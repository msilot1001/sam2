"""Convert SegTrack v2 (data/segtrackv2/raw) into SAM 2 clip folders + per-object bool mask stacks.

    uv run python scripts/segtrack_prepare.py

Writes data/segtrackv2/clips/<video>/{00000.jpg..., obj<k>.npy, frames.json}. Frames are matched to GT by file
stem (worm has one GT frame without an image; monkeydog GT is .png for .bmp frames). Images are re-encoded to JPEG
(quality 95) because the official SAM 2 loader reads JPEG only -- a documented, model-input-side deviation.
GT is binarised as max channel > 127. Never overwrites an existing clip folder.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path("data/segtrackv2/raw/SegTrackv2")
OUT = Path("data/segtrackv2/clips")


def main():
    for vdir in sorted(p for p in (RAW / "JPEGImages").iterdir() if p.is_dir()):
        v, dst = vdir.name, OUT / vdir.name
        if dst.exists():
            continue
        imgs = {p.stem: p for p in vdir.iterdir()}
        g = RAW / "GroundTruth" / v
        objs = sorted((p for p in g.iterdir() if p.is_dir()), key=lambda p: int(p.name)) or [g]
        gts = [{p.stem: p for p in o.iterdir()} for o in objs]
        stems = sorted(set(imgs).intersection(*gts))
        dst.mkdir(parents=True)
        for i, s in enumerate(stems):
            Image.open(imgs[s]).convert("RGB").save(dst / f"{i:05d}.jpg", quality=95)
        for k, gt in enumerate(gts, 1):
            np.save(dst / f"obj{k}.npy", np.stack([np.asarray(Image.open(gt[s]).convert("RGB")).max(-1) > 127 for s in stems]))
        (dst / "frames.json").write_text(json.dumps({"stems": stems, "n_objects": len(gts),
                                                     "unmatched_images": sorted(set(imgs) - set(stems))}))
        print(v, len(stems), "frames", len(gts), "objects")


if __name__ == "__main__":
    main()
