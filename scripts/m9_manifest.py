"""Freeze an M9 evaluation manifest from GT ONLY (run before any SAM 2 inference on the dataset).

    M9_DATA=<root> uv run python scripts/m9_manifest.py <dataset> [videos.txt] [cap_frames] [max_total_frames]

Inclusion rule (M8 definitions, fixed before inference):
  - object visible somewhere; prompt = its FIRST annotated visible frame (LVOS objects may appear late)
  - >= 10 annotated frames from the prompt frame onwards
  - crop unit: vcam.crop_side(prompt mask) is not None (box side x4 < min(H, W)), as in M6-M8
  - small: prompt-frame area < 1% of the frame (primary stratum); tiny: < 0.1%; only small units are included
  - at most 3 small objects per video (seeded choice, seed 9) so one video cannot dominate
  - split by VIDEO: dev/conf from a stable hash of the video id (all objects of a video share the split)
  - optional compute budget (LVOS v2, amendment 4): each unit keeps its first `cap_frames` frames; videos are taken in a
    seeded random order (seed 9) and whole videos are added while the frame total stays <= `max_total_frames`
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rfeas import xdata  # noqa: E402
from spatial import vcam  # noqa: E402

SEED, MAX_PER_VIDEO, MIN_FRAMES = 9, 3, 10


def split_of(dataset, vid):
    return "dev" if int(hashlib.sha256(f"{dataset}/{vid}/m9".encode()).hexdigest(), 16) % 2 == 0 else "conf"


def main(dataset, vlist=None, cap_frames=None, max_total=None):
    r = xdata.root(dataset)
    vids = sorted(p.name for p in (r / "Annotations").iterdir() if p.is_dir())
    if vlist:
        keep = set(Path(vlist).read_text().split())
        vids = [v for v in vids if v in keep]
    rng = random.Random(SEED)
    units, videos, rows = [], {}, []
    for v in vids:
        d = xdata.describe_video(dataset, v)
        videos[v] = {k: d[k] for k in ("n_frames", "H", "W")}
        cand = []
        for oid, o in sorted(d["objects"].items()):
            m0 = np.array(Image.open(r / "Annotations" / v / f"{o['first_frame']}.png")) == oid
            crop = vcam.crop_side(m0)
            row = {"video": v, "obj_id": int(oid), **{k: o[k] for k in o if k != "frames"}, "crop_side": crop,
                   "small": o["area0_frac"] < 0.01, "tiny": o["area0_frac"] < 0.001}
            rows.append(row)
            if crop is not None and row["small"] and o["n_frames_from_prompt"] >= MIN_FRAMES:
                cand.append((oid, o, row))
        rng.shuffle(cand)
        for oid, o, row in sorted(cand[:MAX_PER_VIDEO], key=lambda c: c[0]):
            units.append({"video": v, "obj_id": int(oid), "split": split_of(dataset, v), "frames": o["frames"],
                          **{k: row[k] for k in ("first_frame", "area0_frac", "area_med_frac", "area_min_frac",
                                                 "area_max_frac", "absent_frames", "growth_max", "tiny", "crop_side",
                                                 "n_frames_from_prompt")}})
    budget = None
    if cap_frames:
        cap, lim = int(cap_frames), int(max_total)
        for u in units:
            u["frames"] = u["frames"][:cap]
        order = sorted({u["video"] for u in units})
        random.Random(SEED).shuffle(order)
        keep, tot = set(), 0
        for v in order:
            n = sum(len(u["frames"]) for u in units if u["video"] == v)
            if tot + n <= lim:
                keep.add(v)
                tot += n
        budget = {"cap_frames": cap, "max_total_frames": lim, "eligible_units": len(units),
                  "eligible_videos": len(order), "kept_videos": len(keep)}
        units = [u for u in units if u["video"] in keep]
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S KST"), "dataset": dataset, "rule": __doc__, "seed": SEED,
           "n_videos_scanned": len(vids), "n_units": len(units),
           "n_unit_videos": len({u["video"] for u in units}),
           "split_counts": {s: sum(u["split"] == s for u in units) for s in ("dev", "conf")},
           "n_tiny": sum(u["tiny"] for u in units), "n_frames_total": sum(len(u["frames"]) for u in units), "budget": budget,
           "videos": videos, "units": units}
    dst = xdata.MANIFESTS / f"{dataset}_units.json"
    if dst.exists():
        raise SystemExit(f"{dst} exists: manifests are frozen, refusing to overwrite")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    (xdata.MANIFESTS / f"{dataset}_all_objects.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps({k: out[k] for k in ("dataset", "n_videos_scanned", "n_units", "n_unit_videos", "split_counts",
                                         "n_tiny", "n_frames_total", "budget")}))


if __name__ == "__main__":
    main(*sys.argv[1:])
