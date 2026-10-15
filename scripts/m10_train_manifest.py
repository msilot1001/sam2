"""amend10: freeze the TRAINING unit list for the learned selection head from GT ONLY (before any SAM 2 output on it).

    M9_DATA=<root> python scripts/m10_train_manifest.py <dataset> <area0_max> [tag]

Same unit rule as scripts/m9_manifest.py (first visible annotated frame = prompt, >= 10 annotated frames, crop_side not
None, at most 3 objects per video with seed 9, LVOS v2 first 300 frames) except: prompt-frame area < `area0_max`, and
ONLY videos that appear in no M9 dev/conf unit (whole video excluded if any of its objects is a dev/conf unit) and whose
frames are on disk. Writes outputs/milestone10/train_manifest/<dataset>_units<tag>.json (split = "train").
"""
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

SEED, MAX_PER_VIDEO, MIN_FRAMES, CAP = 9, 3, 10, {"lvosv2": 300}
OUT = Path(__file__).resolve().parents[1] / "outputs/milestone10/train_manifest"


def main(dataset, area0_max, tag=""):
    area0_max = float(area0_max)
    used = {u["video"] for u in json.loads((xdata.MANIFESTS / f"{dataset}_units.json").read_text())["units"]}
    r = xdata.root(dataset)
    vids = sorted(p.name for p in (r / "Annotations").iterdir() if p.is_dir() and p.name not in used
                  and (r / "JPEGImages" / p.name).is_dir())
    rng = random.Random(SEED)
    units = []
    for v in vids:
        d = xdata.describe_video(dataset, v)
        cand = []
        for oid, o in sorted(d["objects"].items()):
            m0 = np.array(Image.open(r / "Annotations" / v / f"{o['first_frame']}.png")) == oid
            if vcam.crop_side(m0) is not None and o["area0_frac"] < area0_max and o["n_frames_from_prompt"] >= MIN_FRAMES:
                cand.append((oid, o))
        rng.shuffle(cand)
        for oid, o in sorted(cand[:MAX_PER_VIDEO], key=lambda c: c[0]):
            fr = o["frames"][:CAP[dataset]] if dataset in CAP else o["frames"]
            units.append({"video": v, "obj_id": int(oid), "split": "train", "frames": fr, "area0_frac": o["area0_frac"],
                          "tiny": o["area0_frac"] < 0.001, "n_frames_from_prompt": o["n_frames_from_prompt"]})
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": dataset, "rule": __doc__, "area0_max": area0_max,
           "seed": SEED, "n_videos_scanned": len(vids), "excluded_dev_conf_videos": len(used), "n_units": len(units),
           "n_unit_videos": len({u["video"] for u in units}), "n_frames_total": sum(len(u["frames"]) for u in units),
           "units": units}
    dst = OUT / f"{dataset}_units{tag}.json"
    if dst.exists():
        raise SystemExit(f"{dst} exists: frozen")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("dataset", "area0_max", "n_videos_scanned", "n_units", "n_unit_videos", "n_frames_total")}))


if __name__ == "__main__":
    main(*sys.argv[1:])
