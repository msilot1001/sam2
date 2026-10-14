"""M9 external datasets (MOSEv2, LVOS v2, VOST) behind the same Unit interface as DAVIS/PUMaVOS/VKITTI.

Layout (DAVIS-style, after selective extraction): <root>/<dataset>/{JPEGImages,Annotations}/<video>/<frame>.{jpg,png};
Annotations are palette PNGs whose pixel value is the object id. Units come ONLY from a frozen manifest
(outputs/milestone9/manifests/<dataset>_units.json, written before any SAM 2 inference); a unit's `frames` start at
its first legal prompt frame (first annotated frame where the object is visible) and keep only frames that have an
annotation file, so sparse annotation is never densified. Data root: $M9_DATA (default <repo>/data/m9).
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from rfeas.units import Unit

REPO = Path(__file__).resolve().parents[2]
DATASETS = ("mosev2", "lvosv2", "vost")
MANIFESTS = REPO / "outputs" / "milestone9" / "manifests"


def root(dataset: str) -> Path:
    return Path(os.environ.get("M9_DATA", REPO / "data" / "m9")) / dataset


def _vid(u: Unit) -> str:
    return u.video.split("/", 1)[1]


@lru_cache(maxsize=4096)
def _ann(dataset: str, vid: str, frame: str) -> np.ndarray:
    return np.array(Image.open(root(dataset) / "Annotations" / vid / f"{frame}.png"))


def gt(u: Unit, t: int) -> np.ndarray:
    a = _ann(u.dataset, _vid(u), u.frames[t])
    return a == u.obj_id


def other(u: Unit, t: int) -> np.ndarray:
    a = _ann(u.dataset, _vid(u), u.frames[t])
    return (a != 0) & (a != u.obj_id) & (a != 255)


def paths(u: Unit) -> list:
    d = root(u.dataset) / "JPEGImages" / _vid(u)
    return [d / f"{f}.jpg" for f in u.frames]


def units(dataset: str, split: str | None = None) -> list:
    """(split, Unit) pairs from the frozen manifest; split in {None, 'dev', 'conf'}."""
    m = json.loads((MANIFESTS / f"{dataset}_units.json").read_text())
    out = []
    for r in m["units"]:
        if split is None or r["split"] == split:
            out.append((r["split"], Unit(dataset, f"{dataset}/{r['video']}", int(r["obj_id"]), tuple(r["frames"]))))
    return out


def describe_video(dataset: str, vid: str) -> dict:
    """GT-only statistics of every object in one video (used to freeze the manifest BEFORE inference)."""
    ad = root(dataset) / "Annotations" / vid
    jd = root(dataset) / "JPEGImages" / vid
    frames = sorted(p.stem for p in ad.glob("*.png"))
    if jd.exists():                                   # images may be fetched only after the manifest is frozen
        frames = [f for f in frames if (jd / f"{f}.jpg").exists()]
    objs = {}
    H = W = None
    for f in frames:
        a = np.array(Image.open(ad / f"{f}.png"))
        H, W = a.shape[:2]
        ids, cnt = np.unique(a, return_counts=True)
        for i, c in zip(ids.tolist(), cnt.tolist()):
            if i in (0, 255):
                continue
            objs.setdefault(i, {}).setdefault("area", {})[f] = c
    out = {"video": vid, "n_frames": len(frames), "H": H, "W": W, "objects": {}}
    for i, o in objs.items():
        present = [f for f in frames if o["area"].get(f, 0) > 0]
        first = present[0]
        k = frames.index(first)
        areas = [o["area"].get(f, 0) for f in frames[k:]]
        vis = [a for a in areas if a > 0]
        out["objects"][i] = {"first_frame": first, "first_index": k, "n_frames_from_prompt": len(frames) - k,
                             "area0_frac": areas[0] / (H * W), "area_med_frac": float(np.median(vis)) / (H * W),
                             "area_min_frac": min(vis) / (H * W), "area_max_frac": max(vis) / (H * W),
                             "absent_frames": int(sum(a == 0 for a in areas)),
                             "growth_max": max(vis) / areas[0], "frames": frames[k:]}
    return out
