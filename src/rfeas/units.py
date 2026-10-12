"""Evaluation units for the risk/memory feasibility study: one (video, target object) each.

DAVIS 2017 val (480p, dense GT; may overlap SAM 2's training data -- reported as a caveat) and
VKITTI2 development scenes (all tracks eligible under the M1 first_area rule). Units, frame ranges
and the dev/val split are fixed from metadata only (no SAM 2 output is consulted).
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DAVIS = Path("data/davis2017/DAVIS")


@dataclass(frozen=True)
class Unit:
    dataset: str          # "davis" | "vkitti"
    video: str            # grouping key: units of one video always share a split
    obj_id: int
    frames: tuple         # original frame ids / names (prompt first)

    @property
    def name(self) -> str:
        return f"{self.dataset}__{self.video}__obj{self.obj_id}"


def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def split_of(video: str) -> str:
    """Video-level dev/val assignment from a fixed hash (never frame-level)."""
    return "dev" if stable_hash(f"rfeas-split|{video}") % 2 == 0 else "val"


# ----------------------------------------------------------------------------- DAVIS
def davis_units(split_file: str = "val") -> tuple[list[Unit], list[dict]]:
    units, excluded = [], []
    for v in (DAVIS / "ImageSets/2017" / f"{split_file}.txt").read_text().split():
        frames = sorted(p.stem for p in (DAVIS / "JPEGImages/480p" / v).glob("*.jpg"))
        ann = DAVIS / "Annotations/480p" / v
        first = np.asarray(Image.open(ann / f"{frames[0]}.png"))
        ids = sorted(int(i) for i in np.unique(first) if i not in (0, 255))
        all_ids = set()
        for f in frames:
            all_ids |= {int(i) for i in np.unique(np.asarray(Image.open(ann / f"{f}.png")))} - {0, 255}
        for oid in sorted(all_ids):
            if oid not in ids:
                excluded.append({"dataset": "davis", "video": v, "obj_id": oid, "reason": "absent in first frame (no prompt)"})
                continue
            units.append(Unit("davis", f"davis/{v}", oid, tuple(frames)))
    return units, excluded


def davis_gt(unit: Unit, frame: str) -> np.ndarray:
    a = np.asarray(Image.open(DAVIS / "Annotations/480p" / unit.video.split("/", 1)[1] / f"{frame}.png"))
    return a == unit.obj_id


def davis_frame_dir(unit: Unit) -> Path:
    return DAVIS / "JPEGImages/480p" / unit.video.split("/", 1)[1]


# ----------------------------------------------------------------------------- VKITTI
def vkitti_units(scenes=("Scene01", "Scene06"), root="data/vkitti2") -> tuple[list[Unit], list[dict], dict]:
    from vkitti.dataset import VKittiDataset
    from vkitti.tracks import select_tracks, summarize_tracks
    units, excluded, dsets = [], [], {}
    for s in scenes:
        ds = VKittiDataset(root, s)
        H, W = ds.image_shape
        summ = summarize_tracks(ds.mask_pixel_counts(), ds.info, ds.bbox, H * W, 0.001)
        sel = select_tracks(summ, H * W, 0.001, 40, max_tracks=10**9)   # every eligible track
        for r in sel.itertuples():
            units.append(Unit("vkitti", f"vkitti/{s}", int(r.track_id),
                              tuple(range(int(r.prompt_frame), int(r.last_frame) + 1))))
        dsets[s] = ds
    return units, excluded, dsets


def check2_select(units, n: int = 12):
    """Pre-registered CHECK-2 subset: first n VAL units ordered by a fixed hash of the unit name."""
    val = [u for u in units if split_of(u.video) == "val"]
    return sorted(val, key=lambda u: stable_hash(f"rfeas-check2|{u.name}"))[:n]
