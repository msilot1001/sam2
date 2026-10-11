"""VKITTI2 RGB / instance-mask / camera-pose loader for one (scene, variation, camera).

Layout (verified against vkitti_2.0.3 archives):
    {root}/{scene}/{variation}/frames/rgb/Camera_{c}/rgb_{frame:05d}.jpg
    {root}/{scene}/{variation}/frames/instanceSegmentation/Camera_{c}/instancegt_{frame:05d}.png
    {root}/{scene}/{variation}/{extrinsic,bbox,info}.txt

All frame ids are ORIGINAL VKITTI2 frame ids (0-based). Nothing here renumbers frames.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .calibration import camera_pose, load_extrinsics, read_table

_FRAME_RE = re.compile(r"_(\d{5})\.(?:jpg|png)$")


def load_rgb(path: Path) -> np.ndarray:
    """(H, W, 3) uint8, RGB order."""
    if not path.exists():
        raise FileNotFoundError(f"missing RGB frame: {path}")
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def load_instance(path: Path) -> np.ndarray:
    """(H, W) uint8 raw palette indices: 0 = background, k = trackID + 1.

    Never convert('RGB') here: that maps indices through the palette and loses track ids.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing instance mask: {path}")
    with Image.open(path) as im:
        if im.mode not in ("P", "L"):
            raise ValueError(f"{path}: expected 8-bit indexed PNG (mode P/L), got mode {im.mode}")
        return np.asarray(im, dtype=np.uint8)


def track_mask(instance: np.ndarray, track_id: int) -> np.ndarray:
    """Binary mask of one vehicle: instance == track_id + 1."""
    if not 0 <= track_id <= 254:
        raise ValueError(f"track_id {track_id} cannot be encoded in an 8-bit instance PNG")
    return instance == track_id + 1


def _frame_ids(directory: Path, prefix: str) -> list[int]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing directory: {directory}")
    ids = sorted(int(m.group(1)) for p in directory.glob(f"{prefix}_*") if (m := _FRAME_RE.search(p.name)))
    if not ids:
        raise FileNotFoundError(f"no {prefix}_*.jpg/png frames in {directory}")
    return ids


class VKittiDataset:
    def __init__(self, root="data/vkitti2", scene="Scene01", variation="clone", camera_id=0):
        self.root, self.scene, self.variation, self.camera_id = Path(root), scene, variation, int(camera_id)
        self.dir = self.root / scene / variation
        if not self.dir.is_dir():
            raise FileNotFoundError(f"missing scene directory: {self.dir} (run scripts/setup_vkitti.py)")
        cam = f"Camera_{self.camera_id}"
        self.rgb_dir = self.dir / "frames" / "rgb" / cam
        self.mask_dir = self.dir / "frames" / "instanceSegmentation" / cam

        self.frame_ids = _frame_ids(self.rgb_dir, "rgb")
        mask_ids = _frame_ids(self.mask_dir, "instancegt")
        if mask_ids != self.frame_ids:
            raise ValueError(f"{self.tag}: RGB/mask frame ids differ: "
                             f"only-rgb={sorted(set(self.frame_ids) - set(mask_ids))[:10]} "
                             f"only-mask={sorted(set(mask_ids) - set(self.frame_ids))[:10]}")
        if self.frame_ids != list(range(self.frame_ids[0], self.frame_ids[-1] + 1)):
            raise ValueError(f"{self.tag}: frame ids are not contiguous")

        all_ext = load_extrinsics(self.dir / "extrinsic.txt")
        self.extrinsics = {f: T for (f, c), T in all_ext.items() if c == self.camera_id}
        missing = [f for f in self.frame_ids if f not in self.extrinsics]
        if missing:
            raise ValueError(f"{self.tag}: no extrinsic for frames {missing[:10]}")

        self.info = read_table(self.dir / "info.txt")
        bbox = read_table(self.dir / "bbox.txt")
        self.bbox = bbox[bbox.cameraID == self.camera_id].reset_index(drop=True)
        self.track_ids = set(self.info.trackID.astype(int))
        self.image_shape = load_rgb(self.rgb_path(self.frame_ids[0])).shape[:2]

    @property
    def tag(self) -> str:
        return f"{self.scene}/{self.variation}/Camera_{self.camera_id}"

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _check_frame(self, frame_id: int) -> int:
        if isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)):
            raise TypeError(f"frame_id must be int, got {type(frame_id).__name__}")
        if not self.frame_ids[0] <= frame_id <= self.frame_ids[-1]:
            raise KeyError(f"{self.tag}: frame {frame_id} not in [{self.frame_ids[0]}, {self.frame_ids[-1]}]")
        return int(frame_id)

    def rgb_path(self, frame_id: int) -> Path:
        return self.rgb_dir / f"rgb_{frame_id:05d}.jpg"

    def mask_path(self, frame_id: int) -> Path:
        return self.mask_dir / f"instancegt_{frame_id:05d}.png"

    def load_rgb(self, frame_id: int) -> np.ndarray:
        rgb = load_rgb(self.rgb_path(self._check_frame(frame_id)))
        if rgb.shape[:2] != self.image_shape:
            raise ValueError(f"{self.tag} frame {frame_id}: RGB shape {rgb.shape[:2]} != {self.image_shape}")
        return rgb

    def load_instance(self, frame_id: int) -> np.ndarray:
        inst = load_instance(self.mask_path(self._check_frame(frame_id)))
        if inst.shape != self.image_shape:
            raise ValueError(f"{self.tag} frame {frame_id}: mask shape {inst.shape} != {self.image_shape}")
        return inst

    def camera(self, frame_id: int) -> dict:
        """Camera pose only -- the sole pose input allowed for memory selection."""
        T = self.extrinsics[self._check_frame(frame_id)]
        position, rotation = camera_pose(T)
        return {"extrinsic": T.copy(), "camera_position": position, "camera_rotation": rotation}

    def get_frame(self, frame_id: int, track_id: int) -> dict:
        if track_id not in self.track_ids:
            raise KeyError(f"{self.tag}: track_id {track_id} not in info.txt")
        mask = track_mask(self.load_instance(frame_id), track_id)
        return {
            "rgb": self.load_rgb(frame_id),
            "mask": mask,
            "original_frame_id": int(frame_id),
            "track_id": int(track_id),
            "track_visible": bool(mask.any()),
            "scene": self.scene,
            "variation": self.variation,
            "camera_id": self.camera_id,
            **self.camera(frame_id),
        }

    def mask_pixel_counts(self) -> pd.DataFrame:
        """Actual per-(frame, track) pixel counts from every instance PNG of this camera."""
        rows = []
        for f in self.frame_ids:
            counts = np.bincount(self.load_instance(f).ravel(), minlength=256)
            rows += [(f, int(v) - 1, int(counts[v])) for v in np.flatnonzero(counts[1:]) + 1]
        return pd.DataFrame(rows, columns=["frame", "trackID", "mask_pixels"])
