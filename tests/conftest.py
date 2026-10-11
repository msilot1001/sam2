"""Tiny synthetic VKITTI2 tree mirroring the real 2.0.3 layout and text headers."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

H, W = 6, 8
BASELINE = 0.532725
FRAMES = [0, 1, 2]
EXT_HEADER = "frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1"
BBOX_HEADER = ("frame cameraID trackID left right top bottom number_pixels "
               "truncation_ratio occupancy_ratio isMoving")


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def cam_to_world(f: int, cam: int) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth camera pose used to synthesise extrinsic.txt."""
    R = rot_y(10.0 * f)
    p = np.array([float(f), -1.65, 2.0 * f]) + R @ np.array([BASELINE * cam, 0, 0])
    return R, p


def instance_frame(f: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    m[0:2, 0:3] = 1          # track 0: 6 px every frame
    if f < 2:
        m[3:6, 4:8] = 5      # track 4: 12 px in frames 0,1, absent in frame 2
    return m


def write_indexed_png(path: Path, idx: np.ndarray) -> None:
    im = Image.fromarray(idx, mode="P")
    # Palette deliberately unrelated to indices so convert('RGB') would destroy track ids.
    im.putpalette([(37 * i) % 256 for i in range(768)])
    im.save(path)


@pytest.fixture
def vkitti_root(tmp_path: Path) -> Path:
    d = tmp_path / "Scene01" / "clone"
    for cam in (0, 1):
        (d / "frames" / "rgb" / f"Camera_{cam}").mkdir(parents=True)
        (d / "frames" / "instanceSegmentation" / f"Camera_{cam}").mkdir(parents=True)
    ext, bbox = [EXT_HEADER], [BBOX_HEADER]
    for f in FRAMES:
        for cam in (0, 1):
            rgb = np.zeros((H, W, 3), np.uint8)
            rgb[..., 0] = 10 + 100 * f  # frame id encoded in red channel
            Image.fromarray(rgb).save(d / "frames" / "rgb" / f"Camera_{cam}" / f"rgb_{f:05d}.jpg", quality=100)
            inst = instance_frame(f)
            write_indexed_png(d / "frames" / "instanceSegmentation" / f"Camera_{cam}" / f"instancegt_{f:05d}.png", inst)
            R, p = cam_to_world(f, cam)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R.T, -R.T @ p  # world->camera
            ext.append(f"{f} {cam} " + " ".join(f"{v:.9f}" for v in T.ravel()))
            for tid in (0, 4):
                n = int((inst == tid + 1).sum())
                if n:
                    bbox.append(f"{f} {cam} {tid} 0 1 0 1 {n} 0 1 {'True' if tid == 4 else 'False'}")
    (d / "extrinsic.txt").write_text("\n".join(ext) + "\n")
    (d / "bbox.txt").write_text("\n".join(bbox) + "\n")
    (d / "info.txt").write_text("trackID label model color\n0 Car Sedan4Door Black\n4 Van Van Red\n7 Car Hatchback Blue\n")
    return tmp_path
