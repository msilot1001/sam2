"""Camera extrinsics from VKITTI2 extrinsic.txt.

extrinsic.txt rows are world->camera: X_cam = R @ X_world + t (verified on Scene01/clone:
T @ pose.txt world_space_* reproduces camera_space_* to 1e-6, and Camera_1 differs from
Camera_0 only by t_x -= 0.532725 m, the documented stereo baseline). Camera coordinates:
x right, y down, z forward.

Only camera poses live here. pose.txt (target-vehicle GT) is deliberately not parsed by the
pipeline so it cannot leak into memory selection.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing VKITTI2 text file: {path}")
    return pd.read_csv(path, sep=" ")


def load_extrinsics(path: Path) -> dict[tuple[int, int], np.ndarray]:
    """{(original_frame_id, camera_id): 4x4 world->camera matrix}."""
    df = read_table(path)
    if df.shape[1] != 18:
        raise ValueError(f"{path}: expected 18 columns (frame cameraID + 16), got {df.shape[1]}")
    out = {}
    for row in df.itertuples(index=False):
        key = (int(row[0]), int(row[1]))
        if key in out:
            raise ValueError(f"{path}: duplicate extrinsic for (frame, camera)={key}")
        out[key] = np.asarray(row[2:], dtype=np.float64).reshape(4, 4)
    return out


def rigid_problems(T: np.ndarray, tol: float = 1e-4) -> list[str]:
    """Empty list if T is a valid 4x4 rigid transform."""
    if T.shape != (4, 4):
        return [f"shape {T.shape} != (4, 4)"]
    if not np.all(np.isfinite(T)):
        return ["contains NaN/Inf"]
    problems = []
    if not np.allclose(T[3], [0, 0, 0, 1], atol=tol):
        problems.append(f"last row {T[3]} != [0 0 0 1]")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        problems.append(f"R not orthonormal (max err {np.abs(R.T @ R - np.eye(3)).max():.2e})")
    det = np.linalg.det(R)
    if abs(det - 1) > tol:
        problems.append(f"det(R)={det:.6f} != +1")
    return problems


def invert_rigid(T: np.ndarray) -> np.ndarray:
    """Exact inverse of a rigid transform: [R t] -> [R^T, -R^T t]."""
    R, t = T[:3, :3], T[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def camera_pose(T_world_to_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(camera_position_world (3,), camera_rotation cam->world (3,3))."""
    problems = rigid_problems(T_world_to_cam)
    if problems:
        raise ValueError(f"invalid extrinsic: {'; '.join(problems)}")
    T_c2w = invert_rigid(T_world_to_cam)
    return T_c2w[:3, 3], T_c2w[:3, :3]


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotations."""
    c = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
