"""Consistency checks for one VKittiDataset. Each returns (ok, details)."""
import numpy as np
import pandas as pd

from .calibration import camera_pose, rigid_problems, rotation_angle_deg

STEREO_BASELINE_M = 0.532725  # documented Camera_1 offset to the right of Camera_0


def check_pixel_counts(counts: pd.DataFrame, bbox: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    """bbox.txt number_pixels vs actual mask pixels; returns mismatching rows (empty = ok)."""
    j = bbox[["frame", "trackID", "number_pixels"]].merge(counts, on=["frame", "trackID"], how="outer")
    j = j.fillna(0)
    bad = j[j.number_pixels != j.mask_pixels]
    return bad.empty, bad


def camera_trajectory(ds) -> pd.DataFrame:
    """Per-frame camera position, step distance and step rotation (from extrinsic.txt only)."""
    rows, prev = [], None
    for f in ds.frame_ids:
        p, R = camera_pose(ds.extrinsics[f])
        step = np.linalg.norm(p - prev[0]) if prev else np.nan
        rot = rotation_angle_deg(prev[1], R) if prev else np.nan
        rows.append((f, *p, step, rot))
        prev = (p, R)
    return pd.DataFrame(rows, columns=["frame", "x", "y", "z", "step_m", "step_rot_deg"])


def check_extrinsics(ds, max_step_m: float, max_step_rot_deg: float) -> tuple[bool, dict]:
    bad = {f: p for f in ds.frame_ids if (p := rigid_problems(ds.extrinsics[f]))}
    traj = camera_trajectory(ds)
    jumps = traj[(traj.step_m > max_step_m) | (traj.step_rot_deg > max_step_rot_deg)]
    details = {
        "invalid_matrices": bad,
        "jumps": jumps.frame.tolist(),
        "step_m": traj.step_m.describe().round(4).to_dict(),
        "step_rot_deg": traj.step_rot_deg.describe().round(4).to_dict(),
        "path_length_m": float(traj.step_m.sum()),
    }
    return not bad and jumps.empty, details


def check_stereo_baseline(extrinsics_all: dict, tol: float = 1e-3) -> tuple[bool, dict]:
    """Camera_1 must sit STEREO_BASELINE_M along Camera_0's +x axis -- verifies world->camera direction."""
    errs = []
    for (f, c), T0 in extrinsics_all.items():
        if c != 0 or (f, 1) not in extrinsics_all:
            continue
        p0, R0 = camera_pose(T0)
        p1, _ = camera_pose(extrinsics_all[(f, 1)])
        offset_in_cam0 = R0.T @ (p1 - p0)
        errs.append(np.abs(offset_in_cam0 - [STEREO_BASELINE_M, 0, 0]).max())
    worst = float(max(errs)) if errs else float("nan")
    return bool(errs) and worst < tol, {"frames_checked": len(errs), "max_abs_err_m": worst}
