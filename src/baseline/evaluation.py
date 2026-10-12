"""Per-frame and per-track evaluation shared by the M2 baseline and dropout runs.

GT masks are read here, after inference. video_idx 0 is the prompted frame (excluded from means).
"""
import hashlib
from pathlib import Path

import pandas as pd

from .metrics import f_measure, j_measure
from .sam2_vos import Clip


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(ds, clip: Clip, masks: dict, cfg) -> pd.DataFrame:
    """GT is read here, after inference. video_idx 0 is the prompted frame (excluded from means)."""
    rows = []
    for idx, f in enumerate(clip.original_frame_ids):
        gt = ds.get_frame(f, clip.track_id)["mask"]
        pr = masks[idx]
        if pr.shape != gt.shape:
            raise ValueError(f"{clip.name} idx {idx}: prediction {pr.shape} vs GT {gt.shape}")
        j, fm = j_measure(pr, gt), f_measure(pr, gt, cfg["boundary_th"])
        f_strict = f_measure(pr, gt, cfg["boundary_th_strict"])  # secondary: DAVIS F saturates on small cars
        rows.append({"scene": clip.scene, "track_id": clip.track_id, "video_idx": idx, "original_frame_id": f,
                     "is_prompt": idx == 0, "gt_pixels": int(gt.sum()), "pred_pixels": int(pr.sum()),
                     "gt_empty": not gt.any(), "J": j, "F": fm, "JF": (j + fm) / 2, "F_strict": f_strict})
    return pd.DataFrame(rows)


def summarize(frames: pd.DataFrame, lost_j: float) -> pd.DataFrame:
    rows = []
    for (scene, tid), g in frames.groupby(["scene", "track_id"], sort=False):
        e = g[~g.is_prompt]
        vis = e[~e.gt_empty]
        emp = e[e.gt_empty]
        rows.append({
            "scene": scene, "track_id": tid, "first_frame": int(g.original_frame_id.min()),
            "last_frame": int(g.original_frame_id.max()), "eval_frames": len(e),
            "gt_empty_frames": len(emp),
            "J": e.J.mean(), "F": e.F.mean(), "JF": e.JF.mean(),                       # DAVIS convention
            "J_visible": vis.J.mean(), "F_visible": vis.F.mean(), "JF_visible": vis.JF.mean(),
            "F_strict": e.F_strict.mean(),
            "false_positive_on_empty_gt": int((emp.pred_pixels > 0).sum()),
            "lost_frames_J_lt": int((vis.J < lost_j).sum()),
            "first_lost_frame": int(vis[vis.J < lost_j].original_frame_id.min()) if (vis.J < lost_j).any() else None,
            "pred_empty_on_visible_gt": int((vis.pred_pixels == 0).sum()),
            "prompt_frame_J": float(g[g.is_prompt].J.iloc[0]),
        })
    return pd.DataFrame(rows)
