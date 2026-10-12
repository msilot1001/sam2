"""Paired comparison of two runs of the same unit on identical frames (prompt excluded)."""
import numpy as np
import pandas as pd

METRICS = ["J", "F", "JF", "F_strict"]


def paired(native: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    a = native[native.video_idx > 0].set_index("video_idx")
    b = other[other.video_idx > 0].set_index("video_idx")
    if list(a.index) != list(b.index) or not (a.gt_empty.to_numpy() == b.gt_empty.to_numpy()).all():
        raise ValueError("runs do not cover identical frames of the same unit")
    out = pd.DataFrame(index=a.index)
    for m in METRICS:
        out[f"{m}_native"], out[f"{m}_policy"] = a[m], b[m]
        out[f"delta_{m}"] = b[m] - a[m]
    out["gt_empty"] = a.gt_empty
    return out.reset_index()


def longest_streak(js, thr: float = 0.5) -> int:
    best = n = 0
    for j in js:
        n = n + 1 if j < thr else 0
        best = max(best, n)
    return best
