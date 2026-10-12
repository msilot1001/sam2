"""Milestone 4 aggregation and failure analysis (pure pandas; no inference).

Aggregation order: frames -> run (track x condition x seed) -> seed mean per track -> scene
(mean of its tracks) -> development set (mean of the 8 tracks, equal track weights, as in M2).
Frame counts are carried along so differences in sample composition stay visible.
"""
import numpy as np
import pandas as pd

METRICS = ["J", "F", "JF", "F_strict"]
SUBSETS = {"all": None, "visible": False, "empty_gt": True}   # value of gt_empty selected
KEYS = ["scene", "track_id", "condition", "seed"]


def run_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """One row per run and subset: matched means, deltas and frame counts."""
    rows = []
    for key, g in paired.groupby(KEYS, sort=False):
        for subset, empty in SUBSETS.items():
            s = g if empty is None else g[g.gt_empty == empty]
            row = dict(zip(KEYS, key)) | {"subset": subset, "frames": int(len(s))}
            for m in METRICS:
                nd, dr = s[f"{m}_no_drop_matched"].mean(), s[f"{m}_drop"].mean()
                row |= {f"{m}_no_drop_matched": nd, f"{m}_drop": dr, f"delta_{m}": dr - nd}
            rows.append(row)
    return pd.DataFrame(rows)


def value_cols():
    return [f"{m}_{k}" for m in METRICS for k in ("no_drop_matched", "drop")] + [f"delta_{m}" for m in METRICS]


def aggregate(runs: pd.DataFrame, subset: str = "all") -> dict[str, pd.DataFrame]:
    """Seed mean per track, then scene and overall as UNWEIGHTED means over tracks."""
    r = runs[runs.subset == subset]
    vc = value_cols()
    track = r.groupby(["scene", "track_id", "condition"]).agg(
        **{c: (c, "mean") for c in vc}, seeds=("seed", "nunique"), frames=("frames", "sum")).reset_index()
    scene = track.groupby(["scene", "condition"]).agg(
        **{c: (c, "mean") for c in vc}, tracks=("track_id", "nunique"), frames=("frames", "sum")).reset_index()
    overall = track.groupby("condition").agg(
        **{c: (c, "mean") for c in vc}, tracks=("track_id", "nunique"), frames=("frames", "sum")).reset_index()
    return {"track": track, "scene": scene, "overall": overall}


def bootstrap_ci(values, n_boot: int = 10000, seed: int = 0, q=(2.5, 97.5)) -> tuple[float, float]:
    """Percentile CI of the mean, resampling tracks (the unit), fixed RNG seed."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), q)
    return float(lo), float(hi)


def failure_classes(paired: pd.DataFrame, thr: float, severity_delta: float = 0.1) -> pd.Series:
    """Per matched VISIBLE-GT frame: compare dropout vs matched no-drop at threshold thr.

    persistent        both below thr, |dJ| < severity_delta   (A: already a no-drop failure)
    new               only the dropout run below thr            (B)
    resolved          only the no-drop run below thr            (C)
    severity_changed  both below thr, |dJ| >= severity_delta    (D)
    ok                neither below thr
    Empty-GT frames are excluded (handled as false positives separately).
    """
    p = paired[~paired.gt_empty]
    fd, fn = p.J_drop < thr, p.J_no_drop_matched < thr
    both = fd & fn
    cls = np.select([both & ((p.J_drop - p.J_no_drop_matched).abs() < severity_delta),
                     both, fd & ~fn, fn & ~fd], ["persistent", "severity_changed", "new", "resolved"], "ok")
    return pd.Series(cls, index=p.index)


def failure_runs(js, thr: float) -> list[int]:
    """Lengths of consecutive (in received order) frames with J < thr."""
    out, n = [], 0
    for j in js:
        if j < thr:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def worst_windows(paired: pd.DataFrame, width: int = 5, k: int = 5, ascending: bool = True) -> pd.DataFrame:
    """Deterministic case selection: per run, the received-order window of `width` matched frames
    with the most negative (ascending) or most positive mean delta_J; then the k most extreme runs,
    at most one per track (ties -> scene, track, condition, seed order)."""
    rows = []
    for key, g in paired.groupby(KEYS, sort=True):
        g = g.sort_values("original_frame_id")
        if len(g) < width:
            continue
        roll = g.delta_J.rolling(width).mean().to_numpy()[width - 1:]
        i = int(np.argmin(roll) if ascending else np.argmax(roll))
        w = g.iloc[i:i + width]
        rows.append(dict(zip(KEYS, key)) | {"start_frame": int(w.original_frame_id.iloc[0]),
                                            "end_frame": int(w.original_frame_id.iloc[-1]),
                                            "window_delta_J": float(roll[i]),
                                            "window_J_drop": float(w.J_drop.mean()),
                                            "window_J_no_drop": float(w.J_no_drop_matched.mean())})
    df = pd.DataFrame(rows).sort_values(["window_delta_J", *KEYS], ascending=[ascending, True, True, True, True])
    return df.drop_duplicates(["scene", "track_id"]).head(k).reset_index(drop=True)
