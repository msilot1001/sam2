"""CHECK 1: causal features, future-failure labels, a small logistic model and its evaluation.

Features at frame t use only signals of frames <= t (shift/rolling over the past). Labels look at
GT of frames t+1..t+H and are used ONLY for fitting/evaluation, never as features.
"""
import numpy as np
import pandas as pd

FEATURES = ["obj_score", "pred_iou", "pred_iou_spread", "log_area", "mean_prob_in_mask", "consistency_iou_prev",
            "log_area_ratio_prev", "d_obj_score", "d_pred_iou", "min3_pred_iou", "min3_obj_score"]


def causal_features(sig: pd.DataFrame) -> pd.DataFrame:
    """sig: one unit's per-frame signals sorted by video_idx (prompt row included, index 0)."""
    s = sig.sort_values("video_idx").reset_index(drop=True)
    f = pd.DataFrame({"video_idx": s.video_idx})
    for c in ("obj_score", "pred_iou", "pred_iou_spread", "mean_prob_in_mask", "consistency_iou_prev"):
        f[c] = s[c]
    f["log_area"] = np.log1p(s["area"])
    f["log_area_ratio_prev"] = np.log(s["area_ratio_prev"])
    # prompt frame (idx 0) signals are not tracking outputs: mask them out of the history
    hist = s.copy()
    hist.loc[hist.video_idx == 0, ["obj_score", "pred_iou"]] = np.nan
    f["d_obj_score"] = hist.obj_score - hist.obj_score.shift(1)
    f["d_pred_iou"] = hist.pred_iou - hist.pred_iou.shift(1)
    f["min3_pred_iou"] = hist.pred_iou.rolling(3, min_periods=1).min()
    f["min3_obj_score"] = hist.obj_score.rolling(3, min_periods=1).min()
    return f


def future_labels(ev: pd.DataFrame, H: int, j_thr: float) -> pd.DataFrame:
    """ev: per-frame evaluation (video_idx, J, gt_empty). label=1 if some VISIBLE-GT frame in
    (t, t+H] has J < j_thr. eligible: t>0, t+H <= last idx, >=1 visible frame in the window."""
    e = ev.sort_values("video_idx").reset_index(drop=True)
    n = len(e)
    J, vis = e.J.to_numpy(), (~e.gt_empty).to_numpy()
    lab, elig, cur_fail = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    for t in range(n):
        if t == 0 or t + H > n - 1:
            continue
        w = slice(t + 1, t + H + 1)
        if not vis[w].any():
            continue
        elig[t] = True
        lab[t] = float(np.any((J[w] < j_thr) & vis[w]))
    cur_fail = vis & (J < j_thr)
    return pd.DataFrame({"video_idx": e.video_idx, "label": lab, "eligible": elig, "current_fail": cur_fail,
                         "current_accurate": J >= 0.5})


def failure_episodes(ev: pd.DataFrame, j_thr: float) -> list[tuple[int, int]]:
    """Maximal runs (video_idx) of visible-GT frames with J < j_thr, prompt excluded."""
    e = ev.sort_values("video_idx")
    bad = ((~e.gt_empty) & (e.J < j_thr) & (e.video_idx > 0)).to_numpy()
    idx = e.video_idx.to_numpy()
    eps, start = [], None
    for i, b in zip(idx, bad):
        if b and start is None:
            start = i
        if not b and start is not None:
            eps.append((start, prev))
            start = None
        prev = i
    if start is not None:
        eps.append((start, prev))
    return eps


# ----------------------------------------------------------------------------- model
class Logistic:
    """L2-regularised logistic regression fit by Newton/IRLS on standardized features."""

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
        Z = self._z(X)
        A = np.c_[np.ones(len(Z)), Z]
        w = np.zeros(A.shape[1])
        R = self.lam * np.eye(A.shape[1])
        R[0, 0] = 0.0
        for _ in range(100):
            p = 1 / (1 + np.exp(-A @ w))
            g = A.T @ (p - y) + R @ w
            Hm = A.T @ (A * (p * (1 - p))[:, None]) + R
            step = np.linalg.solve(Hm, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def _z(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0)   # missing (first frames) -> feature mean

    def predict(self, X):
        return 1 / (1 + np.exp(-(self.w[0] + self._z(X) @ self.w[1:])))

    def to_dict(self):
        return {"lam": self.lam, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["lam"])
        m.mu, m.sd, m.w = np.array(d["mu"]), np.array(d["sd"]), np.array(d["w"])
        return m


# ----------------------------------------------------------------------------- metrics
def auroc(y, s) -> float:
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        m = s == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s) -> float:
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if not y.any():
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    yt = y[order]
    tp = np.cumsum(yt)
    prec = tp / np.arange(1, len(yt) + 1)
    return float((prec * yt).sum() / yt.sum())


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def calibration_bins(y, p, n_bins: int = 10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any():
            rows.append({"bin_lo": a, "bin_hi": b, "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                         "obs_rate": float(y[m].mean())})
    return pd.DataFrame(rows)


def cluster_bootstrap(df: pd.DataFrame, group: str, stat, n: int = 10000, seed: int = 0):
    """Percentile CI of stat(df_resampled) resampling whole groups (videos)."""
    groups = df[group].unique()
    parts = {g: df[df[group] == g] for g in groups}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(groups, len(groups), replace=True)
        v = stat(pd.concat([parts[g] for g in pick], ignore_index=True))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
