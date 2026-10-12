"""DAVIS-2017 region (J) and boundary (F) measures, numpy only.

Follows davis2017-evaluation (metrics.py: db_eval_iou, f_measure, _seg2bmap):
  J = |P∩G| / |P∪G|, and J = 1 when both masks are empty.
  F: boundary maps via _seg2bmap; a boundary pixel matches if the other boundary lies within
     bound_pix = ceil(0.008 * image diagonal) (disk dilation); F = 2PR/(P+R).
     Empty-boundary cases: both empty -> P=R=1; only prediction empty -> P=1, R=0;
     only GT empty -> P=0, R=1.
"""
import numpy as np

BOUND_TH = 0.008


def j_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _check(pred, gt)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask (DAVIS _seg2bmap at native resolution)."""
    seg = seg.astype(bool)
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


def disk_offsets(r: int) -> list[tuple[int, int]]:
    """skimage.morphology.disk(r) footprint as (dy, dx) offsets."""
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r]


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation with a disk of radius r (zero padding outside the image)."""
    H, W = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return out
    padded = np.zeros((H + 2 * r, W + 2 * r), dtype=bool)
    for dy, dx in disk_offsets(r):
        padded[ys + r + dy, xs + r + dx] = True
    return padded[r:r + H, r:r + W]


def f_measure(pred: np.ndarray, gt: np.ndarray, bound_th: float = BOUND_TH) -> float:
    pred, gt = _check(pred, gt)
    bound_pix = bound_th if bound_th >= 1 else int(np.ceil(bound_th * np.linalg.norm(pred.shape)))
    fg_b, gt_b = seg2bmap(pred), seg2bmap(gt)
    n_fg, n_gt = int(fg_b.sum()), int(gt_b.sum())
    if n_fg == 0 and n_gt == 0:
        precision = recall = 1.0
    elif n_fg == 0:
        precision, recall = 1.0, 0.0
    elif n_gt == 0:
        precision, recall = 0.0, 1.0
    else:
        precision = float((fg_b & dilate(gt_b, bound_pix)).sum() / n_fg)
        recall = float((gt_b & dilate(fg_b, bound_pix)).sum() / n_gt)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _check(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError(f"prediction {pred.shape} and GT {gt.shape} must be equal 2-D shapes")
    return pred, gt
