"""F family: global/local mask fusion and evidence-constrained correction (M7). numpy only, frame coordinates.

Inputs are frame-resolution logits: `lL` is NaN outside the crop (unobserved, never treated as background),
`lG` is defined everywhere. `base` is the SGGL output mask. Every function returns a bool mask.
"""
import numpy as np

from spatial import detail


def blend(lL, lG, wL=0.5):
    """F1: weighted logit blend inside the crop (equal weights == probability averaging at 0.5); outside the crop
    nothing is observed by L, so the result there is empty (as SGGL's pasted crop)."""
    obs = ~np.isnan(lL)
    out = np.zeros(lG.shape, bool)
    out[obs] = wL * lL[obs] + (1 - wL) * lG[obs] > 0
    return out


def boundary_band(m, r):
    return detail.dilate(m, r) & ~detail.erode(m, r)


def boundary_refine(base, lL, lG, r=2):
    """F2: change only pixels in an r-px band around the SGGL boundary (inside the crop) to the equal blend."""
    band = boundary_band(base, r) & ~np.isnan(lL)
    out = base.copy()
    out[band] = (np.nan_to_num(lL) + lG)[band] > 0
    return out


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def components(m):
    lab, n = detail.components(m)
    return [lab == k for k in range(1, n + 1)]


def asymmetric_edit(base, lL, lG, g_observable, adj_r=3, add_pL=0.2, del_pG=0.05, del_pL=0.8, symmetric=False):
    """F3: edits proposed by the other view, accepted asymmetrically.
    additions (G-only components): must touch the dilated base mask (same instance) AND L must not deny them where
              it observes (mean p_L >= add_pL; unobserved pixels outside the crop count as 'no denial');
    deletions (base-only components): only if G can observe the target (`g_observable`) AND G is confidently
              negative there (mean p_G < del_pG) AND L itself is not confident (mean p_L < del_pL).
    symmetric=True: add if adjacent & p_G > 0.5, delete if p_G < 0.5 (control).
    Returns (mask, n_added_px, n_deleted_px)."""
    out = base.copy()
    pG, pL = _sig(lG), _sig(np.nan_to_num(lL, nan=0.0))
    obsL = ~np.isnan(lL)
    near = detail.dilate(base, adj_r)
    added = deleted = 0
    gm = lG > 0
    for c in components(gm & ~base):
        if not (c & near).any():
            continue
        if symmetric:
            ok = pG[c].mean() > 0.5
        else:
            co = c & obsL
            ok = (pL[co].mean() if co.any() else 1.0) >= add_pL
        if ok:
            out |= c
            added += int(c.sum())
    if symmetric or g_observable:
        for c in components(base & ~gm):
            if symmetric:
                ok = pG[c].mean() < 0.5
            else:
                ok = pG[c].mean() < del_pG and pL[c].mean() < del_pL
            if ok:
                out &= ~c
                deleted += int(c.sum())
    return out, added, deleted


def instance_union(base, mG, adj_r=5):
    """F4: union with only those G components that touch the (dilated) base instance; distant G regions rejected."""
    out = base.copy()
    near = detail.dilate(base, adj_r)
    for c in components(mG & ~base):
        if (c & near).any():
            out |= c
    return out


def component_oracle(base, other, gt):
    """Offline bound for component-level fusion: flip each connected component of (base xor other) iff that
    increases J w.r.t. GT (greedy, components independent). Uses GT -> diagnostic only."""
    out = base.copy()
    for c in components(base ^ other):
        cand = out ^ c
        if _j(cand, gt) > _j(out, gt):
            out = cand
    return out


def _j(m, g):
    u = (m | g).sum()
    return 1.0 if u == 0 else (m & g).sum() / u
