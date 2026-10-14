"""S family evidence: tracking state, global observability, cross-view support and target identity (M7).

Everything here is causal: it reads only frame-t observations and a running history of earlier frames' model
outputs/decisions. No GT. Thresholds are fixed constants defined here and frozen in commit 274e4d7 before any replay output.
"""
from collections import deque

import numpy as np
import torch
import torch.nn.functional as Fn

from spatial import detail, vcam

G_MIN_SIDE = 24      # px in G's 1024 input: below ~1.5 stride-16 cells G is treated as unable to observe the target
G_REL_WIN = 10       # recent frames for G's tracking reliability
G_REL_MIN = 0.5      # G positive on >= half of the recent frames where the output had a target
Z_MIN = -3.0         # identity: candidate similarity more than 3 sd below its own accepted history -> mismatch
SD_FLOOR = 0.05


def pool_map(sim64, m_view):
    """Area-weighted mean of a 64x64 map under a view-resolution mask (None if empty)."""
    if not m_view.any():
        return None
    w = Fn.interpolate(torch.as_tensor(m_view, dtype=torch.float32)[None, None], size=sim64.shape, mode="area")[0, 0].numpy()
    return float((sim64 * w).sum() / w.sum())


def identity(sim64, m_view, excl_r=2):
    """(in-mask similarity to the frozen frame-0 reference, distractor margin = in-mask minus best cell outside the
    dilated candidate). sim64 is the cosine map in the SAME view as m_view."""
    s_in = pool_map(sim64, m_view)
    if s_in is None:
        return None, None
    cells = Fn.interpolate(torch.as_tensor(m_view, dtype=torch.float32)[None, None], size=sim64.shape, mode="area")[0, 0].numpy() > 0
    out = ~detail.dilate(cells, excl_r)
    return s_in, (s_in - float(sim64[out].max())) if out.any() else 1.0


class Hist:
    """Running per-unit history shared by all policies (each policy owns one instance)."""

    def __init__(self, gt0_box_side):
        self.lost = False               # previous output empty -> next present candidate is a re-acquisition
        self.reacq = False              # re-acquisition not yet confirmed
        self.prev_out = None
        self.last_side = gt0_box_side   # target box side (frame px) of the last non-empty output
        self.g_pos = deque(maxlen=G_REL_WIN)
        self.sims = []                  # identity similarity of accepted L candidates
        self.l_run = 0                  # consecutive frames with a self-gate-valid L candidate
        self.trusted = []

    # ---- evidence
    def g_observable(self, hw):
        H, W = hw
        side_g = self.last_side * 1024.0 / max(H, W)          # conservative: the coarser axis
        rel = (np.mean(self.g_pos) >= G_REL_MIN) if len(self.g_pos) else True
        return bool(side_g >= G_MIN_SIDE and rel)

    def id_z(self, s):
        if s is None or len(self.sims) < 3:
            return 0.0
        mu, sd = float(np.mean(self.sims)), max(float(np.std(self.sims)), SD_FLOOR)
        return (s - mu) / sd

    # ---- update after the decision of frame t
    def update(self, out, selfok, accepted_L, sG, s_id, agree):
        nonempty = bool(out.any())
        if nonempty:
            b = vcam.bbox(out)
            self.last_side = max(b[1] - b[0], b[3] - b[2])
            self.g_pos.append(sG > 0)
        if self.lost and nonempty:
            self.reacq = True
        if self.reacq and accepted_L and agree:
            self.reacq = False
        self.lost = not nonempty
        self.l_run = self.l_run + 1 if selfok else 0
        if accepted_L and s_id is not None:
            self.sims.append(s_id)
        self.prev_out = out
