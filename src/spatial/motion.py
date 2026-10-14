"""M8 family C: motion evidence for SGGL rescue (causal: frames t-1, t only).

  Flow        RAFT-small (torchvision, FlyingChairs+Things weights; no VOS data) on a down-scaled full frame.
  propagate   forward-warp a point set / box of the last confident target position through the flow chain.
  camera      global (camera) translation = median flow outside the predicted target box (robust to the target).

Mask-free by design: the target state carried through flow is a box (centre + size), so a missing mask never stops
propagation and no GT is ever read.
"""
import numpy as np
import torch
import torch.nn.functional as Fn


class Flow:
    def __init__(self, device="cuda", long_side=640, iters=12):
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        self.net = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
        self.device, self.long, self.iters = device, long_side, iters

    @torch.inference_mode()
    def __call__(self, a, b):
        """a, b: 1x3xHxW frames in SAM 2 normalisation (any device). Returns flow a->b as (H, W, 2) np (dx, dy) in px."""
        H, W = a.shape[-2:]
        s = self.long / max(H, W)
        h, w = int(round(H * s / 8)) * 8, int(round(W * s / 8)) * 8
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        def prep(x):                                   # SAM 2 norm -> [0, 1] -> RAFT's [-1, 1]
            x = (x.to(self.device) * std + mean).clamp(0, 1)
            return Fn.interpolate(x, size=(h, w), mode="bilinear", align_corners=False) * 2 - 1
        f = self.net(prep(a), prep(b), num_flow_updates=self.iters)[-1]
        f = Fn.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        f[:, 0] *= W / w
        f[:, 1] *= H / h
        return f[0].permute(1, 2, 0).cpu().numpy()


def box_flow(flow, box, inner=0.5):
    """Median flow (dx, dy) inside the central `inner` fraction of a frame box (y0, y1, x0, x1)."""
    y0, y1, x0, x1 = box
    cy, cx, hh, hw = (y0 + y1) / 2, (x0 + x1) / 2, max((y1 - y0) * inner / 2, 1), max((x1 - x0) * inner / 2, 1)
    H, W = flow.shape[:2]
    a, b = int(max(cy - hh, 0)), int(min(cy + hh + 1, H))
    c, d = int(max(cx - hw, 0)), int(min(cx + hw + 1, W))
    if a >= b or c >= d:
        return np.zeros(2)
    return np.median(flow[a:b, c:d].reshape(-1, 2), 0)


def camera(flow, box=None, margin=2.0):
    """Camera translation (dx, dy): median flow outside `margin` x the target box (whole frame if box is None)."""
    H, W = flow.shape[:2]
    keep = np.ones((H, W), bool)
    if box is not None:
        y0, y1, x0, x1 = box
        cy, cx, hh, hw = (y0 + y1) / 2, (x0 + x1) / 2, (y1 - y0) * margin / 2, (x1 - x0) * margin / 2
        keep[int(max(cy - hh, 0)):int(min(cy + hh, H)), int(max(cx - hw, 0)):int(min(cx + hw, W))] = False
    v = flow[keep] if keep.any() else flow.reshape(-1, 2)
    return np.median(v[:: max(1, len(v) // 20000)], 0)


def move_box(box, dxy, hw):
    y0, y1, x0, x1 = box
    H, W = hw
    dx, dy = float(dxy[0]), float(dxy[1])
    return (min(max(y0 + dy, 0), H - 1), min(max(y1 + dy, 1), H), min(max(x0 + dx, 0), W - 1), min(max(x1 + dx, 1), W))


class Track:
    """Causal target state from SGGL's own outputs + flow: last confident box, raw velocity, flow-carried box."""

    def __init__(self, box0, t0=0):
        self.hist = [(t0, box0)]          # (t, box) of non-empty outputs
        self.fbox = box0                   # flow-propagated box (re-anchored on every non-empty output)
        self.cam = np.zeros(2)
        self.v = np.zeros(2)

    def step_flow(self, flow, hw):
        """Advance the carried box by the median flow inside it (call once per frame t with flow t-1 -> t)."""
        self.cam = camera(flow, self.fbox)
        self.v = box_flow(flow, self.fbox)                  # target displacement t-1 -> t, (dx, dy) px
        self.fbox = move_box(self.fbox, self.v, hw)

    def observe(self, t, box):
        if box is not None:
            self.hist.append((t, box))
            self.fbox = box

    @staticmethod
    def _c(b):
        return ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2)

    def last(self):
        return self.hist[-1]

    def size(self):
        b = self.hist[-1][1]
        return max(b[1] - b[0], b[3] - b[2])

    def velocity(self):
        """Per-frame (vy, vx) between the last two confident boxes (0 if fewer than two)."""
        if len(self.hist) < 2:
            return np.zeros(2)
        (ta, a), (tb, b) = self.hist[-2], self.hist[-1]
        return (np.array(self._c(b)) - np.array(self._c(a))) / max(tb - ta, 1)

    def extrapolate(self, t, k=1.0):
        tb, b = self.hist[-1]
        return tuple(np.array(self._c(b)) + k * self.velocity() * (t - tb))

    def flow_center(self):
        return self._c(self.fbox)


def _gray_small(frame, f=4):
    """1x3xHxW SAM 2-normalised frame -> float32 numpy gray image down-sampled by f (area)."""
    import torch.nn.functional as F_
    g = frame.mean(1, keepdim=True)
    g = F_.avg_pool2d(g, f) if min(g.shape[-2:]) >= 4 * f else g
    return g[0, 0].float().cpu().numpy()


def phase_shift(a, b, f=4):
    """M9 cheap motion: global translation a -> b by phase correlation on f-times down-sampled gray frames.
    Returns (dx, dy) in full-resolution px. No learned model; a few ms on CPU."""
    A, B = _gray_small(a, f), _gray_small(b, f)
    h, w = A.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    FA, FB = np.fft.fft2((A - A.mean()) * win), np.fft.fft2((B - B.mean()) * win)
    R = FB * np.conj(FA)
    R /= np.abs(R) + 1e-9
    r = np.fft.ifft2(R).real
    py, px = np.unravel_index(np.argmax(r), r.shape)
    # sub-pixel refinement (parabolic) on the wrapped peak
    def sub(c, n, m1, m0, p1):
        den = m1 - 2 * m0 + p1
        return c + (0.5 * (m1 - p1) / den if abs(den) > 1e-12 else 0.0)
    y = sub(py, h, r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
    x = sub(px, w, r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])
    y = y - h if y > h / 2 else y
    x = x - w if x > w / 2 else x
    return np.array([x * f, y * f])


class ScaleCtl:
    """M9 conservative crop-scale controller. side tracks 4 x (long side of the latest reliable output box), floored
    at the M6 minimum (min(H, W) / MAX_ZOOM) so tiny targets keep the SGGL crop; bounded EMA (alpha) and bounded
    per-frame change [lo, hi]; held while the output is empty or unreliable."""

    def __init__(self, side0, hw, alpha=0.3, lo=0.8, hi=1.25, k=4.0, max_zoom=4.0):
        self.side, self.hw = float(side0), hw
        self.alpha, self.lo, self.hi, self.k = alpha, lo, hi, k
        self.floor = min(hw) / max_zoom

    def update(self, out_box, reliable):
        if out_box is not None and reliable:
            b = max(out_box[1] - out_box[0], out_box[3] - out_box[2])
            target = min(max(self.k * b, self.floor), min(self.hw))
            new = self.side + self.alpha * (target - self.side)
            self.side = float(np.clip(new, self.lo * self.side, self.hi * self.side))
        return self.side
