"""M8 rescue-candidate generation for SGGL (new evidence when both views fail).

Every function here is CAUSAL (frame-0 prompt + current/past frames only) and NON-STORING: a probe predicts frame t
on an existing SAM 2 state (its memory unchanged) with optional point prompts and/or a replaced input image, and
restores every cache it touches. Candidates are returned in original-frame coordinates.

Pieces (one per research family; the scheduler in scripts/m8.py decides which run):
  heat_proto     A  : frame-0 fg prototype -> cosine heatmap on current full-frame features (global search)
  heat_contrast  B  : multi-token softmax attention of each cell over frame-0 fg (k-means) + bg tokens -> fg posterior
  peaks          A/B: top-k peaks with NMS -> point prompts
  axis_map       F  : separable 1-D view maps: plain crop (OCLR) or context-preserving foveation (CPFR)
  view_image     F  : 1x3x1024x1024 SAM 2 input for a view map; `view_to_frame` maps 256^2 logits back
  probe          G/E: non-storing prediction on a state with point prompts and/or a view image
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as Fn

from memint import engine

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
M = 1024          # SAM 2 input side
STRIDE = 16       # top-level feature stride (64x64 grid at 1024)


# ----------------------------------------------------------------------------- separable view geometry
@dataclass
class AxisMap:
    """Monotone piecewise-linear map input coordinate (px, continuous) <-> output coordinate (px of the 1024 view)."""
    xin: np.ndarray    # knots in input px (increasing)
    xout: np.ndarray   # knots in output px (increasing)

    def fwd(self, x):
        return np.interp(x, self.xin, self.xout, left=np.nan, right=np.nan)

    def inv(self, o):
        return np.interp(o, self.xout, self.xin)

    def mag(self, x):
        """Output px per input px at x."""
        i = np.clip(np.searchsorted(self.xin, x) - 1, 0, len(self.xin) - 2)
        return (self.xout[i + 1] - self.xout[i]) / (self.xin[i + 1] - self.xin[i])


def axis_map(n, c, alpha, fovea=False, m=M):
    """View map for one axis of length n px centred at c with magnification alpha (output px per input px).
    fovea=False: plain crop of width m/alpha (shifted inside [0, n]; if wider than n -> whole axis, identity scale m/n).
    fovea=True : window of width m/(2 alpha) around c at magnification alpha (fovea = half the view); the rest of the
                 axis is compressed linearly (same factor both sides) into the remaining output so the whole axis stays
                 visible (context preserved). Falls back to the plain full-axis map if alpha <= m/n."""
    base = m / n
    if alpha <= base:
        return AxisMap(np.array([0.0, n]), np.array([0.0, m]))
    if not fovea:
        w = m / alpha
        x0 = min(max(c - w / 2, 0.0), n - w)
        return AxisMap(np.array([x0, x0 + w]), np.array([0.0, m]))
    h = m / (4 * alpha)                                      # fovea half-width in input px (fovea = m/2 output px)
    a0, a1 = c - h, c + h
    a0, a1 = (0.0, 2 * h) if a0 < 0 else ((n - 2 * h, float(n)) if a1 > n else (a0, a1))
    beta = (m - 2 * h * alpha) / (n - 2 * h)                  # periphery compression (same both sides)
    o0 = beta * a0
    return AxisMap(np.array([0.0, a0, a1, float(n)]), np.array([0.0, o0, o0 + 2 * h * alpha, float(m)]))


def view_grid(my: AxisMap, mx: AxisMap, hw, m=M):
    """grid_sample grid (1 x m x m x 2, align_corners=False) sampling the frame at the view's output pixel centres."""
    H, W = hw
    o = np.arange(m) + 0.5
    ys, xs = my.inv(o) / H * 2 - 1, mx.inv(o) / W * 2 - 1
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return torch.from_numpy(np.stack([gx, gy], -1)[None].astype(np.float32))


def frame_tensor(path):
    """Full frame as normalised float tensor 1x3xHxW (same normalisation as SAM 2's loader)."""
    from PIL import Image
    im = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).float() / 255.0
    return ((im - MEAN) / STD)[None]


def view_image(frame, my, mx):
    """1x3x1024x1024 SAM 2 input of the view (bilinear; frame 1x3xHxW on any device)."""
    g = view_grid(my, mx, frame.shape[-2:]).to(frame.device)
    return Fn.grid_sample(frame, g, mode="bilinear", padding_mode="border", align_corners=False)


def view_to_frame(logit256, my, mx, hw):
    """View logits (1x1x256x256) -> frame-res logits (H, W); frame pixels outside the view -> NaN (unobserved)."""
    H, W = hw
    oy, ox = my.fwd(np.arange(H) + 0.5), mx.fwd(np.arange(W) + 0.5)
    vy, vx = ~np.isnan(oy), ~np.isnan(ox)
    out = np.full(hw, np.nan, np.float32)
    if not vy.any() or not vx.any():
        return out
    gy = oy[vy] / M * 2 - 1
    gx = ox[vx] / M * 2 - 1
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    g = torch.from_numpy(np.stack([xx, yy], -1)[None].astype(np.float32))
    s = Fn.grid_sample(logit256.float().cpu(), g, mode="bilinear", padding_mode="border", align_corners=False)[0, 0].numpy()
    out[np.ix_(vy, vx)] = s
    return out


def frame_to_view_xy(pts, my, mx):
    """(N, 2) frame (x, y) px -> view (x, y) px (1024 space)."""
    pts = np.asarray(pts, np.float64)
    return np.stack([mx.fwd(pts[:, 0]), my.fwd(pts[:, 1])], 1)


def full_maps(hw):
    H, W = hw
    return AxisMap(np.array([0.0, H]), np.array([0.0, M])), AxisMap(np.array([0.0, W]), np.array([0.0, M]))


def box_maps(box):
    y0, y1, x0, x1 = box
    return AxisMap(np.array([float(y0), y1]), np.array([0.0, M])), AxisMap(np.array([float(x0), x1]), np.array([0.0, M]))


def occupancy_alpha(d_hat, cells):
    """Magnification that makes a target of long side d_hat input px span `cells` stride-16 feature cells."""
    return cells * STRIDE / max(d_hat, 1.0)


# ----------------------------------------------------------------------------- target-conditioned heatmaps
def proto(feat, m):
    """Mask-pooled L2-normalised vector (feat 1xCxhxw, m frame-res bool) or None."""
    w = Fn.interpolate(torch.as_tensor(m, dtype=torch.float32)[None, None], size=feat.shape[-2:], mode="area").to(feat.device)
    if w.sum() <= 0:
        return None
    return Fn.normalize((feat * w).sum((-1, -2)) / w.sum(), dim=1)[0]


def heat_proto(feat, ref):
    """Cosine of every cell to the frame-0 prototype (h x w)."""
    return (Fn.normalize(feat, dim=1)[0] * ref[:, None, None]).sum(0)


def tokens(feat, m, bg, k_fg=8, k_bg=32, iters=10, seed=0):
    """Frame-0 reference tokens: k-means centroids of fg cells (soft-mask weight > 0.5, else the max-weight cells) and
    of bg cells. Returns (fg K1xC, bg K2xC), L2-normalised."""
    f = Fn.normalize(feat, dim=1)[0].flatten(1).T                                   # N x C
    def cells(mm):
        w = Fn.interpolate(torch.as_tensor(mm, dtype=torch.float32)[None, None], size=feat.shape[-2:], mode="area")[0, 0].flatten()
        idx = torch.nonzero(w > 0.5).flatten()
        return idx if len(idx) else torch.topk(w, min(4, int((w > 0).sum()) or 1)).indices
    return _kmeans(f[cells(m).to(f.device)], k_fg, iters, seed), _kmeans(f[cells(bg).to(f.device)], k_bg, iters, seed)


def _kmeans(x, k, iters, seed):
    k = min(k, len(x))
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        for j in range(k):
            if (a == j).any():
                c[j] = x[a == j].mean(0)
    return Fn.normalize(c, dim=1)


def heat_contrast(feat, fg, bg, tau=0.1):
    """Contrastive token attention: each cell attends over [fg; bg] tokens (softmax, temperature tau); returns the
    attention mass on fg tokens (h x w, in [0, 1]). Unlike heat_proto, a cell matching a bg token is suppressed."""
    f = Fn.normalize(feat, dim=1)[0]
    logits = torch.einsum("kc,chw->khw", torch.cat([fg, bg]), f) / tau
    return torch.softmax(logits, 0)[: len(fg)].sum(0)


def peaks(hm, k=3, r=3, min_val=None):
    """Top-k local maxima of a h x w map with square NMS radius r (cells). Returns [(cy, cx, value)] in cells."""
    h = hm.clone().float()
    out = []
    for _ in range(k):
        v, i = h.flatten().max(0)
        if min_val is not None and v < min_val:
            break
        cy, cx = divmod(int(i), h.shape[1])
        out.append((cy, cx, float(v)))
        h[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1] = -1e9
    return out


def cell_to_frame(cy, cx, grid_hw, hw):
    """Cell centre -> frame (x, y) px for a full-frame grid (G resizes HxW to 1024^2 without aspect preservation)."""
    gh, gw = grid_hw
    H, W = hw
    return ((cx + 0.5) * W / gw, (cy + 0.5) * H / gh)


# ----------------------------------------------------------------------------- non-storing prompted / re-imaged probe
@torch.inference_mode()
def probe(pred, s, t, points=None, labels=None, image=None, planner=None, iou_box=None, feats=None):
    """Predict frame t on State `s` WITHOUT storing it. points: (N, 2) view (x, y) px in the 1024 input space of the
    image used (production view if image is None); labels: 1 pos, 0 neg, 2/3 box corners. image: 1x3x1024x1024 input
    replacing frame t's image for this call only; feats: its precomputed pred.forward_image(image) (shared between
    probes of the same view; bit-identical to recomputing). Returns (out dict, 1x1x256x256 CPU logits, score, iou)."""
    pi = None
    if points is not None and len(points):
        pi = {"point_coords": torch.as_tensor(np.asarray(points, np.float32))[None].to(s.st["device"]),
              "point_labels": torch.as_tensor(np.asarray(labels, np.int32))[None].to(s.st["device"])}
    cache = s.st["cached_features"]
    saved = dict(cache)
    pred._memint_planner = planner
    try:
        if image is not None:
            s.st["cached_features"] = {t: (image, feats if feats is not None else pred.forward_image(image.to(s.st["device"])))}
        out, _ = pred._run_single_frame_inference(
            inference_state=s.st, output_dict=s.od, frame_idx=t, batch_size=1, is_init_cond_frame=False,
            point_inputs=pi, mask_inputs=None, reverse=False, run_mem_encoder=True)
    finally:
        pred._memint_planner = None
        s.st["cached_features"] = saved
    return (out, out["pred_masks"].float().cpu(), float(out["object_score_logits"].float().cpu().item()),
            None if iou_box is None else iou_box[0])
