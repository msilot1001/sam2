"""Self-gated global-local SAM 2 (SGGL) dual-state rollout with pluggable policies and side-effect-free probes (M7).

Consolidates scripts/m61.py:run_dual. Two (or more) frozen SAM 2 states run on one video:
  G : full frame (P5 prompt memory), lazily loaded (identity box == official frames bit-for-bit, tested in M6)
  L : fixed object-relative square crop (vcam) whose centre is set by the policy after every frame
  L2: optional second crop state at `scale2` x the L side (same centre trajectory), for cross-scale diagnostics
A *policy* sees only the current/past model outputs of frame t and returns the output mask, the next crop centre and
(optionally) which states admit frame t into their retrievable memory. GT is never passed to the rollout except the
frame-0 prompt; metrics are computed by the caller after the decision.

Probes run extra, NON-STORING predictions on frame t (other memory plan, or modified image features). They never
write into any output dict, and they restore the image-feature cache they touch (tests/test_m7.py proves that a
rollout with probes produces bit-identical production outputs to one without).
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as Fn

from memint import engine, planners
from spatial import vcam


@dataclass
class View:
    name: str
    out: dict                 # SAM 2 compact output of frame t (not stored by `predict`)
    logit: torch.Tensor       # 1x1x256x256 low-res logits (CPU fp32), exact source of the view-res mask
    mask: np.ndarray          # view-resolution bool mask (crop side for crop states)
    score: float              # object score logit
    iou: float                # decoder's predicted IoU of the returned mask
    box: tuple | None = None  # crop box in frame coords (None = full frame)
    pix: torch.Tensor | None = None   # memory-conditioned features 1x256x64x64 (GPU) if capture_pix() was installed


@dataclass
class State:
    st: dict
    od: dict
    frames: vcam.LazyCropFrames
    side: int | None          # None = full frame
    excluded: set = field(default_factory=set)   # non-cond frames hidden from retrieval (memory admission)


def _cond_only(frame_idx, plan, od):
    """Protected initial-reference memory: only the prompt frame's spatial memory and object pointer."""
    return {**plan, "spatial": [e for e in plan["spatial"] if e[1] == "cond"],
            "pointers": [e for e in plan["pointers"] if e[1] == "cond"]}


@torch.inference_mode()
def start_dir(pred, vdir, gt0, p5: bool):
    """Same as scripts/m56.py:start_dir: init_state (video offloaded to CPU) + GT prompt at 0 (+ P5) + preflight."""
    state = pred.init_state(video_path=str(vdir), offload_video_to_cpu=True)
    pred.add_new_mask(state, frame_idx=0, obj_id=1, mask=gt0)
    if p5:
        out = state["temp_output_dict_per_obj"][0]["cond_frame_outputs"][0]
        hi = state["mask_inputs_per_obj"][0][0].to(state["device"]) * 20.0 - 10.0
        out["maskmem_features"], out["maskmem_pos_enc"] = pred._run_memory_encoder(
            state, 0, 1, hi, out["object_score_logits"], is_mask_from_pts=True)
    pred.propagate_in_video_preflight(state)
    return state, state["output_dict_per_obj"][0]


class Dual:
    """Live states for one video. All coordinates are original-frame (H, W) unless noted."""

    def __init__(self, pred, paths, gt0, side=None, scale2=None, official=False):
        self.pred, self.paths, self.hw = pred, list(paths), gt0.shape
        self.side = vcam.crop_side(gt0) if side is None else side
        self._iou = [0.0]
        _hook_iou(pred, self._iou)
        H, W = self.hw
        self.S = {"G": self._start(start_dir, gt0, None, None, p5=True)}
        if official:
            self.S["B0"] = self._start(start_dir, gt0, None, None, p5=False)
        self.center = vcam.center_of(gt0)
        if self.side is not None:
            self.S["L"] = self._start(start_dir, gt0, self.side, self.center, p5=True)
            if scale2:
                self.side2 = min(int(round(self.side * scale2)), min(H, W))
                self.S["L2"] = self._start(start_dir, gt0, self.side2, self.center, p5=True)

    def _start(self, start_dir, gt0, side, center, p5):
        H, W = self.hw
        fr = vcam.LazyCropFrames(self.paths)
        fr.box = (0, H, 0, W) if side is None else vcam.place(center, side, self.hw)
        y0, y1, x0, x1 = fr.box
        res = (H, W) if side is None else (side, side)
        with vcam.lazy_loader(fr, res):
            st, od = start_dir(self.pred, "lazy", gt0[y0:y1, x0:x1], p5=p5)
        return State(st, od, fr, side)

    def set_center(self, center):
        """Place every crop state for the NEXT frame (call before `view`)."""
        if center is not None:
            self.center = center
        for s in self.S.values():
            if s.side is not None:
                s.frames.box = vcam.place(self.center, s.side, self.hw)

    def view(self, name, t, planner=None, store=True) -> View:
        """Predict frame t on state `name`. planner=None -> admission-only plan (native if nothing excluded)."""
        s = self.S[name]
        if planner is None and s.excluded:
            planner = planners.backfill(frozenset(s.excluded), frozenset(s.excluded), L=32)
        self.pred._memint_planner = planner
        try:
            out, mask, _, score = engine.predict(self.pred, s.st, s.od, t)
        finally:
            self.pred._memint_planner = None
        if store:
            s.od["non_cond_frame_outputs"][t] = out
        pix, self.pred._m7_pix = getattr(self.pred, "_m7_pix", None), None
        return View(name, out, out["pred_masks"].float().cpu(), mask, score, self._iou[0],
                    None if s.side is None else s.frames.box, pix)

    def probe(self, name, t, kind, trusted=None, image=None) -> View:
        """Non-storing diagnostic prediction of frame t on state `name` (after its production `view`).
        kind: 'protected' (prompt memory only), 'trusted' (non-cond memory restricted to `trusted` frames),
              'image' (same memory, image features recomputed from the given 1x3x1024x1024 tensor)."""
        s = self.S[name]
        if kind == "protected":
            return self.view(name, t, planner=_cond_only, store=False)
        if kind == "trusted":
            hide = frozenset(set(s.od["non_cond_frame_outputs"]) - set(trusted))
            return self.view(name, t, planner=planners.backfill(hide, hide, L=32), store=False)
        if kind == "image":
            cache = s.st["cached_features"]
            saved = dict(cache)
            try:
                with torch.inference_mode():
                    s.st["cached_features"] = {t: (image, self.pred.forward_image(image.to(s.st["device"])))}
                return self.view(name, t, store=False)
            finally:
                s.st["cached_features"] = saved
        raise ValueError(kind)

    def features(self, name, t):
        """Backbone stride-16 vision features (1x256x64x64, GPU) of frame t from the state's cache."""
        img, bb = self.S[name].st["cached_features"][t]
        return bb["backbone_fpn"][2]

    def close(self):
        for s in self.S.values():
            self.pred.reset_state(s.st)


def capture_pix(pred):
    """Keep the last memory-conditioned feature map (decoder input) on pred._m7_pix (outputs untouched)."""
    if getattr(pred, "_m7_capture", False):
        return
    orig = pred._prepare_memory_conditioned_features

    def f(*a, **k):
        out = orig(*a, **k)
        pred._m7_pix = out
        return out
    pred._prepare_memory_conditioned_features = f
    pred._m7_capture = True


def _hook_iou(pred, box):
    """Record the decoder's predicted IoU of the mask SAM 2 returns (best of multimask or single)."""
    if getattr(pred, "_m7_iou_box", None) is not None:
        pred._m7_iou_box = box
        return
    orig = pred._forward_sam_heads
    pred._m7_iou_box = box

    def f(*a, **k):
        out = orig(*a, **k)
        ious = out[2]
        pred._m7_iou_box[0] = float(ious.max().item()) if ious.shape[-1] > 1 else float(ious.reshape(-1)[0].item())
        return out
    pred._forward_sam_heads = f


# ----------------------------------------------------------------------------- geometry
def crop_to_frame(logit256, box, hw):
    """256x256 crop logits -> frame-res logits (H, W), -inf-free: outside the crop = NaN (unobserved)."""
    y0, y1, x0, x1 = box
    out = np.full(hw, np.nan, np.float32)
    out[y0:y1, x0:x1] = Fn.interpolate(logit256, size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0, 0].numpy()
    return out


def full_to_frame(logit256, hw):
    return Fn.interpolate(logit256, size=tuple(hw), mode="bilinear", align_corners=False)[0, 0].numpy()


def frame_region_to_grid(box, hw, grid=64):
    """Crop box (frame coords) -> fractional box on a full-frame SAM 2 feature grid (G is resized to 1024x1024
    without aspect preservation, so x and y scale separately)."""
    H, W = hw
    y0, y1, x0, x1 = box
    return y0 * grid / H, y1 * grid / H, x0 * grid / W, x1 * grid / W


def sample_global_in_crop(featG, box, hw, out_size):
    """Bilinearly resample full-frame feature map (1xCx64x64) on the crop region -> 1xCxout_size^2 aligned with
    the crop view's own grid (cell centres matched; tested)."""
    H, W = hw
    y0, y1, x0, x1 = box
    s = out_size
    c = (torch.arange(s, dtype=torch.float32) + 0.5) / s                 # crop-normalised cell centres
    ys = (y0 + c * (y1 - y0)) / H * 2 - 1                                 # -> [-1, 1] in frame (align_corners=False)
    xs = (x0 + c * (x1 - x0)) / W * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None].to(featG.device, featG.dtype)
    return Fn.grid_sample(featG, grid, mode="bilinear", padding_mode="border", align_corners=False)


# ----------------------------------------------------------------------------- policies
def self_gate(L: View, hw, box):
    return bool(L.score > 0 and L.mask.any() and not vcam.touches_border(L.mask, box, hw))


def a2x(G: View, L: View, hw):
    """SGGL (M6 A2x): output pasted L iff the self gate passes, else G. Centre follows the output."""
    ok = self_gate(L, hw, L.box)
    return (vcam.paste(L.mask, L.box, hw) if ok else G.mask), ok


def a2v(G: View, L: View, hw):
    """SGGL-V (M6.1 A2v): self gate AND global object score > 0."""
    ok = self_gate(L, hw, L.box) and G.score > 0
    return (vcam.paste(L.mask, L.box, hw) if ok else G.mask), ok
