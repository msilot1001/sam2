"""Instrumented SAM 2 video inference with an optional causal memory-write gate.

Signals: the official video predictor keeps `object_score_logits` in each frame's compact output
but DISCARDS the decoder's predicted IoU. We record it with an instance-level wrapper around
`_forward_sam_heads` (no upstream file is changed; outputs are returned untouched, so predictions
are identical -- verified in tests and against M2).

Gate semantics (labelled intervention): when gate(signals_t, history) returns True after frame t
has been predicted, frame t's entry is removed from `non_cond_frame_outputs`. Later frames then see
a hole at that offset: neither its spatial memory nor its object pointer is retrieved (the memory
was still ENCODED; no older memory is substituted). The frame's own output mask is unchanged.
The prompt (conditioning) memory is never touched. Decisions use only frame t and earlier.
"""
import time
from pathlib import Path

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


class SignalHook:
    """Records per-frame decoder outputs (predicted IoUs of all candidate masks)."""

    def __init__(self, predictor):
        self.p, self.cur, self.ious = predictor, None, {}
        self._orig_track = predictor.track_step
        self._orig_heads = predictor._forward_sam_heads

        def track_step(*a, **k):
            self.cur = k.get("frame_idx", a[0] if a else None)
            return self._orig_track(*a, **k)

        def heads(*a, **k):
            out = self._orig_heads(*a, **k)
            if self.cur is not None:
                self.ious[self.cur] = out[2].detach().float().cpu().numpy().ravel()
            return out

        predictor.track_step, predictor._forward_sam_heads = track_step, heads

    def remove(self):
        del self.p.track_step, self.p._forward_sam_heads   # back to the class methods


def run_unit(predictor, video_dir: Path, prompt_mask: np.ndarray, gate=None, instrument: bool = True) -> dict:
    """Prompt with the GT mask at video_idx 0 only, propagate forward, optionally gate memory writes.

    Returns masks {idx: bool}, per-frame signals, gate decisions and latency.
    """
    import torch
    if prompt_mask.dtype != bool or not prompt_mask.any():
        raise ValueError("prompt mask must be a non-empty bool array")
    hook = SignalHook(predictor) if instrument else None
    masks, sig, skipped, lat = {}, {}, [], {}
    try:
        with torch.inference_mode():
            state = predictor.init_state(video_path=str(video_dir))
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=prompt_mask)
            store = state["output_dict_per_obj"][0]["non_cond_frame_outputs"]
            prev = None
            hist = []
            t = time.perf_counter()
            for idx, _, logits in predictor.propagate_in_video(state):
                m = (logits[0, 0] > 0.0).cpu().numpy()
                masks[idx] = m
                prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
                cur = store.get(idx) or state["output_dict_per_obj"][0]["cond_frame_outputs"].get(idx)
                ious = hook.ious.get(idx) if hook else None
                area = int(m.sum())
                s = {"video_idx": idx,
                     "obj_score": float(cur["object_score_logits"].float().cpu().item()),
                     "pred_iou": float(np.max(ious)) if ious is not None else np.nan,
                     "pred_iou_spread": float(np.max(ious) - np.min(ious)) if ious is not None else np.nan,
                     "area": area,
                     "mean_prob_in_mask": float(prob[m].mean()) if area else 0.0,
                     "consistency_iou_prev": _iou(m, prev) if prev is not None else np.nan,
                     "area_ratio_prev": (area + 1) / (int(prev.sum()) + 1) if prev is not None else np.nan}
                sig[idx] = s
                hist.append(s)
                if gate is not None and idx > 0 and idx in store and gate(s, hist[:-1]):
                    store.pop(idx)               # exclude frame idx's memory + obj pointer from retrieval
                    skipped.append(idx)
                prev = m
                now = time.perf_counter()
                lat[idx] = now - t
                t = now
            n = state["num_frames"]
            predictor.reset_state(state)
    finally:
        if hook:
            hook.remove()
    if sorted(masks) != list(range(n)):
        raise RuntimeError("propagation did not return every frame exactly once")
    return {"masks": masks, "signals": sig, "skipped": skipped, "latency": lat}
