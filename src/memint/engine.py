"""Explicit frame-by-frame SAM 2 propagation with cloneable memory state (M5-Pre).

`propagate_in_video` is a generator over a mutable inference_state, so we step frames ourselves with
the official `_run_single_frame_inference` (the same call the official loop makes) and keep the
per-object output dict explicit. Branches clone that dict (new dicts, shared tensors: SAM 2 never
mutates a stored frame output in place), so interventions in one branch cannot reach another.
"""
import time

import numpy as np
import torch


@torch.inference_mode()
def start(predictor, video_dir, prompt_mask: np.ndarray):
    """init_state + GT prompt at video_idx 0 + official preflight. Returns (state, output_dict)."""
    if prompt_mask.dtype != bool or not prompt_mask.any():
        raise ValueError("prompt mask must be a non-empty bool array")
    state = predictor.init_state(video_path=str(video_dir))
    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=prompt_mask)
    predictor.propagate_in_video_preflight(state)
    return state, state["output_dict_per_obj"][0]


def clone(output_dict: dict) -> dict:
    return {"cond_frame_outputs": dict(output_dict["cond_frame_outputs"]),
            "non_cond_frame_outputs": dict(output_dict["non_cond_frame_outputs"])}


@torch.inference_mode()
def predict(predictor, state, output_dict, frame_idx: int):
    """One frame with whatever planner is installed; does NOT store the result.

    Returns (current_out, mask bool HxW, prob HxW float32, object_score_logit)."""
    current_out, pred_masks_gpu = predictor._run_single_frame_inference(
        inference_state=state, output_dict=output_dict, frame_idx=frame_idx, batch_size=1,
        is_init_cond_frame=False, point_inputs=None, mask_inputs=None, reverse=False, run_mem_encoder=True)
    _, video_res = predictor._get_orig_video_res_output(state, pred_masks_gpu)
    logits = video_res[0, 0]
    return (current_out, (logits > 0.0).cpu().numpy(), torch.sigmoid(logits).float().cpu().numpy(),
            float(current_out["object_score_logits"].float().cpu().item()))


@torch.inference_mode()
def prompt_mask_output(predictor, state):
    """The official output for the prompted frame 0 (as propagate_in_video yields it)."""
    out = state["output_dict_per_obj"][0]["cond_frame_outputs"][0]
    _, video_res = predictor._get_orig_video_res_output(state, out["pred_masks"].to(state["device"]))
    return (video_res[0, 0] > 0.0).cpu().numpy()


@torch.inference_mode()
def rollout(predictor, state, output_dict, first: int, last: int, record=None):
    """Predict and STORE frames first..last into output_dict (in place). Returns {idx: mask}."""
    masks, lat = {}, {}
    for t in range(first, last + 1):
        t0 = time.perf_counter()
        out, m, prob, score = predict(predictor, state, output_dict, t)
        output_dict["non_cond_frame_outputs"][t] = out
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat[t] = time.perf_counter() - t0
        masks[t] = m
        if record is not None:
            record[t] = {"obj_score": score, "area": int(m.sum())}
    return masks, lat
