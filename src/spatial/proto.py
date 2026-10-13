"""M5.5 prototype (training-free, frozen SAM 2): image-guided upsampling of the decoder's 256x256 logits.

Official: video-res mask = bilinear(256x256 logits) > 0. Prototype: the bilinear probability map is filtered with a
guided filter (He et al., TPAMI 2013) whose guide is the CURRENT frame (grey, video resolution), so the upsampled
boundary can follow image edges finer than the decoder's stride-4 grid. Uses only frame t's image and frame t's
logits (causal). Two arms:
  out  -- output only (memory identical to official)
  mem  -- additionally re-encode frame t's memory from the refined map (tests the effect on FUTURE frames)
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window, edge windows normalised by their valid size."""
    k = 2 * r + 1
    num = F.avg_pool2d(x, k, 1, r, count_include_pad=False)
    return num


def guided_filter(I: torch.Tensor, p: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """Grey-guide guided filter. I, p: 1x1xHxW float."""
    mI, mp = box(I, r), box(p, r)
    cov = box(I * p, r) - mI * mp
    var = box(I * I, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a, r) * I + box(b, r)


def grey(path, device) -> torch.Tensor:
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(g)[None, None].to(device)


def refined_prob(logits: torch.Tensor, guide: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """256x256 logits -> guided-filtered probability at the guide's (video) resolution, in [0, 1]."""
    p = torch.sigmoid(F.interpolate(logits.float(), size=guide.shape[-2:], mode="bilinear", align_corners=False))
    return guided_filter(guide, p, r, eps).clamp(0, 1)


@torch.inference_mode()
def predict(predictor, state, output_dict, frame_idx: int, guide: torch.Tensor, r: int, eps: float, feed_memory: bool):
    """Like memint.engine.predict but with the guided output; returns (current_out, mask, official_mask)."""
    current_out, pm = predictor._run_single_frame_inference(
        inference_state=state, output_dict=output_dict, frame_idx=frame_idx, batch_size=1,
        is_init_cond_frame=False, point_inputs=None, mask_inputs=None, reverse=False, run_mem_encoder=True)
    pm = pm.to(state["device"])
    q = refined_prob(pm, guide, r, eps)
    if feed_memory:
        hi = F.interpolate(q, size=(predictor.image_size, predictor.image_size), mode="bilinear", align_corners=False)
        hi = torch.logit(hi.clamp(1e-4, 1 - 1e-4))
        feats, pos = predictor._run_memory_encoder(state, frame_idx, 1, hi, current_out["object_score_logits"],
                                                   is_mask_from_pts=False)
        current_out = {**current_out, "maskmem_features": feats, "maskmem_pos_enc": pos}
    _, vr = predictor._get_orig_video_res_output(state, pm)
    return current_out, (q[0, 0] > 0.5).cpu().numpy(), (vr[0, 0] > 0).cpu().numpy()
