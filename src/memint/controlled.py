"""Controlled SAM 2 memory retrieval for causal interventions (M5-Pre).

`_prepare_memory_conditioned_features` below is a copy of the official SAM 2 implementation
(facebookresearch/sam2 @ 2b90b9f, sam2/modeling/sam2_base.py, Copyright (c) Meta Platforms, Inc.,
Apache-2.0) with ONE change: the lists of (temporal position, source frame) for spatial memories and
object pointers are produced by `native_plan(...)` and then passed through an optional planner.
With no planner the code path is the official one (verified bit-identical in tests). The function is
bound to a predictor INSTANCE; the installed upstream files are never modified.

Plan format
  spatial:  list of (t_pos, store, frame_idx)  store in {"cond", "non_cond"}; t_pos 0 = conditioning
  pointers: list of (pos, store, frame_idx)    pos = signed frame distance used for the pointer sine PE
"""
import types

import torch

from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames


def native_plan(model, frame_idx, output_dict, num_frames, track_in_reverse=False):
    """Exactly the official retrieval choices, expressed as a plan."""
    cond_outputs = output_dict["cond_frame_outputs"]
    selected_cond, unselected_cond = select_closest_cond_frames(frame_idx, cond_outputs, model.max_cond_frames_in_attn)
    spatial = [(0, "cond", t) for t in selected_cond]
    stride = 1 if model.training else model.memory_temporal_stride_for_eval
    for t_pos in range(1, model.num_maskmem):
        t_rel = model.num_maskmem - t_pos
        if t_rel == 1:
            prev = frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
        elif not track_in_reverse:
            prev = ((frame_idx - 2) // stride) * stride - (t_rel - 2) * stride
        else:
            prev = -(-(frame_idx + 2) // stride) * stride + (t_rel - 2) * stride
        if prev in output_dict["non_cond_frame_outputs"]:
            spatial.append((t_pos, "non_cond", prev))
        elif prev in unselected_cond:
            spatial.append((t_pos, "cond", prev))
    pointers = []
    if model.use_obj_ptrs_in_encoder:
        sign = -1 if track_in_reverse else 1
        max_ptrs = min(num_frames, model.max_obj_ptrs_in_encoder)
        if not model.training and model.only_obj_ptrs_in_the_past_for_eval:
            ptr_cond = {t: o for t, o in selected_cond.items() if (t >= frame_idx if track_in_reverse else t <= frame_idx)}
        else:
            ptr_cond = selected_cond
        for t in ptr_cond:
            pos = (frame_idx - t) * sign if model.use_signed_tpos_enc_to_obj_ptrs else abs(frame_idx - t)
            pointers.append((pos, "cond", t))
        for t_diff in range(1, max_ptrs):
            t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
            if t < 0 or (num_frames is not None and t >= num_frames):
                break
            if t in output_dict["non_cond_frame_outputs"]:
                pointers.append((t_diff, "non_cond", t))
            elif t in unselected_cond:
                pointers.append((t_diff, "cond", t))
    return {"spatial": spatial, "pointers": pointers, "max_ptrs": min(num_frames, model.max_obj_ptrs_in_encoder)}


def _fetch(output_dict, store, t):
    return output_dict["cond_frame_outputs" if store == "cond" else "non_cond_frame_outputs"][t]


def _prepare_memory_conditioned_features(self, frame_idx, is_init_cond_frame, current_vision_feats,
                                         current_vision_pos_embeds, feat_sizes, output_dict, num_frames,
                                         track_in_reverse=False):
    B = current_vision_feats[-1].size(1)
    C = self.hidden_dim
    H, W = feat_sizes[-1]
    device = current_vision_feats[-1].device
    if self.num_maskmem == 0:
        return current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

    num_obj_ptr_tokens = 0
    if not is_init_cond_frame:
        to_cat_memory, to_cat_memory_pos_embed = [], []
        assert len(output_dict["cond_frame_outputs"]) > 0
        plan = native_plan(self, frame_idx, output_dict, num_frames, track_in_reverse)
        if self._memint_planner is not None:
            plan = self._memint_planner(frame_idx, plan, output_dict)
        if self._memint_log is not None:
            self._memint_log[frame_idx] = {"spatial": [(p, s, t) for p, s, t in plan["spatial"]],
                                           "pointers": [(p, s, t) for p, s, t in plan["pointers"]]}
        for t_pos, store, t in plan["spatial"]:
            prev = _fetch(output_dict, store, t)
            feats = prev["maskmem_features"].to(device, non_blocking=True)
            to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
            maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
            maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
            maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
            to_cat_memory_pos_embed.append(maskmem_enc)

        if self.use_obj_ptrs_in_encoder:
            max_obj_ptrs_in_encoder = plan["max_ptrs"]
            pos_and_ptrs = [(pos, _fetch(output_dict, store, t)["obj_ptr"]) for pos, store, t in plan["pointers"]]
            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list = zip(*pos_and_ptrs)
                obj_ptrs = torch.stack(ptrs_list, dim=0)
                if self.add_tpos_enc_to_obj_ptrs:
                    t_diff_max = max_obj_ptrs_in_encoder - 1
                    tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                    obj_pos = torch.tensor(pos_list).to(device=device, non_blocking=True)
                    obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                    obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                    obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                else:
                    obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                if self.mem_dim < C:
                    obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                to_cat_memory.append(obj_ptrs)
                to_cat_memory_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0
    else:
        if self.directly_add_no_mem_embed:
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            return pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
        to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

    memory = torch.cat(to_cat_memory, dim=0)
    memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
    pix_feat_with_mem = self.memory_attention(curr=current_vision_feats, curr_pos=current_vision_pos_embeds,
                                              memory=memory, memory_pos=memory_pos_embed,
                                              num_obj_ptr_tokens=num_obj_ptr_tokens)
    return pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)


def install(predictor, planner=None, log: dict | None = None):
    """Bind the controlled retrieval to this predictor instance (idempotent)."""
    predictor._memint_planner = planner
    predictor._memint_log = log
    predictor._prepare_memory_conditioned_features = types.MethodType(_prepare_memory_conditioned_features, predictor)
    return predictor


def uninstall(predictor):
    for a in ("_prepare_memory_conditioned_features", "_memint_planner", "_memint_log"):
        predictor.__dict__.pop(a, None)
