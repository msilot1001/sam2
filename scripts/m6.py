"""M6: overnight research-direction tournament (CUDA FP32, official SAM 2.1 Hiera-T weights, frozen).

    uv run python scripts/m6.py prereg      # freeze candidates, sets, endpoints, gates (before any new output)
    uv run python scripts/m6.py track B1    # P5 reference with full signal logging (also Track B data)
    uv run python scripts/m6.py track A1    # causal fixed-scale virtual-camera crop (Track A)
    uv run python scripts/m6.py track A2    # global-local: A1 crop state + B1 global state, causal selection
    uv run python scripts/m6.py cscale      # Track C: retrospective image-mode scale/decoder diagnostic
    uv run python scripts/m6.py analyze     # reports -> outputs/milestone6/reports/

Reuses M5.5/M5.6 helpers (scripts/m55.py, scripts/m56.py). GT is read only for the frame-0 prompt and for evaluation;
every GT-derived quantity used for diagnosis is labelled retrospective. PUMaVOS C is never touched here.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.dropout_runner import atomic_csv, atomic_json, atomic_npz
from baseline.metrics import j_measure
from memint import controlled, engine
from spatial import detail, vcam


def _load(name):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


m56 = _load("m56")
m55 = m56.m55
OUT = Path("outputs/milestone6")
SETS = ("DAVIS_val", "PUMaVOS_B", "PUMaVOS_A", "SegTrack")
GATE_SETS = ("DAVIS_val", "PUMaVOS_B")          # M5.6 proposal §6: DAVIS val + PUMaVOS B

PREREG = {
    "created_before_any_m6_output": True,
    "model": "SAM 2.1 Hiera-T, official ckpt, CUDA FP32, TF32 off, video offloaded to CPU",
    "conventions": "GT mask prompt at video_idx 0, prompted frame excluded, M5.5 J/F (F 11px, F_strict 2px), original res",
    "sets": {"DAVIS_val": "61 objects (explored in M5.5/M5.6)", "PUMaVOS_A": "9 objects (explored in M5.6)",
             "PUMaVOS_B": "6 objects (development)", "SegTrack": "24 objects (auxiliary control)",
             "PUMaVOS_C": "11 objects, untouched; opened once only for one frozen candidate that passed its gate"},
    "small": "prompt-frame area < 1% of frame (m55.strata 'small'; unchanged since M5.5)",
    "unit_of_analysis": "object (unit) means over all evaluated frames; CIs = 2000x bootstrap clustered by video",
    "arms": {
        "B1": "P5 baseline (M5.5): full-res prompt memory; reference for every comparison",
        "A1": "virtual camera: square crop side s = max(box_h, box_w)/0.25 of the prompt box, clamped to >= min(H,W)/4; "
              "no crop (== B1) if s >= min(H,W). Centre = box centre of the previous OUTPUT mask (prompt GT at t=1); "
              "kept if empty. Crop shifted inside the frame. SAM 2 sees only crops (one scale), output pasted back.",
        "A2": "global-local: A1 crop state and B1 full-frame state run independently (no memory exchange). Frame t "
              "output = crop mask if crop obj score > 0, crop mask non-empty and not touching an interior crop edge; "
              "else the B1 mask. Next crop centre = box centre of this fused output (kept if empty).",
    },
    "A2_trigger": "run A2 only if A1 shows >= 3 small units with dJF >= +0.03 (opportunity) or >= 3 small units with "
                  "dJF <= -0.03 (context loss / escape)",
    "gate_A": {"set": "small units of DAVIS_val + PUMaVOS_B (vs B1)",
               "pass_all": ["mean small-unit dJF >= +0.03",
                            "fraction of small units with dJF < -0.01 below 20%",
                            "mean non-small-unit dJF >= -0.005 (DAVIS_val + PUMaVOS_B)"],
               "secondary_not_gating": "PUMaVOS_A, SegTrack, video-clustered CI, episodes, latency"},
    "track_B": {
        "event": "reappearance: GT visible (area >= 25% of the unit's median visible area) at frame t after >= 1 frame "
                 "that is GT-empty or below 25% of median; eligible frames = t..t+4",
        "absence_type": "out_of_frame if the last visible GT box before absence is within 2 px of the frame border, "
                        "else occluded; annotation-defined absence flagged by visual check (guitar in M5.6)",
        "online_opportunity": "existing SAM 2 output: best of the 3 multimask candidates J >= 0.5 while output J < 0.5 "
                              "(needs a selector); first-frame appearance: M5.6 I1 history reset resolves",
        "B3_trigger": ">= 4 non-annotation reappearance events in >= 3 videos where B1 J < 0.5 on the first eligible "
                      "frame AND an online opportunity exists; else Track B is underpowered and no prototype is built",
        "B3_gate": "if built: mean J over eligible frames +0.10 on triggering events, >= 3 videos improved, no unit "
                   "whole-sequence dJF < -0.01, overall mean dJF >= 0",
    },
    "track_C": {
        "diagnostic": "RETROSPECTIVE upper bound: SAM 2 image mode with GT box (+inner point), best of 3, on every 5th "
                      "GT-visible frame of small units, full frame vs square crop of the A1 side centred on the GT box; "
                      "also the 256 ceiling at both scales",
        "rule": "input scale is a fixable bottleneck if crop oracle - full oracle >= +0.05 mean and positive in >= 60% "
                "of small units; decoder-output bottleneck if 256-ceiling J < 0.8 on >= 25% of small-unit frames",
        "C3": "assessment only, no training in M6",
    },
    "confirmation": "PUMaVOS C evaluated at most once, only for exactly one frozen candidate that passed its gate; "
                    "all 11 units reported; nothing tuned afterwards",
}


PREREG_R2 = {
    "created_after": "round 1 (A1 failed gate_A: small dJF +0.017 < 0.03; A2 trigger not met; B3 trigger not met; "
                     "Track C rules not met). Round 2 tests mechanisms, frozen before any round-2 output.",
    "arms": {
        "A1z": "A1 with zoom target 0.5 (crop side = 2 x prompt box side, same clamp >= min(H,W)/4; no crop if >= min(H,W)). "
               "Dose test of the input-scale mechanism.",
        "A1v": "A1 with variable scale: crop side recomputed every frame from the previous OUTPUT box with A1's rule "
               "(target 0.25, clamp [min(H,W)/4, min(H,W)]); kept if output empty. Memory then mixes scales. "
               "Tests H3 (fixed-scale memory consistency matters).",
        "A2x": "A2 exactly as pre-registered in round 1, run as POST-HOC EXPLORATORY (its trigger was not met): "
               "measures whether the global fallback removes A1's crop-escape regressions. Cannot pass a gate.",
    },
    "gate": "A1z and A1v are judged by the round-1 gate_A unchanged; with 3 looks at the same development data any "
            "pass is provisional and still requires the PUMaVOS C confirmation",
    "H3_rule": "fixed-scale consistency supported if A1 - A1v small-unit dJF >= +0.01 with video-CI lower bound > 0; "
               "refuted if A1v >= A1",
    "dose_rule": "input-scale mechanism supported if image-oracle gain at the A1z scale > gain at the A1 scale on the "
                 "same frames and A1z small dJF > A1 small dJF",
    "C_dose": "Track C diagnostic repeated with the A1z crop side (cscale_frames_z.csv)",
}


PREREG_R3 = {
    "created_after": "round 2 (A1z harmful, H3 refuted, A2x no regressions but < gate). Frozen before computing it.",
    "candidate": "S1 self-assessed scale selection (training-free, causal): run arms {B1, A1, A1z} (fixed scales: full "
                 "frame, 4x and 2x prompt box) in parallel for frames 1..K, K=5; output B1 on frames 1..K; at frame K pick "
                 "the arm with the highest mean SAM 2 predicted IoU (iou_pred, max over 3 candidates) over frames 1..K, "
                 "ties -> B1; output that arm from K+1 on and drop the others. Units where an arm applies no crop use B1 "
                 "for it. Replayed exactly from the logged, deterministic, independent arm runs (one-time switch, no state "
                 "mixing).",
    "gate": "round-1 gate_A unchanged (small dJF >= +0.03, < 20% small units < -0.01, non-small mean >= -0.005) on "
            "DAVIS_val + PUMaVOS_B; this is the 4th look at the development data -> any pass stays provisional until C",
    "ceiling": "RETROSPECTIVE per-unit oracle over the same pool (best whole-sequence JF) reported as an upper bound only",
}


PREREG_R4 = {
    "created_after": "round 3 (S1 failed; per-unit scale-oracle ceiling ~+0.018). Frozen before any round-4 output.",
    "question": "Track C3: is there LEARNABLE headroom beyond SAM 2's frozen mask decoder for small targets, i.e. do "
                "the frozen stride-4/8 features carry information the decoder does not express?",
    "setting": "DIAGNOSTIC, retrospective prompt: SAM 2 image mode, GT box only, multimask_output=False, on the full "
               "frame; both arms get the identical prompt and frozen features. Not a deployable VOS method.",
    "head": "residual conv head on [logit, sigmoid(logit), s0 (32ch @256), up2(s1) (64ch), RGB@256] -> 3x3 conv 64, "
            "3x3 conv 64, 3x3 dilated(2) conv 64 (ReLU), 3x3 conv -> delta; output = logit + delta; last conv "
            "zero-initialised (untrained head == frozen decoder, tested). 1024 -> 256 grid as SAM 2.",
    "train": "DAVIS 2017 train split only (never val/PUMaVOS/SegTrack); every 2nd frame, every visible object; loss "
             "BCE + soft dice at 256 vs area-resized GT, objects with prompt-frame-independent area < 1% of frame "
             "weighted x4; Adam 1e-3, 2 epochs or 25 min wall cap, seed 0, frozen SAM 2 (no_grad).",
    "eval": "(a) small frames: every 5th GT-visible frame of every small unit (DAVIS val, PUMaVOS A+B, SegTrack) "
            "(b) ordinary frames: every 10th GT-visible frame of every non-small DAVIS val + PUMaVOS A+B unit; "
            "J/F at original resolution",
    "rule": "learnable decoder-side headroom supported if small-frame mean dJ >= +0.05 with >= 60% of small units "
            "positive and ordinary-frame mean dJ >= -0.01; otherwise the information is not recoverable by a light "
            "head on frozen features at this budget",
}


PREREG_R5 = {
    "created_after": "round 4 (all tracks failed their gates). Frozen before any A2c output.",
    "arm": "A2c = closest prior art's acceptance rule (Rozumnyi 2025, CTU thesis, parallel-window variant) inside the "
           "same framework as A2x: crop output accepted iff (global mask empty or IoU(crop, global) > 0.5) and "
           "crop-mask box area / crop area <= 0.1; else the B1 mask; next centre from the fused output. Same A1 scale.",
    "question": "is A2's self-gate (crop obj score > 0, not touching an interior crop edge) distinguishable from the "
                "agreement gate? Hypothesis: on targets where B1 is poor, agreement rejects useful crops (A2x > A2c).",
    "report": "small-unit dJF of A2c vs A2x (paired, video CI) and per unit; units with B1 JF < 0.8 listed separately. "
              "Exploratory comparison; no gate, C stays closed.",
}


def prereg_r5():
    m55._freeze(OUT / "preregistration_r5.json", {**PREREG_R5, "rows": []})


def prereg_r4():
    m55._freeze(OUT / "preregistration_r4.json", {**PREREG_R4, "rows": []})


def prereg_r3():
    m55._freeze(OUT / "preregistration_r3.json", {**PREREG_R3, "rows": []})


def prereg_r2():
    m55._freeze(OUT / "preregistration_r2.json", {**PREREG_R2, "rows": []})


def prereg():
    small = _small()
    rows = [{"unit": u.name, "set": s, "small": u.name in small, "crop_side": vcam.crop_side(gt(u, 0))} for s, u in units()]
    m55._freeze(OUT / "preregistration.json", {**PREREG, "rows": rows})


def _small():
    ps = ("outputs/milestone55/manifest.json", "outputs/milestone56/manifest.json", "outputs/milestone55/manifest_segtrack.json")
    return {r["unit"] for p in ps for r in json.loads(Path(p).read_text())["rows"] if r["small"]}


# ----------------------------------------------------------------------------- units
def units(sets=SETS):
    from rfeas.units import davis_units
    sp = {r["unit"]: r["split"] for r in json.loads((m56.OUT / "manifest.json").read_text())["rows"]}
    out = []
    for s in sets:
        if s == "DAVIS_val":
            out += [(s, u) for u in davis_units("val")[0]]
        elif s == "SegTrack":
            out += [(s, u) for u in m55.segtrack_units()]
        else:
            want = {"PUMaVOS_A": "A_exploration", "PUMaVOS_B": "B_development"}[s]
            out += [(s, u) for u in m56.puma_units() if sp[u.name] == want]
    assert not any(sp.get(u.name) == "C_confirmation" for _, u in out)
    return out


def gt(u, t):
    return m56.gt_of(u, u.frames[t])


def paths(u):
    return sorted(m56.frame_dir(u).glob("*.jpg"), key=lambda p: int(p.stem))[:len(u.frames)]


# ----------------------------------------------------------------------------- runs
class Heads:
    """Records the last decoder call's 3 candidate ious and high-res candidates (outputs untouched)."""

    def __init__(self, p):
        self.p, self.orig, self.last = p, p._forward_sam_heads, None

        def f(*a, **k):
            out = self.orig(*a, **k)
            self.last = (out[1].detach(), out[2].detach())
            return out
        p._forward_sam_heads = f

    def close(self):
        self.p.__dict__.pop("_forward_sam_heads", None)


def start_crop(pred, u, side, center):
    frames = vcam.LazyCropFrames(paths(u))
    hw = gt(u, 0).shape
    frames.box = vcam.place(center, side, hw)
    y0, y1, x0, x1 = frames.box
    with vcam.lazy_loader(frames, (side, side)):
        state, od = m56.start_dir(pred, "lazy", gt(u, 0)[y0:y1, x0:x1], p5=True)
    return state, od, frames


def step(pred, state, od, t, heads):
    t0 = time.perf_counter()
    out, m, _, score = engine.predict(pred, state, od, t)
    od["non_cond_frame_outputs"][t] = out
    torch.cuda.synchronize()
    multi, ious = heads.last
    return m, score, ious[0].float().cpu().numpy(), multi, time.perf_counter() - t0


def track(arm):
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    heads = Heads(pred)
    d = OUT / "runs" / arm
    d.mkdir(parents=True, exist_ok=True)
    for s, u in units():
        stem = u.name.replace("/", "_")
        if (d / f"{stem}.csv").exists():
            continue
        if arm != "B1" and vcam.crop_side(gt(u, 0), 0.5 if arm == "A1z" else vcam.TARGET) is None:
            continue                                   # no crop: A1 == A2 == B1 by construction (tested)
        df, masks = run_unit(pred, heads, s, u, arm)
        atomic_npz(d / f"{stem}.npz", **{arm: np.packbits(np.stack(masks))})
        atomic_csv(df, d / f"{stem}.csv")
        print(arm, u.name, round(df.JF.mean(), 4), f"{df.latency_s.median():.3f}s", flush=True)
    heads.close()
    controlled.uninstall(pred)


def run_unit(pred, heads, s, u, arm, n=None, side=None):
    """One object, frames 1..n-1, causal. side overrides the crop rule (tests: full-frame identity box)."""
    stem = u.name.replace("/", "_")
    n, hw = n or len(u.frames), gt(u, 0).shape
    target = 0.5 if arm == "A1z" else vcam.TARGET
    side = side or vcam.crop_side(gt(u, 0), target)
    if arm in ("A2", "A2x", "A2c"):
        g = OUT / "runs/B1" / f"{stem}"
        gm = np.unpackbits(np.load(f"{g}.npz")["B1"])[:(len(u.frames) - 1) * hw[0] * hw[1]].reshape(-1, *hw).astype(bool)
        gs = pd.read_csv(f"{g}.csv").set_index("video_idx").obj_score
    torch.cuda.reset_peak_memory_stats()
    if arm == "B1":
        state, od = m56.start(pred, u, p5=True)
    else:
        center = vcam.center_of(gt(u, 0))
        state, od, frames = start_crop(pred, u, side, center)
    rows, masks, prev = [], [], None
    for t in range(1, n):
        if arm == "A1v" and t > 1 and masks[-1].any():
            b = vcam.bbox(masks[-1])
            side = int(round(min(max(max(b[1] - b[0], b[3] - b[2]) / target, min(hw) / vcam.MAX_ZOOM), min(hw))))
        if arm != "B1":
            frames.box = vcam.place(center, side, hw)
            state["video_height"] = state["video_width"] = side     # output res follows the current crop (A1v)
        m, score, ious, multi, lat = step(pred, state, od, t, heads)
        r = {"set": s, "unit": u.name, "video": u.video, "arm": arm, "video_idx": t, "obj_score": score,
             "iou_pred": float(ious.max()), "iou_cands": json.dumps(np.round(ious, 4).tolist()), "latency_s": lat}
        if arm == "B1":
            out = m
            _, vr = pred._get_orig_video_res_output(state, multi)
            cands = (vr[0] > 0).cpu().numpy()
            r["J_cand_best"] = max(j_measure(c, gt(u, t)) for c in cands)
            r["cand_chosen"] = int(ious.argmax())
        else:
            box = frames.box
            crop_out = vcam.paste(m, box, hw)
            r.update(box=json.dumps(box), side=side, crop_obj_score=score,
                     crop_touch=vcam.touches_border(m, box, hw))
            if arm in ("A1", "A1z", "A1v"):
                out, src = crop_out, "crop"
            else:
                if arm == "A2c":
                    b = vcam.bbox(m)
                    small_box = b is not None and (b[1] - b[0]) * (b[3] - b[2]) / side ** 2 <= 0.1
                    ok = m.any() and small_box and (not gm[t - 1].any() or j_measure(crop_out, gm[t - 1]) > 0.5)
                else:
                    ok = score > 0 and m.any() and not r["crop_touch"]
                out, src = (crop_out, "crop") if ok else (gm[t - 1], "global")
                r["global_obj_score"] = float(gs[t])
            r["source"] = src
            g_ = gt(u, t)
            r["gt_in_crop"] = float(g_[box[0]:box[1], box[2]:box[3]].sum() / g_.sum()) if g_.any() else np.nan
            c = vcam.center_of(out)
            center = c if c is not None else center
        g_ = gt(u, t)
        o = m56.other_gt(u, u.frames[t])
        r.update(m55.jf(out, g_), gt_empty=not g_.any(), gt_area=int(g_.sum()), pred_area=int(out.sum()),
                 other_obj_frac=float((out & o).sum() / out.sum()) if out.any() else 0.0,
                 gt_covered=float((out & g_).sum() / g_.sum()) if g_.any() else np.nan,
                 comp_retention=detail.component_retention(out, g_), thin_recall=detail.thin_recall(out, g_),
                 J_consec_pred=j_measure(out, prev) if prev is not None else np.nan)
        rows.append(r)
        masks.append(out)
        prev = out
    r0 = {"peak_vram_bytes": torch.cuda.max_memory_allocated()}
    pred.reset_state(state)
    return pd.DataFrame(rows).assign(**r0), masks


# ----------------------------------------------------------------------------- Track C diagnostic
def cscale(target=None):
    """RETROSPECTIVE upper bound (GT box prompt, GT-centred crop): does a larger input scale let the frozen
    encoder+decoder express small targets better, and is the 256x256 output a limit? Not deployable."""
    from PIL import Image
    small = _small()
    ipred = m55.build_image_predictor()
    target = float(target or vcam.TARGET)
    rows, f = [], OUT / "reports" / ("cscale_frames.csv" if target == vcam.TARGET else "cscale_frames_z.csv")
    for s, u in units():
        if u.name not in small:
            continue
        side, hw = vcam.crop_side(gt(u, 0), target), gt(u, 0).shape
        ps = paths(u)
        for t in range(0, len(u.frames), 5):
            g = gt(u, t)
            if not g.any():
                continue
            im = np.asarray(Image.open(ps[t]).convert("RGB"))
            with torch.inference_mode():
                ipred.set_image(im)
                r = {"set": s, "unit": u.name, "video": u.video, "video_idx": t, "gt_area": int(g.sum()), "side": side,
                     "J_oracle_full": m55.image_oracle(ipred, g), "J_ceil_full": j_measure(detail.lowres_roundtrip(g), g)}
                if side is not None:
                    y0, y1, x0, x1 = vcam.place(vcam.center_of(g), side, hw)
                    gc = g[y0:y1, x0:x1]
                    r["gt_in_crop"] = float(gc.sum() / g.sum())
                    if gc.any():
                        ipred.set_image(im[y0:y1, x0:x1])
                        r["J_oracle_crop"] = j_measure(vcam.paste(_best_mask(ipred, gc), (y0, y1, x0, x1), hw), g)
                        r["J_ceil_crop"] = j_measure(vcam.paste(detail.lowres_roundtrip(gc), (y0, y1, x0, x1), hw), g)
            rows.append(r)
        print(u.name, flush=True)
    atomic_csv(pd.DataFrame(rows), f)


def _best_mask(ipred, g):
    """image_oracle's argmax mask (same prompts) so the crop result can be scored in full-frame coordinates."""
    ys, xs = np.nonzero(g)
    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
    best, bm = -1, None
    for pts in (None, np.array([m55.inner_point(g)], np.float32)):
        ms, _, _ = ipred.predict(point_coords=pts, point_labels=None if pts is None else np.ones(1), box=box,
                                 multimask_output=True)
        for m in ms:
            j = j_measure(m > 0, g)
            if j > best:
                best, bm = j, m > 0
    return bm


# ----------------------------------------------------------------------------- Track B: re-appearance (CPU, B1 logs)
def reappear_events(vis_area, empty, border, med):
    """GT-only event finder. vis_area[t]: GT area; empty[t]: GT empty; border[t]: GT box within 2 px of the frame
    edge. Returns [(absence_start, reappear_t, type)] for runs of non-visible frames (area < 25% of med) that end
    with a visible frame. type: out_of_frame / occluded (GT empty for the whole run) or partial (never empty)."""
    vis = [a >= 0.25 * med for a in vis_area]
    ev, t, n = [], 1, len(vis)
    while t < n:
        if not vis[t] and vis[t - 1]:
            s = t
            while t < n and not vis[t]:
                t += 1
            if t < n:
                if not any(empty[s:t]):
                    typ = "partial"
                else:
                    typ = "out_of_frame" if border[s - 1] else "occluded"
                ev.append((s, t, typ))
        t += 1
    return ev


ANNOTATION_DEFINED = {"pumavos__pumavos/guitar__obj1": "guitar back side annotated as absent (visual check, M5.6)"}


def breappear():
    F = pd.concat([pd.read_csv(f) for f in sorted((OUT / "runs/B1").glob("*.csv"))], ignore_index=True)
    rows = []
    for (s, name), g in F.groupby(["set", "unit"]):
        g = g.sort_values("video_idx").reset_index(drop=True)
        u = _UNITS()[name]
        g0 = gt(u, 0)
        border = [False] + [_touch(gt(u, t)) for t in g.video_idx]
        area = [int(g0.sum())] + g.gt_area.tolist()
        empty = [False] + g.gt_empty.tolist()
        med = np.median([a for a in area if a > 0])
        J = [np.nan] + g.J.tolist()
        for a0, tr, typ in reappear_events(area, empty, border, med):
            el = range(tr, min(tr + 5, len(area)))
            after = [t for t in range(tr, len(area)) if J[t] >= 0.5]
            row = g.iloc[tr - 1]
            ab = g.iloc[a0 - 1:tr - 1]
            rows.append({"set": s, "unit": name, "video": u.video, "absence_start": a0, "reappear": tr,
                         "absence_len": tr - a0, "type": "annotation_defined" if name in ANNOTATION_DEFINED else typ,
                         "J_first": J[tr], "J_first5": float(np.mean([J[t] for t in el])),
                         "delay": (after[0] - tr) if after else None, "obj_score_first": row.obj_score,
                         "iou_pred_first": row.iou_pred, "pred_gt_area_ratio": row.pred_area / max(row.gt_area, 1),
                         "other_obj_frac_first": row.other_obj_frac, "J_cand_best_first": row.J_cand_best,
                         "absence_obj_score_mean": float(ab.obj_score.mean()),
                         "absence_pred_present_frac": float((ab.pred_area > 0).mean()),
                         "absence_frames_gt_empty": int(ab.gt_empty.sum())})
        print(name, flush=True)
    atomic_csv(pd.DataFrame(rows), OUT / "reports" / "reappear_events.csv")


def _touch(g, px=2):
    b = vcam.bbox(g)
    H, W = g.shape
    return b is not None and (b[0] <= px or b[2] <= px or b[1] >= H - px or b[3] >= W - px)


_U = {}


def _UNITS():
    if not _U:
        _U.update({u.name: u for _, u in units()})
    return _U


# ----------------------------------------------------------------------------- analysis
def _runs(arm):
    fs = sorted((OUT / "runs" / arm).glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def _boot(df, col):
    from rfeas.risk import cluster_bootstrap
    return cluster_bootstrap(df, "video", lambda d: float(d[col].mean()), n=2000)


def compare(arm, B):
    """Per-unit arm vs B1 over every unit of B (units the arm skipped are identical to B1 by construction)."""
    X = _runs(arm)
    small = {r["unit"]: r["small"] for r in json.loads((OUT / "preregistration.json").read_text())["rows"]}
    ub = B.groupby(["set", "unit", "video"]).agg(JF_B1=("JF", "mean"), J_B1=("J", "mean"), F_B1=("F", "mean")).reset_index()
    ux = X.groupby("unit").agg(JF_X=("JF", "mean"), J_X=("J", "mean"), F_X=("F", "mean"), lat_X=("latency_s", "median"),
                               escape=("gt_in_crop", lambda v: float((v < 0.5).mean())),
                               crop_frac=("source", lambda v: float((v == "crop").mean()))).reset_index()
    W = ub.merge(ux, on="unit", how="left")
    W["applied"] = W.JF_X.notna()
    for c in ("JF", "J", "F"):
        W[f"{c}_X"] = W[f"{c}_X"].fillna(W[f"{c}_B1"])
    W["dJF"], W["dJ"], W["dF"] = W.JF_X - W.JF_B1, W.J_X - W.J_B1, W.F_X - W.F_B1
    W["small"] = W.unit.map(small)
    W["arm"] = arm
    return W, X


def gate(W):
    G = W[W.set.isin(GATE_SETS)]
    sm, ns = G[G.small], G[~G.small]
    r = {"n_small": int(len(sm)), "n_small_applied": int(sm.applied.sum()), "small_dJF": float(sm.dJF.mean()),
         "small_dJF_CI_video": _boot(sm, "dJF"), "small_frac_regress_0.01": float((sm.dJF < -0.01).mean()),
         "small_up_0.01": int((sm.dJF > 0.01).sum()), "small_down_0.01": int((sm.dJF < -0.01).sum()),
         "n_nonsmall": int(len(ns)), "n_nonsmall_applied": int(ns.applied.sum()), "nonsmall_dJF": float(ns.dJF.mean())}
    r["pass"] = {"small_dJF>=0.03": r["small_dJF"] >= 0.03, "regress_frac<0.2": r["small_frac_regress_0.01"] < 0.2,
                 "nonsmall_dJF>=-0.005": r["nonsmall_dJF"] >= -0.005}
    r["PASS"] = all(r["pass"].values())
    return r


def analyze():
    rd = OUT / "reports"
    rep = {}
    B = _runs("B1")
    # 0. reproducibility of B1 against M5.6 (PUMaVOS) / M5.5 (DAVIS, SegTrack 'simple' == P5)
    old = pd.concat([pd.read_csv(f) for f in sorted((m56.OUT / "runs").glob("*.csv"))])
    old = pd.concat([old[old.arm == "B1_P5"]] + [pd.read_csv(f).query("arm == 'simple'") for d in ("val", "segtrack")
                                                 for f in sorted(Path(f"outputs/milestone55/proto/{d}").glob("*.csv"))])
    mm = B.merge(old[["unit", "video_idx", "J"]], on=["unit", "video_idx"], suffixes=("", "_old"))
    rep["B1_reproduces_prior_runs"] = {"frames_compared": int(len(mm)), "max_abs_dJ": float((mm.J - mm.J_old).abs().max()),
                                       "frames_differ_1e-6": int(((mm.J - mm.J_old).abs() > 1e-6).sum())}
    rep["B1"] = B.groupby("set").agg(units=("unit", "nunique"), frames=("J", "size"), JF=("JF", "mean"),
                                     lat_median=("latency_s", "median"), vram_GB=("peak_vram_bytes", lambda v: v.max() / 1e9)).reset_index().to_dict("records")
    # 1. Track A
    Ep = []
    for arm in ("A1", "A2", "A1z", "A1v", "A2x", "A2c"):
        if _runs(arm) is None:
            continue
        W, X = compare(arm, B)
        atomic_csv(W, rd / f"trackA_{arm}_per_unit.csv")
        rep[f"gate_{arm}"] = gate(W)
        rep[f"{arm}_by_set"] = {k: {"units": int(len(g)), "applied": int(g.applied.sum()), "dJF": float(g.dJF.mean()),
                                    "dJF_CI": _boot(g, "dJF"), "dJF_applied": float(g[g.applied].dJF.mean()) if g.applied.any() else 0.0}
                                for k, g in W.groupby("set")}
        rep[f"{arm}_applied_units"] = W[W.applied][["set", "unit", "small", "JF_B1", "JF_X", "dJF", "dJ", "dF", "escape",
                                                     "crop_frac", "lat_X"]].round(4).to_dict("records")
        rep[f"{arm}_latency_median_s"] = float(X.latency_s.median())
        rep[f"{arm}_peak_vram_GB"] = float(X.peak_vram_bytes.max() / 1e9)
        for a, F in (("B1", B[B.unit.isin(X.unit.unique())]), (arm, X)):
            e = m56.episode_table(F.assign(arm=a))
            e["set"] = e.unit.map(dict(zip(F.unit, F.set)))
            Ep.append(e)
        if arm == "A1":
            rep["A2_trigger"] = {"small_up_0.03": int(((W.dJF >= 0.03) & W.small).sum()),
                                 "small_down_0.03": int(((W.dJF <= -0.03) & W.small).sum())}
            rep["A2_trigger"]["fire"] = rep["A2_trigger"]["small_up_0.03"] >= 3 or rep["A2_trigger"]["small_down_0.03"] >= 3
    if Ep:
        Ep = pd.concat(Ep, ignore_index=True)
        atomic_csv(Ep, rd / "trackA_episodes.csv")
        rep["trackA_episodes_on_applied_units"] = Ep.groupby("arm").agg(episodes=("unit", "size"), failed_frames=("duration", "sum"),
                                                                          unrecovered=("recovered", lambda r: int((~r).sum()))).reset_index().to_dict("records")
    # 2. Track B
    from rfeas.risk import auroc
    ev = rd / "reappear_events.csv"
    if ev.exists():
        E = pd.read_csv(ev)
        E["fail_first"] = E.J_first < 0.5
        E["cand_opportunity"] = E.fail_first & (E.J_cand_best_first >= 0.5)
        rep["reappear_by_type"] = E.groupby("type").agg(events=("unit", "size"), units=("unit", "nunique"), videos=("video", "nunique"),
                                                        J_first=("J_first", "mean"), J_first5=("J_first5", "mean"),
                                                        fail_first=("fail_first", "sum"), delay_median=("delay", "median"),
                                                        cand_opportunity=("cand_opportunity", "sum"),
                                                        absence_pred_present=("absence_pred_present_frac", "mean")).reset_index().to_dict("records")
        nonann = E[(E.type != "annotation_defined") & E.fail_first]
        trig = nonann[nonann.cand_opportunity]
        rep["B3_trigger"] = {"failed_reappearances_nonannotation": int(len(nonann)), "videos": int(nonann.video.nunique()),
                             "with_online_candidate_opportunity": int(len(trig)), "opportunity_videos": int(trig.video.nunique()),
                             "fire": bool(len(trig) >= 4 and trig.video.nunique() >= 3)}
        rep["reappear_failed_events"] = nonann.round(3).to_dict("records")
    Bv = B[~B.unit.isin(ANNOTATION_DEFINED)]
    rep["presence_detection"] = {k: {"frames": int(len(g)), "gt_empty_frames": int(g.gt_empty.sum()),
                                     "auroc_objscore_vs_gt_present": auroc((~g.gt_empty).astype(int).to_numpy(), g.obj_score.to_numpy())
                                     if g.gt_empty.any() else None,
                                     "false_presence_on_empty": float((g[g.gt_empty].pred_area > 0).mean()) if g.gt_empty.any() else None,
                                     "missed_presence_on_visible": float((g[~g.gt_empty].pred_area == 0).mean())}
                                 for k, g in Bv.groupby("set")}
    vis = Bv[~Bv.gt_empty]
    rep["iou_pred_vs_J_spearman"] = float(vis[["iou_pred", "J"]].corr("spearman").iloc[0, 1])
    rep["identity_frames_other_obj_ge_0.3"] = Bv.groupby("set").other_obj_frac.apply(lambda v: int((v >= 0.3).sum())).to_dict()
    # 3. Track C
    cs = rd / "cscale_frames.csv"
    Cc = None
    if cs.exists():
        C = pd.read_csv(cs)
        Cc = C.dropna(subset=["J_oracle_crop"])
        U = Cc.groupby(["unit", "video"]).agg(frames=("J_oracle_full", "size"), full=("J_oracle_full", "mean"), crop=("J_oracle_crop", "mean"),
                                              ceil_full=("J_ceil_full", "mean"), ceil_crop=("J_ceil_crop", "mean")).reset_index()
        U["gain"] = U.crop - U.full
        atomic_csv(U, rd / "trackC_per_unit.csv")
        rep["trackC"] = {"units_with_crop": int(len(U)), "frames": int(len(Cc)), "oracle_full": float(U.full.mean()), "oracle_crop": float(U.crop.mean()),
                         "gain_mean": float(U.gain.mean()), "gain_CI_video": _boot(U, "gain"), "frac_units_gain_pos": float((U.gain > 0).mean()),
                         "ceil256_full_lt_0.8_frame_frac": float((C.J_ceil_full < 0.8).mean()),
                         "ceil256_crop_lt_0.8_frame_frac": float((Cc.J_ceil_crop < 0.8).mean()),
                         "all_small_units": int(C.unit.nunique()), "all_frames": int(len(C)),
                         "oracle_full_all": float(C.J_oracle_full.mean())}
        rep["trackC"]["input_scale_fixable"] = rep["trackC"]["gain_mean"] >= 0.05 and rep["trackC"]["frac_units_gain_pos"] >= 0.6
        rep["trackC"]["decoder_output_bottleneck"] = rep["trackC"]["ceil256_full_lt_0.8_frame_frac"] >= 0.25
        rep["trackC_per_unit"] = U.round(3).to_dict("records")
    # 4. round 2 mechanism rules
    if _runs("A1") is not None and _runs("A1v") is not None:
        a, v = compare("A1", B)[0], compare("A1v", B)[0]
        d = a[["set", "unit", "video", "small", "JF_X"]].merge(v[["unit", "JF_X"]], on="unit", suffixes=("_A1", "_A1v"))
        d = d[d.set.isin(GATE_SETS) & d.small]
        d["diff"] = d.JF_X_A1 - d.JF_X_A1v
        ci = _boot(d, "diff")
        rep["H3_fixed_minus_variable"] = {"small_units": int(len(d)), "mean": float(d["diff"].mean()), "CI_video": ci,
                                          "supported": bool(d["diff"].mean() >= 0.01 and ci[0] > 0),
                                          "refuted": bool(d["diff"].mean() <= 0)}
    z = rd / "cscale_frames_z.csv"
    if z.exists() and cs.exists() and _runs("A1z") is not None:
        Z = pd.read_csv(z).dropna(subset=["J_oracle_crop"])
        m = Cc.merge(Z[["unit", "video_idx", "J_oracle_crop", "J_ceil_crop"]], on=["unit", "video_idx"], suffixes=("", "_z"))
        gz = rep["gate_A1z"]["small_dJF"]
        rep["dose"] = {"frames_both": int(len(m)), "oracle_full": float(m.J_oracle_full.mean()), "oracle_A1scale": float(m.J_oracle_crop.mean()),
                       "oracle_A1zscale": float(m.J_oracle_crop_z.mean()), "video_small_dJF_A1": rep["gate_A1"]["small_dJF"],
                       "video_small_dJF_A1z": gz}
        rep["dose"]["supported"] = bool(rep["dose"]["oracle_A1zscale"] > rep["dose"]["oracle_A1scale"] and gz > rep["gate_A1"]["small_dJF"])
    atomic_json(rep, rd / "m6_report.json")
    print(json.dumps(rep, indent=1, default=float)[:20000])


# ----------------------------------------------------------------------------- round 3: self-assessed scale selection
POOL, K = ("B1", "A1", "A1z"), 5


def select_scale(iou: dict, k: int = K) -> str:
    """iou: arm -> per-frame iou_pred array (frames 1..). Highest mean over the first k frames; ties -> first in POOL."""
    best, arm = -np.inf, POOL[0]
    for a in POOL:
        if a in iou:
            v = float(np.mean(iou[a][:k]))
            if v > best + 1e-12:
                best, arm = v, a
    return arm


def s1():
    B = _runs("B1")
    arms = {a: _runs(a) for a in POOL}
    small = {r["unit"]: r["small"] for r in json.loads((OUT / "preregistration.json").read_text())["rows"]}
    rows = []
    for (s, u, v), b in B.groupby(["set", "unit", "video"]):
        per = {"B1": b.sort_values("video_idx")}
        for a in POOL[1:]:
            x = arms[a][arms[a].unit == u]
            if len(x):
                per[a] = x.sort_values("video_idx")
        pick = select_scale({a: d.iou_pred.to_numpy() for a, d in per.items()})
        jf = np.concatenate([per["B1"].JF.to_numpy()[:K], per[pick].JF.to_numpy()[K:]])
        rows.append({"set": s, "unit": u, "video": v, "small": small[u], "pick": pick, "JF_B1": per["B1"].JF.mean(),
                     "JF_S1": jf.mean(), **{f"JF_{a}": d.JF.mean() for a, d in per.items()},
                     **{f"iouK_{a}": d.iou_pred.to_numpy()[:K].mean() for a, d in per.items()}})
    W = pd.DataFrame(rows)
    W["dJF"] = W.JF_S1 - W.JF_B1
    W["applied"] = W.pick != "B1"
    W["JF_oracle"] = W[[c for c in W if c.startswith("JF_") and c not in ("JF_S1",)]].max(axis=1)
    W["dJF_oracle"] = W.JF_oracle - W.JF_B1
    atomic_csv(W, OUT / "reports" / "S1_per_unit.csv")
    rep = {"gate_S1": gate(W), "picks": W.pick.value_counts().to_dict(),
           "by_set": {k: {"dJF": float(g.dJF.mean()), "CI": _boot(g, "dJF"), "picks": g.pick.value_counts().to_dict()} for k, g in W.groupby("set")},
           "oracle_retro": {k: {"small": float(g[g.small].dJF_oracle.mean()) if g.small.any() else None,
                                "all": float(g.dJF_oracle.mean())} for k, g in W.groupby("set")},
           "changed_units": W[W.applied].round(4).to_dict("records")}
    atomic_json(rep, OUT / "reports" / "S1_report.json")
    print(json.dumps(rep, indent=1, default=float))


# ----------------------------------------------------------------------------- round 4: Track C3 head pilot
class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        c = torch.nn.Conv2d
        self.net = torch.nn.Sequential(c(101, 64, 3, padding=1), torch.nn.ReLU(), c(64, 64, 3, padding=1), torch.nn.ReLU(),
                                       c(64, 64, 3, padding=2, dilation=2), torch.nn.ReLU(), c(64, 1, 3, padding=1))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, logit, s0, s1, rgb):
        up = torch.nn.functional.interpolate(s1, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        return logit + self.net(torch.cat([logit / 10, torch.sigmoid(logit), s0, up, rgb], 1))


def _frozen(ipred, im, gts):
    """set_image + GT-box prediction per object. Returns (s0, s1, rgb256, [logit 1x1x256x256])."""
    ipred.set_image(im)
    s0, s1 = ipred._features["high_res_feats"]
    rgb = torch.nn.functional.interpolate(torch.from_numpy(im).permute(2, 0, 1)[None].float().cuda() / 255, size=(256, 256),
                                          mode="bilinear", align_corners=False)
    logits = []
    for g in gts:
        b = vcam.bbox(g)
        _, _, lr = ipred.predict(box=np.array([b[2], b[0], b[3], b[1]], np.float32), multimask_output=False, return_logits=True)
        logits.append(torch.from_numpy(lr).cuda()[None])
    return s0, s1, rgb, logits


def _to_frame(ipred, logit, hw):
    return (ipred._transforms.postprocess_masks(logit, hw)[0, 0] > 0).cpu().numpy()


def head_pilot():
    from PIL import Image
    from rfeas.units import davis_units, davis_gt
    torch.manual_seed(0)
    ipred = m55.build_image_predictor()
    head = Head().cuda()
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    tr = {}
    for u in davis_units("train")[0]:
        tr.setdefault(u.video, []).append(u)
    items = [(us, t) for us in tr.values() for t in range(0, len(us[0].frames), 2)]
    rng = np.random.default_rng(0)
    t_start, log = time.time(), []
    for ep in range(2):
        for i in rng.permutation(len(items)):
            if time.time() - t_start > 25 * 60:
                break
            us, t = items[i]
            gts = [davis_gt(u, u.frames[t]) for u in us]
            gts = [g for g in gts if g.any()]
            if not gts:
                continue
            im = np.asarray(Image.open(m55.frame_dir(us[0]) / f"{us[0].frames[t]}.jpg").convert("RGB"))
            with torch.no_grad():
                s0, s1, rgb, logits = _frozen(ipred, im, gts)
            loss = 0
            for g, lg in zip(gts, logits):
                y = torch.nn.functional.interpolate(torch.from_numpy(g)[None, None].float().cuda(), size=(256, 256), mode="area")
                y = (y > 0.5).float()
                out = head(lg, s0, s1, rgb)
                p = torch.sigmoid(out)
                l = torch.nn.functional.binary_cross_entropy_with_logits(out, y) + 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
                loss = loss + l * (4.0 if g.mean() < 0.01 else 1.0)
            opt.zero_grad()
            (loss / len(gts)).backward()
            opt.step()
            log.append({"epoch": ep, "loss": float(loss) / len(gts), "t": time.time() - t_start})
            if len(log) % 200 == 0:
                print(ep, len(log), round(np.mean([x["loss"] for x in log[-200:]]), 4), f"{log[-1]['t']:.0f}s", flush=True)
    torch.save(head.state_dict(), OUT / "head_pilot.pt")
    atomic_csv(pd.DataFrame(log), OUT / "reports" / "head_pilot_trainlog.csv")
    head.eval()
    small = _small()
    rows = []
    for s, u in units():
        sm = u.name in small
        if s == "SegTrack" and not sm:
            continue
        ps = paths(u)
        for t in range(0, len(u.frames), 5 if sm else 10):
            g = gt(u, t)
            if not g.any():
                continue
            im = np.asarray(Image.open(ps[t]).convert("RGB"))
            with torch.no_grad():
                s0, s1, rgb, (lg,) = _frozen(ipred, im, [g])
                a, b = _to_frame(ipred, lg, g.shape), _to_frame(ipred, head(lg, s0, s1, rgb), g.shape)
            rows.append({"set": s, "unit": u.name, "video": u.video, "small": sm, "video_idx": t, "gt_area": int(g.sum()),
                         **{f"{k}_frozen": v for k, v in m55.jf(a, g).items()}, **{f"{k}_head": v for k, v in m55.jf(b, g).items()}})
    E = pd.DataFrame(rows)
    atomic_csv(E, OUT / "reports" / "head_pilot_frames.csv")
    E["dJ"] = E.J_head - E.J_frozen
    U = E.groupby(["set", "unit", "video", "small"]).agg(J_frozen=("J_frozen", "mean"), J_head=("J_head", "mean"), dJ=("dJ", "mean"),
                                                        dF=("F_head", "mean")).reset_index()
    U["dF"] = U.dF - E.groupby("unit").F_frozen.mean().reindex(U.unit).to_numpy()
    sm, od = U[U.small], U[~U.small]
    rep = {"train_steps": len(log), "train_minutes": log[-1]["t"] / 60 if log else 0,
           "small": {"units": int(len(sm)), "dJ": float(sm.dJ.mean()), "CI": _boot(sm, "dJ"), "frac_pos": float((sm.dJ > 0).mean()),
                     "dF": float(sm.dF.mean()), "J_frozen": float(sm.J_frozen.mean()), "J_head": float(sm.J_head.mean())},
           "ordinary": {"units": int(len(od)), "dJ": float(od.dJ.mean()), "CI": _boot(od, "dJ"), "dF": float(od.dF.mean())},
           "per_unit_small": sm.round(4).to_dict("records")}
    rep["supported"] = bool(rep["small"]["dJ"] >= 0.05 and rep["small"]["frac_pos"] >= 0.6 and rep["ordinary"]["dJ"] >= -0.01)
    atomic_json(rep, OUT / "reports" / "head_pilot_report.json")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_unit_small"}, indent=1, default=float))


def _overlay(img, g, pr):
    from spatial.detail import dilate
    a = np.asarray(img.convert("RGB")).copy()
    for m, col in ((g, (0, 255, 0)), (pr, (255, 0, 0))):
        e = dilate(m, 1) & ~m if m.any() else m
        a[e] = col
    return a


def montage(spec=None):
    """Visual checks: `unit:t1,t2,..;unit:...` -> figs/<n>.png (green GT edge, red arm output edge). B1 by default."""
    from PIL import Image
    arm = "B1"
    if spec and spec.startswith("@"):
        arm, spec = spec[1:].split("|", 1)
    tiles = []
    for part in spec.split(";"):
        name, ts = part.split(":")
        u = _UNITS()[name]
        hw = gt(u, 0).shape
        mk = np.unpackbits(np.load(OUT / "runs" / arm / (name.replace("/", "_") + ".npz"))[arm])
        mk = mk[:(len(u.frames) - 1) * hw[0] * hw[1]].reshape(-1, *hw).astype(bool)
        ps = paths(u)
        for t in map(int, ts.split(",")):
            a = _overlay(Image.open(ps[t]), gt(u, t), mk[t - 1] if t > 0 else gt(u, 0))
            tiles.append(np.asarray(Image.fromarray(a).resize((427, 240))))
    rows = [np.concatenate(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4])), 1) for i in range(0, len(tiles), 4)]
    (OUT / "figs").mkdir(exist_ok=True)
    f = OUT / "figs" / f"{arm}_{abs(hash(spec)) % 10**6}.png"
    Image.fromarray(np.concatenate(rows, 0)).save(f)
    print(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("arg", nargs="?")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    fn = globals()[a.what]
    fn(a.arg) if a.arg else fn()


if __name__ == "__main__":
    main()
