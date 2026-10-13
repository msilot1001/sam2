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
        if arm != "B1" and vcam.crop_side(gt(u, 0)) is None:
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
    side = side or vcam.crop_side(gt(u, 0))
    if arm == "A2":
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
        if arm != "B1":
            frames.box = vcam.place(center, side, hw)
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
            if arm == "A1":
                out, src = crop_out, "crop"
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
def cscale():
    """RETROSPECTIVE upper bound (GT box prompt, GT-centred crop): does a larger input scale let the frozen
    encoder+decoder express small targets better, and is the 256x256 output a limit? Not deployable."""
    from PIL import Image
    small = _small()
    ipred = m55.build_image_predictor()
    rows, f = [], OUT / "reports" / "cscale_frames.csv"
    for s, u in units():
        if u.name not in small:
            continue
        side, hw = vcam.crop_side(gt(u, 0)), gt(u, 0).shape
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
