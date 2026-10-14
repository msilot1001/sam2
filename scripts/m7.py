"""M7: SGGL algorithm research tournament (CUDA FP32, frozen SAM 2.1 Hiera-T).

    uv run python scripts/m7.py prereg            # freeze stage-0/1 protocol (before any M7 output)
    uv run python scripts/m7.py rec DEV,REP       # stage-0 recording: live SGGL + all views/probes per frame
    uv run python scripts/m7.py run <arm> <sets>  # causal rollout of a policy arm (see policies in spatial.m7arms)

Sets (crop units only; uncropped units are identical to P5 for every arm):
  DEV  M6 dev crop units (DAVIS val, PUMaVOS A/B, SegTrack; 21 units)       -- repeatedly inspected, development
  REP  DAVIS 2017 train crop units (43 units / 23 videos)                     -- M6.1 replication; discovery of miami-surf
  VK   VKITTI2 Scene01+06 clone crop units (28 units, 2 scenes, synthetic)    -- fresh for SGGL; finalists only
PUMaVOS C is never touched here.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

from baseline.dropout_runner import atomic_csv, atomic_json
from baseline.metrics import j_measure
from memint import controlled
from spatial import sggl, vcam

_s = importlib.util.spec_from_file_location("m6", Path(__file__).with_name("m6.py"))
m6 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m6)
m55, m56 = m6.m55, m6.m56
OUT = Path("outputs/milestone7")
REC = OUT / "rec"

PREREG = {
    "created_before_any_m7_output": True,
    "base": "SGGL = M6 A2x (self gate, centre follows output); SGGL-V = A2x + global score > 0 (M6.1 A2v, post-hoc)",
    "sets": {"DEV": "21 M6 dev crop units (gate set DAVIS_val+PUMaVOS_B = 13, aux PUMaVOS_A + SegTrack = 8)",
             "REP": "43 DAVIS-2017-train crop units / 23 videos (M6.1 replication; miami-surf = discovery case)",
             "VK": "28 VKITTI2 Scene01/06 clone crop units (2 scenes, synthetic, many same-class distractors); "
                   "NOT used during screening/revision; run once per frozen finalist + SGGL + SGGL-V",
             "C": "PUMaVOS C untouched; at most one run for the single final frozen method meeting the final rule"},
    "development_data": "DEV + REP (both previously inspected => every M7 screening result is exploratory)",
    "training_data": "learned modules (T family, any fitted gate) fit on REP videos only (DAVIS train); video-disjoint "
                     "inner split for validation curves; evaluated on DEV (+ VK for finalists). REP is then not an "
                     "evaluation set for that module (reported).",
    "metric": "per-unit mean J&F over frames 1..n-1 (prompt frame excluded, M5.5 J/F); delta vs SGGL and vs P5 (= the "
              "live G state, identical to B1); CIs: 2000x bootstrap clustered by video; small = prompt area < 1%, "
              "tiny = < 0.1%",
    "stage0_recording": "live SGGL rollout on DEV+REP storing per frame: low-res logits of G, L, L2 (2x crop side, same "
                        "centre), L protected-memory probe (prompt memory only), L trusted-memory probe (non-cond memory "
                        "restricted to frames where L and G agreed: both scores > 0 and IoU > 0.5, backfilled), G "
                        "protected probe (scalars), L resolution probe (crop image down/up-sampled to G's effective "
                        "resolution of that region; same memory) on event frames (presence disagreement or IoU<0.5) "
                        "and every 10th frame; frozen frame-0 identity references (mask-pooled stride-16 backbone "
                        "features) and per-frame cosine maps in both views. Production outputs must equal SGGL exactly.",
    "replay_screening": "output-only candidates are first screened by replay on the recorded SGGL trajectory (label: "
                        "REPLAY). Any candidate that changes the output must be confirmed by a causal rollout (centre "
                        "follows its own output) before it can ADVANCE.",
    "pilot_rule": {"ADVANCE": "pooled small crop units of DEV+REP: dJF vs SGGL >= +0.005 AND >= 3 videos improved by > 0.01 "
                              "AND leave-one-video-out min > 0 AND still > 0 without miami-surf AND no unit < -0.05 vs SGGL",
                   "REJECT": "pooled dJF vs SGGL <= 0, or all gain from <= 2 videos with LOVO min <= 0",
                   "INCONCLUSIVE": "otherwise, or a learned module failing its positive control"},
    "learned_positive_control": "overfit 64 fixed REP frames: train-set J gain >= +0.05 over the frozen input mask within "
                                "1500 steps, loss decreasing; else the T result is INCONCLUSIVE (not a negative result)",
    "final_rule": "the final method must beat SGGL on pooled small DEV+REP crop units with video-CI lower bound > 0 in a "
                  "causal rollout, not regress on VK (dJF vs SGGL >= -0.005, 2 scenes => descriptive), and have no "
                  "DEV/REP unit < -0.05 vs SGGL; otherwise KEEP_SGGL or INSUFFICIENT_EVIDENCE",
    "multiplicity": "about 15-25 candidate comparisons on the same development videos; a screening winner is exploratory "
                    "until VK/C or new data; reported as such",
}


def prereg():
    m55._freeze(OUT / "preregistration.json", {**PREREG, "rows": []})


def status(**kw):
    p = OUT / "EXECUTION_STATUS.json"
    s = json.loads(p.read_text()) if p.exists() else {"experiments": [], "known_bugs": [], "completed": []}
    for k, v in kw.items():
        if isinstance(s.get(k), list) and not isinstance(v, list):
            s[k].append(v)
        else:
            s[k] = v
    s["updated_kst"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600))
    atomic_json(s, p)


# ----------------------------------------------------------------------------- data
_VK = {}


def _vk(u):
    from rfeas.units import vkitti_units
    if not _VK:
        _VK.update(vkitti_units()[2])
    return _VK[u.video.split("/", 1)[1]]


def gt(u, t):
    if u.dataset == "vkitti":
        from vkitti.dataset import track_mask
        return track_mask(_vk(u).load_instance(u.frames[t]), u.obj_id)
    return m6.gt(u, t)


def other(u, t):
    """Union of every OTHER annotated object at frame t (offline attribution only)."""
    if u.dataset == "vkitti":
        inst = _vk(u).load_instance(u.frames[t])
        return (inst != 0) & (inst != u.obj_id + 1)
    return m56.other_gt(u, u.frames[t])


def paths(u):
    if u.dataset == "vkitti":
        return [_vk(u).rgb_path(f) for f in u.frames]
    return m6.paths(u)


def units(sets):
    from rfeas.units import davis_units, vkitti_units
    out = []
    for s in sets.split(","):
        if s == "DEV":
            us = [(x, u) for x, u in m6.units(("DAVIS_val", "PUMaVOS_B", "PUMaVOS_A", "SegTrack"))]
        elif s == "REP":
            us = [("DAVIS_train", u) for u in davis_units("train")[0]]
        elif s == "VK":
            us = [("VK", u) for u in vkitti_units()[0]]
        else:
            raise ValueError(s)
        out += [(s, x, u) for x, u in us if vcam.crop_side(gt(u, 0)) is not None]
    return out


def stem(u):
    return u.name.replace("/", "_")


# ----------------------------------------------------------------------------- identity references
def _pool(feat, m, hw_view):
    """Mask-pooled, L2-normalised feature vector (feat 1xCx64x64 GPU, m view-res bool). Soft area weights."""
    w = Fn.interpolate(torch.as_tensor(m, dtype=torch.float32)[None, None], size=feat.shape[-2:], mode="area").to(feat.device)
    if w.sum() <= 0:
        return None
    v = (feat * w).sum((-1, -2)) / w.sum()
    return Fn.normalize(v, dim=1)[0]


def _cos_map(feat, ref):
    return (Fn.normalize(feat, dim=1)[0] * ref[:, None, None]).sum(0)


def _down_up(img, fy, fx):
    """1x3x1024x1024 normalised crop image -> same size after area down-sampling by (fy, fx) + bilinear up.
    With fy = H/side, fx = W/side the crop keeps exactly the pixel density G's 1024x1024 input has on that region."""
    lo = Fn.interpolate(img, size=(max(1, int(round(1024 / max(fy, 1.0)))), max(1, int(round(1024 / max(fx, 1.0))))), mode="area")
    return Fn.interpolate(lo, size=(1024, 1024), mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------- stage 0 recording
def _pasted(v, hw):
    return vcam.paste(v.mask, v.box, hw)


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def record_unit(pred, s, u, n=None, probes=True, official=False):
    """Live SGGL with every view/probe recorded. Returns (DataFrame, arrays)."""
    n = n or len(u.frames)
    g0 = gt(u, 0)
    hw = g0.shape
    H, W = hw
    d = sggl.Dual(pred, paths(u), g0, scale2=2.0, official=official)
    side = d.side
    y0, y1, x0, x1 = d.S["L"].frames.box
    refL = _pool(d.features("L", 0), g0[y0:y1, x0:x1], (side, side))
    refG = _pool(d.features("G", 0), g0, hw)
    trusted, rows = [], []
    A = {k: [] for k in ("G", "L", "L2", "Lp", "Lt", "simL", "simG", "vL", "vG", "vLinG", "ptrL", "ptrG", "box", "box2")}
    # effective G resolution of the crop region: G maps W px -> 1024 (x), H px -> 1024 (y); L maps side px -> 1024
    res_factor = (H / side, W / side)   # G's pixel density on the crop region is 1/res_factor of L's (y, x)
    for t in range(1, n):
        t0 = time.perf_counter()
        G = d.view("G", t)
        L = d.view("L", t)
        lat_prod = time.perf_counter() - t0
        L2 = d.view("L2", t)
        B0 = d.view("B0", t) if official else None
        out, ok = sggl.a2x(G, L, hw)
        r = {"set": s, "unit": u.name, "video": u.video, "video_idx": t, "side": side, "side2": d.side2,
             "y0": L.box[0], "x0": L.box[2], "y0_2": L2.box[0], "x0_2": L2.box[2], "res_fy": res_factor[0], "res_fx": res_factor[1],
             "a2x_crop": ok, "a2v_crop": ok and G.score > 0, "lat_prod_s": lat_prod,
             **{f"score_{v.name}": v.score for v in (G, L, L2)}, **{f"iou_{v.name}": v.iou for v in (G, L, L2)}}
        pL = _pasted(L, hw)
        r["touch_L"] = vcam.touches_border(L.mask, L.box, hw)
        r["touch_L2"] = vcam.touches_border(L2.mask, L2.box, hw)
        r["IoU_LG"] = _iou(pL, G.mask)
        if probes:
            Lp = d.probe("L", t, "protected")
            Gp = d.probe("G", t, "protected")
            Lt = d.probe("L", t, "trusted", trusted=trusted)
            r.update(score_Lp=Lp.score, score_Gp=Gp.score, score_Lt=Lt.score, iou_Lp=Lp.iou, iou_Lt=Lt.iou,
                     IoU_LLp=_iou(L.mask, Lp.mask), IoU_LLt=_iou(L.mask, Lt.mask), IoU_GGp=_iou(G.mask, Gp.mask))
            event = (L.score > 0) != (G.score > 0) or (L.score > 0 and G.score > 0 and r["IoU_LG"] < 0.5)
            r["event"] = bool(event)
            if event or t % 10 == 0:
                img = d.S["L"].frames[t][None]
                Lb = d.probe("L", t, "image", image=_down_up(img, *res_factor))
                r.update(score_Lb=Lb.score, IoU_LLb=_iou(L.mask, Lb.mask), area_Lb=int(Lb.mask.sum()))
            A["Lp"].append(Lp.logit[0, 0].half().numpy())
            A["Lt"].append(Lt.logit[0, 0].half().numpy())
        # online trusted-frame set for the trusted-memory probe (L and G agree on presence and shape)
        if L.score > 0 and G.score > 0 and r["IoU_LG"] > 0.5:
            trusted.append(t)
        fL, fG = d.features("L", t), d.features("G", t)
        A["simL"].append(_cos_map(fL, refL).half().cpu().numpy())
        A["simG"].append(_cos_map(fG, refG).half().cpu().numpy())
        for k, f, m, hwv in (("vL", fL, L.mask, (side, side)), ("vG", fG, G.mask, hw), ("vLinG", fG, pL, hw)):
            v = _pool(f, m, hwv)
            A[k].append(np.zeros(256, np.float16) if v is None else v.half().cpu().numpy())
        A["ptrL"].append(L.out["obj_ptr"][0].half().cpu().numpy())
        A["ptrG"].append(G.out["obj_ptr"][0].half().cpu().numpy())
        for k, v in (("G", G), ("L", L), ("L2", L2)):
            A[k].append(v.logit[0, 0].half().numpy())
        A["box"].append(L.box)
        A["box2"].append(L2.box)
        # ---- offline metrics (GT used only after the decision)
        g, o = gt(u, t), other(u, t)
        r.update(gt_empty=not g.any(), gt_area=int(g.sum()), **m55.jf(out, g),
                 gt_in_crop=float(g[L.box[0]:L.box[1], L.box[2]:L.box[3]].sum() / g.sum()) if g.any() else np.nan)
        cands = {"G": G.mask, "L": pL, "L2": _pasted(L2, hw)}
        if probes:
            cands.update(Lp=_pasted(Lp, hw), Lt=_pasted(Lt, hw), Gp=Gp.mask)
        if official:
            cands["B0"] = B0.mask
        for k, m in cands.items():
            r[f"J_{k}"] = j_measure(m, g)
            r[f"area_{k}"] = int(m.sum())
            r[f"oth_{k}"] = float((m & o).sum() / m.sum()) if m.any() else np.nan
        for k in ("G", "L"):
            r[f"JF_{k}"] = m55.jf(cands[k], g)["JF"]
        rows.append(r)
        c = vcam.center_of(out)
        d.set_center(c)
    peak = torch.cuda.max_memory_allocated()
    d.close()
    arr = {k: np.stack(v) for k, v in A.items() if v}
    arr.update(refL=refL.half().cpu().numpy(), refG=refG.half().cpu().numpy())
    return pd.DataFrame(rows).assign(peak_vram_bytes=peak), arr


def rec(sets="DEV,REP"):
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    REC.mkdir(parents=True, exist_ok=True)
    for s, x, u in units(sets):
        f = REC / f"{stem(u)}.csv"
        if f.exists():
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        df, arr = record_unit(pred, s, u, official=(s == "VK"))
        np.savez_compressed(REC / f"{stem(u)}.tmp.npz", **arr)
        (REC / f"{stem(u)}.tmp.npz").rename(REC / f"{stem(u)}.npz")
        atomic_csv(df.assign(split=s, subset=x), f)
        print(s, u.name, len(df), round(df.JF.mean(), 4), round(df.JF_G.mean(), 4), f"{time.time() - t0:.0f}s", flush=True)
    status(completed=f"stage0 rec {sets}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[a.what](*a.args)


if __name__ == "__main__":
    main()
