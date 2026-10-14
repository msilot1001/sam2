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
import os
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


PREREG_R2 = {
    "created_after": "stage-0 recording + replay of 20 policies on DEV+REP (all S/F/M/R pilots REJECT or INCONCLUSIVE in "
                     "replay; oracles: selection {G,L} +0.018, fusion bound +0.020, component bound +0.024, extra candidates "
                     "{L2,Lp,Lt} +0.033). Frozen before any T-head output and before any causal pilot.",
    "T_heads": {"variants": "T0 (no transfer), T1 (CNN fusion), T2 (+target tokens), T3 (+abstain gate), T4 (+identity gate)",
                "data": "REP cache every 2nd frame on the live SGGL trajectory; fit/val split 4:1 by video hash",
                "positive_control": "T1 and T4 on 64 fixed fit items, 1500 steps, lr 1e-3 cosine, bs 8: pass iff fit J - input J "
                                    ">= +0.05",
                "main": "3000 steps, bs 8, AdamW 1e-3 cosine, wd 1e-4, grad clip 5, seed 0; same for every variant",
                "screen": "DEV, output-only along the live SGGL trajectory (all heads in one pass); selection = SGGL's self gate",
                "attention_claim": "T2/T3/T4 must beat T1 on DEV small units by >= +0.005 (video CI > 0) before any attention-"
                                   "specific benefit is claimed; T1 must beat T0 for any cross-view feature benefit",
                "advance": "pilot rule on DEV small units (REP is training data): dJF vs SGGL >= +0.005, >= 3 videos up, LOVO "
                           "min > 0; then a causal rollout"},
    "causal_pilots": {"why": "replay keeps SGGL's crop trajectory; verify replay ~ causal for the best point estimate of "
                             "each non-learned family",
                      "arms": ["F1-blend.7 (F)", "S2-observe (S)", "M1-always (M)"],
                      "rule": "same pilot rule as stage 1 on causal DEV+REP small units"},
}


def prereg_r2():
    m55._freeze(OUT / "preregistration_r2.json", {**PREREG_R2, "rows": []})


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


def shard(us):
    """Disjoint subset for parallel workers: M7_SHARD="i/n" keeps units with index % n == i (unit order is fixed)."""
    sh = os.environ.get("M7_SHARD")
    if not sh:
        return us
    i, n = map(int, sh.split("/"))
    return [x for k, x in enumerate(us) if k % n == i]


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


# ----------------------------------------------------------------------------- causal rollout of a policy
def live_obs(d, t, views, hw):
    """Observation dict for policies (same fields as m7_replay.observations, fp32 logits)."""
    G, L = views["G"], views["L"]
    o = {"t": t, "hw": hw, "box": L.box, "sG": G.score, "sL": L.score, "touchL": vcam.touches_border(L.mask, L.box, hw),
         "simL": None, "lG": sggl.full_to_frame(G.logit, hw), "mG": G.mask, "mL_view": L.mask}
    for k, v in views.items():
        if k == "G":
            continue
        o[f"s{k}"] = v.score
        o[f"l{k}"] = sggl.crop_to_frame(v.logit, v.box, hw)
        o[f"m{k}"] = vcam.paste(v.mask, v.box, hw)
    return o


def run_policy(pred, s, u, P, n=None, head=None):
    """Causal rollout: the policy's output sets the next crop centre. Probes are run only if the policy needs them."""
    from spatial import policies
    n = n or len(u.frames)
    g0 = gt(u, 0)
    hw = g0.shape
    d = sggl.Dual(pred, paths(u), g0, scale2=2.0 if "L2" in P.needs else None)
    y0, y1, x0, x1 = d.S["L"].frames.box
    refL = _pool(d.features("L", 0), g0[y0:y1, x0:x1], None)
    tok = None
    if head is not None:
        with torch.inference_mode():
            tok = _tokens(d, g0)
    pol, trusted, rows = P(g0), [], []
    for t in range(1, n):
        t0 = time.perf_counter()
        views = {"G": d.view("G", t), "L": d.view("L", t)}
        if "L2" in P.needs:
            views["L2"] = d.view("L2", t)
        if "Lp" in P.needs:
            views["Lp"] = d.probe("L", t, "protected")
        if "Lt" in P.needs:
            views["Lt"] = d.probe("L", t, "trusted", trusted=trusted)
        if head is not None:
            with torch.inference_mode():
                views["L"].logit = head(t_inputs(d, views["G"], views["L"], hw, tok, t)).float().cpu()
            side = views["L"].mask.shape[0]
            views["L"].mask = Fn.interpolate(views["L"].logit, size=(side, side), mode="bilinear", align_corners=False)[0, 0].numpy() > 0
        o = live_obs(d, t, views, hw)
        with torch.inference_mode():
            o["simL"] = _cos_map(d.features("L", t), refL).float().cpu().numpy()
        out, acc = pol(o)
        torch.cuda.synchronize()
        lat = time.perf_counter() - t0
        G, L = views["G"], views["L"]
        if L.score > 0 and G.score > 0 and policies._iou(o["mL"], G.mask) > 0.5:
            trusted.append(t)
        g = gt(u, t)
        rows.append({"set": s, "unit": u.name, "video": u.video, "arm": P.name, "video_idx": t, "acc": acc, "latency_s": lat,
                     "score_G": G.score, "score_L": L.score, "gt_empty": not g.any(), "area0": float(g0.mean()),
                     **m55.jf(out, g), "JF_G": m55.jf(G.mask, g)["JF"], **{f"{k}": v for k, v in pol.log.items()}})
        pol.log = {}
        d.set_center(pol.center(out) if hasattr(pol, "center") else vcam.center_of(out))
    peak = torch.cuda.max_memory_allocated()
    d.close()
    return pd.DataFrame(rows).assign(peak_vram_bytes=peak)


def run(arm, sets="DEV,REP", head_name=""):
    """Causal rollout of policy `arm` (class name or .name in spatial.policies) on the given sets."""
    from spatial import policies
    P = next(c for c in policies.ALL + policies.CAUSAL_ONLY if arm in (c.__name__, c.name))
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    head = None
    if head_name:
        from spatial.feature_fusion import FusionHead
        sggl.capture_pix(pred)
        head = FusionHead(head_name.split("_")[0]).cuda().eval()
        head.load_state_dict(torch.load(HEADS / f"{head_name}.pt"))
    tag = P.name + (f"+{head_name}" if head_name else "")
    dd = OUT / "runs" / tag
    dd.mkdir(parents=True, exist_ok=True)
    for s, x, u in shard(units(sets)):
        f = dd / f"{stem(u)}.csv"
        if f.exists():
            continue
        torch.cuda.reset_peak_memory_stats()
        df = run_policy(pred, s, u, P, head=head)
        atomic_csv(df.assign(subset=x, arm=tag), f)
        print(tag, u.name, round(df.JF.mean(), 4), flush=True)
    if not os.environ.get("M7_SHARD"):
        status(completed=f"run {tag} {sets}")


# ----------------------------------------------------------------------------- T family: feature cache, training
TC = OUT / "tcache"
HEADS = OUT / "heads"


def _soft(m, size=256):
    return Fn.interpolate(torch.as_tensor(m, dtype=torch.float32)[None, None], size=(size, size), mode="area")


def _tokens(d, g0):
    """Frame-0 reference tokens (fg/bg, local and global view) from stride-16 backbone features; prompt only."""
    from spatial import detail
    y0, y1, x0, x1 = d.S["L"].frames.box
    mL = g0[y0:y1, x0:x1]
    bgL = ~detail.dilate(mL, max(2, d.side // 32))
    ringG = np.zeros_like(g0)
    ringG[y0:y1, x0:x1] = True
    bgG = ringG & ~detail.dilate(g0, max(2, d.side // 32))
    fL, fG = d.features("L", 0), d.features("G", 0)
    return torch.stack([_pool(fL, mL, None), _pool(fL, bgL, None), _pool(fG, g0, None), _pool(fG, bgG, None)])


def t_inputs(d, G, L, hw, tok, t):
    """One frame's head inputs on the crop grid (GPU tensors, batch 1)."""
    box = L.box
    return {"memL": L.pix, "bbL": d.features("L", t), "lL": L.logit.cuda(),
            "memG": sggl.sample_global_in_crop(G.pix, box, hw, 32), "bbG": sggl.sample_global_in_crop(d.features("G", t), box, hw, 32),
            "lG": sggl.sample_global_in_crop(G.logit.cuda(), box, hw, 256), "sL": torch.tensor([L.score], device="cuda"),
            "sG": torch.tensor([G.score], device="cuda"), "tok": tok[None]}


def tcache(sets="REP", every="2"):
    """Live SGGL rollout storing T-head inputs (fp16) for every `every`-th frame (training data; DAVIS train only)."""
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    sggl.capture_pix(pred)
    TC.mkdir(parents=True, exist_ok=True)
    for s, x, u in units(sets):
        assert s == "REP", "T heads are trained on DAVIS train (REP) only"
        f = TC / f"{stem(u)}.pt"
        if f.exists():
            continue
        g0 = gt(u, 0)
        hw = g0.shape
        d = sggl.Dual(pred, paths(u), g0)
        with torch.inference_mode():
            tok = _tokens(d, g0)
        keep = {k: [] for k in ("memL", "bbL", "lL", "memG", "bbG", "lG", "sL", "sG", "gt", "t", "a2x")}
        for t in range(1, len(u.frames)):
            G, L = d.view("G", t), d.view("L", t)
            out, ok = sggl.a2x(G, L, hw)
            if t % int(every) == 0:
                with torch.inference_mode():
                    xi = t_inputs(d, G, L, hw, tok, t)
                for k in ("memL", "bbL", "lL", "memG", "bbG", "lG"):
                    keep[k].append(xi[k][0].half().cpu())
                y0, y1, x0, x1 = L.box
                keep["gt"].append((_soft(gt(u, t)[y0:y1, x0:x1])[0] * 255).round().to(torch.uint8))
                keep["sL"].append(L.score), keep["sG"].append(G.score), keep["t"].append(t), keep["a2x"].append(ok)
            d.set_center(vcam.center_of(out))
        d.close()
        blob = {k: (torch.stack(v) if torch.is_tensor(v[0]) else torch.tensor(v)) for k, v in keep.items() if v}
        blob.update(tok=tok.half().cpu(), unit=u.name, video=u.video)
        torch.save(blob, TC / f"{stem(u)}.tmp")
        (TC / f"{stem(u)}.tmp").rename(f)
        print("tcache", u.name, len(keep["t"]), flush=True)
    status(completed=f"tcache {sets} every {every}")


def _tdata(split):
    """REP cache split video-disjointly (fixed hash): 'fit' ~80 % of videos, 'val' ~20 %."""
    from rfeas.units import stable_hash
    items = []
    for f in sorted(TC.glob("*.pt")):
        b = torch.load(f)
        sp = "val" if stable_hash(f"m7-tsplit|{b['video']}") % 5 == 0 else "fit"
        if split in (sp, "all"):
            for i in range(len(b["t"])):
                items.append({**{k: b[k][i] for k in ("memL", "bbL", "lL", "memG", "bbG", "lG", "gt")},
                              "sL": b["sL"][i], "sG": b["sG"][i], "tok": b["tok"], "video": b["video"]})
    return items


_TKEYS = ("memL", "bbL", "lL", "memG", "bbG", "lG", "tok")


class _Stacked:
    """The same items as one stacked tensor per field, resident on `device` (throughput: no per-step CPU stacking or
    host->GPU copies). Batches are value-identical to the list path (tested)."""

    def __init__(self, items, device="cuda"):
        self.n, self.videos = len(items), [it["video"] for it in items]
        self.T = {k: torch.stack([it[k] for it in items]).to(device) for k in _TKEYS + ("gt",)}
        self.sL = torch.tensor([float(it["sL"]) for it in items], device=device)
        self.sG = torch.tensor([float(it["sG"]) for it in items], device=device)

    def __len__(self):
        return self.n


def _videos(items):
    return items.videos if isinstance(items, _Stacked) else [i["video"] for i in items]


def _batch(items, idx):
    if isinstance(items, _Stacked):
        j = torch.as_tensor(np.asarray(list(idx)), device=items.sL.device)
        x = {k: items.T[k][j] for k in _TKEYS}
        x["lL"], x["lG"] = x["lL"].float(), x["lG"].float()
        x["sL"], x["sG"] = items.sL[j], items.sG[j]
        return x, items.T["gt"][j].float() / 255
    x = {k: torch.stack([items[i][k] for i in idx]).cuda() for k in ("memL", "bbL", "lL", "memG", "bbG", "lG", "tok")}
    x["lL"], x["lG"] = x["lL"].float(), x["lG"].float()
    x["sL"] = torch.tensor([float(items[i]["sL"]) for i in idx], device="cuda")
    x["sG"] = torch.tensor([float(items[i]["sG"]) for i in idx], device="cuda")
    y = torch.stack([items[i]["gt"] for i in idx]).cuda().float() / 255
    return x, y


def _jsoft(logit, y):
    """Mean J at the 256 grid (pred > 0 vs target >= 0.5) per sample."""
    p, g = logit > 0, y >= 0.5
    inter, union = (p & g).sum((1, 2, 3)).float(), (p | g).sum((1, 2, 3)).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


def ttrain(variant="T1", steps="3000", n_items="0", lr="1e-3", bs="8", tag=""):
    """Train a head on REP 'fit' (or the first n_items fixed items for the positive control). Cosine lr, AdamW."""
    from spatial.feature_fusion import FusionHead, loss_fn, n_params
    torch.manual_seed(0)
    fit = _tdata("fit")
    if int(n_items):
        rng = np.random.default_rng(0)
        fit = [fit[i] for i in sorted(rng.choice(len(fit), int(n_items), replace=False))]
    val = _tdata("val") if not int(n_items) else fit
    if os.environ.get("M7_TDATA_GPU", "1") == "1":
        fit = _Stacked(fit)
        val = fit if int(n_items) else _Stacked(val)
    head = FusionHead(variant).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=1e-4)
    steps, bs = int(steps), int(bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(1)
    curve, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()

    def evaluate(items):
        head.eval()
        js, j0, ls = [], [], []
        with torch.no_grad():
            for i in range(0, len(items), 16):
                x, y = _batch(items, range(i, min(i + 16, len(items))))
                out = head(x)
                ls.append(float(loss_fn(out, y)) * len(y))
                js += _jsoft(out, y).tolist()
                j0 += _jsoft(x["lL"], y).tolist()
        head.train()
        return sum(ls) / len(items), float(np.mean(js)), float(np.mean(j0))

    for it in range(1, steps + 1):
        x, y = _batch(fit, rng.choice(len(fit), bs, replace=False))
        loss = loss_fn(head(x), y)
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0))
        opt.step()
        sched.step()
        if it % 250 == 0 or it == 1:
            vl, vj, vj0 = evaluate(val)
            curve.append({"step": it, "train_loss": float(loss), "grad_norm": gn, "val_loss": vl, "val_J": vj, "val_J_input": vj0,
                          "lr": sched.get_last_lr()[0]})
            print(variant, curve[-1], flush=True)
    fl, fj, fj0 = evaluate(fit)
    rep = {"variant": variant, "steps": steps, "bs": bs, "lr": float(lr), "schedule": "cosine", "n_fit": len(fit),
           "n_val": len(val), "fit_videos": sorted(set(_videos(fit))), "val_videos": sorted(set(_videos(val))), "data_on_gpu": isinstance(fit, _Stacked),
           "params": n_params(head), "peak_vram_bytes": torch.cuda.max_memory_allocated(), "minutes": (time.time() - t0) / 60,
           "fit_J": fj, "fit_J_input": fj0, "curve": curve}
    HEADS.mkdir(parents=True, exist_ok=True)
    name = f"{variant}{tag}"
    torch.save(head.state_dict(), HEADS / f"{name}.pt")
    atomic_json(rep, HEADS / f"{name}.json")
    print(name, "fit J", round(fj, 4), "input", round(fj0, 4), flush=True)
    return rep


def teval(names, sets="DEV"):
    """Output-only screening of trained heads along the live SGGL trajectory (all heads in one pass). Output of head h
    at frame t = refined local logits if SGGL accepts the crop, else G (selection unchanged)."""
    from spatial.feature_fusion import FusionHead
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    sggl.capture_pix(pred)
    heads = {}
    for nm in names.split(","):
        h = FusionHead(nm.split("_")[0]).cuda().eval()
        h.load_state_dict(torch.load(HEADS / f"{nm}.pt"))
        heads[nm] = h
    dd = OUT / "teval"
    dd.mkdir(parents=True, exist_ok=True)
    for s, x, u in shard(units(sets)):
        f = dd / f"{stem(u)}.csv"
        if f.exists():
            continue
        g0 = gt(u, 0)
        hw = g0.shape
        d = sggl.Dual(pred, paths(u), g0)
        with torch.inference_mode():
            tok = _tokens(d, g0)
        rows = []
        for t in range(1, len(u.frames)):
            G, L = d.view("G", t), d.view("L", t)
            out, ok = sggl.a2x(G, L, hw)
            g = gt(u, t)
            r = {"set": s, "subset": x, "unit": u.name, "video": u.video, "video_idx": t, "area0": float(g0.mean()), "acc": ok,
                 "gt_empty": not g.any(), **{f"{k}|SGGL": v for k, v in m55.jf(out, g).items()}, "JF_G": m55.jf(G.mask, g)["JF"]}
            with torch.inference_mode():
                xi = t_inputs(d, G, L, hw, tok, t)
                for nm, h in heads.items():
                    t0 = time.perf_counter()
                    lg = h(xi).float()
                    torch.cuda.synchronize()
                    side = L.mask.shape[0]
                    mv = Fn.interpolate(lg, size=(side, side), mode="bilinear", align_corners=False)[0, 0].cpu().numpy() > 0
                    o = vcam.paste(mv, L.box, hw) if ok else G.mask
                    r.update({f"{k}|{nm}": v for k, v in m55.jf(o, g).items()})
                    r[f"lat|{nm}"] = time.perf_counter() - t0
                    if h.last_alpha is not None:
                        r[f"alpha|{nm}"] = float(h.last_alpha.mean())
            rows.append(r)
            d.set_center(vcam.center_of(out))
        d.close()
        atomic_csv(pd.DataFrame(rows), f)
        print("teval", u.name, flush=True)
    if not os.environ.get("M7_SHARD"):
        status(completed=f"teval {names} {sets}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[a.what](*a.args)


if __name__ == "__main__":
    main()
