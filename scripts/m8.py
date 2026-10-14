"""M8: diverse algorithm portfolio on top of SGGL (CUDA FP32, frozen SAM 2.1 Hiera-T).

    uv run python scripts/m8.py prereg              # freeze stage-1 protocol (before any M8 output)
    uv run python scripts/m8.py diag DEV,REP        # stage 1: live SGGL + every rescue generator on trigger frames
    uv run python scripts/m8.py run <arm> DEV,REP   # stage 2/3: causal rollout of a rescue arm (spatial.rescue_arms)

Sets are M7's (scripts/m7.py: DEV, REP, VK); PUMaVOS C is never touched. Parallel workers: M7_SHARD=i/n.
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
from spatial import detail, motion, rescue as R, sggl, vcam

_s = importlib.util.spec_from_file_location("m7", Path(__file__).with_name("m7.py"))
m7 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m7)
m55 = m7.m55
OUT = Path("outputs/milestone8")
DIAG = Path(os.environ.get("M8_DIAG_DIR", OUT / "diag"))     # workers: outputs/milestone8/runpod/diag


def status(**kw):
    p = OUT / "EXECUTION_STATUS.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    for k, v in kw.items():
        if isinstance(s.get(k), list) and not isinstance(v, list):
            s[k].append(v)
        else:
            s[k] = v
    s["updated_kst"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600))
    atomic_json(s, p)


PREREG = {
    "created_before_any_m8_output": True,
    "incumbent": "SGGL (M6 A2x; M7 KEEP_SGGL). M3-vel = low-cost control (damp 0.5, unchanged).",
    "data": "DEV + REP crop units (64 units / 34 videos; exploratory, previously inspected). VK (Scene01/06) only for frozen "
            "finalists. PUMaVOS C sealed. Learned modules: fit on REP, evaluated on DEV only.",
    "headroom_disclosed": "stage-0 (M7 rec): co-absence frames with GT present = 131 (14 units / 11 videos); perfect rescue "
                          "of all of them = +0.028 mean unit J&F. The report's +0.03 STRONG_WIN is therefore out of reach "
                          "for a rescue-only method on this data; reported anyway.",
    "stage1": {
        "trigger": "causal, production views only: coabs = both scores <= 0; disagree = exactly one present or IoU(L,G) < 0.5; "
                   "regular = every 10th other frame (calibration). All generators run NON-STORING on the live SGGL trajectory.",
        "generators": {"A_cos": "frame-0 prototype cosine on G stride-16 -> top-3 peaks -> point prompt on G memory",
                       "B_att": "contrastive token attention (8 fg + 32 bg k-means tokens, tau 0.1) -> top-3 -> point prompt",
                       "H_s8": "prototype cosine on stride-8 FPN features (representation) -> top-3 -> point prompt",
                       "G_neg": "B top-1 positive + negative points at A peaks with B posterior < 0.5",
                       "D_ref": "B top-1 point prompt with frame-0-only memory",
                       "A/B_crop": "L-memory crop re-placed at the A / B top-1 peak (attention-guided crop placement)",
                       "E_beam1/2": "L-memory crop at last confident centre + 1x / 2x raw velocity x elapsed frames",
                       "C_flowcrop/C_flowbox": "RAFT-small flow-carried box -> L crop at its centre / G box prompt",
                       "F_oclr/cpfr{8,4}_{G,L}": "matched-magnification plain crop vs context-preserving separable foveation "
                                                "(target long side -> 8 or 4 stride-16 cells), on G and L memory",
                       "Gp/Lp": "M7 protected-memory probes (reference)"},
        "metrics": "per generator on coabs: recall (any of its candidates J > 0.5) when GT present; false-positive rate (non-empty "
                   "with score > 0) when GT empty; localisation hit@1/@3 of heatmap peaks (GT cells dilated by 1 cell); "
                   "video spread of hits",
        "advance_to_causal": "recall >= 0.25 AND >= 2x max(Gp, Lp) recall in the same run AND hits in >= 4 videos AND, with a "
                             "ranker/threshold fit on REP only, FP rate on DEV coabs-absent <= 0.10",
        "falsification": {"attention (B vs A)": "B_att loc@1 and recall not above A_cos (paired, video-clustered) -> no attention "
                                                "benefit for search",
                          "representation (H vs A)": "stride-8 not above stride-16 -> occupancy-of-features hypothesis unsupported",
                          "foveation (cpfr vs oclr, same state, same magnification)": "recall difference CI includes 0 -> "
                                                "context preservation unsupported",
                          "motion (C/E vs L)": "flow/beam recall not above the production L view at coabs -> motion adds no evidence"},
        "rankers": "R0 SAM 2 score; R1 train-free (score > 0, cos_fg - cos_bg, area ratio, flow distance); R2 logistic regression; "
                   "R3 attention ranker over candidate tokens + frame-0 tokens. R2/R3 fit on REP, evaluated on DEV.",
    },
    "stage2_causal_rule": {"PROMISING": "M7 pilot rule vs machine-matched SGGL: dJF >= +0.005 AND >= 3 videos > +0.01 AND LOVO "
                                        "min > 0 AND > 0 without miami-surf AND no unit < -0.05",
                           "STRONG_WIN": "dJF >= +0.03 AND video-CI lower > 0",
                           "REJECTED": "dJF <= 0 or gains from <= 2 videos with LOVO min <= 0",
                           "INCONCLUSIVE": "otherwise",
                           "also_reported": "coabs-subset dJF, false re-acquisition (output non-empty on GT-empty frames), "
                                            "latency, peak VRAM, extra SAM 2 calls"},
    "no_tuning": "each causal arm gets at most one pre-registered revision; thresholds fixed before its causal run",
}


def prereg():
    m55._freeze(OUT / "preregistration.json", {**PREREG, "rows": []})


# ----------------------------------------------------------------------------- causal target context
def trigger(G, L, iou_lg, t):
    """Causal trigger type of frame t from production views only."""
    if G.score <= 0 and L.score <= 0:
        return "coabs"
    if (G.score > 0) != (L.score > 0) or iou_lg < 0.5:
        return "disagree"
    return "regular" if t % 10 == 0 else None


@torch.inference_mode()
def raw_s8(pred, s, t):
    """Raw 256-d stride-8 FPN-neck features of the state's cached frame-t image. (SAM 2's forward_image replaces
    backbone_fpn[1] by the decoder's conv_s1 projection (64-d high-res detail features), so it is recomputed here.)"""
    img = s.st["cached_features"][t][0]
    return pred.image_encoder(img.to(s.st["device"]))["backbone_fpn"][1]


class Refs:
    """Frame-0 references (prompt only): G stride-16 prototype + k-means fg/bg tokens, stride-8 prototype."""

    def __init__(self, d, g0):
        fG = d.features("G", 0)
        f8 = raw_s8(d.pred, d.S["G"], 0)
        self.p16 = R.proto(fG, g0)
        self.p8 = R.proto(f8, g0)
        bg = ~detail.dilate(g0, max(4, int(np.sqrt(g0.sum()))))
        self.fg, self.bg = R.tokens(fG, g0, bg)
        self.pbg = R.proto(fG, bg)


def cand_feats(featG, m, refs):
    v = R.proto(featG, m)
    if v is None:
        return None, np.nan, np.nan
    return v, float(v @ refs.p16), float(v @ refs.pbg)


# ----------------------------------------------------------------------------- stage 1: generator diagnostics
def generators(pred, d, t, hw, frame, refs, trk, iou_box, only=None):
    """Rescue generators on frame t (non-storing). only: set of generator names (None = all, stage 1).
    Returns ([(gen, frame-res logit (NaN = unobserved), score, iou, centre-hint)], maps, peaks)."""
    H, W = hw
    SG, SL = d.S["G"], d.S["L"]
    want = (lambda k: True) if only is None else (lambda k: any(o == k or o.startswith(k + "#") for o in only))
    featG = d.features("G", t)
    side = d.side
    sx, sy = R.M / W, R.M / H
    maps = {}
    if want("A_cos") or want("A_cos_crop") or want("G_neg"):
        maps["A_cos"] = R.heat_proto(featG, refs.p16)
    if want("B_att") or want("B_att_crop") or want("G_neg") or want("D_ref"):
        maps["B_att"] = R.heat_contrast(featG, refs.fg, refs.bg)
    if want("H_s8"):
        maps["H_s8"] = R.heat_proto(raw_s8(pred, SG, t), refs.p8)
    pk = {k: R.peaks(v, 3, 3 if k != "H_s8" else 6) for k, v in maps.items()}
    out = []

    def keep(name):
        return only is None or name in only

    def G_prompt(name, pts, labels, planner=None):
        if keep(name):
            o, lg, sc, _ = R.probe(pred, SG, t, np.asarray(pts) * [sx, sy], labels, planner=planner)
            out.append((name, sggl.full_to_frame(lg, hw), sc, iou_box[0], o))

    vcache = {}

    def view(name, s, my, mx, pts=None, labels=None, planner=None):
        if keep(name):
            key = (tuple(my.xin), tuple(my.xout), tuple(mx.xin), tuple(mx.xout))
            if key not in vcache:                         # one backbone pass per distinct view image
                img = R.view_image(frame, my, mx)
                vcache[key] = (img, pred.forward_image(img.to(s.st["device"])))
            img, bb = vcache[key]
            vp = None if pts is None else R.frame_to_view_xy(pts, my, mx)
            o, lg, sc, _ = R.probe(pred, s, t, vp, labels, image=img, planner=planner, feats=bb)
            out.append((name, R.view_to_frame(lg, my, mx, hw), sc, iou_box[0], o))

    # A / B / H: global target-conditioned search -> point prompts on G's memory
    for k in ("A_cos", "B_att", "H_s8"):
        if k in maps:
            gh, gw = maps[k].shape
            for i, (cy, cx, v) in enumerate(pk[k]):
                G_prompt(f"{k}#{i}", [R.cell_to_frame(cy, cx, (gh, gw), hw)], [1])
    if "B_att" in maps:
        # G: target-vs-distractor negative prompts (pos = B top-1, neg = A peaks the contrastive map calls background)
        by, bx, _ = pk["B_att"][0]
        pos = R.cell_to_frame(by, bx, (64, 64), hw)
        if "A_cos" in maps:
            negs = [R.cell_to_frame(cy, cx, (64, 64), hw) for cy, cx, _ in pk["A_cos"]
                    if max(abs(cy - by), abs(cx - bx)) > 3 and float(maps["B_att"][cy, cx]) < 0.5]
            G_prompt("G_neg", [pos] + negs, [1] + [0] * len(negs))
        # D: reference-conditioned re-prompt (frame-0 memory only) at the attention peak
        G_prompt("D_ref", [pos], [1], planner=sggl._cond_only)
    # B/A-guided crop placement on L's memory (no prompt)
    for k in ("A_cos", "B_att"):
        if k in maps:
            cy, cx, _ = pk[k][0]
            x, y = R.cell_to_frame(cy, cx, (64, 64), hw)
            view(f"{k}_crop", SL, *R.box_maps(vcam.place((y, x), side, hw)))
    # E: motion search beam (L's memory, crop side unchanged): last centre + 1x / 2x raw velocity extrapolation
    for k in (1.0, 2.0):
        view(f"E_beam{k:g}", SL, *R.box_maps(vcam.place(trk.extrapolate(t, k), side, hw)))
    # C: flow-carried box -> L crop at its centre; G box prompt
    fy, fx = trk.flow_center()
    view("C_flowcrop", SL, *R.box_maps(vcam.place((fy, fx), side, hw)))
    y0, y1, x0, x1 = trk.fbox
    ey, ex = (y1 - y0) * 0.1 + 2, (x1 - x0) * 0.1 + 2
    G_prompt("C_flowbox", [(x0 - ex, y0 - ey), (x1 + ex, y1 + ey)], [2, 3])
    # F: occupancy-controlled crop vs context-preserving foveation, matched magnification, centred on the flow box
    dh = trk.size()
    for cells in (8, 4):
        a = R.occupancy_alpha(dh, cells)
        for fov in (False, True):
            nm = f"F_{'cpfr' if fov else 'oclr'}{cells}"
            if any(keep(nm + x) for x in ("_G", "_L", "_G_pt", "_L_pt")):
                my, mx = R.axis_map(H, fy, a, fov), R.axis_map(W, fx, a, fov)
                view(nm + "_G", SG, my, mx)
                view(nm + "_G_pt", SG, my, mx, [(fx, fy)], [1])      # amendment 1: + point prompt at the predicted centre
                if cells == 8:
                    view(nm + "_L", SL, my, mx)
                    view(nm + "_L_pt", SL, my, mx, [(fx, fy)], [1])
    # M7 references: protected memory on both states
    for nm, s in (("Gp", "G"), ("Lp", "L")):
        if keep(nm):
            v = d.probe(s, t, "protected")
            out.append((nm, sggl.full_to_frame(v.logit, hw) if v.box is None else sggl.crop_to_frame(v.logit, v.box, hw),
                        v.score, v.iou, v.out))
    return out, maps, pk


def diag_unit(pred, flow, s, u, n=None, only=None):
    n = n or len(u.frames)
    g0 = m7.gt(u, 0)
    hw = g0.shape
    ps = m7.paths(u)
    d = sggl.Dual(pred, ps, g0)
    refs = Refs(d, g0)
    trk = motion.Track(vcam.bbox(g0))
    iou_box = d._iou
    prev = R.frame_tensor(ps[0]).cuda()
    rows, crows, vecs = [], [], []
    for t in range(1, n):
        t0 = time.perf_counter()
        G, L = d.view("G", t), d.view("L", t)
        out, ok = sggl.a2x(G, L, hw)
        torch.cuda.synchronize()
        lat_sggl = time.perf_counter() - t0
        frame = R.frame_tensor(ps[t]).cuda()
        t1 = time.perf_counter()
        trk.step_flow(flow(prev, frame), hw)
        lat_flow = time.perf_counter() - t1
        pL = vcam.paste(L.mask, L.box, hw)
        iou_lg = m7._iou(pL, G.mask)
        trig = trigger(G, L, iou_lg, t)
        g, oth = m7.gt(u, t), m7.other(u, t)
        r = {"set": s, "unit": u.name, "video": u.video, "video_idx": t, "area0": float(g0.mean()), "acc": ok, "trig": trig,
             "score_G": G.score, "score_L": L.score, "gt_empty": not g.any(), "gt_area": int(g.sum()), **m55.jf(out, g),
             "JF_G": m55.jf(G.mask, g)["JF"], "J_L": j_measure(pL, g), "lat_sggl": lat_sggl, "lat_flow": lat_flow,
             "fbox_hit": bool(g.any() and g[int(trk.fbox[0]):int(np.ceil(trk.fbox[1])), int(trk.fbox[2]):int(np.ceil(trk.fbox[3]))].any()),
             "gt_in_crop": float(g[L.box[0]:L.box[1], L.box[2]:L.box[3]].sum() / g.sum()) if g.any() else np.nan}
        if trig is not None:
            t2 = time.perf_counter()
            with torch.inference_mode():
                cands, maps, pk = generators(pred, d, t, hw, frame, refs, trk, iou_box, only=only)
                featG = d.features("G", t)
                gg = Fn.interpolate(torch.as_tensor(detail.dilate(g, 0), dtype=torch.float32)[None, None], size=(64, 64), mode="area")[0, 0] > 0
                gg = Fn.max_pool2d(gg[None, None].float(), 3, 1, 1)[0, 0] > 0           # GT cells dilated by one cell
                g8 = Fn.max_pool2d((Fn.interpolate(torch.as_tensor(g, dtype=torch.float32)[None, None], size=(128, 128), mode="area") > 0).float(), 5, 1, 2)[0, 0] > 0
                for k, pl in pk.items():
                    gk = g8 if k == "H_s8" else gg
                    hits = [bool(gk[cy, cx]) for cy, cx, _ in pl]
                    r[f"loc1_{k}"] = hits[0] if g.any() else np.nan
                    r[f"loc3_{k}"] = any(hits) if g.any() else np.nan
                    r[f"peak_{k}"] = pl[0][2]
                    if g.any():                                    # best heat on the target vs the global max
                        r[f"heat_gt_{k}"] = float(maps[k][gk.to(maps[k].device)].max())
                r["lat_rescue"] = time.perf_counter() - t2
                r["n_cand"] = len(cands)
                cy_, cx_ = trk.flow_center()
                for name, lg, sc, iou, _ in cands:
                    m = np.nan_to_num(lg, nan=-1.0) > 0
                    v, cos_fg, cos_bg = cand_feats(featG, m, refs)
                    c = vcam.center_of(m)
                    crows.append({"unit": u.name, "video": u.video, "video_idx": t, "trig": trig, "gen": name, "score": sc,
                                  "iou": iou, "area": int(m.sum()), "cos_fg": cos_fg, "cos_bg": cos_bg,
                                  "dist_flow": np.nan if c is None else float(np.hypot(c[0] - cy_, c[1] - cx_) / d.side),
                                  "area_ratio": float(m.sum() / max(g0.sum(), 1)),
                                  "J": j_measure(m, g), "oth": float((m & oth).sum() / m.sum()) if m.any() else np.nan,
                                  "gt_empty": not g.any()})
                    vecs.append(np.zeros(256, np.float16) if v is None else v.half().cpu().numpy())
        rows.append(r)
        trk.observe(t, vcam.bbox(out))
        d.set_center(vcam.center_of(out))
        prev = frame
    peak = torch.cuda.max_memory_allocated()
    d.close()
    return pd.DataFrame(rows).assign(peak_vram_bytes=peak), pd.DataFrame(crows), np.stack(vecs) if vecs else np.zeros((0, 256), np.float16)


def diag(sets="DEV,REP"):
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    flow = motion.Flow()
    DIAG.mkdir(parents=True, exist_ok=True)
    skip = set(os.environ.get("M8_SKIP", "").split(","))      # units already recorded on another machine
    for s, x, u in m7.shard(m7.units(sets)):
        f = DIAG / f"{m7.stem(u)}.csv"
        if f.exists() or m7.stem(u) in skip:
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        df, cd, vec = diag_unit(pred, flow, s, u)
        np.save(DIAG / f"{m7.stem(u)}.vec.npy", vec)
        atomic_csv(cd, DIAG / f"{m7.stem(u)}.cand.csv")
        atomic_csv(df.assign(subset=x), f)
        print("diag", u.name, len(df), int(df.trig.notna().sum()), round(df.JF.mean(), 4), f"{time.time() - t0:.0f}s", flush=True)
    if not os.environ.get("M7_SHARD"):
        status(completed=f"diag {sets}")


F_PT = {"F_oclr8_G_pt", "F_cpfr8_G_pt", "F_oclr8_L_pt", "F_cpfr8_L_pt", "F_oclr4_G_pt", "F_cpfr4_G_pt"}
H8 = {"H_s8#0", "H_s8#1", "H_s8#2"}     # amendment 2: shard-0 H_s8 used decoder-projected (conv_s1) features


def diag1b(sets="DEV,REP"):
    """Amendment 1: prompted F generators for units whose diag ran before they existed (same trajectory, same triggers)."""
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    flow = motion.Flow()
    for s, x, u in m7.shard(m7.units(sets)):
        st = m7.stem(u)
        f = DIAG / f"{st}.cand1b.csv"
        if f.exists() or not (DIAG / f"{st}.cand.csv").exists():
            continue
        c0 = pd.read_csv(DIAG / f"{st}.cand.csv")
        if len(c0) and c0.gen.isin(F_PT).any():           # recorded after amendments 1-2
            continue
        df, cd, vec = diag_unit(pred, flow, s, u, only=F_PT | H8)
        ref = pd.read_csv(DIAG / f"{st}.csv")
        assert np.allclose(df.JF.to_numpy(), ref.JF.to_numpy(), atol=1e-9), "trajectory differs from the stage-1 diag"
        cols = ["unit", "video_idx"] + [c for c in df.columns if c.endswith("_H_s8")]
        atomic_csv(df[df.trig.notna()][cols], DIAG / f"{st}.diag1b.csv")
        np.save(DIAG / f"{st}.vec1b.npy", vec)
        atomic_csv(cd, f)
        print("diag1b", u.name, len(cd), flush=True)


def refs(sets="DEV,REP"):
    """Frame-0 reference tokens per unit (prompt frame only) for the set ranker R3."""
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    for s, x, u in m7.units(sets):
        f = DIAG / f"{m7.stem(u)}.refs.pt"
        if f.exists():
            continue
        g0 = m7.gt(u, 0)
        d = sggl.Dual(pred, m7.paths(u), g0)
        with torch.inference_mode():
            rf = Refs(d, g0)
        torch.save({"fg": rf.fg.cpu(), "bg": rf.bg.cpu(), "p16": rf.p16.cpu(), "pbg": rf.pbg.cpu()}, f)
        d.close()


# ----------------------------------------------------------------------------- stage 2/3: causal rescue arms
# Each arm: SGGL every frame; on a causal trigger it runs `gens` (non-storing), ranks them with `ranker`, and if the
# best candidate is accepted outputs it and re-centres the crop on it (and optionally writes it into the producing
# state's memory: `admit`). `center` = "out" (SGGL) or "m3vel" (M3-vel damped velocity). Arms are frozen in
# outputs/milestone8/arms.json before their first output.
ARMS_FILE = OUT / "arms.json"
RANKERS = OUT / "rankers"


def arms():
    return json.loads(ARMS_FILE.read_text()) if ARMS_FILE.exists() else {}


def freeze_arm(name, spec_json):
    """Add arm `name` to arms.json before its first output (refuses to change an existing arm)."""
    a = arms()
    spec = {"name": name, **json.loads(spec_json), "frozen_kst": time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600))}
    if name in a:
        if {k: v for k, v in a[name].items() if k != "frozen_kst"} != {k: v for k, v in spec.items() if k != "frozen_kst"}:
            raise SystemExit(f"arm {name} is frozen and differs")
        return
    assert not any((d / name).exists() for d in (OUT / "runs", OUT / "runpod" / "runs")), "arm already has outputs"
    a[name] = spec
    atomic_json(a, ARMS_FILE)
    print("froze arm", name)


class Rank:
    def __init__(self, spec):
        from spatial import ranker
        self.spec, self.rk = spec, ranker
        if spec["kind"] == "lr":
            self.lr = ranker.LR.load(json.loads((RANKERS / f"{spec['name']}.json").read_text()))
        elif spec["kind"] == "set":
            blob = torch.load(RANKERS / f"{spec['name']}.pt", weights_only=False)
            self.nets = []
            for st in blob["states"]:                          # two video-fold models, scores averaged
                n = ranker.SetRanker(len(blob["gens"])).eval()
                n.load_state_dict(st)
                self.nets.append(n)
            self.gens, self.mu, self.sd = blob["gens"], blob["mu"], blob["sd"]

    def __call__(self, cdf, vecs, refs_t):
        """Returns (index of best candidate or None, its score)."""
        if not len(cdf):
            return None, -np.inf
        k = self.spec["kind"]
        if k == "score":
            s = cdf.score.to_numpy()
        elif k == "r1":
            s = self.rk.r1(cdf)
        elif k == "lr":
            s = self.lr(self.rk.scalars(cdf))
        else:
            with torch.no_grad():
                X = torch.from_numpy(np.concatenate([vecs, (self.rk.scalars(cdf) - self.mu) / self.sd], 1)).float()[None]
                gid = torch.tensor([self.gens.index(g.split("#")[0]) for g in cdf.gen])[None]
                ref = torch.cat([refs_t["fg"], refs_t["bg"]]).float()[None]
                rt = torch.tensor([0] * len(refs_t["fg"]) + [1] * len(refs_t["bg"]))[None]
                s = 0
                for net in self.nets:
                    lo = net(X, gid, ref, rt, torch.zeros(1, len(cdf), dtype=torch.bool))[0]
                    s = s + (lo[:-1] - lo[-1]).numpy() / len(self.nets)      # margin over abstaining
        i = int(np.argmax(s))
        return (i, float(s[i])) if s[i] > self.spec.get("thr", 0.0) else (None, float(s[i]))


def m3_center(state, out):
    c = vcam.center_of(out)
    if c is None:
        state["prev"] = None
        return None
    p = state.get("prev")
    state["prev"] = c
    return c if p is None else (c[0] + 0.5 * (c[0] - p[0]), c[1] + 0.5 * (c[1] - p[1]))


def run_arm(pred, flow, s, u, A, rank, n=None):
    n = n or len(u.frames)
    g0 = m7.gt(u, 0)
    hw = g0.shape
    ps = m7.paths(u)
    d = sggl.Dual(pred, ps, g0)
    rf = Refs(d, g0)
    refs_t = {"fg": rf.fg.cpu(), "bg": rf.bg.cpu()}
    trk = motion.Track(vcam.bbox(g0))
    use_flow = flow is not None
    prev = R.frame_tensor(ps[0]).cuda() if use_flow else None
    m3, rows, held = {}, [], 0
    for t in range(1, n):
        t0 = time.perf_counter()
        G, L = d.view("G", t), d.view("L", t)
        out, ok = sggl.a2x(G, L, hw)
        frame = R.frame_tensor(ps[t]).cuda() if (use_flow or A["gens"]) else None
        if use_flow:
            trk.step_flow(flow(prev, frame), hw)
        trig = G.score <= 0 and L.score <= 0
        res = {"rescued": False, "gen": None, "rscore": np.nan, "calls": 2}
        if trig and A["gens"]:
            with torch.inference_mode():
                cands, _, _ = generators(pred, d, t, hw, frame, rf, trk, d._iou, only=set(A["gens"]))
                featG = d.features("G", t)
                cr, vecs = [], []
                for name, lg, sc, iou, o in cands:
                    m = np.nan_to_num(lg, nan=-1.0) > 0
                    v, cf, cb = cand_feats(featG, m, rf)
                    c = vcam.center_of(m)
                    fy, fx = trk.flow_center()
                    cr.append({"gen": name, "score": sc, "iou": iou, "area": int(m.sum()), "cos_fg": cf, "cos_bg": cb,
                               "dist_flow": np.nan if c is None else float(np.hypot(c[0] - fy, c[1] - fx) / d.side),
                               "area_ratio": float(m.sum() / max(g0.sum(), 1))})
                    vecs.append(np.zeros(256, np.float32) if v is None else v.float().cpu().numpy())
                cdf = pd.DataFrame(cr)
                i, sc = rank(cdf, np.stack(vecs), refs_t)
            res.update(calls=2 + len(cands), rscore=sc)
            if i is not None:
                name, lg, _, _, o = cands[i]
                m = np.nan_to_num(lg, nan=-1.0) > 0
                if m.any():
                    out = m
                    res.update(rescued=True, gen=name)
                    if A.get("admit"):
                        st = d.S["G"] if name.startswith(("A_cos#", "B_att#", "H_s8#", "G_neg", "D_ref", "C_flowbox")) or name.endswith("_G") else d.S["L"]
                        st.od["non_cond_frame_outputs"][t] = o
        torch.cuda.synchronize()
        lat = time.perf_counter() - t0
        g = m7.gt(u, t)
        rows.append({"set": s, "unit": u.name, "video": u.video, "arm": A["name"], "video_idx": t, "acc": ok, "trig": trig,
                     "latency_s": lat, "score_G": G.score, "score_L": L.score, "gt_empty": not g.any(),
                     "area0": float(g0.mean()), **m55.jf(out, g), "JF_G": m55.jf(G.mask, g)["JF"], **res})
        trk.observe(t, vcam.bbox(out))
        if A.get("center") == "m3vel":
            c = m3_center(m3, out)
        elif A.get("center") == "flowvel":
            # C3: next centre = current target centre (output, else flow-carried box) + damped FLOW displacement of the
            # target region (dense motion, not mask-centre differences; keeps following the target while both views miss)
            c0 = vcam.center_of(out) or trk.flow_center()
            c = (c0[0] + 0.5 * trk.v[1], c0[1] + 0.5 * trk.v[0])
        else:
            c = vcam.center_of(out)
        d.set_center(c)
        prev = frame
    peak = torch.cuda.max_memory_allocated()
    d.close()
    return pd.DataFrame(rows).assign(peak_vram_bytes=peak)


def run(arm, sets="DEV,REP"):
    A = arms()[arm]
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    needs_flow = any(g.startswith(("C_", "F_")) for g in A["gens"]) or A.get("center") == "flowvel" or A.get("flow", False)
    flow = motion.Flow() if needs_flow else None
    rank = Rank(A["ranker"]) if A["gens"] else None
    dd = Path(os.environ.get("M8_RUNS_DIR", OUT / "runs")) / arm      # workers: M8_RUNS_DIR=outputs/milestone8/runpod/runs
    dd.mkdir(parents=True, exist_ok=True)
    env = {"machine": os.environ.get("M8_MACHINE", "ubuntu-rtx2080"), "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "fill_hole_area": pred.fill_hole_area, "arm": A,
           "code": json.loads(Path("CODE_SYNC.json").read_text()) if Path("CODE_SYNC.json").exists() else "ubuntu working tree"}
    shard_tag = os.environ.get("M7_SHARD", "all").replace("/", "of")
    atomic_json(env, dd / f"ENV_{env['machine']}_{shard_tag}.json")
    us = m7.units(sets)
    if os.environ.get("M8_SUBSET"):                       # benchmarking: fixed unit subset k/m, then M7_SHARD within it
        k, m = map(int, os.environ["M8_SUBSET"].split("/"))
        us = [x for i, x in enumerate(us) if i % m == k]
    for s, x, u in m7.shard(us):
        f = dd / f"{m7.stem(u)}.csv"
        if f.exists():
            continue
        torch.cuda.reset_peak_memory_stats()
        df = run_arm(pred, flow, s, u, A, rank)
        atomic_csv(df.assign(subset=x), f)
        print(arm, u.name, round(df.JF.mean(), 4), int(df.rescued.sum()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[a.what](*a.args)


if __name__ == "__main__":
    main()
