"""M6.1: overnight challenger tracks against the A2x incumbent (CUDA FP32, frozen SAM 2.1 Hiera-T).

    uv run python scripts/m61.py prereg        # freeze round-1 protocols (before any M6.1 output)
    uv run python scripts/m61.py tplr          # Track A: tracker-prompted local re-segmentation (new candidate masks)
    uv run python scripts/m61.py dtrain        # Track D: B1 + A1 logs on DAVIS *train* crop units (calibrator fitting)
    uv run python scripts/m61.py gates         # Track D: gate comparison (CPU, replay of logged arms)
    uv run python scripts/m61.py head_pc       # Track B: positive control (overfit 32 samples)
    uv run python scripts/m61.py head_train    # Track B: train refinement head v2 on DAVIS train
    uv run python scripts/m61.py head_image    # Track B: image-level eval (GT box prompt, retrospective)
    uv run python scripts/m61.py head_video    # Track B: B1 + head (output-only) video eval
    uv run python scripts/m61.py analyze

Reuses scripts/m6.py (units, GT, crop wrapper, run_unit) and M5.x helpers. PUMaVOS C is never touched.
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

from baseline.dropout_runner import atomic_csv, atomic_json, atomic_npz
from baseline.metrics import j_measure
from memint import controlled, engine
from spatial import vcam

_s = importlib.util.spec_from_file_location("m6", Path(__file__).with_name("m6.py"))
m6 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m6)
m55, m56 = m6.m55, m6.m56
OUT = Path("outputs/milestone61")
M6R = Path("outputs/milestone6/runs")
GATE_SETS = m6.GATE_SETS

PREREG = {
    "created_before_any_m61_output": True,
    "incumbent": "B2 = A2x (M6): small-unit dJF +0.018 vs P5 on DAVIS_val+PUMaVOS_B (exploratory)",
    "baselines": {"B0": "official", "B1": "P5", "B2": "A2x self-gated global-local", "B3": "A2c CTU agreement gate"},
    "track_A_TPLR": {
        "hypothesis": "SAM 2 image mode re-run on a zoomed local crop, prompted only by the online global tracker's own "
                      "box, yields masks that carry information absent from both the global (B1) and the crop-tracker "
                      "(A1) candidates.",
        "method": "frame t: if B1 obj score > 0 and B1 mask non-empty: crop side s = A1 rule (vcam.crop_side of the "
                  "prompt), centred on the B1 box centre (frame t, online); SAM 2 image predictor on the crop, box "
                  "prompt = B1 box in crop coords, multimask_output=False; output T1 = pasted local mask, else B1. "
                  "Output-only: B1 memory untouched.",
        "units": "every unit where A1 applied a crop (21 dev units, M6)",
        "candidate_value": "retrospective per-frame oracle over {B1, A1, T1} minus over {B1, A1} on small gate-set units "
                           ">= +0.01 => T1 adds information; else stop Track A",
        "deployable_success": "T1 small-unit dJF >= A2x small dJF + 0.01 on the gate set AND applied non-small mean "
                              ">= -0.005",
    },
    "track_D_calibration": {
        "fit_data": "B1 and A1 logged on DAVIS *train* units where A1 crops (43 units, 23 videos); never dev/confirm",
        "replay_variant": "A2o: crop trajectory of A1 (centre from crop output), output chosen per frame between A1 "
                          "and B1 by a gate; exactly replayable from logs (no state mixing). Gates compared on the same "
                          "variant: none (always crop), CTU agreement (IoU>0.5 & box/crop<=0.1), self (obj>0, no "
                          "interior edge), iou-only (iou_L > iou_G), calibrated logistic.",
        "calibrated": "logistic regression (sklearn-free, numpy IRLS, L2 1e-2) on standardized features [iou_L, iou_G, "
                      "score_L, score_G, IoU(L,G), touch_L, log(area_L+1), log(area_G+1), log crop side]; target "
                      "1[J_L > J_G]; frames where both J equal are dropped from fitting; gate p > 0.5",
        "success": "calibrated small-unit dJF (gate set) >= self gate + 0.005 with video-CI of the paired difference "
                   "excluding 0; else retain the A2x self gate",
        "also": "scale-bias verification: calibration of iou_pred vs J by arm and area bin on dev logs (CPU)",
    },
    "track_B_head_v2": {
        "hypothesis": "a light residual head on frozen SAM 2 decoder logits + stride-4/8 features + RGB, supervised at "
                      "1024 with a boundary-weighted loss, corrects systematic mask errors the frozen decoder makes.",
        "architecture": "256-grid branch: [logit/10, sigmoid, s0(32), up(s1)(64), rgb256(3)] -> 96ch convs dilations "
                        "1,2,4,8 -> delta256 (zero-init); 1024 branch: [up4(logit256+delta256), rgb1024] -> 3x conv 16 "
                        "-> delta1024 (zero-init); output logit1024. Untrained head == frozen decoder (tested).",
        "training": "DAVIS 2017 train only; image mode; box prompt from GT box with +-10% jitter; loss BCE + dice + 5x "
                    "BCE weight in a 3 px GT boundary band, all at 1024; AdamW 1e-3 cosine, 30 min cap, seed 0",
        "positive_control": "32 fixed DAVIS-train object-frames, 400 steps; pass iff train mean J(head) - J(frozen) "
                            ">= +0.05 on those samples. If it fails: Track B 'technically inconclusive', no main run",
        "image_eval": "retrospective GT-box frames (same as M6 round 4: small every 5th, ordinary every 10th)",
        "video_eval": "B1 + head on decoder output (output-only, memory untouched) on DAVIS_val + PUMaVOS_B (+A, "
                      "SegTrack if time); primary: overall unit-mean dJF vs B1 >= +0.01 with video-CI lower bound > 0 "
                      "on DAVIS_val+PUMaVOS_B and no set mean below -0.003; secondary: small-unit dJF, F, F_strict",
    },
    "track_C": "CPU diagnostics only unless >= 4 non-annotation independent videos with a correctable temporal "
               "opportunity (agent report); then pre-register separately",
    "confirmation": "PUMaVOS C at most once, for exactly one frozen method meeting its own success rule",
}


def prereg():
    m55._freeze(OUT / "preregistration.json", {**PREREG, "rows": []})


def status(**kw):
    p = OUT / "EXECUTION_STATUS.json"
    s = json.loads(p.read_text())
    for k, v in kw.items():
        if isinstance(s.get(k), list) and not isinstance(v, list):
            s[k].append(v)
        else:
            s[k] = v
    s["updated_kst"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600))
    atomic_json(s, p)


def _masks(arm, u, root=M6R):
    hw, n = m6.gt(u, 0).shape, len(u.frames)
    a = np.load(root / arm / (u.name.replace("/", "_") + ".npz"))[arm]
    return np.unpackbits(a)[:(n - 1) * hw[0] * hw[1]].reshape(n - 1, *hw).astype(bool)


# ----------------------------------------------------------------------------- Track A: TPLR
def tplr():
    from PIL import Image
    ipred = m55.build_image_predictor()
    d = OUT / "trackA" / "T1"
    d.mkdir(parents=True, exist_ok=True)
    for f in sorted((M6R / "A1").glob("*.csv")):
        A = pd.read_csv(f)
        name = A.unit[0]
        if (d / f"{f.stem}.csv").exists():
            continue
        u = m6._UNITS()[name]
        B = pd.read_csv(M6R / "B1" / f.name).set_index("video_idx")
        G = _masks("B1", u)
        hw = G.shape[1:]
        side = vcam.crop_side(m6.gt(u, 0))
        ps = m6.paths(u)
        rows, out = [], []
        for t in range(1, len(u.frames)):
            g, gm = m6.gt(u, t), G[t - 1]
            t0 = time.perf_counter()
            r = {"set": A.set[0], "unit": name, "video": u.video, "video_idx": t, "used_local": False, "iou_T": np.nan}
            o = gm
            if B.obj_score[t] > 0 and gm.any():
                box = vcam.place(vcam.center_of(gm), side, hw)
                y0, y1, x0, x1 = box
                b = vcam.bbox(gm[y0:y1, x0:x1])
                if b is not None:
                    with torch.inference_mode():
                        ipred.set_image(np.asarray(Image.open(ps[t]).convert("RGB"))[y0:y1, x0:x1])
                        ms, ious, _ = ipred.predict(box=np.array([b[2], b[0], b[3], b[1]], np.float32), multimask_output=False)
                    o = vcam.paste(ms[0] > 0, box, hw)
                    r.update(used_local=True, iou_T=float(ious[0]), box=json.dumps(box))
            r["latency_s"] = time.perf_counter() - t0
            r.update(m55.jf(o, g), gt_empty=not g.any())
            rows.append(r)
            out.append(o)
        atomic_npz(d / f"{f.stem}.npz", T1=np.packbits(np.stack(out)))
        atomic_csv(pd.DataFrame(rows), d / f"{f.stem}.csv")
        print("T1", name, round(pd.DataFrame(rows).JF.mean(), 4), round(B.JF.mean(), 4), flush=True)
    status(completed="trackA tplr run")


# ----------------------------------------------------------------------------- Track D: DAVIS train logs
def dtrain():
    from rfeas.units import davis_units
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    heads = m6.Heads(pred)
    for u in davis_units("train")[0]:
        if vcam.crop_side(m6.gt(u, 0)) is None:
            continue
        for arm in ("B1", "A1"):
            d = OUT / "trackD" / "runs" / arm
            d.mkdir(parents=True, exist_ok=True)
            stem = u.name.replace("/", "_")
            if (d / f"{stem}.csv").exists():
                continue
            df, masks = m6.run_unit(pred, heads, "DAVIS_train", u, arm)
            atomic_npz(d / f"{stem}.npz", **{arm: np.packbits(np.stack(masks))})
            atomic_csv(df, d / f"{stem}.csv")
            print(arm, u.name, round(df.JF.mean(), 4), flush=True)
    heads.close()
    status(completed="trackD dtrain run")


def _pairs(root):
    """Per-frame joined B1 (global) / A1 (crop) logs + mask-level features. root: dir with B1/ and A1/."""
    rows = []
    for f in sorted((root / "A1").glob("*.csv")):
        L = pd.read_csv(f)
        G = pd.read_csv(root / "B1" / f.name).set_index("video_idx")
        units = m6._UNITS() if root == M6R else _TRAIN_UNITS()
        u = units[L.unit[0]]
        GM, LM = _masks("B1", u, root), _masks("A1", u, root)
        side = int(L.side.iloc[0])
        for i, r in L.iterrows():
            t = int(r.video_idx)
            gm, lm = GM[t - 1], LM[t - 1]
            box = json.loads(r.box)
            b = vcam.bbox(lm)
            rows.append({"set": r.set, "unit": r.unit, "video": r.video, "video_idx": t, "J_L": r.J, "J_G": G.J[t],
                         "JF_L": r.JF, "JF_G": G.JF[t], "iou_L": r.iou_pred, "iou_G": G.iou_pred[t],
                         "score_L": r.obj_score, "score_G": G.obj_score[t], "agree": j_measure(lm, gm),
                         "touch": bool(r.crop_touch), "area_L": int(lm.sum()), "area_G": int(gm.sum()), "side": side,
                         "box_ratio": ((b[1] - b[0]) * (b[3] - b[2]) / side ** 2) if b is not None else 0.0,
                         "gt_empty": bool(r.gt_empty)})
    return pd.DataFrame(rows)


_TU = {}


def _TRAIN_UNITS():
    if not _TU:
        from rfeas.units import davis_units
        _TU.update({u.name: u for u in davis_units("train")[0]})
    return _TU


FEATS = ["iou_L", "iou_G", "score_L", "score_G", "agree", "touch", "lareaL", "lareaG", "lside"]


def _X(P):
    P = P.assign(lareaL=np.log1p(P.area_L), lareaG=np.log1p(P.area_G), lside=np.log(P.side), touch=P.touch.astype(float))
    return P[FEATS].to_numpy(float)


def fit_logistic(X, y, l2=1e-2, iters=50):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, Z.T @ (p - y) + l2 * w)
    return {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist()}


def predict_logistic(m, X):
    Z = np.c_[np.ones(len(X)), (X - np.array(m["mu"])) / np.array(m["sd"])]
    return 1 / (1 + np.exp(-Z @ np.array(m["w"])))


def gates():
    tr = _pairs(OUT / "trackD" / "runs")
    tr = tr[~tr.gt_empty]
    fit = tr[tr.J_L != tr.J_G]
    model = fit_logistic(_X(fit), (fit.J_L > fit.J_G).astype(float).to_numpy())
    atomic_json({**model, "features": FEATS, "n_fit_frames": int(len(fit)), "n_fit_units": int(fit.unit.nunique()),
                 "n_fit_videos": int(fit.video.nunique())}, OUT / "trackD" / "calibrator.json")
    dv = _pairs(M6R)
    p = predict_logistic(model, _X(dv))
    rules = {"none": np.ones(len(dv), bool),
             "ctu": (dv.agree > 0.5) | (dv.area_G == 0),
             "self": (dv.score_L > 0) & (dv.area_L > 0) & ~dv.touch,
             "iou": dv.iou_L > dv.iou_G,
             "calibrated": p > 0.5}
    rules["ctu"] = rules["ctu"] & (dv.box_ratio <= 0.1) & (dv.area_L > 0)
    rules = {k: np.asarray(v, bool) for k, v in rules.items()}
    for k, v in rules.items():
        dv[f"JF_{k}"] = np.where(v, dv.JF_L, dv.JF_G)
        dv[f"use_{k}"] = v
    dv["p_cal"] = p
    dv["oracle"] = dv[["JF_L", "JF_G"]].max(axis=1)
    atomic_csv(dv, OUT / "trackD" / "gates_dev_frames.csv")
    # whole-unit means (units without crop are identical to B1 for every gate -> dJF 0)
    B = m6._runs("B1")
    sm = {r["unit"]: r["small"] for r in json.loads((m6.OUT / "preregistration.json").read_text())["rows"]}
    ub = B.groupby(["set", "unit", "video"]).JF.mean().rename("JF_B1").reset_index()
    U = ub.merge(dv.groupby("unit")[[c for c in dv if c.startswith("JF_")] + ["oracle"]].mean().reset_index(), on="unit", how="left")
    # frame 1..n-1 coverage of the replay equals the unit's frames (A1 logs every frame)
    for k in list(rules) + ["oracle"]:
        col = k if k == "oracle" else f"JF_{k}"
        U[f"d_{k}"] = (U[col] - U.JF_B1).fillna(0)
    U["small"] = U.unit.map(sm)
    atomic_csv(U, OUT / "trackD" / "gates_per_unit.csv")
    G = U[U.set.isin(GATE_SETS)]
    rep = {"fit": {"frames": int(len(fit)), "units": int(fit.unit.nunique()), "videos": int(fit.video.nunique()),
                   "frac_crop_better": float((fit.J_L > fit.J_G).mean())}}
    for k in list(rules) + ["oracle"]:
        g = G[G.small]
        rep[k] = {"small_dJF": float(g[f"d_{k}"].mean()), "CI": m6._boot(g, f"d_{k}"),
                  "nonsmall_dJF": float(G[~G.small][f"d_{k}"].mean()),
                  "all_dev_dJF": float(U[f"d_{k}"].mean())}
        if k in rules:
            rules[k] = np.asarray(rules[k], bool)
            rep[k]["adopt_rate"] = float(rules[k].mean())
            rep[k]["wrong_adopt"] = float(((rules[k]) & (dv.JF_L < dv.JF_G - 0.05)).mean())
            rep[k]["missed_helpful"] = float(((~rules[k]) & (dv.JF_L > dv.JF_G + 0.05)).mean())
    g = G[G.small].assign(diff=G[G.small].d_calibrated - G[G.small].d_self)
    rep["calibrated_minus_self"] = {"mean": float(g["diff"].mean()), "CI": m6._boot(g, "diff")}
    rep["success"] = bool(rep["calibrated_minus_self"]["mean"] >= 0.005 and rep["calibrated_minus_self"]["CI"][0] > 0)
    # scale-bias verification (D1): predicted IoU minus true J by arm and area bin, dev logs
    vis = dv[~dv.gt_empty]
    cal = []
    for arm, iou, J in (("global", "iou_G", "J_G"), ("crop", "iou_L", "J_L")):
        for lo, hi in ((0, 500), (500, 2000), (2000, 8000), (8000, 10 ** 9)):
            x = vis[(vis.area_G >= lo) & (vis.area_G < hi)]
            if len(x):
                cal.append({"arm": arm, "area_bin": f"{lo}-{hi}", "frames": int(len(x)), "units": int(x.unit.nunique()),
                            "mean_iou_pred": float(x[iou].mean()), "mean_J": float(x[J].mean()),
                            "bias": float((x[iou] - x[J]).mean()),
                            "spearman": float(x[[iou, J]].corr("spearman").iloc[0, 1])})
    rep["scale_bias"] = cal
    ub_ = vis.groupby("unit").apply(lambda x: float(((x.iou_L - x.J_L) - (x.iou_G - x.J_G)).mean()))
    rep["crop_minus_global_bias_per_unit"] = {"units": int(len(ub_)), "positive": int((ub_ > 0).sum()), "median": float(ub_.median())}
    atomic_json(rep, OUT / "trackD" / "gates_report.json")
    print(json.dumps(rep, indent=1, default=float))
    status(completed="trackD gates")


# ----------------------------------------------------------------------------- Track B: refinement head v2
class HeadV2(torch.nn.Module):
    def __init__(self, c=96):
        super().__init__()
        C = torch.nn.Conv2d
        layers, cin = [], 101
        for dil in (1, 2, 4, 8):
            layers += [C(cin, c, 3, padding=dil, dilation=dil), torch.nn.ReLU()]
            cin = c
        self.low = torch.nn.Sequential(*layers, C(c, 1, 3, padding=1))
        self.high = torch.nn.Sequential(C(4, 16, 3, padding=1), torch.nn.ReLU(), C(16, 16, 3, padding=1), torch.nn.ReLU(),
                                        C(16, 1, 3, padding=1))
        for z in (self.low[-1], self.high[-1]):
            torch.nn.init.zeros_(z.weight)
            torch.nn.init.zeros_(z.bias)

    def forward(self, logit, s0, s1, rgb1024):
        up = Fn.interpolate(s1, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        rgb256 = Fn.interpolate(rgb1024, size=(256, 256), mode="bilinear", align_corners=False)
        l256 = logit + self.low(torch.cat([logit / 10, torch.sigmoid(logit), s0, up, rgb256], 1))
        l1024 = Fn.interpolate(l256, size=(1024, 1024), mode="bilinear", align_corners=False)
        return l1024 + self.high(torch.cat([l1024 / 10, rgb1024], 1))


def _rgb1024(im):
    return Fn.interpolate(torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)[None].float().cuda() / 255,
                          size=(1024, 1024), mode="bilinear", align_corners=False)


def _gt1024(g):
    return (Fn.interpolate(torch.from_numpy(g)[None, None].float().cuda(), size=(1024, 1024), mode="area") > 0.5).float()


def _loss(out, y):
    p = torch.sigmoid(out)
    band = (Fn.max_pool2d(y, 7, 1, 3) - (-Fn.max_pool2d(-y, 7, 1, 3))) > 0           # 3 px boundary band
    w = 1 + 4 * band.float()
    bce = (Fn.binary_cross_entropy_with_logits(out, y, reduction="none") * w).sum() / w.sum()
    return bce + 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)


def _samples(n_every=1):
    from rfeas.units import davis_units
    tr = {}
    for u in davis_units("train")[0]:
        tr.setdefault(u.video, []).append(u)
    return [(us, t) for us in tr.values() for t in range(0, len(us[0].frames), n_every)]


def _frozen_box(ipred, im, g, rng=None):
    ipred.set_image(im)
    b = vcam.bbox(g)
    box = np.array([b[2], b[0], b[3], b[1]], np.float32)
    if rng is not None:
        w, h = box[2] - box[0], box[3] - box[1]
        box = box + rng.uniform(-0.1, 0.1, 4).astype(np.float32) * np.array([w, h, w, h], np.float32)
    _, _, lr = ipred.predict(box=box, multimask_output=False, return_logits=True)
    s0, s1 = ipred._features["high_res_feats"]
    return torch.from_numpy(lr).cuda()[None], s0, s1


def _to_frame_1024(l1024, hw):
    return (Fn.interpolate(l1024, size=hw, mode="bilinear", align_corners=False)[0, 0] > 0).cpu().numpy()


def _train(head, items, minutes, steps=None, log_every=100, tag="train", rng_seed=0):
    from PIL import Image
    from rfeas.units import davis_gt
    ipred = m55.build_image_predictor()
    opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=1e-4)
    total = steps or 10 ** 9
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + np.cos(np.pi * min(k / max(total, 1), 1))) if steps else 1.0)
    rng = np.random.default_rng(rng_seed)
    t0, log, k = time.time(), [], 0
    while True:
        for i in rng.permutation(len(items)):
            if (steps and k >= steps) or time.time() - t0 > minutes * 60:
                return log
            us, t = items[i]
            im = np.asarray(Image.open(m55.frame_dir(us[0]) / f"{us[0].frames[t]}.jpg").convert("RGB"))
            gts = [g for g in (davis_gt(u, u.frames[t]) for u in us) if g.any()]
            if not gts:
                continue
            rgb = _rgb1024(im)
            loss = 0
            for g in gts:
                with torch.no_grad():
                    lg, s0, s1 = _frozen_box(ipred, im, g, rng if steps is None else None)
                loss = loss + _loss(head(lg, s0, s1, rgb), _gt1024(g))
            opt.zero_grad()
            (loss / len(gts)).backward()
            opt.step()
            sched.step()
            k += 1
            log.append({"step": k, "loss": float(loss.detach()) / len(gts), "t": time.time() - t0})
            if k % log_every == 0:
                print(tag, k, round(np.mean([x["loss"] for x in log[-log_every:]]), 4), f"{log[-1]['t']:.0f}s", flush=True)


def _eval_items(head, items):
    from PIL import Image
    from rfeas.units import davis_gt
    ipred = m55.build_image_predictor()
    rows = []
    for us, t in items:
        im = np.asarray(Image.open(m55.frame_dir(us[0]) / f"{us[0].frames[t]}.jpg").convert("RGB"))
        rgb = _rgb1024(im)
        for u in us:
            g = davis_gt(u, u.frames[t])
            if not g.any():
                continue
            with torch.no_grad():
                lg, s0, s1 = _frozen_box(ipred, im, g)
                a = _to_frame_1024(Fn.interpolate(lg, size=(1024, 1024), mode="bilinear", align_corners=False), g.shape)
                b = _to_frame_1024(head(lg, s0, s1, rgb), g.shape)
            rows.append({"unit": u.name, "t": t, "J_frozen": j_measure(a, g), "J_head": j_measure(b, g)})
    return pd.DataFrame(rows)


def head_pc():
    torch.manual_seed(0)
    items = _samples()
    rng = np.random.default_rng(1)
    pc = [items[i] for i in rng.choice(len(items), 32, replace=False)]
    head = HeadV2().cuda()
    log = _train(head, pc, minutes=12, steps=400, log_every=50, tag="pc")
    E = _eval_items(head, pc)
    rep = {"steps": len(log), "minutes": log[-1]["t"] / 60, "loss_first50": float(np.mean([x["loss"] for x in log[:50]])),
           "loss_last50": float(np.mean([x["loss"] for x in log[-50:]])), "J_frozen": float(E.J_frozen.mean()),
           "J_head": float(E.J_head.mean()), "objects": int(len(E))}
    rep["pass"] = rep["J_head"] - rep["J_frozen"] >= 0.05
    (OUT / "trackB").mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame(log), OUT / "trackB" / "pc_trainlog.csv")
    atomic_json(rep, OUT / "trackB" / "positive_control.json")
    print(json.dumps(rep, indent=1))
    status(completed=f"trackB positive control pass={rep['pass']}")


PREREG_PC2 = {
    "created_after": "head_pc failed (train J 0.809 -> 0.814 after 400 steps); diagnostic of the optimization, not of the task",
    "setup": "the same 32 DAVIS-train frames (75 objects), frozen SAM 2 outputs cached once; batch = all objects of one frame",
    "variants": {"V1": "HeadV2 as pre-registered, AdamW 1e-3 constant, 3000 steps",
                 "V2": "V1 + fixed feature standardization (s0 / 0.084, s1 / 0.177; constants measured on one frame)",
                 "V3": "V2 with lr 3e-4"},
    "pass": "train J(head) - J(frozen) >= +0.05 on the 75 objects (same rule as head_pc); 20 min wall cap in total",
    "next": "if a variant passes: main training with that variant (30 min) and the round-1 evaluation protocol unchanged; "
            "else Track B = technically inconclusive",
}


class HeadV2N(HeadV2):
    def forward(self, logit, s0, s1, rgb1024):
        return super().forward(logit, s0 / 0.084, s1 / 0.177, rgb1024)


def head_pc2():
    from PIL import Image
    from rfeas.units import davis_gt
    m55._freeze(OUT / "trackB" / "preregistration_pc2.json", {**PREREG_PC2, "rows": []})
    items = _samples()
    rng = np.random.default_rng(1)
    pc = [items[i] for i in rng.choice(len(items), 32, replace=False)]
    ipred = m55.build_image_predictor()
    cache = []
    for us, t in pc:
        im = np.asarray(Image.open(m55.frame_dir(us[0]) / f"{us[0].frames[t]}.jpg").convert("RGB"))
        objs = []
        for u in us:
            g = davis_gt(u, u.frames[t])
            if g.any():
                with torch.no_grad():
                    lg, s0, s1 = _frozen_box(ipred, im, g)
                objs.append((lg, g, _gt1024(g)))
        if objs:
            cache.append((s0.clone(), s1.clone(), _rgb1024(im), objs))
    del ipred
    rep = {}
    for name, cls, lr in (("V1", HeadV2, 1e-3), ("V2", HeadV2N, 1e-3), ("V3", HeadV2N, 3e-4)):
        torch.manual_seed(0)
        head = cls().cuda()
        opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-4)
        t0, log = time.time(), []
        for k in range(3000):
            if time.time() - t0 > 6 * 60:
                break
            s0, s1, rgb, objs = cache[k % len(cache)]
            loss = sum(_loss(head(lg, s0, s1, rgb), y) for lg, _, y in objs) / len(objs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            log.append(float(loss.detach()))
        jf, jh = [], []
        with torch.no_grad():
            for s0, s1, rgb, objs in cache:
                for lg, g, _ in objs:
                    jf.append(j_measure(_to_frame_1024(Fn.interpolate(lg, size=(1024, 1024), mode="bilinear", align_corners=False), g.shape), g))
                    jh.append(j_measure(_to_frame_1024(head(lg, s0, s1, rgb), g.shape), g))
        rep[name] = {"steps": len(log), "minutes": (time.time() - t0) / 60, "loss_first100": float(np.mean(log[:100])),
                     "loss_last100": float(np.mean(log[-100:])), "J_frozen": float(np.mean(jf)), "J_head": float(np.mean(jh)),
                     "pass": float(np.mean(jh) - np.mean(jf)) >= 0.05}
        print(name, json.dumps(rep[name]), flush=True)
    atomic_json(rep, OUT / "trackB" / "positive_control2.json")
    status(completed=f"trackB pc2 {[(k, v['pass']) for k, v in rep.items()]}")


def head_train():
    torch.manual_seed(0)
    head = HeadV2N().cuda()                                  # V2: winner of the pre-registered pc2 diagnostic
    log = _train(head, _samples(), minutes=30, tag="train")
    (OUT / "trackB").mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), OUT / "trackB" / "head_v2.pt")
    atomic_csv(pd.DataFrame(log), OUT / "trackB" / "trainlog.csv")
    status(completed=f"trackB head_train steps={len(log)}")


def _load_head():
    h = HeadV2N().cuda()
    h.load_state_dict(torch.load(OUT / "trackB" / "head_v2.pt"))
    return h.eval()


def head_image():
    """RETROSPECTIVE image-level eval (GT box prompt) on the M6 round-4 frame set."""
    from PIL import Image
    head = _load_head()
    ipred = m55.build_image_predictor()
    small = m6._small()
    rows = []
    for s, u in m6.units():
        sm = u.name in small
        if s == "SegTrack" and not sm:
            continue
        ps = m6.paths(u)
        for t in range(0, len(u.frames), 5 if sm else 10):
            g = m6.gt(u, t)
            if not g.any():
                continue
            im = np.asarray(Image.open(ps[t]).convert("RGB"))
            with torch.no_grad():
                lg, s0, s1 = _frozen_box(ipred, im, g)
                a = _to_frame_1024(Fn.interpolate(lg, size=(1024, 1024), mode="bilinear", align_corners=False), g.shape)
                b = _to_frame_1024(head(lg, s0, s1, _rgb1024(im)), g.shape)
            rows.append({"set": s, "unit": u.name, "video": u.video, "small": sm, "video_idx": t,
                         **{f"{k}_frozen": v for k, v in m55.jf(a, g).items()}, **{f"{k}_head": v for k, v in m55.jf(b, g).items()}})
    E = pd.DataFrame(rows)
    atomic_csv(E, OUT / "trackB" / "image_eval_frames.csv")
    E["dJ"], E["dJF"] = E.J_head - E.J_frozen, E.JF_head - E.JF_frozen
    U = E.groupby(["set", "unit", "video", "small"])[["dJ", "dJF"]].mean().reset_index()
    rep = {k: {"units": int(len(g)), "dJ": float(g.dJ.mean()), "dJF": float(g.dJF.mean()), "CI_dJF": m6._boot(g, "dJF"),
               "frac_pos": float((g.dJF > 0).mean())} for k, g in (("small", U[U.small]), ("ordinary", U[~U.small]))}
    atomic_json(rep, OUT / "trackB" / "image_eval.json")
    print(json.dumps(rep, indent=1))
    status(completed="trackB head_image")


def head_video(sets="DAVIS_val,PUMaVOS_B,PUMaVOS_A,SegTrack"):
    """B1 + head applied to the decoder output of every frame (output-only; memory untouched)."""
    from PIL import Image
    head = _load_head()
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    d = OUT / "trackB" / "video"
    d.mkdir(parents=True, exist_ok=True)
    for s, u in m6.units(tuple(sets.split(","))):
        stem = u.name.replace("/", "_")
        if (d / f"{stem}.csv").exists():
            continue
        state, od = m56.start(pred, u, p5=True)
        ps, rows = m6.paths(u), []
        for t in range(1, len(u.frames)):
            t0 = time.perf_counter()
            out, m, _, score = engine.predict(pred, state, od, t)
            od["non_cond_frame_outputs"][t] = out
            g = m6.gt(u, t)
            with torch.no_grad():
                feats = state["cached_features"][t][1]["backbone_fpn"]
                im = np.asarray(Image.open(ps[t]).convert("RGB"))
                l1024 = head(out["pred_masks"].to("cuda").float(), feats[0], feats[1], _rgb1024(im))
                lg = out["pred_masks"].to("cuda").float()
                pm = _to_frame_1024(Fn.interpolate(lg, size=(1024, 1024), mode="bilinear", align_corners=False), g.shape)
                hm = m if score <= 0 else _to_frame_1024(l1024, g.shape)   # SAM 2 says absent: keep its output
            torch.cuda.synchronize()
            rows.append({"set": s, "unit": u.name, "video": u.video, "video_idx": t, "obj_score": score,
                         "latency_s": time.perf_counter() - t0, "gt_empty": not g.any(),
                         **{f"{k}_B1": v for k, v in m55.jf(m, g).items()}, **{f"{k}_head": v for k, v in m55.jf(hm, g).items()},
                         "J_path1024": j_measure(m if score <= 0 else pm, g)})
        pred.reset_state(state)
        df = pd.DataFrame(rows).assign(peak_vram_bytes=torch.cuda.max_memory_allocated())
        atomic_csv(df, d / f"{stem}.csv")
        print(u.name, round(df.JF_B1.mean(), 4), round(df.JF_head.mean(), 4), flush=True)
    controlled.uninstall(pred)
    status(completed=f"trackB head_video {sets}")


# ----------------------------------------------------------------------------- Track A (round 2): memory write-back
PREREG_R2 = {
    "created_after": "round 1: TPLR rejected (no new information), calibrated gate no gain over self gate (selection "
                     "saturated: self gate reaches ~89% of the A2o selection oracle), head v2 image-level negative. "
                     "Frozen before any A3 output.",
    "hypothesis": "selection among fixed global/local candidates is saturated; the global tracker's own memory keeps "
                  "propagating its coarse small-target masks. Writing the accepted local (crop) mask back into the global "
                  "state's memory (exact: the mask is already in full-frame coordinates, re-encoded like P5) makes the "
                  "global candidate itself better on later frames, i.e. it creates NEW candidates.",
    "arms": {"A2d": "live dual-state A2x (G and L run together, no write-back); must reproduce the logged A2x exactly",
             "A3": "A2d + when the crop is accepted at t, re-encode G's memory for frame t from the fused full-res mask "
                   "(detail.to_model -> +-10 logits, G's object score)",
             "A3s": "A3 + when the crop is rejected and G's mask is non-empty, re-encode L's memory for frame t from G's "
                    "mask cropped to the current crop box (exact crop+resize, G's object score)"},
    "units": "the 21 dev units where A1/A2x crop (same as M6)",
    "success": "gate-set small-unit dJF (vs B1, 15 small units, uncropped = 0) >= A2x + 0.01 AND no cropped unit below "
               "-0.01 vs A2x AND retrospective per-frame oracle over {G', L'} exceeds the oracle over {B1, A1} by >= 0.01",
    "secondary": "live G' alone vs B1 (does write-back improve the global tracker itself?), frames 21+, latency",
}


def prereg_r2():
    m55._freeze(OUT / "preregistration_r2.json", {**PREREG_R2, "rows": []})


def run_dual(pred, s, u, arm, n=None):
    from spatial import detail
    n, hw = n or len(u.frames), m6.gt(u, 0).shape
    side = vcam.crop_side(m6.gt(u, 0))
    stG, odG = m56.start(pred, u, p5=True)
    center = vcam.center_of(m6.gt(u, 0))
    stL, odL, frames = m6.start_crop(pred, u, side, center)
    rows, masks = [], []
    for t in range(1, n):
        frames.box = box = vcam.place(center, side, hw)
        stL["video_height"] = stL["video_width"] = side
        t0 = time.perf_counter()
        oG, mG, _, sG = engine.predict(pred, stG, odG, t)
        odG["non_cond_frame_outputs"][t] = oG
        oL, mL, _, sL = engine.predict(pred, stL, odL, t)
        odL["non_cond_frame_outputs"][t] = oL
        touch = vcam.touches_border(mL, box, hw)
        ok = sL > 0 and mL.any() and not touch
        if arm == "A2v":                                   # cross-view presence veto (round 5)
            ok = ok and sG > 0
        out = vcam.paste(mL, box, hw) if ok else mG
        wrote = ""
        with torch.inference_mode():
            if arm in ("A3", "A3s") and ok:
                hi = detail.to_model(out).to("cuda") * 20.0 - 10.0
                oG["maskmem_features"], oG["maskmem_pos_enc"] = pred._run_memory_encoder(stG, t, 1, hi, oG["object_score_logits"], True)
                wrote = "G"
            if arm == "A3s" and not ok and mG.any():
                y0, y1, x0, x1 = box
                hi = detail.to_model(mG[y0:y1, x0:x1]).to("cuda") * 20.0 - 10.0
                oL["maskmem_features"], oL["maskmem_pos_enc"] = pred._run_memory_encoder(stL, t, 1, hi, oG["object_score_logits"], True)
                wrote = "L"
        torch.cuda.synchronize()
        g = m6.gt(u, t)
        rows.append({"set": s, "unit": u.name, "video": u.video, "arm": arm, "video_idx": t, "source": "crop" if ok else "global",
                     "wrote": wrote, "score_G": sG, "score_L": sL, "latency_s": time.perf_counter() - t0, "gt_empty": not g.any(),
                     **m55.jf(out, g), "JF_G": m55.jf(mG, g)["JF"], "JF_L": m55.jf(vcam.paste(mL, box, hw), g)["JF"],
                     "J_G": j_measure(mG, g), "J_L": j_measure(vcam.paste(mL, box, hw), g)})
        masks.append(out)
        c = vcam.center_of(out)
        center = c if c is not None else center
    peak = torch.cuda.max_memory_allocated()
    pred.reset_state(stG)
    pred.reset_state(stL)
    return pd.DataFrame(rows).assign(peak_vram_bytes=peak), masks


def dual(arms="A2d,A3,A3s"):
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    for arm in arms.split(","):
        d = OUT / "trackA" / arm
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted((M6R / "A2x").glob("*.csv")):
            if (d / f.name).exists():
                continue
            u = m6._UNITS()[pd.read_csv(f, nrows=1).unit[0]]
            s = pd.read_csv(f, nrows=1).set[0]
            torch.cuda.reset_peak_memory_stats()
            df, masks = run_dual(pred, s, u, arm)
            atomic_npz(d / f"{f.stem}.npz", **{arm: np.packbits(np.stack(masks))})
            atomic_csv(df, d / f.name)
            print(arm, u.name, round(df.JF.mean(), 4), round(df.JF_G.mean(), 4), flush=True)
    status(completed=f"trackA dual {arms}")


# ----------------------------------------------------------------------------- round 3: cost-matched generic control
PREREG_R3 = {
    "created_after": "round 2: A3 +0.0022 [+0.0003, +0.006] vs A2x on small units (below its +0.01 bar); no challenger "
                     "has beaten A2x. Frozen before any F2 output.",
    "question": "is A2x's gain specific to its global-local mechanism, or would any 2x-compute ensemble give it? "
                "F2 is the cost-matched generic control the incumbent must beat.",
    "F2": "two independent full-frame P5 states: original video and horizontally flipped video (lazy flip of the official "
          "frame tensor, flipped prompt); output = mean of the two sigmoid probability maps (flipped one flipped back "
          "exactly) > 0.5. Output-only, no memory exchange. Applied to every unit of DAVIS_val + PUMaVOS_B (67).",
    "decision_rules": {"incumbent_specific": "A2x small-unit dJF (gate set, 15) exceeds F2's by >= 0.005 -> the targeted "
                                             "mechanism matters; else A2x's gain is not specific",
                       "F2_as_challenger": "F2 overall dJF >= A2x overall + 0.003 with video CI of the paired difference "
                                           "> 0 (F2 is not novel; it would only change the baseline)"},
}


def prereg_r3():
    m55._freeze(OUT / "preregistration_r3.json", {**PREREG_R3, "rows": []})


class LazyFlip(vcam.LazyCropFrames):
    def __getitem__(self, t):
        return torch.flip(super().__getitem__(t), dims=[-1])


def run_flip(pred, s, u, n=None):
    n, hw = n or len(u.frames), m6.gt(u, 0).shape
    st, od = m56.start(pred, u, p5=True)
    fr = LazyFlip(m6.paths(u))
    fr.box = (0, hw[0], 0, hw[1])
    with vcam.lazy_loader(fr, hw):
        sf, odf = m56.start_dir(pred, "lazy", m6.gt(u, 0)[:, ::-1].copy(), p5=True)
    rows, masks = [], []
    for t in range(1, n):
        t0 = time.perf_counter()
        o1, m1, p1, s1 = engine.predict(pred, st, od, t)
        od["non_cond_frame_outputs"][t] = o1
        o2, m2, p2, s2 = engine.predict(pred, sf, odf, t)
        odf["non_cond_frame_outputs"][t] = o2
        out = (p1 + p2[:, ::-1]) / 2 > 0.5
        torch.cuda.synchronize()
        g = m6.gt(u, t)
        rows.append({"set": s, "unit": u.name, "video": u.video, "arm": "F2", "video_idx": t, "latency_s": time.perf_counter() - t0,
                     "gt_empty": not g.any(), **m55.jf(out, g), "JF_orig": m55.jf(m1, g)["JF"], "JF_flip": m55.jf(m2[:, ::-1], g)["JF"]})
        masks.append(out)
    peak = torch.cuda.max_memory_allocated()
    pred.reset_state(st)
    pred.reset_state(sf)
    return pd.DataFrame(rows).assign(peak_vram_bytes=peak), masks


def flip2(sets="DAVIS_val,PUMaVOS_B"):
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    d = OUT / "flip2"
    d.mkdir(parents=True, exist_ok=True)
    for s, u in m6.units(tuple(sets.split(","))):
        stem = u.name.replace("/", "_")
        if (d / f"{stem}.csv").exists():
            continue
        torch.cuda.reset_peak_memory_stats()
        df, masks = run_flip(pred, s, u)
        atomic_csv(df, d / f"{stem}.csv")
        print(u.name, round(df.JF.mean(), 4), round(df.JF_orig.mean(), 4), flush=True)
    status(completed=f"F2 flip control {sets}")


PREREG_R4 = {
    "created_after": "round 3: F2 (cost-matched flip ensemble) +0.001 overall; A2x - F2 +0.018 small [+0.009, +0.026]; A3 - A2x "
                     "+0.0022 [+0.0002, +0.0054]. Frozen before any round-4 output.",
    "question": "do A2x and A3 replicate on more independent videos? DAVIS 2017 *train* units where the A1 rule crops "
                "(43 objects, 23 videos): never used to design A2x/A3 (only to fit the rejected Track-D calibrator, whose "
                "B1/A1 logs are reused as the B1 reference). Caveat: DAVIS train may overlap SAM 2's own training data.",
    "arms": "A2d (= A2x) and A3 via run_dual, unchanged code/parameters",
    "report": "unit-mean dJF vs B1 for all 43 objects and the small (<1%) subset; video-clustered CI; A3 - A2d paired",
    "replicates_if": "A2d small-subset dJF > 0 with video-CI lower bound > 0 and no more than 10% of objects below -0.01; "
                     "A3 - A2d >= 0 (CI reported). Not a confirmation set; C stays closed.",
}


def prereg_r4():
    m55._freeze(OUT / "preregistration_r4.json", {**PREREG_R4, "rows": []})


def replicate(arms="A2d,A3"):
    from rfeas.units import davis_units
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    for arm in arms.split(","):
        d = OUT / "replication" / arm
        d.mkdir(parents=True, exist_ok=True)
        for u in davis_units("train")[0]:
            if vcam.crop_side(m6.gt(u, 0)) is None:
                continue
            stem = u.name.replace("/", "_")
            if (d / f"{stem}.csv").exists():
                continue
            torch.cuda.reset_peak_memory_stats()
            df, masks = run_dual(pred, "DAVIS_train", u, arm)
            atomic_csv(df, d / f"{stem}.csv")
            print(arm, u.name, round(df.JF.mean(), 4), flush=True)
    status(completed=f"replication {arms}")


PREREG_R5 = {
    "created_after": "round-4 replication on DAVIS train: A2x small +0.009 [-0.019, +0.032] (22 videos; 17 up / 2 down by "
                     "video); one catastrophic unit miami-surf obj5 -0.477: after a true absence the LOCAL tracker re-acquires "
                     "a wrong object with high confidence (score_L up to 7.8) while the GLOBAL tracker correctly reports absence "
                     "(score_G < 0, J_G = 1.0) and the self gate adopts the crop for ~35 frames. EXPLORATORY: the failure "
                     "was found on the replication set, so results on that set are not independent.",
    "arm": "A2v = A2x self gate AND global object score > 0 (cross-view presence veto); everything else unchanged",
    "evaluation": {"dev_gate_set": "21 dev crop units (as A2x); primary: non-inferiority, small-unit dJF (15, uncropped=0) "
                                   ">= A2x - 0.003",
                   "replication_set": "43 DAVIS-train crop units: miami-surf obj5 dJF >= -0.05 and small-unit mean >= A2x "
                                      "(descriptive, not independent)"},
    "interpretation": "if both hold, cross-view presence consistency is a measured reliability component of the global-local "
                      "direction (still exploratory); if dev non-inferiority fails, the veto removes A2x's benefit",
}


def prereg_r5():
    m55._freeze(OUT / "preregistration_r5.json", {**PREREG_R5, "rows": []})


def veto(which="dev,train"):
    from rfeas.units import davis_units
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    for w in which.split(","):
        d = OUT / ("trackA" if w == "dev" else "replication") / "A2v"
        d.mkdir(parents=True, exist_ok=True)
        if w == "dev":
            todo = [(pd.read_csv(f, nrows=1).set[0], m6._UNITS()[pd.read_csv(f, nrows=1).unit[0]]) for f in sorted((M6R / "A2x").glob("*.csv"))]
        else:
            todo = [("DAVIS_train", u) for u in davis_units("train")[0] if vcam.crop_side(m6.gt(u, 0)) is not None]
        for s, u in todo:
            stem = u.name.replace("/", "_")
            if (d / f"{stem}.csv").exists():
                continue
            torch.cuda.reset_peak_memory_stats()
            df, masks = run_dual(pred, s, u, "A2v")
            atomic_csv(df, d / f"{stem}.csv")
            print("A2v", u.name, round(df.JF.mean(), 4), flush=True)
    status(completed=f"A2v veto {which}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("arg", nargs="?")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fn = globals()[a.what]
    fn(a.arg) if a.arg else fn()


if __name__ == "__main__":
    main()
