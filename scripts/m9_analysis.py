"""M9 analysis (CPU). Never writes into outputs/milestone8.

    uv run python scripts/m9_analysis.py stage1 <dataset>          # minimal-generator recall / FP / localisation / paired tests
    uv run python scripts/m9_analysis.py taxonomy <dataset>        # failure taxonomy of the machine-matched SGGL trajectory
    uv run python scripts/m9_analysis.py causal <arm> [<arm>...]   # arm vs machine-matched SGGL, per dataset + pooled

Data locations (diag rows = SGGL trajectory = machine-matched baseline; FLOWVEL etc. = causal runs):
  RunPod: outputs/milestone9/runpod/diag/<dataset>/, outputs/milestone9/runpod/runs/<arm>/
  Ubuntu: outputs/milestone9/diag/<dataset>/,        outputs/milestone9/runs/<arm>/
Taxonomy needs GT bounding boxes; they are precomputed where the data lives (scripts/m9_gtbox.py) into
outputs/milestone9/gtbox/<dataset>.csv.
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rfeas.risk import cluster_bootstrap  # noqa: E402

OUT = Path("outputs/milestone9")
MACH = {"runpod": OUT / "runpod", "ubuntu": OUT}
GOOD = 0.5


def _frames(d):
    return [f for f in sorted(d.glob("*.csv")) if not f.name.endswith((".cand.csv", ".cand1b.csv", ".diag1b.csv"))]


def diag_dirs(ds):
    return {m: p / "diag" / ds for m, p in MACH.items() if (p / "diag" / ds).exists()}


def load_diag(ds):
    F, C = [], []
    dirs = diag_dirs(ds)
    for m, p in MACH.items():                            # datasets without a diag run: plain SGGL arm rows
        if m not in dirs and (p / "runs" / "SGGL").exists() and list((p / "runs" / "SGGL").glob(f"{ds}__*.csv")):
            for f in sorted((p / "runs" / "SGGL").glob(f"{ds}__*.csv")):
                fr = pd.read_csv(f).assign(machine=m, stem=f.stem)
                fr["trig"] = np.where(fr.trig.astype(bool), "coabs", None)
                F.append(fr)
    for m, d in dirs.items():
        for f in _frames(d):
            fr = pd.read_csv(f).assign(machine=m, stem=f.stem)
            F.append(fr)
            cf = d / f"{f.stem}.cand.csv"
            if cf.exists() and cf.stat().st_size > 1:
                c = pd.read_csv(cf)
                if len(c):
                    C.append(c.assign(machine=m, stem=f.stem))
    F = pd.concat(F, ignore_index=True)
    F["gt_empty"] = F.gt_empty.astype(bool)
    C = pd.concat(C, ignore_index=True) if C else pd.DataFrame()
    if len(C):
        C["fam"] = C.gen.str.split("#").str[0]
        C["present"] = (C.score > 0) & (C.area > 0)
        C["gt_empty"] = C.gt_empty.astype(bool)
    return F, C


# ----------------------------------------------------------------------------- stage 1
def stage1(ds):
    F, C = load_diag(ds)
    co = F[F.trig == "coabs"]
    rep = {"dataset": ds, "n_units": int(F.unit.nunique()), "n_videos": int(F.video.nunique()), "frames": int(len(F)),
           "trigger_counts": F.trig.value_counts().to_dict(), "coabs_gt_present": int((~co.gt_empty).sum()),
           "coabs_gt_empty": int(co.gt_empty.sum()), "gt_empty_frames": int(F.gt_empty.sum()),
           "coabs_gt_present_units": int(co[~co.gt_empty].unit.nunique()),
           "coabs_gt_present_videos": int(co[~co.gt_empty].video.nunique()),
           "latency_s": {"sggl": float(F.lat_sggl.mean()), "flow": float(F.lat_flow.mean()),
                         "rescue_per_trigger": float(F.lat_rescue.mean()) if "lat_rescue" in F else None}}
    rows, tests = [], {}
    if len(C):
        X = C[C.trig == "coabs"]
        g = X.groupby(["unit", "video", "video_idx", "gt_empty", "fam"])
        T = g.agg(best_J=("J", "max"), any_present=("present", "any")).reset_index()
        T["hit"] = (~T.gt_empty) & (T.best_J > GOOD)
        T["fp"] = T.gt_empty & T.any_present
        P, A = T[~T.gt_empty], T[T.gt_empty]
        for f, d in P.groupby("fam"):
            a = A[A.fam == f]
            rows.append({"gen": f, "recall": d.hit.mean(), "mean_best_J": d.best_J.mean(),
                         "videos_hit": int(d[d.hit].video.nunique()), "units_hit": int(d[d.hit].unit.nunique()),
                         "fp_rate_gt_empty": a.fp.mean() if len(a) else np.nan, "n_present": len(d), "n_empty": len(a)})
        U = P.groupby(["unit", "video", "video_idx"]).best_J.max()
        rep["oracle_any_generator_recall"] = float((U > GOOD).mean()) if len(U) else None
        rep["oracle_any_generator_videos"] = int(U[U > GOOD].reset_index().video.nunique()) if len(U) else 0
        T1 = X.assign(fam=X.gen).groupby(["unit", "video", "video_idx", "gt_empty", "fam"]).J.max().reset_index()
        T1["hit"] = (~T1.gt_empty) & (T1.J > GOOD)

        def paired(a, b, frame_set):
            x = frame_set[frame_set.fam == a].set_index(["unit", "video", "video_idx"]).hit.astype(float)
            y = frame_set[frame_set.fam == b].set_index(["unit", "video", "video_idx"]).hit.astype(float)
            dd = (x - y).dropna().reset_index(name="d")
            if not len(dd):
                return None
            return {"delta": float(dd.d.mean()), "CI": cluster_bootstrap(dd, "video", lambda q: q.d.mean(), n=2000),
                    "n_frames": len(dd), "n_videos": int(dd.video.nunique())}
        P1 = T1[~T1.gt_empty]
        tests["B_att_vs_A_cos_recall_top3"] = paired("B_att", "A_cos", P)
        tests["B_att#0_vs_A_cos#0_hit1"] = paired("B_att#0", "A_cos#0", P1)
        tests["C_flowcrop_vs_Lp"] = paired("C_flowcrop", "Lp", P)
        tests["F_oclr8_L_vs_Lp"] = paired("F_oclr8_L", "Lp", P)
        # train-free score>0 rule per generator (replay, J-only, frame level): rescued vs false-positive frames
        rule = []
        for f, Y in X.groupby("fam"):
            b = Y.loc[Y.groupby(["unit", "video_idx"]).score.idxmax()]
            acc = b.score > 0
            pres = ~b.gt_empty
            rule.append({"gen": f, "accepted": int(acc.sum()), "rescued": int((acc & pres & (b.J > GOOD)).sum()),
                         "fp": int((acc & ~pres).sum()),
                         "net_dJ_frames": float(np.where(acc & pres, b.J, 0).sum() - (acc & ~pres).sum()),
                         "videos_rescued": int(b[acc & pres & (b.J > GOOD)].video.nunique())})
        rep["score_rule_replay"] = rule
    if "loc1_A_cos" in F:
        Pf = F[(F.trig == "coabs") & ~F.gt_empty]
        rep["localisation_coabs"] = {k: {"loc1": float(Pf[f"loc1_{k}"].mean()), "loc3": float(Pf[f"loc3_{k}"].mean())}
                                     for k in ("A_cos", "B_att") if f"loc1_{k}" in Pf}
    rep["generators"] = rows
    rep["paired_tests"] = tests
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / f"stage1_{ds}.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("generators", "score_rule_replay")}, indent=1, default=float))
    if rows:
        print(pd.DataFrame(rows).sort_values("recall", ascending=False).round(3).to_string(index=False))
        print(pd.DataFrame(rep["score_rule_replay"]).sort_values("net_dJ_frames", ascending=False).to_string(index=False))


# ----------------------------------------------------------------------------- taxonomy
def classify(r):
    """Pre-registered order (preregistration.json failure_taxonomy)."""
    if r.area_ratio < 0.25 or r.gt_area < 64:
        return "OBSERVABILITY"
    if r.gt_long_over_side > 0.6 or r.area_ratio > 4:
        return "SCALE"
    if r.gt_in_crop < 0.5:
        return "POSITION"
    return "PRESENCE_ABSENT_CLAIM" if r.out_empty else "PRESENCE_WRONG_TARGET"


def taxonomy(ds):
    F, _ = load_diag(ds)
    box = pd.read_csv(OUT / "gtbox" / f"{ds}.csv")          # unit, video_idx, gt_long, prompt_area, crop_side
    X = F.merge(box, on=["unit", "video_idx"], how="left")
    X["area_ratio"] = X.gt_area / X.prompt_area
    X["gt_long_over_side"] = X.gt_long / X.crop_side
    X["out_empty"] = ~X.acc.astype(bool) & (X.score_G <= 0)    # SGGL output empty: L rejected by the gate and G absent
    X["loss"] = 1 - X.JF
    miss = X[~X.gt_empty & (X.J < GOOD)].copy()
    miss["cat"] = miss.apply(classify, axis=1)
    miss["coabs"] = miss.trig == "coabs"
    tot = miss.loss.sum()
    cats = []
    for c, d in miss.groupby("cat"):
        cats.append({"category": c, "frames": len(d), "loss": float(d.loss.sum()), "loss_share": float(d.loss.sum() / tot),
                     "units": int(d.unit.nunique()), "videos": int(d.video.nunique()),
                     "coabs_frames": int(d.coabs.sum()),
                     "top_video_share": float(d.groupby("video").loss.sum().max() / d.loss.sum())})
    cats = sorted(cats, key=lambda c: -c["loss"])
    dom = next((c["category"] for c in cats if c["loss_share"] >= 0.5 and c["videos"] >= 3), "MIXED")
    small = X[X.area0 < 0.01]
    rep = {"dataset": ds, "frames": int(len(X)), "gt_present_frames": int((~X.gt_empty).sum()),
           "missed_frames": int(len(miss)), "missed_loss": float(tot), "categories": cats, "dominant": dom,
           "false_reacq_frames": int((X.gt_empty & (X.J < 1)).sum()), "gt_empty_frames": int(X.gt_empty.sum()),
           "sggl_small_unit_JF_mean": float(small.groupby("unit").JF.mean().mean())}
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / f"taxonomy_{ds}.json").write_text(json.dumps(rep, indent=1, default=float))
    miss.to_csv(OUT / "reports" / f"taxonomy_frames_{ds}.csv", index=False)
    print(json.dumps(rep, indent=1, default=float))


# ----------------------------------------------------------------------------- causal
def _unit_deltas(arm):
    rows = []
    for m, p in MACH.items():
        rd = p / "runs" / arm
        if not rd.exists():
            continue
        for f in sorted(rd.glob("*.csv")):
            ds = f.stem.split("__")[0]
            bd = p / "diag" / ds / f.name                     # SGGL trajectory: diag rows, else the plain SGGL arm
            if not bd.exists():
                bd = p / "runs" / "SGGL" / f.name
            if not bd.exists() or arm == "SGGL":
                continue
            a, b = pd.read_csv(f), pd.read_csv(bd)
            b = b.assign(coabs_b=(b.score_G <= 0) & (b.score_L <= 0))
            mm = a.merge(b[["video_idx", "J", "F", "JF", "coabs_b"]], on="video_idx", suffixes=("", "_b"))
            assert len(mm) == len(a) == len(b), (arm, f.name)
            co_b = mm.coabs_b.to_numpy()
            rows.append({"dataset": ds, "machine": m, "unit": a.unit.iloc[0], "video": a.video.iloc[0],
                         "set": a.set.iloc[0], "area0": a.area0.iloc[0],
                         "small": a.area0.iloc[0] < 0.01, "tiny": a.area0.iloc[0] < 0.001,
                         "d": a.JF.mean() - mm.JF_b.mean(), "dJ": a.J.mean() - mm.J_b.mean(), "dF": a.F.mean() - mm.F_b.mean(),
                         "base_JF": mm.JF_b.mean(), "arm_JF": a.JF.mean(),
                         "coabs_frames_base": int(co_b.sum()),
                         "coabs_dsum": float((mm.JF - mm.JF_b).to_numpy()[co_b].sum()),
                         "fp_arm": int((a.gt_empty.astype(bool) & (a.J < 1)).sum()),
                         "fp_base": int((a.gt_empty.astype(bool) & (mm.J_b < 1)).sum()),
                         "lat": a.latency_s.mean(), "vram": a.peak_vram_bytes.max()})
    return pd.DataFrame(rows)


def _summ(S, label):
    vid = S.groupby("video").d.mean()
    lovo = [S[S.video != v].d.mean() for v in vid.index] if len(vid) > 1 else [np.nan]
    pos, neg = int((vid > 0).sum()), int((vid < 0).sum())
    n = pos + neg
    p_sign = min(1.0, 2 * sum(comb(n, k) for k in range(max(pos, neg), n + 1)) / 2 ** n) if n else 1.0
    gain = S.d.clip(lower=0).groupby(S.video).sum().sort_values(ascending=False)
    top2 = float(S[S.video.isin(vid.sort_values(ascending=False).index[:2])].d.sum() / S.d.sum()) if S.d.sum() > 0 else np.nan
    return {"label": label, "n_units": int(len(S)), "n_videos": int(S.video.nunique()), "dJF": float(S.d.mean()),
            "dJ": float(S.dJ.mean()), "dF": float(S.dF.mean()),
            "CI_video": cluster_bootstrap(S, "video", lambda q: q.d.mean(), n=2000) if S.video.nunique() > 1 else [np.nan, np.nan],
            "videos_up_0.01": int((vid > 0.01).sum()), "videos_down_0.01": int((vid < -0.01).sum()),
            "videos_pos_neg": [pos, neg], "sign_test_p": p_sign, "lovo_min": float(np.nanmin(lovo)),
            "top2_share_of_net_gain": top2, "worst_unit": [S.loc[S.d.idxmin(), "unit"], float(S.d.min())],
            "best_unit": [S.loc[S.d.idxmax(), "unit"], float(S.d.max())], "n_units_lt_-0.05": int((S.d < -0.05).sum()),
            "tiny_dJF": float(S[S.tiny].d.mean()) if S.tiny.any() else None, "n_tiny": int(S.tiny.sum()),
            "base_JF": float(S.base_JF.mean()), "coabs_frames_base": int(S.coabs_frames_base.sum()),
            "coabs_dJF_per_frame": float(S.coabs_dsum.sum() / max(S.coabs_frames_base.sum(), 1)),
            "false_reacq_arm_vs_base": [int(S.fp_arm.sum()), int(S.fp_base.sum())],
            "latency_s": float(S.lat.mean()), "peak_vram_gb": float(S.vram.max() / 1e9)}


def causal(*arms):
    out = {}
    for arm in arms:
        U = _unit_deltas(arm)
        if not len(U):
            print(arm, "no units yet")
            continue
        U.to_csv(OUT / "reports" / f"causal_units_{arm}.csv", index=False)
        S = U[U.small]
        res = {"pooled": _summ(S, "pooled")}
        for ds, d in S.groupby("dataset"):
            res[ds] = _summ(d, ds)
            for sp, e in d.groupby("set"):
                res[f"{ds}:{sp}"] = _summ(e, f"{ds}:{sp}")
        # Holm across per-dataset bootstrap tests: one-sided p from the bootstrap sign of the mean
        out[arm] = res
        for k, r in res.items():
            print(f"{arm:10s} {k:16s} n={r['n_units']:3d}/{r['n_videos']:3d}v dJF={r['dJF']:+.4f} CI=[{r['CI_video'][0]:+.4f},{r['CI_video'][1]:+.4f}] "
                  f"up/dn={r['videos_up_0.01']}/{r['videos_down_0.01']} sign_p={r['sign_test_p']:.3f} lovo={r['lovo_min']:+.4f} "
                  f"top2={r['top2_share_of_net_gain'] if r['top2_share_of_net_gain'] == r['top2_share_of_net_gain'] else float('nan'):.2f} "
                  f"worst={r['worst_unit'][1]:+.3f} <-0.05:{r['n_units_lt_-0.05']} FP={r['false_reacq_arm_vs_base']}")
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    prev = json.loads((OUT / "reports" / "causal.json").read_text()) if (OUT / "reports" / "causal.json").exists() else {}
    prev.update(out)
    (OUT / "reports" / "causal.json").write_text(json.dumps(prev, indent=1, default=float))


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
