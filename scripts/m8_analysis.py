"""M8 analysis (CPU only; reads outputs/milestone8, never runs SAM 2).

    uv run python scripts/m8_analysis.py stage1      # generator recall / FP / localisation + paired falsification tests
    uv run python scripts/m8_analysis.py rankers     # fit R2 (logistic) and R3 (set attention) on REP, evaluate on DEV
    uv run python scripts/m8_analysis.py causal      # every finished causal arm vs machine-matched SGGL (diag baseline)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rfeas.risk import cluster_bootstrap  # noqa: E402
from spatial import ranker  # noqa: E402

torch.set_num_threads(2)      # leave the CPU to the GPU workers' data path

OUT = Path("outputs/milestone8")
DIAG = OUT / "diag"
RANKERS = OUT / "rankers"
GOOD, BAD = 0.5, 0.1          # candidate J thresholds: hit / clear miss


def fam(g):
    return g.split("#")[0]


DIAGS = (DIAG, OUT / "runpod" / "diag")      # Ubuntu 2080 units + units recorded on the RunPod worker


def load():
    F, C, V = [], [], []
    for f in sorted(p for d in DIAGS for p in d.glob("*.csv")):
        DIAG = f.parent
        if f.name.endswith(".cand.csv"):
            continue
        st = f.name[:-4]
        cf = DIAG / f"{st}.cand.csv"
        if not cf.exists():
            continue
        fr = pd.read_csv(f)
        b1 = DIAG / f"{st}.cand1b.csv"
        redo = b1.exists()                                   # amendment 2: H_s8 of this unit comes from stage 1b
        if redo:
            h = pd.read_csv(DIAG / f"{st}.diag1b.csv")
            fr = fr.drop(columns=[c for c in fr.columns if c.endswith("_H_s8")]).merge(h, on=["unit", "video_idx"], how="left")
        F.append(fr)
        for cfile, vfile in ((cf, DIAG / f"{st}.vec.npy"), (b1, DIAG / f"{st}.vec1b.npy")):
            if not cfile.exists() or cfile.stat().st_size <= 1:
                continue
            c = pd.read_csv(cfile)
            if len(c):
                v = np.load(vfile).astype(np.float32)
                if redo and cfile == cf:
                    keep = ~c.gen.str.startswith("H_s8").to_numpy()
                    c, v = c[keep], v[keep]
                c["stem"] = st
                C.append(c)
                V.append(v)
    F = pd.concat(F, ignore_index=True)
    C = pd.concat(C, ignore_index=True)
    C["fam"] = C.gen.map(fam)
    C["split"] = C.unit.map(F.drop_duplicates("unit").set_index("unit").set)
    C["small"] = C.unit.map(F.drop_duplicates("unit").set_index("unit").area0) < 0.01
    C["present"] = (C.score > 0) & (C.area > 0)
    return F, C, np.concatenate(V)


def per_frame(C, trig="coabs"):
    """Frame x family table: hit (any candidate J > GOOD), fp (any present candidate on GT-empty frame), best J."""
    X = C[C.trig == trig]
    g = X.groupby(["unit", "video", "video_idx", "gt_empty", "fam"])
    T = g.agg(best_J=("J", "max"), any_present=("present", "any"), top_J=("J", "first")).reset_index()
    T["hit"] = (~T.gt_empty) & (T.best_J > GOOD)
    T["fp"] = T.gt_empty & T.any_present
    return T


def boot_mean(df, col, n=2000):
    return cluster_bootstrap(df, "video", lambda d: d[col].mean(), n=n)


def stage1():
    F, C, _ = load()
    rep = {"n_units": int(F.unit.nunique()), "n_videos": int(F.video.nunique()), "frames": int(len(F)),
           "trigger_counts": F.trig.value_counts().to_dict(),
           "coabs_gt_present": int(((F.trig == "coabs") & ~F.gt_empty).sum()),
           "coabs_gt_empty": int(((F.trig == "coabs") & F.gt_empty).sum()),
           "latency_s": {"sggl_per_frame": float(F.lat_sggl.mean()), "flow_per_frame": float(F.lat_flow.mean()),
                         "rescue_all_generators_per_trigger": float(F.lat_rescue.mean())},
           "peak_vram_gb": float(F.peak_vram_bytes.max() / 1e9)}
    T = per_frame(C)
    P, A = T[~T.gt_empty], T[T.gt_empty]
    rows = []
    for f, d in P.groupby("fam"):
        a = A[A.fam == f]
        rows.append({"gen": f, "recall": d.hit.mean(), "mean_best_J": d.best_J.mean(), "top_J": d.top_J.mean(),
                     "videos_hit": int(d[d.hit].video.nunique()), "units_hit": int(d[d.hit].unit.nunique()),
                     "fp_rate_gt_empty": a.fp.mean() if len(a) else np.nan, "n_present": len(d), "n_empty": len(a)})
    G = pd.DataFrame(rows).sort_values("recall", ascending=False)
    ref = G[G.gen.isin(["Gp", "Lp"])].recall.max()
    G["adv_recall"] = (G.recall >= 0.25) & (G.recall >= 2 * ref) & (G.videos_hit >= 4)
    rep["generators_coabs"] = G.round(4).to_dict("records")
    rep["reference_recall_max_Gp_Lp"] = float(ref)
    # union of everything (oracle over all generators) and per-family oracle headroom (J-only, frames of coabs)
    U = P.groupby(["unit", "video", "video_idx"]).agg(best=("best_J", "max")).reset_index()
    rep["oracle_any_generator_recall"] = float((U.best > GOOD).mean())
    # heatmap localisation on coabs GT-present frames
    Pf = F[(F.trig == "coabs") & ~F.gt_empty]
    rep["localisation_coabs"] = {k: {"loc1": float(Pf[f"loc1_{k}"].mean()), "loc3": float(Pf[f"loc3_{k}"].mean())}
                                 for k in ("A_cos", "B_att", "H_s8")}
    # paired falsification tests (video-clustered)
    tests = {}
    # single-candidate generators are compared with the single top-1 candidate (B_att#0), not the top-3 family
    T1 = per_frame(C.assign(fam=C.gen))
    P1 = T1[~T1.gt_empty]

    def paired(a, b, metric="hit", frame_set=P):
        fs = P1 if "#" in a or "#" in b else frame_set
        x = fs[fs.fam == a].set_index(["unit", "video", "video_idx"])[metric].astype(float)
        y = fs[fs.fam == b].set_index(["unit", "video", "video_idx"])[metric].astype(float)
        d = (x - y).dropna().reset_index(name="d")
        return {"delta": float(d.d.mean()), "CI": boot_mean(d, "d"), "n_frames": len(d), "n_videos": int(d.video.nunique())}
    tests["B_att_vs_A_cos_recall"] = paired("B_att", "A_cos")
    tests["H_s8_vs_A_cos_recall"] = paired("H_s8", "A_cos")
    for st in ("G", "L"):
        tests[f"cpfr8_vs_oclr8_{st}_recall"] = paired(f"F_cpfr8_{st}", f"F_oclr8_{st}")
        tests[f"cpfr8_vs_oclr8_{st}_bestJ"] = paired(f"F_cpfr8_{st}", f"F_oclr8_{st}", "best_J")
    tests["cpfr4_vs_oclr4_G_recall"] = paired("F_cpfr4_G", "F_oclr4_G")
    tests["G_neg_vs_B_att#0_recall"] = paired("G_neg", "B_att#0")
    tests["D_ref_vs_B_att#0_recall"] = paired("D_ref", "B_att#0")
    for st in ("G", "L"):                                     # amendment 1: prompted F variants = primary F test
        tests[f"cpfr8pt_vs_oclr8pt_{st}_recall"] = paired(f"F_cpfr8_{st}_pt", f"F_oclr8_{st}_pt")
    tests["cpfr4pt_vs_oclr4pt_G_recall"] = paired("F_cpfr4_G_pt", "F_oclr4_G_pt")
    tests["C_flowcrop_vs_Lp_recall"] = paired("C_flowcrop", "Lp")
    tests["E_beam1_vs_Lp_recall"] = paired("E_beam1", "Lp")
    for k in ("A_cos", "B_att", "H_s8"):
        pass
    d = Pf.assign(d=Pf.loc1_B_att.astype(float) - Pf.loc1_A_cos.astype(float))
    tests["B_att_vs_A_cos_loc1"] = {"delta": float(d.d.mean()), "CI": boot_mean(d, "d")}
    d = Pf.assign(d=Pf.loc1_H_s8.astype(float) - Pf.loc1_A_cos.astype(float))
    tests["H_s8_vs_A_cos_loc1"] = {"delta": float(d.d.mean()), "CI": boot_mean(d, "d")}
    rep["paired_tests"] = tests
    # per-video hits for the best generators
    rep["hits_by_video"] = P[P.hit].groupby(["fam", "video"]).size().unstack(fill_value=0).to_dict("index")
    # disagreement-trigger and regular frames: identity of the candidates (top J) — ranking difficulty context
    for trig in ("disagree", "regular"):
        Tt = per_frame(C, trig)
        rep[f"{trig}_recall"] = Tt[~Tt.gt_empty].groupby("fam").hit.mean().round(3).to_dict()
    (OUT / "stage1_report.json").write_text(json.dumps(rep, indent=1, default=float))
    G.to_csv(OUT / "stage1_generators.csv", index=False)
    print(json.dumps({k: rep[k] for k in ("n_units", "trigger_counts", "coabs_gt_present", "coabs_gt_empty", "latency_s",
                                          "oracle_any_generator_recall", "localisation_coabs")}, indent=1, default=float))
    print(G.round(3).to_string(index=False))
    print(json.dumps(tests, indent=1, default=float))


# ----------------------------------------------------------------------------- rankers
def _ranker_data(C, V, gens):
    keep = C.fam.isin(gens) & C.trig.isin(["coabs", "disagree", "regular"])
    X = C[keep].copy()
    X["y"] = np.where(X.J > GOOD, 1, np.where((X.J < BAD) | X.gt_empty, 0, -1))
    return X, V[keep.to_numpy()]


def _accept_eval(X, s, thr):
    """Frame-level outcome of 'output the best-scored candidate if its score > thr' on coabs frames."""
    X = X.assign(s=s)
    X = X[X.trig == "coabs"]
    best = X.loc[X.groupby(["unit", "video_idx"]).s.idxmax()]
    acc = best.s > thr
    pres = ~best.gt_empty
    return {"frames_present": int(pres.sum()), "frames_empty": int((~pres).sum()),
            "rescue_rate_present": float((acc & pres & (best.J > GOOD)).sum() / max(pres.sum(), 1)),
            "wrong_rescue_present": float((acc & pres & (best.J <= GOOD)).sum() / max(pres.sum(), 1)),
            "fp_rate_empty": float((acc & ~pres).sum() / max((~pres).sum(), 1)),
            "mean_J_gain_present": float(np.where(acc & pres, best.J, 0).sum() / max(pres.sum(), 1)),
            "videos_rescued": int(best[acc & pres & (best.J > GOOD)].video.nunique()),
            # frame-level J change vs SGGL's empty output: +J on GT-present frames, -1 on GT-empty frames (J(empty,empty)=1)
            "net_dJ_frames": float(np.where(acc & pres, best.J, 0).sum() - (acc & ~pres).sum())}


def _thr_best_net(X, s, fp=0.10):
    """Threshold maximising net_dJ_frames on the fit set subject to fp_rate_empty <= fp (candidates: observed scores)."""
    X = X.assign(s=s)
    X = X[X.trig == "coabs"]
    best = X.loc[X.groupby(["unit", "video_idx"]).s.idxmax()]
    top = _thr_for_fp(X, X.s.to_numpy(), fp)
    cands = np.unique(np.concatenate([best.s.to_numpy(), [top]]))
    cands = cands[np.isfinite(cands) & (cands >= top)]
    vals = [(_accept_eval(X, X.s.to_numpy(), c)["net_dJ_frames"], c) for c in cands]
    return max(vals)[1] if vals else top


def _thr_for_fp(X, s, fp=0.10):
    """Lowest threshold whose coabs GT-empty acceptance rate is <= fp (fit set)."""
    X = X.assign(s=s)
    X = X[X.trig == "coabs"]
    best = X.loc[X.groupby(["unit", "video_idx"]).s.idxmax()]
    e = np.sort(best[best.gt_empty].s.to_numpy())[::-1]
    k = int(np.floor(fp * len(e)))
    return float(e[k]) if k < len(e) else float(e[-1] - 1e-6)


def rankers(gens=None, name=None):
    """Cross-fitted rankers: direction 'REP->DEV' (pre-registered) and 'DEV->REP' (added so causal arms can be evaluated
    out-of-sample on all of DEV+REP). Each direction: fit on one split, threshold on that split (max net_dJ_frames s.t.
    fp_rate_empty <= 0.10; R3 uses out-of-fold scores), apply to the other split."""
    F, C, V = load()
    gens = (gens.split(",") if isinstance(gens, str) and gens else gens) or sorted(C.fam.unique())
    X, XV = _ranker_data(C, V, gens)
    RANKERS.mkdir(parents=True, exist_ok=True)
    tag = name or ("_".join(sorted(gens)) if len(gens) < 6 else f"pool{len(gens)}")
    res = {"gens": gens, "fit_data": "coabs+disagree+regular candidates of the fit split",
           "threshold": "max net_dJ_frames on the fit split's coabs frames s.t. fp_rate_empty <= 0.10 (R3: out-of-fold)"}
    for fs, es in (("REP", "DEV"), ("DEV", "REP")):
        fit, ev = X.split == fs, X.split == es
        S = {"R0_score": X.score.to_numpy(), "R1_trainfree": ranker.r1(X)}
        lab = fit & (X.y >= 0)
        lr = ranker.LR().fit(ranker.scalars(X[lab]), X.y[lab].to_numpy())
        S["R2_logistic"] = lr(ranker.scalars(X))
        (RANKERS / f"R2_{tag}_fit{fs}.json").write_text(json.dumps(lr.state()))
        S["R3_set"] = _fit_set(X, XV, fit, gens, f"{tag}_fit{fs}")
        for k, sc in S.items():
            thr = _thr_best_net(X[fit], sc[fit.to_numpy()])
            res.setdefault(k, {})[f"fit{fs}"] = {"thr": thr, "fit_eval": _accept_eval(X[fit], sc[fit.to_numpy()], thr),
                                                  "heldout_eval": _accept_eval(X[ev], sc[ev.to_numpy()], thr)}
    for k in res:
        if k.startswith("R"):
            h = [res[k][f"fit{f}"]["heldout_eval"] for f in ("REP", "DEV")]
            res[k]["pooled_heldout"] = {m: h[0][m] + h[1][m] for m in ("net_dJ_frames", "frames_present", "frames_empty")}
            res[k]["pooled_heldout"]["rescued_frames"] = sum(x["rescue_rate_present"] * x["frames_present"] for x in h)
            res[k]["pooled_heldout"]["fp_frames"] = sum(x["fp_rate_empty"] * x["frames_empty"] for x in h)
            print(tag, k, {a: round(b, 2) for a, b in res[k]["pooled_heldout"].items()},
                  "| REP->DEV", {a: round(b, 3) for a, b in res[k]["fitREP"]["heldout_eval"].items() if a in ("rescue_rate_present", "fp_rate_empty", "net_dJ_frames")})
    (OUT / f"rankers_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def _fit_set(X, XV, fit, gens, tag, steps=1500, seed=0):
    """R3: one training example per (unit, frame); target = argmax-J candidate if J > GOOD else 'none'."""
    torch.manual_seed(seed)
    fams = sorted(set(gens))
    Z = ranker.scalars(X)
    mu, sd = Z[fit.to_numpy()].mean(0), Z[fit.to_numpy()].std(0) + 1e-6
    feats = np.concatenate([XV, (Z - mu) / sd], 1).astype(np.float32)
    refs = {}
    for st in X.stem.unique():
        r = torch.load(DIAG / f"{st}.refs.pt")
        refs[st] = (torch.cat([r["fg"], r["bg"]]).float(), torch.tensor([0] * len(r["fg"]) + [1] * len(r["bg"])))
    groups = list(X.groupby(["stem", "video_idx"]).indices.items())
    fam_id = np.array([fams.index(f) for f in X.fam])
    Jall = X.J.to_numpy()
    feats_t = torch.from_numpy(feats)
    def batch(idx_groups):
        K = max(len(ix) for _, ix in idx_groups)
        R_ = max(len(refs[k[0]][0]) for k, _ in idx_groups)
        B = len(idx_groups)
        cand = torch.zeros(B, K, feats.shape[1])
        gid = torch.zeros(B, K, dtype=torch.long)
        pad = torch.ones(B, K, dtype=torch.bool)
        ref = torch.zeros(B, R_, 256)
        rt = torch.ones(B, R_, dtype=torch.long)
        y = torch.zeros(B, dtype=torch.long)
        for b, ((st, _), ix) in enumerate(idx_groups):
            cand[b, :len(ix)] = feats_t[ix]
            gid[b, :len(ix)] = torch.from_numpy(fam_id[ix])
            pad[b, :len(ix)] = False
            rr, tt = refs[st]
            ref[b, :len(rr)], rt[b, :len(rr)] = rr, tt
            J = Jall[ix]
            y[b] = int(J.argmax()) if J.max() > GOOD else K               # K = 'none' slot after padding
        return cand, gid, ref, rt, pad, y
    def train(gs):
        torch.manual_seed(seed)
        net = ranker.SetRanker(len(fams))
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        rng = np.random.default_rng(seed)
        for step in range(steps):
            pick = [gs[i] for i in rng.choice(len(gs), min(32, len(gs)), replace=False)]
            cand, gid, ref, rt, pad, y = batch(pick)
            loss = torch.nn.functional.cross_entropy(net(cand, gid, ref, rt, pad), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return net.eval()

    def score(net, gs, s):
        with torch.no_grad():
            for i in range(0, len(gs), 64):
                chunk = gs[i:i + 64]
                cand, gid, ref, rt, pad, _ = batch(chunk)
                lo = net(cand, gid, ref, rt, pad)
                for b, (_, ix) in enumerate(chunk):
                    s[ix] = (lo[b, :len(ix)] - lo[b, -1]).numpy()

    fit_groups = [g for g in groups if fit.iloc[g[1][0]]]
    ev_groups = [g for g in groups if not fit.iloc[g[1][0]]]
    s = np.zeros(len(X), np.float32)
    # REP rows get out-of-fold scores (2 video-disjoint folds) so the REP-fitted threshold is not in-sample
    vids = sorted(X.video[fit].unique())
    fold = {v: i % 2 for i, v in enumerate(vids)}
    # DEV rows get the MEAN of the two fold models (same score scale as the out-of-fold REP scores the threshold is
    # fitted on; a refit on all REP would shift the scale)
    sd_ev = np.zeros(len(X), np.float32)
    states = []
    for k in (0, 1):
        tr = [g for g in fit_groups if fold[X.video.iloc[g[1][0]]] != k]
        te = [g for g in fit_groups if fold[X.video.iloc[g[1][0]]] == k]
        net = train(tr)
        score(net, te, s)
        tmp = np.zeros(len(X), np.float32)
        score(net, ev_groups, tmp)
        sd_ev += tmp / 2
        states.append(net.state_dict())
    ev_ix = np.concatenate([ix for _, ix in ev_groups]) if ev_groups else np.zeros(0, int)
    s[ev_ix] = sd_ev[ev_ix]
    torch.save({"states": states, "gens": fams, "mu": mu, "sd": sd, "steps": steps, "seed": seed,
                "folds": fold}, RANKERS / f"R3_{tag}.pt")
    return s




# ----------------------------------------------------------------------------- causal arms
def _base_frames():
    """Machine-matched SGGL per unit: RunPod `runs/SGGL` for pod arms, the Ubuntu diag trajectory for Ubuntu arms."""
    pod = {f.stem: pd.read_csv(f) for f in (OUT / "runpod/runs/SGGL").glob("*.csv")}
    ubu = {f.stem: pd.read_csv(f) for f in DIAG.glob("*.csv") if not f.name.endswith((".cand.csv", ".cand1b.csv", ".diag1b.csv"))}
    return {"runpod": pod, "ubuntu": ubu}


def _pilot(U, label):
    S = U[U.small].copy()
    vid = S.groupby("video").d.mean()
    lovo = [S[S.video != v].d.mean() for v in vid.index]
    r = {"arm": label, "n_units": int(len(S)), "n_videos": int(S.video.nunique()), "dJF": float(S.d.mean()),
         "CI_video": cluster_bootstrap(S, "video", lambda x: x.d.mean(), n=2000),
         "dJ": float(S.dJ.mean()), "dF": float(S.dF.mean()),
         "videos_up_0.01": int((vid > 0.01).sum()), "videos_down_0.01": int((vid < -0.01).sum()),
         "lovo_min": float(min(lovo)), "without_miami": float(S[S.video != "davis/miami-surf"].d.mean()),
         "worst_unit": [S.loc[S.d.idxmin(), "unit"], float(S.d.min())], "best_unit": [S.loc[S.d.idxmax(), "unit"], float(S.d.max())],
         "tiny_dJF": float(S[S.tiny].d.mean()) if S.tiny.any() else None, "n_tiny": int(S.tiny.sum()),
         "all_units_dJF": float(U.d.mean()),
         "coabs_frames": int(U.coabs_n.sum()), "coabs_dJF_framesum": float(U.coabs_dsum.sum()),
         "false_reacq_frames": int(U.fp_arm.sum()), "false_reacq_frames_SGGL": int(U.fp_base.sum()),
         "rescued_frames": int(U.rescued.sum()),
         "latency_s_mean": float(U.lat.mean()), "latency_s_SGGL": float(U.lat_base.mean()) if U.lat_base.notna().any() else None,
         "extra_sam2_calls_per_frame": float(U.calls.mean() - 2), "peak_vram_gb": float(U.vram.max() / 1e9),
         "per_video": vid.round(4).to_dict()}
    adv = r["dJF"] >= 0.005 and r["videos_up_0.01"] >= 3 and r["lovo_min"] > 0 and r["without_miami"] > 0 and r["worst_unit"][1] >= -0.05
    rej = r["dJF"] <= 0 or (r["videos_up_0.01"] <= 2 and r["lovo_min"] <= 0)
    r["verdict"] = "PROMISING" if adv else ("REJECTED" if rej else "INCONCLUSIVE")
    if r["dJF"] >= 0.03 and r["CI_video"][0] > 0:
        r["verdict"] = "STRONG_WIN"
    return r


def causal(*names):
    base = _base_frames()
    out = {}
    dirs = {"runpod": OUT / "runpod/runs", "ubuntu": OUT / "runs"}
    arms = sorted({p.name for d in dirs.values() if d.exists() for p in d.iterdir() if p.is_dir() and p.name != "SGGL"})
    rows_all = []
    for arm in (names or arms):
        rows = []
        for mach, d in dirs.items():
            for f in sorted((d / arm).glob("*.csv")) if (d / arm).exists() else []:
                b = base[mach].get(f.stem)
                if b is None:
                    continue
                a = pd.read_csv(f)
                m = a.merge(b[["video_idx", "J", "F", "JF"] + (["latency_s"] if "latency_s" in b else [])],
                            on="video_idx", suffixes=("", "_b"))
                assert len(m) == len(a) == len(b), (arm, f.stem)
                co = (a.score_G <= 0) & (a.score_L <= 0)
                rows.append({"unit": a.unit.iloc[0], "video": a.video.iloc[0], "machine": mach, "set": a.set.iloc[0],
                             "area0": a.area0.iloc[0], "small": a.area0.iloc[0] < 0.01, "tiny": a.area0.iloc[0] < 0.001,
                             "d": a.JF.mean() - m.JF_b.mean(), "dJ": a.J.mean() - m.J_b.mean(), "dF": a.F.mean() - m.F_b.mean(),
                             "coabs_n": int(co.sum()), "coabs_dsum": float((m.JF - m.JF_b)[co.to_numpy()].sum()),
                             "fp_arm": int((a.gt_empty & (a.J < 1)).sum()), "fp_base": int((a.gt_empty & (m.J_b < 1)).sum()),
                             "rescued": int(a.get("rescued", pd.Series(False)).sum()), "lat": a.latency_s.mean(),
                             "lat_base": m.latency_s_b.mean() if "latency_s_b" in m else np.nan,
                             "calls": a.calls.mean() if "calls" in a else 2.0, "vram": a.peak_vram_bytes.max()})
        if not rows:
            continue
        U = pd.DataFrame(rows)
        V = U[U.set == "VK"]
        U = U[U.set != "VK"]
        if len(V):                                          # VKITTI finalists: descriptive, object-clustered
            V = V.assign(scene=V.video)
            out[f"{arm}@VK"] = {"n_units": int(len(V)), "dJF": float(V.d.mean()),
                                "CI_object": cluster_bootstrap(V.assign(o=V.unit), "o", lambda x: x.d.mean(), n=2000),
                                "per_scene": V.groupby("scene").d.mean().round(4).to_dict(),
                                "units_up_0.01": int((V.d > 0.01).sum()), "units_down_0.01": int((V.d < -0.01).sum()),
                                "worst_unit": [V.loc[V.d.idxmin(), "unit"], float(V.d.min())],
                                "non_regression_pass": bool(V.d.mean() >= -0.005)}
            print(f"{arm}@VK dJF={V.d.mean():+.4f} per_scene={out[f'{arm}@VK']['per_scene']}")
        if not len(U):
            continue
        out[arm] = _pilot(U, arm)
        out[arm]["machines"] = U.machine.value_counts().to_dict()
        rows_all.append(U.assign(arm=arm))
        r = out[arm]
        print(f"{arm:20s} n={r['n_units']:2d} dJF={r['dJF']:+.4f} CI=[{r['CI_video'][0]:+.4f},{r['CI_video'][1]:+.4f}] "
              f"up/down={r['videos_up_0.01']}/{r['videos_down_0.01']} lovo={r['lovo_min']:+.4f} worst={r['worst_unit'][1]:+.3f} "
              f"rescued={r['rescued_frames']} FPreacq={r['false_reacq_frames']}vs{r['false_reacq_frames_SGGL']} {r['verdict']}")
    (OUT / "causal_report.json").write_text(json.dumps(out, indent=1, default=float))
    if rows_all:
        pd.concat(rows_all).to_csv(OUT / "causal_per_unit.csv", index=False)
    return out

def pairwise(*pairs):
    """Paired arm-vs-arm difference on small units, same machine (RunPod), video-clustered CI. pairs: 'A-B' strings."""
    U = pd.read_csv(OUT / "causal_per_unit.csv")
    U = U[U.small & (U.machine == "runpod")]
    res = {}
    for pr in pairs or ("FLOWCARRY-FLOWVEL", "CAMCARRY-FLOWVEL", "FLOWCARRY-CAMCARRY", "FLOWVEL-FV_PRESENT", "FLOWVEL-M3vel",
                        "K_OCLR_FLOWVEL-FLOWVEL", "K_ADMIT_FLOWVEL-FLOWVEL", "K_OCLR_M3-M3vel", "K_OCLR_M3-RESC_OCLR",
                        "RESC_OCLR_ADMIT-RESC_OCLR", "RESC_CPFR-RESC_OCLR"):
        a, b = pr.split("-", 1)
        m = U[U.arm == a][["unit", "video", "d"]].merge(U[U.arm == b][["unit", "d"]], on="unit", suffixes=("_a", "_b"))
        m["x"] = m.d_a - m.d_b
        res[pr] = {"delta": float(m.x.mean()), "CI": cluster_bootstrap(m, "video", lambda d: d.x.mean(), n=2000), "n": len(m)}
        print(f"{pr:28s} n={len(m)} delta={m.x.mean():+.4f} CI=[{res[pr]['CI'][0]:+.4f},{res[pr]['CI'][1]:+.4f}]")
    (OUT / "pairwise_report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
