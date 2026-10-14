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


def load():
    F, C, V = [], [], []
    for f in sorted(DIAG.glob("*.csv")):
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
    def paired(a, b, metric="hit", frame_set=P):
        x = frame_set[frame_set.fam == a].set_index(["unit", "video", "video_idx"])[metric].astype(float)
        y = frame_set[frame_set.fam == b].set_index(["unit", "video", "video_idx"])[metric].astype(float)
        d = (x - y).dropna().reset_index(name="d")
        return {"delta": float(d.d.mean()), "CI": boot_mean(d, "d"), "n_frames": len(d), "n_videos": int(d.video.nunique())}
    tests["B_att_vs_A_cos_recall"] = paired("B_att", "A_cos")
    tests["H_s8_vs_A_cos_recall"] = paired("H_s8", "A_cos")
    for st in ("G", "L"):
        tests[f"cpfr8_vs_oclr8_{st}_recall"] = paired(f"F_cpfr8_{st}", f"F_oclr8_{st}")
        tests[f"cpfr8_vs_oclr8_{st}_bestJ"] = paired(f"F_cpfr8_{st}", f"F_oclr8_{st}", "best_J")
    tests["cpfr4_vs_oclr4_G_recall"] = paired("F_cpfr4_G", "F_oclr4_G")
    tests["G_neg_vs_B_att#0_recall"] = paired("G_neg", "B_att")
    tests["D_ref_vs_B_att_recall"] = paired("D_ref", "B_att")
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


def rankers(gens=None):
    F, C, V = load()
    gens = (gens.split(",") if isinstance(gens, str) else gens) or sorted(C.fam.unique())
    X, XV = _ranker_data(C, V, gens)
    fit, ev = X.split == "REP", X.split == "DEV"
    RANKERS.mkdir(parents=True, exist_ok=True)
    res = {"gens": gens, "fit": "REP coabs+disagree+regular candidates", "eval": "DEV coabs frames",
           "threshold": "max net_dJ_frames on REP coabs subject to REP fp_rate_empty <= 0.10 (R3: out-of-fold REP scores)"}
    S = {}
    S["R0_score"] = X.score.to_numpy()
    S["R1_trainfree"] = ranker.r1(X)
    lab = fit & (X.y >= 0)
    lr = ranker.LR().fit(ranker.scalars(X[lab]), X.y[lab].to_numpy())
    S["R2_logistic"] = lr(ranker.scalars(X))
    tag = "_".join(sorted(gens)) if len(gens) < 6 else f"pool{len(gens)}"
    (RANKERS / f"R2_{tag}.json").write_text(json.dumps(lr.state()))
    S["R3_set"] = _fit_set(X, XV, fit, gens, tag)
    for k, s in S.items():
        thr = _thr_best_net(X[fit], s[fit.to_numpy()])
        res[k] = {"thr_fit_REP": thr, "REP": _accept_eval(X[fit], s[fit.to_numpy()], thr),
                  "DEV": _accept_eval(X[ev], s[ev.to_numpy()], thr)}
    (OUT / f"rankers_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    for k in S:
        print(k, "thr", round(res[k]["thr_fit_REP"], 3), "DEV", {a: round(b, 3) for a, b in res[k]["DEV"].items()})
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


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
