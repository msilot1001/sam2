"""M10 analysis (CPU). Reads outputs/milestone10/<machine>/runs/<arm>/*.csv, writes outputs/milestone10/reports/.

    uv run python scripts/m10_analysis.py table [arm ...]     # oracle table: per-dataset paired dJ&F vs OR0 (+ offline)
    uv run python scripts/m10_analysis.py states [arm ...]    # where (tracker state x scene category) each arm recovers
    uv run python scripts/m10_analysis.py episodes [arm ...]  # wrong-region / loss episodes: onset, duration, recovery
    uv run python scripts/m10_analysis.py parity              # OR0 vs M9 SGGL rows (same frames)

Primary estimand (preregistration.json): per dataset, video-equal mean of (arm - OR0) J&F; 95% video-cluster
bootstrap. All JSON is written with allow_nan=False (non-finite -> null).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/milestone10")
RUNS = Path(__import__("os").environ.get("M10_RUNS", OUT / "runpod" / "runs"))
REP = OUT / "reports"
TAU = 0.5
B = 10000
OFFLINE = ("OR1off", "OR5off", "OR6off")
DSETS = ("mosev2", "vost", "lvosv2")
B_ = 10000


def finite(o):
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    if isinstance(o, dict):
        return {str(k): finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def save(obj, name):
    REP.mkdir(parents=True, exist_ok=True)
    (REP / name).write_text(json.dumps(finite(obj), indent=1, allow_nan=False) + "\n")


def load(arm):
    fs = sorted((RUNS / arm).glob("*.csv"))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in fs if f.stat().st_size > 1], ignore_index=True)   # OR9: units w/o GT frames
    df["ds"] = df.set.str.split(":").str[0].str.lower()
    for c in ("gt_empty", "acc", "ok_G", "ok_L", "adm_G", "adm_L", "rew_G", "rew_L"):
        if c in df:
            df[c] = df[c].astype(bool)
    return df


def offline(b, kind):
    """Framewise hindsight oracles computed from OR0 rows (no rollout)."""
    o = b.copy()
    if kind == "OR1off":
        jf = np.where(o.gt_empty, 1.0, o.JF)
        j = np.where(o.gt_empty, 1.0, o.J)
    elif kind == "OR5off":
        jf, j = o.JF_best, o[[f"cJ{k}" for k in range(7)]].max(axis=1)
    else:                                                            # OR6off: perfect target-region detector
        alt_jf = np.where(o.acc, o.JF_G, o.JF_L)
        alt_j = np.where(o.acc, o.J_G, o.J_L)
        prec = np.where(o.out_area > 0, o.out_inter_gt / o.out_area.clip(lower=1), 1.0)
        ok = o.J >= TAU
        alt_ok = alt_j >= TAU
        wrong = ~ok & ~alt_ok & (prec < TAU)
        jf = np.select([ok, alt_ok, wrong], [o.JF, alt_jf, o.gt_empty.astype(float)], o.JF)
        j = np.select([ok, alt_ok, wrong], [o.J, alt_j, o.gt_empty.astype(float)], o.J)
    o["JF"], o["J"], o["arm"] = jf, j, kind
    o["F"] = 2 * o.JF - o.J
    return o


def paired(a, b):
    """Frame-aligned arm vs OR0 rows -> per-unit deltas (asserts identical frame sets)."""
    k = ["unit", "video_idx"]
    m = a[k + ["ds", "video", "JF", "J", "F", "area0"]].merge(b[k + ["JF", "J", "F"]], on=k, suffixes=("", "_0"))
    assert len(m) == len(a) == len(b[b.unit.isin(a.unit.unique())]), "frame sets differ"
    u = m.groupby(["ds", "video", "unit"]).agg(JF=("JF", "mean"), JF0=("JF_0", "mean"), J=("J", "mean"),
                                                J0=("J_0", "mean"), F=("F", "mean"), F0=("F_0", "mean"),
                                                area0=("area0", "first")).reset_index()
    u["d"], u["dJ"], u["dF"] = u.JF - u.JF0, u.J - u.J0, u.F - u.F0
    return u


def vstats(u, rng):
    """Video-equal estimand, video-cluster bootstrap, LOVO, top-2 share, sign counts."""
    v = u.groupby("video").agg(d=("d", "mean"), dJ=("dJ", "mean"), dF=("dF", "mean"), JF0=("JF0", "mean")).reset_index()
    x = v.d.to_numpy()
    n = len(x)
    bs = x[rng.integers(0, n, (B, n))].mean(1)
    lovo = [(x.sum() - xi) / (n - 1) for xi in x] if n > 1 else [np.nan]
    pos = np.sort(x[x > 0])[::-1]
    net = x.sum()
    tiny = u[u.area0 < 0.001]
    return {"n_units": len(u), "n_videos": n, "dJF": x.mean(), "ci": [np.percentile(bs, 2.5), np.percentile(bs, 97.5)],
            "dJ": v.dJ.mean(), "dF": v.dF.mean(), "dJF_unit_weighted": u.d.mean(),
            "dJF_tiny_units": tiny.d.mean() if len(tiny) else None, "n_tiny": len(tiny),
            "JF0": v.JF0.mean(), "recovered_frac": x.mean() / (1 - v.JF0.mean()),
            "lovo_min": min(lovo), "lovo_max": max(lovo),
            "top2_share": pos[:2].sum() / net if net > 0 else None,
            "videos_up": int((x > 0.01).sum()), "videos_down": int((x < -0.01).sum()),
            "worst_video": v.loc[v.d.idxmin(), "video"] if n else None, "worst_video_d": x.min() if n else None,
            "units_below_m005": int((u.d < -0.05).sum()), "units_above_005": int((u.d > 0.05).sum())}


def table(*arms):
    b = load("OR0")
    arms = arms or [p.name for p in sorted(RUNS.iterdir()) if p.is_dir() and p.name not in ("OR0", "OR9")] + list(OFFLINE)
    rng = np.random.default_rng(10)
    res = {"baseline": {ds: {"units": int(g.unit.nunique()), "videos": int(g.video.nunique()), "frames": len(g),
                             "JF_video_equal": g.groupby(["video", "unit"]).JF.mean().groupby("video").mean().mean()}
                        for ds, g in b.groupby("ds")}, "arms": {}}
    for arm in arms:
        a = offline(b, arm) if arm in OFFLINE else load(arm)
        if not len(a):
            continue
        r = {"kind": "offline-framewise" if arm in OFFLINE else "online-rollout", "per_dataset": {}}
        us = []
        for ds, g in a.groupby("ds"):
            u = paired(g, b[b.ds == ds])
            us.append(u)
            r["per_dataset"][ds] = {**vstats(u, rng), "complete": g.unit.nunique() == b[b.ds == ds].unit.nunique()}
        U = pd.concat(us)
        U["video"] = U.ds + "/" + U.video
        r["pooled_secondary"] = vstats(U, rng)
        if "adm_G" in a:
            r["admit_rate"] = {"G": a.adm_G.mean(), "L": a.adm_L.mean()}
            r["rewrite_rate"] = {"G": a.rew_G.mean(), "L": a.rew_L.mean()}
        r["latency_s_mean"] = a.latency_s.mean() if "latency_s" in a else None
        r["peak_vram_gb"] = a.peak_vram_bytes.max() / 2**30 if "peak_vram_bytes" in a else None
        res["arms"][arm] = r
        print(arm, {ds: (round(x["dJF"], 4), [round(c, 4) for c in x["ci"]], x["n_units"])
                    for ds, x in r["per_dataset"].items()}, flush=True)
    save(res, "oracle_table.json")


def state(o):
    """Tracker/output state of each OR0 frame (GT-based, offline)."""
    prec = np.where(o.out_area > 0, o.out_inter_gt / o.out_area.clip(lower=1), 1.0)
    oth = np.where(o.out_area > 0, o.out_inter_other / o.out_area.clip(lower=1), 0.0)
    return np.select([o.gt_empty & (o.out_area == 0), o.gt_empty, o.J >= TAU, o.out_area == 0,
                      (prec < TAU) & (oth >= TAU), prec < TAU],
                     ["absent_ok", "false_present", "correct", "empty", "wrong_object", "wrong_region"], "partial")


def scene(ds):
    """M9 per-frame scene/failure category of SGGL misses (identical trajectory: see parity)."""
    p = Path(f"outputs/milestone9/reports/taxonomy_frames_{ds}.csv")
    if not p.exists():
        return None
    t = pd.read_csv(p)
    t = t[t.machine == t.machine.iloc[0]] if "machine" in t else t
    return t[["unit", "video_idx", "cat"]].drop_duplicates(["unit", "video_idx"])


def states(*arms):
    b = load("OR0")
    b["state"] = state(b)
    b["loss0"] = 1 - b.JF
    arms = arms or [p.name for p in sorted(RUNS.iterdir()) if p.is_dir() and p.name not in ("OR0", "OR9")] + list(OFFLINE)
    res = {"OR0_loss_share": {}, "recovery": {}}
    for ds, g in b.groupby("ds"):
        tot = g.loss0.sum()
        res["OR0_loss_share"][ds] = {"frames": g.state.value_counts().to_dict(),
                                     "loss_share": (g.groupby("state").loss0.sum() / tot).to_dict()}
        sc = scene(ds)
        if sc is not None:
            m = g.merge(sc, on=["unit", "video_idx"], how="left")
            res["OR0_loss_share"][ds]["by_m9_category_x_state"] = (
                m[m.J < TAU].groupby(["cat", "state"]).loss0.sum() / tot).round(4).unstack(fill_value=0).to_dict()
    for arm in arms:
        a = offline(b, arm) if arm in OFFLINE else load(arm)
        if not len(a):
            continue
        res["recovery"][arm] = {}
        for ds, g in b.groupby("ds"):
            m = g[["unit", "video_idx", "state", "JF"]].merge(a[["unit", "video_idx", "JF"]], on=["unit", "video_idx"],
                                                               suffixes=("0", ""))
            if not len(m):
                continue
            m["gain"] = m.JF - m.JF0
            loss = (1 - m.JF0).groupby(m.state).sum()
            res["recovery"][arm][ds] = {"gain_by_state": m.groupby("state").gain.sum().round(2).to_dict(),
                                        "recovered_frac_by_state": (m.groupby("state").gain.sum() / loss).round(4).to_dict(),
                                        "frames": len(m)}
    save(res, "states.json")


def episodes(*arms):
    """Episodes = maximal runs of frames with GT present and J < 0.5 (loss) or output wrong-region (drift)."""
    def ep(g, flag):
        out = []
        for unit, x in g.groupby("unit"):
            f = flag[x.index].to_numpy()
            j = x.JF.to_numpy()
            i = 0
            while i < len(f):
                if f[i]:
                    k = i
                    while k < len(f) and f[k]:
                        k += 1
                    out.append({"unit": unit, "onset": int(x.video_idx.iloc[i]), "dur": k - i,
                                "recovered": k < len(f), "loss": float((1 - j[i:k]).sum())})
                    i = k
                else:
                    i += 1
        return pd.DataFrame(out)

    def stats(g):
        r = {}
        for name, fl in (("loss", (~g.gt_empty) & (g.J < TAU)), ("drift", g.state.isin(["wrong_region", "wrong_object"]))):
            e = ep(g, fl)
            r[name] = {"episodes": len(e), "per_unit": len(e) / g.unit.nunique(),
                       "mean_dur": e.dur.mean() if len(e) else None, "median_dur": e.dur.median() if len(e) else None,
                       "recovered_frac": e.recovered.mean() if len(e) else None,
                       "loss_sum": e.loss.sum() if len(e) else 0.0,
                       "frames_in_episodes": int(e.dur.sum()) if len(e) else 0}
        return r
    arms = arms or [p.name for p in sorted(RUNS.iterdir()) if p.is_dir() and p.name not in ("OR0", "OR9")]
    b = load("OR0").sort_values(["unit", "video_idx"]).reset_index(drop=True)
    b["state"] = state(b)
    res = {"OR0_all": {ds: stats(g) for ds, g in b.groupby("ds")}}
    for arm in arms:                                        # each arm next to OR0 on exactly the same units
        a = load(arm)
        if not len(a):
            continue
        a = a.sort_values(["unit", "video_idx"]).reset_index(drop=True)
        a["state"] = state(a)
        res[arm] = {}
        for ds, g in a.groupby("ds"):
            g0 = b[b.unit.isin(set(g.unit))].reset_index(drop=True)
            res[arm][ds] = {"units": int(g.unit.nunique()), "arm": stats(g.reset_index(drop=True)), "OR0": stats(g0)}
    save(res, "episodes.json")


def pool():
    """Where is the correct candidate in SGGL misses (GT present, J < 0.5)? First matching class, in order:
    own_view_misselect (output view scored present, an unchosen hypothesis of it has J >= 0.5),
    own_view_suppressed (output view scored absent, one of its raw hypotheses has J >= 0.5),
    other_view_output (the other view's actual mask is correct: gate picked the wrong view),
    other_view_hyp (only an unchosen/suppressed hypothesis of the other view is correct),
    none (no pool member reaches J >= 0.5: candidate generation failure)."""
    b = load("OR0")
    G, L = b[["cJ0", "cJ1", "cJ2"]].max(axis=1), b[["cJ3", "cJ4", "cJ5"]].max(axis=1)
    own_hyp, oth_hyp = np.where(b.acc, L, G), np.where(b.acc, G, L)
    own_sc, oth_j = np.where(b.acc, b.score_L, b.score_G), np.where(b.acc, b.J_G, b.J_L)
    miss = ~b.gt_empty & (b.J < TAU)
    cls = np.select([own_hyp >= TAU, oth_j >= TAU, oth_hyp >= TAU], [np.where(own_sc > 0, "own_view_misselect",
                     "own_view_suppressed"), "other_view_output", "other_view_hyp"], "none")
    b["cls"], b["loss"] = cls, 1 - b.JF
    b["gain_best"] = b.JF_best - b.JF
    res = {}
    for ds, g in b.groupby("ds"):
        m = g[miss[g.index]]
        pres = g[~g.gt_empty]
        res[ds] = {"present_frames": len(pres), "sggl_hit_rate": (pres.J >= TAU).mean(),
                   "pool_hit_rate": (pres[[f"cJ{k}" for k in range(6)]].max(axis=1) >= TAU).mean(),
                   "miss_frames": len(m), "miss_loss_share_of_total_loss": m.loss.sum() / g.loss.sum(),
                   "classes_frames": m.cls.value_counts().to_dict(),
                   "classes_loss_share": (m.groupby("cls").loss.sum() / m.loss.sum()).round(4).to_dict(),
                   "classes_recoverable_gain": m.groupby("cls").gain_best.sum().round(2).to_dict(),
                   "absent_false_present_loss_share": g[g.gt_empty].loss.sum() / g.loss.sum(),
                   "units": int(g.unit.nunique())}
    print(json.dumps(finite(res), indent=1))
    save(res, "pool.json")


def or9():
    """OR9 (amend2): GT-box-prompted image-mode SAM 2 vs OR0 on the same GT-present frames (video-equal)."""
    b, o = load("OR0"), load("OR9")
    m = o.merge(b[["unit", "video_idx", "JF", "J"]], on=["unit", "video_idx"])
    rng = np.random.default_rng(10)
    res = {}
    for ds, g in m.groupby("ds"):
        res[ds] = {}
        for v in ("full", "crop"):
            u = g.groupby(["ds", "video", "unit"]).agg(JF=(f"JF_{v}", "mean"), JF0=("JF", "mean"), J=(f"J_{v}", "mean"),
                                                        J0=("J", "mean"), F=(f"F_{v}", "mean"), area0=("area0", "first")).reset_index()
            u["F0"] = 2 * u.JF0 - u.J0
            u["d"], u["dJ"], u["dF"] = u.JF - u.JF0, u.J - u.J0, u.F - u.F0
            res[ds][v] = {**vstats(u, rng), "JF_oracle": u.groupby("video").JF.mean().mean(),
                          "tiny_JF_oracle": u[u.area0 < 0.001].JF.mean() if (u.area0 < 0.001).any() else None}
        res[ds]["frames"], res[ds]["units"] = len(g), int(g.unit.nunique())
        print(ds, {v: (round(res[ds][v]["JF_oracle"], 3), round(res[ds][v]["JF0"], 3)) for v in ("full", "crop")})
    save(res, "or9.json")


PAIRS = {"C48": ("OR4", "OR8"), "C58": ("OR5S", "OR8"), "C47": ("OR4", "OR7"), "C75": ("OR5S", "OR7")}


def interact():
    """Per dataset on units common to OR0, A, B, AB: standalone dA, dB; joint dAB; conditional (dAB - dA, dAB - dB);
    interaction dAB - dA - dB. Video-equal; one bootstrap over videos resamples all four jointly."""
    b = load("OR0")
    rng = np.random.default_rng(10)
    res = {}
    for ab, (A, B) in PAIRS.items():
        X = {k: load(k) for k in (A, B, ab)}
        if any(not len(x) for x in X.values()):
            continue
        res[ab] = {"A": A, "B": B}
        for ds in DSETS:
            common = set(b[b.ds == ds].unit)
            for x in X.values():
                common &= set(x[x.ds == ds].unit)
            if not common:
                continue
            V = {}
            for k, x in X.items():
                u = paired(x[x.unit.isin(common)], b[b.unit.isin(common)])
                V[k] = u.groupby("video").d.mean()
            v = pd.DataFrame(V)
            n = len(v)
            idx = rng.integers(0, n, (B_, n))
            def ci(col):
                bs = col.to_numpy()[idx].mean(1)
                return [np.percentile(bs, 2.5), np.percentile(bs, 97.5)]
            inter = v[ab] - v[A] - v[B]
            res[ab][ds] = {"units": len(common), "videos": n, "dA": v[A].mean(), "dB": v[B].mean(), "dAB": v[ab].mean(),
                           "ci_AB": ci(v[ab]), "cond_B_given_A": (v[ab] - v[A]).mean(), "ci_cond_B_given_A": ci(v[ab] - v[A]),
                           "cond_A_given_B": (v[ab] - v[B]).mean(), "ci_cond_A_given_B": ci(v[ab] - v[B]),
                           "interaction": inter.mean(), "ci_interaction": ci(inter)}
            r = res[ab][ds]
            print(ab, ds, r["units"], f"dA {r['dA']:+.3f} dB {r['dB']:+.3f} dAB {r['dAB']:+.3f} int {r['interaction']:+.3f}")
    save(res, "interactions.json")


FEATS = ("piou", "cos", "tiou", "cdist", "xiou", "larea")


def hyp_table(o):
    """Long table: one row per (frame, hypothesis k=0..5) with GT-free features and the GT J (cJk)."""
    rows = []
    for k in range(6):
        h = o[["ds", "video", "unit", "video_idx", "gt_empty", "J", "acc", "score_G", "score_L"]].copy()
        h["k"], h["view"] = k, "G" if k < 3 else "L"
        for f in ("piou", "cos", "tiou", "cdist", "xiou", "area"):
            c = f"h{f}{k}" if f == "area" and f"harea{k}" in o else f"{f}{k}"
            h[f] = o[c].to_numpy()
        h["cJ"] = o[f"cJ{k}"].to_numpy()
        h["vscore"] = np.where(k < 3, o.score_G, o.score_L)
        rows.append(h)
    H = pd.concat(rows, ignore_index=True)
    H["larea"] = np.log1p(H.area)
    H["cos"] = H["cos"].fillna(-1)
    H["cdist"] = H["cdist"].fillna(H.cdist.max())
    return H


def logit_fit(X, y, lam=1.0, it=25):
    """Ridge logistic regression by Newton/IRLS on standardised features; returns a scoring function (log-odds)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    R = lam * np.eye(Z.shape[1])
    R[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + R @ w
        Hs = (Z * (p * (1 - p))[:, None]).T @ Z + R
        w -= np.linalg.solve(Hs, g)
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w


def selector(arm="OR0F"):
    """Offline (framewise, dev) agreement of GT-free selection rules with the GT-best raw hypothesis, on GT-present
    frames where some hypothesis has J >= 0.5. Rules are fixed here BEFORE looking at results:
      R_sggl: SGGL's own output J; R_piou: argmax decoder IoU; R_cos: argmax frame-0 cosine; R_tiou: argmax IoU with
      the previous output; R_sum: argmax of z(piou)+z(cos)+z(tiou)-z(cdist) (z within frame); R_lr: logistic
      regression on FEATS (+view, vscore), 2-fold video-cluster cross-fitting within each dataset."""
    o = load(arm)
    H = hyp_table(o)
    key = ["unit", "video_idx"]
    z = lambda c: (H[c] - H.groupby(key)[c].transform("mean")) / (H.groupby(key)[c].transform("std") + 1e-6)
    H["s_sum"] = z("piou") + z("cos") + z("tiou") - z("cdist")
    H["s_lr"] = np.nan
    X = np.c_[H[list(FEATS)].to_numpy(), (H.view == "L").to_numpy(), H.vscore.to_numpy()]
    y = (H.cJ >= TAU).to_numpy()
    for ds in H.ds.unique():
        vids = np.array(sorted(H[H.ds == ds].video.unique()))
        fold = {v: i % 2 for i, v in enumerate(np.random.default_rng(10).permutation(vids))}
        for f in (0, 1):
            tr = (H.ds == ds) & (H.video.map(fold) != f)
            te = (H.ds == ds) & (H.video.map(fold) == f)
            H.loc[te, "s_lr"] = logit_fit(X[tr], y[tr])(X[te])
    res = {}
    for ds, g in H.groupby("ds"):
        fr = g.groupby(key)
        best = fr.cJ.max()
        pres = fr.gt_empty.first()                      # True = GT absent
        ok = best[(best >= TAU) & ~pres].index          # GT present and some hypothesis is correct
        r = {"frames_with_a_correct_hypothesis": len(ok), "present_frames": int((~pres).sum())}
        sg = fr.J.first()
        r["R_sggl_hit"] = float((sg.loc[ok] >= TAU).mean())
        r["R_sggl_meanJ_present"] = float(sg[~pres].mean())
        for name, col in (("R_piou", "piou"), ("R_cos", "cos"), ("R_tiou", "tiou"), ("R_sum", "s_sum"), ("R_lr", "s_lr")):
            pick = g.loc[g.groupby(key)[col].idxmax()].set_index(key).cJ
            r[f"{name}_hit"] = float((pick.loc[ok] >= TAU).mean())
            r[f"{name}_meanJ_present"] = float(pick[~pres.loc[pick.index]].mean())
        r["oracle_meanJ_present"] = float(best[~pres].mean())
        res[ds] = r
        print(ds, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    save(res, "selector_offline.json")


RULES = ("R_piou", "R_sum4", "R_anchor", "R_cos", "R_lr")


def rule_scores(H):
    """Add s_<rule> columns (amend5). z within frame over the 6 hypotheses; R_lr cross-fitted by video within dataset."""
    key = ["unit", "video_idx"]
    z = lambda c: (H[c] - H.groupby(key)[c].transform("mean")) / (H.groupby(key)[c].transform("std") + 1e-6)
    H["s_R_piou"] = H.piou
    H["s_R_sum4"] = z("piou") + z("cos") + z("tiou") - z("cdist")
    H["s_R_anchor"] = z("piou") + z("cos") + z("xiou")
    H["s_R_cos"] = H["cos"]
    X = np.c_[H[list(FEATS)].to_numpy(), (H.view == "L").to_numpy(), H.vscore.to_numpy()]
    y = (H.cJ >= TAU).to_numpy().astype(float)
    H["s_R_lr"] = np.nan
    for ds in H.ds.unique():
        vids = np.array(sorted(H[H.ds == ds].video.unique()))
        fold = {v: i % 2 for i, v in enumerate(np.random.default_rng(10).permutation(vids))}
        for f in (0, 1):
            tr = ((H.ds == ds) & (H.video.map(fold) != f)).to_numpy()
            te = ((H.ds == ds) & (H.video.map(fold) == f)).to_numpy()
            H.loc[te, "s_R_lr"] = logit_fit(X[tr], y[tr])(X[te])
    return H


def frame_choices(o, H, rule):
    """Per frame: SGGL hypothesis, rule pick, margin, J of both (GT used only for scoring the choice)."""
    F = o[["ds", "video", "unit", "video_idx", "gt_empty", "acc", "out_area", "J", "state"]].copy().reset_index(drop=True)
    S = H.pivot_table(index=["unit", "video_idx"], columns="k", values=f"s_{rule}").reindex(pd.MultiIndex.from_frame(F[["unit", "video_idx"]])).to_numpy()
    P = H.pivot_table(index=["unit", "video_idx"], columns="k", values="piou").reindex(pd.MultiIndex.from_frame(F[["unit", "video_idx"]])).to_numpy()
    Jc = H.pivot_table(index=["unit", "video_idx"], columns="k", values="cJ").reindex(pd.MultiIndex.from_frame(F[["unit", "video_idx"]])).to_numpy()
    view0 = np.where(F.acc.to_numpy(), 3, 0)
    k_sg = view0 + np.argmax(np.stack([P[np.arange(len(F)), view0 + i] for i in range(3)], 1), 1)
    pick = np.argmax(S, 1)
    r = np.arange(len(F))
    F["k_sggl"], F["pick"] = k_sg, pick
    F["margin"] = S[r, pick] - S[r, k_sg]
    F["J_sggl"], F["J_pick"] = Jc[r, k_sg], Jc[r, pick]
    F["active"] = F.out_area > 0
    return F


def gate_eval(F, tau, rng):
    sw = F.active & (F.pick != F.k_sggl) & (F.margin >= tau)
    dJ = np.where(sw, F.J_pick - F.J_sggl, 0.0)
    G = F.assign(dJ=dJ, sw=sw)
    out = {}
    for ds, g in G.groupby("ds"):
        pres = g[~g.gt_empty]
        v = pres.groupby(["video", "unit"]).dJ.mean().groupby("video").mean().to_numpy()
        bs = v[rng.integers(0, len(v), (B_, len(v)))].mean(1)
        s_ = g[g.sw]
        drift = pres[pres.state.isin(["wrong_region", "wrong_object", "empty"])]
        corr = pres[pres.state == "correct"]
        out[ds] = {"dJ_present": v.mean(), "ci": [np.percentile(bs, 2.5), np.percentile(bs, 97.5)],
                   "switches": int(len(s_)), "switch_rate_active": float(len(s_) / max(g.active.sum(), 1)),
                   "precision": float((s_.J_pick > s_.J_sggl).mean()) if len(s_) else None,
                   "worse_share": float((s_.J_pick < s_.J_sggl).mean()) if len(s_) else None,
                   "dJ_drift_frames": float(drift.dJ.mean()) if len(drift) else None,
                   "dJ_correct_frames": float(corr.dJ.mean()) if len(corr) else None,
                   "dJ_sum_switches": float(s_.eval("J_pick - J_sggl").sum()) if len(s_) else 0.0}
    return out


def gate(arm="OR0F"):
    """amend5 offline gate (review 11d-1). Writes reports/selector_gate.json."""
    o = load(arm)
    o["state"] = state(o)
    H = rule_scores(hyp_table(o))
    rng = np.random.default_rng(10)
    res = {"amend": "preregistration_amend5.json", "rules": {}}
    for rule in RULES:
        F = frame_choices(o, H, rule)
        pool = F[F.ds.isin(["mosev2", "vost"]) & F.active & (F.pick != F.k_sggl)]
        pos = pool.margin[pool.margin > 0].to_numpy()
        grid = [0.0] + (list(np.quantile(pos, np.arange(0.05, 1.0, 0.05))) if len(pos) else [])
        curve, tau_star = [], None
        for tau in grid:
            sw = pool[pool.margin >= tau]
            prec = float((sw.J_pick > sw.J_sggl).mean()) if len(sw) else None
            curve.append({"tau": float(tau), "switches": int(len(sw)), "precision_pooled": prec})
            if tau_star is None and prec is not None and prec >= 0.70:
                tau_star = float(tau)
        r = {"tau_star": tau_star, "curve": curve, "at_tau0": gate_eval(F, 0.0, rng)}
        if tau_star is not None:
            ev = gate_eval(F, tau_star, rng)
            r["at_tau_star"] = ev
            md, vo = ev.get("mosev2"), ev.get("vost")
            r["pass"] = bool(md and vo and md["ci"][0] > 0 and vo["ci"][0] > 0
                             and (md["precision"] or 0) >= 0.6 and (vo["precision"] or 0) >= 0.6
                             and (md["dJ_drift_frames"] or 0) > 0 and (vo["dJ_drift_frames"] or 0) > 0
                             and (md["dJ_correct_frames"] or 0) >= -0.005 and (vo["dJ_correct_frames"] or 0) >= -0.005)
        else:
            r["pass"] = False
        res["rules"][rule] = r
        e = r.get("at_tau_star", {})
        print(rule, "tau*", None if tau_star is None else round(tau_star, 4), "PASS" if r["pass"] else "fail",
              {ds: (round(x["dJ_present"], 4), [round(c, 4) for c in x["ci"]], x["switches"], x["precision"] and round(x["precision"], 3),
                    x["dJ_drift_frames"] and round(x["dJ_drift_frames"], 4), x["dJ_correct_frames"] and round(x["dJ_correct_frames"], 4)) for ds, x in e.items()})
    passing = [k for k, v in res["rules"].items() if v["pass"]]
    tot = lambda k: sum(res["rules"][k]["at_tau_star"][d]["dJ_present"] for d in ("mosev2", "vost"))
    best = None
    if passing:
        best = max(passing, key=tot)
        simple = [k for k in passing if k != "R_lr"]
        if best == "R_lr" and simple:
            alt = max(simple, key=tot)
            lr, al = res["rules"]["R_lr"]["at_tau_star"], res["rules"][alt]["at_tau_star"]
            if not all(lr[d]["dJ_present"] - al[d]["dJ_present"] > 0.005 for d in ("mosev2", "vost")):
                best = alt
    res["passing"], res["chosen"] = passing, best
    print("passing", passing, "chosen", best)
    save(res, "selector_gate.json")


def switch_precision(x):
    """Realized precision of the disagree-gated switch on an arm's own rows (xsw = picked hypothesis, -1 if none)."""
    sw = x[x.xsw >= 0]
    if not len(sw):
        return None, 0
    P = sw[[f"piou{k}" for k in range(6)]].to_numpy()
    J = sw[[f"cJ{k}" for k in range(6)]].to_numpy()
    kv = np.where(sw.acc.to_numpy(), 3, 0)
    ks = kv + np.argmax(np.stack([P[np.arange(len(sw)), kv + i] for i in range(3)], 1), 1)
    r = np.arange(len(sw))
    jp, js = J[r, sw.xsw.to_numpy().astype(int)], J[r, ks]
    nt = jp != js
    switch_precision.last_no_ties = float((jp[nt] > js[nt]).mean()) if nt.any() else None
    return float((jp > js).mean()), int(len(sw))


def degraded():
    """amend8: VXM as a degraded oracle. Per dataset: selector precision (offline OR0F disagree frames; realized on the
    arm's own switch frames) vs dJ&F (all frames, GT-present part), with VXO and reference points."""
    b = load("OR0")
    o = load("OR0F")
    o["state"] = state(o)
    H = rule_scores(hyp_table(o))
    F = frame_choices(o, H, "R_piou")
    X = H.pivot_table(index=["unit", "video_idx"], columns="k", values="xiou").reindex(pd.MultiIndex.from_frame(F[["unit", "video_idx"]])).to_numpy()
    F["xiou_sg"] = X[np.arange(len(F)), F.k_sggl]
    off = F[F.active & (F.pick != F.k_sggl) & (F.margin > 0) & (F.xiou_sg < 0.5)]
    rng = np.random.default_rng(10)
    nt = lambda g: float((g.J_pick > g.J_sggl)[g.J_pick != g.J_sggl].mean())
    res = {"offline_disagree_precision": {ds: {"precision": float((g.J_pick > g.J_sggl).mean()), "precision_no_ties": nt(g), "switches": len(g),
                                               "dJ_sum": float((g.J_pick - g.J_sggl).sum())} for ds, g in off.groupby("ds")},
           "arms": {}}
    for arm in ("VXM", "VXO", "N1M", "N2X", "OR5M", "OR7R"):
        x = load(arm)
        if not len(x):
            continue
        res["arms"][arm] = {}
        for ds, g in x.groupby("ds"):
            g0 = b[b.ds == ds]
            u = paired(g, g0)
            st = vstats(u, rng)
            m = g[["unit", "video_idx", "JF", "gt_empty"]].merge(g0[["unit", "video_idx", "JF"]], on=["unit", "video_idx"], suffixes=("", "_0"))
            m["d"] = m.JF - m.JF_0
            pres = m.groupby("unit").apply(lambda q: q.d[~q.gt_empty].sum() / len(q))
            vid = g.groupby("unit").video.first()
            dpres = pres.groupby(vid).mean().mean()
            prec, nsw = switch_precision(g) if arm in ("VXO", "VXM") else (None, 0)
            prec_nt = switch_precision.last_no_ties if arm in ("VXO", "VXM") and nsw else None
            if arm == "N2X":                       # controls: actual intervention counts (xsw there is only logged)
                nsw = int((g.xk >= 0).sum())
            elif arm == "N1M":
                nsw = int((g.rew_G | g.rew_L).sum())
            res["arms"][arm][ds] = {"dJF": st["dJF"], "ci": st["ci"], "dJF_present_part": float(dpres), "units": st["n_units"],
                                    "complete": g.unit.nunique() == g0.unit.nunique(),
                                    "realized_precision": prec, "realized_precision_no_ties": prec_nt, "switches": nsw,
                                    "switch_per_100f": 100 * nsw / len(g) if nsw else 0.0,
                                    "units_below_m005": st["units_below_m005"]}
    for arm, r in res["arms"].items():
        print(arm, {ds: (round(v["dJF"], 4), [round(c, 4) for c in v["ci"]], v["realized_precision"] and round(v["realized_precision"], 2), v["switches"], v["units"]) for ds, v in r.items()})
    print("offline", {ds: round(v["precision"], 2) for ds, v in res["offline_disagree_precision"].items()})
    save(res, "degraded.json")


def parity():
    b = load("OR0")
    res = {}
    for ds, g in b.groupby("ds"):
        d = Path("outputs/milestone9/runpod/diag") / ds
        src = d if d.exists() else Path("outputs/milestone9/runpod/runs/SGGL")
        m9 = pd.concat([pd.read_csv(f) for f in sorted(src.glob(f"{ds}__*.csv"))
                        if not f.name.endswith((".cand.csv", ".cand1b.csv", ".diag1b.csv"))], ignore_index=True)
        m = g[["unit", "video_idx", "JF"]].merge(m9[["unit", "video_idx", "JF"]], on=["unit", "video_idx"], suffixes=("", "_m9"))
        dd = (m.JF - m.JF_m9).abs()
        res[ds] = {"frames": len(m), "max_abs": dd.max(), "mean_abs": dd.mean(), "frames_diff_gt_1e-6": int((dd > 1e-6).sum()),
                   "units_identical": int((m.assign(z=dd).groupby("unit").z.max() < 1e-6).sum()), "units": int(m.unit.nunique()),
                   "m9_source": str(src)}
    print(res)
    save(res, "parity.json")


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
