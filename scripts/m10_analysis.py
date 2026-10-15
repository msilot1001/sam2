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
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
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
        a = load(arm).sort_values(["unit", "video_idx"]).reset_index(drop=True)
        if not len(a):
            continue
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
