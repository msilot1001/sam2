"""Analyses for the risk/memory feasibility study.

    uv run python scripts/rfeas_analyze.py check1    # needs all native runs; writes risk_model.json
    uv run python scripts/rfeas_analyze.py check2    # needs gated + single-frame runs
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from rfeas.pairing import longest_streak, paired  # noqa: E402
from rfeas.risk import (FEATURES, Logistic, auroc, average_precision, brier, calibration_bins, causal_features,  # noqa: E402
                        cluster_bootstrap, failure_episodes, future_labels)

CFG = yaml.safe_load(Path("configs/rfeas.yaml").read_text())
OUT = Path(CFG["output_dir"])
REP, FIG = OUT / "reports", OUT / "figures"
C1 = CFG["check1"]


def load_runs(policy: str) -> dict[str, pd.DataFrame]:
    units = pd.read_csv(OUT / "units.csv")
    out = {}
    for u in units.itertuples():
        d = OUT / "runs" / policy / u.unit
        if (d / "record.json").exists():
            out[u.unit] = pd.read_csv(d / "signals.csv").assign(unit=u.unit, video=u.video, split=u.split,
                                                                dataset=u.dataset)
    return out


def build_frames(native: dict, H: int) -> pd.DataFrame:
    rows = []
    for name, df in native.items():
        f = causal_features(df)
        lab = future_labels(df[["video_idx", "J", "gt_empty"]], H, C1["event_j"])
        sev = future_labels(df[["video_idx", "J", "gt_empty"]], H, C1["severe_j"])
        fut_min = [df.J.iloc[t + 1:t + 1 + H][~df.gt_empty.iloc[t + 1:t + 1 + H]].min() if t + H < len(df) else np.nan
                   for t in range(len(df))]
        rows.append(f.merge(lab, on="video_idx").assign(
            severe=sev.label.to_numpy(), J=df.J.to_numpy(), future_min_J=fut_min, unit=name,
            video=df.video.iloc[0], split=df.split.iloc[0], dataset=df.dataset.iloc[0]))
    return pd.concat(rows, ignore_index=True)


def score_table(fr: pd.DataFrame, model, platt) -> pd.DataFrame:
    s = fr.copy()
    s["A_presence"] = -s.obj_score
    s["B_pred_iou"] = -s.pred_iou
    s["C_consistency"] = -s.consistency_iou_prev.fillna(s.consistency_iou_prev.mean())
    s["D_logistic"] = model.predict(s[FEATURES].to_numpy())
    if platt is not None:
        s["E_platt"] = platt.predict(s[["D_logistic"]].to_numpy())
    return s


def metrics(df: pd.DataFrame, score: str, label: str = "label", ci: bool = True) -> dict:
    y, sc = df[label].to_numpy(), df[score].to_numpy()
    r = {"score": score, "frames": int(len(df)), "positives": int(y.sum()), "auroc": auroc(y, sc),
         "ap": average_precision(y, sc), "base_rate": float(y.mean()) if len(y) else np.nan}
    if ci and y.sum() > 0:
        r["auroc_ci"] = cluster_bootstrap(df, "video", lambda d: auroc(d[label], d[score]), n=2000)
        r["ap_ci"] = cluster_bootstrap(df, "video", lambda d: average_precision(d[label], d[score]), n=2000)
    return r


def check1():
    for d in (REP, FIG):
        d.mkdir(parents=True, exist_ok=True)
    native = load_runs("native")
    units = pd.read_csv(OUT / "units.csv")
    report = {"units_total": len(units), "units_with_native_run": len(native)}
    if len(native) != len(units):
        print(f"WARNING: {len(units) - len(native)} native runs missing")
    allf = pd.concat(native.values(), ignore_index=True)
    # descriptive: native tracking quality and failure episodes (all frames, prompt excluded)
    eps = []
    for name, df in native.items():
        for thr, tag in ((C1["event_j"], "J<0.5"), (C1["severe_j"], "J<0.1")):
            for a, b in failure_episodes(df, thr):
                eps.append({"unit": name, "video": df.video.iloc[0], "split": df.split.iloc[0],
                            "dataset": df.dataset.iloc[0], "type": tag, "start": a, "end": b, "length": b - a + 1})
    eps = pd.DataFrame(eps, columns=["unit", "video", "split", "dataset", "type", "start", "end", "length"])
    eps.to_csv(REP / "check1_failure_episodes.csv", index=False)
    nf = allf[allf.video_idx > 0]
    report["native_quality"] = nf.groupby("dataset").agg(frames=("J", "size"), mean_J=("J", "mean"),
                                                        mean_JF=("JF", "mean"),
                                                        vis_frames_J_lt_0_5=("J", lambda s: int((s < .5).sum()))).reset_index().to_dict("records")
    report["episodes"] = eps.groupby(["type", "split"]).agg(episodes=("unit", "size"), videos=("video", "nunique"),
                                                            units=("unit", "nunique")).reset_index().to_dict("records")
    res = {}
    for H in (C1["horizon"], C1["horizon_sensitivity"]):
        fr = build_frames(native, H)
        el = fr[fr.eligible]
        dev, val = el[el.split == "dev"], el[el.split == "val"]
        dev_eps = eps[(eps.split == "dev") & (eps.type == "J<0.5")]
        model = Logistic(1.0).fit(dev[FEATURES].to_numpy(), dev.label.to_numpy())
        dev_risk = model.predict(dev[FEATURES].to_numpy())
        platt = None
        if len(dev_eps) >= C1["min_episodes_for_claims"] and dev.label.sum() > 0:
            platt = Logistic(1e-6).fit(dev_risk[:, None], dev.label.to_numpy())
        thr = float(np.percentile(dev_risk, 90))
        if H == C1["horizon"]:
            (OUT / "risk_model.json").write_text(json.dumps({"model": model.to_dict(), "threshold": thr,
                                                             "features": FEATURES, "horizon": H,
                                                             "fit_on": "dev eligible frames"}, indent=2))
        sv = score_table(val, model, platt)
        sd = score_table(dev, model, platt)
        scores = ["A_presence", "B_pred_iou", "C_consistency", "D_logistic"] + (["E_platt"] if platt else [])
        out = {"dev": {"frames": len(dev), "positives": int(dev.label.sum()), "units": dev.unit.nunique(),
                       "videos": dev.video.nunique(), "episodes_J<0.5": len(dev_eps)},
               "val": {"frames": len(val), "positives": int(val.label.sum()), "units": val.unit.nunique(),
                       "videos": val.video.nunique(), "videos_with_positive": int(val[val.label > 0].video.nunique()),
                       "severe_positives": int(val.severe.sum())},
               "calibration_fitted": platt is not None, "threshold_dev_p90": thr}
        out["val_all"] = [metrics(sv, s) for s in scores]
        out["val_currently_accurate"] = [metrics(sv[sv.current_accurate], s) for s in scores]
        out["val_severe"] = [metrics(sv, s, "severe", ci=False) for s in scores]
        out["dev_all"] = [metrics(sd, s, ci=False) for s in scores]
        base = dev.label.mean()
        out["brier"] = {"constant_dev_rate": brier(val.label, np.full(len(val), base)),
                        "D_logistic": brier(val.label, sv.D_logistic),
                        **({"E_platt": brier(val.label, sv.E_platt)} if platt else {})}
        # separate associations
        out["assoc_current_error_auroc"] = {s: auroc(sv.current_fail, sv[s]) for s in scores}
        acc = sv[sv.current_accurate & sv.future_min_J.notna()]
        out["assoc_error_propagation_spearman"] = {
            s: float(pd.Series(acc[s].rank()).corr(pd.Series((-acc.future_min_J).rank()))) for s in scores}
        # alarm behaviour of the D model at the gate threshold
        flag = sv.D_logistic >= thr
        out["alarm"] = {"false_alarm_rate": float(flag[sv.label == 0].mean()) if (sv.label == 0).any() else np.nan,
                        "recall": float(flag[sv.label == 1].mean()) if (sv.label == 1).any() else np.nan,
                        "flagged_fraction": float(flag.mean())}
        # lead time: for each val J<0.5 episode, earliest alarm in the H frames before onset
        lead = []
        risk_by = {u: g.set_index("video_idx") for u, g in score_table(fr[fr.split == "val"], model, platt).groupby("unit")}
        for e in eps[(eps.split == "val") & (eps.type == "J<0.5")].itertuples():
            g = risk_by.get(e.unit)
            if g is None:
                continue
            before = [t for t in range(e.start - H, e.start) if t > 0 and t in g.index]
            hits = [t for t in before if g.loc[t, "D_logistic"] >= thr]
            lead.append({"unit": e.unit, "onset": e.start, "alarmed": bool(hits),
                         "lead_frames": e.start - min(hits) if hits else 0, "prior_frames": len(before)})
        lead = pd.DataFrame(lead, columns=["unit", "onset", "alarmed", "lead_frames", "prior_frames"])
        out["lead_time"] = {"episodes": len(lead), "alarmed": int(lead.alarmed.sum()) if len(lead) else 0,
                            "median_lead_if_alarmed": float(lead[lead.alarmed].lead_frames.median()) if lead.alarmed.any() else None}
        pv = []
        for v, g in sv.groupby("video"):
            if 0 < g.label.sum() < len(g):
                pv.append({"video": v, "frames": len(g), "positives": int(g.label.sum()),
                           **{s: auroc(g.label, g[s]) for s in scores}})
        out["per_video_auroc"] = pv
        res[f"H{H}"] = out
        if H == C1["horizon"]:
            sv.to_csv(REP / "check1_val_scores.csv", index=False)
            lead.to_csv(REP / "check1_lead_time.csv", index=False)
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
            for s in ["D_logistic"] + (["E_platt"] if platt else []):
                cb = calibration_bins(sv.label, sv[s], 10)
                axes[0].plot(cb.mean_pred, cb.obs_rate, "o-", label=s)
            axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
            axes[0].axhline(base, color="0.6", ls="--", lw=0.8, label=f"constant dev rate {base:.3f}")
            axes[0].set(xlabel="predicted risk", ylabel="observed future-failure rate", title=f"VAL calibration (H={H})")
            axes[0].legend(fontsize=8)
            for s in scores:
                y, sc = sv.label.to_numpy(), sv[s].to_numpy()
                o = np.argsort(-sc)
                tp, fp = np.cumsum(y[o]), np.cumsum(1 - y[o])
                if y.sum() and (1 - y).sum():
                    axes[1].plot(fp / (1 - y).sum(), tp / y.sum(), label=f"{s} AUROC {auroc(y, sc):.2f}")
            axes[1].plot([0, 1], [0, 1], "k:", lw=0.8)
            axes[1].set(xlabel="false positive rate", ylabel="true positive rate", title="VAL ROC (pooled frames; see CI)")
            axes[1].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(FIG / "check1_calibration_roc.png", dpi=110)
            plt.close(fig)
    report["check1"] = res
    (REP / "check1_report.json").write_text(json.dumps(report, indent=2, default=lambda o: o.item()
                                                        if isinstance(o, np.generic) else str(o)))
    print(json.dumps(report, indent=1, default=str)[:12000])


def check2():
    native = load_runs("native")
    policies = ["presence_gate", "iou_gate", "consistency_gate", "risk_gate", "random_matched",
                "single_maxrisk", "single_random"]
    rows, frames = [], []
    for p in policies:
        runs = load_runs(p)
        for u, df in runs.items():
            nat = native[u]
            pr = paired(nat, df)
            rec = json.loads((OUT / "runs" / p / u / "record.json").read_text())
            n_nonprompt = len(df) - 1
            vis = ~pr.gt_empty
            nj, pj = nat[nat.video_idx > 0], df[df.video_idx > 0]
            row = {"policy": p, "unit": u, "video": nat.video.iloc[0], "dataset": nat.dataset.iloc[0],
                   "frames": n_nonprompt, "skipped": len(rec["skipped"]), "skip_frac": len(rec["skipped"]) / n_nonprompt,
                   **{f"delta_{m}": pr[f"delta_{m}"].mean() for m in ["J", "F", "JF", "F_strict"]},
                   "JF_native": pr.JF_native.mean(), "JF_policy": pr.JF_policy.mean(),
                   "loss_frames_native": int(((pr.J_native < 0.1) & vis).sum()),
                   "loss_frames_policy": int(((pr.J_policy < 0.1) & vis).sum()),
                   "fail_frames_native": int(((pr.J_native < 0.5) & vis).sum()),
                   "fail_frames_policy": int(((pr.J_policy < 0.5) & vis).sum()),
                   "streak_native": longest_streak(nj.J[~nj.gt_empty]), "streak_policy": longest_streak(pj.J[~pj.gt_empty]),
                   "latency_ratio": pj.latency_s.median() / nj.latency_s.median()}
            if p.startswith("single"):
                t = rec["skip_frame"]
                after = pr[pr.video_idx > t]
                row |= {"skip_frame": t, "frames_changed_after": int((after.delta_J.abs() > 1e-9).sum()),
                        "delta_JF_after": after.delta_JF.mean() if len(after) else np.nan,
                        "native_fail_after_10": bool(((after.J_native < .5) & ~after.gt_empty).head(10).any()),
                        "policy_fail_after_10": bool(((after.J_policy < .5) & ~after.gt_empty).head(10).any())}
            rows.append(row)
            frames.append(pr.assign(policy=p, unit=u))
    R = pd.DataFrame(rows)
    REP.mkdir(parents=True, exist_ok=True)
    R.to_csv(REP / "check2_units.csv", index=False)
    pd.concat(frames).to_csv(REP / "check2_paired_frames.csv.gz", index=False, compression="gzip")
    g = R[~R.policy.str.startswith("single")]
    summ = []
    for p, d in g.groupby("policy"):
        ci = cluster_bootstrap(d, "video", lambda x: x.delta_JF.mean(), n=5000)
        summ.append({"policy": p, "units": len(d), "mean_skip_frac": d.skip_frac.mean(),
                     "delta_JF": d.delta_JF.mean(), "delta_JF_ci": ci, "delta_J": d.delta_J.mean(),
                     "delta_F_strict": d.delta_F_strict.mean(),
                     "helpful_units(>+0.01)": int((d.delta_JF > 0.01).sum()), "harmful_units(<-0.01)": int((d.delta_JF < -0.01).sum()),
                     "loss_frames_native": int(d.loss_frames_native.sum()), "loss_frames_policy": int(d.loss_frames_policy.sum()),
                     "fail_frames_native": int(d.fail_frames_native.sum()), "fail_frames_policy": int(d.fail_frames_policy.sum()),
                     "max_streak_native": int(d.streak_native.max()), "max_streak_policy": int(d.streak_policy.max()),
                     "median_latency_ratio": float(d.latency_ratio.median())})
    S = pd.DataFrame(summ)
    S.to_csv(REP / "check2_summary.csv", index=False)
    sing = R[R.policy.str.startswith("single")]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(S.round(4).to_string())
    print(sing[["policy", "unit", "skip_frame", "frames_changed_after", "delta_JF_after", "native_fail_after_10",
                "policy_fail_after_10"]].round(4).to_string())
    print(g.pivot_table(index="unit", columns="policy", values="delta_JF").round(4).to_string())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    order = ["presence_gate", "iou_gate", "consistency_gate", "risk_gate", "random_matched"]
    for i, p in enumerate(order):
        d = g[g.policy == p]
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-.12, .12, len(d)), d.delta_JF, s=18)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xticks=range(len(order)), xticklabels=order, ylabel="unit mean delta J&F vs native",
           title="CHECK 2: per-unit paired change (12 pre-registered VAL units)")
    fig.tight_layout()
    fig.savefig(FIG / "check2_policy_deltas.png", dpi=110)
    plt.close(fig)
    (REP / "check2_report.json").write_text(json.dumps({"summary": S.to_dict("records"),
                                                        "single": sing.to_dict("records")}, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check1", "check2"])
    {"check1": check1, "check2": check2}[ap.parse_args().what]()
