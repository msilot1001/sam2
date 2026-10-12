"""Milestone 4 analysis: validate the 96 runs, aggregate, failure/recovery analysis, figures.

    uv run python scripts/analyze_dropout_baseline.py

Reads outputs/milestone4/runs (never runs inference). Writes outputs/milestone4/{reports,figures}.
"""
import hashlib
import json
from importlib.util import find_spec
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from baseline import frame_drops as fd  # noqa: E402
from baseline.dropout_analysis import (METRICS, aggregate, bootstrap_ci, failure_classes, failure_runs,  # noqa: E402
                                       run_summary, worst_windows)
from baseline.dropout_eval import drop_runs  # noqa: E402
from baseline.dropout_runner import RunSpec, experiment_matrix, manifest_path, run_dir, validate_run  # noqa: E402
from baseline.metrics import seg2bmap  # noqa: E402
from vkitti.calibration import rotation_angle_deg  # noqa: E402
from vkitti.dataset import VKittiDataset  # noqa: E402

import os  # noqa: E402

CFG = yaml.safe_load(Path(os.environ.get("M4_CONFIG", "configs/m4_dropout_baseline.yaml")).read_text())
M3 = yaml.safe_load(Path(CFG["m3_config"]).read_text())
COND_SRC = yaml.safe_load(Path(CFG["conditions_from"]).read_text()) if "conditions_from" in CFG else M3
M2 = yaml.safe_load(Path(M3["m2_config"]).read_text())
OUT = Path(CFG["output_dir"])
REP, FIG = OUT / "reports", OUT / "figures"
CONDS = list(COND_SRC["conditions"])
SEEDS = COND_SRC["seeds"]
BURSTS = [c for c, v in COND_SRC["conditions"].items() if v["kind"] == "burst"]
TIMELINE = CFG.get("timeline_condition", "burst_50")
COLORS = dict(zip(CONDS, ["#4C72B0", "#1F3B73", "#DD8452", "#8C3B0F"]))
THR, LOST = M3["recovery"]["j_threshold"], M2["evaluation"]["lost_j"]
OFFICIAL_FILES = ["modeling/sam2_base.py", "sam2_video_predictor.py", "modeling/memory_attention.py",
                  "modeling/memory_encoder.py", "configs/sam2.1/sam2.1_hiera_t.yaml", "build_sam.py"]


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)
    return str(FIG / name)


def tag(track):
    return f"{track.scene} t{track.track_id}"


# ----------------------------------------------------------------------------- load + validate
def load_all():
    specs = experiment_matrix(CFG["tracks"], CONDS, SEEDS)
    ds = {s: VKittiDataset(M2["root"], s, M2["variation"], M2["camera_id"]) for s in CFG["tracks"]}
    shape = next(iter(ds.values())).image_shape
    pkg = Path(find_spec("sam2").origin).parent
    current = {f: hashlib.sha256((pkg / f).read_bytes()).hexdigest() for f in OFFICIAL_FILES}
    paired, perf, events, recs, problems, patterns = [], [], [], [], [], {}
    model0 = None
    for s in specs:
        p = fd.load(manifest_path(Path(CFG["manifests_dir"]), s))
        patterns[s.name] = p
        d = run_dir(OUT, s)
        rec = json.loads((d / "record.json").read_text()) if (d / "record.json").exists() else None
        model = rec["model"] if rec else {}
        ok, why = validate_run(d, p, model, shape) if rec else (False, "no record")
        if not ok:
            problems.append((s.name, why))
            continue
        if model.get("sam2_file_sha256") != current:
            problems.append((s.name, "official SAM 2 files differ from the run"))
        model0 = model0 or model
        if {k: v for k, v in model.items()} != model0:
            problems.append((s.name, "model info differs between runs"))
        key = {"scene": s.scene, "track_id": s.track_id, "condition": s.condition, "seed": s.seed}
        pr = pd.read_csv(d / "paired.csv").assign(**key)
        pf = pd.read_csv(d / "per_frame.csv").assign(**key)
        ev = pd.read_csv(d / "events.csv").assign(**key) if (d / "events.csv").stat().st_size > 1 else pd.DataFrame()
        rec_ids = set(p.received_ids) - {p.meta["prompt_original_frame_id"]}
        if set(pr.original_frame_id) != rec_ids or pr.original_frame_id.duplicated().any():
            problems.append((s.name, "paired frames != received non-prompt frames"))
        if pf.original_frame_id.duplicated().any():
            problems.append((s.name, "duplicate per-frame outputs"))
        paired.append(pr)
        perf.append(pf)
        events.append(ev)
        recs.append(key | {"actual_drop_rate": rec["manifest"]["actual_drop_rate"],
                           "received": rec["manifest"]["received"], **rec["timing"],
                           "mps_peak_gb": (rec["memory"]["mps_driver_peak_bytes"] or 0) / 1e9,
                           "system_free_percent_before": rec["memory"]["system_free_percent_before"],
                           "pred_bytes": (d / "predictions.npz").stat().st_size})
    return (specs, ds, patterns, pd.concat(paired, ignore_index=True), pd.concat(perf, ignore_index=True),
            pd.concat(events, ignore_index=True), pd.DataFrame(recs), problems, model0)


def add_camera_motion(paired, ds, patterns):
    """For each matched frame: original gap since the previous received frame and the camera
    translation/rotation over that gap (camera pose from extrinsic.txt; retrospective analysis only)."""
    rows = []
    for (sc, tr, c, sd), g in paired.groupby(["scene", "track_id", "condition", "seed"]):
        rec = patterns[f"{sc}_track{tr:03d}/{c}_seed{sd}"].received_ids
        prev = dict(zip(rec[1:], rec[:-1]))
        for f in g.original_frame_id:
            a, b = ds[sc].camera(prev[f]), ds[sc].camera(f)
            rows.append({"scene": sc, "track_id": tr, "condition": c, "seed": sd, "original_frame_id": f,
                         "gap_original": f - prev[f],
                         "cam_dT_m": float(np.linalg.norm(b["camera_position"] - a["camera_position"])),
                         "cam_dR_deg": rotation_angle_deg(a["camera_rotation"], b["camera_rotation"])})
    return paired.merge(pd.DataFrame(rows), on=["scene", "track_id", "condition", "seed", "original_frame_id"])


# ----------------------------------------------------------------------------- figures
def fig_timelines(m2pf, paired, patterns):
    tracks = list(m2pf.groupby(["scene", "track_id"], sort=False))
    fig, axes = plt.subplots(len(tracks), 1, figsize=(13, 2.1 * len(tracks)), squeeze=False)
    for ax, ((sc, tr), g) in zip(axes[:, 0], tracks):
        p = patterns[f"{sc}_track{tr:03d}/{TIMELINE}_seed0"]
        for s, e in drop_runs(p.frames):
            ax.axvspan(s - 0.5, e + 0.5, color="#C44E52", alpha=0.15, lw=0)
        e = g[~g.is_prompt]
        ax.plot(e.original_frame_id, e.J, color="0.55", lw=1, label="no-drop J (M2, all frames)")
        d = paired[(paired.scene == sc) & (paired.track_id == tr) & (paired.condition == TIMELINE) & (paired.seed == 0)]
        ax.plot(d.original_frame_id, d.J_drop, ".", ms=4, color="#8C3B0F", label=f"{TIMELINE} seed0 J (received frames)")
        ax.set(ylim=(-0.02, 1.02), ylabel=f"{sc}\ntrack {tr}")
    axes[0, 0].legend(ncol=2, fontsize=8, loc="lower left")
    axes[-1, 0].set_xlabel("original VKITTI frame (red = dropped burst)")
    return save(fig, f"01_matched_J_over_time_{TIMELINE}_seed0.png")


def fig_delta_hist(paired, metric, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    bins = np.linspace(-1, 1, 81)
    for c in CONDS:
        v = paired[paired.condition == c][f"delta_{metric}"]
        axes[0].hist(v, bins=bins, histtype="step", lw=1.5, color=COLORS[c], label=f"{c} (n={len(v)})")
    axes[0].set(yscale="log", xlabel=f"per-frame delta {metric} (dropout - matched no-drop), full range [-1, 1]",
                ylabel="frames (log scale)", title=f"per-frame paired delta {metric}")
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].legend(fontsize=8)
    r = paired.groupby(["condition", "scene", "track_id", "seed"])[f"delta_{metric}"].mean().reset_index()
    for i, c in enumerate(CONDS):
        v = r[r.condition == c][f"delta_{metric}"]
        axes[1].scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=14,
                        color=COLORS[c])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set(xticks=range(4), xticklabels=CONDS, ylabel=f"run mean delta {metric}",
                title=f"run-level mean delta {metric} (24 runs per condition; note y-scale)")
    return save(fig, name)


def fig_condition(agg_all, cis):
    o = agg_all["overall"].set_index("condition").loc[CONDS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(4)
    for k, (col, lab) in enumerate([("JF_no_drop_matched", "no-drop (matched frames)"), ("JF_drop", "dropout")]):
        axes[0].bar(x + (k - 0.5) * 0.35, o[col], 0.35, label=lab, color=["0.65", "#4C72B0"][k])
    axes[0].set(xticks=x, xticklabels=CONDS, ylim=(0, 1), ylabel="J&F (mean of 8 tracks)",
                title="Accuracy level, full 0-1 axis")
    axes[0].legend(loc="lower right")
    for k, m in enumerate(METRICS):
        lo = [cis[(c, m)][0] for c in CONDS]
        hi = [cis[(c, m)][1] for c in CONDS]
        v = o[f"delta_{m}"].to_numpy()
        axes[1].errorbar(x + (k - 1.5) * 0.18, v, yerr=[v - lo, np.array(hi) - v], fmt="o", capsize=3, label=m)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set(xticks=x, xticklabels=CONDS, ylabel="paired delta (dropout - no-drop)",
                title="Paired change, 95% bootstrap CI over 8 tracks (zoomed axis, see ticks)")
    axes[1].legend(fontsize=8)
    return save(fig, "04_condition_comparison.png")


def fig_scene(agg_all):
    s = agg_all["scene"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    scenes = sorted(s.scene.unique())
    for i, sc in enumerate(scenes):
        v = s[s.scene == sc].set_index("condition").loc[CONDS]
        ax.bar(np.arange(4) + (i - 0.5) * 0.35, v.delta_JF, 0.35, label=sc)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xticks=range(4), xticklabels=CONDS, ylabel="delta J&F (mean of 4 tracks)",
           title="Scene-level paired change in J&F (zoomed axis)")
    ax.legend()
    return save(fig, "05_scene_comparison.png")


def fig_seeds(runs):
    r = runs[runs.subset == "all"]
    tracks = list(r.groupby(["scene", "track_id"], sort=False).groups)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for ax, c in zip(axes, CONDS):
        for i, (sc, tr) in enumerate(tracks):
            v = r[(r.condition == c) & (r.scene == sc) & (r.track_id == tr)].sort_values("seed")
            ax.plot([i] * len(v), v.delta_J, "-", color="0.8", zorder=0)
            ax.scatter([i] * len(v), v.delta_J, c=v.seed, cmap="viridis", vmin=0, vmax=2, s=25)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(title=c, xticks=range(len(tracks)), xticklabels=[f"{s[-2:]}\nt{t}" for s, t in tracks])
    axes[0].set_ylabel("run delta J (dots = seeds 0/1/2)")
    fig.suptitle("Seed variability per track")
    return save(fig, "06_seed_variability.png")


def fig_size(paired):
    v = paired[~paired.gt_empty].copy()
    bins = [0, 500, 1000, 2000, 4000, 10000, 1e6]
    v["size_bin"] = pd.cut(v.gt_pixels, bins)
    t = v.groupby(["size_bin", "condition"], observed=True).agg(dJ=("delta_J", "mean"), n=("delta_J", "size"),
                                                                Jnd=("J_no_drop_matched", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    labels = ["<=500", "500-1k", "1k-2k", "2k-4k", "4k-10k", ">10k"]
    for c in CONDS:
        g = t[t.condition == c]
        axes[0].plot(range(len(g)), g.dJ, "o-", color=COLORS[c], label=c)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set(xticks=range(len(labels)), xticklabels=labels, xlabel="GT mask pixels", ylabel="mean delta J",
                title="Degradation by object size (visible-GT frames)")
    axes[0].legend(fontsize=8)
    g = t[t.condition == CONDS[0]]
    axes[1].bar(range(len(g)), g.Jnd, color="0.6")
    axes[1].set(xticks=range(len(labels)), xticklabels=labels, ylim=(0, 1), xlabel="GT mask pixels",
                ylabel="no-drop J (matched, random_30 frames)", title="Accuracy level by object size")
    return save(fig, "07_object_size_vs_degradation.png"), t


def fig_recovery(ev):
    b = ev[ev.condition.isin(BURSTS)]
    fig, axes = plt.subplots(1, len(BURSTS), figsize=(7 * len(BURSTS), 4.5), sharey=True, squeeze=False)
    axes = axes[0]
    cls_col = {"intact_at_first_frame": "#55A868", "recovered_later": "#4C72B0", "unrecovered_next_burst": "#C44E52",
               "unrecovered_end": "#8C3B0F", "insufficient_observation": "#BBBBBB", "no_observation": "#BBBBBB"}
    for ax, c in zip(axes, BURSTS):
        for wj, cls in b[b.condition == c][["window_J", "class"]].itertuples(index=False):
            js = json.loads(wj) if isinstance(wj, str) else []
            if js:
                ax.plot(range(1, len(js) + 1), js, "-", lw=0.6, alpha=0.5, color=cls_col[cls])
        ax.axhline(THR, color="k", ls=":", lw=0.8)
        ax.set(title=f"{c}: J after each burst (colour = class)", xlabel="received frames after burst", xlim=(0.5, 15.5))
    axes[0].set_ylabel("J")
    handles = [plt.Line2D([], [], color=v, label=k) for k, v in cls_col.items() if k != "no_observation"]
    axes[1].legend(handles=handles, fontsize=7, loc="lower right")
    return save(fig, "08_burst_recovery_timelines.png")


def fig_censoring(ev):
    b = ev[ev.condition.isin(BURSTS)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for c in BURSTS:
        w = b[b.condition == c].window_length
        axes[0].hist(w, bins=np.arange(-0.5, w.max() + 1.5), histtype="step", lw=1.5, color=COLORS[c],
                     label=f"{c} (n={len(w)})")
    axes[0].axvline(1.5, color="k", ls=":", lw=0.8)
    axes[0].axvline(4.5, color="k", ls="--", lw=0.8)
    axes[0].set(xlabel="observation window (received frames before next burst / clip end)", ylabel="burst events",
                title="Window length (dotted: <2 cannot confirm, dashed: >=5 needed for within-5)")
    axes[0].legend()
    t = b.groupby(["condition", "class"]).size().unstack(fill_value=0)
    t.plot.bar(stacked=True, ax=axes[1], rot=0)
    axes[1].set(ylabel="burst events", title="Event classes")
    axes[1].legend(fontsize=7)
    return save(fig, "09_censoring_distribution.png")


def overlay(rgb, gt, pr):
    out = rgb.astype(np.float32)
    out[pr] = 0.5 * out[pr] + 0.5 * np.array([255, 0, 0])
    e = seg2bmap(gt)
    out[e | np.roll(e, 1, 0) | np.roll(e, 1, 1)] = [0, 255, 0]
    return out.astype(np.uint8)


def crop(gt, a, b, shape, pad=50):
    ys, xs = np.nonzero(gt | a | b)
    if len(ys) == 0:
        return slice(0, shape[0]), slice(0, shape[1])
    return (slice(max(ys.min() - pad, 0), min(ys.max() + pad, shape[0])),
            slice(max(xs.min() - pad, 0), min(xs.max() + pad, shape[1])))


def fig_case(ds, sc, tr, cond, seed, frames, title, name):
    """RGB | GT | no-drop prediction | dropout prediction (crop around target)."""
    d = run_dir(OUT, RunSpec(sc, tr, cond, seed))
    if not (d / "record.json").exists():
        return f"SKIPPED {name}: run {sc} t{tr} {cond} s{seed} not available"
    z = np.load(d / "predictions.npz")
    drop = dict(zip(z["original_frame_ids"].tolist(), z["masks"]))
    m2 = np.load(Path(M3["m2_output_dir"]) / "predictions" / f"{sc}_track{tr:03d}.npz")
    nd = dict(zip(m2["original_frame_ids"].tolist(), m2["masks"]))
    pr = pd.read_csv(d / "paired.csv").set_index("original_frame_id")
    frames = [f for f in frames if f in nd]   # stay inside the clip
    fig, axes = plt.subplots(len(frames), 4, figsize=(18, 2.9 * len(frames)), squeeze=False)
    for r, f in enumerate(frames):
        s = ds[sc].get_frame(f, tr)
        gt = s["mask"]
        a = nd[f]
        b = drop.get(f, np.zeros_like(gt))
        ys, xs = crop(gt, a, b, gt.shape)
        jd = f"{pr.loc[f, 'J_drop']:.2f}" if f in pr.index else "dropped"
        jn = f"{pr.loc[f, 'J_no_drop_matched']:.2f}" if f in pr.index else "-"
        panels = [(s["rgb"][ys, xs], f"frame {f}  RGB"), (gt[ys, xs], f"GT ({int(gt.sum())} px)"),
                  (overlay(s["rgb"], gt, a)[ys, xs], f"no-drop pred, J={jn}"),
                  (overlay(s["rgb"], gt, b)[ys, xs], f"{cond} s{seed} pred, J={jd}" if f in drop else "frame DROPPED")]
        for c, (img, t) in enumerate(panels):
            axes[r, c].imshow(img, cmap="gray" if img.ndim == 2 else None)
            axes[r, c].set_title(t, fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle(title + "   (red = prediction, green = GT contour)")
    return save(fig, name, rect=(0, 0, 1, 0.97))


# ----------------------------------------------------------------------------- main
def main():
    REP.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    specs, ds, patterns, paired, perf, ev, recs, problems, model = load_all()
    print(f"valid runs: {len(recs)}/{len(specs)}; problems: {problems[:5]}")
    for m in METRICS:
        assert np.allclose(paired[f"delta_{m}"], paired[f"{m}_drop"] - paired[f"{m}_no_drop_matched"])
    paired = add_camera_motion(paired, ds, patterns)
    m2pf = pd.read_csv(Path(M3["m2_output_dir"]) / "reports" / "per_frame_metrics.csv")

    # Aggregation.
    runs = run_summary(paired)
    runs = runs.merge(recs[["scene", "track_id", "condition", "seed", "actual_drop_rate", "received"]],
                      on=["scene", "track_id", "condition", "seed"])
    agg = {s: aggregate(runs, s) for s in ("all", "visible", "empty_gt")}
    cis, sig = {}, []
    for c in CONDS:
        t = agg["all"]["track"][agg["all"]["track"].condition == c]
        for m in METRICS:
            cis[(c, m)] = bootstrap_ci(t[f"delta_{m}"])
            sig.append({"condition": c, "metric": m, "mean_delta": t[f"delta_{m}"].mean(), "ci_lo": cis[(c, m)][0],
                        "ci_hi": cis[(c, m)][1], "tracks_negative": int((t[f"delta_{m}"] < 0).sum()),
                        "tracks_positive": int((t[f"delta_{m}"] > 0).sum()), "tracks": len(t)})
    sig = pd.DataFrame(sig)
    rate = recs.groupby("condition").actual_drop_rate.agg(["min", "mean", "max"]).loc[CONDS]
    seed_overall = runs[runs.subset == "all"].groupby(["condition", "seed"])[["delta_J", "delta_JF"]].mean().reset_index()
    cond_level = agg["all"]["overall"].set_index("condition").loc[CONDS]
    for s in ("all", "visible", "empty_gt"):
        for lvl, df in agg[s].items():
            df.to_csv(REP / f"agg_{s}_{lvl}.csv", index=False)
    runs.to_csv(REP / "run_summary.csv", index=False)
    paired.to_csv(REP / "paired_frames_all.csv", index=False)
    sig.to_csv(REP / "condition_delta_ci.csv", index=False)

    # Failure analysis (visible GT for J-based failures; empty GT for false positives).
    fail = {}
    for thr, nm in ((LOST, "J_lt_0.1"), (THR, "J_lt_0.5")):
        cl = paired.loc[~paired.gt_empty].assign(cls=failure_classes(paired, thr))
        fail[nm] = cl.groupby(["condition", "cls"]).size().unstack(fill_value=0).reindex(CONDS).fillna(0).astype(int)
        cl[cl.cls != "ok"].to_csv(REP / f"failure_frames_{nm}.csv", index=False)
    vis = paired[~paired.gt_empty]
    emp = paired[paired.gt_empty]
    fc = []
    for c in CONDS:
        vc, ec = vis[vis.condition == c], emp[emp.condition == c]
        fc.append({"condition": c, "visible_frames": len(vc),
                   "J_lt_0.1_drop": int((vc.J_drop < LOST).sum()), "J_lt_0.1_no_drop": int((vc.J_no_drop_matched < LOST).sum()),
                   "J_lt_0.5_drop": int((vc.J_drop < THR).sum()), "J_lt_0.5_no_drop": int((vc.J_no_drop_matched < THR).sum()),
                   "pred_empty_visible_drop": int((vc.pred_pixels == 0).sum()),
                   "pred_empty_visible_no_drop": int((vc.pred_pixels_no_drop == 0).sum()),
                   "empty_gt_frames": len(ec), "fp_on_empty_gt_drop": int((ec.pred_pixels > 0).sum()),
                   "fp_on_empty_gt_no_drop": int((ec.pred_pixels_no_drop > 0).sum())})
    fail_counts = pd.DataFrame(fc).set_index("condition")
    streaks = []
    for key, g in paired.sort_values("original_frame_id").groupby(["scene", "track_id", "condition", "seed"]):
        for who, col in (("drop", "J_drop"), ("no_drop", "J_no_drop_matched")):
            for L in failure_runs(g[col].tolist(), THR):
                streaks.append({"scene": key[0], "track_id": key[1], "condition": key[2], "seed": key[3], "run": who,
                                "length": L})
    streaks = pd.DataFrame(streaks, columns=["scene", "track_id", "condition", "seed", "run", "length"])
    streak_tab = streaks.groupby(["condition", "run"]).length.agg(["count", "max", "mean"]) if len(streaks) else None
    fail_counts.to_csv(REP / "failure_counts.csv")
    streaks.to_csv(REP / "failure_streaks_J_lt_0.5.csv", index=False)

    # Recovery (burst conditions) + censoring.
    b = ev[ev.condition.isin(BURSTS)].copy()
    rec_rows = []
    for c, g in b.groupby("condition"):
        deg = g[g.degraded_after_drop]
        det = deg[deg.within_n_determinable]
        lit = g[g.within_n_determinable]
        onset = pd.to_numeric(deg[deg["class"] == "recovered_later"].recovery_onset_received_k)
        censored = g["class"].isin(["insufficient_observation", "no_observation", "unrecovered_next_burst",
                                    "unrecovered_end"])
        rec_rows.append({
            "condition": c, "events": len(g), "window_ge_2": int((g.window_length >= 2).sum()),
            "window_ge_5": int((g.window_length >= 5).sum()), "window_median": float(g.window_length.median()),
            "intact_at_first_frame": int((g["class"] == "intact_at_first_frame").sum()),
            "degraded": len(deg), "lost": int(g.lost_after_drop.sum()),
            "recovered_later": int((g["class"] == "recovered_later").sum()),
            "degraded_recovered": int((deg["class"] == "recovered_later").sum()),
            "unrecovered_next_burst": int((g["class"] == "unrecovered_next_burst").sum()),
            "unrecovered_end": int((g["class"] == "unrecovered_end").sum()),
            "insufficient_observation": int((g["class"] == "insufficient_observation").sum()),
            "no_observation": int((g["class"] == "no_observation").sum()),
            "censored_fraction": float(censored.mean()),
            "censored_by_next_burst": int((censored & (g.censored_by == "next_drop")).sum()),
            "censored_by_clip_end": int((censored & (g.censored_by == "clip_end")).sum()),
            "onset_k_median": float(onset.median()) if len(onset) else None,
            "degraded_within5_denominator": len(det),
            "degraded_within5_rate": float(det.confirmed_within_n.astype(bool).mean()) if len(det) else None,
            "plan_literal_within5_denominator": len(lit),
            "plan_literal_within5_rate": float(lit.confirmed_within_n.astype(bool).mean()) if len(lit) else None,
            "visible_class_intact": int((g.visible_class == "intact_at_first_frame").sum()),
            "visible_class_recovered_later": int((g.visible_class == "recovered_later").sum()),
            "first_frame_gt_empty": int((g.first_frame_gt_visible == False).sum()),  # noqa: E712
            "degraded_where_no_drop_also_degraded": int(deg.nodrop_degraded_first.sum()),
            "nodrop_class_not_intact": int((g.nodrop_class != "intact_at_first_frame").sum()),
        })
    recovery = pd.DataFrame(rec_rows)
    recovery.to_csv(REP / "burst_recovery_summary.csv", index=False)
    b.to_csv(REP / "burst_events_all.csv", index=False)
    deg_events = b[b.degraded_after_drop | (b.visible_class != b["class"])]
    deg_events.to_csv(REP / "burst_events_degraded_or_visible_diff.csv", index=False)

    # Size / camera-motion relations (retrospective).
    size_fig, size_tab = fig_size(paired)
    size_tab.to_csv(REP / "delta_by_object_size.csv", index=False)
    v = paired[~paired.gt_empty]
    gap_tab = v.groupby(["condition", pd.cut(v.gap_original, [0, 1, 2, 4, 6, 100])], observed=True).agg(
        frames=("delta_J", "size"), dJ=("delta_J", "mean")).reset_index()
    cam_tab = v.groupby(pd.cut(v.cam_dT_m, [-0.01, 0.05, 1, 3, 6, 100]), observed=True).agg(
        frames=("delta_J", "size"), dJ=("delta_J", "mean"), dJ_min=("delta_J", "min")).reset_index()
    gap_tab.to_csv(REP / "delta_by_gap.csv", index=False)
    cam_tab.to_csv(REP / "delta_by_camera_motion.csv", index=False)
    track_size = v.groupby(["scene", "track_id"]).gt_pixels.median().rename("median_gt_px").reset_index().merge(
        agg["all"]["track"].groupby(["scene", "track_id"]).delta_J.mean().rename("mean_delta_J").reset_index())
    track_size.to_csv(REP / "track_size_vs_delta.csv", index=False)

    # Case studies (deterministic): worst / best 5-frame windows, near-zero, fixed cases.
    worst = worst_windows(paired, 5, 5, ascending=True)
    best = worst_windows(paired, 5, 5, ascending=False)
    worst.to_csv(REP / "case_worst_windows.csv", index=False)
    best.to_csv(REP / "case_best_windows.csv", index=False)
    figs = [fig_timelines(m2pf, paired, patterns), fig_delta_hist(paired, "J", "02_delta_J_distribution.png"),
            fig_delta_hist(paired, "JF", "03_delta_JF_distribution.png"), fig_condition(agg["all"], cis),
            fig_scene(agg["all"]), fig_seeds(runs), size_fig, fig_recovery(ev), fig_censoring(ev)]
    w = worst.iloc[0]
    figs.append(fig_case(ds, w.scene, int(w.track_id), w.condition, int(w.seed),
                         list(range(int(w.start_frame) - 2, int(w.end_frame) + 1)),
                         f"Worst paired window: {w.scene} t{w.track_id} {w.condition} s{w.seed}", "10a_case_worst.png"))
    bst = best.iloc[0]
    figs.append(fig_case(ds, bst.scene, int(bst.track_id), bst.condition, int(bst.seed),
                         list(range(int(bst.start_frame) - 2, int(bst.end_frame) + 1)),
                         f"Most improved window: {bst.scene} t{bst.track_id} {bst.condition} s{bst.seed}",
                         "10b_case_improved.png"))
    figs.append(fig_case(ds, "Scene01", 86, "burst_50", 0, list(range(376, 384)),
                         "Scene01 t86 frame 380 (no-drop loss) under burst_50 s0", "10c_case_t86_f380_burst50.png"))
    figs.append(fig_case(ds, "Scene01", 86, "random_50", 1, list(range(376, 384)),
                         "Scene01 t86 frame 380 under random_50 s1", "10d_case_t86_f380_random50.png"))
    figs.append(fig_case(ds, "Scene06", 12, "burst_50", 0, [89, 90, 91, 92, 95, 96, 97, 98],
                         "Scene06 t12 occlusion 92-95 and reappearance, burst_50 s0", "10e_case_t12_occlusion.png"))
    near0 = runs[(runs.subset == "all")].assign(a=lambda d: d.delta_J.abs()).sort_values(["a", "scene", "track_id"]).iloc[0]
    g = paired[(paired.scene == near0.scene) & (paired.track_id == near0.track_id) & (paired.condition == near0.condition)
               & (paired.seed == near0.seed)].original_frame_id.tolist()
    figs.append(fig_case(ds, near0.scene, int(near0.track_id), near0.condition, int(near0.seed), g[len(g) // 2:len(g) // 2 + 4],
                         f"Near-zero effect run: {near0.scene} t{near0.track_id} {near0.condition} s{near0.seed}",
                         "10f_case_near_zero.png"))

    # Runtime / storage, compared to M2 on the same tracks.
    m2rt = pd.read_csv(Path(M3["m2_output_dir"]) / "reports" / "runtime.csv")
    m2rt["m2_total_s"] = m2rt.init_s + m2rt.latency_mean_s * (m2rt.frames - 1) + m2rt.prompt_frame_s
    rt = recs.merge(m2rt[["scene", "track_id", "frames", "m2_total_s"]].rename(columns={"frames": "m2_frames"}),
                    on=["scene", "track_id"])
    rt["received_fraction"] = rt.received / rt.m2_frames
    rt["time_fraction_vs_m2"] = rt.wall_s / rt.m2_total_s
    runtime = {"runs": int(len(recs)), "total_inference_wall_s": float(recs.wall_s.sum()),
               "total_frames": int(recs.frames.sum()), "init_s_median": float(recs.init_s.median()),
               "init_s_max": float(recs.init_s.max()), "latency_median_of_run_medians_s": float(recs.latency_median_s.median()),
               "latency_steady_mean_s": float(recs.latency_steady_mean_s.mean()),
               "fps_steady": float(1 / recs.latency_steady_mean_s.mean()), "mps_peak_gb_max": float(recs.mps_peak_gb.max()),
               "system_free_percent_min": float(recs.system_free_percent_before.min()),
               "prediction_storage_mb": float(recs.pred_bytes.sum() / 1e6),
               "time_vs_received_fraction_ratio_median": float((rt.time_fraction_vs_m2 / rt.received_fraction).median()),
               "device": model["device"], "torch": model["torch"], "precision": model["precision"]}
    rt.to_csv(REP / "runtime_runs.csv", index=False)

    report = {"valid_runs": len(recs), "expected_runs": len(specs), "problems": problems, "model": model,
              "frames": {"paired_total": int(len(paired)), "visible": int((~paired.gt_empty).sum()),
                         "empty_gt": int(paired.gt_empty.sum())},
              "achieved_rate": rate.round(4).reset_index().to_dict("records"),
              "overall_all": cond_level.round(5).reset_index().to_dict("records"),
              "overall_visible": agg["visible"]["overall"].round(5).to_dict("records"),
              "overall_empty_gt": agg["empty_gt"]["overall"].round(5).to_dict("records"),
              "scene_all": agg["all"]["scene"].round(5).to_dict("records"),
              "delta_ci": sig.round(5).to_dict("records"), "seed_overall": seed_overall.round(5).to_dict("records"),
              "failure_counts": fail_counts.reset_index().to_dict("records"),
              "failure_classes": {k: v.reset_index().to_dict("records") for k, v in fail.items()},
              "failure_streaks": streak_tab.reset_index().to_dict("records") if streak_tab is not None else [],
              "recovery": recovery.to_dict("records"), "gap": gap_tab.astype({"gap_original": str}).to_dict("records"),
              "camera_motion": cam_tab.astype({"cam_dT_m": str}).to_dict("records"),
              "track_size": track_size.to_dict("records"),
              "worst_windows": worst.to_dict("records"), "best_windows": best.to_dict("records"),
              "near_zero_run": near0[["scene", "track_id", "condition", "seed", "delta_J"]].to_dict(),
              "runtime": runtime, "figures": figs}
    (REP / "m4_report.json").write_text(json.dumps(report, indent=2, default=lambda o: o.item()
                                                   if isinstance(o, np.generic) else str(o)))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("achieved rate\n", rate.round(4))
    print("overall (mean of 8 tracks, all matched frames)\n", cond_level[[c for c in cond_level.columns]].round(4))
    print("overall visible\n", agg["visible"]["overall"].round(4))
    print("overall empty_gt\n", agg["empty_gt"]["overall"].round(4))
    print("scene\n", agg["all"]["scene"][["scene", "condition", "JF_no_drop_matched", "JF_drop", "delta_J", "delta_F",
                                         "delta_JF", "delta_F_strict", "frames"]].round(4))
    print("CIs\n", sig.round(4))
    print("seed overall\n", seed_overall.round(4))
    print("failure counts\n", fail_counts)
    for k, t in fail.items():
        print(k, "\n", t)
    print("streaks\n", streak_tab)
    print("recovery\n", recovery.T)
    print("gap\n", gap_tab)
    print("camera\n", cam_tab)
    print("size\n", size_tab)
    print("track size\n", track_size)
    print("worst\n", worst)
    print("best\n", best)
    print("runtime", json.dumps(runtime, indent=1))


if __name__ == "__main__":
    main()
