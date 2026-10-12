"""Milestone 1: do the four memory policies pick different memories? (no frame drops, no SAM2)

    uv run python scripts/analyze_memory_selection.py [--config configs/m1_memory.yaml]

For every selected track of every DEV scene, stream frames prompt_frame..last_frame, ask each policy
for its picks, then add the current frame to the bank (as if predicted). Writes CSV/JSON summaries
and figures to outputs/milestone1/.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from memsel.selector import (BINS, POLICIES, Memory, MemoryBank, MemorySelector, age_bin,  # noqa: E402
                             estimate_pose_scale, pose_cost, pose_difference)
from vkitti.dataset import VKittiDataset  # noqa: E402
from vkitti.tracks import select_tracks, summarize_tracks  # noqa: E402
from vkitti.validation import camera_trajectory  # noqa: E402

COLORS = dict(zip(POLICIES, ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]))
BIN_NAMES = [f"{lo}-{hi}" if hi else f"{lo}+" for lo, hi in BINS]


def camera_memories(ds) -> dict[int, Memory]:
    out = {}
    for f in ds.frame_ids:
        c = ds.camera(f)
        out[f] = Memory(f, c["camera_position"], c["camera_rotation"])
    return out


def run_scene(ds, mems, tracks, selectors, scale, L):
    """Rows: one per (track, current frame, policy, chosen memory); plus per-frame bin stats."""
    picks, frames = [], []
    for tr in tracks.itertuples():
        bank = MemoryBank(mems[tr.prompt_frame], L)
        for t in range(tr.prompt_frame + 1, tr.last_frame + 1):
            cur = mems[t]
            pool = [c for c in bank.candidates if c is not bank.recent]
            bins = np.bincount([age_bin(t - c.original_frame_id) for c in pool], minlength=len(BINS))
            costs = [pose_cost(cur, c, scale) for c in pool]
            ages = [t - c.original_frame_id for c in pool]
            assert sorted(ages) == list(range(2, len(ages) + 2)), "no-drop stream must give a contiguous pool"
            diffs = np.array([pose_difference(cur, c) for c in pool]).reshape(-1, 2)
            rho = spearman(ages, costs) if len(pool) >= 5 else np.nan
            frames.append({"scene": ds.scene, "track_id": tr.track_id, "current": t, "pool": len(pool),
                           "full_bank": len(bank.candidates) == L, "age_cost_spearman": rho,
                           "pool_dT_max": diffs[:, 0].max(initial=0), "pool_dR_max": diffs[:, 1].max(initial=0),
                           **{f"bin_{n}": int(k) for n, k in zip(BIN_NAMES, bins)}})
            for p, sel in selectors.items():
                for m in sel.select(cur, bank).chosen:
                    dt, dr = pose_difference(cur, m)
                    picks.append({"scene": ds.scene, "track_id": tr.track_id, "current": t, "policy": p,
                                  "memory": m.original_frame_id, "age": t - m.original_frame_id,
                                  "bin": BIN_NAMES[age_bin(t - m.original_frame_id)],
                                  "dT_m": dt, "dR_deg": dr, "cost": pose_cost(cur, m, scale)})
            bank.add(cur)  # only after every policy has selected for frame t
    return pd.DataFrame(picks), pd.DataFrame(frames)


def spearman(a, b) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def pick_sets(picks: pd.DataFrame) -> pd.DataFrame:
    """index (scene, track_id, current), columns policy -> frozenset of chosen frames."""
    s = picks.groupby(["scene", "track_id", "current", "policy"]).memory.agg(frozenset).unstack("policy")
    return s.apply(lambda col: col.map(lambda v: v if isinstance(v, frozenset) else frozenset()))


def overlap(a: frozenset, b: frozenset) -> float:
    n = max(len(a), len(b))
    return len(a & b) / n if n else np.nan


def overlap_matrix(sets: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame(1.0, index=list(POLICIES), columns=list(POLICIES))
    for p, q in combinations(POLICIES, 2):
        m.loc[p, q] = m.loc[q, p] = np.nanmean([overlap(a, b) for a, b in zip(sets[p], sets[q])])
    return m


def within_bin_comparison(picks: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """time-distributed vs pose+dt, bin by bin, on frames where the bin has candidates.

    Both policies take one memory per non-empty bin, so any difference inside a bin comes from the
    pose cost alone. chance_agreement = E[1/m] = agreement if pose+dt picked uniformly in the bin.
    """
    key = ["scene", "track_id", "current", "bin"]
    td = picks[picks.policy == "time-distributed"].drop_duplicates(key).set_index(key)
    pd_ = picks[picks.policy == "pose+dt"].drop_duplicates(key).set_index(key)
    j = td.join(pd_, lsuffix="_td", rsuffix="_pdt", how="inner").reset_index()
    fr = frames.melt(id_vars=["scene", "track_id", "current"], value_vars=[f"bin_{n}" for n in BIN_NAMES],
                     var_name="bin", value_name="n_cand")
    fr["bin"] = fr.bin.str[4:]
    j = j.merge(fr, on=["scene", "track_id", "current", "bin"])
    j["same"] = j.memory_td == j.memory_pdt
    # Recency control: is the pose+dt pick simply the youngest candidate of its bin? (no drops:
    # the pool is contiguous, so the youngest age in a bin is max(lo, 2) -- age 1 is the fixed recent.)
    lo = {n: max(b[0], 2) for n, b in zip(BIN_NAMES, BINS)}
    j["pdt_youngest"] = j.age_pdt == j.bin.map(lo)
    rows = []
    for (scene, b), g in j.groupby(["scene", "bin"]):
        rows.append({"scene": scene, "bin": b, "frames": len(g), "mean_candidates": g.n_cand.mean(),
                     "agreement": g.same.mean(), "chance_agreement": (1 / g.n_cand).mean(),
                     "pdt_is_youngest_in_bin": g.pdt_youngest.mean(),
                     "cost_td": g.cost_td.mean(), "cost_pdt": g.cost_pdt.mean(),
                     "dT_td": g.dT_m_td.mean(), "dT_pdt": g.dT_m_pdt.mean(),
                     "dR_td": g.dR_deg_td.mean(), "dR_pdt": g.dR_deg_pdt.mean()})
    out = pd.DataFrame(rows)
    out["bin"] = pd.Categorical(out.bin, BIN_NAMES)
    return out.sort_values(["scene", "bin"]).reset_index(drop=True)


def camera_state(traj: pd.DataFrame, cfg) -> pd.Series:
    st = np.where(traj.step_m < cfg["stopped_step_m"], "stopped",
                  np.where(traj.step_rot_deg > cfg["turning_step_deg"], "turning", "straight"))
    return pd.Series(st, index=traj.frame)


# ----------------------------------------------------------------------------- figures
def fig_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def plot_trajectory_with_picks(scene, traj, mems, example, picks, out):
    t, tid = example["current"], example["track_id"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=True, sharey=True)
    lo = max(int(traj.frame.min()), t - 40)
    win = traj[(traj.frame >= lo) & (traj.frame <= t)]
    for ax, p in zip(axes, POLICIES):
        ax.plot(win.x, win.z, "-", color="0.8", zorder=0)
        ax.scatter(win.x, win.z, c="0.75", s=8, label="other frames")
        ch = picks[(picks.track_id == tid) & (picks.current == t) & (picks.policy == p)].memory
        pts = np.array([mems[f].camera_position for f in ch])
        ax.scatter(pts[:, 0], pts[:, 2], c=COLORS[p], s=50, label="selected", zorder=3)
        for f, (x, _, z) in zip(ch, pts):
            ax.annotate(str(f), (x, z), fontsize=7, xytext=(3, 3), textcoords="offset points")
        c = mems[t].camera_position
        r = mems[t - 1].camera_position
        ax.scatter([r[0]], [r[2]], marker="s", c="k", s=40, label=f"recent {t - 1}")
        ax.scatter([c[0]], [c[2]], marker="*", c="gold", edgecolors="k", s=200, label=f"current {t}", zorder=4)
        ax.set(title=p, xlabel="world x [m]", aspect="equal")
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("world z [m]")
    fig.suptitle(f"{scene} track {tid}: camera positions of selected memories at frame {t} (last 40 frames shown)")
    return fig_save(fig, out / f"{scene}_trajectory_selected_frame{t:05d}.png")


def plot_current_vs_selected(scene, picks, track, prompt, out):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    d = picks[picks.track_id == track]
    for ax, p in zip(axes, POLICIES):
        g = d[d.policy == p]
        ax.scatter(g.current, g.memory, s=3, c=COLORS[p])
        x = np.arange(d.current.min(), d.current.max() + 1)
        ax.plot(x, x - 1, "k-", lw=0.6, label="recent (fixed)")
        ax.axhline(prompt, color="0.5", ls="--", lw=0.8, label="prompt (fixed)")
        ax.set(title=p, xlabel="current original frame")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("selected memory original frame")
    fig.suptitle(f"{scene} track {track}: selected memory frames per current frame (5 policy slots)")
    return fig_save(fig, out / f"{scene}_track{track:03d}_current_vs_selected.png")


def plot_overlap(mats: dict, out):
    fig, axes = plt.subplots(1, len(mats), figsize=(7.5 * len(mats), 5.5), gridspec_kw={"wspace": 0.55})
    for ax, (name, m) in zip(np.atleast_1d(axes), mats.items()):
        im = ax.imshow(m.values, vmin=0, vmax=1, cmap="viridis")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=POLICIES, yticklabels=POLICIES, title=name)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center",
                        color="w" if m.values[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean overlap of the 5 policy picks")
    fig.savefig(out / "policy_overlap_matrix.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out / "policy_overlap_matrix.png")


def plot_age_hist(picks, out):
    scenes = sorted(picks.scene.unique())
    fig, axes = plt.subplots(len(scenes), 4, figsize=(18, 3.6 * len(scenes)), sharex=True, squeeze=False)
    for r, s in enumerate(scenes):
        for c, p in enumerate(POLICIES):
            g = picks[(picks.scene == s) & (picks.policy == p)]
            axes[r, c].hist(g.age, bins=np.arange(0.5, 33.5, 1), color=COLORS[p])
            axes[r, c].set(title=f"{s} {p} (median {g.age.median():.0f})", xlabel="memory age [original frames]")
    return fig_save(fig, out / "memory_age_distribution.png")


def plot_pose_diff(picks, out):
    scenes = sorted(picks.scene.unique())
    fig, axes = plt.subplots(len(scenes), 3, figsize=(16, 3.8 * len(scenes)), squeeze=False)
    for r, s in enumerate(scenes):
        for c, (col, lab) in enumerate([("dT_m", "translation diff [m]"), ("dR_deg", "rotation diff [deg]"),
                                        ("cost", "pose cost C")]):
            data = [picks[(picks.scene == s) & (picks.policy == p)][col] for p in POLICIES]
            bp = axes[r, c].boxplot(data, tick_labels=POLICIES, showfliers=False, patch_artist=True)
            for patch, p in zip(bp["boxes"], POLICIES):
                patch.set_facecolor(COLORS[p])
            axes[r, c].set(title=f"{s}: {lab}")
            plt.setp(axes[r, c].get_xticklabels(), rotation=20)
    return fig_save(fig, out / "selected_memory_pose_difference.png")


def plot_scene_comparison(summary: pd.DataFrame, out):
    metrics = ["overlap_recentK_poseonly", "overlap_timedist_posedt", "identical_recentK_poseonly",
               "identical_timedist_posedt"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(summary)
    for i, (scene, row) in enumerate(summary.iterrows()):
        ax.bar(np.arange(len(metrics)) + i * w, row[metrics], w, label=scene)
    ax.set(xticks=np.arange(len(metrics)) + w * (len(summary) - 1) / 2, ylim=(0, 1),
           ylabel="fraction (full-bank frames)", title="Policy agreement by development scene")
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.axhline(0.95, color="r", ls=":", lw=1, label="95% (research-process doc criterion)")
    ax.legend()
    return fig_save(fig, out / "scene_policy_comparison.png")


def plot_rgb_strip(ds, picks, example, out):
    t, tid = example["current"], example["track_id"]
    fig, axes = plt.subplots(4, 6, figsize=(24, 6.2))
    for r, p in enumerate(POLICIES):
        ch = sorted(picks[(picks.track_id == tid) & (picks.current == t) & (picks.policy == p)].memory)
        for c in range(6):
            ax = axes[r, c]
            ax.axis("off")
            f = t if c == 0 else (ch[c - 1] if c - 1 < len(ch) else None)
            if f is None:
                continue
            ax.imshow(ds.load_rgb(f)[::2, ::2])
            ax.set_title(("CURRENT " if c == 0 else f"{p}: ") + f"frame {f}" + ("" if c == 0 else f" (age {t - f})"),
                         fontsize=8)
    fig.suptitle(f"{ds.tag} track {tid}: current RGB and the 5 policy-selected memory frames")
    return fig_save(fig, out / f"{ds.scene}_rgb_current_vs_selected_frame{t:05d}.png")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m1_memory.yaml")
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    if set(cfg["dev_scenes"]) & set(cfg["test_scenes"]):
        raise SystemExit("dev and test scenes overlap")
    out = Path(cfg["output_dir"])
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    mc, sel_cfg = cfg["memory"], cfg["track_selection"]

    data = {}
    for scene in cfg["dev_scenes"]:
        ds = VKittiDataset(cfg["root"], scene, cfg["variation"], cfg["camera_id"])
        H, W = ds.image_shape
        summary = summarize_tracks(ds.mask_pixel_counts(), ds.info, ds.bbox, H * W, sel_cfg["min_area_frac"])
        tracks = select_tracks(summary, H * W, **sel_cfg)
        data[scene] = (ds, camera_memories(ds), tracks)

    # One pose scale for all dev scenes (pooled), fixed before any SAM2 result exists.
    scale = estimate_pose_scale([list(m.values()) for _, m, _ in data.values()], mc["L"], mc["scale_percentile"], mc["w_t"])
    per_scene_scale = {s: estimate_pose_scale([list(m.values())], mc["L"], mc["scale_percentile"])
                       for s, (_, m, _) in data.items()}
    selectors = {p: MemorySelector(p, scale, K=mc["K"]) for p in POLICIES}

    all_picks, all_frames, report, figures = [], [], {}, []
    for scene, (ds, mems, tracks) in data.items():
        picks, frames = run_scene(ds, mems, tracks, selectors, scale, mc["L"])
        traj = camera_trajectory(ds)
        frames["camera_state"] = frames.current.map(camera_state(traj, cfg["camera_state"]))
        all_picks.append(picks)
        all_frames.append(frames)

        sets = pick_sets(picks).join(frames.set_index(["scene", "track_id", "current"])[["full_bank", "camera_state"]])
        full = sets[sets.full_bank]
        mat_all, mat_full = overlap_matrix(sets), overlap_matrix(full)
        ov = lambda df, p, q: float(np.nanmean([overlap(a, b) for a, b in zip(df[p], df[q])]))  # noqa: E731
        same = lambda df, p, q: float(np.mean([a == b for a, b in zip(df[p], df[q])]))  # noqa: E731
        f_full = frames[frames.full_bank]
        pool_max = f_full.groupby("camera_state")[["pool_dT_max", "pool_dR_max"]].median()
        by_state = {st: {"frames": int(len(g)), "overlap_recentK_poseonly": ov(g, "recent-K", "pose-only"),
                         "overlap_timedist_posedt": ov(g, "time-distributed", "pose+dt"),
                         "median_pool_dT_max_m": float(pool_max.loc[st, "pool_dT_max"]),
                         "median_pool_dR_max_deg": float(pool_max.loc[st, "pool_dR_max"])}
                    for st, g in full.groupby("camera_state")}
        report[scene] = {
            "frames": len(ds), "tracks": tracks[["track_id", "label", "first_visible_frame", "prompt_frame",
                                                 "prompt_area", "observable_frames", "last_frame", "motion"]]
                                         .to_dict("records"),
            "current_frames_analysed": int(len(frames)), "full_bank_frames": int(frames.full_bank.sum()),
            "overlap_all_frames": mat_all.round(4).to_dict(), "overlap_full_bank": mat_full.round(4).to_dict(),
            "identical_full_bank": {f"{p}|{q}": same(full, p, q) for p, q in combinations(POLICIES, 2)},
            "overlap_by_camera_state_full_bank": by_state,
            "age_cost_spearman_full_bank": {"mean": float(f_full.age_cost_spearman.mean()),
                                            "nan_frames": int(f_full.age_cost_spearman.isna().sum())},
            "bins_full_bank": {n: {"mean_candidates": float(f_full[f"bin_{n}"].mean()),
                                   "empty_frac": float((f_full[f"bin_{n}"] == 0).mean())} for n in BIN_NAMES},
            "bins_all_frames_empty_frac": {n: float((frames[f"bin_{n}"] == 0).mean()) for n in BIN_NAMES},
            "selected_stats": picks.groupby("policy")[["age", "dT_m", "dR_deg", "cost"]].median().round(4).to_dict(),
            "own_scene_scale": {"s_t": per_scene_scale[scene].s_t, "s_r_deg": per_scene_scale[scene].s_r_deg},
        }

        # Example frame: full-bank frame (any selected track) where recent-K and pose-only differ most;
        # ties -> first in (track rank, frame) order.
        top = int(tracks.track_id.iloc[0])
        cand = full.loc[scene].copy()
        cand["ov"] = [overlap(a, b) for a, b in zip(cand["recent-K"], cand["pose-only"])]
        rank = {t: i for i, t in enumerate(tracks.track_id)}
        cand = cand.reset_index().assign(r=lambda d: d.track_id.map(rank)).sort_values(["ov", "r", "current"])
        example = {"track_id": int(cand.track_id.iloc[0]), "current": int(cand.current.iloc[0]),
                   "overlap_recentK_poseonly": float(cand.ov.iloc[0])}
        report[scene]["example_frame"] = example
        figures += [plot_trajectory_with_picks(scene, traj, mems, example, picks, figs),
                    plot_current_vs_selected(scene, picks, top, int(tracks.prompt_frame.iloc[0]), figs),
                    plot_rgb_strip(ds, picks, example, figs)]
        report[scene]["_mats"] = mat_full  # popped before JSON

    picks = pd.concat(all_picks, ignore_index=True)
    frames = pd.concat(all_frames, ignore_index=True)
    within = within_bin_comparison(picks, frames[frames.full_bank])
    mats = {f"{s} (full bank)": report[s].pop("_mats") for s in report}
    summary = pd.DataFrame({s: {
        "overlap_recentK_poseonly": r["overlap_full_bank"]["recent-K"]["pose-only"],
        "overlap_timedist_posedt": r["overlap_full_bank"]["time-distributed"]["pose+dt"],
        "identical_recentK_poseonly": r["identical_full_bank"]["recent-K|pose-only"],
        "identical_timedist_posedt": r["identical_full_bank"]["time-distributed|pose+dt"]} for s, r in report.items()}).T
    figures += [plot_overlap(mats, figs), plot_age_hist(picks, figs), plot_pose_diff(picks, figs),
                plot_scene_comparison(summary, figs)]

    picks.to_csv(out / "reports" / "m1_picks.csv", index=False)
    frames.to_csv(out / "reports" / "m1_frames.csv", index=False)
    within.to_csv(out / "reports" / "m1_within_bin_td_vs_posedt.csv", index=False)
    summary.to_csv(out / "reports" / "m1_scene_summary.csv")
    meta = {"pose_scale_pooled_dev": {"s_t": scale.s_t, "s_r_deg": scale.s_r_deg, "w_t": scale.w_t},
            "K": mc["K"], "L": mc["L"], "bins": BIN_NAMES, "scenes": report, "figures": figures}
    (out / "reports" / "m1_report.json").write_text(json.dumps(meta, indent=2, default=lambda o: o.item()
                                                               if isinstance(o, np.generic) else str(o)))

    print(f"pose scale (pooled dev, p{mc['scale_percentile']}): sT={scale.s_t:.3f} m, sR={scale.s_r_deg:.3f} deg")
    for s, r in report.items():
        print(f"\n== {s}: {r['current_frames_analysed']} current frames ({r['full_bank_frames']} full bank), "
              f"tracks {[t['track_id'] for t in r['tracks']]}")
        print("overlap (full bank):\n" + pd.DataFrame(r["overlap_full_bank"]).round(3).to_string())
        print("identical sets:", {k: round(v, 3) for k, v in r["identical_full_bank"].items()})
        print("by camera state:", json.dumps(r["overlap_by_camera_state_full_bank"], default=float))
        print("age-cost spearman:", r["age_cost_spearman_full_bank"], "own-scene scale:", r["own_scene_scale"])
    print("\nwithin-bin time-distributed vs pose+dt (full bank):\n" + within.round(3).to_string(index=False))
    print("\nfigures:\n  " + "\n  ".join(figures))


if __name__ == "__main__":
    main()
