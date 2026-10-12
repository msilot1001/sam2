"""Milestone 4.5 feasibility analyses (no SAM 2 inference here):

  A  memory availability at every dropout-induced failure (M4 + stress runs)
  C  selection opportunity of the 4 policies + recent-per-bin on every real dropout manifest
  D  geometry-only return-visit study on synthetic camera paths (L = 16/32/64/128)
  N  pose-noise tolerance sweep (real dev sequences + synthetic jitter)

    uv run python scripts/m45_feasibility.py

Outputs: outputs/milestone45/{reports,figures,geometry}. Uses only camera poses (extrinsic.txt),
frame ids, manifests and saved per-frame J for retrospective stratification.
"""
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from baseline import frame_drops as fd  # noqa: E402
from baseline.dropout_runner import experiment_matrix, manifest_path, run_dir  # noqa: E402
from memsel import synthetic as syn  # noqa: E402
from memsel.opportunity import compare, overlap, simulate  # noqa: E402
from memsel.selector import (BINS, DIAGNOSTIC_POLICIES, POLICIES, Memory, MemoryBank, MemorySelector,  # noqa: E402
                             PoseScale, estimate_pose_scale, pose_difference)
from vkitti.dataset import VKittiDataset  # noqa: E402

OUT = Path("outputs/milestone45")
REP, FIG, GEO = OUT / "reports", OUT / "figures", OUT / "geometry"
ALL_POLICIES = POLICIES + DIAGNOSTIC_POLICIES
K = 7
TRACKS = {"Scene01": [90, 86, 63, 54], "Scene06": [12, 9, 11, 13]}
M3 = yaml.safe_load(Path("configs/m3_frame_drop.yaml").read_text())
ST = yaml.safe_load(Path("configs/m45_stress.yaml").read_text())
FAMILIES = {  # name: (conditions, seeds, manifests dir, runs dir)
    "M4": (list(M3["conditions"]), M3["seeds"], Path("outputs/milestone3/manifests"), Path("outputs/milestone4")),
    "stress": (list(ST["conditions"]), ST["seeds"], Path(ST["manifests_dir"]), Path("outputs/milestone45/stress_runs")),
}
TOLERANCES = [(0.0, 0.0), (0.005, 0.01), (0.01, 0.05), (0.05, 0.1), (0.10, 0.25), (0.25, 0.5)]  # (m, deg)
REVISIT = {"max_dT_m": 2.0, "max_dR_deg": 10.0, "min_age": 21}


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def camera_states(ds) -> dict:
    out, prev = {}, None
    for f in ds.frame_ids:
        c = ds.camera(f)
        if prev is None:
            out[f] = "start"
        else:
            dt, dr = pose_difference(Memory(f, c["camera_position"], c["camera_rotation"]), prev)
            out[f] = "stopped" if dt < 0.05 else ("turning" if dr > 0.5 else "straight")
        prev = Memory(f, c["camera_position"], c["camera_rotation"])
    return out


def load_frames(runs_root: Path, spec) -> pd.DataFrame | None:
    d = run_dir(runs_root, spec)
    if not (d / "record.json").exists():
        return None
    return pd.read_csv(d / "paired.csv")


# ----------------------------------------------------------------------------- PART A
def failure_audit(mems_by_scene, selectors, ds_by_scene):
    """Every frame with J_drop < 0.5 while matched no-drop J >= 0.5 (a NEW failure)."""
    rows = []
    for fam, (conds, seeds, man, runs) in FAMILIES.items():
        for spec in experiment_matrix(TRACKS, conds, seeds):
            pr = load_frames(runs, spec)
            if pr is None:
                continue
            new = pr[(~pr.gt_empty) & (pr.J_drop < 0.5) & (pr.J_no_drop_matched >= 0.5)]
            if new.empty:
                continue
            p = fd.load(manifest_path(man, spec))
            rec = list(p.received_ids)
            sim = simulate(mems_by_scene[spec.scene], rec, selectors).set_index("original_frame_id")
            mems = mems_by_scene[spec.scene]
            fail_ids = set(pr[(~pr.gt_empty) & (pr.J_drop < 0.5)].original_frame_id)
            for f in new.original_frame_id:
                i = rec.index(f)
                # onset = first frame of the consecutive (received-order) failure streak containing f
                j = i
                while j - 1 >= 1 and rec[j - 1] in fail_ids:
                    j -= 1
                onset = rec[j]
                s = sim.loc[f]
                bank_ids = [x for x in rec[max(1, i - 32):i]]               # L=32 FIFO of received non-prompt
                sam2_ids = rec[max(1, i - 6):i]                               # official: last 6 received
                outside = [x for x in bank_ids if x not in sam2_ids and x < onset]
                jmap = dict(zip(pr.original_frame_id, pr.J_drop))
                clean_default = [x for x in sam2_ids if x < onset and jmap.get(x, 0) >= 0.5]
                clean_outside = [x for x in outside if jmap.get(x, 0) >= 0.5]
                dT_p, dR_p = pose_difference(mems[f], mems[rec[0]])
                dT_r, dR_r = pose_difference(mems[f], mems[rec[i - 1]])
                differ = {f"{a}!={b}": set(s[a]) != set(s[b]) for a, b in
                          [("recent-K", "pose-only"), ("recent-per-bin", "pose+dt"), ("time-distributed", "pose+dt")]}
                clean_frac = {f"clean_frac|{name}": (np.mean([jmap.get(x, 0) >= 0.5 for x in s[name]]) if len(s[name]) else np.nan)
                              for name in ALL_POLICIES}
                row = pr.set_index("original_frame_id").loc[f]
                rows.append({
                    "family": fam, "scene": spec.scene, "track_id": spec.track_id, "condition": spec.condition,
                    "seed": spec.seed, "original_frame_id": f, "video_idx": i, "gap": f - rec[i - 1],
                    "failure_onset": onset, "J_drop": row.J_drop, "J_no_drop": row.J_no_drop_matched,
                    "gt_pixels": row.gt_pixels,
                    "noncond_available": i - 1, "in_L32_bank": len(bank_ids), "sam2_active_spatial": 1 + min(i - 1, 6),
                    "prompt_memory_available": True, "sam2_uses_all_available": i - 1 <= 6,
                    "pre_onset_candidates_outside_sam2_set": len(outside),
                    "clean_memories_in_sam2_set": len(clean_default),     # pre-onset, J_drop >= 0.5
                    "clean_candidates_outside_sam2_set": len(clean_outside),
                    "selection_can_change_set": s["selection_possible"],
                    "cam_dT_since_prompt": dT_p, "cam_dR_since_prompt": dR_p,
                    "cam_dT_since_prev_received": dT_r, "cam_dR_since_prev_received": dR_r,
                    **differ, **clean_frac,
                    "verdict": ("unaddressable: SAM 2 already attends every available memory"
                                if i - 1 <= 6 else
                                "unaddressable: no clean pre-failure memory outside the default set"
                                if not clean_outside else
                                "default set already holds clean memories; selection could only swap in other clean ones"
                                if clean_default else
                                "potentially addressable: clean memories exist only outside the default set"),
                })
    return pd.DataFrame(rows)


def case_timeline(mems, man, spec, runs, selectors, frames):
    p = fd.load(manifest_path(man, spec))
    rec = list(p.received_ids)
    sim = simulate(mems, rec, selectors).set_index("original_frame_id")
    pr = load_frames(runs, spec).set_index("original_frame_id")
    nd = pd.read_csv("outputs/milestone2/reports/per_frame_metrics.csv")
    nd = nd[(nd.scene == spec.scene) & (nd.track_id == spec.track_id)].set_index("original_frame_id")
    rows = []
    for f in frames:
        dropped = f not in rec
        r = {"original_frame_id": f, "received": not dropped, "gt_pixels": int(nd.loc[f, "gt_pixels"]),
             "J_no_drop": float(nd.loc[f, "J"])}
        if not dropped:
            i = rec.index(f)
            s = sim.loc[f] if f in sim.index else None
            dT_p, dR_p = pose_difference(mems[f], mems[rec[0]])
            r |= {"video_idx": i, "gap": (f - rec[i - 1]) if i else None,
                  "J_drop": float(pr.loc[f, "J_drop"]) if f in pr.index else None,
                  "noncond_available": max(i - 1, 0), "in_L32_bank": min(max(i - 1, 0), 32),
                  "sam2_active_spatial": 1 + min(max(i - 1, 0), 6), "prompt_memory_available": True,
                  "selection_possible": bool(s["selection_possible"]) if s is not None else False,
                  "cam_dT_since_prompt": round(dT_p, 3), "cam_dR_since_prompt": round(dR_p, 3),
                  "sam2_memories_original_ids": rec[max(1, i - 6):i] if i else [],
                  "recent-K": s["recent-K"] if s is not None else (), "pose-only": s["pose-only"] if s is not None else (),
                  "pose+dt": s["pose+dt"] if s is not None else ()}
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- PART D
SYNTH = {  # name: (poses, turn radius for plausibility, speed m/frame, note)
    "straight_v1.0": (syn.straight(200, 1.0), None, 1.0),
    "turn_R15_v0.75": (syn.turn(0.75, 15.0), 15.0, 0.75),
    "stationary_exact": (syn.stationary(200), None, 0.0),
    "stationary_jitter": (syn.stationary(200), None, 0.0),
    "uturn_R6_v0.45": (syn.u_turn(0.45, 6.0, 60), 6.0, 0.45),
    "loop_R5.1_v0.45": (syn.loop(0.45, 5.1, 2.5, 10), 5.1, 0.45),
    "loop_R15_v0.77": (syn.loop(0.77, 15.0, 2.2, 10), 15.0, 0.77),
    "loop_R5_v1.0_IMPLAUSIBLE": (syn.loop(1.0, 5.0, 3.0, 10), 5.0, 1.0),
}


def revisit_lags(mems: list[Memory]) -> list[int | None]:
    """For each frame, age of the NEWEST earlier frame that is a 'revisit' (old + similar pose)."""
    out = []
    for i, cur in enumerate(mems):
        lag = None
        for j in range(i - REVISIT["min_age"], -1, -1):
            dt, dr = pose_difference(cur, mems[j])
            if dt <= REVISIT["max_dT_m"] and dr <= REVISIT["max_dR_deg"]:
                lag = i - j
                break
        out.append(lag)
    return out


def geometry_study(scale):
    rows, curves, lag_rows = [], [], []
    for name, (poses, radius, v) in SYNTH.items():
        jit = name == "stationary_jitter"
        mems = syn.to_memories(poses, jitter_t=0.005 if jit else 0.0, jitter_r_deg=0.02 if jit else 0.0, seed=0)
        pl = syn.plausibility(v, radius)
        lags = revisit_lags(mems)
        lag_rows += [{"trajectory": name, "frame": i, "revisit_lag": g} for i, g in enumerate(lags)]
        md = {m.original_frame_id: m for m in mems}
        rec = [m.original_frame_id for m in mems]
        for L in (16, 32, 64, 128):
            sels = {p: MemorySelector(p, scale) for p in ALL_POLICIES}
            sim = simulate(md, rec, sels, L)
            ov1, id1 = compare(sim, "recent-K", "pose-only")
            ov2, id2 = compare(sim, "recent-per-bin", "pose+dt")
            lag_in_L = [g is not None and g <= L - 1 for g in lags[1:]]
            rows.append({"trajectory": name, "L": L, "speed_mps@10Hz": pl["speed_mps"],
                         "lateral_acc@10Hz": pl["lateral_acc"], "plausible": pl["plausible"],
                         "frames": len(sim), "frames_with_revisit_anywhere": int(sum(g is not None for g in lags[1:])),
                         "frames_with_revisit_inside_L": int(sum(lag_in_L)),
                         "min_revisit_lag": min((g for g in lags if g is not None), default=None),
                         "overlap_recentK_poseonly": ov1, "identical_recentK_poseonly": id1,
                         "overlap_recentperbin_posedt": ov2, "identical_recentperbin_posedt": id2,
                         "pose_only_mean_age": float(sim["pose-only|age"].mean()),
                         "recent_K_mean_age": float(sim["recent-K|age"].mean())})
            curves.append(sim.assign(trajectory=name, L=L)[["trajectory", "L", "original_frame_id", "recent-K",
                                                            "pose-only", "pose+dt", "recent-per-bin"]])
    return pd.DataFrame(rows), pd.concat(curves), pd.DataFrame(lag_rows)


def plot_paths(scale):
    names = list(SYNTH)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for ax, name in zip(axes.ravel(), names):
        poses, radius, v = SYNTH[name]
        jit = name == "stationary_jitter"
        mems = syn.to_memories(poses, jitter_t=0.005 if jit else 0.0, jitter_r_deg=0.02 if jit else 0.0, seed=0)
        md = {m.original_frame_id: m for m in mems}
        rec = [m.original_frame_id for m in mems]
        t = int(len(rec) * 0.85)
        xs = [m.camera_position[0] for m in mems]
        zs = [m.camera_position[2] for m in mems]
        ax.plot(xs, zs, "-", color="0.8")
        for L, mk, col in ((32, "o", "#C44E52"), (128, "s", "#4C72B0")):
            sim = simulate(md, rec[: t + 1], {"pose-only": MemorySelector("pose-only", scale)}, L)
            ch = sim.iloc[-1]["pose-only"]
            ax.scatter([md[i].camera_position[0] for i in ch], [md[i].camera_position[2] for i in ch], marker=mk,
                       facecolors="none", edgecolors=col, s=70, label=f"pose-only picks, L={L} (ages {sorted(t - i for i in ch)})")
        c = md[rec[t]].camera_position
        ax.scatter([c[0]], [c[2]], marker="*", s=200, c="gold", edgecolors="k", zorder=5, label=f"current frame {t}")
        ax.set(title=name, xlabel="x [m]", ylabel="z [m]")
        if max(xs) - min(xs) > 1:
            ax.set_aspect("equal")
        else:  # straight / stationary: avoid a degenerate equal-aspect axis
            ax.set_xlim(min(xs) - 0.05 - 10 * (max(zs) - min(zs) > 1), max(xs) + 0.05 + 10 * (max(zs) - min(zs) > 1))
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("SYNTHETIC geometry only (no images): camera paths and pose-only selections at one frame")
    return save(fig, FIG / "D1_synthetic_paths_selected_memories.png")


# ----------------------------------------------------------------------------- main
def main():
    for d in (REP, FIG, GEO):
        d.mkdir(parents=True, exist_ok=True)
    ds = {s: VKittiDataset("data/vkitti2", s) for s in TRACKS}
    mems = {s: {f: Memory(f, d.camera(f)["camera_position"], d.camera(f)["camera_rotation"]) for f in d.frame_ids}
            for s, d in ds.items()}
    cam = {s: camera_states(d) for s, d in ds.items()}
    scale = estimate_pose_scale([list(m.values()) for m in mems.values()], 32, 95, 0.5)  # == M1 pooled dev scale
    sels = {p: MemorySelector(p, scale) for p in ALL_POLICIES}
    report = {"pose_scale": {"s_t": scale.s_t, "s_r_deg": scale.s_r_deg, "w_t": scale.w_t}}

    # ---- PART A
    audit = failure_audit(mems, sels, ds)
    audit.to_csv(REP / "A_new_failure_memory_audit.csv", index=False)
    from baseline.dropout_runner import RunSpec
    tA = case_timeline(mems["Scene06"], FAMILIES["M4"][2], RunSpec("Scene06", 9, "burst_30", 2), FAMILIES["M4"][3],
                       sels, list(range(69, 84)))
    tB = case_timeline(mems["Scene06"], FAMILIES["M4"][2], RunSpec("Scene06", 12, "random_50", 1), FAMILIES["M4"][3],
                       sels, list(range(94, 122)))
    tA.to_csv(REP / "A_caseA_t9_burst30_s2_timeline.csv", index=False)
    tB.to_csv(REP / "A_caseB_t12_random50_s1_timeline.csv", index=False)
    report["A_verdicts"] = audit.groupby(["family", "verdict"]).size().reset_index(name="frames").to_dict("records") \
        if len(audit) else []
    report["A_runs_with_new_failures"] = audit.groupby(["family", "scene", "track_id", "condition", "seed"]).size() \
        .reset_index(name="frames").to_dict("records") if len(audit) else []

    # ---- PART C
    opp_rows, frame_rows = [], []
    for fam, (conds, seeds, man, runs) in FAMILIES.items():
        for spec in experiment_matrix(TRACKS, conds, seeds):
            p = fd.load(manifest_path(man, spec))
            sim = simulate(mems[spec.scene], list(p.received_ids), sels, 32, cam[spec.scene])
            pr = load_frames(runs, spec)
            if pr is not None:
                sim = sim.merge(pr[["original_frame_id", "J_drop", "J_no_drop_matched", "gt_empty"]],
                                on="original_frame_id", how="left")
            sim = sim.assign(family=fam, scene=spec.scene, track_id=spec.track_id, condition=spec.condition,
                             seed=spec.seed)
            frame_rows.append(sim)
    # no-drop reference (the M2 clip itself) for comparison
    m2map = pd.read_csv("outputs/milestone2/reports/video_index_mapping.csv")
    for (sc, tr), g in m2map.groupby(["scene", "track_id"]):
        sim = simulate(mems[sc], g.original_frame_id.tolist(), sels, 32, cam[sc])
        frame_rows.append(sim.assign(family="no_drop", scene=sc, track_id=tr, condition="no_drop", seed=0))
    F = pd.concat(frame_rows, ignore_index=True)
    F["failure_frame"] = (F.get("J_drop") < 0.5) & (~F.get("gt_empty", pd.Series(False, index=F.index)).fillna(True).astype(bool))
    pairs = [("recent-K", "pose-only"), ("recent-per-bin", "pose+dt"), ("time-distributed", "pose+dt"),
             ("recent-K", "pose+dt"), ("recent-K", "recent-per-bin")]

    def summarize(g):
        r = {"frames": len(g), "selection_possible": g.selection_possible.mean(), "full_bank": g.full_bank.mean(),
             "pool_mean": g.pool.mean()}
        for a, b in pairs:
            ov, ident = compare(g, a, b)
            r[f"ov|{a}|{b}"], r[f"diff|{a}|{b}"] = ov, 1 - ident
        sp = g[g.selection_possible]
        for a, b in pairs[:2]:
            r[f"diff_when_possible|{a}|{b}"] = 1 - compare(sp, a, b)[1] if len(sp) else np.nan
        for p in ALL_POLICIES:
            r[f"age|{p}"] = g[f"{p}|age"].mean()
            r[f"dT|{p}"] = g[f"{p}|dT"].mean()
            r[f"dR|{p}"] = g[f"{p}|dR"].mean()
        return pd.Series(r)

    by_cond = F.groupby(["family", "condition"]).apply(summarize, include_groups=False).reset_index()
    by_state = F[F.camera_state.isin(["stopped", "straight", "turning"])].groupby(
        ["family", "camera_state"]).apply(summarize, include_groups=False).reset_index()
    fail = F[F.failure_frame.fillna(False)]
    by_fail = fail.groupby(["family"]).apply(summarize, include_groups=False).reset_index() if len(fail) else pd.DataFrame()
    full = F[F.full_bank].groupby(["family", "condition"]).apply(summarize, include_groups=False).reset_index()
    by_cond.to_csv(REP / "C_selection_opportunity_by_condition.csv", index=False)
    by_state.to_csv(REP / "C_selection_opportunity_by_camera_state.csv", index=False)
    full.to_csv(REP / "C_selection_opportunity_full_bank.csv", index=False)
    if len(by_fail):
        by_fail.to_csv(REP / "C_selection_opportunity_at_failures.csv", index=False)
    F.drop(columns=list(ALL_POLICIES)).to_csv(REP / "C_frames.csv.gz", index=False, compression="gzip")

    # ---- NOISE tolerance sweep on real data (stationary vs moving), all drop families + no-drop
    tol_rows = []
    for tt, tr_ in TOLERANCES:
        sc = PoseScale(scale.s_t, scale.s_r_deg, scale.w_t, tt, tr_)
        s2 = {p: MemorySelector(p, sc) for p in ("recent-K", "pose-only", "pose+dt", "recent-per-bin")}
        rows = []
        for fam, (conds, seeds, man, runs) in list(FAMILIES.items()) + [("no_drop", (None, None, None, None))]:
            specs = (experiment_matrix(TRACKS, conds, seeds) if conds else
                     [type("S", (), {"scene": s, "track_id": t})() for s, ts in TRACKS.items() for t in ts])
            for spec in specs:
                rec = (list(fd.load(manifest_path(man, spec)).received_ids) if conds else
                       m2map[(m2map.scene == spec.scene) & (m2map.track_id == spec.track_id)].original_frame_id.tolist())
                rows.append(simulate(mems[spec.scene], rec, s2, 32, cam[spec.scene]).assign(
                    family=fam, scene=spec.scene, track_id=spec.track_id,
                    condition=getattr(spec, "condition", "no_drop"), seed=getattr(spec, "seed", 0)))
        G = pd.concat(rows)
        key = ["family", "scene", "track_id", "condition", "seed", "original_frame_id"]
        G = G.merge(F[key + ["failure_frame"]], on=key, how="left")
        for st, g in [("stopped", G[(G.camera_state == "stopped") & G.failure_frame.fillna(False)]),
                      ("moving", G[G.camera_state.isin(["straight", "turning"]) & G.failure_frame.fillna(False)])]:
            tol_rows.append({"tol_t_m": tt, "tol_r_deg": tr_, "camera_state": f"FAILURE_{st}", "family": "ALL",
                             "frames": len(g), "diff_recentK_poseonly": 1 - compare(g, "recent-K", "pose-only")[1],
                             "diff_recentperbin_posedt": 1 - compare(g, "recent-per-bin", "pose+dt")[1]})
        for st, g in [("stopped", G[G.camera_state == "stopped"]), ("moving", G[G.camera_state.isin(["straight", "turning"])]),
                      ("all", G)]:
            for fam, gg in list(g.groupby("family")) + [("ALL", g)]:
                tol_rows.append({"tol_t_m": tt, "tol_r_deg": tr_, "camera_state": st, "family": fam, "frames": len(gg),
                                 "diff_recentK_poseonly": 1 - compare(gg, "recent-K", "pose-only")[1],
                                 "diff_recentperbin_posedt": 1 - compare(gg, "recent-per-bin", "pose+dt")[1]})
    tol = pd.DataFrame(tol_rows)
    tol.to_csv(REP / "N_pose_tolerance_sweep.csv", index=False)

    # ---- PART D (synthetic)
    geo, curves, lags = geometry_study(scale)
    geo.to_csv(GEO / "D_geometry_summary.csv", index=False)
    lags.to_csv(GEO / "D_revisit_lags.csv", index=False)
    pmin, vopt = syn.min_loop_period_frames()
    report["D_min_plausible_loop_period_frames_at_10Hz"] = {"frames": pmin, "speed_mps": vopt,
                                                            "assumptions": "R>=5 m, lateral acc <= 4 m/s^2, 10 Hz"}
    syn_tol = []
    jm = syn.to_memories(syn.stationary(200), 0.005, 0.02, seed=0)
    for tt, tr_ in TOLERANCES:
        sc = PoseScale(scale.s_t, scale.s_r_deg, scale.w_t, tt, tr_)
        sim = simulate({m.original_frame_id: m for m in jm}, [m.original_frame_id for m in jm],
                       {p: MemorySelector(p, sc) for p in ("recent-K", "pose-only")}, 32)
        syn_tol.append({"tol_t_m": tt, "tol_r_deg": tr_, "diff_recentK_poseonly": 1 - compare(sim, "recent-K", "pose-only")[1]})
    pd.DataFrame(syn_tol).to_csv(GEO / "D_stationary_jitter_tolerance.csv", index=False)

    # ---- figures
    figs = [plot_paths(scale)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for name, g in geo.groupby("trajectory", sort=False):
        axes[0].plot(g.L, 1 - g.identical_recentK_poseonly, "o-", label=name)
        axes[1].plot(g.L, 1 - g.identical_recentperbin_posedt, "o-", label=name)
    for ax, t in zip(axes, ["pose-only differs from recent-K", "pose+dt differs from recent-per-bin"]):
        ax.set(xscale="log", xticks=[16, 32, 64, 128], xticklabels=["16", "32", "64", "128"], ylim=(-0.02, 1.02),
               xlabel="candidate bank L", ylabel="fraction of frames", title=f"SYNTHETIC: {t}")
    axes[1].legend(fontsize=7)
    figs.append(save(fig, FIG / "D2_policy_divergence_vs_L.png"))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, g in lags.groupby("trajectory", sort=False):
        v = g.revisit_lag.dropna()
        if len(v):
            ax.hist(v, bins=np.arange(0, 400, 8), histtype="step", lw=1.5, label=f"{name} (n={len(v)})")
    for L in (16, 32, 64, 128):
        ax.axvline(L, color="k", ls=":", lw=0.8)
        ax.text(L, ax.get_ylim()[1] * 0.95, f"L={L}", fontsize=7)
    ax.set(xlabel="revisit lag [frames] (newest earlier frame within 2 m / 10 deg, age >= 21)", ylabel="frames",
           title="SYNTHETIC: revisit lag vs candidate-bank size")
    ax.legend(fontsize=7)
    figs.append(save(fig, FIG / "D3_revisit_lag_distribution.png"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    t_all = tol[(tol.family == "ALL")]
    for st, g in t_all.groupby("camera_state"):
        lab = [f"{a*100:g}cm/{b:g}°" for a, b in zip(g.tol_t_m, g.tol_r_deg)]
        axes[0].plot(range(len(g)), g.diff_recentK_poseonly, "o-", label=st)
        axes[1].plot(range(len(g)), g.diff_recentperbin_posedt, "o-", label=st)
    for ax, t in zip(axes, ["pose-only != recent-K", "pose+dt != recent-per-bin"]):
        ax.set(xticks=range(len(TOLERANCES)), xticklabels=[f"{a*100:g}cm/{b:g}°" for a, b in TOLERANCES],
               ylim=(-0.02, 1.02), ylabel="fraction of frames", xlabel="pose tolerance (translation / rotation)",
               title=f"REAL dev scenes, all drop conditions: {t}")
        ax.legend()
    figs.append(save(fig, FIG / "N1_pose_tolerance_sweep.png"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    bc = by_cond.copy()
    bc["label"] = bc.family + ":" + bc.condition
    x = np.arange(len(bc))
    axes[0].bar(x - 0.2, bc["diff|recent-K|pose-only"], 0.4, label="pose-only != recent-K")
    axes[0].bar(x + 0.2, bc["diff|recent-per-bin|pose+dt"], 0.4, label="pose+dt != recent-per-bin")
    axes[0].set(xticks=x, ylim=(0, 1), ylabel="fraction of frames", title="REAL: selection differs (all frames)")
    axes[0].set_xticklabels(bc.label, rotation=60, ha="right", fontsize=7)
    axes[0].legend(fontsize=8)
    axes[1].bar(x, bc.selection_possible, color="0.5")
    axes[1].set(xticks=x, ylim=(0, 1), ylabel="fraction of frames", title="REAL: more candidates than policy slots")
    axes[1].set_xticklabels(bc.label, rotation=60, ha="right", fontsize=7)
    figs.append(save(fig, FIG / "C1_selection_opportunity_by_condition.png"))

    report |= {"C_by_condition": by_cond.round(4).to_dict("records"), "C_by_camera_state": by_state.round(4).to_dict("records"),
               "C_at_failures": by_fail.round(4).to_dict("records") if len(by_fail) else [],
               "N_tolerance": tol[tol.family == "ALL"].round(4).to_dict("records"),
               "D_geometry": geo.round(4).to_dict("records"), "figures": figs}
    (REP / "m45_feasibility_report.json").write_text(json.dumps(report, indent=2, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print("scale", report["pose_scale"])
    print("A audit\n", audit[["family", "scene", "track_id", "condition", "seed", "original_frame_id", "video_idx", "gap",
                              "noncond_available", "sam2_uses_all_available", "pre_onset_candidates_outside_sam2_set",
                              "clean_memories_in_sam2_set", "clean_candidates_outside_sam2_set",
                              "J_drop", "J_no_drop", "verdict"]].to_string() if len(audit) else "no new failures")
    print("caseA\n", tA.to_string())
    print("caseB\n", tB.to_string())
    cols = ["family", "condition", "frames", "selection_possible", "full_bank", "diff|recent-K|pose-only",
            "diff|recent-per-bin|pose+dt", "diff_when_possible|recent-K|pose-only",
            "diff_when_possible|recent-per-bin|pose+dt", "age|recent-K", "age|pose-only", "dT|pose-only", "dT|recent-K"]
    print("C by condition\n", by_cond[cols].round(3).to_string())
    print("C by camera state\n", by_state[cols[:2] + cols[2:9]].rename(columns={"condition": "x"}).round(3).to_string()
          if "condition" in by_state else by_state[["family", "camera_state", "frames", "selection_possible",
                                                     "diff|recent-K|pose-only", "diff|recent-per-bin|pose+dt"]].round(3).to_string())
    if len(by_fail):
        print("C at failure frames\n", by_fail[["family", "frames", "selection_possible", "diff|recent-K|pose-only",
                                               "diff|recent-per-bin|pose+dt"]].round(3).to_string())
    print("tolerance\n", tol[tol.family == "ALL"].round(3).to_string())
    print("geometry\n", geo.round(3).to_string())
    print("synthetic stationary jitter tolerance\n", pd.DataFrame(syn_tol).round(3).to_string())
    print("min plausible loop period", pmin, vopt)


if __name__ == "__main__":
    main()
