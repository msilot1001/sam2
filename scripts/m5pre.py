"""M5-Pre: causal memory interventions on SAM 2.1 Tiny (CUDA FP32, official weights).

    uv run python scripts/m5pre.py partA          # depletion ablation on DAVIS pigs obj2 (A0-A5)
    uv run python scripts/m5pre.py manifest       # freeze Part-B intervention manifest (no inference)
    uv run python scripts/m5pre.py partB          # counterfactual memory utility (resumable)
    uv run python scripts/m5pre.py analyze        # Part-B / Part-C tables

Outputs: outputs/milestone5pre/. Ground truth is read only for evaluation, after predictions.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.dropout_runner import atomic_csv, atomic_json
from baseline.metrics import f_measure, j_measure
from baseline.sam2_vos import Clip, write_clip_folder
from memint import controlled, engine, planners
from rfeas.pairing import longest_streak
from rfeas.units import Unit, davis_frame_dir, davis_gt, davis_units, stable_hash, vkitti_units

OUT = Path("outputs/milestone5pre")
PIGS_SKIP = [46, 47, 48, 62, 63, 64, 65, 66, 67]   # exact risk-gate decisions recorded in the MPS run (rfeas)
L_POOL = 32


def build_predictor():
    from sam2.build_sam import build_sam2_video_predictor
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt",
                                      device=torch.device("cuda"), apply_postprocessing=True)


def units_index():
    d, _ = davis_units("val")
    v, _, dsets = vkitti_units(("Scene01", "Scene06"))
    return {u.name: u for u in d + v}, dsets


def prepare(u: Unit, dsets):
    if u.dataset == "davis":
        return davis_frame_dir(u), davis_gt(u, u.frames[0]), (lambda f: davis_gt(u, f))
    ds = dsets[u.video.split("/")[1]]
    vdir = write_clip_folder(ds, Clip(ds.scene, u.obj_id, u.frames, variant="m5pre"), OUT / "clips")
    return vdir, ds.get_frame(u.frames[0], u.obj_id)["mask"], (lambda f: ds.get_frame(f, u.obj_id)["mask"])


def other_objects_gt(u: Unit, frame) -> np.ndarray | None:
    """Union of the OTHER annotated objects (DAVIS) -- for distractor-capture diagnostics only."""
    if u.dataset != "davis":
        return None
    from PIL import Image
    from rfeas.units import DAVIS
    a = np.asarray(Image.open(DAVIS / "Annotations/480p" / u.video.split("/", 1)[1] / f"{frame}.png"))
    return (a != 0) & (a != 255) & (a != u.obj_id)


def jf(pr, gt):
    j, f = j_measure(pr, gt), f_measure(pr, gt)
    return j, f, (j + f) / 2


# ----------------------------------------------------------------------------- PART A
def part_a():
    pred = build_predictor()
    idx, dsets = units_index()
    u = idx["davis__davis/pigs__obj2"]
    vdir, prompt, gt_of = prepare(u, dsets)
    n = len(u.frames)
    S = frozenset(PIGS_SKIP)
    conds = {
        "A0_native": None,
        "A1_skip_both": planners.suppress(S, S),
        "A2_spatial_only": planners.suppress(S, frozenset()),
        "A3_pointer_only": planners.suppress(frozenset(), S),
        "A4_backfill_ptr_suppressed": planners.backfill(S, S, L_POOL),
        "A5_backfill_ptr_native": planners.backfill(S, frozenset(), L_POOL),
    }
    rows, logs, summary = [], {}, []
    for name, planner in conds.items():
        log = {}
        controlled.install(pred, planner, log)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        state, od = engine.start(pred, vdir, prompt)
        masks, lat = engine.rollout(pred, state, od, 1, n - 1)
        masks[0] = engine.prompt_mask_output(pred, state)
        wall = time.perf_counter() - t0
        pred.reset_state(state)
        controlled.uninstall(pred)
        for i, f in enumerate(u.frames):
            gt, other = gt_of(f), other_objects_gt(u, f)
            j, fm, jfv = jf(masks[i], gt)
            plan = log.get(i, {"spatial": [], "pointers": []})
            sp = [t for p, s, t in plan["spatial"] if s == "non_cond"]
            pt = [t for p, s, t in plan["pointers"] if s == "non_cond"]
            rows.append({"condition": name, "video_idx": i, "J": j, "F": fm, "JF": jfv, "pred_px": int(masks[i].sum()),
                         "gt_px": int(gt.sum()),
                         "distractor_frac": float((masks[i] & other).sum() / max(masks[i].sum(), 1)) if other is not None else np.nan,
                         "spatial_sources": sp, "pointer_sources": pt, "n_spatial": len(sp), "n_pointers": len(pt),
                         "missing_spatial_slots": min(i - 1, 6) - len(sp) if i > 0 else 0,
                         "spatial_ages": [i - t for t in sp], "max_spatial_age": max([i - t for t in sp], default=0)})
        logs[name] = {k: v for k, v in log.items()}
        d = pd.DataFrame([r for r in rows if r["condition"] == name and r["video_idx"] > 0])
        bad = d.J < 0.5
        onset = int(d[bad].video_idx.min()) if bad.any() else None
        rec = None
        if onset is not None:
            after = d[d.video_idx > onset].reset_index(drop=True)
            ok = (after.J >= 0.5).to_numpy()
            rec = next((int(after.video_idx[k]) - onset for k in range(len(ok) - 1) if ok[k] and ok[k + 1]), None)
        summary.append({"condition": name, "mean_J": d.J.mean(), "mean_JF": d.JF.mean(),
                        "frames_J_lt_0.5": int(bad.sum()), "frames_J_lt_0.1": int((d.J < 0.1).sum()),
                        "longest_J_lt_0.5_streak": longest_streak(d.J.tolist()),
                        "first_failure": onset, "frames_to_recovery": rec,
                        "max_distractor_frac": float(d.distractor_frac.max()),
                        "frames_distractor_gt_0.3": int((d.distractor_frac > 0.3).sum()),
                        "min_spatial_slots_after_skips": int(d[d.video_idx >= 47].n_spatial.min()),
                        "min_pointers_after_skips": int(d[d.video_idx >= 47].n_pointers.min()),
                        "max_spatial_age": int(d.max_spatial_age.max()),
                        "wall_s": wall, "mean_frame_s": float(np.mean(list(lat.values()))),
                        "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6})
    (OUT / "partA").mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows)
    atomic_csv(R.astype({c: str for c in ("spatial_sources", "pointer_sources", "spatial_ages")}), OUT / "partA" / "per_frame.csv")
    S_ = pd.DataFrame(summary)
    atomic_csv(S_, OUT / "partA" / "summary.csv")
    atomic_json({"unit": u.name, "skip_frames": PIGS_SKIP, "L": L_POOL, "summary": S_.to_dict("records"),
                 "plans": logs}, OUT / "partA" / "partA_report.json")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(S_.round(4).to_string(index=False))
    w = R[R.video_idx.between(60, 79)].pivot(index="video_idx", columns="condition", values="J").round(3)
    print(w.to_string())


# ----------------------------------------------------------------------------- manifest (Part B)
GROUP_A = ["davis__davis/pigs__obj2", "davis__davis/gold-fish__obj1", "davis__davis/gold-fish__obj2",
           "vkitti__vkitti/Scene06__obj9", "vkitti__vkitti/Scene01__obj86"]
N_GROUP_B = 7
FRAMES_PER_SEQ = 6
MIN_IDX = 9            # >= 8 earlier non-cond frames: 6 native slots + >= 1 replacement candidate
ROLLOUT = 10


def manifest():
    idx, _ = units_index()
    cands = sorted((n for n in idx if n not in GROUP_A), key=lambda n: stable_hash(f"m5pre-groupB|{n}"))
    group_b = []
    for n in cands:
        if len(idx[n].frames) >= MIN_IDX + ROLLOUT + FRAMES_PER_SEQ:
            group_b.append(n)
        if len(group_b) == N_GROUP_B:
            break
    rows, excluded = [], []
    for grp, names in (("A_known", GROUP_A), ("B_control", group_b)):
        for n in names:
            u = idx[n]
            elig = list(range(MIN_IDX, len(u.frames) - ROLLOUT))
            pick = sorted(elig, key=lambda t: stable_hash(f"m5pre-frame|{n}|{t}"))[:FRAMES_PER_SEQ]
            for t in sorted(pick):
                rows.append({"group": grp, "unit": n, "frame_idx": t, "rule": "hash"})
            excluded.append({"unit": n, "excluded_frames": f"idx < {MIN_IDX} (too few memories) or > n-1-{ROLLOUT} (no rollout room)"})
    if (OUT / "manifest.json").exists():
        old = json.loads((OUT / "manifest.json").read_text())
        if old["rows"] != rows:
            raise SystemExit("manifest already frozen with different content; refusing to overwrite")
        print("manifest unchanged")
        return
    atomic_json({"frozen_unix": time.time(), "rule": "Group A = prespecified known cases; Group B = first "
                 f"{N_GROUP_B} other units (>= {MIN_IDX + ROLLOUT + FRAMES_PER_SEQ} frames) by sha256('m5pre-groupB|unit'); "
                 f"{FRAMES_PER_SEQ} frames per unit by sha256('m5pre-frame|unit|t') among idx in [{MIN_IDX}, n-1-{ROLLOUT}]",
                 "rows": rows, "excluded": excluded}, OUT / "manifest.json")
    print(pd.DataFrame(rows).groupby(["group", "unit"]).frame_idx.apply(list).to_string())


# ----------------------------------------------------------------------------- PART B
def interventions(frame_idx, plan, od, unit_name):
    """Prespecified counterfactuals at one frame. Returns {name: inner planner}."""
    slots = planners.slot_sources(plan)               # most recent first
    pool = planners.candidate_pool(frame_idx, plan, od, L_POOL)
    out = {"B0_native": lambda f, p, o: p}
    if len(slots) < 2 or not pool:
        return out, slots, pool
    for k, (tp, src) in enumerate(slots):
        out[f"B1_loo|{src}"] = planners.leave_one_out(src)
        out[f"B4_recency|{src}"] = planners.replace(src, pool[0])
        r = planners.deterministic_choice(pool, f"m5pre-rand|{unit_name}|{frame_idx}|{src}")
        out[f"B3_random|{src}"] = planners.replace(src, r)
        out[f"B2_oldest|{src}"] = planners.replace(src, pool[-1])
        out[f"B5_pos_only|{src}"] = planners.reposition(src, 1)
    return out, slots, pool


ROLLOUT_SET = ("B0_native", "B1_loo|oldest", "B1_loo|newest", "B4_recency|oldest", "B3_random|oldest")


def part_b():
    pred = build_predictor()
    idx, dsets = units_index()
    man = json.loads((OUT / "manifest.json").read_text())
    by_unit = pd.DataFrame(man["rows"]).groupby("unit")
    (OUT / "partB").mkdir(parents=True, exist_ok=True)
    for name, g in by_unit:
        dest = OUT / "partB" / (name.replace("/", "~") + ".json")
        if dest.exists() and json.loads(dest.read_text()).get("status") == "complete":
            print(f"{name}: done, skipped", flush=True)
            continue
        u = idx[name]
        vdir, prompt, gt_of = prepare(u, dsets)
        gts = {i: gt_of(f) for i, f in enumerate(u.frames)}           # evaluation only
        others = {i: other_objects_gt(u, f) for i, f in enumerate(u.frames)}
        log = {}
        controlled.install(pred, None, log)
        state, od = engine.start(pred, vdir, prompt)
        targets = sorted(g.frame_idx.tolist())
        records, t_start = [], time.perf_counter()
        native_masks = {}
        for t in range(1, max(targets) + 1):
            if t in targets:
                base_plan = None
                pred._memint_planner = None
                log.clear()
                _out, m_nat, _p, s_nat = engine.predict(pred, state, od, t)
                base_plan = log[t]
                ivs, slots, pool = interventions(t, {"spatial": base_plan["spatial"], "pointers": base_plan["pointers"]},
                                                 od, name)
                gt, other = gts[t], others[t]
                for iv_name, inner in ivs.items():
                    pred._memint_planner = planners.at_frame(t, inner)
                    log.clear()
                    out_i, m_i, _, s_i = engine.predict(pred, state, od, t)
                    j, fm, jfv = jf(m_i, gt)
                    rec = {"unit": name, "group": g.group.iloc[0], "frame_idx": t, "intervention": iv_name,
                           "J": j, "F": fm, "JF": jfv, "obj_score": s_i, "pred_px": int(m_i.sum()),
                           "mask_changed_px": int((m_i ^ m_nat).sum()), "gt_px": int(gt.sum()),
                           "distractor_frac": float((m_i & other).sum() / max(m_i.sum(), 1)) if other is not None else np.nan,
                           "spatial": [x[2] for x in log[t]["spatial"] if x[1] == "non_cond"],
                           "n_slots_native": len(slots), "pool_size": len(pool)}
                    records.append(rec)
                # short rollouts from the identical pre-intervention state (cloned output dicts)
                for rname in ROLLOUT_SET:
                    key = {"B0_native": "B0_native",
                           "B1_loo|oldest": f"B1_loo|{slots[-1][1]}" if len(slots) >= 2 and pool else None,
                           "B1_loo|newest": f"B1_loo|{slots[0][1]}" if len(slots) >= 2 and pool else None,
                           "B4_recency|oldest": f"B4_recency|{slots[-1][1]}" if len(slots) >= 2 and pool else None,
                           "B3_random|oldest": f"B3_random|{slots[-1][1]}" if len(slots) >= 2 and pool else None}[rname]
                    if key is None:
                        continue
                    branch = engine.clone(od)
                    pred._memint_planner = planners.at_frame(t, ivs[key])
                    bm, _ = engine.rollout(pred, state, branch, t, min(t + ROLLOUT, len(u.frames) - 1))
                    for tt, m in bm.items():
                        j, fm, jfv = jf(m, gts[tt])
                        records.append({"unit": name, "group": g.group.iloc[0], "frame_idx": t,
                                        "intervention": f"ROLL|{rname}", "rollout_t": tt - t, "J": j, "F": fm, "JF": jfv,
                                        "pred_px": int(m.sum()), "gt_px": int(gts[tt].sum())})
                    del branch
                pred._memint_planner = None
                od["non_cond_frame_outputs"][t] = _out          # continue the NATIVE trajectory
                native_masks[t] = m_nat
            else:
                pred._memint_planner = None
                out, m, _, _ = engine.predict(pred, state, od, t)
                od["non_cond_frame_outputs"][t] = out
        pred.reset_state(state)
        controlled.uninstall(pred)
        atomic_json({"status": "complete", "unit": name, "frames": targets, "wall_s": time.perf_counter() - t_start,
                     "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6, "records": records}, dest)
        print(f"{name}: {len(records)} records", flush=True)


MEANINGFUL = 0.01   # |delta J| threshold for "helpful"/"harmful" (fixed before Part B results were read)


def analyze():
    from rfeas.risk import auroc, cluster_bootstrap
    recs = []
    for f in sorted((OUT / "partB").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("status") == "complete":
            recs += d["records"]
    R = pd.DataFrame(recs)
    imm = R[~R.intervention.str.startswith("ROLL|")].copy()
    base = imm[imm.intervention == "B0_native"].set_index(["unit", "frame_idx"])[["J", "JF", "obj_score", "pred_px"]]
    iv = imm[imm.intervention != "B0_native"].join(base, on=["unit", "frame_idx"], rsuffix="_native")
    iv["family"] = iv.intervention.str.split("|").str[0]
    iv["slot_frame"] = iv.intervention.str.split("|").str[1].astype(int)
    iv["slot_age"] = iv.frame_idx - iv.slot_frame
    iv["dJ"], iv["dJF"] = iv.J - iv.J_native, iv.JF - iv.JF_native
    iv["d_obj"] = iv.obj_score - iv.obj_score_native
    iv["effect"] = np.select([iv.dJ > MEANINGFUL, iv.dJ < -MEANINGFUL], ["helpful", "harmful"], "neutral")
    frames = base.reset_index()
    opp = imm.groupby(["unit", "frame_idx"]).intervention.nunique()
    rows = []
    for (grp, fam), g in iv.groupby(["group", "family"]):
        rows.append({"group": grp, "family": fam, "interventions": len(g), "frames": g.groupby(["unit", "frame_idx"]).ngroups,
                     "units": g.unit.nunique(), "helpful": int((g.effect == "helpful").sum()),
                     "harmful": int((g.effect == "harmful").sum()), "neutral": int((g.effect == "neutral").sum()),
                     "mask_changed_frac": float((g.mask_changed_px > 0).mean()),
                     "mean_dJ": g.dJ.mean(), "mean_abs_dJ": g.dJ.abs().mean(), "max_dJ": g.dJ.max(), "min_dJ": g.dJ.min(),
                     "mean_dJ_ci_units": cluster_bootstrap(g, "unit", lambda x: x.dJ.mean(), n=2000)})
    fam = pd.DataFrame(rows)
    # oracle upper bound (retrospective GT; NOT deployable)
    orc = iv.groupby(["group", "unit", "frame_idx"]).dJ.max().reset_index()
    oracle = orc.groupby("group").agg(frames=("dJ", "size"), mean_best_dJ=("dJ", "mean"),
                                      frames_best_gt_0_01=("dJ", lambda s: int((s > MEANINGFUL).sum()))).reset_index()
    # rollouts
    ro = R[R.intervention.str.startswith("ROLL|")].copy()
    ro["variant"] = ro.intervention.str[5:]
    nat = ro[ro.variant == "B0_native"].set_index(["unit", "frame_idx", "rollout_t"])[["J", "JF"]]
    ro = ro[ro.variant != "B0_native"].join(nat, on=["unit", "frame_idx", "rollout_t"], rsuffix="_nat")
    roll = []
    for (grp, v), g in ro.groupby(["group", "variant"]):
        per = g.groupby(["unit", "frame_idx"]).apply(lambda x: pd.Series({
            "imm_dJ": float(x[x.rollout_t == 0].J.iloc[0] - x[x.rollout_t == 0].J_nat.iloc[0]),
            "d_JF_1_5": float((x[x.rollout_t.between(1, 5)].JF - x[x.rollout_t.between(1, 5)].JF_nat).mean()),
            "d_JF_1_10": float((x[x.rollout_t >= 1].JF - x[x.rollout_t >= 1].JF_nat).mean()),
            "d_minJ_10": float(x[x.rollout_t >= 1].J.min() - x[x.rollout_t >= 1].J_nat.min()),
            "fail_iv": int((x[x.rollout_t >= 1].J < 0.5).sum()), "fail_nat": int((x[x.rollout_t >= 1].J_nat < 0.5).sum())}),
            include_groups=False)
        roll.append({"group": grp, "variant": v, "frames": len(per), "mean_imm_dJ": per.imm_dJ.mean(),
                     "mean_dJF_1_5": per.d_JF_1_5.mean(), "mean_dJF_1_10": per.d_JF_1_10.mean(),
                     "rollouts_dJF10_gt_0.01": int((per.d_JF_1_10 > MEANINGFUL).sum()),
                     "rollouts_dJF10_lt_-0.01": int((per.d_JF_1_10 < -MEANINGFUL).sum()),
                     "fail_frames_iv": int(per.fail_iv.sum()), "fail_frames_native": int(per.fail_nat.sum()),
                     "corr_imm_vs_rollout": float(per.imm_dJ.corr(per.d_JF_1_10)) if per.imm_dJ.std() > 0 else np.nan})
    roll = pd.DataFrame(roll)
    # Part C gate + simple causal predictors of LOO / replacement utility (labels from GT, features causal)
    pc = {}
    lab = iv[iv.family.isin(["B1_loo", "B4_recency", "B3_random", "B2_oldest"])]
    n_meaningful = int((lab.effect != "neutral").sum())
    pc["meaningful_interventions"] = n_meaningful
    pc["units_with_meaningful"] = int(lab[lab.effect != "neutral"].unit.nunique())
    pc["proceed"] = bool(n_meaningful >= 30 and pc["units_with_meaningful"] >= 4)
    if pc["proceed"]:
        loo = iv[iv.family == "B1_loo"].copy()
        loo["harm_if_removed"] = (loo.dJ < -MEANINGFUL).astype(int)     # memory was useful
        loo["help_if_removed"] = (loo.dJ > MEANINGFUL).astype(int)      # memory was harmful
        feats = {"recency (newer = more useful)": -loo.slot_age, "slot age": loo.slot_age,
                 "native mask area (log)": np.log1p(loo.pred_px_native)}
        pc["loo_useful_auroc"] = {k: auroc(loo.harm_if_removed, v) for k, v in feats.items()}
        pc["loo_harmful_auroc"] = {k: auroc(loo.help_if_removed, v) for k, v in feats.items()}
        pc["loo_useful_by_age"] = loo.groupby("slot_age").agg(n=("dJ", "size"), useful=("harm_if_removed", "mean"),
                                                             harmful=("help_if_removed", "mean"), mean_dJ=("dJ", "mean")).reset_index().to_dict("records")
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    iv.to_csv(OUT / "reports" / "partB_interventions.csv", index=False)
    ro.to_csv(OUT / "reports" / "partB_rollouts.csv", index=False)
    fam.to_csv(OUT / "reports" / "partB_families.csv", index=False)
    roll.to_csv(OUT / "reports" / "partB_rollout_summary.csv", index=False)
    atomic_json({"threshold_dJ": MEANINGFUL, "frames": len(frames), "units": frames.unit.nunique(),
                 "interventions_per_frame": opp.describe().to_dict(), "families": fam.to_dict("records"),
                 "oracle_upper_bound_NOT_DEPLOYABLE": oracle.to_dict("records"), "rollouts": roll.to_dict("records"),
                 "partC": pc}, OUT / "reports" / "partB_report.json")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(fam.round(4).to_string(index=False))
    print(oracle.round(4).to_string(index=False))
    print(roll.round(4).to_string(index=False))
    print(json.dumps(pc, indent=1, default=float))


def candidates():
    """DIAGNOSTIC (retrospective GT, NOT deployable): on the Part-B units' native runs, compare the mask
    SAM 2 chose with its other multimask candidates. Large gaps at failure frames would point to
    candidate selection / identity rather than memory content."""
    pred = build_predictor()
    idx, dsets = units_index()
    units = sorted({r["unit"] for r in json.loads((OUT / "manifest.json").read_text())["rows"]})
    store = {}
    orig = pred._forward_sam_heads

    def heads(*a, **k):
        out = orig(*a, **k)
        store["multi"], store["ious"] = out[1].detach(), out[2].detach().float().cpu().numpy().ravel()
        return out
    pred._forward_sam_heads = heads
    rows = []
    controlled.install(pred, None, {})
    for name in units:
        u = idx[name]
        vdir, prompt, gt_of = prepare(u, dsets)
        state, od = engine.start(pred, vdir, prompt)
        for t in range(1, len(u.frames)):
            out, m, _, score = engine.predict(pred, state, od, t)
            od["non_cond_frame_outputs"][t] = out
            with torch.inference_mode():
                _, vr = pred._get_orig_video_res_output(state, store["multi"])
            cands = (vr[0] > 0).cpu().numpy()
            gt = gt_of(u.frames[t])
            js = [j_measure(c, gt) for c in cands]
            rows.append({"unit": name, "video_idx": t, "J_chosen": j_measure(m, gt), "J_best_candidate": max(js),
                         "n_candidates": len(js), "pred_ious": store["ious"].tolist(), "obj_score": score,
                         "gt_empty": not gt.any()})
        pred.reset_state(state)
    controlled.uninstall(pred)
    del pred._forward_sam_heads
    D = pd.DataFrame(rows)
    D["gap"] = D.J_best_candidate - D.J_chosen
    vis = D[~D.gt_empty]
    fail = vis[vis.J_chosen < 0.5]
    summ = {"frames": len(vis), "failure_frames": len(fail), "failure_units": int(fail.unit.nunique()),
            "failure_frames_with_candidate_J_ge_0.5": int((fail.J_best_candidate >= 0.5).sum()),
            "mean_gap_failure": float(fail.gap.mean()) if len(fail) else None,
            "mean_gap_all": float(vis.gap.mean()),
            "by_unit": fail.groupby("unit").agg(frames=("gap", "size"), mean_J=("J_chosen", "mean"),
                                                mean_best=("J_best_candidate", "mean")).reset_index().to_dict("records")}
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "reports" / "candidate_oracle_frames.csv", index=False)
    atomic_json(summ, OUT / "reports" / "candidate_oracle_report.json")
    print(json.dumps(summ, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["partA", "manifest", "partB", "analyze", "candidates"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"partA": part_a, "manifest": manifest, "partB": part_b, "analyze": analyze, "candidates": candidates}[a.what]()


if __name__ == "__main__":
    main()
