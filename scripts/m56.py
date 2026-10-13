"""M5.6: comprehensive SAM 2.1 failure audit (CUDA FP32, official weights, frozen).

    uv run python scripts/m56.py prepare_pumavos   # data/pumavos/raw -> clip folders (first CAP frames)
    uv run python scripts/m56.py manifest          # freeze PUMaVOS A/B/C split, episode + diagnostic rules
    uv run python scripts/m56.py run               # B0 official / B1 P5 on PUMaVOS A+B (resumable)
    uv run python scripts/m56.py diagnose          # per-episode controlled interventions (resumable)
    uv run python scripts/m56.py analyze           # reports -> outputs/milestone56/reports/

Reuses M5.5 (scripts/m55.py) helpers and M5-Pre (memint) controlled memory retrieval. GT is read only for the
prompt at video_idx 0 and for evaluation; every GT-based quantity below is labelled retrospective / not deployable.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.dropout_runner import atomic_csv, atomic_json, atomic_npz
from baseline.metrics import f_measure, j_measure
from memint import controlled, engine, planners
from rfeas.units import Unit, stable_hash
from spatial import detail

_spec = importlib.util.spec_from_file_location("m55", Path(__file__).with_name("m55.py"))
m55 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m55)

OUT = Path("outputs/milestone56")
PUMA_RAW = Path("data/pumavos/raw")
PUMA = Path("data/pumavos/clips")
CAP = 300                 # frames per PUMaVOS video in the pilot (prompt + 299), pre-registered
FAIL_J = 0.5
MERGE_GAP = 2             # failed runs separated by <= 2 good frames form one episode
MAX_EP_PER_UNIT = 3       # diagnose at most the first 3 episodes of a unit (no single unit dominates)
WINDOW = 10               # intervention rollout length (onset .. onset+9)
RECOVER = {"min_mean_J": 0.5, "min_gain": 0.15}


# ----------------------------------------------------------------------------- episodes (pure)
def episodes(js, merge_gap: int = MERGE_GAP, thr: float = FAIL_J):
    """Failure episodes over a per-frame J sequence (positions 0..n-1). Runs of J < thr separated by at most
    merge_gap good frames merge. Returns [(start, end_inclusive, recovered_at or None)]."""
    runs, i, n = [], 0, len(js)
    while i < n:
        if js[i] < thr:
            j = i
            while j + 1 < n and js[j + 1] < thr:
                j += 1
            if runs and i - runs[-1][1] - 1 <= merge_gap:
                runs[-1][1] = j
            else:
                runs.append([i, j])
            i = j + 1
        else:
            i += 1
    return [(s, e, e + 1 if e + 1 < n else None) for s, e in runs]


def episode_table(frames: pd.DataFrame, arm_col="arm") -> pd.DataFrame:
    """frames: unit, arm, video_idx, J (all evaluated frames, sorted)."""
    rows = []
    for (u, a), g in frames.sort_values("video_idx").groupby(["unit", arm_col]):
        idx, js = g.video_idx.to_numpy(), g.J.to_numpy()
        for k, (s, e, rec) in enumerate(episodes(js)):
            rows.append({"unit": u, arm_col: a, "episode": k, "onset": int(idx[s]), "end": int(idx[e]),
                         "duration": int(idx[e] - idx[s] + 1), "recovered": rec is not None,
                         "frames_to_recover": int(idx[rec] - idx[s]) if rec is not None else None,
                         "mean_J": float(js[s:e + 1].mean()), "frames_total": len(js)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- data
def prepare_pumavos():
    """First CAP frames of every PUMaVOS video -> data/pumavos/clips/<v>/{00000.jpg.., obj<k>.npy, meta.json}.
    Frames whose JPEG size differs from the annotation size (5 videos are 1080p) are resized to the annotation
    size (Lanczos, JPEG q95); others are symlinked unchanged. Never overwrites an existing clip folder."""
    from PIL import Image
    root = PUMA_RAW / "PUBLIC_PUMaVOS"
    for vdir in sorted((root / "JPEGImages").iterdir()):
        v, dst = vdir.name, PUMA / vdir.name
        if dst.exists():
            continue
        jpgs = sorted(vdir.iterdir())[:CAP]
        anns = [root / "Annotations" / v / (j.stem + ".png") for j in jpgs]
        assert all(a.exists() for a in anns), v
        A = np.stack([np.asarray(Image.open(a)) for a in anns])
        ids = sorted(int(i) for i in np.unique(A) if i not in (0, 255))
        dst.mkdir(parents=True)
        H, W = A.shape[1:]
        resized = False
        for i, j in enumerate(jpgs):
            im = Image.open(j)
            if im.size != (W, H):
                im.convert("RGB").resize((W, H), Image.LANCZOS).save(dst / f"{i:05d}.jpg", quality=95)
                resized = True
            else:
                (dst / f"{i:05d}.jpg").symlink_to(j.resolve())
        for k in ids:
            np.save(dst / f"obj{k}.npy", A == k)
        (dst / "meta.json").write_text(json.dumps({"n_frames_total": len(list(vdir.iterdir())), "n_frames": len(jpgs),
                                                   "ids": ids, "hw": [int(H), int(W)], "resized_from_jpeg": resized,
                                                   "void_255": int((A == 255).sum())}))
        print(v, len(jpgs), ids, (H, W), "resized" if resized else "")


def puma_units() -> list[Unit]:
    us = []
    for v in sorted(p for p in PUMA.iterdir() if p.is_dir()):
        meta = json.loads((v / "meta.json").read_text())
        first = {k for k in meta["ids"] if np.load(v / f"obj{k}.npy", mmap_mode="r")[0].any()}
        us += [Unit("pumavos", f"pumavos/{v.name}", k, tuple(range(meta["n_frames"]))) for k in meta["ids"] if k in first]
    return us


_PM = {}


def gt_of(u: Unit, t) -> np.ndarray:
    if u.dataset != "pumavos":
        return m55.gt_of(u, t)
    key = (u.video, u.obj_id)
    if key not in _PM:
        _PM[key] = np.load(PUMA / u.video.split("/", 1)[1] / f"obj{u.obj_id}.npy")
    return _PM[key][t]


def frame_dir(u: Unit) -> Path:
    return PUMA / u.video.split("/", 1)[1] if u.dataset == "pumavos" else m55.frame_dir(u)


def other_gt(u: Unit, t) -> np.ndarray:
    if u.dataset != "pumavos":
        return m55.other_gt(u, t)
    ids = json.loads((frame_dir(u) / "meta.json").read_text())["ids"]
    return np.logical_or.reduce([gt_of(Unit(u.dataset, u.video, k, u.frames), t) for k in ids if k != u.obj_id]
                                or [np.zeros_like(gt_of(u, t))])


# ----------------------------------------------------------------------------- manifest
CLASS_RULES = [   # observational episode labels, evaluated at the onset frame under B1; first match wins
    ("E_occlusion_absence", "GT area at onset < 25% of the unit's median visible GT area, or GT empty"),
    ("F_identity_distractor", ">= 30% of predicted pixels on another annotated object, or predicted area > 2x GT "
                              "with < 50% of GT covered (capture of something else)"),
    ("A_spatial_resolution", "256 ceiling J at onset < 0.5 (GT -> official low-res logits -> video res)"),
    ("BC_static_segmentation", "image oracle J (GT box / box+inner point, best of 3) < 0.5"),
    ("D_memory_propagation", "image oracle >= 0.5 and >= video J + 0.2"),
    ("H_unresolved", "anything else"),
]
INTERVENTIONS = {   # all from the identical B1 pre-state at the onset frame t0, rollout t0 .. t0+WINDOW-1
    "I0_native": "B1 continuation (control)",
    "I1_history_reset": "exclude every non-prompt memory (spatial + pointer) older than t0; memories formed at >= t0 allowed",
    "I2_older_backfill": "exclude memories of frames t0-6..t0-1, refill spatial slots from older frames (L=32) [M5-Pre A4]",
    "I3_gt_reprompt": "RETROSPECTIVE: GT mask at t0 added as a conditioning frame (full-res memory); not deployable",
    "I4_multimask_oracle": "RETROSPECTIVE: best of SAM 2's 3 candidate masks at t0 (observational; no state change)",
}


def manifest():
    rows = []
    for u in puma_units():
        g0 = gt_of(u, 0)
        meta = json.loads((frame_dir(u) / "meta.json").read_text())
        rows.append({"unit": u.name, "video": u.video, "n_frames": len(u.frames), "n_frames_total": meta["n_frames_total"],
                     "multi_object": len(meta["ids"]) > 1, "resized": meta["resized_from_jpeg"], **m55.strata(g0)})
    vids = pd.DataFrame(rows).groupby("video").agg(detail=("detail", "max")).reset_index()
    split = {}
    for _, g in vids.groupby("detail"):
        order = sorted(g.video, key=lambda v: stable_hash(f"m56-split|{v}"))
        nA, nB = -(-len(order) // 3), len(order) // 4
        for i, v in enumerate(order):
            split[v] = "A_exploration" if i < nA else ("B_development" if i < nA + nB else "C_confirmation")
    for r in rows:
        r["split"] = split[r["video"]]
    m = {"dataset": "PUMaVOS (CC BY 4.0), zip sha256 ccd06263...a087a4", "cap_frames": CAP, "rows": rows,
         "split_rule": "per detail stratum (video-level max of prompt-frame strata): hash order, first ceil(n/3) A, "
                       "next floor(n/4) B, rest C; C is never run in M5.6",
         "baselines": {"B0": "official", "B1": "P5 full-res prompt memory (M5.5, frozen; not tuned here)"},
         "fail_J": FAIL_J, "merge_gap": MERGE_GAP, "max_episodes_per_unit": MAX_EP_PER_UNIT, "window": WINDOW,
         "recover_rule": RECOVER, "class_rules": CLASS_RULES, "interventions": INTERVENTIONS,
         "diagnosed_sets": ["DAVIS val (M5.5 exploration set)", "PUMaVOS A", "PUMaVOS B"],
         "scale_probe": "RETROSPECTIVE fixed GT-union crop (10% margin) re-run of whole clip for units with episodes "
                        "whose median GT area < 1% of frame"}
    m55._freeze(OUT / "manifest.json", m)


# ----------------------------------------------------------------------------- B0 / B1 runs
def start(pred, u: Unit, p5: bool):
    return start_dir(pred, frame_dir(u), gt_of(u, u.frames[0]), p5)


@torch.inference_mode()
def start_dir(pred, vdir, gt0, p5: bool):
    """init_state (video offloaded to CPU: identical values, bounded VRAM) + GT prompt at 0 (+ P5) + preflight."""
    state = pred.init_state(video_path=str(vdir), offload_video_to_cpu=True)
    pred.add_new_mask(state, frame_idx=0, obj_id=1, mask=gt0)
    if p5:
        out = state["temp_output_dict_per_obj"][0]["cond_frame_outputs"][0]
        hi = state["mask_inputs_per_obj"][0][0].to(state["device"]) * 20.0 - 10.0
        out["maskmem_features"], out["maskmem_pos_enc"] = pred._run_memory_encoder(
            state, 0, 1, hi, out["object_score_logits"], is_mask_from_pts=True)
    pred.propagate_in_video_preflight(state)
    return state, state["output_dict_per_obj"][0]


def pilot_units():
    m = json.loads((OUT / "manifest.json").read_text())
    us = {u.name: u for u in puma_units()}
    return [us[r["unit"]] for r in m["rows"] if r["split"] in ("A_exploration", "B_development")]


def run():
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    d = OUT / "runs"
    d.mkdir(parents=True, exist_ok=True)
    for u in pilot_units():
        stem = u.name.replace("/", "_")
        if (d / f"{stem}.csv").exists():
            continue
        rows, store = [], {}
        for arm, p5 in (("B0_official", False), ("B1_P5", True)):
            torch.cuda.reset_peak_memory_stats()
            state, od = start(pred, u, p5)
            masks, lat = engine.rollout(pred, state, od, 1, len(u.frames) - 1, record=(rec := {}))
            vram = torch.cuda.max_memory_allocated()
            pred.reset_state(state)
            store[arm] = np.packbits(np.stack([masks[t] for t in range(1, len(u.frames))]))
            prev = None
            for t in range(1, len(u.frames)):
                g, pr = gt_of(u, t), masks[t]
                c = detail.lowres_roundtrip(g)
                o = other_gt(u, t)
                rows.append({"unit": u.name, "arm": arm, "video_idx": t, **m55.jf(pr, g), "gt_empty": not g.any(),
                             "gt_area": int(g.sum()), "pred_area": int(pr.sum()), "J_ceil256": j_measure(c, g),
                             "gt_covered": float((pr & g).sum() / g.sum()) if g.any() else np.nan,
                             "other_obj_frac": float((pr & o).sum() / pr.sum()) if pr.any() else 0.0,
                             "obj_score": rec[t]["obj_score"],
                             "J_consec_pred": j_measure(pr, prev) if prev is not None else np.nan,
                             "latency_s": lat[t], "peak_vram_bytes": vram})
                prev = pr
        atomic_npz(d / f"{stem}.npz", shape=np.array([len(u.frames) - 1, *gt_of(u, 0).shape]), **store)
        df = pd.DataFrame(rows)
        atomic_csv(df, d / f"{stem}.csv")
        print(u.name, df.groupby("arm").JF.mean().round(4).to_dict(), flush=True)
    controlled.uninstall(pred)


# ----------------------------------------------------------------------------- per-episode diagnostics
def history_reset(t0):
    def planner(frame_idx, plan, od):
        if frame_idx < t0:
            return plan
        keep = lambda e: e[1] != "non_cond" or e[2] >= t0
        return {**plan, "spatial": [e for e in plan["spatial"] if keep(e)],
                "pointers": [e for e in plan["pointers"] if keep(e)]}
    return planner


def b1_frames():
    """B1 per-frame J for every diagnosed unit: DAVIS val from M5.5 ('simple' arm == P5), PUMaVOS A+B from runs/."""
    davis = pd.concat([pd.read_csv(f) for f in sorted(Path("outputs/milestone55/proto/val").glob("*.csv"))])
    davis = davis[davis.arm == "simple"].assign(arm="B1_P5", set="DAVIS_val")
    pm = pd.concat([pd.read_csv(f) for f in sorted((OUT / "runs").glob("*.csv"))])
    sp = {r["unit"]: r["split"] for r in json.loads((OUT / "manifest.json").read_text())["rows"]}
    pm = pm[pm.arm == "B1_P5"].assign(set=lambda d: d.unit.map(sp))
    return pd.concat([davis[["unit", "arm", "set", "video_idx", "J", "gt_empty"]],
                      pm[["unit", "arm", "set", "video_idx", "J", "gt_empty"]]], ignore_index=True)


def all_units():
    from rfeas.units import davis_units
    return {u.name: u for u in davis_units("val")[0] + puma_units()}


def diagnose():
    from PIL import Image
    F = b1_frames()
    E = episode_table(F)
    E = E[E.episode < MAX_EP_PER_UNIT]
    E = E[E.onset + 1 < E.frames_total]                       # need at least one frame after onset
    units = all_units()
    pred = m55.build_predictor()
    ipred = m55.build_image_predictor()
    controlled.install(pred, None, {})
    hook, orig = {}, pred._forward_sam_heads

    def heads(*a, **k):
        out = orig(*a, **k)
        hook["multi"] = out[1].detach()
        return out
    pred._forward_sam_heads = heads
    d = OUT / "diagnose"
    d.mkdir(parents=True, exist_ok=True)
    for name, eps in E.groupby("unit"):
        f = d / f"{name.replace('/', '_')}.json"
        if f.exists():
            continue
        u = units[name]
        n = len(u.frames)
        gts = {t: gt_of(u, u.frames[t]) for t in range(n)}
        med = np.median([g.sum() for g in gts.values() if g.any()])
        state, od = start(pred, u, p5=True)
        res, done = [], 0
        for ep in eps.itertuples():
            t0 = ep.onset
            if t0 - 1 > done:
                engine.rollout(pred, state, od, done + 1, t0 - 1)
                done = t0 - 1
            last = min(t0 + WINDOW - 1, n - 1)
            win = range(t0 + 1, last + 1)
            r = {"unit": name, "episode": ep.episode, "onset": t0, "duration": ep.duration, "recovered": ep.recovered,
                 "frames_to_recover": ep.frames_to_recover, "window_end": last}
            out = {}
            for arm in ("I0_native", "I1_history_reset", "I2_older_backfill", "I3_gt_reprompt"):
                c = engine.clone(od)
                pred._memint_planner = {"I1_history_reset": history_reset(t0),
                                        "I2_older_backfill": planners.backfill(set(range(t0 - 6, t0)), set(range(t0 - 6, t0)), 32)
                                        }.get(arm)
                if arm == "I3_gt_reprompt":
                    with torch.inference_mode():
                        cur, _ = pred._run_single_frame_inference(
                            inference_state=state, output_dict=c, frame_idx=t0, batch_size=1, is_init_cond_frame=False,
                            point_inputs=None, mask_inputs=detail.to_model(gts[t0]).to("cuda"), reverse=False,
                            run_mem_encoder=True)
                    c["cond_frame_outputs"][t0] = cur
                    masks, _ = engine.rollout(pred, state, c, t0 + 1, last) if last > t0 else ({}, {})
                else:
                    masks, _ = engine.rollout(pred, state, c, t0, last)
                pred._memint_planner = None
                out[arm] = {t: masks[t] for t in win}
                r[f"J_{arm}"] = float(np.mean([j_measure(masks[t], gts[t]) for t in win])) if len(win) else np.nan
                if arm == "I0_native":
                    r["J_onset_native"] = j_measure(masks[t0], gts[t0])
                    r["J_onset_recorded"] = float(F[(F.unit == name) & (F.video_idx == t0)].J.iloc[0])
                    r["pred_area_onset"] = int(masks[t0].sum())
                    o = other_gt(u, u.frames[t0])
                    r["other_obj_frac"] = float((masks[t0] & o).sum() / masks[t0].sum()) if masks[t0].any() else 0.0
                    r["gt_covered"] = float((masks[t0] & gts[t0]).sum() / gts[t0].sum()) if gts[t0].any() else np.nan
            # observational probes at the onset frame (same pre-state; nothing stored)
            with torch.inference_mode():
                cur, mk, _, score = engine.predict(pred, state, engine.clone(od), t0)
                _, vr = pred._get_orig_video_res_output(state, hook["multi"])
            r["obj_score_onset"] = score
            r["J_multimask_best"] = max(j_measure(c, gts[t0]) for c in (vr[0] > 0).cpu().numpy())
            r["gt_area_onset"] = int(gts[t0].sum())
            r["gt_area_ratio_to_median"] = float(gts[t0].sum() / med) if med else np.nan
            r["area_frac_onset"] = float(gts[t0].mean())
            r["J_ceil256_onset"] = j_measure(detail.lowres_roundtrip(gts[t0]), gts[t0])
            with torch.inference_mode():
                ipred.set_image(np.asarray(Image.open(sorted(frame_dir(u).glob("*.jpg"))[t0]).convert("RGB")))
                r["J_image_oracle_onset"] = m55.image_oracle(ipred, gts[t0])
            res.append(r)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        pred.reset_state(state)
        atomic_json(res, f)
    pred._forward_sam_heads = orig
    controlled.uninstall(pred)



def crop_box(gts, margin=0.10):
    """RETROSPECTIVE: union of GT boxes over the clip, expanded by margin * box size, clamped. (y0, y1, x0, x1)."""
    ys, xs = np.nonzero(np.logical_or.reduce(list(gts)))
    H, W = gts[0].shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = int(np.ceil(margin * (y1 - y0))), int(np.ceil(margin * (x1 - x0)))
    return max(0, y0 - my), min(H, y1 + my), max(0, x0 - mx), min(W, x1 + mx)


def scale_probe():
    """RETROSPECTIVE upper bound on spatial scale: B1 on a fixed GT-union crop vs B1 on the full frame, for small
    units (median GT area < 1% of frame) that have B1 failure episodes. Same model, prompt, frames; only scale."""
    from PIL import Image
    F = b1_frames()
    E = episode_table(F)
    units = all_units()
    pred = m55.build_predictor()
    rows = []
    for name in sorted(E.unit.unique()):
        u = units[name]
        n = len(u.frames)
        gts = [gt_of(u, u.frames[t]) for t in range(n)]
        if np.median([g.mean() for g in gts if g.any()]) >= 0.01:
            continue
        y0, y1, x0, x1 = crop_box(gts)
        cd = OUT / "crops" / name.replace("/", "_")
        if not cd.exists():
            cd.mkdir(parents=True)
            for t, j in enumerate(sorted(frame_dir(u).glob("*.jpg"))[:n]):
                Image.open(j).convert("RGB").crop((x0, y0, x1, y1)).save(cd / f"{t:05d}.jpg", quality=95)
        state, od = start_dir(pred, cd, gts[0][y0:y1, x0:x1], p5=True)
        masks, _ = engine.rollout(pred, state, od, 1, n - 1)
        pred.reset_state(state)
        fb = F[F.unit == name].set_index("video_idx").J
        eps = E[E.unit == name]
        epf = {t for e in eps.itertuples() for t in range(e.onset, e.end + 1)}
        for t in range(1, n):
            full = np.zeros_like(gts[t])
            full[y0:y1, x0:x1] = masks[t]
            rows.append({"unit": name, "video_idx": t, "J_crop": j_measure(full, gts[t]), "J_full_B1": float(fb[t]),
                         "in_episode": t in epf, "zoom": float(np.sqrt(gts[0].size / ((y1 - y0) * (x1 - x0))))})
        print(name, pd.DataFrame([r for r in rows if r["unit"] == name])[["J_crop", "J_full_B1"]].mean().round(3).to_dict(), flush=True)
    atomic_csv(pd.DataFrame(rows), OUT / "reports" / "scale_probe_frames.csv")



# ----------------------------------------------------------------------------- analysis
def _boot_units(df, col, group="video", n=2000):
    from rfeas.risk import cluster_bootstrap
    return cluster_bootstrap(df, group, lambda d: float(d[col].mean()), n=n)


def classify(r) -> str:
    if r.gt_area_onset == 0 or r.gt_area_ratio_to_median < 0.25:
        return "E_occlusion_absence"
    if r.other_obj_frac >= 0.30 or (r.pred_area_onset > 2 * max(r.gt_area_onset, 1) and r.gt_covered < 0.5):
        return "F_identity_distractor"
    if r.J_ceil256_onset < 0.5:
        return "A_spatial_resolution"
    if r.J_image_oracle_onset < 0.5:
        return "BC_static_segmentation"
    if r.J_image_oracle_onset >= r.J_onset_native + 0.2:
        return "D_memory_propagation"
    return "H_unresolved"


def resolves(x, base):
    return (x >= RECOVER["min_mean_J"]) and (x - base >= RECOVER["min_gain"])


def analyze():
    rd = OUT / "reports"
    rd.mkdir(parents=True, exist_ok=True)
    rep = {}
    # 1. B0 vs B1 on PUMaVOS A+B (+ DAVIS val / SegTrack from M5.5, same code path)
    sp = {r["unit"]: r["split"] for r in json.loads((OUT / "manifest.json").read_text())["rows"]}
    st = {r["unit"]: r for r in json.loads((OUT / "manifest.json").read_text())["rows"]}
    P = pd.concat([pd.read_csv(f) for f in sorted((OUT / "runs").glob("*.csv"))], ignore_index=True)
    P["set"] = P.unit.map(sp)
    P["video"] = P.unit.str.split("__").str[1]
    U = P.groupby(["unit", "video", "set", "arm"]).agg(J=("J", "mean"), F=("F", "mean"), JF=("JF", "mean"),
                                                       F_strict=("F_strict", "mean")).reset_index()
    W = U.pivot_table(index=["unit", "video", "set"], columns="arm", values=["J", "F", "JF", "F_strict"]).reset_index()
    W.columns = ["_".join(c).strip("_") for c in W.columns]
    W["dJF"] = W.JF_B1_P5 - W.JF_B0_official
    W["detail"] = W.unit.map(lambda u: st[u]["detail"])
    atomic_csv(W, rd / "pumavos_B0_B1_per_unit.csv")
    rep["pumavos_B0_B1"] = {s: {"n_units": int(len(g)), "JF_B0": float(g.JF_B0_official.mean()), "JF_B1": float(g.JF_B1_P5.mean()),
                                "dJF": float(g.dJF.mean()), "dJF_CI": _boot_units(g, "dJF"),
                                "dF_strict": float((g.F_strict_B1_P5 - g.F_strict_B0_official).mean()),
                                "units_up_0.01": int((g.dJF > 0.01).sum()), "units_down_0.01": int((g.dJF < -0.01).sum())}
                            for s, g in list(W.groupby("set")) + [("A+B", W)]}
    bins = pd.cut(P.gt_area, [-1, 0, 500, 2000, 8000, 32000, 10**9], labels=["empty", "<500", "500-2k", "2k-8k", "8k-32k", ">32k"])
    rep["pumavos_J_by_area"] = P.groupby([bins, "arm"], observed=True).J.agg(["size", "mean"]).reset_index().astype({"gt_area": str}).to_dict("records")
    rep["pumavos_runtime"] = {a: {"latency_s_median": float(g.latency_s.median()), "peak_vram_GB": float(g.peak_vram_bytes.max() / 1e9)}
                              for a, g in P.groupby("arm")}
    # 2. episodes, B0 vs B1, every set
    davis = pd.concat([pd.read_csv(f) for f in sorted(Path("outputs/milestone55/proto/val").glob("*.csv"))])
    seg = pd.concat([pd.read_csv(f) for f in sorted(Path("outputs/milestone55/proto/segtrack").glob("*.csv"))])
    frames = pd.concat([davis[davis.arm.isin(["official", "simple"])].assign(set="DAVIS_val"),
                        seg[seg.arm.isin(["official", "simple"])].assign(set="SegTrack"),
                        P.assign(arm=P.arm.map({"B0_official": "official", "B1_P5": "simple"}))], ignore_index=True)
    frames["arm"] = frames.arm.map({"official": "B0", "simple": "B1"})
    Ep = []
    for s_, g in frames.groupby("set"):
        e = episode_table(g)
        e["set"] = s_
        Ep.append(e)
    Ep = pd.concat(Ep, ignore_index=True)
    Ep["video"] = Ep.unit.str.split("__").str[1]
    atomic_csv(Ep, rd / "episodes.csv")
    rep["episodes"] = Ep.groupby(["set", "arm"]).agg(episodes=("unit", "size"), units=("unit", "nunique"), videos=("video", "nunique"),
                                                     failed_frames=("duration", "sum"), median_duration=("duration", "median"),
                                                     max_duration=("duration", "max"), unrecovered=("recovered", lambda r: int((~r).sum())),
                                                     median_frames_to_recover=("frames_to_recover", "median")).reset_index().to_dict("records")
    tot = frames.groupby(["set", "arm"]).agg(units=("unit", "nunique"), frames=("J", "size")).reset_index()
    rep["frames_evaluated"] = tot.to_dict("records")
    # which B0 episodes P5 removes (no overlapping B1 episode)
    res = []
    for (s_, u), g in Ep.groupby(["set", "unit"]):
        b1 = g[g.arm == "B1"]
        for e in g[g.arm == "B0"].itertuples():
            ov = b1[(b1.onset <= e.end) & (b1.end >= e.onset)]
            res.append({"set": s_, "unit": u, "onset": e.onset, "duration": e.duration, "resolved_by_P5": ov.empty})
    R = pd.DataFrame(res)
    rep["B0_episodes_resolved_by_P5"] = R.groupby("set").agg(B0_episodes=("unit", "size"), resolved=("resolved_by_P5", "sum"),
                                                              resolved_units=("unit", lambda u: int(u[R.loc[u.index, "resolved_by_P5"]].nunique()))).reset_index().to_dict("records")
    # 3. diagnostics on B1 episodes
    D = pd.DataFrame([r for f in sorted((OUT / "diagnose").glob("*.json")) for r in json.loads(f.read_text())])
    if len(D):
        D["set"] = D.unit.map(lambda u: "DAVIS_val" if u.startswith("davis") else sp.get(u))
        D["video"] = D.unit.str.split("__").str[1]
        D["cls"] = [classify(r) for r in D.itertuples()]
        D["deterministic"] = (D.J_onset_native - D.J_onset_recorded).abs() < 1e-9
        base = D.J_I0_native
        D["memory_intervention_resolves"] = [resolves(max(a, b), c) for a, b, c in zip(D.J_I1_history_reset, D.J_I2_older_backfill, base)]
        D["gt_reprompt_resolves"] = [resolves(a, c) for a, c in zip(D.J_I3_gt_reprompt, base)]
        D["candidate_gap"] = D.J_multimask_best - D.J_onset_native
        D["mechanism_test"] = np.where(D.memory_intervention_resolves, "memory_history_causal",
                                       np.where(D.gt_reprompt_resolves, "needs_new_target_information", "not_recoverable_by_reprompt"))
        atomic_csv(D, rd / "diagnose_episodes.csv")
        agg = lambda g: {"episodes": int(len(g)), "units": int(g.unit.nunique()), "videos": int(g.video.nunique()),
                         "dJ_I1": float((g.J_I1_history_reset - g.J_I0_native).mean()),
                         "dJ_I2": float((g.J_I2_older_backfill - g.J_I0_native).mean()),
                         "dJ_I3_retro": float((g.J_I3_gt_reprompt - g.J_I0_native).mean()),
                         "memory_resolves": int(g.memory_intervention_resolves.sum()),
                         "gt_reprompt_resolves": int(g.gt_reprompt_resolves.sum()),
                         "candidate_gap_ge_0.15": int((g.candidate_gap >= 0.15).sum()),
                         "obj_score_negative": int((g.obj_score_onset < 0).sum())}
        rep["diagnose_all"] = agg(D)
        rep["diagnose_deterministic"] = bool(D.deterministic.all())
        rep["diagnose_by_set"] = {s_: agg(g) for s_, g in D.groupby("set")}
        rep["diagnose_by_class"] = {c: agg(g) for c, g in D.groupby("cls")}
        rep["diagnose_by_mechanism_test"] = {c: agg(g) for c, g in D.groupby("mechanism_test")}
        rep["diagnose_class_x_test"] = pd.crosstab(D.cls, D.mechanism_test).to_dict()
    sc = rd / "scale_probe_frames.csv"
    if sc.exists():
        S = pd.read_csv(sc)
        rep["scale_probe"] = {"units": S.groupby("unit").agg(J_crop=("J_crop", "mean"), J_full=("J_full_B1", "mean"),
                                                             zoom=("zoom", "first")).reset_index().to_dict("records"),
                              "episode_frames": {"n": int(S.in_episode.sum()), "J_crop": float(S[S.in_episode].J_crop.mean()),
                                                 "J_full": float(S[S.in_episode].J_full_B1.mean())}}
    atomic_json(rep, rd / "m56_report.json")
    print(json.dumps(rep, indent=1, default=float)[:12000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[a.what]()


if __name__ == "__main__":
    main()
