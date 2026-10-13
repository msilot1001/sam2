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
@torch.inference_mode()
def start(pred, u: Unit, p5: bool):
    """init_state (video offloaded to CPU: identical values, bounded VRAM) + GT prompt at 0 (+ P5) + preflight."""
    state = pred.init_state(video_path=str(frame_dir(u)), offload_video_to_cpu=True)
    pred.add_new_mask(state, frame_idx=0, obj_id=1, mask=gt_of(u, 0))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[a.what]()


if __name__ == "__main__":
    main()
