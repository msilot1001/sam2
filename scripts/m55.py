"""M5.5: spatial-detail feasibility on SAM 2.1 Tiny (CUDA FP32, official weights, frozen).

    uv run python scripts/m55.py audit       # prompt-frame fidelity audit (paragliding-launch o3, kite-surf o1)
    uv run python scripts/m55.py manifest    # freeze units, strata, rules, ablation/prototype grid (no inference)
    uv run python scripts/m55.py survey      # native run + image-oracle on every manifest unit (resumable)
    uv run python scripts/m55.py ablate      # single-factor prompt ablations, first 20 frames (resumable)
    uv run python scripts/m55.py analyze     # tables -> outputs/milestone55/reports/

Ground truth is read only for evaluation, after the predictions of a unit exist, and for the GT prompt
at video_idx 0. Every "oracle" quantity is retrospective and labelled NOT deployable.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.dropout_runner import atomic_csv, atomic_json, atomic_npz
from baseline.metrics import f_measure, j_measure
from memint import engine
from rfeas.units import DAVIS, Unit, davis_frame_dir, davis_gt, davis_units, stable_hash
from spatial import detail

OUT = Path("outputs/milestone55")
CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CKPT = "checkpoints/sam2.1_hiera_tiny.pt"
AUDIT = [("paragliding-launch", 3), ("kite-surf", 1)]


# ----------------------------------------------------------------------------- dataset dispatch (DAVIS / SegTrack v2)
SEGTRACK = Path("data/segtrackv2/clips")
_ST = {}


def segtrack_units() -> list[Unit]:
    us = []
    for v in sorted(p for p in SEGTRACK.iterdir() if p.is_dir()):
        info = json.loads((v / "frames.json").read_text())
        us += [Unit("segtrack", f"segtrack/{v.name}", k, tuple(range(len(info["stems"]))))
               for k in range(1, info["n_objects"] + 1)]
    return us


def _st(u: Unit, k: int) -> np.ndarray:
    key = (u.video, k)
    if key not in _ST:
        _ST[key] = np.load(SEGTRACK / u.video.split("/", 1)[1] / f"obj{k}.npy")
    return _ST[key]


def gt_of(u: Unit, frame) -> np.ndarray:
    return davis_gt(u, frame) if u.dataset == "davis" else _st(u, u.obj_id)[frame]


def frame_dir(u: Unit) -> Path:
    return davis_frame_dir(u) if u.dataset == "davis" else SEGTRACK / u.video.split("/", 1)[1]


def units_of(split) -> list[Unit]:
    return segtrack_units() if split == "segtrack" else davis_units(split)[0]


def build_predictor():
    from sam2.build_sam import build_sam2_video_predictor
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pred = build_sam2_video_predictor(CFG, CKPT, device=torch.device("cuda"), apply_postprocessing=True)
    # Environment parity (M7 migration): every M5-M7 result was produced where sam2._C is missing, so SAM 2 skips its
    # hole-filling post-process. Make that explicit so machines WITH the extension (RTX 5090) give identical outputs
    # (verified: outputs/milestone7/repro, 164/164 frames identical J&F with this setting, 1 differing frame without).
    pred.fill_hole_area = 0
    return pred


def build_image_predictor():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2(CFG, CKPT, device=torch.device("cuda"), apply_postprocessing=True))


def unit_of(video: str, obj: int, split="val") -> Unit:
    return next(u for u in davis_units(split)[0] if u.video == f"davis/{video}" and u.obj_id == obj)


def jf(pr, gt):
    j, f = j_measure(pr, gt), f_measure(pr, gt)
    return {"J": j, "F": f, "JF": (j + f) / 2, "F_strict": f_measure(pr, gt, 2)}


class MemEncHook:
    """Records the high-res mask each memory-encoder call receives (instance-level; outputs untouched)."""

    def __init__(self, p):
        self.p, self.seen, self._orig = p, {}, p._run_memory_encoder

        def run(inference_state, frame_idx, batch_size, high_res_masks, object_score_logits, is_mask_from_pts):
            self.seen[frame_idx] = (high_res_masks.detach().float().cpu(), bool(is_mask_from_pts))
            return self._orig(inference_state, frame_idx, batch_size, high_res_masks, object_score_logits, is_mask_from_pts)
        p._run_memory_encoder = run

    def remove(self):
        del self.p._run_memory_encoder


def best_shift(pred, gt, r=3):
    """Integer (dy, dx) in [-r, r]^2 maximising J(shifted pred, gt): an alignment check."""
    best = (j_measure(pred, gt), 0, 0)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy or dx:
                best = max(best, (j_measure(np.roll(pred, (dy, dx), (0, 1)), gt), dy, dx))
    return {"J_best": best[0], "dy": best[1], "dx": best[2]}


# ----------------------------------------------------------------------------- 1. prompt-frame audit
def audit():
    from PIL import Image
    pred = build_predictor()
    rows, frames_rows = [], []
    for video, obj in AUDIT:
        u = unit_of(video, obj)
        png = Image.open(DAVIS / "Annotations/480p" / video / f"{u.frames[0]}.png")
        raw = np.asarray(png)
        gt0 = davis_gt(u, u.frames[0])
        jpgs = sorted((DAVIS / "JPEGImages/480p" / video).glob("*.jpg"))
        r = {"unit": u.name, "png_mode": png.mode, "ids_in_prompt_png": sorted(int(i) for i in np.unique(raw)),
             "void_255_pixels": int((raw == 255).sum()), "gt_shape": list(gt0.shape), "gt_area": int(gt0.sum()),
             "jpg_size_wh": list(Image.open(jpgs[0]).size), "prompt_frame_name": u.frames[0],
             "first_jpg": jpgs[0].stem, "n_frames": len(u.frames),
             "jpg_names_are_0..n-1": [p.stem for p in jpgs] == [f"{i:05d}" for i in range(len(jpgs))],
             "thin_fraction_r2": float(detail.thin_part(gt0).sum() / gt0.sum()),
             "components": detail.components(gt0)[1]}
        # stage-by-stage reconstruction of the official path (no model needed)
        m1024 = detail.to_model(gt0)
        r["J_stage_input1024"] = j_measure(detail.model_roundtrip(gt0), gt0)
        low = detail.mask_as_output_lowres(m1024)
        r["J_stage_lowres256_output"] = j_measure(detail.to_video(low, gt0.shape), gt0)
        mem_in = torch.nn.functional.interpolate(low, (1024, 1024), mode="bilinear", align_corners=False)
        r["J_stage_memenc_input"] = j_measure(detail.to_video((mem_in > 0).float() * 20 - 10, gt0.shape), gt0)
        # the real official pipeline, instrumented
        hook = MemEncHook(pred)
        state, od = engine.start(pred, davis_frame_dir(u), gt0)
        assert (state["video_height"], state["video_width"]) == gt0.shape
        out0 = engine.prompt_mask_output(pred, state)
        r["J_official_prompt_output"] = j_measure(out0, gt0)
        r["official_prompt_output_equals_reconstruction"] = bool(np.array_equal(out0, detail.to_video(low, gt0.shape)))
        hm, binar = hook.seen[0]
        r["memenc_prompt_input_res"] = list(hm.shape[-2:])
        r["memenc_prompt_binarized"] = binar
        r["memenc_prompt_input_maxabs_diff_vs_reconstruction"] = float((hm - mem_in).abs().max())
        r["J_official_memenc_input"] = j_measure(detail.to_video((hm > 0).float() * 20 - 10, gt0.shape), gt0)
        r["prompt_output_alignment"] = best_shift(out0, gt0)
        masks, _ = engine.rollout(pred, state, od, 1, 20)
        pred.reset_state(state)
        hook.remove()
        for t in range(1, 21):
            g = davis_gt(u, u.frames[t])
            c = detail.lowres_roundtrip(g)
            frames_rows.append({"unit": u.name, "video_idx": t, **jf(masks[t], g),
                                "J_ceiling256": j_measure(c, g), "F_strict_ceiling256": f_measure(c, g, 2),
                                "J_vs_ceiling_mask": j_measure(masks[t], c), **{f"shift_{k}": v for k, v in best_shift(masks[t], g).items()}})
        fr = pd.DataFrame([x for x in frames_rows if x["unit"] == u.name])
        r["J_first_propagated"] = float(fr.J.iloc[0])
        r["J_frames1_20"] = float(fr.J.mean())
        r["J_ceiling256_frames1_20"] = float(fr.J_ceiling256.mean())
        r["frames_best_shift_nonzero"] = int(((fr.shift_dy != 0) | (fr.shift_dx != 0)).sum())
        r["mean_J_gain_best_shift"] = float((fr.shift_J_best - fr.J).mean())
        # decoder path instead of mask-as-output (diagnostic config flip on this instance only)
        orig_flag = pred.use_mask_input_as_output_without_sam
        pred.use_mask_input_as_output_without_sam = False
        state, od = engine.start(pred, davis_frame_dir(u), gt0)
        r["J_decoder_path_prompt_output"] = j_measure(engine.prompt_mask_output(pred, state), gt0)
        m2, _ = engine.rollout(pred, state, od, 1, 20)
        r["J_decoder_path_frames1_20"] = float(np.mean([j_measure(m2[t], davis_gt(u, u.frames[t])) for t in range(1, 21)]))
        pred.reset_state(state)
        pred.use_mask_input_as_output_without_sam = orig_flag
        rows.append(r)
        print(json.dumps(r, indent=1, default=float))
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    atomic_json(rows, OUT / "audit" / "prompt_audit.json")
    atomic_csv(pd.DataFrame(frames_rows), OUT / "audit" / "prompt_audit_frames.csv")

# ----------------------------------------------------------------------------- 2. frozen manifest
STRATA_RULES = {"small": "prompt-frame area < 1% of the frame",
                "thin": "prompt-frame thin fraction (opening r=2) >= 0.30",
                "multi": ">= 3 prompt-frame components with area >= 20 px",
                "detail": "small or thin or multi", "plain": "not detail"}
FAIL_J = 0.5
CLASS_RULES = [  # applied in order to GT-visible frames with J < FAIL_J; first match wins (pre-registered)
    ("F_occlusion_identity", "GT area < 25% of the unit's median GT area (heavy occlusion / exit), or >= 30% of "
                             "predicted pixels lie on other annotated objects"),
    ("D_output_resolution", "256x256 ceiling J (GT -> official low-res logits -> video res) < 0.5"),
    ("BC_encoder_decoder", "image-oracle J (frozen SAM 2 image mode, GT box / box+inner point, best of 3 masks) < 0.5"),
    ("E_memory_propagation", "image oracle >= 0.5 and >= video J + 0.2 (the frozen decoder can segment it in "
                             "isolation; the video path does not)"),
    ("unresolved", "anything else"),
]
A_RULE = "A_prompt_preprocessing: unit-level; 1024 prompt representation J < 0.9 (official resize)"
G_RULE = "G_annotation: not automatable; reported only from manual inspection of named frames"
ABLATIONS = {   # one factor each, prompt + 20 frames, same units (P6 is a diagnostic config flip, not preprocessing)
    "P0_official": {}, "P1_nearest": {"mode": "nearest"}, "P2_bilinear_noaa": {"mode": "bilinear_noaa"},
    "P3_thr025": {"thr": 0.25}, "P4_thr075": {"thr": 0.75}, "P5_fullres_prompt_memory": {"fullres_mem": True},
    "P6_decoder_path": {"decoder_path": True}}
ABL_FRAMES = 20


def strata(gt0: np.ndarray) -> dict:
    lab, n = detail.components(gt0)
    areas = np.bincount(lab.ravel())[1:]
    s = {"area_frac": float(gt0.mean()), "thin_frac": float(detail.thin_part(gt0).sum() / gt0.sum()),
         "n_comp20": int((areas >= 20).sum())}
    s.update(small=s["area_frac"] < 0.01, thin=s["thin_frac"] >= 0.30, multi=s["n_comp20"] >= 3)
    s["detail"] = s["small"] or s["thin"] or s["multi"]
    return s


def manifest(split=None):
    if split == "segtrack":     # zero-shot generalisation set, frozen separately (added after the DAVIS freeze)
        rows = [{"unit": u.name, "split": "segtrack", "role": "zero_shot_not_in_sam2_training", "n_frames": len(u.frames),
                 **strata(gt_of(u, 0))} for u in segtrack_units()]
        return _freeze(OUT / "manifest_segtrack.json", {"rows": rows, "strata_rules": STRATA_RULES, "fail_J": FAIL_J,
                       "class_rules": CLASS_RULES, "selection": "census of every SegTrack v2 object"})
    path = OUT / "manifest.json"
    rows = []
    for split, role in (("val", "primary"), ("train", "secondary_sam2_training_overlap")):
        for u in davis_units(split)[0]:
            rows.append({"unit": u.name, "split": split, "role": role, "n_frames": len(u.frames),
                         **strata(davis_gt(u, u.frames[0]))})
    m = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "model": {"cfg": CFG, "ckpt": CKPT, "precision": "fp32, tf32 off"},
         "rows": rows, "strata_rules": STRATA_RULES, "fail_J": FAIL_J, "class_rules": CLASS_RULES,
         "A_rule": A_RULE, "G_rule": G_RULE, "ablations": ABLATIONS, "ablation_frames": ABL_FRAMES,
         "ablation_units": "all primary (val) units", "selection": "census of every first-frame object (no sampling)"}
    _freeze(path, m)


def _freeze(path, m):
    m = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), **{k: v for k, v in m.items() if k != "created"}}
    if path.exists():
        old = json.loads(path.read_text())
        if {k: v for k, v in old.items() if k != "created"} != {k: v for k, v in m.items() if k != "created"}:
            sys.exit(f"{path} is frozen and differs; refusing to overwrite")
        print("manifest unchanged")
        return
    atomic_json(m, path)
    print(f"froze {len(m["rows"])} units -> {path}")


def load_manifest(split="val"):
    return json.loads((OUT / ("manifest_segtrack.json" if split == "segtrack" else "manifest.json")).read_text())


# ----------------------------------------------------------------------------- 3. survey (native + oracle)
def inner_point(gt: np.ndarray):
    """Deterministic interior point: first raster pixel of the deepest non-empty erosion."""
    cur = gt.astype(bool)
    while True:
        nxt = detail.erode(cur, 1)
        if not nxt.any():
            ys, xs = np.nonzero(cur)
            return int(xs[0]), int(ys[0])
        cur = nxt


def image_oracle(ipred, gt: np.ndarray) -> float:
    """RETROSPECTIVE, NOT deployable: frozen SAM 2 image mode prompted from GT (box, box+inner point);
    best J over the 3 multimask outputs. Measures what the encoder+decoder can express in isolation."""
    if not gt.any():
        return float("nan")
    ys, xs = np.nonzero(gt)
    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
    best = 0.0
    for pts in (None, np.array([inner_point(gt)], np.float32)):
        ms, _, _ = ipred.predict(point_coords=pts, point_labels=None if pts is None else np.ones(1),
                                 box=box, multimask_output=True)
        best = max(best, *(j_measure(m > 0, gt) for m in ms))
    return best


def other_gt(u: Unit, frame) -> np.ndarray:
    if u.dataset == "segtrack":
        n = json.loads((frame_dir(u) / "frames.json").read_text())["n_objects"]
        return np.logical_or.reduce([_st(u, k)[frame] for k in range(1, n + 1) if k != u.obj_id] or [np.zeros_like(gt_of(u, frame))])
    from PIL import Image
    split_root = DAVIS / "Annotations/480p" / u.video.split("/", 1)[1]
    a = np.asarray(Image.open(split_root / f"{frame}.png"))
    return (a != 0) & (a != 255) & (a != u.obj_id)


def frame_metrics(pr, gt, prev_pr, prev_gt, oth) -> dict:
    c = detail.lowres_roundtrip(gt)
    r = {**jf(pr, gt), "gt_area": int(gt.sum()), "pred_area": int(pr.sum()), "gt_empty": not gt.any(),
         "J_ceil256": j_measure(c, gt), "F_strict_ceil256": f_measure(c, gt, 2),
         "thin_recall": detail.thin_recall(pr, gt), "thin_recall_ceil256": detail.thin_recall(c, gt),
         "comp_retention": detail.component_retention(pr, gt), "comp_retention_ceil256": detail.component_retention(c, gt),
         "other_obj_frac": float((pr & oth).sum() / pr.sum()) if pr.any() else 0.0}
    if prev_pr is not None:
        r["Fs_consec_pred"] = f_measure(pr, prev_pr, 2)
        r["Fs_consec_gt"] = f_measure(gt, prev_gt, 2)
    return r


def _units_by_video(split):
    us = {u.name: u for u in units_of(split)}
    names = [r["unit"] for r in load_manifest(split)["rows"] if r["split"] == split]
    vids = {}
    for n in names:
        vids.setdefault(us[n].video, []).append(us[n])
    return vids


def survey(split="val"):
    from PIL import Image
    pred, ipred = build_predictor(), build_image_predictor()
    d = OUT / "survey" / split
    d.mkdir(parents=True, exist_ok=True)
    for video, units in _units_by_video(split).items():
        todo = [u for u in units if not (d / f"{u.name.replace('/', '_')}.csv").exists()]
        if not todo:
            continue
        runs = {}
        for u in todo:                                     # predictions first (no GT beyond the prompt)
            torch.cuda.reset_peak_memory_stats()
            state, od = engine.start(pred, frame_dir(u), gt_of(u, u.frames[0]))
            t0 = time.perf_counter()
            masks, lat = engine.rollout(pred, state, od, 1, len(u.frames) - 1, record=(rec := {}))
            runs[u.name] = (masks, lat, rec, time.perf_counter() - t0, torch.cuda.max_memory_allocated())
            pred.reset_state(state)
        jpgs = sorted(frame_dir(todo[0]).glob("*.jpg"))
        orc = {}
        with torch.inference_mode():
            for t in range(1, len(jpgs)):
                ipred.set_image(np.asarray(Image.open(jpgs[t]).convert("RGB")))
                for u in todo:
                    orc[(u.name, t)] = image_oracle(ipred, gt_of(u, u.frames[t]))
        for u in todo:
            masks, lat, rec, wall, vram = runs[u.name]
            rows, prev = [], (None, None)
            for t in range(1, len(u.frames)):
                g = gt_of(u, u.frames[t])
                rows.append({"unit": u.name, "video_idx": t, **frame_metrics(masks[t], g, *prev, other_gt(u, u.frames[t])),
                             "J_image_oracle": orc[(u.name, t)], "obj_score": rec[t]["obj_score"], "latency_s": lat[t]})
                prev = (masks[t], g)
            stem = u.name.replace("/", "_")
            atomic_npz(d / f"{stem}.npz", masks=np.packbits(np.stack([masks[t] for t in range(1, len(u.frames))])),
                       shape=np.array([len(u.frames) - 1, *masks[1].shape]))
            df = pd.DataFrame(rows).assign(wall_s=wall, peak_vram_bytes=vram)
            atomic_csv(df, d / f"{stem}.csv")
            print(f"{u.name}: J {df.J.mean():.3f}  ceil {df.J_ceil256.mean():.3f}  oracle {df.J_image_oracle.mean():.3f}", flush=True)


# ----------------------------------------------------------------------------- 5. prompt ablations
@torch.inference_mode()
def start_variant(pred, vdir, gt0: np.ndarray, mode=None, thr=0.5, fullres_mem=False, decoder_path=False):
    """engine.start with exactly one prompt-path factor changed. mode/thr: we resize the prompt to 1024 ourselves
    (add_new_mask then skips its own resize). fullres_mem: encode the prompt memory from the 1024 mask instead of the
    official 256x256 round trip. decoder_path: use_mask_input_as_output_without_sam=False (caller restores)."""
    mask = gt0 if mode is None and thr == 0.5 else detail.to_model(gt0, mode or "official", thr)[0, 0].numpy() > 0.5
    pred.use_mask_input_as_output_without_sam = not decoder_path
    state = pred.init_state(video_path=str(vdir))
    pred.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask)
    if fullres_mem:
        out = state["temp_output_dict_per_obj"][0]["cond_frame_outputs"][0]
        hi = state["mask_inputs_per_obj"][0][0].to(state["device"]) * 20.0 - 10.0
        out["maskmem_features"], out["maskmem_pos_enc"] = pred._run_memory_encoder(
            state, 0, 1, hi, out["object_score_logits"], is_mask_from_pts=True)
    pred.propagate_in_video_preflight(state)
    return state, state["output_dict_per_obj"][0]


def ablate():
    m = load_manifest()
    pred = build_predictor()
    d = OUT / "ablate"
    d.mkdir(parents=True, exist_ok=True)
    us = {u.name: u for u in davis_units("val")[0]}
    for name in [r["unit"] for r in m["rows"] if r["split"] == "val"]:
        f = d / f"{name.replace('/', '_')}.csv"
        if f.exists():
            continue
        u, rows = us[name], []
        gt0 = davis_gt(u, u.frames[0])
        last = min(m["ablation_frames"], len(u.frames) - 1)
        for var, kw in m["ablations"].items():
            hook = MemEncHook(pred)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            state, od = start_variant(pred, davis_frame_dir(u), gt0, **kw)
            out0 = engine.prompt_mask_output(pred, state)
            masks, lat = engine.rollout(pred, state, od, 1, last)
            wall = time.perf_counter() - t0
            pred.reset_state(state)
            pred.use_mask_input_as_output_without_sam = True
            hook.remove()
            mem0 = (hook.seen[0][0] > 0).float() * 20 - 10
            init = {"J_prompt_1024": j_measure(detail.model_roundtrip(gt0, kw.get("mode", "official"), kw.get("thr", 0.5)), gt0),
                    "J_prompt_output": j_measure(out0, gt0), "J_prompt_memenc_input": j_measure(detail.to_video(mem0, gt0.shape), gt0),
                    "thin_recall_memenc_input": detail.thin_recall(detail.to_video(mem0, gt0.shape), gt0)}
            prev = (None, None)
            for t in range(1, last + 1):
                g = davis_gt(u, u.frames[t])
                pr = masks[t]
                rows.append({"unit": name, "variant": var, "video_idx": t, **init, **jf(pr, g), "gt_empty": not g.any(),
                             "thin_recall": detail.thin_recall(pr, g), "comp_retention": detail.component_retention(pr, g),
                             "pred_area": int(pr.sum()), "gt_area": int(g.sum()),
                             "Fs_consec_pred": f_measure(pr, prev[0], 2) if prev[0] is not None else np.nan,
                             "latency_s": lat[t], "wall_s": wall, "peak_vram_bytes": torch.cuda.max_memory_allocated()})
                prev = (pr, g)
        atomic_csv(pd.DataFrame(rows), f)
        D = pd.DataFrame(rows).groupby("variant").JF.mean()
        print(name, " ".join(f"{k[:3]}={v:.3f}" for k, v in D.items()), flush=True)


def _boot(df, col, group="video", n=2000):
    from rfeas.risk import cluster_bootstrap
    return cluster_bootstrap(df, group, lambda d: float(d[col].mean()), n=n)


def classify(r, med_area) -> str:
    """Pre-registered failure classes (CLASS_RULES), first match wins. Uses retrospective GT: diagnostic only."""
    if r.gt_area < 0.25 * med_area or r.other_obj_frac >= 0.30:
        return "F_occlusion_identity"
    if r.J_ceil256 < 0.5:
        return "D_output_resolution"
    if r.J_image_oracle < 0.5:
        return "BC_encoder_decoder"
    if r.J_image_oracle >= r.J + 0.2:
        return "E_memory_propagation"
    return "unresolved"


def load_survey(split):
    m = {r["unit"]: r for r in load_manifest(split)["rows"] if r["split"] == split}
    D = pd.concat([pd.read_csv(f) for f in sorted((OUT / "survey" / split).glob("*.csv"))], ignore_index=True)
    st = pd.DataFrame(m.values())[["unit", "area_frac", "thin_frac", "n_comp20", "small", "thin", "multi", "detail"]]
    D = D.merge(st, on="unit")
    D["video"] = D.unit.str.split("__").str[1]
    return D, len(m)


def survey_report(split):
    D, n_manifest = load_survey(split)
    vis = D[~D.gt_empty].copy()
    med = vis.groupby("unit").gt_area.median()
    vis["med_area"] = vis.unit.map(med)
    U = vis.groupby(["unit", "video"]).agg(J=("J", "mean"), F=("F", "mean"), JF=("JF", "mean"), F_strict=("F_strict", "mean"),
                                           J_ceil256=("J_ceil256", "mean"), Fs_ceil256=("F_strict_ceil256", "mean"),
                                           J_oracle=("J_image_oracle", "mean"), thin_recall=("thin_recall", "mean"),
                                           thin_recall_ceil=("thin_recall_ceil256", "mean"), comp_ret=("comp_retention", "mean"),
                                           comp_ret_ceil=("comp_retention_ceil256", "mean"), detail=("detail", "first"),
                                           small=("small", "first"), thin=("thin", "first"), multi=("multi", "first"),
                                           frames=("J", "size")).reset_index()
    U["headroom_ceil_minus_J"] = U.J_ceil256 - U.J
    rep = {"split": split, "units_done": int(D.unit.nunique()), "units_manifest": n_manifest, "frames_visible": len(vis),
           "frames_gt_empty": int(D.gt_empty.sum())}
    for name, sub in (("all", U), ("detail", U[U.detail]), ("plain", U[~U.detail]), ("thin", U[U.thin]), ("small", U[U.small])):
        if len(sub):
            rep[f"units_{name}"] = {"n": len(sub), **{c: float(sub[c].mean()) for c in
                                    ["J", "F", "F_strict", "JF", "J_ceil256", "Fs_ceil256", "J_oracle", "thin_recall",
                                     "thin_recall_ceil", "comp_ret", "comp_ret_ceil", "headroom_ceil_minus_J"]}}
    if U.detail.any() and (~U.detail).any():
        rep["detail_minus_plain_J"] = float(U[U.detail].J.mean() - U[~U.detail].J.mean())
        rep["detail_minus_plain_F_strict"] = float(U[U.detail].F_strict.mean() - U[~U.detail].F_strict.mean())
    # error share explained by the 256 output resolution: (1 - J_ceil) / (1 - J), pooled over frames
    for name, sub in (("all", vis), ("detail", vis[vis.detail]), ("plain", vis[~vis.detail])):
        if len(sub):
            rep[f"error_share_resolution_{name}"] = float((1 - sub.J_ceil256).sum() / max((1 - sub.J).sum(), 1e-9))
    # small-object IoU by GT area fraction of the frame
    bins = pd.cut(vis.gt_area, [0, 500, 2000, 8000, 32000, 10**9], labels=["<500px", "500-2k", "2k-8k", "8k-32k", ">32k"])
    rep["J_by_gt_area_px"] = vis.groupby(bins, observed=True).agg(frames=("J", "size"), J=("J", "mean"), J_ceil256=("J_ceil256", "mean"),
                                                                 F_strict=("F_strict", "mean")).reset_index().astype({"gt_area": str}).to_dict("records")
    # failures
    fail = vis[vis.J < FAIL_J].copy()
    fail["cls"] = [classify(r, r.med_area) for r in fail.itertuples()]
    rep["failure_frames"] = len(fail)
    rep["failure_units"] = int(fail.unit.nunique())
    rep["failure_classes"] = {c: {"frames": int((fail.cls == c).sum()), "units": int(fail[fail.cls == c].unit.nunique())}
                              for c, _ in CLASS_RULES}
    rep["failure_by_unit"] = fail.groupby(["unit", "cls"]).size().unstack(fill_value=0).reset_index().to_dict("records")
    # temporal: persistence and boundary stability
    pers = []
    for u, g in vis.sort_values("video_idx").groupby("unit"):
        f = (g.J < FAIL_J).to_numpy()
        if f[:-1].sum():
            pers.append(f[1:][f[:-1]].mean())
    rep["P_fail_next_given_fail"] = float(np.mean(pers)) if pers else None
    cons = D.dropna(subset=["Fs_consec_pred"])
    rep["boundary_consistency"] = {"Fs_consec_pred": float(cons.Fs_consec_pred.mean()), "Fs_consec_gt": float(cons.Fs_consec_gt.mean()),
                                   "detail_pred": float(cons[cons.detail].Fs_consec_pred.mean()) if cons.detail.any() else None,
                                   "detail_gt": float(cons[cons.detail].Fs_consec_gt.mean()) if cons.detail.any() else None}
    early_late = []
    for u, g in vis.sort_values("video_idx").groupby("unit"):
        if len(g) >= 20:
            h = (g.J_ceil256 - g.J).to_numpy()
            early_late.append({"unit": u, "gap_first10": float(h[:10].mean()), "gap_last10": float(h[-10:].mean())})
    EL = pd.DataFrame(early_late)
    rep["ceiling_gap_first10_vs_last10"] = {"first10": float(EL.gap_first10.mean()), "last10": float(EL.gap_last10.mean()),
                                           "units_gap_grew_gt_0.05": int(((EL.gap_last10 - EL.gap_first10) > 0.05).sum()), "n": len(EL)}
    rep["latency_s_per_frame_median"] = float(D.latency_s.median())
    rep["peak_vram_GB"] = float(D.peak_vram_bytes.max() / 1e9)
    rd = OUT / "reports"
    rd.mkdir(parents=True, exist_ok=True)
    atomic_csv(U, rd / f"survey_{split}_units.csv")
    atomic_csv(fail[["unit", "video_idx", "J", "J_ceil256", "J_image_oracle", "gt_area", "med_area", "other_obj_frac", "cls"]],
               rd / f"survey_{split}_failures.csv")
    atomic_json(rep, rd / f"survey_{split}_report.json")
    return rep


def ablation_report():
    D = pd.concat([pd.read_csv(f) for f in sorted((OUT / "ablate").glob("*.csv"))], ignore_index=True)
    st = pd.DataFrame(load_manifest()["rows"])[["unit", "detail", "thin", "small"]]
    D = D.merge(st, on="unit")
    D["video"] = D.unit.str.split("__").str[1]
    base = D[D.variant == "P0_official"].set_index(["unit", "video_idx"])
    rows, unit_rows = [], []
    for var, g in D.groupby("variant"):
        g = g.set_index(["unit", "video_idx"])
        if list(g.index) != list(base.index) or not (g.gt_empty == base.gt_empty).all():
            raise ValueError(f"{var}: frames not paired with P0")
        d = g[["J", "F", "JF", "F_strict", "thin_recall", "comp_retention"]] - base[["J", "F", "JF", "F_strict", "thin_recall", "comp_retention"]]
        d["video"], d["detail"] = g.video, g.detail
        first = d.xs(1, level="video_idx")
        per_unit = d.groupby(level="unit").agg({"J": "mean", "F": "mean", "JF": "mean", "F_strict": "mean", "thin_recall": "mean",
                                                "comp_retention": "mean", "video": "first", "detail": "first"}).reset_index()
        per_unit["variant"] = var
        unit_rows.append(per_unit)
        init = g.groupby(level="unit")[["J_prompt_1024", "J_prompt_output", "J_prompt_memenc_input", "thin_recall_memenc_input"]].first()
        r = {"variant": var, "n_units": len(per_unit),
             "J_prompt_1024": float(init.J_prompt_1024.mean()), "J_prompt_output": float(init.J_prompt_output.mean()),
             "J_prompt_memenc_input": float(init.J_prompt_memenc_input.mean()),
             "thin_recall_memenc_input": float(init.thin_recall_memenc_input.mean()),
             "dJ_first_frame": float(first.J.mean()), "dJF_20": float(per_unit.JF.mean()), "dJ_20": float(per_unit.J.mean()),
             "dF_20": float(per_unit.F.mean()), "dF_strict_20": float(per_unit.F_strict.mean()),
             "dthin_recall_20": float(per_unit.thin_recall.mean()), "dcomp_ret_20": float(per_unit.comp_retention.mean()),
             "dJF_20_detail": float(per_unit[per_unit.detail].JF.mean()), "dJF_20_plain": float(per_unit[~per_unit.detail].JF.mean()),
             "units_up_0.01": int((per_unit.JF > 0.01).sum()), "units_down_0.01": int((per_unit.JF < -0.01).sum()),
             "dJF_20_CI": _boot(per_unit, "JF") if var != "P0_official" else (0.0, 0.0),
             "wall_s_per_unit": float(g.groupby(level="unit").wall_s.first().mean()),
             "latency_s_median": float(g.latency_s.median()), "peak_vram_GB": float(g.peak_vram_bytes.max() / 1e9)}
        rows.append(r)
    R = pd.DataFrame(rows)
    rd = OUT / "reports"
    rd.mkdir(parents=True, exist_ok=True)
    atomic_csv(R, rd / "ablation_summary.csv")
    atomic_csv(pd.concat(unit_rows), rd / "ablation_per_unit.csv")
    return R



def proto_report(split):
    D = pd.concat([pd.read_csv(f) for f in sorted((OUT / "proto" / split).glob("*.csv"))], ignore_index=True)
    st = pd.DataFrame(load_manifest(split)["rows"])[["unit", "detail", "thin", "small"]]
    D = D.merge(st, on="unit")
    D["video"] = D.unit.str.split("__").str[1]
    D["late"] = D.video_idx > 20
    base = D[D.arm == "official"].set_index(["unit", "video_idx"])
    cols = ["J", "F", "JF", "F_strict", "thin_recall", "comp_retention", "J_consec_pred"]
    rows, per_unit = [], []
    for arm, g in D.groupby("arm"):
        g = g.set_index(["unit", "video_idx"])
        if list(g.index) != list(base.index) or not (g.gt_empty == base.gt_empty).all():
            raise ValueError(f"{arm}: frames not paired with official")
        d = g[cols] - base[cols]
        d["video"], d["detail"], d["late"] = g.video, g.detail, g.late
        pu = d.groupby(level="unit").agg({**{c: "mean" for c in cols}, "video": "first", "detail": "first"}).reset_index()
        early = d[~d.late].groupby(level="unit").JF.mean()
        late = d[d.late].groupby(level="unit").JF.mean()
        pu["JF_frames1_20"], pu["JF_frames21_end"] = pu.unit.map(early), pu.unit.map(late)
        pu["arm"] = arm
        per_unit.append(pu)
        rows.append({"arm": arm, "n_units": len(pu), "abs_JF": float(g.groupby(level="unit").JF.mean().mean()),
                     **{f"d{c}": float(pu[c].mean()) for c in cols},
                     "dJF_CI": _boot(pu, "JF") if arm != "official" else (0.0, 0.0),
                     "dJF_detail": float(pu[pu.detail].JF.mean()) if pu.detail.any() else None,
                     "dJF_plain": float(pu[~pu.detail].JF.mean()),
                     "dJF_frames1_20": float(pu.JF_frames1_20.mean()), "dJF_frames21_end": float(pu.JF_frames21_end.mean()),
                     "dJF_first_frame": float(d.xs(1, level="video_idx").JF.mean()),
                     "units_up_0.01": int((pu.JF > 0.01).sum()), "units_down_0.01": int((pu.JF < -0.01).sum()),
                     "latency_s_median": float(g.latency_s.median()), "peak_vram_GB": float(g.peak_vram_bytes.max() / 1e9)})
    R = pd.DataFrame(rows)
    rd = OUT / "reports"
    atomic_csv(R, rd / f"proto_{split}_summary.csv")
    atomic_csv(pd.concat(per_unit), rd / f"proto_{split}_per_unit.csv")
    return R


def analyze():
    for split in ("val", "segtrack"):
        if (OUT / "survey" / split).exists():
            print(json.dumps(survey_report(split), indent=1, default=float)[:6000])
    if (OUT / "ablate").exists():
        print(ablation_report().to_string())
    for split in ("val", "segtrack"):
        if (OUT / "proto" / split).exists():
            print(proto_report(split).to_string())


# ----------------------------------------------------------------------------- 7. prototype: guided upsampling
TUNE_GRID = {"r": [1, 2, 4, 8], "eps": [1e-4, 1e-3, 1e-2, 1e-1]}
TUNE_FRAMES = 20
N_TUNE_PER_STRATUM = 12


def tune_units():
    """Deterministic DAVIS-TRAIN subset (never val/SegTrack): first 12 detail + 12 plain units by a fixed hash."""
    rows = [r for r in load_manifest()["rows"] if r["split"] == "train"]
    key = lambda r: stable_hash(f"m55-tune|{r['unit']}")
    return [r["unit"] for r in sorted([r for r in rows if r["detail"]], key=key)[:N_TUNE_PER_STRATUM]] + \
           [r["unit"] for r in sorted([r for r in rows if not r["detail"]], key=key)[:N_TUNE_PER_STRATUM]]


def tune():
    """Pick (r, eps) for the guided output on DAVIS train only; output arm (memory untouched) so one native run
    per unit suffices. Writes reports/proto_tune.json; the chosen values are then frozen in manifest_proto.json."""
    from spatial import proto
    pred = build_predictor()
    us = {u.name: u for u in davis_units("train")[0]}
    store = {}
    rows = []
    for name in tune_units():
        u = us[name]
        state, od = engine.start(pred, frame_dir(u), gt_of(u, u.frames[0]))
        jpgs = sorted(frame_dir(u).glob("*.jpg"))
        last = min(TUNE_FRAMES, len(u.frames) - 1)
        with torch.inference_mode():
            for t in range(1, last + 1):
                out, pm = pred._run_single_frame_inference(
                    inference_state=state, output_dict=od, frame_idx=t, batch_size=1, is_init_cond_frame=False,
                    point_inputs=None, mask_inputs=None, reverse=False, run_mem_encoder=True)
                od["non_cond_frame_outputs"][t] = out
                g, guide = gt_of(u, u.frames[t]), proto.grey(jpgs[t], "cuda")
                for r in TUNE_GRID["r"]:
                    for eps in TUNE_GRID["eps"]:
                        q = proto.refined_prob(pm.to("cuda"), guide, r, eps)[0, 0] > 0.5
                        rows.append({"unit": name, "video_idx": t, "r": r, "eps": eps, **jf(q.cpu().numpy(), g)})
                _, vr = pred._get_orig_video_res_output(state, pm)
                rows.append({"unit": name, "video_idx": t, "r": 0, "eps": 0.0, **jf((vr[0, 0] > 0).cpu().numpy(), g)})
        pred.reset_state(state)
        print(name, flush=True)
    D = pd.DataFrame(rows)
    U = D.groupby(["r", "eps", "unit"]).JF.mean().groupby(level=["r", "eps"]).mean().reset_index()
    best = U[U.r > 0].sort_values("JF", ascending=False).iloc[0]
    rep = {"units": tune_units(), "grid": U.to_dict("records"), "official_JF": float(U[U.r == 0].JF.iloc[0]),
           "chosen": {"r": int(best.r), "eps": float(best.eps), "JF": float(best.JF)}}
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    atomic_csv(D, OUT / "reports" / "proto_tune_frames.csv")
    atomic_json(rep, OUT / "reports" / "proto_tune.json")
    print(json.dumps(rep["chosen"]), "official", rep["official_JF"])



def proto_manifest(simple_variant: str):
    """Freeze the prototype comparison BEFORE running it: chosen (r, eps) from DAVIS-train tuning, the simple
    correction (selected from the val 20-frame ablation -- disclosed), arms, units, metrics."""
    t = json.loads((OUT / "reports" / "proto_tune.json").read_text())
    _freeze(OUT / "manifest_proto.json", {
        "rows": [{"unit": r["unit"], "split": r["split"]} for sp in ("val", "segtrack") for r in load_manifest(sp)["rows"]
                 if r["split"] == sp],
        "guided": {"r": t["chosen"]["r"], "eps": t["chosen"]["eps"], "selected_on": "DAVIS train subset (tune_units), output arm"},
        "simple_correction": {"variant": simple_variant, "kwargs": ABLATIONS[simple_variant],
                              "selected_on": "DAVIS val 20-frame ablation (same units as primary eval: disclosed leak; "
                                             "SegTrack is untouched)"},
        "arms": ["official", "simple", "guided_out", "guided_mem", "simple_guided_mem"],
        "arm_notes": {"simple_guided_mem": "combination (two factors) -- reported separately, not a single-factor test"},
        "frames": "all frames of each unit"})


def proto_run(split):
    from spatial import proto
    m = json.loads((OUT / "manifest_proto.json").read_text())
    r, eps = m["guided"]["r"], m["guided"]["eps"]
    simple_kw = m["simple_correction"]["kwargs"]
    pred = build_predictor()
    us = {u.name: u for u in units_of(split)}
    d = OUT / "proto" / split
    d.mkdir(parents=True, exist_ok=True)
    for name in [x["unit"] for x in m["rows"] if x["split"] == split]:
        f = d / f"{name.replace('/', '_')}.csv"
        if f.exists():
            continue
        u = us[name]
        n = len(u.frames)
        gt0 = gt_of(u, u.frames[0])
        jpgs = sorted(frame_dir(u).glob("*.jpg"))
        masks, lat, vram = {a: {} for a in m["arms"]}, {a: {} for a in m["arms"]}, {}
        for arm in ("official", "guided_mem", "simple", "simple_guided_mem"):
            torch.cuda.reset_peak_memory_stats()
            if arm.startswith("simple"):
                state, od = start_variant(pred, frame_dir(u), gt0, **simple_kw)
                pred.use_mask_input_as_output_without_sam = True
            else:
                state, od = engine.start(pred, frame_dir(u), gt0)
            for t in range(1, n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if arm == "simple":
                    out, mk, _, _ = engine.predict(pred, state, od, t)
                else:
                    out, mk, off = proto.predict(pred, state, od, t, proto.grey(jpgs[t], "cuda"), r, eps,
                                                 feed_memory=arm.endswith("guided_mem"))
                od["non_cond_frame_outputs"][t] = out
                torch.cuda.synchronize()
                lat[arm][t] = time.perf_counter() - t0
                if arm == "official":
                    masks["official"][t], masks["guided_out"][t] = off, mk
                    lat["guided_out"][t] = lat[arm][t]
                else:
                    masks[arm][t] = mk
            vram[arm] = torch.cuda.max_memory_allocated()
            pred.reset_state(state)
        vram["guided_out"] = vram["official"]
        rows = []
        for arm in m["arms"]:
            prev = None
            for t in range(1, n):
                g, pr = gt_of(u, u.frames[t]), masks[arm][t]
                rows.append({"unit": name, "arm": arm, "video_idx": t, **jf(pr, g), "gt_empty": not g.any(),
                             "gt_area": int(g.sum()), "pred_area": int(pr.sum()),
                             "thin_recall": detail.thin_recall(pr, g), "comp_retention": detail.component_retention(pr, g),
                             "Fs_consec_pred": f_measure(pr, prev, 2) if prev is not None else np.nan,
                             "J_consec_pred": j_measure(pr, prev) if prev is not None else np.nan,
                             "latency_s": lat[arm][t], "peak_vram_bytes": vram[arm]})
                prev = pr
        atomic_csv(pd.DataFrame(rows), f)
        S = pd.DataFrame(rows).groupby("arm").JF.mean()
        print(name, " ".join(f"{k}={v:.4f}" for k, v in S.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["audit", "manifest", "survey", "ablate", "analyze", "tune", "proto", "proto_manifest"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--simple", default="P5_fullres_prompt_memory")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "survey":
        return survey(a.split)
    if a.what == "manifest":
        return manifest(a.split if a.split == "segtrack" else None)
    if a.what == "proto":
        return proto_run(a.split)
    if a.what == "proto_manifest":
        return proto_manifest(a.simple)
    {"audit": audit, "ablate": ablate, "analyze": analyze, "tune": tune}[a.what]()


if __name__ == "__main__":
    main()
