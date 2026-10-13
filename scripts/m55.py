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
    return build_sam2_video_predictor(CFG, CKPT, device=torch.device("cuda"), apply_postprocessing=True)


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


def analyze():
    raise SystemExit("analyze: added after the survey/ablation runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["audit", "manifest", "survey", "ablate", "analyze"])
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "survey":
        return survey(a.split)
    if a.what == "manifest":
        return manifest(a.split if a.split == "segtrack" else None)
    {"audit": audit, "ablate": ablate, "analyze": analyze}[a.what]()


if __name__ == "__main__":
    main()
