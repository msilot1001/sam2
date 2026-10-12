"""Milestone 4: resumable runner for the 96 frame-drop baseline experiments.

One run = one M3 manifest (scene, track, condition, seed) through the unmodified SAM 2 predictor.
Run directory: <out>/runs/<scene>_trackNNN/<condition>_seed<s>/
  predictions.npz   masks (R, H, W) bool + original_frame_ids (received order)
  per_frame.csv     evaluate() rows for received frames (prompt included, flagged)
  paired.csv        matched no-drop vs dropout, non-prompt received frames
  events.csv        dropped-run recovery events (J rule + visible-target + no-drop diagnostics)
  latency.csv       per video_idx latency
  record.json       written LAST, atomically: the completion marker
A run is skipped only if validate_run() passes; an invalid directory is moved (not deleted) to
<out>/invalid/ and the run is repeated. Failures are appended to <out>/failures.jsonl.
"""
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import frame_drops as fd
from .dropout_eval import check_m2_reference, paired_frames, recovery_events
from .evaluation import evaluate
from .sam2_vos import clip_from_pattern, run_clip, verify_clip_folder, write_clip_folder

RECORD_SCHEMA = 1


@dataclass(frozen=True)
class RunSpec:
    scene: str
    track_id: int
    condition: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.scene}_track{self.track_id:03d}/{self.condition}_seed{self.seed}"


def experiment_matrix(tracks: dict, conditions, seeds) -> list[RunSpec]:
    """tracks: {scene: [track ids]} in fixed order -> scene x track x condition x seed."""
    return [RunSpec(s, t, c, sd) for s, ts in tracks.items() for t in ts for c in conditions for sd in seeds]


def run_dir(out: Path, spec: RunSpec) -> Path:
    return Path(out) / "runs" / spec.name


def manifest_path(manifests: Path, spec: RunSpec) -> Path:
    return fd.paths(manifests, spec.scene, spec.track_id, f"{spec.condition}_seed{spec.seed}")[0]


# ----------------------------------------------------------------------------- atomic I/O
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    atomic_write_bytes(path, df.to_csv(index=False, lineterminator="\n").encode())


def atomic_json(obj, path: Path) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, default=_json_default) + "\n").encode())


def atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# ----------------------------------------------------------------------------- validation
def validate_run(d: Path, pattern, model_info: dict, image_shape=None) -> tuple[bool, str]:
    """True only if the run directory is a complete, consistent result for this manifest+model."""
    d = Path(d)
    rec_p = d / "record.json"
    if not rec_p.exists():
        return False, "no record.json"
    try:
        rec = json.loads(rec_p.read_text())
    except json.JSONDecodeError as e:
        return False, f"record.json unreadable: {e}"
    if rec.get("schema") != RECORD_SCHEMA or rec.get("status") != "complete":
        return False, f"status {rec.get('status')}"
    if rec["manifest"]["frames_csv_sha256"] != pattern.csv_sha256:
        return False, "manifest changed since the run"
    for k in ("sam2_upstream_commit", "checkpoint_sha256", "config", "device", "precision", "apply_postprocessing"):
        if rec["model"].get(k) != model_info.get(k):
            return False, f"model mismatch: {k}"
    npz = d / "predictions.npz"
    if not npz.exists():
        return False, "predictions.npz missing"
    if sha256_file(npz) != rec["predictions_sha256"]:
        return False, "predictions.npz corrupt (sha256)"
    try:
        z = np.load(npz)
        masks, ids = z["masks"], z["original_frame_ids"]
    except Exception as e:  # noqa: BLE001 - any read error means corrupt
        return False, f"predictions unreadable: {e}"
    received = list(pattern.received_ids)
    if ids.tolist() != received:
        return False, "prediction frame ids != manifest received frames"
    if masks.dtype != bool or masks.ndim != 3 or masks.shape[0] != len(received):
        return False, f"bad mask array {masks.dtype} {masks.shape}"
    if image_shape is not None and tuple(masks.shape[1:]) != tuple(image_shape):
        return False, f"mask size {masks.shape[1:]} != {image_shape}"
    try:
        pf = pd.read_csv(d / "per_frame.csv")
        pr = pd.read_csv(d / "paired.csv")
    except FileNotFoundError as e:
        return False, f"missing {e.filename}"
    if pf.original_frame_id.tolist() != received or pr.original_frame_id.tolist() != received[1:]:
        return False, "metric tables do not cover exactly the received frames"
    return True, ""


# ----------------------------------------------------------------------------- one run
def system_free_percent() -> float | None:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return float(m.group(1)) if m else None
    except (OSError, subprocess.SubprocessError):
        return None


def wait_for_memory(min_free: float, tries: int = 5, sleep_s: float = 60.0) -> float | None:
    for _ in range(tries):
        free = system_free_percent()
        if free is None or free >= min_free:
            return free
        time.sleep(sleep_s)
    raise MemoryError(f"system memory free {free}% < {min_free}% after {tries} checks")


def evaluate_run(ds, pattern, clip, masks: dict, ref: pd.DataFrame, eval_cfg: dict, rec_cfg: dict):
    """(per_frame, paired, events) for one run. GT is read here only, after inference."""
    per_frame = evaluate(ds, clip, masks, eval_cfg)
    paired = paired_frames(ref, per_frame)
    paired = paired.merge(per_frame[["original_frame_id", "gt_pixels", "pred_pixels"]], on="original_frame_id")
    paired = paired.merge(ref[["original_frame_id", "pred_pixels"]].rename(columns={"pred_pixels": "pred_pixels_no_drop"}),
                          on="original_frame_id")
    j = dict(zip(per_frame.original_frame_id, per_frame.J))
    vis = dict(zip(per_frame.original_frame_id, ~per_frame.gt_empty))
    events = recovery_events(pattern.frames, j, rec_cfg["j_threshold"], eval_cfg["lost_j"],
                             rec_cfg["within_received_frames"], visible_by_frame=vis,
                             ref_j_by_frame=dict(zip(ref.original_frame_id, ref.J)))
    return per_frame, paired, events


def run_one(spec: RunSpec, ctx: dict) -> dict:
    """Inference + evaluation + atomic save for one manifest. ctx keys: predictor, ds, manifests,
    out, m2_dir, sam2_cfg, eval_cfg, rec_cfg, model_info, min_free_percent."""
    import torch

    pattern = fd.load(manifest_path(ctx["manifests"], spec))
    ref = check_m2_reference(ctx["m2_dir"], pattern, ctx["sam2_cfg"], ctx["eval_cfg"], ctx["eval_cfg"])
    ds = ctx["ds"]
    clip = clip_from_pattern(pattern)
    vdir = write_clip_folder(ds, clip, Path(ctx["out"]) / "clips")
    verify_clip_folder(ds, clip, vdir)
    dropped = {ds.rgb_path(f).resolve() for f in pattern.meta["dropped_original_frame_ids"]}
    if dropped & {p.resolve() for p in vdir.glob("*.jpg")}:
        raise RuntimeError("a dropped frame is present in the clip folder")
    free_before = wait_for_memory(ctx.get("min_free_percent", 10))
    prompt = ds.get_frame(clip.original_frame_ids[0], clip.track_id)["mask"]   # the only GT given to SAM 2
    t0 = time.perf_counter()
    res = run_clip(ctx["predictor"], vdir, prompt)
    wall = time.perf_counter() - t0
    masks = res["masks"]
    if sorted(masks) != list(range(len(clip.original_frame_ids))):
        raise RuntimeError("SAM 2 output indices != received video indices")
    per_frame, paired, events = evaluate_run(ds, pattern, clip, masks, ref, ctx["eval_cfg"], ctx["rec_cfg"])

    d = run_dir(ctx["out"], spec)
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").unlink(missing_ok=True)   # incomplete until the record is rewritten
    atomic_npz(d / "predictions.npz", masks=np.stack([masks[i] for i in range(len(masks))]),
               original_frame_ids=np.array(clip.original_frame_ids))
    atomic_csv(per_frame, d / "per_frame.csv")
    atomic_csv(paired, d / "paired.csv")
    atomic_csv(events, d / "events.csv")
    lat = pd.Series(res["latency_s"]).sort_index()
    atomic_csv(pd.DataFrame({"video_idx": lat.index, "original_frame_id": clip.original_frame_ids,
                             "latency_s": lat.values}), d / "latency.csv")
    record = {
        "schema": RECORD_SCHEMA, "status": "complete", "spec": spec.__dict__, "variant": clip.variant,
        "manifest": {"json": str(manifest_path(ctx["manifests"], spec)),
                     "frames_csv_sha256": pattern.csv_sha256,
                     "actual_drop_rate": pattern.meta["actual_drop_rate"],
                     "received": pattern.meta["total_received_frames"], "dropped": pattern.meta["total_dropped_frames"]},
        "model": ctx["model_info"], "predictions_sha256": sha256_file(d / "predictions.npz"),
        "timing": {"wall_s": wall, "init_s": res["init_s"], "prompt_frame_s": float(lat.iloc[0]),
                   "latency_mean_s": float(lat.iloc[1:].mean()) if len(lat) > 1 else None,
                   "latency_median_s": float(lat.iloc[1:].median()) if len(lat) > 1 else None,
                   "latency_steady_mean_s": float(lat.iloc[7:].mean()) if len(lat) > 7 else None,
                   "frames": int(len(lat))},
        "memory": {"mps_driver_peak_bytes": max(res["mps_driver_bytes"], default=None),
                   "system_free_percent_before": free_before},
        "finished_unix": time.time(),
    }
    atomic_json(record, d / "record.json")
    del res, masks
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return record


def reevaluate_run(spec: RunSpec, ctx: dict) -> dict:
    """--eval-only: recompute tables from saved predictions; keeps timing/memory of the inference run."""
    d = run_dir(ctx["out"], spec)
    pattern = fd.load(manifest_path(ctx["manifests"], spec))
    ok, why = validate_run(d, pattern, ctx["model_info"])
    if not ok:
        raise RuntimeError(f"cannot re-evaluate invalid run: {why}")
    ref = check_m2_reference(ctx["m2_dir"], pattern, ctx["sam2_cfg"], ctx["eval_cfg"], ctx["eval_cfg"])
    clip = clip_from_pattern(pattern)
    z = np.load(d / "predictions.npz")
    masks = {i: m for i, m in enumerate(z["masks"])}
    per_frame, paired, events = evaluate_run(ctx["ds"], pattern, clip, masks, ref, ctx["eval_cfg"], ctx["rec_cfg"])
    rec = json.loads((d / "record.json").read_text())
    (d / "record.json").unlink()
    atomic_csv(per_frame, d / "per_frame.csv")
    atomic_csv(paired, d / "paired.csv")
    atomic_csv(events, d / "events.csv")
    rec["reevaluated_unix"] = time.time()
    atomic_json(rec, d / "record.json")
    return rec


def quarantine(d: Path, out: Path, reason: str) -> Path | None:
    """Move an invalid run directory aside (never delete results)."""
    if not Path(d).exists():
        return None
    dest = Path(out) / "invalid" / f"{Path(d).parent.name}__{Path(d).name}__{int(time.time())}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), dest)
    (dest / "INVALID_REASON.txt").write_text(reason + "\n")
    return dest


def log_failure(out: Path, spec: RunSpec, err: BaseException) -> None:
    with open(Path(out) / "failures.jsonl", "a") as f:
        f.write(json.dumps({"run": spec.name, "time": time.time(), "error": repr(err),
                            "traceback": traceback.format_exc()}) + "\n")


def run_matrix(specs, ctx_for, out: Path, model_info: dict, image_shape, run=run_one, log=print) -> dict:
    """Resumable loop. ctx_for(spec) -> ctx. Returns {run name: status}."""
    status = {}
    for i, spec in enumerate(specs, 1):
        d = run_dir(out, spec)
        try:
            ctx = ctx_for(spec)
            pattern = fd.load(manifest_path(ctx["manifests"], spec))
            ok, why = validate_run(d, pattern, model_info, image_shape)
            if ok:
                status[spec.name] = "skipped_valid"
                log(f"[{i}/{len(specs)}] {spec.name}: valid, skipped")
                continue
            if d.exists():
                moved = quarantine(d, out, why)
                log(f"[{i}/{len(specs)}] {spec.name}: invalid ({why}) -> {moved}")
            rec = run(spec, ctx)
            ok, why = validate_run(d, pattern, model_info, image_shape)
            if not ok:
                raise RuntimeError(f"run finished but failed validation: {why}")
            status[spec.name] = "completed"
            t = rec["timing"]
            log(f"[{i}/{len(specs)}] {spec.name}: done, {t['frames']} frames, {t['wall_s']:.1f}s")
        except Exception as e:  # noqa: BLE001 - record and continue with the next run
            status[spec.name] = f"failed: {e!r}"
            log_failure(out, spec, e)
            log(f"[{i}/{len(specs)}] {spec.name}: FAILED {e!r}")
    return status
