"""Milestone 4 runner: official SAM 2.1 Tiny on the 96 M3 frame-drop manifests (resumable).

    uv run python scripts/run_dropout_baseline.py --all                      # run / resume everything
    uv run python scripts/run_dropout_baseline.py --scene Scene01 --condition burst_50 --seed 0
    uv run python scripts/run_dropout_baseline.py --all --status             # validate only, no inference
    uv run python scripts/run_dropout_baseline.py --all --eval-only          # recompute metrics from saved masks

Completed runs are validated (record, manifest sha, model, prediction sha/frames) and skipped;
invalid ones are moved to outputs/milestone4/invalid/ and repeated. Runs are sequential.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from importlib.util import find_spec  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from baseline import frame_drops as fd  # noqa: E402
from baseline.dropout_runner import (atomic_json, experiment_matrix, manifest_path, reevaluate_run,  # noqa: E402
                                     run_dir, run_matrix, validate_run)
from baseline.evaluation import sha256  # noqa: E402
from vkitti.dataset import VKittiDataset  # noqa: E402
from vkitti.tracks import select_tracks, summarize_tracks  # noqa: E402

OFFICIAL_FILES = ["modeling/sam2_base.py", "sam2_video_predictor.py", "modeling/memory_attention.py",
                  "modeling/memory_encoder.py", "configs/sam2.1/sam2.1_hiera_t.yaml", "build_sam.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m4_dropout_baseline.yaml")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scene")
    ap.add_argument("--track", type=int)
    ap.add_argument("--condition")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--status", action="store_true", help="validate saved runs only")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    if not (a.all or a.scene or a.track is not None or a.condition or a.seed is not None):
        ap.error("select runs with --all or filters")

    cfg = yaml.safe_load(Path(a.config).read_text())
    m3 = yaml.safe_load(Path(cfg["m3_config"]).read_text())
    m2 = yaml.safe_load(Path(m3["m2_config"]).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    sc = m2["sam2"]

    # Data: the same tracks as M1-M3, or stop.
    datasets = {}
    for scene, expected in cfg["tracks"].items():
        ds = VKittiDataset(m2["root"], scene, m2["variation"], m2["camera_id"])
        H, W = ds.image_shape
        t = select_tracks(summarize_tracks(ds.mask_pixel_counts(), ds.info, ds.bbox, H * W,
                                           m2["track_selection"]["min_area_frac"]), H * W, **m2["track_selection"])
        if t.track_id.tolist() != expected:
            raise SystemExit(f"{scene}: selected tracks {t.track_id.tolist()} != {expected}")
        datasets[scene] = ds
    image_shape = next(iter(datasets.values())).image_shape

    cond_src = yaml.safe_load(Path(cfg["conditions_from"]).read_text()) if "conditions_from" in cfg else m3
    specs = [s for s in experiment_matrix(cfg["tracks"], list(cond_src["conditions"]), cond_src["seeds"])
             if (a.scene in (None, s.scene)) and (a.track in (None, s.track_id))
             and (a.condition in (None, s.condition)) and (a.seed in (None, s.seed))]

    import torch
    pkg = Path(find_spec("sam2").origin).parent
    model_info = {
        "sam2_upstream_commit": sc["upstream_commit"], "checkpoint_sha256": sha256(Path(sc["checkpoint"])),
        "config": sc["config"], "apply_postprocessing": sc["apply_postprocessing"], "device": cfg["device"],
        "precision": "float32 (no autocast)", "torch": torch.__version__, "python": platform.python_version(),
        "sam2_file_sha256": {f: hashlib.sha256((pkg / f).read_bytes()).hexdigest() for f in OFFICIAL_FILES},
    }
    if model_info["checkpoint_sha256"] != sc["checkpoint_sha256"]:
        raise SystemExit("checkpoint sha256 differs from the M2 configuration")
    base_ctx = {"manifests": Path(cfg["manifests_dir"]), "out": out, "m2_dir": Path(m3["m2_output_dir"]),
                "sam2_cfg": sc, "eval_cfg": m2["evaluation"], "rec_cfg": m3["recovery"], "model_info": model_info,
                "min_free_percent": cfg["min_free_memory_percent"]}

    if a.status:
        bad = []
        for s in specs:
            ok, why = validate_run(run_dir(out, s), fd.load(manifest_path(base_ctx["manifests"], s)), model_info,
                                   image_shape)
            if not ok:
                bad.append((s.name, why))
        print(f"{len(specs) - len(bad)}/{len(specs)} valid")
        for n, w in bad:
            print(f"  INVALID {n}: {w}")
        return

    if a.eval_only:
        for s in specs:
            reevaluate_run(s, {**base_ctx, "ds": datasets[s.scene]})
            print(f"re-evaluated {s.name}", flush=True)
        return

    from sam2.build_sam import build_sam2_video_predictor
    t0 = time.perf_counter()
    predictor = build_sam2_video_predictor(sc["config"], sc["checkpoint"], device=torch.device(cfg["device"]),
                                           apply_postprocessing=sc["apply_postprocessing"])
    build_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    status = run_matrix(specs, lambda s: {**base_ctx, "ds": datasets[s.scene], "predictor": predictor}, out,
                        model_info, image_shape, log=lambda m: print(m, flush=True))
    session = {"started_unix": time.time() - (time.perf_counter() - t0), "build_s": build_s,
               "loop_wall_s": time.perf_counter() - t1, "model_info": model_info, "runs": status,
               "selected": len(specs)}
    sessions = out / "sessions"
    sessions.mkdir(exist_ok=True)
    atomic_json(session, sessions / f"session_{int(time.time())}.json")
    n_ok = sum(v in ("completed", "skipped_valid") for v in status.values())
    print(f"{n_ok}/{len(specs)} runs valid after this session")


if __name__ == "__main__":
    main()
