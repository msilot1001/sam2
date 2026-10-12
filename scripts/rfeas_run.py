"""Resumable SAM 2 runs for the risk/memory feasibility study.

    uv run python scripts/rfeas_run.py native --max-minutes 22          # all units, instrumented
    uv run python scripts/rfeas_run.py gated  --max-minutes 22          # CHECK-2 policies (needs risk model)
    uv run python scripts/rfeas_run.py single --max-minutes 22          # single-frame skip interventions

Each (unit, policy) is stored in outputs/research_feasibility/runs/<policy>/<unit>/ with
masks.npz, signals.csv and record.json (written last; a run is complete only if all three agree).
Ground truth is read only AFTER inference, for evaluation.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from baseline.dropout_runner import atomic_csv, atomic_json, atomic_npz, sha256_file  # noqa: E402
from baseline.metrics import f_measure, j_measure  # noqa: E402
from baseline.sam2_vos import Clip, write_clip_folder  # noqa: E402
from rfeas import gates  # noqa: E402
from rfeas.instrument import run_unit  # noqa: E402
from rfeas.units import (check2_select, davis_frame_dir, davis_gt, davis_units, split_of, stable_hash,  # noqa: E402
                         vkitti_units)

CFG = yaml.safe_load(Path("configs/rfeas.yaml").read_text())
OUT = Path(CFG["output_dir"])


def all_units():
    d, d_ex = davis_units(CFG["datasets"]["davis"]["split_file"])
    v, v_ex, dsets = vkitti_units(tuple(CFG["datasets"]["vkitti"]["scenes"]))
    return d + v, d_ex + v_ex, dsets


def check2_units(units):
    return check2_select(units, 12)


def prepare(u, dsets):
    """(frame directory for SAM 2, prompt mask, GT loader)."""
    if u.dataset == "davis":
        return davis_frame_dir(u), davis_gt(u, u.frames[0]), (lambda f: davis_gt(u, f))
    ds = dsets[u.video.split("/")[1]]
    clip = Clip(ds.scene, u.obj_id, u.frames, variant="rfeas")
    vdir = write_clip_folder(ds, clip, OUT / "clips")
    return vdir, ds.get_frame(u.frames[0], u.obj_id)["mask"], (lambda f: ds.get_frame(f, u.obj_id)["mask"])


def evaluate(u, masks, gt_of):
    rows = []
    for i, f in enumerate(u.frames):
        gt, pr = gt_of(f), masks[i]
        rows.append({"video_idx": i, "frame": f, "gt_pixels": int(gt.sum()), "pred_pixels": int(pr.sum()),
                     "gt_empty": not gt.any(), "J": j_measure(pr, gt), "F": f_measure(pr, gt),
                     "F_strict": f_measure(pr, gt, 2)})
    df = pd.DataFrame(rows)
    df["JF"] = (df.J + df.F) / 2
    return df


def run_dir(policy, u):
    return OUT / "runs" / policy / u.name


def valid(d: Path, u) -> bool:
    try:
        rec = json.loads((d / "record.json").read_text())
        return (rec["status"] == "complete" and rec["masks_sha256"] == sha256_file(d / "masks.npz")
                and len(pd.read_csv(d / "signals.csv")) == len(u.frames))
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def do_run(pred, u, policy, dsets, gate=None, extra=None):
    d = run_dir(policy, u)
    if valid(d, u):
        return "skipped"
    vdir, prompt, gt_of = prepare(u, dsets)
    t0 = time.perf_counter()
    res = run_unit(pred, vdir, prompt, gate=gate)
    wall = time.perf_counter() - t0
    if len(res["masks"]) != len(u.frames):
        raise RuntimeError(f"{u.name}: {len(res['masks'])} outputs for {len(u.frames)} frames")
    ev = evaluate(u, res["masks"], gt_of)                      # GT only after inference
    sig = pd.DataFrame([res["signals"][i] for i in range(len(u.frames))])
    sig["skipped_write"] = sig.video_idx.isin(res["skipped"])
    sig["latency_s"] = [res["latency"][i] for i in range(len(u.frames))]
    table = sig.merge(ev, on="video_idx")
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").unlink(missing_ok=True)
    atomic_npz(d / "masks.npz", masks=np.stack([res["masks"][i] for i in range(len(u.frames))]))
    atomic_csv(table, d / "signals.csv")
    atomic_json({"status": "complete", "unit": u.name, "policy": policy, "split": split_of(u.video),
                 "frames": len(u.frames), "skipped": res["skipped"], "wall_s": wall,
                 "masks_sha256": sha256_file(d / "masks.npz"), **(extra or {})}, d / "record.json")
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["native", "gated", "single"])
    ap.add_argument("--max-minutes", type=float, default=1e9)
    a = ap.parse_args()
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    m2 = yaml.safe_load(Path("configs/m2_baseline.yaml").read_text())["sam2"]
    if sha256_file(Path(m2["checkpoint"])) != m2["checkpoint_sha256"]:
        raise SystemExit("checkpoint sha256 mismatch")
    pred = build_sam2_video_predictor(m2["config"], m2["checkpoint"], device=torch.device(CFG["device"]),
                                      apply_postprocessing=m2["apply_postprocessing"])
    units, excluded, dsets = all_units()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"unit": u.name, "dataset": u.dataset, "video": u.video, "obj_id": u.obj_id,
                   "frames": len(u.frames), "split": split_of(u.video)} for u in units]).to_csv(OUT / "units.csv", index=False)
    pd.DataFrame(excluded, columns=["dataset", "video", "obj_id", "reason"]).to_csv(OUT / "excluded_units.csv", index=False)
    jobs = []
    if a.mode == "native":
        jobs = [(u, "native", None, None) for u in units]
    else:
        c2 = check2_units(units)
        model = gates.load_risk_model(OUT / "risk_model.json")
        if a.mode == "gated":
            for u in c2:
                jobs += [(u, "presence_gate", gates.presence_gate, None), (u, "iou_gate", gates.iou_gate, None),
                         (u, "consistency_gate", gates.consistency_gate, None),
                         (u, "risk_gate", gates.make_risk_gate(model), None)]
            for u in c2:   # needs the risk_gate run of the same unit for its budget
                jobs.append((u, "random_matched", "RANDOM", None))
        else:
            for u in c2:
                nat = pd.read_csv(run_dir("native", u) / "signals.csv")
                risk = gates.risk_scores(model, nat)
                t_hi = int(np.nanargmax(np.where(nat.video_idx > 0, risk, np.nan)))
                t_rand = 1 + stable_hash(f"rfeas-single|{u.name}") % (len(u.frames) - 1)
                for tag, t in (("single_maxrisk", t_hi), ("single_random", t_rand)):
                    jobs.append((u, tag, gates.skip_exactly(t), {"skip_frame": t}))
    t_start = time.time()
    for u, policy, gate, extra in jobs:
        if (time.time() - t_start) / 60 > a.max_minutes:
            print("time budget reached; rerun to resume", flush=True)
            break
        if gate == "RANDOM":
            rd = run_dir("risk_gate", u)
            if not valid(rd, u):
                print(f"{u.name}: risk_gate missing, random_matched postponed", flush=True)
                continue
            rec = json.loads((rd / "record.json").read_text())
            frac = len(rec["skipped"]) / max(len(u.frames) - 1, 1)
            gate, extra = gates.random_gate(frac, f"rfeas-random|{u.name}"), {"matched_fraction": frac}
        status = do_run(pred, u, policy, dsets, gate, extra)
        print(f"{policy} {u.name}: {status}", flush=True)
    n = sum(valid(run_dir(p, u), u) for u, p, _, _ in jobs)
    print(f"{n}/{len(jobs)} {a.mode} runs valid", flush=True)


if __name__ == "__main__":
    main()
