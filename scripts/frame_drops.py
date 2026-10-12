"""Milestone 3: generate/validate frame-drop manifests and run a short SAM 2 smoke test.

    uv run python scripts/frame_drops.py generate             # manifests + figures for all dev tracks
    uv run python scripts/frame_drops.py smoke --device mps   # SAM 2 on the configured dropped clips

Manifests (outputs/milestone3/manifests/<scene>_trackNNN/<condition>_seed<s>.{json,csv}) are the
source of truth for M4. Re-running never replaces an existing manifest with a different pattern.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from baseline import frame_drops as fd  # noqa: E402
from baseline.dropout_eval import (check_m2_reference, paired_frames, paired_summary,  # noqa: E402
                                   recovery_events, recovery_summary)
from baseline.evaluation import evaluate, sha256  # noqa: E402
from baseline.sam2_vos import clip_from_pattern, make_clip, run_clip, verify_clip_folder, write_clip_folder  # noqa: E402
from vkitti.dataset import VKittiDataset  # noqa: E402
from vkitti.tracks import select_tracks, summarize_tracks  # noqa: E402

EXPECTED_TRACKS = {"Scene01": [90, 86, 63, 54], "Scene06": [12, 9, 11, 13]}  # fixed in M1/M2


def load_cfg(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["m2"] = yaml.safe_load(Path(cfg["m2_config"]).read_text())
    return cfg


def dev_tracks(cfg):
    m2 = cfg["m2"]
    out = {}
    for scene in m2["scenes"]:
        ds = VKittiDataset(m2["root"], scene, m2["variation"], m2["camera_id"])
        H, W = ds.image_shape
        t = select_tracks(summarize_tracks(ds.mask_pixel_counts(), ds.info, ds.bbox, H * W,
                                           m2["track_selection"]["min_area_frac"]), H * W, **m2["track_selection"])
        if t.track_id.tolist() != EXPECTED_TRACKS[scene]:
            raise SystemExit(f"{scene}: track selection changed ({t.track_id.tolist()}), refusing to continue")
        out[scene] = (ds, t)
    return out


# ----------------------------------------------------------------------------- figures
def plot_track_timeline(patterns, title, path):
    fig, ax = plt.subplots(figsize=(14, 0.38 * len(patterns) + 1.4))
    for r, p in enumerate(patterns):
        f = p.frames
        ax.scatter(f[f.is_received].original_frame_id, [r] * f.is_received.sum(), marker="|", s=90, c="#4C72B0")
        ax.scatter(f[f.is_dropped].original_frame_id, [r] * f.is_dropped.sum(), marker="|", s=90, c="#DDDDDD")
        for b in p.meta["bursts"]:
            ax.add_patch(plt.Rectangle((b["start"] - 0.5, r - 0.35), 4, 0.7, fc="none", ec="#C44E52", lw=1))
        ax.scatter([p.meta["prompt_original_frame_id"]], [r], marker="*", s=80, c="gold", edgecolors="k", zorder=3)
    ax.set_yticks(range(len(patterns)))
    ax.set_yticklabels([f"{p.name} ({p.meta['actual_drop_rate']:.0%})" for p in patterns], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="original VKITTI frame  (blue = received, grey = dropped, red box = burst, star = prompt)", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return str(path)


def plot_rates(summary: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, g in summary.groupby("condition"):
        ax.scatter(g.requested_drop_rate + np.random.default_rng(0).uniform(-.008, .008, len(g)), g.actual_drop_rate,
                   label=cond, s=25, alpha=0.8)
    ax.plot([0.25, 0.55], [0.25, 0.55], "k:", lw=1)
    ax.set(xlabel="requested drop rate (jittered)", ylabel="achieved drop rate (dropped / post-prompt frames)",
           title="Requested vs achieved drop rate (8 tracks x 3 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return str(path)


def plot_index_map(patterns, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for p in patterns:
        rec = p.frames[p.frames.is_received]
        ax.step(rec.original_frame_id, rec.received_video_idx, where="post", label=p.name)
    f0 = patterns[0].frames
    ax.plot(f0.original_frame_id, f0.original_frame_id - f0.original_frame_id.iloc[0], "k:", lw=1, label="no drop")
    ax.set(xlabel="original VKITTI frame", ylabel="SAM 2 video index (received order)", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return str(path)


def plot_recovery_windows(p, ev, title, path, j=None):
    fig, ax = plt.subplots(figsize=(14, 3.6))
    for e in ev.itertuples():
        ax.axvspan(e.run_start - 0.5, e.run_end + 0.5, color="#C44E52", alpha=0.25, lw=0)
        if e.window_frames:
            ax.axvspan(e.window_frames[0] - 0.5, e.window_frames[-1] + 0.5, color="#55A868", alpha=0.12, lw=0)
            ax.text(e.run_start, 1.04, f"{e.window_length}" + ("*" if e.censored_by == "clip_end" else ""), fontsize=7)
    if j is not None:
        ax.plot(j.original_frame_id, j.J_drop, ".-", lw=1, label="J (dropout run)")
        ax.plot(j.original_frame_id, j.J_no_drop_matched, ".", ms=3, c="0.4", label="J (no-drop, matched)")
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.legend(fontsize=8, loc="lower left")
    ax.set(ylim=(-0.02, 1.1), xlabel="original VKITTI frame  (red = dropped run, green = observation window;"
           " number = window length in received frames, * = ends at clip end)", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------------------- commands
def cmd_generate(cfg):
    out = Path(cfg["output_dir"])
    man, figs, rep = out / "manifests", out / "figures", out / "reports"
    for d in (man, figs, rep):
        d.mkdir(parents=True, exist_ok=True)
    m2 = cfg["m2"]
    rows, figures, events = [], [], []
    for scene, (ds, tracks) in dev_tracks(cfg).items():
        for tr in tracks.itertuples():
            clip = make_clip(scene, tr)  # the M2 no-drop clip: prompt .. last observable frame
            pats = []
            for cond, spec in cfg["conditions"].items():
                for seed in cfg["seeds"]:
                    p = fd.generate(scene, m2["variation"], m2["camera_id"], int(tr.track_id), clip.original_frame_ids,
                                    cond, spec["kind"], spec["rate"], seed, cfg["burst_length"],
                                    cfg["min_gap_between_bursts"])
                    jpath, _ = fd.save(p, man)
                    p = fd.load(jpath)  # round trip + regeneration check
                    pats.append(p)
                    m = p.meta
                    rows.append({k: m[k] for k in ("scene", "track_id", "condition", "seed", "requested_drop_rate",
                                                   "actual_drop_rate", "post_prompt_frames", "total_received_frames",
                                                   "total_dropped_frames", "requested_number_of_bursts",
                                                   "number_of_bursts", "request_fits")}
                                | {"manifest": str(jpath)})
                    if m["kind"] == "burst":
                        ev = recovery_events(p.frames, within=cfg["recovery"]["within_received_frames"])
                        events.append(ev.assign(scene=scene, track_id=int(tr.track_id), condition=cond, seed=seed))
            tag = f"{scene}_track{int(tr.track_id):03d}"
            figures.append(plot_track_timeline(pats, f"{tag}: received/dropped frames per condition and seed",
                                               figs / f"{tag}_timeline.png"))
            if int(tr.track_id) in (86, 12):  # one representative track per scene
                seed0 = [p for p in pats if p.meta["seed"] == 0]
                figures.append(plot_index_map(seed0, f"{tag}: received video index vs original frame (seed 0)",
                                              figs / f"{tag}_index_map.png"))
                b = next(p for p in pats if p.meta["condition"] == "burst_50" and p.meta["seed"] == 0)
                figures.append(plot_recovery_windows(b, recovery_events(b.frames), f"{tag} burst_50 seed0: bursts "
                                                     "and recovery observation windows", figs / f"{tag}_burst_windows.png"))
    summary = pd.DataFrame(rows)
    summary.to_csv(rep / "manifest_summary.csv", index=False)
    figures.append(plot_rates(summary, figs / "requested_vs_actual_rate.png"))
    ev = pd.concat(events, ignore_index=True)
    ev.drop(columns=["window_frames"]).to_csv(rep / "burst_events_structure.csv", index=False)
    agg = summary.groupby("condition").agg(manifests=("seed", "size"), rate_min=("actual_drop_rate", "min"),
                                           rate_mean=("actual_drop_rate", "mean"), rate_max=("actual_drop_rate", "max"),
                                           bursts_min=("number_of_bursts", "min"), bursts_max=("number_of_bursts", "max"),
                                           all_requests_fit=("request_fits", "all"))
    evs = ev.groupby("condition").agg(runs=("run_index", "size"), window_mean=("window_length", "mean"),
                                      window_lt2=("window_length", lambda s: int((s < 2).sum())),
                                      determinable_within5=("within_n_determinable", "sum"),
                                      ends_at_clip_end=("censored_by", lambda s: int((s == "clip_end").sum())))
    report = {"config": {k: cfg[k] for k in ("seeds", "burst_length", "min_gap_between_bursts", "conditions", "recovery")},
              "manifests": len(summary), "by_condition": agg.reset_index().to_dict("records"),
              "burst_windows": evs.reset_index().to_dict("records"), "figures": figures}
    (rep / "generate_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(agg.round(4).to_string())
    print(evs.round(2).to_string())
    print(f"{len(summary)} manifests validated; figures: {len(figures)}")


def cmd_smoke(cfg, device_name):
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    m2, sc = cfg["m2"], cfg["m2"]["sam2"]
    out = Path(cfg["output_dir"]) / "smoke"
    for d in ("clips", "predictions", "reports", "figures"):
        (out / d).mkdir(parents=True, exist_ok=True)
    if sha256(Path(sc["checkpoint"])) != sc["checkpoint_sha256"]:
        raise SystemExit("checkpoint sha256 mismatch")
    pred = build_sam2_video_predictor(sc["config"], sc["checkpoint"], device=torch.device(device_name),
                                      apply_postprocessing=sc["apply_postprocessing"])
    m2_dir = Path(cfg["m2_output_dir"])
    tracks = dev_tracks(cfg)
    results = []
    for job in cfg["smoke"]:
        ds = tracks[job["scene"]][0]
        jpath, _ = fd.paths(Path(cfg["output_dir"]) / "manifests", job["scene"], job["track_id"],
                            f"{job['condition']}_seed{job['seed']}")
        p = fd.load(jpath)
        ref = check_m2_reference(m2_dir, p, sc, m2["evaluation"], m2["evaluation"])
        clip = clip_from_pattern(p)
        vdir = write_clip_folder(ds, clip, out / "clips")
        verify_clip_folder(ds, clip, vdir)
        dropped_paths = {ds.rgb_path(f).resolve() for f in p.meta["dropped_original_frame_ids"]}
        assert not dropped_paths & {x.resolve() for x in vdir.glob("*.jpg")}, "dropped frame present in clip"
        prompt = ds.get_frame(clip.original_frame_ids[0], clip.track_id)["mask"]  # identical to M2's prompt
        res = run_clip(pred, vdir, prompt)
        masks = res["masks"]
        assert sorted(masks) == list(range(len(clip.original_frame_ids))), "output indices != received indices"
        assert all(masks[i].shape == ds.image_shape for i in masks)
        np.savez_compressed(out / "predictions" / f"{clip.variant}_{clip.name}.npz",
                            masks=np.stack([masks[i] for i in range(len(masks))]),
                            original_frame_ids=np.array(clip.original_frame_ids))
        # Causality check: before the first dropped frame the inputs are identical to M2's, so the
        # predictions must be too (a model using future frames would differ).
        m2_masks = np.load(m2_dir / "predictions" / f"{clip.name}.npz")["masks"]
        first_drop = p.meta["dropped_original_frame_ids"][0]
        prefix = [i for i, f in enumerate(clip.original_frame_ids) if f < first_drop]
        prefix_same = [bool(np.array_equal(masks[i], m2_masks[i])) for i in prefix]
        frames = evaluate(ds, clip, masks, m2["evaluation"])
        frames.to_csv(out / "reports" / f"{clip.variant}_{clip.name}_per_frame.csv", index=False)
        pf = paired_frames(ref, frames)
        pf.to_csv(out / "reports" / f"{clip.variant}_{clip.name}_paired.csv", index=False)
        ev = recovery_events(p.frames, dict(zip(frames.original_frame_id, frames.J)), cfg["recovery"]["j_threshold"],
                             m2["evaluation"]["lost_j"], cfg["recovery"]["within_received_frames"])
        ev.to_csv(out / "reports" / f"{clip.variant}_{clip.name}_recovery_events.csv", index=False)
        lat = pd.Series(res["latency_s"]).sort_index()
        r = {"scene": clip.scene, "track_id": clip.track_id, "condition": p.meta["condition"], "seed": p.meta["seed"],
             "received_frames": len(clip.original_frame_ids), "dropped_frames": p.meta["total_dropped_frames"],
             "sam2_output_frames": len(masks), "prefix_frames_before_first_drop": len(prefix),
             "prefix_identical_to_m2": all(prefix_same), "prompt_frame_identical_to_m2": prefix_same[0],
             **paired_summary(pf), "recovery": recovery_summary(ev, cfg["recovery"]["within_received_frames"]),
             "median_latency_s": float(lat.iloc[1:].median()), "device": device_name}
        results.append(r)
        plot_recovery_windows(p, ev, f"{clip.name} {clip.variant}: J with dropped runs and observation windows",
                              out / "figures" / f"{clip.variant}_{clip.name}_recovery.png", pf)
        print(json.dumps(r, indent=1, default=str))
        if len(ev):
            print(ev[["run_start", "run_end", "first_received_after", "window_length", "first_J", "class",
                      "recovery_onset_received_k", "confirmed_within_n"]].to_string(index=False))
    (out / "reports" / "smoke_report.json").write_text(json.dumps(
        {"device": device_name, "torch": torch.__version__, "m2_reference": str(m2_dir), "jobs": results}, indent=2,
        default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["generate", "smoke"])
    ap.add_argument("--config", default="configs/m3_frame_drop.yaml")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    cmd_generate(cfg) if a.command == "generate" else cmd_smoke(cfg, a.device)


if __name__ == "__main__":
    main()
