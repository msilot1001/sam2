"""Milestone 2: official SAM 2.1 Hiera-Tiny baseline on VKITTI2 dev tracks (no frame drops).

    uv run python scripts/run_sam2_baseline.py --device mps                 # all dev tracks
    uv run python scripts/run_sam2_baseline.py --device mps --smoke 10      # first track, 10 frames
    uv run python scripts/run_sam2_baseline.py --eval-only                  # re-evaluate saved masks

Writes predictions (npz), per-frame/track/scene metrics, runtime and figures to outputs/milestone2/.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # as in the official notebooks; logged below

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from baseline.metrics import f_measure, j_measure, seg2bmap  # noqa: E402
from baseline.sam2_vos import Clip, make_clip, run_clip, write_clip_folder  # noqa: E402
from vkitti.dataset import VKittiDataset  # noqa: E402
from vkitti.tracks import select_tracks, summarize_tracks  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(ds, clip: Clip, masks: dict, cfg) -> pd.DataFrame:
    """GT is read here, after inference. video_idx 0 is the prompted frame (excluded from means)."""
    rows = []
    for idx, f in enumerate(clip.original_frame_ids):
        gt = ds.get_frame(f, clip.track_id)["mask"]
        pr = masks[idx]
        if pr.shape != gt.shape:
            raise ValueError(f"{clip.name} idx {idx}: prediction {pr.shape} vs GT {gt.shape}")
        j, fm = j_measure(pr, gt), f_measure(pr, gt, cfg["boundary_th"])
        f_strict = f_measure(pr, gt, cfg["boundary_th_strict"])  # secondary: DAVIS F saturates on small cars
        rows.append({"scene": clip.scene, "track_id": clip.track_id, "video_idx": idx, "original_frame_id": f,
                     "is_prompt": idx == 0, "gt_pixels": int(gt.sum()), "pred_pixels": int(pr.sum()),
                     "gt_empty": not gt.any(), "J": j, "F": fm, "JF": (j + fm) / 2, "F_strict": f_strict})
    return pd.DataFrame(rows)


def summarize(frames: pd.DataFrame, lost_j: float) -> pd.DataFrame:
    rows = []
    for (scene, tid), g in frames.groupby(["scene", "track_id"], sort=False):
        e = g[~g.is_prompt]
        vis = e[~e.gt_empty]
        emp = e[e.gt_empty]
        rows.append({
            "scene": scene, "track_id": tid, "first_frame": int(g.original_frame_id.min()),
            "last_frame": int(g.original_frame_id.max()), "eval_frames": len(e),
            "gt_empty_frames": len(emp),
            "J": e.J.mean(), "F": e.F.mean(), "JF": e.JF.mean(),                       # DAVIS convention
            "J_visible": vis.J.mean(), "F_visible": vis.F.mean(), "JF_visible": vis.JF.mean(),
            "F_strict": e.F_strict.mean(),
            "false_positive_on_empty_gt": int((emp.pred_pixels > 0).sum()),
            "lost_frames_J_lt": int((vis.J < lost_j).sum()),
            "first_lost_frame": int(vis[vis.J < lost_j].original_frame_id.min()) if (vis.J < lost_j).any() else None,
            "pred_empty_on_visible_gt": int((vis.pred_pixels == 0).sum()),
            "prompt_frame_J": float(g[g.is_prompt].J.iloc[0]),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def overlay_panel(rgb, gt, pr):
    out = rgb.astype(np.float32)
    out[pr] = 0.5 * out[pr] + 0.5 * np.array([255, 0, 0])
    edge = seg2bmap(gt)
    edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)  # 2-px GT contour
    out[edge] = [0, 255, 0]
    return out.astype(np.uint8)


def crop_box(gt, pr, shape, pad=40):
    ys, xs = np.nonzero(gt | pr)
    if len(ys) == 0:
        return slice(0, shape[0]), slice(0, shape[1])
    return (slice(max(ys.min() - pad, 0), min(ys.max() + pad, shape[0])),
            slice(max(xs.min() - pad, 0), min(xs.max() + pad, shape[1])))


def side_by_side(ds, clip, masks, rows: pd.DataFrame, path: Path, title: str):
    fig, axes = plt.subplots(len(rows), 5, figsize=(26, 2.6 * len(rows)), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 3, 3, 3, 1.4]})
    for r, row in enumerate(rows.itertuples()):
        s = ds.get_frame(row.original_frame_id, clip.track_id)
        rgb, gt, pr = s["rgb"], s["mask"], masks[row.video_idx]
        ov = overlay_panel(rgb, gt, pr)
        ys, xs = crop_box(gt, pr, gt.shape)
        panels = [(rgb, "RGB"), (gt, "GT mask"), (pr, "SAM2 prediction"),
                  (ov, "overlay: pred red, GT green contour"), (ov[ys, xs], "zoom")]
        for c, (img, name) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.axis("off")
            ax.set_title(name if c else f"orig frame {row.original_frame_id} (video idx {row.video_idx})  "
                         f"J={row.J:.3f} F={row.F:.3f}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return str(path)


def plot_timelines(frames, path):
    tracks = list(frames.groupby(["scene", "track_id"], sort=False))
    fig, axes = plt.subplots(len(tracks), 1, figsize=(12, 2.2 * len(tracks)), squeeze=False)
    for ax, ((scene, tid), g) in zip(axes[:, 0], tracks):
        ax.plot(g.original_frame_id, g.J, label="J", lw=1)
        ax.plot(g.original_frame_id, g.F, label="F", lw=1)
        ax.plot(g.original_frame_id, g.JF, label="J&F", lw=1.5, color="k")
        for f in g[g.gt_empty].original_frame_id:
            ax.axvspan(f - 0.5, f + 0.5, color="0.85", lw=0)
        ax.axvline(g.original_frame_id.iloc[0], color="g", ls="--", lw=0.8)
        ax.set(ylim=(-0.02, 1.02), ylabel=f"{scene}\ntrack {tid}")
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("original VKITTI frame (green dashed = prompt, grey = empty GT)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return str(path)


def plot_track_bars(tracks, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(tracks))
    for k, (col, lab) in enumerate([("J", "J"), ("F", "F"), ("JF", "J&F")]):
        ax.bar(x + (k - 1) * 0.27, tracks[col], 0.27, label=lab)
    ax.set(xticks=x, ylim=(0, 1), ylabel="mean over non-prompt frames",
           title="SAM 2.1 Hiera-Tiny baseline per track (no frame drops)")
    ax.set_xticklabels([f"{r.scene}\ntrack {r.track_id}" for r in tracks.itertuples()], fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_baseline.yaml")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--smoke", type=int, help="first track of the first scene only, this many frames")
    ap.add_argument("--eval-only", action="store_true", help="reuse saved predictions")
    ap.add_argument("--out", help="override output_dir")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    out = Path(a.out or cfg["output_dir"]) / ("smoke" if a.smoke else "")
    for sub in ("predictions", "clips", "reports", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    sc, ev = cfg["sam2"], cfg["evaluation"]

    # Data: tracks + clips are fixed before the model is touched.
    clips, datasets = [], {}
    for scene in cfg["scenes"]:
        ds = VKittiDataset(cfg["root"], scene, cfg["variation"], cfg["camera_id"])
        H, W = ds.image_shape
        tracks = select_tracks(summarize_tracks(ds.mask_pixel_counts(), ds.info, ds.bbox, H * W,
                                                cfg["track_selection"]["min_area_frac"]), H * W, **cfg["track_selection"])
        datasets[scene] = ds
        clips += [make_clip(scene, row, a.smoke) for row in tracks.itertuples()]
    if a.smoke:
        clips = clips[:1]

    env, runtime = {}, []
    if a.eval_only:  # keep the inference-run records in the regenerated report
        env = json.loads((out / "reports" / "environment.json").read_text())
        runtime = pd.read_csv(out / "reports" / "runtime.csv").to_dict("records")
    else:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        ckpt = Path(sc["checkpoint"])
        digest = sha256(ckpt)
        if digest != sc["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint sha256 {digest} != expected {sc['checkpoint_sha256']}")
        device = torch.device(a.device)
        t0 = time.perf_counter()
        predictor = build_sam2_video_predictor(sc["config"], str(ckpt), device=device,
                                               apply_postprocessing=sc["apply_postprocessing"])
        build_s = time.perf_counter() - t0
        try:
            import sam2._C  # noqa: F401
            has_c = True
        except ImportError:
            has_c = False
        import sam2
        env = {
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "torch": torch.__version__, "device": str(device), "precision": "float32 (no autocast; official MPS/CPU path)",
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "sam2_upstream_commit": sc["upstream_commit"], "sam2_package": sam2.__file__,
            "config": sc["config"], "checkpoint": str(ckpt), "checkpoint_sha256": digest,
            "checkpoint_bytes": ckpt.stat().st_size, "apply_postprocessing": sc["apply_postprocessing"],
            "sam2_C_extension": has_c,
            "model": {"num_maskmem": predictor.num_maskmem, "image_size": predictor.image_size,
                      "memory_temporal_stride_for_eval": predictor.memory_temporal_stride_for_eval,
                      "max_cond_frames_in_attn": predictor.max_cond_frames_in_attn,
                      "use_obj_ptrs_in_encoder": predictor.use_obj_ptrs_in_encoder,
                      "max_obj_ptrs_in_encoder": predictor.max_obj_ptrs_in_encoder,
                      "fill_hole_area": predictor.fill_hole_area,
                      "non_overlap_masks": predictor.non_overlap_masks},
            "build_s": build_s,
        }
        for clip in clips:
            ds = datasets[clip.scene]
            vdir = write_clip_folder(ds, clip, out / "clips")
            prompt = ds.get_frame(clip.original_frame_ids[0], clip.track_id)["mask"]  # the ONLY GT given
            res = run_clip(predictor, vdir, prompt)
            np.savez_compressed(out / "predictions" / f"{clip.name}.npz",
                                masks=np.stack([res["masks"][i] for i in range(len(clip.original_frame_ids))]),
                                original_frame_ids=np.array(clip.original_frame_ids))
            lat = pd.Series(res["latency_s"]).sort_index()
            runtime.append({"scene": clip.scene, "track_id": clip.track_id, "frames": len(lat),
                            "init_s": res["init_s"], "prompt_frame_s": lat.iloc[0],
                            "latency_mean_s": lat.iloc[1:].mean(), "latency_median_s": lat.iloc[1:].median(),
                            "latency_p90_s": lat.iloc[1:].quantile(0.9),
                            "latency_steady_mean_s": lat.iloc[7:].mean(),  # memory bank full (num_maskmem=7)
                            "mps_driver_peak_bytes": max(res["mps_driver_bytes"], default=None)})
            pd.DataFrame({"video_idx": lat.index, "latency_s": lat.values}).to_csv(
                out / "reports" / f"{clip.name}_latency.csv", index=False)
            print(f"[{clip.name}] {len(lat)} frames, init {res['init_s']:.2f}s, "
                  f"median {lat.iloc[1:].median() * 1000:.0f} ms/frame", flush=True)
        env["process_max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
        (out / "reports" / "environment.json").write_text(json.dumps(env, indent=2, default=str))
        pd.DataFrame(runtime).to_csv(out / "reports" / "runtime.csv", index=False)

    # Evaluation (GT loaded only here).
    frames = []
    masks_by_clip = {}
    for clip in clips:
        z = np.load(out / "predictions" / f"{clip.name}.npz")
        if tuple(z["original_frame_ids"]) != clip.original_frame_ids:
            raise SystemExit(f"{clip.name}: saved frame mapping differs from the clip definition")
        masks_by_clip[clip.name] = {i: m for i, m in enumerate(z["masks"])}
        frames.append(evaluate(datasets[clip.scene], clip, masks_by_clip[clip.name], ev))
    frames = pd.concat(frames, ignore_index=True)
    tracks = summarize(frames, ev["lost_j"])
    scenes = tracks.groupby("scene")[["J", "F", "JF", "J_visible", "F_visible", "JF_visible", "F_strict"]].mean()
    scenes["tracks"] = tracks.groupby("scene").size()
    scenes["eval_frames"] = tracks.groupby("scene").eval_frames.sum()
    overall = tracks[["J", "F", "JF", "J_visible", "F_visible", "JF_visible", "F_strict"]].mean()
    frames.to_csv(out / "reports" / "per_frame_metrics.csv", index=False)
    tracks.to_csv(out / "reports" / "per_track_metrics.csv", index=False)
    scenes.to_csv(out / "reports" / "per_scene_metrics.csv")
    pd.concat([c.mapping().assign(scene=c.scene, track_id=c.track_id) for c in clips]).to_csv(
        out / "reports" / "video_index_mapping.csv", index=False)

    # Figures.
    figs = [plot_timelines(frames, out / "figures" / "jf_over_time.png"),
            plot_track_bars(tracks, out / "figures" / "per_track_jf.png")]
    for clip in clips:
        g = frames[(frames.scene == clip.scene) & (frames.track_id == clip.track_id)]
        e = g[~g.is_prompt & ~g.gt_empty]
        picks = sorted({0, int(e.video_idx.iloc[0]), int(e.video_idx.iloc[len(e) // 2]), int(e.video_idx.iloc[-1]),
                        int(e.loc[e.JF.idxmin()].video_idx)})
        figs.append(side_by_side(datasets[clip.scene], clip, masks_by_clip[clip.name], g.set_index("video_idx").loc[picks]
                                 .reset_index(), out / "figures" / f"{clip.name}_side_by_side.png",
                                 f"{clip.name}: prompt, first/mid/last evaluated frame and worst J&F frame"))
    vis = frames[~frames.is_prompt & ~frames.gt_empty]
    for kind, sel in (("success", vis.nlargest(3, "JF")), ("failure", vis.nsmallest(3, "JF"))):
        rows, masks_cat = [], {}
        fig_rows = []
        for r in sel.itertuples():
            c = next(c for c in clips if c.scene == r.scene and c.track_id == r.track_id)
            fig_rows.append(side_by_side(datasets[c.scene], c, masks_by_clip[c.name],
                                         frames[(frames.scene == c.scene) & (frames.track_id == c.track_id)
                                                & (frames.video_idx == r.video_idx)],
                                         out / "figures" / f"{kind}_{c.name}_frame{r.original_frame_id:05d}.png",
                                         f"{kind} example: {c.name}"))
        figs += fig_rows

    report = {"environment": env, "clips": [{"name": c.name, "scene": c.scene, "track_id": c.track_id,
                                             "first": c.original_frame_ids[0], "last": c.original_frame_ids[-1],
                                             "frames": len(c.original_frame_ids)} for c in clips],
              "overall_mean_of_tracks": overall.to_dict(), "scenes": scenes.reset_index().to_dict("records"),
              "tracks": tracks.to_dict("records"), "runtime": runtime, "figures": figs}
    (out / "reports" / "m2_report.json").write_text(json.dumps(report, indent=2, default=lambda o: o.item()
                                                               if isinstance(o, np.generic) else str(o)))
    pd.set_option("display.width", 200)
    print(tracks[["scene", "track_id", "first_frame", "last_frame", "eval_frames", "gt_empty_frames", "J", "F",
                  "JF", "JF_visible", "F_strict", "lost_frames_J_lt", "false_positive_on_empty_gt"]].round(4).to_string(index=False))
    print(scenes.round(4).to_string())
    print("overall (mean of tracks):", overall.round(4).to_dict())


if __name__ == "__main__":
    main()
