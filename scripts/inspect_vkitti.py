"""Milestone 0 smoke test on real VKITTI2 data.

    uv run python scripts/inspect_vkitti.py [--config configs/vkitti_m0.yaml]

Checks paths, RGB/mask/extrinsic frame correspondence, bbox pixel counts, extrinsic validity,
stereo baseline; selects target tracks; writes overlays, trajectories and a JSON report to
outputs/milestone0/. Exits 1 if any check fails.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from vkitti.calibration import load_extrinsics
from vkitti.dataset import VKittiDataset
from vkitti.tracks import RULES, select_tracks, summarize_tracks
from vkitti.validation import (camera_trajectory, check_extrinsics, check_pixel_counts,
                               check_stereo_baseline)
from vkitti.visualization import save_overlay, save_trajectory_plots


def overlay_frames(row) -> list[int]:
    """Prompt, middle and last visible frame of a track."""
    first, last = int(row.prompt_frame), int(row.last_frame)
    return sorted({first, (first + last) // 2, last})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vkitti_m0.yaml")
    ap.add_argument("--scene", help="override config scene (dev scenes only: Scene01, Scene06)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["scene"] = args.scene or cfg["scene"]
    out = Path(cfg["output_dir"])
    for sub in ("overlays", "trajectories", "reports"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    ds = VKittiDataset(cfg["root"], cfg["scene"], cfg["variation"], cfg["camera_id"])
    prefix = ds.tag.replace("/", "_")
    H, W = ds.image_shape
    checks = {}
    print(f"[{ds.tag}] {len(ds)} frames ({ds.frame_ids[0]}..{ds.frame_ids[-1]}), image {W}x{H}, "
          f"{len(ds.track_ids)} tracks in info.txt")

    # Full pass over RGB: every frame decodes, same shape (load_rgb raises otherwise).
    for f in ds.frame_ids:
        ds.load_rgb(f)
    checks["rgb_mask_extrinsic_frames_aligned"] = True  # enforced in VKittiDataset.__init__ and above

    counts = ds.mask_pixel_counts()
    unknown = sorted(set(counts.trackID) - ds.track_ids)
    checks["mask_ids_in_info_txt"] = not unknown
    ok, bad = check_pixel_counts(counts, ds.bbox)
    checks["bbox_number_pixels_match_masks"] = ok
    ok, ext = check_extrinsics(ds, **cfg["extrinsic_checks"])
    checks["extrinsics_valid_and_continuous"] = ok
    ok, stereo = check_stereo_baseline(load_extrinsics(ds.dir / "extrinsic.txt"))
    checks["stereo_baseline_confirms_world_to_camera"] = ok

    sel_cfg = cfg["track_selection"]
    summary = summarize_tracks(counts, ds.info, ds.bbox, H * W, sel_cfg["min_area_frac"])
    selected = select_tracks(summary, H * W, **sel_cfg)
    # Eligible counts under both pre-registered rules (no cap), for the M0 -> M0.5 comparison.
    eligible = {r: select_tracks(summary, H * W, sel_cfg["min_area_frac"], sel_cfg["min_frames"], 10**9, r)
                for r in RULES}
    checks["at_least_one_track_selected"] = not selected.empty
    summary.to_csv(out / "reports" / f"{prefix}_tracks.csv", index=False)
    selected.to_csv(out / "reports" / f"{prefix}_selected_tracks.csv", index=False)

    overlays = []
    for row in selected.itertuples():
        for f in overlay_frames(row):
            s = ds.get_frame(f, int(row.track_id))
            assert s["mask"].sum() == counts.query("frame == @f and trackID == @row.track_id").mask_pixels.sum()
            p = out / "overlays" / f"{prefix}_track{row.track_id:03d}_frame{f:05d}.png"
            save_overlay(s, p)
            overlays.append(str(p))

    traj = camera_trajectory(ds)
    traj.to_csv(out / "reports" / f"{prefix}_camera_trajectory.csv", index=False)
    trajectories = [str(p) for p in save_trajectory_plots(traj, ds.tag, out / "trajectories")]

    report = {
        "dataset": {"scene": ds.scene, "variation": ds.variation, "camera_id": ds.camera_id,
                    "frames": len(ds), "first_frame": ds.frame_ids[0], "last_frame": ds.frame_ids[-1],
                    "image_hw": [H, W], "tracks_in_info": len(ds.track_ids),
                    "tracks_visible_in_masks": int(summary.shape[0])},
        "checks": checks,
        "pixel_count_mismatches": len(bad),
        "unknown_mask_track_ids": unknown,
        "extrinsics": ext,
        "stereo_baseline": stereo,
        "track_selection": {**sel_cfg, "min_area_px": sel_cfg["min_area_frac"] * H * W,
                            "eligible_before_cap": {r: len(e) for r, e in eligible.items()},
                            "eligible_motion": {r: e.motion.value_counts().to_dict() for r, e in eligible.items()},
                            "selected": selected.to_dict("records")},
        "outputs": {"overlays": overlays, "trajectories": trajectories},
    }
    rp = out / "reports" / f"{prefix}_m0_report.json"
    rp.write_text(json.dumps(report, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  camera path {ext['path_length_m']:.1f} m, step mean {ext['step_m']['mean']:.3f} m, "
          f"max {ext['step_m']['max']:.3f} m; stereo max err {stereo['max_abs_err_m']:.2e} m")
    print(f"  eligible tracks by rule: { {r: len(e) for r, e in eligible.items()} }; "
          f"selected ({sel_cfg['rule']}): {selected.track_id.tolist()}")
    print(selected[["track_id", "label", "first_visible_frame", "prompt_frame", "prompt_area",
                    "observable_frames", "last_frame", "motion"]].to_string(index=False))
    print(f"  report: {rp}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
