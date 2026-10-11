"""Per-track statistics and deterministic target-vehicle selection.

Criteria (mentor_plan_revision.md §8), fixed before any SAM2 result exists:
  prompt frame  = first frame in which the track has >= 1 instance-mask pixel
  area          : mask pixels at the prompt frame >= min_area_frac * H * W  (0.1%)
  observability : number of frames with >= 1 mask pixel >= min_frames      (40)
Ranking among eligible tracks: more visible frames, then larger prompt area, then lower trackID.
"""
import pandas as pd


def summarize_tracks(counts: pd.DataFrame, info: pd.DataFrame, bbox: pd.DataFrame) -> pd.DataFrame:
    """One row per track that is visible in at least one mask.

    counts: frame, trackID, mask_pixels (from VKittiDataset.mask_pixel_counts)
    bbox:   bbox.txt rows for the same camera (used only for isMoving metadata)
    """
    visible = counts[counts.mask_pixels > 0].sort_values(["trackID", "frame"])
    g = visible.groupby("trackID")
    first = g.first()
    summary = pd.DataFrame({
        "prompt_frame": first.frame,
        "prompt_area": first.mask_pixels,
        "last_frame": g.frame.max(),
        "visible_frames": g.size(),
    })
    summary["span_frames"] = summary.last_frame - summary.prompt_frame + 1
    moving = bbox.groupby("trackID").isMoving.agg(lambda s: s.astype(str).eq("True").mean())
    summary["moving_frac"] = moving.reindex(summary.index)
    summary = summary.join(info.set_index("trackID")[["label", "model", "color"]])
    return summary.reset_index().rename(columns={"trackID": "track_id"})


def select_tracks(summary: pd.DataFrame, image_area: int, min_area_frac: float = 0.001,
                  min_frames: int = 40, max_tracks: int = 4) -> pd.DataFrame:
    min_area = min_area_frac * image_area
    ok = summary[(summary.prompt_area >= min_area) & (summary.visible_frames >= min_frames)]
    ranked = ok.sort_values(["visible_frames", "prompt_area", "track_id"], ascending=[False, False, True])
    return ranked.head(max_tracks).reset_index(drop=True)
