"""Per-track statistics and deterministic target-vehicle selection.

Two pre-registered rules (fixed on camera/mask statistics only, never on SAM2 results):

  first_area (default, M0.5)
    prompt frame  = first frame whose mask area >= min_area_frac * H * W  (0.1%)
    observability = frames with >= 1 mask pixel strictly AFTER the prompt frame >= min_frames (40)
  first_visible (M0, kept for reproducible comparison)
    prompt frame  = first frame with >= 1 mask pixel; its area must be >= min_area_frac * H * W
    observability = frames with >= 1 mask pixel, prompt frame included, >= min_frames

Ranking among eligible tracks: more observable frames, larger prompt area, lower trackID.
"""
import pandas as pd

RULES = ("first_area", "first_visible")


def summarize_tracks(counts: pd.DataFrame, info: pd.DataFrame, bbox: pd.DataFrame,
                     image_area: int, min_area_frac: float = 0.001) -> pd.DataFrame:
    """One row per track visible in at least one mask, with the columns both rules need.

    counts: frame, trackID, mask_pixels (from VKittiDataset.mask_pixel_counts)
    bbox:   bbox.txt rows for the same camera (isMoving metadata only)
    """
    min_area = min_area_frac * image_area
    visible = counts[counts.mask_pixels > 0].sort_values(["trackID", "frame"])
    rows = []
    for tid, g in visible.groupby("trackID"):
        big = g[g.mask_pixels >= min_area]
        prompt = big.iloc[0] if len(big) else None
        rows.append({
            "track_id": tid,
            "first_visible_frame": g.frame.iloc[0],
            "first_visible_area": g.mask_pixels.iloc[0],
            "visible_frames": len(g),
            "last_frame": g.frame.iloc[-1],
            "area_prompt_frame": None if prompt is None else prompt.frame,
            "area_prompt_area": None if prompt is None else prompt.mask_pixels,
            "visible_after_area_prompt": 0 if prompt is None else int((g.frame > prompt.frame).sum()),
        })
    summary = pd.DataFrame(rows).set_index("track_id")
    summary["area_prompt_frame"] = summary.area_prompt_frame.astype("Int64")
    summary["area_prompt_area"] = summary.area_prompt_area.astype("Int64")
    moving = bbox.groupby("trackID").isMoving.agg(lambda s: s.astype(str).eq("True").mean())
    summary["moving_frac"] = moving.reindex(summary.index)
    summary["motion"] = summary.moving_frac.map(lambda m: "static" if m == 0 else "moving" if m == 1 else "mixed")
    summary = summary.join(info.set_index("trackID")[["label", "model", "color"]])
    return summary.reset_index()


def select_tracks(summary: pd.DataFrame, image_area: int, min_area_frac: float = 0.001,
                  min_frames: int = 40, max_tracks: int = 4, rule: str = "first_area") -> pd.DataFrame:
    """Eligible tracks ranked deterministically; adds prompt_frame / prompt_area / observable_frames."""
    if rule == "first_area":
        s = summary.assign(prompt_frame=summary.area_prompt_frame, prompt_area=summary.area_prompt_area,
                           observable_frames=summary.visible_after_area_prompt)
        ok = s[s.prompt_frame.notna() & (s.observable_frames >= min_frames)]
        ok = ok[ok.prompt_area >= min_area_frac * image_area]  # guards a summary built with another threshold
    elif rule == "first_visible":
        s = summary.assign(prompt_frame=summary.first_visible_frame, prompt_area=summary.first_visible_area,
                           observable_frames=summary.visible_frames)
        ok = s[(s.prompt_area >= min_area_frac * image_area) & (s.observable_frames >= min_frames)]
    else:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    ranked = ok.sort_values(["observable_frames", "prompt_area", "track_id"], ascending=[False, False, True])
    out = ranked.head(max_tracks).reset_index(drop=True)
    out["prompt_frame"] = out.prompt_frame.astype(int)
    out["prompt_area"] = out.prompt_area.astype(int)
    return out
