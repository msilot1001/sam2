import pandas as pd
import pytest

from vkitti.dataset import VKittiDataset
from vkitti.tracks import select_tracks, summarize_tracks

AREA = 48  # 6x8 fixture image


def summary_of(vkitti_root, frac=0.001):
    ds = VKittiDataset(vkitti_root)
    return summarize_tracks(ds.mask_pixel_counts(), ds.info, ds.bbox, AREA, frac)


def test_summary_from_real_mask_counts(vkitti_root):
    s = summary_of(vkitti_root).set_index("track_id")
    assert list(s.index) == [0, 4]  # track 7 is in info.txt but never visible
    assert s.loc[0, ["first_visible_frame", "first_visible_area", "last_frame", "visible_frames"]].tolist() == [0, 6, 2, 3]
    assert s.loc[4, ["first_visible_frame", "first_visible_area", "last_frame", "visible_frames"]].tolist() == [0, 12, 1, 2]
    assert s.loc[4, "moving_frac"] == 1.0 and s.loc[0, "moving_frac"] == 0.0
    assert s.loc[4, "motion"] == "moving" and s.loc[0, "motion"] == "static"
    assert s.loc[4, "label"] == "Van"


def test_first_visible_rule_thresholds(vkitti_root):
    s = summary_of(vkitti_root)
    sel = lambda frac, n: select_tracks(s, AREA, frac, n, rule="first_visible").track_id.tolist()  # noqa: E731
    assert sel(0.1, 3) == [0]
    assert sel(0.2, 2) == [4]
    assert sel(0.3, 1) == []


def counts_frame(rows):
    return pd.DataFrame(rows, columns=["frame", "trackID", "mask_pixels"])


def synthetic_summary():
    # Track 1: enters tiny (5 px) at frame 0, first >= 10 px at frame 3, visible through frame 9.
    # Track 2: >= 10 px from its first frame 2, visible through frame 5.
    rows = [(f, 1, 5 if f < 3 else 20) for f in range(10)] + [(f, 2, 15) for f in range(2, 6)]
    info = pd.DataFrame({"trackID": [1, 2], "label": ["Car", "Van"], "model": ["a", "b"], "color": ["c", "d"]})
    bbox = pd.DataFrame({"trackID": [1, 2], "isMoving": [False, True]})
    return summarize_tracks(counts_frame(rows), info, bbox, image_area=1000, min_area_frac=0.01)


def test_first_area_rule_prompt_is_first_frame_over_threshold():
    s = synthetic_summary().set_index("track_id")
    assert s.loc[1, ["first_visible_frame", "area_prompt_frame", "visible_after_area_prompt"]].tolist() == [0, 3, 6]
    assert s.loc[2, ["first_visible_frame", "area_prompt_frame", "visible_after_area_prompt"]].tolist() == [2, 2, 3]
    sel = select_tracks(synthetic_summary(), 1000, 0.01, min_frames=6, rule="first_area")
    assert sel.track_id.tolist() == [1] and sel.prompt_frame.tolist() == [3] and sel.observable_frames.tolist() == [6]
    # The M0 rule rejects track 1 (5 px at first appearance) but keeps track 2.
    assert select_tracks(synthetic_summary(), 1000, 0.01, 4, rule="first_visible").track_id.tolist() == [2]
    # Frames counted strictly after the prompt: 6 is enough, 7 is not.
    assert select_tracks(synthetic_summary(), 1000, 0.01, 7).empty


def test_never_large_enough_track_is_ineligible():
    rows = [(f, 3, 2) for f in range(100)]
    info = pd.DataFrame({"trackID": [3], "label": ["Car"], "model": ["a"], "color": ["c"]})
    s = summarize_tracks(counts_frame(rows), info, pd.DataFrame({"trackID": [3], "isMoving": [False]}), 1000, 0.01)
    assert s.area_prompt_frame.isna().all() and select_tracks(s, 1000, 0.01, 1).empty


def test_unknown_rule():
    with pytest.raises(ValueError):
        select_tracks(synthetic_summary(), 1000, rule="bogus")


def test_ranking_and_tiebreak_deterministic():
    s = pd.DataFrame({"track_id": [9, 3, 5, 1], "first_visible_area": [1000, 1000, 2000, 1000],
                      "first_visible_frame": 0, "visible_frames": [50, 50, 50, 80],
                      "area_prompt_frame": 0, "area_prompt_area": [1000, 1000, 2000, 1000],
                      "visible_after_area_prompt": [49, 49, 49, 79]})
    for rule in ("first_area", "first_visible"):
        r1 = select_tracks(s, 10**5, max_tracks=3, rule=rule).track_id.tolist()
        r2 = select_tracks(s.sample(frac=1, random_state=0), 10**5, max_tracks=3, rule=rule).track_id.tolist()
        assert r1 == r2 == [1, 5, 3]  # observable frames desc, area desc, track id asc
