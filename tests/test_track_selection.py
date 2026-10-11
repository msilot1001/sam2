import pandas as pd

from vkitti.dataset import VKittiDataset
from vkitti.tracks import select_tracks, summarize_tracks


def summary_of(vkitti_root):
    ds = VKittiDataset(vkitti_root)
    return summarize_tracks(ds.mask_pixel_counts(), ds.info, ds.bbox)


def test_summary_from_real_mask_counts(vkitti_root):
    s = summary_of(vkitti_root).set_index("track_id")
    assert list(s.index) == [0, 4]  # track 7 is in info.txt but never visible
    assert s.loc[0, ["prompt_frame", "prompt_area", "last_frame", "visible_frames"]].tolist() == [0, 6, 2, 3]
    assert s.loc[4, ["prompt_frame", "prompt_area", "last_frame", "visible_frames"]].tolist() == [0, 12, 1, 2]
    assert s.loc[4, "moving_frac"] == 1.0 and s.loc[0, "moving_frac"] == 0.0
    assert s.loc[4, "label"] == "Van"


def test_thresholds(vkitti_root):
    s = summary_of(vkitti_root)
    area = 48  # 6x8 image
    assert select_tracks(s, area, min_area_frac=0.1, min_frames=3).track_id.tolist() == [0]
    assert select_tracks(s, area, min_area_frac=0.2, min_frames=2).track_id.tolist() == [4]
    assert select_tracks(s, area, min_area_frac=0.3, min_frames=1).empty


def test_ranking_and_tiebreak_deterministic():
    s = pd.DataFrame({"track_id": [9, 3, 5, 1], "prompt_area": [1000, 1000, 2000, 1000],
                      "visible_frames": [50, 50, 50, 80]})
    r1 = select_tracks(s, 10**5, max_tracks=3).track_id.tolist()
    r2 = select_tracks(s.sample(frac=1, random_state=0), 10**5, max_tracks=3).track_id.tolist()
    assert r1 == r2 == [1, 5, 3]  # frames desc, area desc, track id asc
