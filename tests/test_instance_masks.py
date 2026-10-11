import numpy as np
import pytest
from conftest import instance_frame
from PIL import Image

from vkitti.dataset import VKittiDataset, load_instance, track_mask
from vkitti.validation import check_pixel_counts


def test_indexed_png_values_preserved(vkitti_root):
    ds = VKittiDataset(vkitti_root)
    inst = ds.load_instance(0)
    assert inst.dtype == np.uint8
    np.testing.assert_array_equal(inst, instance_frame(0))
    # The naive path loses ids: palette colours are not the indices.
    with Image.open(ds.mask_path(0)) as im:
        assert not np.array_equal(np.asarray(im.convert("RGB"))[..., 0], inst)


def test_track_id_plus_one_rule(vkitti_root):
    ds = VKittiDataset(vkitti_root)
    m = ds.get_frame(0, 4)["mask"]
    assert m.dtype == bool and m.sum() == 12 and m[3:6, 4:8].all()
    assert ds.get_frame(0, 0)["mask"].sum() == 6


def test_absent_track_gives_empty_mask(vkitti_root):
    s = VKittiDataset(vkitti_root).get_frame(2, 4)
    assert s["mask"].sum() == 0 and s["track_visible"] is False
    # In info.txt but never rendered.
    assert VKittiDataset(vkitti_root).get_frame(0, 7)["mask"].sum() == 0


def test_track_mask_rejects_unencodable_ids():
    with pytest.raises(ValueError):
        track_mask(np.zeros((2, 2), np.uint8), 255)
    with pytest.raises(ValueError):
        track_mask(np.zeros((2, 2), np.uint8), -1)


def test_rgb_mode_png_rejected(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(p)
    with pytest.raises(ValueError, match="indexed"):
        load_instance(p)


def test_bbox_pixel_counts_match(vkitti_root):
    ds = VKittiDataset(vkitti_root)
    counts = ds.mask_pixel_counts()
    ok, bad = check_pixel_counts(counts, ds.bbox)
    assert ok and bad.empty
    ds.bbox.loc[0, "number_pixels"] += 1
    ok, bad = check_pixel_counts(counts, ds.bbox)
    assert not ok and len(bad) == 1
