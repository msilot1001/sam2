from pathlib import Path

import numpy as np
import pytest
from conftest import FRAMES, H, W, cam_to_world

from vkitti.dataset import VKittiDataset

REAL_ROOT = Path(__file__).resolve().parents[1] / "data" / "vkitti2"


def test_rgb_shape_dtype_and_frame_ids(vkitti_root):
    ds = VKittiDataset(vkitti_root)
    assert ds.frame_ids == FRAMES and len(ds) == 3 and ds.image_shape == (H, W)
    rgb = ds.load_rgb(1)
    assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8


def test_get_frame_keys_and_correspondence(vkitti_root):
    ds = VKittiDataset(vkitti_root)
    for f in FRAMES:
        s = ds.get_frame(frame_id=f, track_id=0)
        assert s["original_frame_id"] == f and s["track_id"] == 0
        assert abs(int(s["rgb"][..., 0].mean()) - (10 + 100 * f)) <= 3  # RGB really from frame f
        R, p = cam_to_world(f, 0)
        np.testing.assert_allclose(s["camera_position"], p, atol=1e-6)
        np.testing.assert_allclose(s["camera_rotation"], R, atol=1e-6)
        assert s["extrinsic"].shape == (4, 4)
        assert s["mask"].shape == s["rgb"].shape[:2]
        assert not any("pose" in k and "camera" not in k for k in s)  # no target GT pose leaks in


def test_camera_ids_not_mixed(vkitti_root):
    p0 = VKittiDataset(vkitti_root, camera_id=0).camera(1)["camera_position"]
    p1 = VKittiDataset(vkitti_root, camera_id=1).camera(1)["camera_position"]
    np.testing.assert_allclose(p0, cam_to_world(1, 0)[1], atol=1e-6)
    np.testing.assert_allclose(p1, cam_to_world(1, 1)[1], atol=1e-6)


def test_invalid_frame_and_track(vkitti_root):
    ds = VKittiDataset(vkitti_root)
    with pytest.raises(KeyError):
        ds.get_frame(3, 0)
    with pytest.raises(KeyError):
        ds.get_frame(-1, 0)
    with pytest.raises(TypeError):
        ds.get_frame(1.0, 0)
    with pytest.raises(KeyError, match="track_id 99"):
        ds.get_frame(0, 99)


def test_missing_files(vkitti_root):
    with pytest.raises(FileNotFoundError):
        VKittiDataset(vkitti_root, scene="Scene02")
    d = vkitti_root / "Scene01" / "clone"
    (d / "frames" / "instanceSegmentation" / "Camera_0" / "instancegt_00001.png").unlink()
    with pytest.raises(ValueError, match="RGB/mask frame ids differ"):
        VKittiDataset(vkitti_root)


def test_missing_rgb_after_init(vkitti_root):
    ds = VKittiDataset(vkitti_root)
    ds.rgb_path(2).unlink()
    with pytest.raises(FileNotFoundError):
        ds.load_rgb(2)


def test_wrong_resolution_rejected(vkitti_root):
    from conftest import write_indexed_png
    ds = VKittiDataset(vkitti_root)
    write_indexed_png(ds.mask_path(2), np.zeros((H + 1, W), np.uint8))
    with pytest.raises(ValueError, match="mask shape"):
        ds.load_instance(2)


def test_missing_extrinsic(vkitti_root):
    p = vkitti_root / "Scene01" / "clone" / "extrinsic.txt"
    p.write_text("\n".join(l for l in p.read_text().splitlines() if not l.startswith("2 0 ")) + "\n")
    with pytest.raises(ValueError, match="no extrinsic"):
        VKittiDataset(vkitti_root)


@pytest.mark.realdata
@pytest.mark.skipif(not (REAL_ROOT / "Scene01/clone/frames/rgb/Camera_0").is_dir(), reason="real VKITTI2 not present")
def test_real_scene01_clone_cam0():
    ds = VKittiDataset(REAL_ROOT)
    assert len(ds) > 0 and ds.image_shape == (375, 1242)
    s = ds.get_frame(frame_id=50, track_id=4)
    assert s["rgb"].shape == (375, 1242, 3) and s["mask"].dtype == bool
    row = ds.bbox[(ds.bbox.frame == 50) & (ds.bbox.trackID == 4)]
    assert int(s["mask"].sum()) == (int(row.number_pixels.iloc[0]) if len(row) else 0)
