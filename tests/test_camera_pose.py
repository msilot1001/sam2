import numpy as np
import pytest
from conftest import rot_y

from vkitti.calibration import camera_pose, invert_rigid, load_extrinsics, rigid_problems, rotation_angle_deg
from vkitti.dataset import VKittiDataset
from vkitti.validation import camera_trajectory, check_extrinsics, check_stereo_baseline


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def test_extrinsic_parsing_keyed_by_frame_and_camera(vkitti_root):
    ext = load_extrinsics(vkitti_root / "Scene01/clone/extrinsic.txt")
    assert set(ext) == {(f, c) for f in range(3) for c in (0, 1)}
    assert all(T.shape == (4, 4) and not rigid_problems(T) for T in ext.values())
    assert not np.allclose(ext[(1, 0)], ext[(1, 1)])


def test_invert_rigid_matches_numpy_and_roundtrips():
    T = make_T(rot_y(33) @ rot_y(-5).T, [1.0, -2.0, 3.5])
    np.testing.assert_allclose(invert_rigid(T), np.linalg.inv(T), atol=1e-12)
    np.testing.assert_allclose(invert_rigid(T) @ T, np.eye(4), atol=1e-12)


def test_world_to_camera_inversion_gives_camera_center():
    R_c2w, p = rot_y(40), np.array([5.0, -1.6, 9.0])
    T_w2c = make_T(R_c2w.T, -R_c2w.T @ p)
    pos, rot = camera_pose(T_w2c)
    np.testing.assert_allclose(pos, p, atol=1e-12)
    np.testing.assert_allclose(rot, R_c2w, atol=1e-12)
    np.testing.assert_allclose((T_w2c @ np.append(p, 1))[:3], 0, atol=1e-12)  # centre maps to origin


@pytest.mark.parametrize("bad, msg", [
    (make_T(np.diag([1, 1, -1]), [0, 0, 0]), "det"),
    (make_T(np.eye(3) * 2, [0, 0, 0]), "orthonormal"),
    (np.vstack([make_T(np.eye(3), [0, 0, 0])[:3], [0, 0, 1, 1]]), "last row"),
    (make_T(np.full((3, 3), np.nan), [0, 0, 0]), "NaN"),
    (np.eye(3), "shape"),
])
def test_invalid_matrices_detected(bad, msg):
    assert any(msg in p for p in rigid_problems(bad))
    with pytest.raises(ValueError):
        camera_pose(bad)


def test_rotation_angle():
    assert rotation_angle_deg(rot_y(10), rot_y(35)) == pytest.approx(25)


def test_trajectory_and_continuity(vkitti_root):
    ds = VKittiDataset(vkitti_root)
    traj = camera_trajectory(ds)
    np.testing.assert_allclose(traj.step_m.iloc[1:], np.sqrt(5), atol=1e-6)
    np.testing.assert_allclose(traj.step_rot_deg.iloc[1:], 10, atol=1e-4)
    assert check_extrinsics(ds, max_step_m=3, max_step_rot_deg=15)[0]
    ok, d = check_extrinsics(ds, max_step_m=2, max_step_rot_deg=15)
    assert not ok and d["jumps"] == [1, 2]


def test_stereo_baseline_confirms_direction(vkitti_root):
    ext = load_extrinsics(vkitti_root / "Scene01/clone/extrinsic.txt")
    ok, d = check_stereo_baseline(ext)
    assert ok and d["frames_checked"] == 3
    # Misreading the file as camera->world must fail this check.
    flipped = {k: invert_rigid(T) for k, T in ext.items()}
    assert not check_stereo_baseline(flipped)[0]
