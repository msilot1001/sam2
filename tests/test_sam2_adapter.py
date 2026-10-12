import hashlib
import inspect
from collections import namedtuple
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from conftest import FRAMES, H, W

torch = pytest.importorskip("torch")
from baseline.sam2_vos import Clip, make_clip, run_clip, write_clip_folder  # noqa: E402
from vkitti.dataset import VKittiDataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
REAL = ROOT / "data" / "vkitti2" / "Scene01" / "clone" / "frames" / "rgb" / "Camera_0"
Row = namedtuple("Row", "track_id prompt_frame last_frame")

# sha256 of the files at upstream commit 2b90b9f (recorded from a fresh clone, 2026-10-12).
OFFICIAL_SHA256 = {
    "modeling/sam2_base.py": "69a46b44e8625f509791352bd09beaafe589cd7d872721384e63d37f9fdc6e41",
    "sam2_video_predictor.py": "912555920a77f72ded07839efa33fcceecc79f3523abbec783f0be46ee2c55c2",
    "modeling/memory_attention.py": "07358cb7f58ec3788e88ff4e6415f8a9d628a3963529d6e2c915494a5347c8e2",
    "modeling/memory_encoder.py": "73f7089ae5fdacdfcaaf3deca1ca6d9f84d89e7deb9cf97589622a9ef17ba42f",
    "configs/sam2.1/sam2.1_hiera_t.yaml": "f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d",
    "build_sam.py": "856d64d71e44407401297b551884fb4cd1aba8082de0ff5fc60fac3dd42f094d",
}


class FakePredictor:
    """Records every call; predicts the prompt mask at every frame."""
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def init_state(self, video_path):
        n = len(list(Path(video_path).glob("*.jpg")))
        self.calls.append(("init_state", video_path))
        return {"num_frames": n}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        self.calls.append(("add_new_mask", frame_idx, obj_id, mask.copy()))
        self.mask = mask

    def propagate_in_video(self, state):
        self.calls.append(("propagate_in_video",))
        logits = torch.where(torch.from_numpy(self.mask), 5.0, -5.0)[None, None]
        for i in range(state["num_frames"]):
            yield i, [1], logits

    def reset_state(self, state):
        self.calls.append(("reset_state",))


def clip_for(root, last=2):
    return make_clip("Scene01", Row(4, 0, last))


def test_clip_mapping_and_determinism(vkitti_root, tmp_path):
    ds = VKittiDataset(vkitti_root)
    c = make_clip("Scene01", Row(0, 1, 2))
    assert c == make_clip("Scene01", Row(0, 1, 2))
    assert c.original_frame_ids == (1, 2)
    assert c.mapping().to_dict("records") == [{"video_idx": 0, "original_frame_id": 1},
                                              {"video_idx": 1, "original_frame_id": 2}]
    d = write_clip_folder(ds, c, tmp_path)
    assert sorted(p.name for p in d.glob("*.jpg")) == ["00000.jpg", "00001.jpg"]
    assert (d / "00000.jpg").resolve() == ds.rgb_path(1).resolve()   # video idx 0 = original frame 1
    assert (d / "00001.jpg").resolve() == ds.rgb_path(2).resolve()
    assert make_clip("Scene01", Row(0, 0, 2), max_frames=2).original_frame_ids == (0, 1)


def test_official_loader_sees_frames_in_mapping_order(vkitti_root, tmp_path):
    from sam2.utils.misc import load_video_frames_from_jpg_images
    ds = VKittiDataset(vkitti_root)
    c = make_clip("Scene01", Row(0, 0, 2))
    d = write_clip_folder(ds, c, tmp_path)
    imgs, h, w = load_video_frames_from_jpg_images(str(d), image_size=16, offload_video_to_cpu=True,
                                                   compute_device=torch.device("cpu"))
    assert (h, w) == (H, W) and imgs.shape[0] == len(FRAMES)
    red = imgs[:, 0].mean(dim=(1, 2)).numpy()       # fixture encodes the frame id in the red channel
    assert np.all(np.diff(red) > 0)


def test_only_initial_gt_prompt(vkitti_root, tmp_path):
    ds = VKittiDataset(vkitti_root)
    c = clip_for(vkitti_root)
    d = write_clip_folder(ds, c, tmp_path)
    prompt = ds.get_frame(c.original_frame_ids[0], c.track_id)["mask"]
    fake = FakePredictor()
    res = run_clip(fake, d, prompt)
    mask_calls = [x for x in fake.calls if x[0] == "add_new_mask"]
    assert len(mask_calls) == 1 and mask_calls[0][1] == 0 and mask_calls[0][2] == 1
    np.testing.assert_array_equal(mask_calls[0][3], prompt)
    assert [x[0] for x in fake.calls] == ["init_state", "add_new_mask", "propagate_in_video", "reset_state"]
    assert sorted(res["masks"]) == [0, 1, 2]
    assert all(m.shape == (H, W) and m.dtype == bool for m in res["masks"].values())
    # The adapter cannot reach later GT: it receives no dataset, only the prompt mask.
    assert list(inspect.signature(run_clip).parameters) == ["predictor", "video_dir", "prompt_mask", "obj_id"]


def test_prompt_validation(tmp_path):
    with pytest.raises(ValueError):
        run_clip(FakePredictor(), tmp_path, np.zeros((H, W), bool))
    with pytest.raises(ValueError):
        run_clip(FakePredictor(), tmp_path, np.ones((H, W), np.uint8))


def test_scene_mismatch_rejected(vkitti_root, tmp_path):
    with pytest.raises(ValueError):
        write_clip_folder(VKittiDataset(vkitti_root), Clip("Scene06", 0, (0, 1)), tmp_path)


def test_official_sam2_files_unmodified():
    pkg = Path(find_spec("sam2").origin).parent
    for rel, digest in OFFICIAL_SHA256.items():
        assert hashlib.sha256((pkg / rel).read_bytes()).hexdigest() == digest, rel


@pytest.mark.realdata
@pytest.mark.skipif(not (CKPT.exists() and REAL.is_dir()), reason="checkpoint or VKITTI2 data missing")
def test_real_sam2_cpu_three_frames(tmp_path):
    from sam2.build_sam import build_sam2_video_predictor

    from baseline.metrics import j_measure
    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", str(CKPT), device=torch.device("cpu"))
    # Default memory behaviour is untouched.
    assert (pred.num_maskmem, pred.memory_temporal_stride_for_eval, pred.max_cond_frames_in_attn) == (7, 1, -1)
    assert pred.use_obj_ptrs_in_encoder and pred.add_tpos_enc_to_obj_ptrs
    ds = VKittiDataset(ROOT / "data" / "vkitti2", "Scene01")
    c = make_clip("Scene01", Row(90, 130, 235), max_frames=3)
    d = write_clip_folder(ds, c, tmp_path)
    res = run_clip(pred, d, ds.get_frame(130, 90)["mask"])
    for i, f in enumerate(c.original_frame_ids):
        assert res["masks"][i].shape == (375, 1242)
        assert j_measure(res["masks"][i], ds.get_frame(f, 90)["mask"]) > 0.5
