"""Thin adapter between VKittiDataset and the UNMODIFIED official SAM 2 video predictor.

Index spaces are kept apart on purpose:
  video_idx          0..N-1, what SAM 2 sees (files {video_idx:05d}.jpg)
  original_frame_id  VKITTI2 frame number (rgb_{original:05d}.jpg)
The clip folder holds symlinks to the original JPEGs, so SAM 2 decodes the exact dataset bytes.

`run_clip` receives the prompt mask and nothing else from the ground truth: GT for later frames is
loaded only by the evaluation, after inference.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Clip:
    scene: str
    track_id: int
    original_frame_ids: tuple[int, ...]   # video_idx i <-> original_frame_ids[i]; [0] is the prompt frame

    @property
    def name(self) -> str:
        return f"{self.scene}_track{self.track_id:03d}"

    def mapping(self) -> pd.DataFrame:
        return pd.DataFrame({"video_idx": range(len(self.original_frame_ids)),
                             "original_frame_id": self.original_frame_ids})


def make_clip(scene: str, track_row, max_frames: int | None = None) -> Clip:
    """Prompt frame .. last observable frame (inclusive), contiguous, no drops (M2)."""
    first, last = int(track_row.prompt_frame), int(track_row.last_frame)
    ids = tuple(range(first, last + 1))
    return Clip(scene, int(track_row.track_id), ids[:max_frames] if max_frames else ids)


def write_clip_folder(ds, clip: Clip, root: Path) -> Path:
    """Folder of {video_idx:05d}.jpg symlinks to the original RGB files + mapping.csv."""
    if ds.scene != clip.scene:
        raise ValueError(f"dataset scene {ds.scene} != clip scene {clip.scene}")
    d = root / clip.name
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.jpg"):
        old.unlink()
    for i, f in enumerate(clip.original_frame_ids):
        src = ds.rgb_path(f).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        (d / f"{i:05d}.jpg").symlink_to(src)
    clip.mapping().to_csv(d / "mapping.csv", index=False)
    return d


def sync(device) -> None:
    import torch
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run_clip(predictor, video_dir: Path, prompt_mask: np.ndarray, obj_id: int = 1) -> dict:
    """Prompt with the GT mask at video_idx 0 only, then propagate forward with default settings.

    Returns {"masks": {video_idx: bool (H, W)}, "init_s", "latency_s": {video_idx: s}, "mem": [...]}.
    """
    import torch
    if prompt_mask.dtype != bool or prompt_mask.ndim != 2 or not prompt_mask.any():
        raise ValueError("prompt mask must be a non-empty 2-D bool array")
    device = predictor.device
    with torch.inference_mode():
        t0 = time.perf_counter()
        state = predictor.init_state(video_path=str(video_dir))
        predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=prompt_mask)
        sync(device)
        init_s = time.perf_counter() - t0
        masks, latency, mem = {}, {}, []
        t = time.perf_counter()
        for idx, obj_ids, logits in predictor.propagate_in_video(state):  # forward from idx 0
            if list(obj_ids) != [obj_id]:
                raise RuntimeError(f"unexpected object ids {obj_ids}")
            masks[idx] = (logits[0, 0] > 0.0).cpu().numpy()  # .cpu() also synchronises
            now = time.perf_counter()
            latency[idx] = now - t
            t = now
            if device.type == "mps":
                mem.append(torch.mps.driver_allocated_memory())
        n = state["num_frames"]
        predictor.reset_state(state)
    if sorted(masks) != list(range(n)):
        raise RuntimeError("propagation did not return every frame exactly once")
    return {"masks": masks, "init_s": init_s, "latency_s": latency, "mps_driver_bytes": mem}
