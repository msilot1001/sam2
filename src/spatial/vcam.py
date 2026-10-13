"""M6 Track A: causal "virtual camera" input for the frozen SAM 2 video predictor.

SAM 2 is fed a synthetic video: every frame is a fixed-size square crop of the original frame (size chosen once
from the prompt mask), whose centre follows the previous frame's online prediction. Because the crop size never
changes, the synthetic video is an ordinary pan-only video at one scale; all SAM 2 memory stays in that video's
coordinates and nothing is ever transformed between coordinate systems. Only the output mask is pasted back.

Causality: `LazyCropFrames[t]` is materialised only when SAM 2 asks for frame t, using the box set by the caller
after frame t-1 was predicted. Nothing reads GT except the prompt at frame 0.
"""
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image

TARGET = 0.25        # object box side / crop side (pre-registered, M5.6 proposal)
MAX_ZOOM = 4.0       # crop side >= min(H, W) / MAX_ZOOM (pre-registered in M6)


def bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1) if len(ys) else None


def crop_side(gt0: np.ndarray):
    """Fixed crop side from the prompt mask, or None (no crop: identical to the full-frame baseline)."""
    H, W = gt0.shape
    y0, y1, x0, x1 = bbox(gt0)
    s = max(y1 - y0, x1 - x0) / TARGET
    if s >= min(H, W):
        return None
    return int(round(max(s, min(H, W) / MAX_ZOOM)))


def place(center, side, hw):
    """Square box of `side` centred at (cy, cx), shifted to lie inside the frame. Returns (y0, y1, x0, x1)."""
    H, W = hw
    y0 = int(round(center[0] - side / 2))
    x0 = int(round(center[1] - side / 2))
    y0, x0 = min(max(y0, 0), H - side), min(max(x0, 0), W - side)
    return y0, y0 + side, x0, x0 + side


def center_of(m: np.ndarray):
    b = bbox(m)
    return None if b is None else ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2)


def paste(crop_mask: np.ndarray, box, hw) -> np.ndarray:
    out = np.zeros(hw, bool)
    y0, y1, x0, x1 = box
    out[y0:y1, x0:x1] = crop_mask
    return out


def touches_border(crop_mask: np.ndarray, box, hw) -> bool:
    """Mask touches a crop edge that is not also a frame edge."""
    y0, y1, x0, x1 = box
    H, W = hw
    return bool((y0 > 0 and crop_mask[0].any()) or (y1 < H and crop_mask[-1].any())
                or (x0 > 0 and crop_mask[:, 0].any()) or (x1 < W and crop_mask[:, -1].any()))


class LazyCropFrames:
    """Indexable like SAM 2's frame tensor. Reproduces sam2.utils.misc preprocessing exactly (identity box ==
    official frames bit-for-bit). `box` must be set by the caller before SAM 2 requests frame t."""

    def __init__(self, paths, image_size=1024, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.paths, self.size = list(paths), image_size
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.box, self.boxes = None, {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, t):
        y0, y1, x0, x1 = self.box
        im = Image.open(self.paths[t]).convert("RGB")
        if (y0, y1, x0, x1) != (0, im.size[1], 0, im.size[0]):
            im = im.crop((x0, y0, x1, y1))
        img = torch.from_numpy(np.array(im.resize((self.size, self.size))) / 255.0).permute(2, 0, 1)
        x = torch.zeros(3, self.size, self.size, dtype=torch.float32)
        x[:] = img
        x -= self.mean
        x /= self.std
        self.boxes[t] = self.box
        return x


@contextmanager
def lazy_loader(frames: LazyCropFrames, hw):
    """Make predictor.init_state use `frames` (video res = hw) instead of loading the whole directory."""
    import sam2.sam2_video_predictor as vp
    orig = vp.load_video_frames
    vp.load_video_frames = lambda **k: (frames, hw[0], hw[1])
    try:
        yield
    finally:
        vp.load_video_frames = orig
