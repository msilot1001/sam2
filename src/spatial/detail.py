"""Spatial-detail mask measures and SAM 2 mask-resolution round trips (M5.5). numpy/torch only.

All functions take 2-D bool masks at the evaluation (video) resolution. Morphology uses square
(3x3 / (2r+1)^2) structuring elements via max-pooling; components use 8-connectivity.
"""
import numpy as np
import torch
import torch.nn.functional as F

IMAGE_SIZE = 1024          # SAM 2 input resolution
LOW_RES = IMAGE_SIZE // 4  # mask-decoder / stored pred_masks resolution


def _t(m):
    return torch.as_tensor(np.ascontiguousarray(m), dtype=torch.float32)[None, None]


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    return F.max_pool2d(_t(m), 2 * r + 1, 1, r)[0, 0].numpy() > 0.5 if r > 0 else m.astype(bool)


def erode(m: np.ndarray, r: int) -> np.ndarray:
    return ~dilate(~m.astype(bool), r)


def opening(m: np.ndarray, r: int) -> np.ndarray:
    return dilate(erode(m, r), r) & m.astype(bool)


def thin_part(m: np.ndarray, r: int = 2) -> np.ndarray:
    """Pixels of m in structures narrower than 2r+1 px (removed by an opening of radius r)."""
    return m.astype(bool) & ~opening(m, r)


def components(m: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected component labels (0 = background), labels 1..n in raster order of first pixel."""
    m = m.astype(bool)
    if not m.any():
        return np.zeros(m.shape, np.int64), 0
    H, W = m.shape
    lab = torch.where(torch.from_numpy(m), torch.arange(1, H * W + 1, dtype=torch.float64).reshape(H, W), 0.)
    fg = torch.from_numpy(m)
    while True:                           # propagate the max id inside each component until stable
        nxt = F.max_pool2d(lab[None, None], 3, 1, 1)[0, 0] * fg
        for _ in range(16):
            nxt = F.max_pool2d(nxt[None, None], 3, 1, 1)[0, 0] * fg
        if torch.equal(nxt, lab):
            break
        lab = nxt
    ids, inv = np.unique(lab.numpy().astype(np.int64), return_inverse=True)
    inv = inv.reshape(H, W)
    # relabel to 1..n (0 stays 0 because ids[0] == 0 when background exists)
    if ids[0] != 0:
        inv = inv + 1
    return inv, int(len(ids) - (1 if ids[0] == 0 else 0))


def component_retention(pred: np.ndarray, gt: np.ndarray, min_area: int = 20, cover: float = 0.5) -> float:
    """Fraction of GT components (area >= min_area) with >= `cover` of their pixels predicted. NaN if none."""
    lab, n = components(gt)
    if n == 0:
        return float("nan")
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    hit = np.bincount(lab.ravel(), weights=pred.astype(bool).ravel(), minlength=n + 1)
    keep = np.nonzero(areas[1:] >= min_area)[0] + 1
    return float("nan") if len(keep) == 0 else float(np.mean(hit[keep] / areas[keep] >= cover))


def thin_recall(pred: np.ndarray, gt: np.ndarray, r: int = 2) -> float:
    """Recall of GT thin structures (see thin_part). NaN if GT has none."""
    tp = thin_part(gt, r)
    n = tp.sum()
    return float("nan") if n == 0 else float((pred.astype(bool) & tp).sum() / n)


# ----------------------------------------------------------------------------- SAM 2 resolution path
def to_model(mask: np.ndarray, mode: str = "official", thr: float = 0.5) -> torch.Tensor:
    """Video-res bool mask -> 1x1x1024x1024 float {0,1}, as add_new_mask does ('official' =
    bilinear+antialias, threshold >= thr). Variants change exactly one factor."""
    x = _t(mask)
    if mode == "official":
        y = F.interpolate(x, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False, antialias=True)
    elif mode == "nearest":
        y = F.interpolate(x, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest-exact")
    elif mode == "bilinear_noaa":
        y = F.interpolate(x, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False, antialias=False)
    else:
        raise ValueError(mode)
    return (y >= thr).float()


def to_video(logits: torch.Tensor, hw) -> np.ndarray:
    """Any-res logits (1x1xhxw) -> video-res bool, as _get_orig_video_res_output (bilinear, > 0)."""
    return (F.interpolate(logits.float().cpu(), size=tuple(hw), mode="bilinear", align_corners=False)[0, 0] > 0).numpy()


def mask_as_output_lowres(mask_1024: torch.Tensor) -> torch.Tensor:
    """_use_mask_as_output: {0,1} -> logits -10/+10 -> antialiased bilinear /4 (256x256)."""
    hi = mask_1024.float() * 20.0 - 10.0
    return F.interpolate(hi, (LOW_RES, LOW_RES), mode="bilinear", align_corners=False, antialias=True)


def lowres_roundtrip(mask: np.ndarray) -> np.ndarray:
    """Best case of SAM 2's output path for a mask: official prompt resize -> 256x256 low-res logits ->
    bilinear back to video res. Upper bound on what a 256x256 logit map, upsampled officially, can express
    when built from this mask (diagnostic ceiling; uses GT, not deployable)."""
    return to_video(mask_as_output_lowres(to_model(mask)), mask.shape)


def model_roundtrip(mask: np.ndarray, mode: str = "official", thr: float = 0.5) -> np.ndarray:
    """1024x1024 representation only (prompt resize, then back): isolates prompt-resampling loss."""
    return to_video(to_model(mask, mode, thr) * 20.0 - 10.0, mask.shape)
