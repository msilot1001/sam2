from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"rgb {rgb.shape[:2]} and mask {mask.shape} differ")
    out = rgb.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(color, np.float32)
    return out.astype(np.uint8)


def save_overlay(sample: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.42, 4.2), dpi=100)
    ax.imshow(overlay(sample["rgb"], sample["mask"]))
    ax.set_title(f"{sample['scene']}/{sample['variation']}/Camera_{sample['camera_id']}  "
                 f"frame {sample['original_frame_id']}  track {sample['track_id']}  "
                 f"mask px {int(sample['mask'].sum())}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_trajectory_plots(traj, tag: str, out_dir: Path) -> list[Path]:
    """traj: DataFrame from validation.camera_trajectory. World y points down in VKITTI2."""
    paths, prefix = [], tag.replace("/", "_")
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(traj.x, traj.z, c=traj.frame, s=6, cmap="viridis")
    ax.plot(traj.x.iloc[0], traj.z.iloc[0], "go", label=f"start (frame {traj.frame.iloc[0]})")
    ax.plot(traj.x.iloc[-1], traj.z.iloc[-1], "rs", label=f"end (frame {traj.frame.iloc[-1]})")
    ax.set(xlabel="world x [m]", ylabel="world z [m]", title=f"{tag} camera trajectory (top view x-z)", aspect="equal")
    ax.legend()
    fig.colorbar(sc, label="original frame id")
    paths.append(_save(fig, out_dir / f"{prefix}_trajectory_2d.png"))

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot(traj.x, traj.z, -traj.y)
    ax.set(xlabel="x [m]", ylabel="z [m]", zlabel="-y (up) [m]", title=f"{tag} camera trajectory 3D")
    paths.append(_save(fig, out_dir / f"{prefix}_trajectory_3d.png"))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for c in "xyz":
        axes[0].plot(traj.frame, traj[c], label=c)
    axes[0].set(ylabel="position [m]")
    axes[0].legend()
    axes[1].plot(traj.frame, traj.step_m)
    axes[1].set(ylabel="step translation [m]")
    axes[2].plot(traj.frame, traj.step_rot_deg)
    axes[2].set(ylabel="step rotation [deg]", xlabel="original frame id")
    fig.suptitle(f"{tag} camera pose vs frame")
    paths.append(_save(fig, out_dir / f"{prefix}_trajectory_timeseries.png"))
    return paths


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
