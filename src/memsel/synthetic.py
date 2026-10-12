"""Deterministic synthetic CAMERA-POSE trajectories for selector feasibility (M4.5).

Geometry only: positions, rotations and frame ids. No images, no masks -- these are never paired
with VKITTI RGB or used to report segmentation accuracy.

Conventions follow VKITTI2: world y points DOWN, the vehicle drives in the x-z plane; camera axes
x right, y down, z forward; Memory.camera_rotation maps camera -> world. A heading psi turns the
forward axis to world (sin psi, 0, cos psi), i.e. rotation about the world y axis.

Speeds are metres per ORIGINAL frame. Physical plausibility (see `plausibility`) needs a frame
rate; VKITTI2's rate is not documented, so 10 Hz (KITTI, MILESTONE_1 §4) is an explicit assumption.
"""
import math

import numpy as np

from .selector import Memory

CAMERA_HEIGHT_M = 1.65          # y = -1.65 (up is -y)
MIN_TURN_RADIUS_M = 5.0         # passenger-car kerb-to-kerb radius ~5-6 m
MAX_LATERAL_ACC = 4.0           # m/s^2, upper bound of ordinary (non-emergency) driving


def rot_y(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _walk(segments, v: float):
    """segments: [("straight", n_frames) | ("arc", radius_m, angle_rad, sign)]; sign +1 = left."""
    x = z = psi = 0.0
    poses = [(x, z, psi)]
    for seg in segments:
        if seg[0] == "straight":
            for _ in range(seg[1]):
                x += v * math.sin(psi)
                z += v * math.cos(psi)
                poses.append((x, z, psi))
        else:
            _, radius, angle, sign = seg
            n = max(1, round(radius * angle / v))
            dpsi = sign * angle / n
            for _ in range(n):
                # move along the chord of a constant-curvature arc
                psi_mid = psi + dpsi / 2
                step = 2 * radius * math.sin(abs(dpsi) / 2)
                x += step * math.sin(psi_mid)
                z += step * math.cos(psi_mid)
                psi += dpsi
                poses.append((x, z, psi))
    return poses


def to_memories(poses, jitter_t: float = 0.0, jitter_r_deg: float = 0.0, seed: int = 0) -> list[Memory]:
    rng = np.random.default_rng(seed)
    out = []
    for i, (x, z, psi) in enumerate(poses):
        p = np.array([x, -CAMERA_HEIGHT_M, z]) + (rng.normal(0, jitter_t, 3) if jitter_t else 0.0)
        a = psi + (math.radians(rng.normal(0, jitter_r_deg)) if jitter_r_deg else 0.0)
        out.append(Memory(i, p, rot_y(a)))
    return out


def straight(n: int = 200, v: float = 1.0):
    return _walk([("straight", n)], v)


def turn(v: float = 1.0, radius: float = 15.0, before: int = 60, after: int = 100):
    return _walk([("straight", before), ("arc", radius, math.pi / 2, +1), ("straight", after)], v)


def stationary(n: int = 200):
    return [(0.0, 0.0, 0.0)] * (n + 1)


def u_turn(v: float = 1.0, radius: float = 6.0, leg: int = 60):
    return _walk([("straight", leg), ("arc", radius, math.pi, +1), ("straight", leg)], v)


def loop(v: float = 1.0, radius: float = 10.0, laps: float = 2.0, lead: int = 10):
    return _walk([("straight", lead), ("arc", radius, 2 * math.pi * laps, +1), ("straight", 10)], v)


def plausibility(v_m_per_frame: float, radius_m: float | None, fps: float = 10.0) -> dict:
    """Lateral acceleration v^2/R at the ASSUMED frame rate; flags physically implausible turns."""
    v = v_m_per_frame * fps
    if radius_m is None:
        return {"speed_mps": v, "lateral_acc": 0.0, "plausible": True}
    a = v * v / radius_m
    return {"speed_mps": v, "lateral_acc": a,
            "plausible": radius_m >= MIN_TURN_RADIUS_M and a <= MAX_LATERAL_ACC}


def min_loop_period_frames(fps: float = 10.0, a_max: float = MAX_LATERAL_ACC, r_min: float = MIN_TURN_RADIUS_M):
    """Shortest closed-loop revisit (same position AND heading) a plausible car can make, in frames.

    Period P = 2*pi*R*fps / v with R >= max(r_min, v^2/a_max); minimised at v = sqrt(r_min * a_max).
    """
    v = math.sqrt(r_min * a_max)
    return 2 * math.pi * r_min * fps / v, v
