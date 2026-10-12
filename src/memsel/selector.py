"""Memory selection policies for SAM 2.1 (mentor_plan_revision.md §5-§7), independent of SAM2.

Active set (K=7): prompt memory + most recent received memory + up to K-2=5 policy picks.
The bank keeps at most L=32 non-conditioning candidates (FIFO, oldest evicted); the prompt is
held separately. A frame enters the bank only through MemoryBank.add(), i.e. after its own
prediction, so the current frame can never be its own candidate.

Inputs are camera pose + ORIGINAL frame id only. Memory has no field for masks or target GT
pose, so neither can reach a policy. Δt is the original-frame-id difference (FPS unverified).

All ties go to the more recent candidate (smaller age). Returned sets are sorted by time.
"""
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from vkitti.calibration import rotation_angle_deg

POLICIES = ("recent-K", "time-distributed", "pose-only", "pose+dt")
BINS = ((1, 2), (3, 5), (6, 10), (11, 20), (21, None))  # age in original frames; None = open
MIN_SCALE_T_M = 0.01      # floors for sT / sR (plan §13): a near-stationary dev set must not
MIN_SCALE_R_DEG = 0.01    # turn tiny pose noise into full-range cost differences


@dataclass(frozen=True, slots=True, eq=False)  # identity equality: arrays have no truth value
class Memory:
    original_frame_id: int
    camera_position: np.ndarray   # (3,) world
    camera_rotation: np.ndarray   # (3,3) camera->world

    def __post_init__(self):
        if np.shape(self.camera_position) != (3,) or np.shape(self.camera_rotation) != (3, 3):
            raise ValueError("camera_position must be (3,), camera_rotation (3,3)")
        if not (np.all(np.isfinite(self.camera_position)) and np.all(np.isfinite(self.camera_rotation))):
            raise ValueError("camera pose contains NaN/Inf")


@dataclass(frozen=True)
class PoseScale:
    """C = w_t * clip(dT/s_t, 0, 1) + (1 - w_t) * clip(dR/s_r, 0, 1)."""
    s_t: float
    s_r_deg: float
    w_t: float = 0.5

    def __post_init__(self):
        if not 0 <= self.w_t <= 1:
            raise ValueError("w_t must be in [0, 1]")
        object.__setattr__(self, "s_t", max(float(self.s_t), MIN_SCALE_T_M))
        object.__setattr__(self, "s_r_deg", max(float(self.s_r_deg), MIN_SCALE_R_DEG))


def estimate_pose_scale(sequences, L: int = 32, q: float = 95.0, w_t: float = 0.5) -> PoseScale:
    """sT, sR = q-th percentile of (current, candidate) pose differences over every frame of the
    given DEVELOPMENT sequences, candidates being the previous L frames (no-drop window)."""
    dts, drs = [], []
    for seq in sequences:  # each: time-ordered list of Memory
        for i, cur in enumerate(seq):
            for cand in seq[max(0, i - L):i]:
                dt, dr = pose_difference(cur, cand)
                dts.append(dt)
                drs.append(dr)
    if not dts:
        raise ValueError("no frame pairs to estimate the pose scale from")
    return PoseScale(np.percentile(dts, q), np.percentile(drs, q), w_t)


def pose_difference(a: Memory, b: Memory) -> tuple[float, float]:
    """(translation [m], rotation [deg]) between two camera poses."""
    return (float(np.linalg.norm(a.camera_position - b.camera_position)),
            rotation_angle_deg(a.camera_rotation, b.camera_rotation))


def pose_cost(current: Memory, cand: Memory, scale: PoseScale) -> float:
    dt, dr = pose_difference(current, cand)
    return scale.w_t * min(dt / scale.s_t, 1.0) + (1 - scale.w_t) * min(dr / scale.s_r_deg, 1.0)


def age_bin(age: int, bins=BINS) -> int | None:
    for i, (lo, hi) in enumerate(bins):
        if age >= lo and (hi is None or age <= hi):
            return i
    return None


class MemoryBank:
    def __init__(self, prompt: Memory, L: int = 32):
        if L < 1:
            raise ValueError("L must be >= 1")
        self.prompt, self.L = prompt, L
        self._q: deque[Memory] = deque(maxlen=L)

    def add(self, m: Memory) -> None:
        """Call only after m's frame has been predicted and encoded."""
        if not isinstance(m, Memory):
            raise TypeError("bank accepts Memory objects only")
        if m.original_frame_id <= self.recent.original_frame_id:
            raise ValueError(f"frame {m.original_frame_id} is not newer than {self.recent.original_frame_id}")
        self._q.append(m)

    @property
    def candidates(self) -> tuple[Memory, ...]:
        return tuple(self._q)

    @property
    def recent(self) -> Memory:
        """Most recent received memory; the prompt itself until anything else arrives."""
        return self._q[-1] if self._q else self.prompt


class Selection(NamedTuple):
    active: tuple[Memory, ...]   # prompt + recent + chosen, unique, time-ordered
    chosen: tuple[Memory, ...]   # policy picks only, time-ordered


class MemorySelector:
    def __init__(self, policy: str, scale: PoseScale | None = None, K: int = 7, bins=BINS):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
        if policy.startswith("pose") and scale is None:
            raise ValueError(f"{policy} needs a PoseScale")
        self.n = K - 2
        if policy in ("time-distributed", "pose+dt") and len(bins) != self.n:
            raise ValueError(f"{len(bins)} time bins but K-2={self.n} slots")
        self.policy, self.scale, self.K, self.bins = policy, scale, K, bins

    def select(self, current: Memory, bank: MemoryBank) -> Selection:
        cur = current.original_frame_id
        if bank.prompt.original_frame_id >= cur or any(c.original_frame_id >= cur for c in bank.candidates):
            raise ValueError(f"bank holds frames not strictly before current frame {cur}")
        recent = bank.recent
        pool = [c for c in bank.candidates if c is not recent]
        age = {c.original_frame_id: cur - c.original_frame_id for c in pool}
        key_recent = lambda c: age[c.original_frame_id]  # noqa: E731
        if self.policy == "recent-K":
            chosen = sorted(pool, key=key_recent)[: self.n]
        else:
            cost = {c.original_frame_id: pose_cost(current, c, self.scale) for c in pool} if self.scale else {}
            method = {"time-distributed": self._time_distributed, "pose-only": self._pose_only,
                      "pose+dt": self._pose_dt}[self.policy]
            chosen = method(pool, age, cost, (bank.prompt.original_frame_id, recent.original_frame_id))
        fixed = {bank.prompt.original_frame_id: bank.prompt, recent.original_frame_id: recent}
        active = {**fixed, **{c.original_frame_id: c for c in chosen}}
        by_time = lambda ms: tuple(sorted(ms, key=lambda m: m.original_frame_id))  # noqa: E731
        return Selection(by_time(active.values()), by_time(chosen))

    def _binned(self, pool, age):
        groups = [[] for _ in self.bins]
        for c in pool:
            b = age_bin(age[c.original_frame_id], self.bins)
            if b is not None:
                groups[b].append(c)
        return groups

    def _pose_only(self, pool, age, cost, fixed_ids):
        return sorted(pool, key=lambda c: (cost[c.original_frame_id], age[c.original_frame_id]))[: self.n]

    def _pose_dt(self, pool, age, cost, fixed_ids):
        key = lambda c: (cost[c.original_frame_id], age[c.original_frame_id])  # noqa: E731
        chosen = [min(g, key=key) for g in self._binned(pool, age) if g]
        rest = sorted((c for c in pool if c not in chosen), key=key)
        return chosen + rest[: self.n - len(chosen)]

    def _time_distributed(self, pool, age, cost, fixed_ids):
        chosen = []
        for (lo, hi), g in zip(self.bins, self._binned(pool, age)):
            if not g:
                continue
            target = (lo + hi) / 2 if hi is not None else float(np.median([age[c.original_frame_id] for c in g]))
            chosen.append(min(g, key=lambda c: (abs(age[c.original_frame_id] - target), age[c.original_frame_id])))
        # Empty bins: repeatedly take the candidate whose minimum time gap to everything already
        # active (prompt, recent, earlier picks) is largest; ties -> more recent.
        rest = [c for c in pool if c not in chosen]
        while len(chosen) < self.n and rest:
            taken = list(fixed_ids) + [c.original_frame_id for c in chosen]
            best = max(rest, key=lambda c: (min(abs(c.original_frame_id - t) for t in taken), -age[c.original_frame_id]))
            chosen.append(best)
            rest.remove(best)
        return chosen
