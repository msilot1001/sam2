"""Deterministic frame-drop patterns and their manifests (Milestone 3).

Population: the post-prompt frames of a clip (prompt = original_frame_ids[0], never dropped).
Randomness: no RNG library. Candidates are ranked by sha256("<rng_key>|<i>") and the first k are
taken, i.e. a uniformly random k-subset that is identical on every machine and Python/numpy
version. rng_key = "m3|<scene>|<variation>|cam<c>|track<t>|<condition>|seed<s>" (no hash()).
Ground truth (masks, visibility, motion) is never an input.

random_<p>: drop round_half_up(p * n) post-prompt frames.
burst_<p>:  k = round_half_up(p * n / L) bursts of exactly L consecutive ORIGINAL frames, separated
            by >= min_gap received frames (so bursts never merge); k is capped at the largest count
            that fits; no partial bursts. Burst starts are a uniform draw over all valid layouts
            (stars and bars: choose k of slack + k slots).
actual_drop_rate = dropped / n, with n = post-prompt frames.
"""
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1
FRAME_COLUMNS = ["original_frame_id", "is_prompt", "is_received", "is_dropped", "received_video_idx"]


class ManifestError(ValueError):
    pass


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def hash_order(key: str, n: int) -> list[int]:
    return sorted(range(n), key=lambda i: hashlib.sha256(f"{key}|{i}".encode()).digest())


def rng_key(scene, variation, camera_id, track_id, condition, seed) -> str:
    return f"m3|{scene}|{variation}|cam{camera_id}|track{track_id}|{condition}|seed{seed}"


def random_drop_positions(n: int, rate: float, key: str) -> list[int]:
    k = min(round_half_up(rate * n), n)
    return sorted(hash_order(key, n)[:k])


def burst_drop_positions(n: int, rate: float, key: str, L: int = 4, gap: int = 1) -> tuple[list[int], int, bool]:
    """(burst start positions in 0..n-1, requested burst count, whether the request fit)."""
    k_req = round_half_up(rate * n / L)
    k_max = (n + gap) // (L + gap) if n >= L else 0
    k = min(k_req, k_max)
    if k == 0:
        return [], k_req, k_req == 0
    slack = n - (L * k + gap * (k - 1))
    chosen = sorted(hash_order(key, slack + k)[:k])
    starts = [c + i * (L + gap - 1) for i, c in enumerate(chosen)]
    return starts, k_req, k == k_req


@dataclass
class DropPattern:
    meta: dict
    frames: pd.DataFrame = field(repr=False)
    csv_sha256: str | None = None          # set by load(): sha256 of the saved frames CSV

    @property
    def received_ids(self) -> tuple[int, ...]:
        return tuple(int(f) for f in self.frames[self.frames.is_received].original_frame_id)

    @property
    def name(self) -> str:
        return f"{self.meta['condition']}_seed{self.meta['seed']}"


def generate(scene: str, variation: str, camera_id: int, track_id: int, original_frame_ids, condition: str,
             kind: str, rate: float, seed: int, burst_length: int = 4, min_gap: int = 1) -> DropPattern:
    ids = [int(f) for f in original_frame_ids]
    if not ids or ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError("original_frame_ids must be a non-empty contiguous increasing range")
    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")
    n = len(ids) - 1
    key = rng_key(scene, variation, camera_id, track_id, condition, seed)
    bursts, k_req, fits = [], None, True
    if kind == "random":
        drop_pos = random_drop_positions(n, rate, key)
    elif kind == "burst":
        starts, k_req, fits = burst_drop_positions(n, rate, key, burst_length, min_gap)
        drop_pos = [s + j for s in starts for j in range(burst_length)]
        bursts = [{"start": ids[1 + s], "end": ids[s + burst_length], "length": burst_length} for s in starts]
    else:
        raise ValueError(f"unknown kind {kind!r}")
    dropped = {ids[1 + p] for p in drop_pos}
    frames = pd.DataFrame({"original_frame_id": ids})
    frames["is_prompt"] = frames.original_frame_id == ids[0]
    frames["is_dropped"] = frames.original_frame_id.isin(dropped)
    frames["is_received"] = ~frames.is_dropped
    frames["received_video_idx"] = pd.Series(pd.NA, index=frames.index, dtype="Int64")
    frames.loc[frames.is_received, "received_video_idx"] = range(int(frames.is_received.sum()))
    frames = frames[FRAME_COLUMNS]
    meta = {
        "schema_version": SCHEMA_VERSION, "scene": scene, "variation": variation, "camera_id": camera_id,
        "track_id": track_id, "condition": condition, "kind": kind, "seed": seed, "rng_key": key,
        "prompt_original_frame_id": ids[0], "original_start_frame": ids[0], "original_end_frame": ids[-1],
        "requested_drop_rate": rate, "actual_drop_rate": len(dropped) / n if n else 0.0,
        "rate_denominator": "post_prompt_frames", "rounding": "round_half_up",
        "burst_length": burst_length if kind == "burst" else None,
        "min_gap_between_bursts": min_gap if kind == "burst" else None,
        "requested_number_of_bursts": k_req, "number_of_bursts": len(bursts), "request_fits": fits,
        "bursts": bursts, "dropped_original_frame_ids": sorted(dropped),
        "total_original_frames": len(ids), "post_prompt_frames": n,
        "total_received_frames": len(ids) - len(dropped), "total_dropped_frames": len(dropped),
    }
    pattern = DropPattern(meta, frames)
    validate(pattern)
    return pattern


def validate(p: DropPattern) -> None:
    """Structural invariants; raises ManifestError."""
    m, f = p.meta, p.frames

    def need(cond, msg):
        if not cond:
            raise ManifestError(f"{m.get('scene')}/track{m.get('track_id')}/{m.get('condition')}/seed{m.get('seed')}: {msg}")

    need(list(f.columns) == FRAME_COLUMNS, f"columns {list(f.columns)}")
    ids = f.original_frame_id.tolist()
    need(ids == list(range(m["original_start_frame"], m["original_end_frame"] + 1)), "original ids not the full contiguous range")
    need(ids[0] == m["prompt_original_frame_id"] and f.is_prompt.tolist() == [True] + [False] * (len(ids) - 1),
         "prompt must be the first and only prompt frame")
    need(bool(f.is_received.iloc[0]) and not bool(f.is_dropped.iloc[0]), "prompt frame dropped")
    need((f.is_received ^ f.is_dropped).all(), "each frame must be exactly one of received/dropped")
    rec = f[f.is_received]
    need(rec.received_video_idx.tolist() == list(range(len(rec))), "received_video_idx not contiguous 0..R-1 in order")
    need(f[f.is_dropped].received_video_idx.isna().all(), "dropped frame has a video index")
    dropped = f[f.is_dropped].original_frame_id.tolist()
    need(dropped == m["dropped_original_frame_ids"], "dropped ids disagree with metadata")
    need(m["total_original_frames"] == len(f) and m["total_dropped_frames"] == len(dropped)
         and m["total_received_frames"] == len(rec), "totals disagree")
    n = len(f) - 1
    need(abs(m["actual_drop_rate"] - (len(dropped) / n if n else 0.0)) < 1e-12, "actual_drop_rate wrong")
    if m["kind"] == "burst":
        L, gap = m["burst_length"], m["min_gap_between_bursts"]
        need(m["number_of_bursts"] == len(m["bursts"]), "burst count")
        covered = []
        for b in m["bursts"]:
            need(b["end"] - b["start"] + 1 == L == b["length"], f"burst {b} length != {L}")
            covered += list(range(b["start"], b["end"] + 1))
        need(covered == dropped, "dropped frames are not exactly the union of the bursts")
        for a, b in zip(m["bursts"], m["bursts"][1:]):
            need(b["start"] - a["end"] - 1 >= gap, f"bursts {a} and {b} closer than {gap} received frame(s)")
    else:
        need(m["bursts"] == [] and m["burst_length"] is None, "random pattern carries bursts")


# ----------------------------------------------------------------------------- manifest I/O
def paths(root: Path, scene: str, track_id: int, name: str) -> tuple[Path, Path]:
    d = Path(root) / f"{scene}_track{track_id:03d}"
    return d / f"{name}.json", d / f"{name}.csv"


def _csv_bytes(frames: pd.DataFrame) -> bytes:
    return frames.to_csv(index=False, lineterminator="\n").encode()


def save(p: DropPattern, root: Path) -> tuple[Path, Path]:
    """Write CSV + JSON. An existing manifest is never replaced: identical -> no-op, different -> error."""
    jpath, cpath = paths(root, p.meta["scene"], p.meta["track_id"], p.name)
    data = _csv_bytes(p.frames)
    meta = {**p.meta, "frames_csv": cpath.name, "frames_csv_sha256": hashlib.sha256(data).hexdigest()}
    if jpath.exists() or cpath.exists():
        old = load(jpath)
        if old.meta != p.meta or not old.frames.equals(p.frames):
            raise ManifestError(f"{jpath} exists with a different pattern; refusing to overwrite")
        return jpath, cpath
    jpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_bytes(data)
    jpath.write_text(json.dumps(meta, indent=2) + "\n")
    return jpath, cpath


def load(jpath: Path, regenerate_check: bool = True) -> DropPattern:
    """Load + validate a manifest; optionally confirm the generator still produces the same pattern."""
    jpath = Path(jpath)
    meta = json.loads(jpath.read_text())
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"{jpath}: unsupported schema {meta.get('schema_version')}")
    cpath = jpath.with_name(meta["frames_csv"])
    data = cpath.read_bytes()
    if hashlib.sha256(data).hexdigest() != meta["frames_csv_sha256"]:
        raise ManifestError(f"{cpath}: sha256 does not match the JSON metadata (file edited?)")
    frames = pd.read_csv(cpath, dtype={"received_video_idx": "Int64"})
    p = DropPattern({k: v for k, v in meta.items() if k not in ("frames_csv", "frames_csv_sha256")}, frames)
    validate(p)
    if regenerate_check:
        m = p.meta
        again = generate(m["scene"], m["variation"], m["camera_id"], m["track_id"], frames.original_frame_id,
                         m["condition"], m["kind"], m["requested_drop_rate"], m["seed"],
                         m["burst_length"] or 4, m["min_gap_between_bursts"] or 1)
        if again.meta != p.meta or not again.frames.equals(frames):
            raise ManifestError(f"{jpath}: generator no longer reproduces this manifest")
    p.csv_sha256 = meta["frames_csv_sha256"]
    return p
