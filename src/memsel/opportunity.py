"""Selection-opportunity simulation over a received-frame sequence (M4.5). No SAM 2 inference.

Replays received frames through MemoryBank (FIFO L, only frames received BEFORE the current one)
and records the 5 policy-selected memories per policy; the fixed prompt/recent memories are kept
out of the policy columns so overlaps compare only the slots a policy can change.
"""
import numpy as np
import pandas as pd

from .selector import MemoryBank, pose_difference

K = 7


def overlap(a: set, b: set) -> float:
    n = max(len(a), len(b))
    return len(a & b) / n if n else np.nan


# ----------------------------------------------------------------------------- simulation core
def simulate(mems: dict, received: list[int], selectors: dict, L: int = 32, cam_state: dict | None = None):
    """Replay a received-frame sequence through MemoryBank; one row per non-prompt received frame.

    Policy columns hold the 5 policy-selected ORIGINAL frame ids (fixed prompt/recent excluded).
    sam2_noncond = memories official SAM 2 attends (last min(idx-1, 6) received frames).
    """
    bank = MemoryBank(mems[received[0]], L)
    rows = []
    for idx, f in enumerate(received[1:], start=1):
        cur = mems[f]
        pool = [c for c in bank.candidates if c is not bank.recent]
        row = {"video_idx": idx, "original_frame_id": f, "gap": f - received[idx - 1], "bank": len(bank.candidates),
               "pool": len(pool), "sam2_noncond": min(idx - 1, K - 1), "full_bank": len(bank.candidates) == L,
               "selection_possible": len(pool) > K - 2, "camera_state": cam_state.get(f) if cam_state else None}
        for name, sel in selectors.items():
            ch = sel.select(cur, bank).chosen
            row[name] = tuple(m.original_frame_id for m in ch)
            if ch:
                d = np.array([pose_difference(cur, m) for m in ch])
                row[f"{name}|age"] = float(np.mean([f - m.original_frame_id for m in ch]))
                row[f"{name}|dT"], row[f"{name}|dR"] = float(d[:, 0].mean()), float(d[:, 1].mean())
        rows.append(row)
        bank.add(cur)
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, a: str, b: str) -> tuple[float, float]:
    """(mean overlap, identical-set rate) of two policy columns."""
    if df.empty:
        return np.nan, np.nan
    sa, sb = df[a].map(set), df[b].map(set)
    ov = [overlap(x, y) for x, y in zip(sa, sb)]
    return float(np.nanmean(ov)), float(np.mean([x == y for x, y in zip(sa, sb)]))
