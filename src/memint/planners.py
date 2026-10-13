"""Memory-retrieval planners (pure functions of the plan + past frame ids; no GT, no future frames).

Every planner returns a NEW plan; the conditioning prompt entries (t_pos 0 / store "cond") are kept.
Temporal positions of substituted spatial memories follow the source frame's true age:
t_pos = num_maskmem - age, clamped to 1 (SAM 2 has no embedding for ages > num_maskmem - 1), so an
older substitute is encoded as "the farthest slot". Several memories may share t_pos 1 (as all
conditioning frames share t_pos 0 natively).
"""
import hashlib

NUM_MASKMEM = 7


def tpos_for_age(age: int) -> int:
    return max(1, NUM_MASKMEM - age)


def _nonprompt(entries):
    return [e for e in entries if e[1] == "non_cond"]


def suppress(spatial_frames=frozenset(), pointer_frames=frozenset()):
    """A1/A2/A3: hide these frames' spatial memories and/or object pointers from all later frames."""
    sf, pf = frozenset(spatial_frames), frozenset(pointer_frames)

    def planner(frame_idx, plan, output_dict):
        return {**plan,
                "spatial": [e for e in plan["spatial"] if not (e[1] == "non_cond" and e[2] in sf)],
                "pointers": [e for e in plan["pointers"] if not (e[1] == "non_cond" and e[2] in pf)]}
    return planner


def backfill(spatial_frames=frozenset(), pointer_frames=frozenset(), L: int = 32):
    """A4/A5: suppress like `suppress`, then refill each EMPTY spatial slot (relative to the native
    number of non-conditioning slots) with the most recent earlier, non-suppressed, unused encoded
    memory within the last L frames. Pointers follow `pointer_frames` (no pointer backfill)."""
    sf = frozenset(spatial_frames)
    base = suppress(spatial_frames, pointer_frames)

    def planner(frame_idx, plan, output_dict):
        native_n = len(_nonprompt(plan["spatial"]))
        p = base(frame_idx, plan, output_dict)
        used = {e[2] for e in p["spatial"]}
        need = native_n - len(_nonprompt(p["spatial"]))
        pool = sorted((t for t in output_dict["non_cond_frame_outputs"]
                       if frame_idx - L <= t < frame_idx and t not in sf and t not in used), reverse=True)
        add = [(tpos_for_age(frame_idx - t), "non_cond", t) for t in pool[:max(need, 0)]]
        return {**p, "spatial": p["spatial"] + add}
    return planner


def at_frame(target: int, inner):
    """Apply `inner` only when predicting frame `target` (single-step counterfactuals)."""
    def planner(frame_idx, plan, output_dict):
        return inner(frame_idx, plan, output_dict) if frame_idx == target else plan
    return planner


# ----------------------------------------------------------------------------- Part B (one frame)
def slot_sources(plan):
    """Native non-conditioning spatial memories, most recent first: [(t_pos, frame_idx)]."""
    return sorted([(p, t) for p, s, t in plan["spatial"] if s == "non_cond"], key=lambda x: -x[1])


def candidate_pool(frame_idx, plan, output_dict, L: int = 32):
    used = {t for _, s, t in plan["spatial"] if s == "non_cond"}
    return sorted((t for t in output_dict["non_cond_frame_outputs"] if frame_idx - L <= t < frame_idx and t not in used),
                  reverse=True)


def leave_one_out(slot_frame: int):
    def inner(frame_idx, plan, output_dict):
        return {**plan, "spatial": [e for e in plan["spatial"] if not (e[1] == "non_cond" and e[2] == slot_frame)]}
    return inner


def replace(slot_frame: int, new_frame: int):
    def inner(frame_idx, plan, output_dict):
        sp = [e for e in plan["spatial"] if not (e[1] == "non_cond" and e[2] == slot_frame)]
        return {**plan, "spatial": sp + [(tpos_for_age(frame_idx - new_frame), "non_cond", new_frame)]}
    return inner


def reposition(slot_frame: int, t_pos: int = 1):
    """Positional-encoding control: same source memory, temporal position forced to t_pos."""
    def inner(frame_idx, plan, output_dict):
        return {**plan, "spatial": [(t_pos, s, t) if (s == "non_cond" and t == slot_frame) else (p, s, t)
                                    for p, s, t in plan["spatial"]]}
    return inner


def deterministic_choice(items, key: str):
    items = list(items)
    if not items:
        return None
    return min(items, key=lambda x: hashlib.sha256(f"{key}|{x}".encode()).digest())
