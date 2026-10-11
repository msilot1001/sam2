import dataclasses

import numpy as np
import pytest
from conftest import rot_y

from memsel.selector import (BINS, POLICIES, Memory, MemoryBank, MemorySelector, PoseScale, age_bin,
                             estimate_pose_scale, pose_cost, pose_difference)

SCALE = PoseScale(s_t=10.0, s_r_deg=10.0, w_t=0.5)


def mem(fid, x=None, yaw=0.0):
    return Memory(fid, np.array([float(fid if x is None else x), 0.0, 0.0]), rot_y(yaw))


def stream(prompt, frames, L=32):
    bank = MemoryBank(mem(prompt), L)
    for f in frames:
        bank.add(mem(f))
    return bank


def picks(policy, current, bank, scale=SCALE):
    return [m.original_frame_id for m in MemorySelector(policy, scale).select(current, bank).chosen]


@pytest.mark.parametrize("policy", POLICIES)
def test_only_past_frames_and_limits(policy):
    bank = MemoryBank(mem(0), L=32)
    sel = MemorySelector(policy, SCALE)
    for t in range(1, 120):
        cur = mem(t)
        s = sel.select(cur, bank)
        ids = [m.original_frame_id for m in s.active]
        assert all(i < t for i in ids)
        assert len(ids) == len(set(ids)) <= 7 and len(s.chosen) <= 5
        assert ids[0] == 0                                   # prompt always kept
        assert bank.recent.original_frame_id in ids          # most recent always kept
        assert ids == sorted(ids)
        bank.add(cur)                                        # only after "prediction"
        assert len(bank.candidates) <= 32


def test_current_frame_cannot_be_candidate():
    bank = stream(0, range(1, 10))
    with pytest.raises(ValueError, match="strictly before"):
        MemorySelector("recent-K").select(mem(9), bank)
    with pytest.raises(ValueError, match="not newer"):
        bank.add(mem(9))
    with pytest.raises(TypeError):
        bank.add({"original_frame_id": 10, "mask": None})


def test_fifo_eviction_keeps_last_L():
    bank = stream(0, range(1, 41), L=32)
    assert [m.original_frame_id for m in bank.candidates] == list(range(9, 41))


def test_initial_frames_and_prompt_recent_dedup():
    bank = MemoryBank(mem(5))
    s = MemorySelector("pose+dt", SCALE).select(mem(6), bank)
    assert [m.original_frame_id for m in s.active] == [5] and s.chosen == ()  # prompt == recent counted once
    bank.add(mem(6))
    bank.add(mem(7))
    for p in POLICIES:
        s = MemorySelector(p, SCALE).select(mem(8), bank)
        assert [m.original_frame_id for m in s.active] == [5, 6, 7]  # no fake/duplicate fill


def test_recent_k_no_drop():
    assert picks("recent-K", mem(100), stream(0, range(60, 100))) == [94, 95, 96, 97, 98]


def test_time_distributed_bin_centres_and_tie():
    # current 100, recent 99. Bins by age: [1,2]->98, [3,5]->96 (centre 4), [6,10]->92 (centre 8),
    # [11,20]: ages 15/16 tie around 15.5 -> 85 (more recent), [21+]: ages 21..32 median 26.5 -> 26/27 tie -> 74.
    assert picks("time-distributed", mem(100), stream(0, range(68, 100))) == [74, 85, 92, 96, 98]


def test_time_distributed_empty_bins_filled_by_spread():
    # Ages 1(recent),2,25..30 -> bins [3,5],[6,10],[11,20] empty.
    bank = stream(0, [70, 71, 72, 73, 74, 75, 98, 99])
    # Bins: 98 (age 2), 21+ median 27.5 -> 72/73 tie -> 73. Fill vs {0,99,98,73}: 70 (gap 3),
    # then 75 (gap 2), then 74/72/71 all gap 1 -> most recent 74.
    assert picks("time-distributed", mem(100), bank) == [70, 73, 74, 75, 98]


def test_pose_only_prefers_pose_and_ties_to_recent():
    frames = list(range(80, 100))
    bank = MemoryBank(mem(0))
    for f in frames:  # all candidates at the same pose except 81, 85 which sit exactly at the current pose
        bank.add(mem(f, x=0.0 if f in (81, 85) else 5.0))
    cur = mem(100, x=0.0)
    assert picks("pose-only", cur, bank) == [81, 85, 96, 97, 98]
    # Identical poses everywhere -> pure tie -> identical to recent-K.
    flat = MemoryBank(mem(0))
    for f in frames:
        flat.add(mem(f, x=3.0))
    assert picks("pose-only", mem(100, x=3.0), flat) == picks("recent-K", mem(100), flat)


def test_pose_dt_picks_lowest_cost_within_each_bin():
    bank = MemoryBank(mem(0))
    for f in range(68, 100):
        bank.add(mem(f, x=0.0 if f in (77, 85, 93, 97) else 4.0))
    s = picks("pose+dt", mem(100, x=0.0), bank)
    # bin [1,2]: only 98; [3,5]: 97 is at pose; [6,10]: 93; [11,20]: 85; [21+]: 77.
    assert s == [77, 85, 93, 97, 98]
    assert s != picks("time-distributed", mem(100, x=0.0), bank)


def test_pose_dt_empty_bins_filled_by_cost():
    bank = MemoryBank(mem(0))
    for f in [70, 71, 72, 98, 99]:
        bank.add(mem(f, x=0.0 if f == 71 else 9.0))
    assert picks("pose+dt", mem(100, x=0.0), bank) == [70, 71, 72, 98]  # only 4 non-recent candidates


@pytest.mark.parametrize("policy", POLICIES)
def test_deterministic(policy):
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(60, 3))
    def run():
        bank = MemoryBank(Memory(0, pos[0], rot_y(0)))
        for f in range(1, 50):
            bank.add(Memory(f, pos[f], rot_y(f * 3)))
        return picks(policy, Memory(50, pos[50], rot_y(150)), bank)
    assert run() == run()


def test_pose_difference_and_cost():
    a = Memory(0, np.zeros(3), rot_y(0))
    b = Memory(1, np.array([3.0, 4.0, 0.0]), rot_y(30))
    dt, dr = pose_difference(a, b)
    assert dt == pytest.approx(5) and dr == pytest.approx(30)
    assert pose_cost(a, b, PoseScale(10, 60, 0.5)) == pytest.approx(0.5 * 0.5 + 0.5 * 0.5)
    assert pose_cost(a, b, PoseScale(1, 1, 0.25)) == pytest.approx(1.0)    # clipped
    assert pose_cost(a, b, PoseScale(10, 60, 1.0)) == pytest.approx(0.5)   # translation only
    assert PoseScale(0, 0).s_t > 0 and PoseScale(0, 0).s_r_deg > 0          # floors
    with pytest.raises(ValueError):
        PoseScale(1, 1, 1.5)


def test_estimate_pose_scale():
    seq = [Memory(f, np.array([float(f), 0, 0]), rot_y(0)) for f in range(10)]
    s = estimate_pose_scale([seq], L=2, q=100)
    assert s.s_t == pytest.approx(2.0) and s.s_r_deg == pytest.approx(0.01)  # rotation floor


def test_no_gt_mask_or_target_pose_can_enter():
    assert [f.name for f in dataclasses.fields(Memory)] == ["original_frame_id", "camera_position",
                                                            "camera_rotation"]
    m = mem(1)
    with pytest.raises((AttributeError, TypeError)):
        m.mask = np.ones((2, 2), bool)
    with pytest.raises(TypeError):
        Memory(1, np.zeros(3), np.eye(3), target_pose=np.zeros(3))
    import inspect
    import memsel.selector as module
    src = inspect.getsource(module)
    assert not any(s in src for s in ("pose.txt", "load_instance", "bbox", "mask_pixel"))


def test_bins_and_config_errors():
    assert [age_bin(a) for a in (1, 2, 3, 5, 6, 10, 11, 20, 21, 999)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert age_bin(0) is None and len(BINS) == 5
    with pytest.raises(ValueError):
        MemorySelector("pose-only")
    with pytest.raises(ValueError):
        MemorySelector("nope")
    with pytest.raises(ValueError):
        MemorySelector("time-distributed", K=6)
