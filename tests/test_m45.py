import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import rot_y

from baseline import frame_drops as fd
from memsel import synthetic as syn
from memsel.opportunity import compare, overlap, simulate
from memsel.selector import DIAGNOSTIC_POLICIES, POLICIES, Memory, MemoryBank, MemorySelector, PoseScale, pose_cost
from vkitti.calibration import rigid_problems

ROOT = Path(__file__).resolve().parents[1]
SCALE = PoseScale(10.0, 10.0, 0.5)


def mem(fid, x=None, yaw=0.0):
    return Memory(fid, np.array([float(fid if x is None else x), 0.0, 0.0]), rot_y(yaw))


# ----------------------------------------------------------------------------- selector additions
def test_primary_policies_unchanged_and_diagnostic_separate():
    assert POLICIES == ("recent-K", "time-distributed", "pose-only", "pose+dt")
    assert DIAGNOSTIC_POLICIES == ("recent-per-bin",)


def test_recent_per_bin_newest_in_each_bin_and_fill():
    bank = MemoryBank(mem(0))
    for f in range(68, 100):
        bank.add(mem(f))
    pick = [m.original_frame_id for m in MemorySelector("recent-per-bin").select(mem(100), bank).chosen]
    assert pick == [79, 89, 94, 97, 98]          # newest of ages [2], [3-5], [6-10], [11-20], [21+]
    gappy = MemoryBank(mem(0))
    for f in [70, 71, 72, 98, 99]:
        gappy.add(mem(f))
    pick = [m.original_frame_id for m in MemorySelector("recent-per-bin").select(mem(100), gappy).chosen]
    assert pick == [70, 71, 72, 98]              # bins [2] and [21+] + newest-first fill
    a = [m.original_frame_id for m in MemorySelector("recent-per-bin").select(mem(100), bank).chosen]
    assert a == [79, 89, 94, 97, 98]             # deterministic


def test_tolerance_default_is_original_behaviour_and_ties_small_noise():
    a, b = mem(0, x=0.0), mem(1, x=0.004)
    assert pose_cost(a, b, SCALE) > 0
    assert pose_cost(a, b, PoseScale(10, 10, 0.5, tol_t_m=0.005, tol_r_deg=0.01)) == 0.0
    bank = MemoryBank(mem(0))
    for f in range(1, 40):
        bank.add(Memory(f, np.array([0.001 * ((f * 7) % 5), 0.0, 0.0]), rot_y(0)))  # sub-cm jitter
    cur = Memory(40, np.zeros(3), rot_y(0))
    raw = MemorySelector("pose-only", SCALE).select(cur, bank).chosen
    tol = MemorySelector("pose-only", PoseScale(10, 10, 0.5, 0.01, 0.05)).select(cur, bank).chosen
    rk = MemorySelector("recent-K").select(cur, bank).chosen
    assert [m.original_frame_id for m in tol] == [m.original_frame_id for m in rk]
    assert [m.original_frame_id for m in raw] != [m.original_frame_id for m in rk]


# ----------------------------------------------------------------------------- opportunity simulation
def test_simulate_respects_budget_past_only_and_excludes_fixed():
    mems = {f: mem(f) for f in range(0, 120)}
    received = [0] + [f for f in range(1, 120) if f % 3 != 0]
    sels = {p: MemorySelector(p, SCALE) for p in POLICIES + DIAGNOSTIC_POLICIES}
    df = simulate(mems, received, sels, L=16)
    assert df.bank.max() <= 16 and df.pool.max() <= 15
    for r in df.itertuples():
        prev = received[r.video_idx - 1]
        for p in sels:
            ids = df.loc[r.Index, p]
            assert len(ids) <= 5 and all(i < r.original_frame_id for i in ids)   # past only, K-2 slots
            assert prev not in ids and 0 not in ids                               # fixed memories excluded
            assert all(i in received for i in ids)                                # dropped frames never used
    assert df.sam2_noncond.max() == 6 and (df.sam2_noncond == np.minimum(df.video_idx - 1, 6)).all()
    assert (df.selection_possible == (df.pool > 5)).all()


def test_compare_and_overlap():
    df = pd.DataFrame({"a": [(1, 2, 3), (4, 5)], "b": [(1, 2, 9), (4, 5)]})
    ov, ident = compare(df, "a", "b")
    assert ov == pytest.approx((2 / 3 + 1) / 2) and ident == 0.5
    assert np.isnan(overlap(set(), set()))


# ----------------------------------------------------------------------------- synthetic geometry
@pytest.mark.parametrize("poses", [syn.straight(50), syn.turn(0.75, 15.0), syn.u_turn(0.45, 6.0), syn.loop(0.45, 5.1, 1.2)])
def test_synthetic_poses_are_valid_rigid_and_continuous(poses):
    ms = syn.to_memories(poses)
    for m in ms:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = m.camera_rotation.T, -m.camera_rotation.T @ m.camera_position
        assert not rigid_problems(T)
        assert m.camera_position[1] == pytest.approx(-syn.CAMERA_HEIGHT_M)        # y down: camera 1.65 m up
    steps = [np.linalg.norm(b.camera_position - a.camera_position) for a, b in zip(ms, ms[1:])]
    assert max(steps) < 1.01


def test_heading_convention_matches_camera_forward():
    ms = syn.to_memories(syn.turn(1.0, 10.0, before=5, after=5))
    a, b = ms[-2], ms[-1]
    fwd = b.camera_rotation @ np.array([0, 0, 1.0])           # camera z axis in world
    step = (b.camera_position - a.camera_position) / np.linalg.norm(b.camera_position - a.camera_position)
    assert np.allclose(fwd, step, atol=1e-9)


def test_loop_revisit_lag_and_plausibility():
    pmin, v = syn.min_loop_period_frames()
    assert pmin == pytest.approx(2 * math.pi * 5 * 10 / math.sqrt(20)) and pmin > 64
    assert syn.plausibility(0.45, 5.1)["plausible"] and not syn.plausibility(1.0, 5.0)["plausible"]
    ms = syn.to_memories(syn.loop(0.45, 5.1, 2.0, lead=0))
    period = round(2 * math.pi * 5.1 / 0.45)
    i = period + 5
    dt = np.linalg.norm(ms[i].camera_position - ms[i - period].camera_position)
    assert dt < 0.5                                                             # back at the same place


# ----------------------------------------------------------------------------- stress manifests
COND = {"stress_b8_30": (0.30, 8), "stress_b8_50": (0.50, 8), "stress_b16_30": (0.30, 16), "stress_b16_50": (0.50, 16)}


@pytest.mark.parametrize("cond", COND)
def test_stress_generation_deterministic_and_bursts_exact(cond):
    rate, L = COND[cond]
    p = fd.generate("Scene01", "clone", 0, 86, range(339, 426), cond, "burst", rate, 0, L, 1, "m45")
    q = fd.generate("Scene01", "clone", 0, 86, range(339, 426), cond, "burst", rate, 0, L, 1, "m45")
    assert p.meta == q.meta and p.frames.equals(q.frames) and p.meta["namespace"] == "m45"
    assert all(b["length"] == L == b["end"] - b["start"] + 1 for b in p.meta["bursts"])
    assert bool(p.frames.is_received.iloc[0])
    assert p.meta["rng_key"].startswith("m45|")
    other = fd.generate("Scene01", "clone", 0, 86, range(339, 426), cond, "burst", rate, 0, L, 1, "m3")
    assert "namespace" not in other.meta                     # M3 manifests unchanged in format


@pytest.mark.realdata
@pytest.mark.skipif(not (ROOT / "outputs/milestone45/stress_manifests").is_dir(), reason="stress manifests missing")
def test_saved_stress_manifests_reproduce_and_m3_unchanged():
    js = sorted((ROOT / "outputs/milestone45/stress_manifests").glob("*/*.json"))
    assert len(js) == 96
    for j in js:
        p = fd.load(j)
        assert p.meta["namespace"] == "m45" and p.meta["condition"] in COND
    for j in sorted((ROOT / "outputs/milestone3/manifests").glob("*/*.json")):
        assert "namespace" not in json.loads(j.read_text())
        fd.load(j)                                           # sha256 + regeneration still exact


@pytest.mark.realdata
@pytest.mark.skipif(not (ROOT / "outputs/milestone4/reports/run_summary.csv").exists(), reason="M4 outputs missing")
def test_m4_results_preserved():
    r = pd.read_csv(ROOT / "outputs/milestone4/reports/run_summary.csv")
    assert len(r[r.subset == "all"]) == 96
    o = pd.read_csv(ROOT / "outputs/milestone4/reports/agg_all_overall.csv").set_index("condition")
    assert o.loc["burst_30", "delta_JF"] == pytest.approx(-0.0021, abs=1e-4)
    assert o.loc["random_30", "delta_JF"] == pytest.approx(0.0005, abs=1e-4)
