import importlib.util
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from memint import planners

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
DAVIS = ROOT / "data/davis2017/DAVIS"


def fake_od(frames):
    return {"cond_frame_outputs": {0: {}}, "non_cond_frame_outputs": {t: {} for t in frames}}


def native_like(frame_idx, od):
    """The native plan shape for a fully populated past (prompt + last 6 + up to 15 pointers)."""
    sp = [(0, "cond", 0)] + [(7 - r, "non_cond", frame_idx - r) for r in range(6, 0, -1) if frame_idx - r in od["non_cond_frame_outputs"]]
    pt = [(frame_idx, "cond", 0)] + [(d, "non_cond", frame_idx - d) for d in range(1, 16) if frame_idx - d in od["non_cond_frame_outputs"]]
    return {"spatial": sp, "pointers": pt, "max_ptrs": 16}


# ----------------------------------------------------------------------------- planners (pure)
def test_spatial_and_pointer_suppression_are_independent():
    od = fake_od(range(1, 50))
    plan = native_like(50, od)
    a2 = planners.suppress({46, 47, 48}, set())(50, plan, od)
    assert {t for _, s, t in a2["spatial"] if s == "non_cond"} == {44, 45, 49}
    assert {46, 47, 48} <= {t for _, s, t in a2["pointers"]}                 # pointers untouched
    a3 = planners.suppress(set(), {46, 47, 48})(50, plan, od)
    assert {t for _, s, t in a3["spatial"]} == {t for _, s, t in plan["spatial"]}
    assert not {46, 47, 48} & {t for _, s, t in a3["pointers"]}
    assert (0, "cond", 0) in a2["spatial"] and (50, "cond", 0) in a3["pointers"]   # prompt preserved


def test_backfill_deterministic_bounded_budget_and_past_only():
    od = fake_od(range(1, 70))
    plan = native_like(68, od)
    S = {62, 63, 64, 65, 66, 67}
    p = planners.backfill(S, S, L=32)(68, plan, od)
    nc = [(tp, t) for tp, s, t in p["spatial"] if s == "non_cond"]
    assert [t for _, t in sorted(nc, key=lambda x: -x[1])] == [61, 60, 59, 58, 57, 56]   # most recent valid first
    assert all(tp == 1 for tp, _ in nc)                                     # true age > 6 -> farthest slot
    assert len(nc) == len([e for e in plan["spatial"] if e[1] == "non_cond"])         # budget kept
    assert p == planners.backfill(S, S, L=32)(68, plan, od)
    small = planners.backfill(S, S, L=8)(68, plan, od)
    assert all(68 - 8 <= t < 68 for _, s, t in small["spatial"] if s == "non_cond")    # pool bound
    assert len([1 for _, s, _ in small["spatial"] if s == "non_cond"]) == 2            # only 60, 61 eligible


@pytest.mark.parametrize("seed", range(10))
def test_no_planner_ever_reads_future_or_current_frames(seed):
    rng = np.random.default_rng(seed)
    frames = sorted(rng.choice(np.arange(1, 80), 40, replace=False).tolist())
    od = fake_od(frames)
    for t in range(10, 80):
        past = fake_od([f for f in frames if f < t])
        plan = native_like(t, past)
        pool = planners.candidate_pool(t, plan, past)
        ps = [planners.backfill(set(frames[::3]), set(), 32), planners.suppress(set(frames[::2]), set(frames[::2]))]
        slots = planners.slot_sources(plan)
        if slots and pool:
            ps += [planners.at_frame(t, planners.replace(slots[0][1], pool[0])),
                   planners.at_frame(t, planners.leave_one_out(slots[-1][1]))]
        for pl in ps:
            out = pl(t, plan, past)
            assert all(src < t for _, s, src in out["spatial"] + out["pointers"] if s == "non_cond")
            assert all(src in past["non_cond_frame_outputs"] for _, s, src in out["spatial"] if s == "non_cond")
    del od


def test_counterfactual_planners_frame_identity_and_positions():
    od = fake_od(range(1, 40))
    plan = native_like(30, od)
    slots = planners.slot_sources(plan)
    assert [t for _, t in slots] == [29, 28, 27, 26, 25, 24]
    pool = planners.candidate_pool(30, plan, od, 32)
    assert pool[0] == 23 and pool[-1] == 1 and all(t not in {24, 25, 26, 27, 28, 29} for t in pool)
    rep = planners.replace(24, 23)(30, plan, od)
    assert (1, "non_cond", 23) in rep["spatial"] and 24 not in [t for _, s, t in rep["spatial"] if s == "non_cond"]
    loo = planners.leave_one_out(29)(30, plan, od)
    assert len(loo["spatial"]) == len(plan["spatial"]) - 1
    pos = planners.reposition(29, 1)(30, plan, od)
    assert (1, "non_cond", 29) in pos["spatial"] and (6, "non_cond", 29) not in pos["spatial"]
    assert planners.at_frame(31, planners.leave_one_out(29))(30, plan, od) == plan       # only at target frame
    assert planners.deterministic_choice(pool, "k") == planners.deterministic_choice(list(reversed(pool)), "k")
    assert planners.tpos_for_age(1) == 6 and planners.tpos_for_age(6) == 1 and planners.tpos_for_age(40) == 1


def test_planners_and_engine_use_no_ground_truth():
    import memint.controlled as c
    import memint.engine as e
    import io
    import tokenize
    for mod in (planners, e, c):
        toks = tokenize.generate_tokens(io.StringIO(inspect.getsource(mod)).readline)
        code = " ".join(t.string for t in toks if t.type not in (tokenize.STRING, tokenize.COMMENT))  # code only
        assert not re.search(r"\bgt\b|ground.truth|annotation|get_frame|davis_gt", code.lower())


# ----------------------------------------------------------------------------- engine (real SAM 2)
@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    if not (CKPT.exists() and DAVIS.is_dir()):
        pytest.skip("checkpoint or DAVIS missing")
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor
    d = tmp_path_factory.mktemp("clip")
    for i, f in enumerate(sorted((DAVIS / "JPEGImages/480p/blackswan").glob("*.jpg"))[:20]):
        (d / f"{i:05d}.jpg").symlink_to(f.resolve())
    prompt = np.asarray(Image.open(DAVIS / "Annotations/480p/blackswan/00000.png")) == 1
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", str(CKPT), device=dev)
    return pred, d, prompt


@pytest.mark.realdata
def test_controlled_native_equals_official_and_suppression_semantics(setup):
    from memint import controlled, engine
    from rfeas.instrument import run_unit
    pred, d, prompt = setup
    official = run_unit(pred, d, prompt, instrument=False)
    assert next(iter(pred.parameters())).dtype.is_floating_point and str(next(iter(pred.parameters())).dtype) == "torch.float32"
    log = {}
    controlled.install(pred, None, log)
    state, od = engine.start(pred, d, prompt)
    m, _ = engine.rollout(pred, state, od, 1, 19)
    pred.reset_state(state)
    assert all(np.array_equal(m[i], official["masks"][i]) for i in range(1, 20))
    assert [t for _, s, t in log[15]["spatial"] if s == "non_cond"] == [9, 10, 11, 12, 13, 14]
    pop = run_unit(pred, d, prompt, gate=lambda s, h: s["video_idx"] in (5, 6, 7), instrument=False)
    pred._memint_planner = planners.suppress({5, 6, 7}, {5, 6, 7})
    log.clear()
    state, od = engine.start(pred, d, prompt)
    m1, _ = engine.rollout(pred, state, od, 1, 19)
    pred.reset_state(state)
    assert all(np.array_equal(m1[i], pop["masks"][i]) for i in range(1, 20))   # A1 == earlier pop gate
    assert not {5, 6, 7} & {t for _, s, t in log[10]["spatial"] + log[10]["pointers"] if s == "non_cond"}
    controlled.uninstall(pred)
    assert "_prepare_memory_conditioned_features" not in vars(pred)


@pytest.mark.realdata
def test_clone_replay_and_no_cross_contamination(setup):
    from memint import controlled, engine
    pred, d, prompt = setup
    controlled.install(pred, None, {})
    state, od = engine.start(pred, d, prompt)
    engine.rollout(pred, state, od, 1, 9)
    snapshot = {k: dict(v) for k, v in od.items()}
    a = engine.clone(od)
    ma, _ = engine.rollout(pred, state, a, 10, 14)
    b = engine.clone(od)                                       # same intervention twice -> identical
    pred._memint_planner = planners.at_frame(10, planners.leave_one_out(9))
    mb1, _ = engine.rollout(pred, state, b, 10, 14)
    c = engine.clone(od)
    mb2, _ = engine.rollout(pred, state, c, 10, 14)
    pred._memint_planner = None
    assert all(np.array_equal(mb1[t], mb2[t]) for t in mb1)
    assert {k: set(v) for k, v in od.items()} == {k: set(v) for k, v in snapshot.items()}   # original untouched
    assert all(od[k][t] is snapshot[k][t] for k in od for t in od[k])
    rest, _ = engine.rollout(pred, state, od, 10, 14)            # native continuation == clone branch 'a'
    assert all(np.array_equal(rest[t], ma[t]) for t in rest)
    pred.reset_state(state)
    controlled.uninstall(pred)


def test_paired_evaluation_identical_frames():
    from rfeas.pairing import paired
    a = pd.DataFrame({"video_idx": [0, 1, 2], "gt_empty": False, "J": [1, .9, .2], "F": 1.0, "JF": 1.0, "F_strict": 1.0})
    with pytest.raises(ValueError):
        paired(a, a.iloc[[0, 2, 1]].reset_index(drop=True).assign(video_idx=[0, 2, 1]))


def test_manifest_freeze_refuses_change(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("m5pre", ROOT / "scripts/m5pre.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "OUT", tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"rows": [{"unit": "x"}]}))
    from rfeas.units import Unit
    fake = {n: Unit("davis", n, 1, tuple(range(40))) for n in m.GROUP_A + [f"davis__x{i}__obj1" for i in range(10)]}
    monkeypatch.setattr(m, "units_index", lambda: (fake, {}))
    with pytest.raises(SystemExit):
        m.manifest()
    assert m.build_predictor.__code__.co_names.count("allow_tf32") >= 1


@pytest.mark.realdata
@pytest.mark.skipif(not (ROOT / "outputs/milestone4/reports/agg_all_overall.csv").exists(), reason="earlier outputs missing")
def test_earlier_results_preserved():
    o = pd.read_csv(ROOT / "outputs/milestone4/reports/agg_all_overall.csv").set_index("condition")
    assert o.loc["burst_30", "delta_JF"] == pytest.approx(-0.0021, abs=1e-4)
    r = json.loads((ROOT / "outputs/research_feasibility/reports/check1_report.json").read_text())
    assert r["units_total"] == 95
    m2 = pd.read_csv(ROOT / "outputs/milestone2/reports/per_track_metrics.csv")
    assert m2.JF.mean() == pytest.approx(0.9511, abs=1e-4)
