import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baseline import dropout_runner as dr
from baseline import frame_drops as fd
from baseline.dropout_analysis import aggregate, bootstrap_ci, failure_classes, failure_runs, run_summary, worst_windows
from baseline.dropout_eval import check_m2_reference, recovery_events, recovery_summary

ROOT = Path(__file__).resolve().parents[1]
TRACKS = {"Scene01": [90, 86, 63, 54], "Scene06": [12, 9, 11, 13]}
CONDS = ["random_30", "random_50", "burst_30", "burst_50"]
MODEL = {"sam2_upstream_commit": "c", "checkpoint_sha256": "k", "config": "cfg", "device": "mps",
         "precision": "float32 (no autocast)", "apply_postprocessing": True}
SHAPE = (6, 8)


# ----------------------------------------------------------------------------- matrix
def test_experiment_matrix_is_exactly_96_unique():
    specs = dr.experiment_matrix(TRACKS, CONDS, [0, 1, 2])
    assert len(specs) == 96 == len({s.name for s in specs})
    assert specs[0] == dr.RunSpec("Scene01", 90, "random_30", 0)
    assert pd.Series([s.condition for s in specs]).value_counts().to_dict() == {c: 24 for c in CONDS}


@pytest.mark.realdata
@pytest.mark.skipif(not (ROOT / "outputs/milestone3/manifests").is_dir(), reason="M3 manifests missing")
def test_every_spec_has_its_m3_manifest():
    man = ROOT / "outputs/milestone3/manifests"
    for s in dr.experiment_matrix(TRACKS, CONDS, [0, 1, 2]):
        p = fd.load(dr.manifest_path(man, s))
        assert (p.meta["scene"], p.meta["track_id"], p.meta["condition"], p.meta["seed"]) == \
            (s.scene, s.track_id, s.condition, s.seed)


# ----------------------------------------------------------------------------- fake runs
@pytest.fixture
def setup(tmp_path):
    man = tmp_path / "manifests"
    specs = [dr.RunSpec("Scene01", 7, "burst_50", 0), dr.RunSpec("Scene01", 7, "random_30", 1)]
    kinds = {"burst_50": ("burst", 0.5), "random_30": ("random", 0.3)}
    for s in specs:
        fd.save(fd.generate("Scene01", "clone", 0, 7, range(10, 31), s.condition, *kinds[s.condition], s.seed), man)
    return tmp_path, man, specs


def fake_run(out, man):
    """Writes a complete, valid run directory the way run_one does, without SAM 2."""
    def run(spec, ctx):
        p = fd.load(dr.manifest_path(man, spec))
        ids = list(p.received_ids)
        d = dr.run_dir(out, spec)
        d.mkdir(parents=True, exist_ok=True)
        dr.atomic_npz(d / "predictions.npz", masks=np.zeros((len(ids), *SHAPE), bool), original_frame_ids=np.array(ids))
        dr.atomic_csv(pd.DataFrame({"original_frame_id": ids}), d / "per_frame.csv")
        dr.atomic_csv(pd.DataFrame({"original_frame_id": ids[1:]}), d / "paired.csv")
        rec = {"schema": dr.RECORD_SCHEMA, "status": "complete", "model": MODEL,
               "manifest": {"frames_csv_sha256": p.csv_sha256},
               "predictions_sha256": dr.sha256_file(d / "predictions.npz"), "timing": {"frames": len(ids), "wall_s": 0.0}}
        dr.atomic_json(rec, d / "record.json")
        return rec
    return run


def test_resume_skips_valid_and_reruns_invalid(setup):
    out, man, specs = setup
    calls = []
    run = fake_run(out, man)

    def counting(spec, ctx):
        calls.append(spec.name)
        return run(spec, ctx)

    ctx = lambda s: {"manifests": man}  # noqa: E731
    st = dr.run_matrix(specs, ctx, out, MODEL, SHAPE, run=counting, log=lambda m: None)
    assert set(st.values()) == {"completed"} and len(calls) == 2
    st = dr.run_matrix(specs, ctx, out, MODEL, SHAPE, run=counting, log=lambda m: None)
    assert set(st.values()) == {"skipped_valid"} and len(calls) == 2          # resume: nothing re-run
    # Corrupt one prediction file -> detected, quarantined (not deleted), re-run.
    npz = dr.run_dir(out, specs[0]) / "predictions.npz"
    npz.write_bytes(npz.read_bytes()[:-10] + b"0123456789")
    ok, why = dr.validate_run(dr.run_dir(out, specs[0]), fd.load(dr.manifest_path(man, specs[0])), MODEL, SHAPE)
    assert not ok and "sha256" in why
    st = dr.run_matrix(specs, ctx, out, MODEL, SHAPE, run=counting, log=lambda m: None)
    assert st[specs[0].name] == "completed" and st[specs[1].name] == "skipped_valid" and len(calls) == 3
    q = list((out / "invalid").iterdir())
    assert len(q) == 1 and (q[0] / "INVALID_REASON.txt").read_text().startswith("predictions.npz corrupt")


def test_validate_run_detects_problems(setup):
    out, man, specs = setup
    fake_run(out, man)(specs[0], None)
    d, p = dr.run_dir(out, specs[0]), fd.load(dr.manifest_path(man, specs[0]))
    assert dr.validate_run(d, p, MODEL, SHAPE) == (True, "")
    assert not dr.validate_run(d, p, {**MODEL, "checkpoint_sha256": "other"}, SHAPE)[0]
    assert not dr.validate_run(d, p, MODEL, (7, 8))[0]
    other = fd.load(dr.manifest_path(man, specs[1]))
    assert not dr.validate_run(d, other, MODEL, SHAPE)[0]                     # different manifest
    (d / "paired.csv").unlink()
    assert dr.validate_run(d, p, MODEL, SHAPE)[1].startswith("missing")
    (d / "record.json").write_text("{not json")
    assert not dr.validate_run(d, p, MODEL, SHAPE)[0]
    (d / "record.json").unlink()
    assert dr.validate_run(d, p, MODEL, SHAPE) == (False, "no record.json")


def test_failure_is_logged_and_loop_continues(setup):
    out, man, specs = setup

    def boom(spec, ctx):
        if spec.condition == "burst_50":
            raise RuntimeError("simulated MPS failure")
        return fake_run(out, man)(spec, ctx)

    st = dr.run_matrix(specs, lambda s: {"manifests": man}, out, MODEL, SHAPE, run=boom, log=lambda m: None)
    assert st[specs[0].name].startswith("failed") and st[specs[1].name] == "completed"
    lines = (out / "failures.jsonl").read_text().splitlines()
    assert len(lines) == 1 and "simulated MPS failure" in json.loads(lines[0])["error"]
    assert not (dr.run_dir(out, specs[0]) / "record.json").exists()


def test_atomic_writes_leave_no_temp_files(tmp_path):
    dr.atomic_json({"a": np.int64(1)}, tmp_path / "x.json")
    dr.atomic_npz(tmp_path / "y.npz", a=np.ones(2))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "y.npz"]


def test_runner_never_reads_target_pose():
    src = inspect.getsource(dr)
    assert "pose.txt" not in src and "get_frame(clip.original_frame_ids[0]" in src  # prompt is the only GT


# ----------------------------------------------------------------------------- M2 reference validation
def fake_m2(tmp_path, sam2_cfg, ev_cfg, start=10, end=30, prompt=10):
    d = tmp_path / "m2"
    (d / "reports").mkdir(parents=True)
    (d / "reports" / "environment.json").write_text(json.dumps(
        {"checkpoint_sha256": sam2_cfg["checkpoint_sha256"], "sam2_upstream_commit": sam2_cfg["upstream_commit"],
         "config": sam2_cfg["config"], "apply_postprocessing": True}))
    ids = list(range(start, end + 1))
    pd.DataFrame({"video_idx": range(len(ids)), "original_frame_id": ids, "scene": "Scene01", "track_id": 7}).to_csv(
        d / "reports" / "video_index_mapping.csv", index=False)
    pd.DataFrame({"scene": "Scene01", "track_id": 7, "original_frame_id": ids, "is_prompt": [f == prompt for f in ids],
                  "gt_empty": False, "J": 1.0, "F": 1.0, "JF": 1.0, "F_strict": 1.0}).to_csv(
        d / "reports" / "per_frame_metrics.csv", index=False)
    return d


def test_m2_reference_validation(tmp_path):
    sc = {"checkpoint_sha256": "k", "upstream_commit": "c", "config": "cfg", "apply_postprocessing": True}
    ev = {"boundary_th": 0.008, "boundary_th_strict": 2, "lost_j": 0.1}
    p = fd.generate("Scene01", "clone", 0, 7, range(10, 31), "burst_30", "burst", 0.3, 0)
    ref = check_m2_reference(fake_m2(tmp_path, sc, ev), p, sc, ev, ev)
    assert ref.original_frame_id.tolist() == list(range(10, 31))
    with pytest.raises(ValueError, match="checkpoint"):
        check_m2_reference(tmp_path / "m2", p, {**sc, "checkpoint_sha256": "x"}, ev, ev)
    with pytest.raises(ValueError, match="evaluation"):
        check_m2_reference(tmp_path / "m2", p, sc, ev, {**ev, "lost_j": 0.2})
    p2 = fd.generate("Scene01", "clone", 0, 7, range(10, 30), "burst_30", "burst", 0.3, 0)
    with pytest.raises(ValueError, match="clip frames"):
        check_m2_reference(tmp_path / "m2", p2, sc, ev, ev)


# ----------------------------------------------------------------------------- recovery diagnostics
def test_visible_target_recovery_and_nodrop_reference():
    ids = list(range(12))
    frames = pd.DataFrame({"original_frame_id": ids, "is_dropped": [1 <= f <= 4 for f in ids]})
    # After the burst the target is fully occluded for 2 frames: empty prediction scores J=1 (DAVIS).
    j = {0: 1, 5: 1.0, 6: 1.0, 7: 0.3, 8: 0.8, 9: 0.9, 10: 0.9, 11: 0.9}
    vis = {f: f not in (5, 6) for f in j}
    ref = {f: 0.9 for f in ids}
    ev = recovery_events(frames, j, visible_by_frame=vis, ref_j_by_frame=ref)
    assert ev["class"].tolist() == ["intact_at_first_frame"]                 # primary J rule (unchanged)
    assert ev.visible_class.tolist() == ["recovered_later"] and ev.visible_onset_received_k.tolist() == [4]
    assert ev.first_frame_gt_visible.tolist() == [False]
    assert ev.nodrop_class.tolist() == ["intact_at_first_frame"] and ev.nodrop_first_J.tolist() == [0.9]


def test_recovery_denominators():
    rows = [  # degraded & long window recovered, degraded & long unrecovered, intact long, degraded short
        {"class": "recovered_later", "degraded_after_drop": True, "lost_after_drop": False,
         "within_n_determinable": True, "confirmed_within_n": True},
        {"class": "unrecovered_end", "degraded_after_drop": True, "lost_after_drop": True,
         "within_n_determinable": True, "confirmed_within_n": False},
        {"class": "intact_at_first_frame", "degraded_after_drop": False, "lost_after_drop": False,
         "within_n_determinable": True, "confirmed_within_n": True},
        {"class": "insufficient_observation", "degraded_after_drop": True, "lost_after_drop": False,
         "within_n_determinable": False, "confirmed_within_n": None},
    ]
    s = recovery_summary(pd.DataFrame(rows))
    assert s["degraded_runs"] == 3 and s["degraded_recovered"] == 1 and s["degraded_censored"] == 2
    assert s["degraded_within_5_denominator"] == 2 and s["degraded_recovered_within_5_rate"] == 0.5
    assert s["plan_literal_within_5_denominator"] == 3 and s["plan_literal_within_5_rate"] == pytest.approx(2 / 3)


# ----------------------------------------------------------------------------- aggregation
def paired_rows(scene, track, cond, seed, jd, jn, empty=None):
    n = len(jd)
    d = {"scene": scene, "track_id": track, "condition": cond, "seed": seed, "original_frame_id": range(n),
         "gt_empty": empty or [False] * n}
    for m in ["J", "F", "JF", "F_strict"]:
        d[f"{m}_drop"], d[f"{m}_no_drop_matched"] = jd, jn
    return pd.DataFrame(d)


def test_aggregation_weights_tracks_equally():
    p = pd.concat([
        paired_rows("S1", 1, "c", 0, [0.0] * 100, [1.0] * 100),   # long track, delta -1
        paired_rows("S1", 2, "c", 0, [1.0] * 2, [1.0] * 2),       # short track, delta 0
        paired_rows("S1", 1, "c", 1, [0.5] * 100, [1.0] * 100),   # seed 1 of track 1, delta -0.5
        paired_rows("S1", 2, "c", 1, [1.0] * 2, [1.0] * 2),
    ])
    runs = run_summary(p)
    agg = aggregate(runs)
    t = agg["track"].set_index("track_id")
    assert t.loc[1, "delta_J"] == pytest.approx(-0.75) and t.loc[2, "delta_J"] == 0
    assert agg["overall"].delta_J.iloc[0] == pytest.approx(-0.375)          # not frame-weighted (-0.74)
    assert agg["overall"].frames.iloc[0] == 204 and agg["scene"].tracks.iloc[0] == 2


def test_subsets_split_visible_and_empty():
    p = paired_rows("S", 1, "c", 0, [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [False, True, False])
    r = run_summary(p).set_index("subset")
    assert r.loc["all", "frames"] == 3 and r.loc["visible", "frames"] == 2 and r.loc["empty_gt", "frames"] == 1
    assert r.loc["visible", "delta_J"] == 0 and r.loc["empty_gt", "delta_J"] == -1


def test_failure_classes_and_runs():
    p = pd.DataFrame({"gt_empty": [False] * 6 + [True],
                      "J_drop": [0.9, 0.05, 0.9, 0.05, 0.05, 0.4, 0.0],
                      "J_no_drop_matched": [0.9, 0.9, 0.05, 0.06, 0.45, 0.45, 1.0]})
    c = failure_classes(p, 0.1)
    assert c.tolist() == ["ok", "new", "resolved", "persistent", "new", "ok"]
    c5 = failure_classes(p, 0.5)
    assert c5.tolist() == ["ok", "new", "resolved", "persistent", "severity_changed", "persistent"]
    assert failure_runs([0.9, 0.2, 0.3, 0.9, 0.1], 0.5) == [2, 1]


def test_worst_windows_deterministic():
    p = pd.concat([paired_rows("S", 1, "c", 0, [1, 1, 0, 0, 0, 1, 1], [1] * 7),
                   paired_rows("S", 2, "c", 0, [1, 1, 1, 1, 1, 1, 1], [1] * 7)])
    for m in ["J"]:
        p[f"delta_{m}"] = p[f"{m}_drop"] - p[f"{m}_no_drop_matched"]
    w = worst_windows(p, width=3, k=1)
    assert w.track_id.tolist() == [1] and (w.start_frame.iloc[0], w.end_frame.iloc[0]) == (2, 4)
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0)


# ----------------------------------------------------------------------------- earlier milestones preserved
@pytest.mark.realdata
@pytest.mark.skipif(not (ROOT / "outputs/milestone2/reports/per_track_metrics.csv").exists(), reason="M2 outputs missing")
def test_m2_reference_results_unchanged():
    t = pd.read_csv(ROOT / "outputs/milestone2/reports/per_track_metrics.csv")
    assert len(t) == 8 and t.eval_frames.sum() == 657
    assert t.JF.mean() == pytest.approx(0.9511, abs=1e-4) and t.J.mean() == pytest.approx(0.9070, abs=1e-4)
