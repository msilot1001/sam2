import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rfeas import gates
from rfeas.pairing import longest_streak, paired
from rfeas.risk import (FEATURES, Logistic, auroc, average_precision, brier, calibration_bins, causal_features,
                        cluster_bootstrap, failure_episodes, future_labels)
from rfeas.units import Unit, check2_select, split_of, stable_hash

ROOT = Path(__file__).resolve().parents[1]
DAVIS = ROOT / "data/davis2017/DAVIS"
CKPT = ROOT / "checkpoints/sam2.1_hiera_tiny.pt"


def sig_table(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"video_idx": range(n), "obj_score": rng.normal(5, 2, n), "pred_iou": rng.uniform(.5, 1, n),
                         "pred_iou_spread": rng.uniform(0, .1, n), "area": rng.integers(100, 1000, n),
                         "mean_prob_in_mask": rng.uniform(.8, 1, n),
                         "consistency_iou_prev": np.r_[np.nan, rng.uniform(.5, 1, n - 1)],
                         "area_ratio_prev": np.r_[np.nan, rng.uniform(.8, 1.2, n - 1)]})


# ----------------------------------------------------------------------------- leakage / features
def test_causal_features_ignore_future_rows():
    s = sig_table(10)
    f = causal_features(s)
    s2 = s.copy()
    s2.loc[6:, ["obj_score", "pred_iou", "area"]] = [-99, 0.0, 1]
    f2 = causal_features(s2)
    pd.testing.assert_frame_equal(f.iloc[:6], f2.iloc[:6])
    assert set(f.columns[1:]) == set(FEATURES)
    assert np.isnan(f.loc[1, "d_pred_iou"])          # prompt row is not a tracking output


def test_online_gate_features_equal_offline():
    s = sig_table(12)
    model = {"model": Logistic().fit(np.random.default_rng(1).normal(size=(50, len(FEATURES))),
                                     np.r_[np.zeros(40), np.ones(10)]), "threshold": 0.3}
    offline = gates.risk_scores(model, s)
    rows = s.to_dict("records")
    online = [gates.risk_scores(model, pd.DataFrame(rows[: t + 1]))[-1] for t in range(len(rows))]
    assert np.allclose(offline, online)


# ----------------------------------------------------------------------------- labels
def ev_table(J, empty=None):
    return pd.DataFrame({"video_idx": range(len(J)), "J": J, "gt_empty": empty or [False] * len(J)})


def test_future_labels_window_eligibility_and_censoring():
    J = [1, .9, .9, .9, .3, .9, .9, .9, .9, .9]
    lab = future_labels(ev_table(J), H=2, j_thr=0.5)
    assert lab.eligible.tolist() == [False, True, True, True, True, True, True, True, False, False]
    assert lab.label.tolist()[1:8] == [0, 1, 1, 0, 0, 0, 0]    # failure at 4 seen from t=2,3 only
    assert lab.current_fail.tolist()[4] and not lab.current_accurate.tolist()[4]
    # windows with only empty GT are not eligible; empty-GT frames never count as failures
    lab = future_labels(ev_table([1, .9, 0, 0, .9], [False, False, True, True, False]), H=2, j_thr=0.5)
    assert lab.eligible.tolist() == [False, False, True, False, False] and lab.label.sum() == 0


def test_failure_episodes():
    assert failure_episodes(ev_table([0, .9, .2, .3, .9, .1]), 0.5) == [(2, 3), (5, 5)]


# ----------------------------------------------------------------------------- metrics
def test_auroc_ap_brier_calibration():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([.1, .4, .35, .8, .2, .9])
    pairs = [(a, b) for a in np.where(y == 1)[0] for b in np.where(y == 0)[0]]
    brute = np.mean([1.0 if s[a] > s[b] else 0.5 if s[a] == s[b] else 0.0 for a, b in pairs])
    assert auroc(y, s) == pytest.approx(brute)
    assert auroc([1, 0], [.5, .5]) == 0.5
    assert average_precision([1, 0, 1], [.9, .8, .1]) == pytest.approx((1 + 2 / 3) / 2)
    assert brier([1, 0], [1, 0]) == 0 and brier([1, 0], [0.5, 0.5]) == 0.25
    cb = calibration_bins([0, 1, 1, 0], [.05, .95, .91, .02], 10)
    assert cb.n.sum() == 4 and cb.iloc[-1].obs_rate == 1.0


def test_logistic_learns_separable_signal_and_roundtrips():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] > 0.5).astype(float)
    m = Logistic(1.0).fit(X, y)
    assert auroc(y, m.predict(X)) > 0.95
    assert np.allclose(Logistic.from_dict(m.to_dict()).predict(X), m.predict(X))


def test_cluster_bootstrap_resamples_groups():
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, "v": [1.0] * 5 + [3.0] * 5})
    lo, hi = cluster_bootstrap(df, "g", lambda d: d.v.mean(), n=500)
    assert 1.0 <= lo <= hi <= 3.0 and lo < 2 < hi


# ----------------------------------------------------------------------------- splits / selection
def test_video_level_split_and_deterministic_selection():
    us = [Unit("davis", f"davis/v{i}", o, ("00000",)) for i in range(40) for o in (1, 2)]
    for u in us:
        assert split_of(u.video) == split_of(u.video)
    assert all(split_of(a.video) == split_of(b.video) for a, b in zip(us[::2], us[1::2]))  # objects share split
    sel = check2_select(us, 12)
    assert sel == check2_select(list(reversed(us)), 12) and len(sel) == 12
    assert all(split_of(u.video) == "val" for u in sel)
    assert stable_hash("x") == stable_hash("x")


# ----------------------------------------------------------------------------- gates
def test_gate_thresholds():
    s = {"obj_score": -0.1, "pred_iou": 0.69, "consistency_iou_prev": 0.49, "video_idx": 3}
    assert gates.presence_gate(s, []) and gates.iou_gate(s, []) and gates.consistency_gate(s, [])
    s = {"obj_score": 0.0, "pred_iou": 0.7, "consistency_iou_prev": np.nan, "video_idx": 3}
    assert not gates.presence_gate(s, []) and not gates.iou_gate(s, []) and not gates.consistency_gate(s, [])
    g = gates.random_gate(0.3, "k")
    frac = np.mean([g({"video_idx": i}, []) for i in range(2000)])
    assert 0.25 < frac < 0.35 and g({"video_idx": 5}, []) == g({"video_idx": 5}, [])
    assert gates.skip_exactly(4)({"video_idx": 4}, []) and not gates.skip_exactly(4)({"video_idx": 5}, [])


def test_paired_and_streak():
    a = pd.DataFrame({"video_idx": [0, 1, 2], "gt_empty": False, "J": [1, .9, .2], "F": 1.0, "JF": 1.0, "F_strict": 1.0})
    b = a.assign(J=[1, .8, .6])
    p = paired(a, b)
    assert p.video_idx.tolist() == [1, 2] and np.allclose(p.delta_J, [-.1, .4])
    with pytest.raises(ValueError):
        paired(a, b.iloc[:2])
    assert longest_streak([.9, .1, .2, .9, .3]) == 2


def test_run_unit_receives_no_ground_truth_beyond_prompt():
    from rfeas.instrument import run_unit
    assert list(inspect.signature(run_unit).parameters) == ["predictor", "video_dir", "prompt_mask", "gate", "instrument"]
    import re
    src = inspect.getsource(run_unit).split('"""')[2]          # code only, docstring excluded
    assert not re.search(r"\bgt\b|ground|annotation|get_frame", src.lower())


# ----------------------------------------------------------------------------- real SAM 2 (CPU)
@pytest.mark.realdata
@pytest.mark.skipif(not (CKPT.exists() and DAVIS.is_dir()), reason="checkpoint or DAVIS missing")
def test_instrumentation_equivalence_skip_semantics_and_restoration(tmp_path):
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from PIL import Image

    from rfeas.instrument import run_unit
    src = DAVIS / "JPEGImages/480p/blackswan"
    for i, f in enumerate(sorted(src.glob("*.jpg"))[:18]):           # 18 > 16: no obj-ptr length effect
        (tmp_path / f"{i:05d}.jpg").symlink_to(f.resolve())
    prompt = np.asarray(Image.open(DAVIS / "Annotations/480p/blackswan/00000.png")) == 1
    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", str(CKPT), device=torch.device("cpu"))
    plain = run_unit(pred, tmp_path, prompt, instrument=False)
    inst = run_unit(pred, tmp_path, prompt, instrument=True)
    off = run_unit(pred, tmp_path, prompt, gate=lambda s, h: False)
    assert "track_step" not in vars(pred) and "_forward_sam_heads" not in vars(pred)   # hook removed
    for i in range(18):
        assert np.array_equal(plain["masks"][i], inst["masks"][i]) and np.array_equal(plain["masks"][i], off["masks"][i])
    assert all(np.isfinite(inst["signals"][i]["pred_iou"]) for i in range(1, 18))
    assert np.isnan(plain["signals"][3]["pred_iou"])                    # uninstrumented: IoU unavailable
    skip = run_unit(pred, tmp_path, prompt, gate=gates.skip_exactly(5))
    assert skip["skipped"] == [5]
    assert all(np.array_equal(skip["masks"][i], plain["masks"][i]) for i in range(6))   # up to and incl. t
    again = run_unit(pred, tmp_path, prompt, instrument=False)                         # state fully restored
    assert all(np.array_equal(again["masks"][i], plain["masks"][i]) for i in range(18))
