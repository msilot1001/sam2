import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("m61", ROOT / "scripts/m61.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_untrained_head_v2_is_bilinear_identity():
    m = load()
    h = m.HeadV2()
    lg = torch.randn(1, 1, 256, 256)
    out = h(lg, torch.randn(1, 32, 256, 256), torch.randn(1, 64, 128, 128), torch.rand(1, 3, 1024, 1024))
    assert torch.allclose(out, Fn.interpolate(lg, size=(1024, 1024), mode="bilinear", align_corners=False))


def test_logistic_recovers_separable_rule():
    m = load()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] - X[:, 1] > 0).astype(float)
    mdl = m.fit_logistic(X, y)
    assert ((m.predict_logistic(mdl, X) > 0.5) == (y > 0.5)).mean() > 0.95


def test_boundary_loss_zero_grad_free_and_finite():
    m = load()
    y = torch.zeros(1, 1, 1024, 1024)
    y[..., 100:200, 100:200] = 1
    good, bad = m._loss(y * 20 - 10, y), m._loss(-(y * 20 - 10), y)
    assert torch.isfinite(good) and good < bad


import pytest


@pytest.mark.realdata
def test_dual_disabled_reproduces_logged_A2x_and_is_causal():
    import pandas as pd
    if not torch.cuda.is_available() or not (ROOT / "outputs/milestone6/runs/A2x").exists():
        pytest.skip("no CUDA / M6 runs")
    from memint import controlled
    m = load()
    pred = m.m55.build_predictor()
    controlled.install(pred, None, {})
    u = m.m6._UNITS()["davis__davis/lab-coat__obj1"]
    ref = pd.read_csv(ROOT / "outputs/milestone6/runs/A2x/davis__davis_lab-coat__obj1.csv")
    a, ma = m.run_dual(pred, "DAVIS_val", u, "A2d", n=7)
    assert a.JF.round(10).tolist() == ref.JF.round(10).tolist()[:6]               # live dual == logged A2x
    b, mb = m.run_dual(pred, "DAVIS_val", u, "A3", n=5)
    c, mc = m.run_dual(pred, "DAVIS_val", u, "A3", n=8)
    assert all(np.array_equal(x, y) for x, y in zip(mb, mc[:4]))                 # no future dependence
