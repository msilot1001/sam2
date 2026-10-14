import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from spatial import sggl

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("m7", ROOT / "scripts/m7.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_global_feature_resampling_is_aligned_with_crop_grid():
    H, W, g = 480, 854, 64
    yy, xx = torch.meshgrid((torch.arange(g) + 0.5) / g * H, (torch.arange(g) + 0.5) / g * W, indexing="ij")
    feat = torch.stack([yy, xx])[None]                       # cell centres in frame pixels (G squashes to 64x64)
    box = (100, 292, 300, 492)
    s = sggl.sample_global_in_crop(feat, box, (H, W), 32)
    c = (torch.arange(32) + 0.5) / 32 * 192
    ey, ex = torch.meshgrid(100 + c, 300 + c, indexing="ij")
    assert torch.allclose(s[0, 0], ey, atol=1e-3) and torch.allclose(s[0, 1], ex, atol=1e-3)


def test_crop_logits_paste_into_frame_with_unobserved_nan():
    lg = torch.full((1, 1, 256, 256), -5.0)
    lg[..., :128, :] = 5.0
    f = sggl.crop_to_frame(lg, (10, 110, 20, 120), (200, 300))
    assert np.isnan(f[0, 0]) and np.isnan(f[150, 150])
    assert f[20, 50] > 0 and f[100, 50] < 0 and not np.isnan(f[10:110, 20:120]).any()


def test_down_up_is_identity_without_degradation_and_blurs_otherwise():
    m = load()
    x = torch.randn(1, 3, 1024, 1024)
    assert torch.allclose(m._down_up(x, 1.0, 1.0), x, atol=1e-5)
    y = m._down_up(x, 2.25, 4.0)
    assert y.shape == x.shape and y.std() < 0.5 * x.std()


@pytest.mark.realdata
def test_recording_rollout_reproduces_sggl_probes_are_isolated_and_causal():
    import pandas as pd
    if not torch.cuda.is_available() or not (ROOT / "outputs/milestone6/runs/A2x").exists():
        pytest.skip("no CUDA / M6 runs")
    from memint import controlled
    m = load()
    pred = m.m55.build_predictor()
    controlled.install(pred, None, {})
    u = m.m6._UNITS()["davis__davis/lab-coat__obj1"]
    ref = pd.read_csv(ROOT / "outputs/milestone6/runs/A2x/davis__davis_lab-coat__obj1.csv")
    a, A = m.record_unit(pred, "DEV", u, n=9, probes=False)
    b, B = m.record_unit(pred, "DEV", u, n=9, probes=True)
    assert a.JF.round(10).tolist() == ref.JF.round(10).tolist()[:8]                 # == logged SGGL (A2x)
    for k in ("G", "L", "L2", "simL", "simG", "ptrL", "ptrG", "box"):
        assert np.array_equal(A[k], B[k]), k                                         # probes leave production untouched
    assert a.JF.tolist() == b.JF.tolist() and a.J_L2.tolist() == b.J_L2.tolist()
    c, C = m.record_unit(pred, "DEV", u, n=5, probes=True)
    assert np.array_equal(C["L"], B["L"][:4]) and c.JF.tolist() == b.JF.tolist()[:4]   # no future dependence
    assert (b.IoU_LLp >= 0).all() and b.score_Lb.notna().any()


def _tx(B=2):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    return {"memL": r(B, 256, 64, 64), "bbL": r(B, 256, 64, 64), "memG": r(B, 256, 32, 32), "bbG": r(B, 256, 32, 32),
            "lL": r(B, 1, 256, 256) * 5, "lG": r(B, 1, 256, 256) * 5, "sL": r(B), "sG": r(B), "tok": r(B, 4, 256)}


def test_fusion_heads_start_as_identity_and_get_gradients():
    from spatial.feature_fusion import VARIANTS, FusionHead, loss_fn
    x = _tx()
    y = (torch.rand(2, 1, 256, 256) > 0.5).float()
    for v in VARIANTS:
        h = FusionHead(v)
        out = h(x)
        assert torch.equal(out, x["lL"]), v                                    # untrained head == local logits
        loss_fn(out, y).backward()
        assert h.out.weight.grad is not None and h.out.weight.grad.abs().sum() > 0, v
        assert torch.isfinite(loss_fn(out, y))


def test_no_transfer_head_ignores_global_inputs_and_gate_bounds():
    from spatial.feature_fusion import FusionHead
    x = _tx()
    h0 = FusionHead("T0")
    torch.nn.init.normal_(h0.out.weight)
    a = h0(x)
    x2 = dict(x, memG=torch.zeros_like(x["memG"]), lG=-x["lG"], bbG=torch.zeros_like(x["bbG"]))
    assert torch.equal(a, h0(x2))                                               # T0 never reads the global view
    h4 = FusionHead("T4")
    h4(x)
    assert h4.last_alpha.min() >= 0 and h4.last_alpha.max() <= 1


def test_fusion_policies_respect_bounds():
    from spatial import mask_fusion
    rng = np.random.default_rng(0)
    lG = rng.normal(size=(40, 50)).astype(np.float32)
    lL = np.full((40, 50), np.nan, np.float32)
    lL[5:30, 10:35] = rng.normal(size=(25, 25))
    b = mask_fusion.blend(lL, lG)
    assert not b[np.isnan(lL)].any()                                           # unobserved != background evidence
    base = lL > 0
    base = np.nan_to_num(base).astype(bool)
    m, a, d = mask_fusion.asymmetric_edit(base, lL, lG, g_observable=False)
    assert d == 0 and (m | base).sum() == m.sum()                              # G unobservable -> no deletions
    u = mask_fusion.instance_union(base, lG > 0)
    assert (u & base).sum() == base.sum() and (u <= (base | (lG > 0))).all()


@pytest.mark.realdata
def test_causal_policy_rollout_reproduces_sggl_and_sgglv():
    import pandas as pd
    if not torch.cuda.is_available() or not (ROOT / "outputs/milestone6/runs/A2x").exists():
        pytest.skip("no CUDA / M6 runs")
    from memint import controlled
    from spatial import policies
    m = load()
    pred = m.m55.build_predictor()
    controlled.install(pred, None, {})
    u = m.m6._UNITS()["davis__davis/lab-coat__obj1"]
    ref = pd.read_csv(ROOT / "outputs/milestone6/runs/A2x/davis__davis_lab-coat__obj1.csv")
    a = m.run_policy(pred, "DEV", u, policies.Policy, n=9)
    assert a.JF.round(10).tolist() == ref.JF.round(10).tolist()[:8]
    v = pd.read_csv(ROOT / "outputs/milestone61/trackA/A2v/davis__davis_lab-coat__obj1.csv")
    b = m.run_policy(pred, "DEV", u, policies.SGGLV, n=9)
    assert b.JF.round(10).tolist() == v.JF.round(10).tolist()[:8]
    c = m.run_policy(pred, "DEV", u, policies.M1Verify, n=6)                      # probe-using policy runs causally
    assert len(c) == 5 and c.JF.notna().all()
