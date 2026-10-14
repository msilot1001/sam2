import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from spatial import motion, ranker, rescue as R

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("m8", ROOT / "scripts/m8.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_view_maps_crop_and_foveation_geometry():
    n, c, a = 854, 400.0, 6.0
    crop, fov = R.axis_map(n, c, a, False), R.axis_map(n, c, a, True)
    assert np.isclose(crop.xin[1] - crop.xin[0], R.M / a)                          # plain crop width = M / alpha
    assert fov.xin[0] == 0 and fov.xin[-1] == n and fov.xout[0] == 0 and fov.xout[-1] == R.M   # whole axis kept
    assert np.isclose(fov.mag(c), a) and np.isclose(crop.mag(c), a)                # matched magnification at the target
    x = np.linspace(0, n, 101)
    assert np.allclose(fov.inv(fov.fwd(x)), x)                                     # invertible everywhere
    ident = R.axis_map(n, c, 0.5, True)                                            # alpha below full-frame scale
    assert np.allclose(ident.xout, [0, R.M]) and np.allclose(ident.xin, [0, n])


def test_view_image_and_inverse_mask_round_trip():
    H, W = 120, 200
    m = np.zeros((H, W), bool)
    m[50:62, 140:150] = True
    img = torch.from_numpy(m.astype(np.float32))[None, None] * 20 - 10            # logit-like single channel
    for fov in (False, True):
        my, mx = R.axis_map(H, 56, 20.0, fov), R.axis_map(W, 145, 20.0, fov)
        v = R.view_image(img, my, mx)                                              # 1x1x1024x1024
        lg = torch.nn.functional.interpolate(v, size=(256, 256), mode="area")
        back = R.view_to_frame(lg, my, mx, (H, W))
        seen = ~np.isnan(back)
        assert seen[56, 145] and (seen.all() if fov else not seen.all())
        b = np.nan_to_num(back, nan=-1) > 0
        assert (b & m).sum() / (b | m).sum() > 0.9
        p = R.frame_to_view_xy([(145.0, 56.0)], my, mx)[0]
        assert v[0, 0, int(p[1]), int(p[0])] > 0


def test_contrastive_attention_suppresses_background_matches_and_peaks_nms():
    C = 8
    e = torch.eye(C)
    feat = torch.zeros(1, C, 10, 10) + e[7][None, :, None, None] * 0.1
    feat[0, :, 2, 2] = e[0]                                                         # target
    feat[0, :, 7, 7] = 0.6 * e[0] + 0.8 * e[1]                                     # distractor: similar, but bg-like
    fg, bg = e[:1], e[1:3]
    att = R.heat_contrast(feat, fg, bg, tau=0.1)
    cos = R.heat_proto(feat, e[0])
    assert cos[7, 7] > 0.5 and att[7, 7] < 0.2 and att[2, 2] > 0.9
    pk = R.peaks(cos, 2, 1)
    assert pk[0][:2] == (2, 2) and pk[1][:2] == (7, 7)


def test_motion_track_velocity_extrapolation_and_flow_carry():
    tr = motion.Track((10, 20, 10, 20))
    tr.observe(2, (14, 24, 18, 28))                                               # moved (+4, +8) in 2 frames
    assert np.allclose(tr.velocity(), [2, 4]) and np.allclose(tr.extrapolate(5, 1.0), (25, 35))
    flow = np.zeros((60, 80, 2), np.float32)
    flow[..., 0], flow[..., 1] = 3.0, -1.0
    tr.step_flow(flow, (60, 80))
    assert np.allclose(tr.flow_center(), (18, 26)) and np.allclose(tr.cam, [3, -1])
    tr.observe(3, None)                                                            # empty output keeps the carried box
    assert np.allclose(tr.flow_center(), (18, 26))


def test_rankers_shapes_padding_and_lr():
    df = pd.DataFrame({"score": [2.0, -1.0], "iou": [0.8, 0.2], "cos_fg": [0.9, np.nan], "cos_bg": [0.1, np.nan],
                       "area_ratio": [1.0, 0.0], "dist_flow": [0.1, np.nan]})
    X = ranker.scalars(df)
    assert X.shape == (2, len(ranker.SCALARS)) and np.isfinite(X).all()
    assert ranker.r1(df)[1] == -np.inf
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, X.shape[1])).astype(np.float32)
    lr = ranker.LR().fit(Z, Z[:, 0] > 0)
    assert (lr(Z) > 0).mean() > 0.4 and ((lr(Z) > 0) == (Z[:, 0] > 0)).mean() > 0.9
    net = ranker.SetRanker(5)
    out = net(torch.randn(2, 4, 256 + X.shape[1]), torch.zeros(2, 4, dtype=torch.long), torch.randn(2, 6, 256),
              torch.zeros(2, 6, dtype=torch.long), torch.tensor([[False] * 4, [False, False, True, True]]))
    assert out.shape == (2, 5) and (out[1, 2:4] < -1e3).all()


@pytest.mark.realdata
def test_rescue_probes_are_non_storing_and_empty_arm_equals_sggl():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from memint import controlled
    from spatial import sggl
    m = load()
    pred = m.m55.build_predictor()
    controlled.install(pred, None, {})
    u = m.m7.m6._UNITS()["davis__davis/lab-coat__obj1"]
    m.trigger = lambda G, L, i, t: "regular"                                       # probe every frame
    flow = motion.Flow()
    a, cand, vec = m.diag_unit(pred, flow, "DEV", u, n=6)
    assert len(cand) and len(vec) == len(cand) and set(cand.gen) >= {"B_att#0", "F_cpfr8_G", "C_flowbox", "E_beam1"}
    b = m.run_arm(pred, None, "DEV", u, {"name": "SGGL", "gens": []}, None, n=6)
    assert a.JF.round(12).tolist() == b.JF.round(12).tolist()                      # probes never touched production
    assert pred.fill_hole_area == 0


@pytest.mark.realdata
def test_shared_view_backbone_is_bit_identical_to_recomputing():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from memint import controlled
    m = load()
    pred = m.m55.build_predictor()
    controlled.install(pred, None, {})
    u = m.m7.m6._UNITS()["davis__davis/lab-coat__obj1"]
    m.trigger = lambda G, L, i, t: "regular"
    flow = motion.Flow()
    _, a, va = m.diag_unit(pred, flow, "DEV", u, n=4)
    orig = R.probe
    m.R.probe = lambda *x, feats=None, **k: orig(*x, **k)                         # force one backbone pass per probe
    try:
        _, b, vb = m.diag_unit(pred, flow, "DEV", u, n=4)
    finally:
        m.R.probe = orig
    assert a.score.tolist() == b.score.tolist() and a.J.tolist() == b.J.tolist() and np.array_equal(va, vb)
