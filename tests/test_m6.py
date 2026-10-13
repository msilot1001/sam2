import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from spatial import vcam

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("m6", ROOT / "scripts/m6.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_crop_rules():
    g = np.zeros((480, 854), bool)
    g[100:110, 200:230] = True                         # box side 30 -> s = 120 = min(H,W)/4 (the clamp)
    assert vcam.crop_side(g) == 120
    g[:] = False
    g[100:102, 200:202] = True                         # tiny: clamped to the max zoom
    assert vcam.crop_side(g) == 120
    g[:] = False
    g[0:130, 0:10] = True                              # 4 * 130 >= 480: no crop
    assert vcam.crop_side(g) is None
    for c in [(0, 0), (479, 853), (240, 427), (-50, 900)]:
        y0, y1, x0, x1 = vcam.place(c, 200, (480, 854))
        assert y1 - y0 == 200 and x1 - x0 == 200 and y0 >= 0 and x0 >= 0 and y1 <= 480 and x1 <= 854
    m = np.zeros((200, 200), bool)
    m[5, 5] = True
    box = vcam.place((240, 427), 200, (480, 854))
    p = vcam.paste(m, box, (480, 854))
    assert p.sum() == 1 and p[box[0] + 5, box[2] + 5]
    assert not vcam.touches_border(m, box, (480, 854))
    m[0, 50] = True
    assert vcam.touches_border(m, box, (480, 854))
    assert not vcam.touches_border(m, (0, 200, 300, 500), (480, 854))  # top crop edge == frame edge


def test_lazy_identity_matches_official_loader():
    m6 = load()
    from rfeas.units import davis_units
    us = davis_units("val")[0]
    if not us:
        pytest.skip("no DAVIS")
    u = us[0]
    from sam2.utils.misc import load_video_frames
    ref, H, W = load_video_frames(str(m6.m56.frame_dir(u)), 1024, offload_video_to_cpu=True)
    f = vcam.LazyCropFrames(m6.paths(u))
    f.box = (0, H, 0, W)
    assert len(f) == len(ref) and all(torch.equal(f[t], ref[t]) for t in (0, 1, len(ref) - 1))


def test_units_exclude_confirmation():
    m6 = load()
    if not (ROOT / "outputs/milestone56/manifest.json").exists():
        pytest.skip("no manifest")
    C = {r["unit"] for r in json.loads((ROOT / "outputs/milestone56/manifest.json").read_text())["rows"]
         if r["split"] == "C_confirmation"}
    assert C and not (C & {u.name for _, u in m6.units()})


@pytest.mark.realdata
def test_identity_crop_equals_B1_and_crop_arm_is_causal():
    if not torch.cuda.is_available() or not (ROOT / "checkpoints/sam2.1_hiera_tiny.pt").exists():
        pytest.skip("no CUDA/checkpoint")
    from memint import controlled, engine
    m6 = load()
    pred = m6.m55.build_predictor()
    controlled.install(pred, None, {})
    u = next(u for s, u in m6.units(("DAVIS_val",)) if u.video == "davis/kite-surf" and u.obj_id == 2)
    s, od = m6.m56.start(pred, u, p5=True)
    ref, _ = engine.rollout(pred, s, od, 1, 4)
    pred.reset_state(s)
    H, W = m6.gt(u, 0).shape
    f = vcam.LazyCropFrames(m6.paths(u))
    f.box = (0, H, 0, W)
    with vcam.lazy_loader(f, (H, W)):
        s, od = m6.m56.start_dir(pred, "lazy", m6.gt(u, 0), p5=True)
    got, _ = engine.rollout(pred, s, od, 1, 4)
    pred.reset_state(s)
    assert all(np.array_equal(ref[t], got[t]) for t in ref)                    # disabled wrapper == B1
    heads = m6.Heads(pred)
    a, ma = m6.run_unit(pred, heads, "DAVIS_val", u, "A1", n=5)
    b, mb = m6.run_unit(pred, heads, "DAVIS_val", u, "A1", n=8)
    heads.close()
    assert all(np.array_equal(x, y) for x, y in zip(ma, mb[:4]))               # no dependence on later frames
    assert a.box.tolist() == b.box.tolist()[:4] and a.JF.tolist() == b.JF.tolist()[:4]


def test_reappear_events():
    m6 = load()
    area = [100, 100, 0, 0, 100, 10, 100, 0]
    empty = [a == 0 for a in area]
    border = [False, True, False, False, False, False, False, False]
    ev = m6.reappear_events(area, empty, border, 100)
    assert ev == [(2, 4, "out_of_frame"), (5, 6, "partial")]            # trailing absence never reappears
    assert m6.reappear_events([100, 0, 100], [False, True, False], [False] * 3, 100) == [(1, 2, "occluded")]


def test_zoom_target_rule():
    g = np.zeros((480, 854), bool)
    g[100:200, 100:150] = True                         # side 100: A1 s=400, A1z s=200
    assert vcam.crop_side(g) == 400 and vcam.crop_side(g, 0.5) == 200
    g[:, :] = False
    g[0:300, 0:10] = True
    assert vcam.crop_side(g, 0.5) is None


def test_select_scale_causal_and_ties():
    m6 = load()
    iou = {"B1": np.array([.8] * 5 + [0.0] * 50), "A1": np.array([.8] * 5 + [1.0] * 50), "A1z": np.array([.7] * 55)}
    assert m6.select_scale(iou) == "B1"                                  # tie -> B1; later frames ignored
    iou["A1"][:5] = .81
    assert m6.select_scale(iou) == "A1"
    assert m6.select_scale({"B1": np.array([.5] * 5)}) == "B1"
