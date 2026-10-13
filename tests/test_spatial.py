import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from baseline.metrics import j_measure
from spatial import detail

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
DAVIS = ROOT / "data/davis2017/DAVIS"


def load_m55():
    spec = importlib.util.spec_from_file_location("m55", ROOT / "scripts/m55.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bfs_count(m):
    seen, n = np.zeros_like(m), 0
    H, W = m.shape
    for y, x in zip(*np.nonzero(m)):
        if seen[y, x]:
            continue
        n += 1
        st = [(y, x)]
        seen[y, x] = True
        while st:
            a, b = st.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    c, d = a + dy, b + dx
                    if 0 <= c < H and 0 <= d < W and m[c, d] and not seen[c, d]:
                        seen[c, d] = True
                        st.append((c, d))
    return n


@pytest.mark.parametrize("seed", range(5))
def test_components_match_bfs(seed):
    m = np.random.default_rng(seed).random((60, 90)) > 0.6
    lab, n = detail.components(m)
    assert n == _bfs_count(m)
    assert set(np.unique(lab[m])) == set(range(1, n + 1)) and (lab[~m] == 0).all()
    s = np.zeros((40, 40), bool)
    s[2:38, 5] = True                                   # one long 1-px line
    assert detail.components(s)[1] == 1


def test_thin_part_and_retention():
    m = np.zeros((50, 50), bool)
    m[10:30, 10:30] = True                              # thick block
    m[40, 5:45] = True                                  # 1-px line
    tp = detail.thin_part(m, 2)
    assert tp[40, 5:45].all() and not tp[15:25, 15:25].any()
    assert detail.thin_recall(m, m) == 1.0 and detail.thin_recall(m & ~tp, m) == 0.0
    assert detail.component_retention(m, m) == 1.0
    assert detail.component_retention(m & ~tp, m) == 0.5


def test_mask_resizing_matches_official_add_new_mask():
    rng = np.random.default_rng(0)
    m = rng.random((48, 85)) > 0.7
    x = torch.tensor(m)[None, None].float()
    ref = (torch.nn.functional.interpolate(x, size=(1024, 1024), align_corners=False, mode="bilinear",
                                           antialias=True) >= 0.5).float()     # code copied from add_new_mask
    assert torch.equal(detail.to_model(m), ref)
    assert detail.to_model(m, "nearest").shape == ref.shape
    with pytest.raises(ValueError):
        detail.to_model(m, "bicubic")


def test_gt_alignment_roundtrip_is_identity_for_large_shapes_and_unshifted():
    m = np.zeros((480, 854), bool)
    m[100:300, 200:600] = True
    assert j_measure(detail.model_roundtrip(m), m) > 0.995
    rt = detail.lowres_roundtrip(m)
    assert j_measure(rt, m) > 0.97
    ys, xs = np.nonzero(rt)
    assert abs(ys.mean() - 199.5) < 1 and abs(xs.mean() - 399.5) < 1          # no systematic shift
    line = np.zeros((480, 854), bool)
    line[240, 100:700] = True
    assert j_measure(detail.lowres_roundtrip(line), line) < 0.5                # 1-px detail below 256 resolution


def test_manifest_deterministic_and_frozen(tmp_path, monkeypatch):
    m = load_m55()
    gt = np.zeros((480, 854), bool)
    gt[10:20, 10:20] = True
    assert m.strata(gt) == m.strata(gt.copy()) and m.strata(gt)["small"]
    monkeypatch.setattr(m, "OUT", tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"rows": []}))
    monkeypatch.setattr(m, "davis_units", lambda split: ([], []))
    with pytest.raises(SystemExit):
        m.manifest()
    assert m.inner_point(gt) == m.inner_point(gt)


def test_no_future_or_gt_in_prediction_paths():
    import inspect
    import io
    import tokenize
    m = load_m55()
    for fn in (m.start_variant,):
        toks = tokenize.generate_tokens(io.StringIO(inspect.getsource(fn)).readline)
        code = " ".join(t.string for t in toks if t.type not in (tokenize.STRING, tokenize.COMMENT))
        assert "davis_gt" not in code and "frames[t" not in code
    src = inspect.getsource(m.survey)
    assert src.index("engine.rollout") < src.index("image_oracle")            # predictions before any later-frame GT


def test_paired_ablation_frames_identical():
    from rfeas.pairing import paired
    import pandas as pd
    a = pd.DataFrame({"video_idx": [1, 2, 3], "gt_empty": False, "J": [1, .9, .2], "F": 1.0, "JF": 1.0, "F_strict": 1.0})
    with pytest.raises(ValueError):
        paired(a, a.iloc[:2])


@pytest.fixture(scope="module")
def pred():
    if not (CKPT.exists() and DAVIS.is_dir() and torch.cuda.is_available()):
        pytest.skip("checkpoint, DAVIS or CUDA missing")
    return load_m55().build_predictor()


@pytest.mark.realdata
def test_native_equivalence_and_prompt_fidelity(pred):
    from memint import engine
    from rfeas.units import davis_units, davis_frame_dir, davis_gt
    m = load_m55()
    u = next(x for x in davis_units("val")[0] if x.video == "davis/kite-surf" and x.obj_id == 1)
    gt0 = davis_gt(u, u.frames[0])
    s, od = engine.start(pred, davis_frame_dir(u), gt0)
    out0 = engine.prompt_mask_output(pred, s)
    ref, _ = engine.rollout(pred, s, od, 1, 5)
    pred.reset_state(s)
    assert np.array_equal(out0, detail.lowres_roundtrip(gt0))           # prompt output == 256 round trip
    for kw in ({}, {"mode": "official"}):                              # our own official resize == add_new_mask's
        s, od = m.start_variant(pred, davis_frame_dir(u), gt0, **kw)
        got, _ = engine.rollout(pred, s, od, 1, 5)
        pred.reset_state(s)
        pred.use_mask_input_as_output_without_sam = True
        assert all(np.array_equal(got[t], ref[t]) for t in ref)
    s, od = m.start_variant(pred, davis_frame_dir(u), gt0, fullres_mem=True)
    got, _ = engine.rollout(pred, s, od, 1, 5)
    pred.reset_state(s)
    assert any(not np.array_equal(got[t], ref[t]) for t in ref)         # the factor actually changes memory


def test_guided_filter_identity_and_causality():
    from spatial import proto
    I = torch.rand(1, 1, 30, 40)
    p = (torch.rand(1, 1, 30, 40) > 0.5).float()
    assert torch.isfinite(proto.guided_filter(I, p, 2, 1e-3)).all()
    flat = torch.full((1, 1, 30, 40), 0.3)
    assert torch.allclose(proto.guided_filter(I, flat, 3, 1e-3), flat, atol=1e-5)       # constant input preserved
    import inspect
    src = inspect.getsource(proto.predict)
    assert "frame_idx + 1" not in src and "gt" not in src.lower().replace("guide", "")


@pytest.mark.realdata
def test_prototype_official_branch_is_native(pred):
    from memint import engine
    from rfeas.units import davis_units, davis_frame_dir, davis_gt
    from spatial import proto
    u = next(x for x in davis_units("val")[0] if x.video == "davis/kite-surf" and x.obj_id == 1)
    gt0, jp = davis_gt(u, u.frames[0]), sorted(davis_frame_dir(u).glob("*.jpg"))
    s, od = engine.start(pred, davis_frame_dir(u), gt0)
    ref, _ = engine.rollout(pred, s, od, 1, 4)
    pred.reset_state(s)
    for feed in (False, True):
        s, od = engine.start(pred, davis_frame_dir(u), gt0)
        offs = {}
        for t in range(1, 5):
            out, mk, off = proto.predict(pred, s, od, t, proto.grey(jp[t], "cuda"), 2, 1e-2, feed)
            od["non_cond_frame_outputs"][t] = out
            offs[t] = off
        pred.reset_state(s)
        same = all(np.array_equal(offs[t], ref[t]) for t in ref)
        assert same if not feed else (np.array_equal(offs[1], ref[1]) and not same)   # memory arm alters only later frames
