import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("m56", ROOT / "scripts/m56.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_episodes_merge_and_recovery():
    m = load()
    assert m.episodes([1, .2, .3, 1, 1, .1, 1, 1, 1, 1, .4]) == [(1, 5, 6), (10, 10, None)]
    assert m.episodes([1, .2, 1, 1, 1, .2]) == [(1, 1, 2), (5, 5, None)]          # gap 3 > 2: separate
    assert m.episodes([.1, .1]) == [(0, 1, None)] and m.episodes([1, 1]) == []
    import pandas as pd
    df = pd.DataFrame({"unit": "u", "arm": "a", "video_idx": [3, 1, 2, 4], "J": [.1, 1, .2, 1]})
    e = m.episode_table(df)
    assert e.onset.tolist() == [2] and e.duration.tolist() == [2] and e.frames_to_recover.tolist() == [2]


def test_history_reset_is_causal_and_targeted():
    m = load()
    plan = {"spatial": [(0, "cond", 0)] + [(7 - r, "non_cond", 30 - r) for r in range(6, 0, -1)],
            "pointers": [(30, "cond", 0)] + [(d, "non_cond", 30 - d) for d in range(1, 16)], "max_ptrs": 16}
    p = m.history_reset(28)
    assert p(27, plan, {}) == plan                                         # before t0: untouched
    out = p(30, plan, {})
    assert {t for _, s, t in out["spatial"] if s == "non_cond"} == {28, 29}
    assert {t for _, s, t in out["pointers"] if s == "non_cond"} == {28, 29}
    assert (0, "cond", 0) in out["spatial"] and (30, "cond", 0) in out["pointers"]


def test_crop_box_covers_all_gt_and_clamps():
    m = load()
    g = [np.zeros((50, 80), bool) for _ in range(3)]
    g[0][10:12, 5:7] = True
    g[2][40:49, 70:79] = True
    y0, y1, x0, x1 = m.crop_box(g)
    for x in g:
        ys, xs = np.nonzero(x)
        if len(ys):
            assert y0 <= ys.min() and ys.max() < y1 and x0 <= xs.min() and xs.max() < x1
    assert y1 <= 50 and x1 <= 80 and y0 >= 0 and x0 >= 0


def test_manifest_split_is_video_level_and_frozen(tmp_path, monkeypatch):
    p = ROOT / "outputs/milestone56/manifest.json"
    if not p.exists():
        pytest.skip("manifest not built")
    rows = json.loads(p.read_text())["rows"]
    by_video = {}
    for r in rows:
        by_video.setdefault(r["video"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_video.values())                     # never split inside a video
    assert {"A_exploration", "B_development", "C_confirmation"} == {r["split"] for r in rows}
    m = load()
    (tmp_path / "manifest.json").write_text(json.dumps({"rows": []}))
    with pytest.raises(SystemExit):                                        # the shared freeze refuses a change
        m.m55._freeze(tmp_path / "manifest.json", {"rows": [{"unit": "x"}]})


def test_confirmation_split_never_run():
    m = load()
    runs = ROOT / "outputs/milestone56/runs"
    p = ROOT / "outputs/milestone56/manifest.json"
    if not p.exists():
        pytest.skip("manifest not built")
    C = {r["unit"].replace("/", "_") for r in json.loads(p.read_text())["rows"] if r["split"] == "C_confirmation"}
    done = {f.stem for f in runs.glob("*.csv")} if runs.exists() else set()
    assert not (C & done)


@pytest.mark.realdata
def test_offload_start_equals_engine_start():
    if not torch.cuda.is_available() or not (ROOT / "checkpoints/sam2.1_hiera_tiny.pt").exists():
        pytest.skip("no CUDA/checkpoint")
    from memint import engine
    from rfeas.units import davis_units
    m = load()
    pred = m.m55.build_predictor()
    u = next(x for x in davis_units("val")[0] if x.video == "davis/kite-surf" and x.obj_id == 1)
    s, od = engine.start(pred, m.frame_dir(u), m.gt_of(u, u.frames[0]))
    ref, _ = engine.rollout(pred, s, od, 1, 4)
    pred.reset_state(s)
    s, od = m.start(pred, u, p5=False)
    got, _ = engine.rollout(pred, s, od, 1, 4)
    pred.reset_state(s)
    assert all(np.array_equal(ref[t], got[t]) for t in ref)
    s, od = m.m55.start_variant(pred, m.frame_dir(u), m.gt_of(u, u.frames[0]), fullres_mem=True)
    p5ref, _ = engine.rollout(pred, s, od, 1, 4)
    pred.reset_state(s)
    s, od = m.start(pred, u, p5=True)
    got, _ = engine.rollout(pred, s, od, 1, 4)
    pred.reset_state(s)
    assert all(np.array_equal(p5ref[t], got[t]) for t in p5ref)              # B1 == M5.5 P5
