import json
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from baseline import frame_drops as fd
from baseline.dropout_eval import drop_runs, paired_frames, paired_summary, recovery_events, recovery_summary
from baseline.sam2_vos import Clip, clip_from_pattern, make_clip, verify_clip_folder, write_clip_folder
from vkitti.dataset import VKittiDataset

CONDS = {"random_30": ("random", 0.3), "random_50": ("random", 0.5), "burst_30": ("burst", 0.3),
         "burst_50": ("burst", 0.5)}


def gen(cond="burst_30", seed=0, ids=range(100, 187), track=86, scene="Scene01"):
    kind, rate = CONDS[cond]
    return fd.generate(scene, "clone", 0, track, ids, cond, kind, rate, seed)


# ----------------------------------------------------------------------------- generation
@pytest.mark.parametrize("cond", CONDS)
@pytest.mark.parametrize("n", [1, 2, 4, 5, 9, 10, 86, 135])
def test_invariants_and_prompt_kept(cond, n):
    p = gen(cond, ids=range(50, 51 + n))
    f = p.frames
    assert f.original_frame_id.tolist() == list(range(50, 51 + n))            # originals intact, ordered
    assert bool(f.is_received.iloc[0]) and bool(f.is_prompt.iloc[0])          # prompt never dropped
    assert p.received_ids[0] == 50 and list(p.received_ids) == sorted(set(p.received_ids))
    rec = f[f.is_received]
    assert rec.received_video_idx.tolist() == list(range(len(rec)))           # contiguous, no extras
    assert len(p.received_ids) + p.meta["total_dropped_frames"] == n + 1


def test_rate_rounding():
    assert [fd.round_half_up(x) for x in (0.5, 1.5, 2.5, 2.49, 25.8)] == [1, 2, 3, 2, 26]
    p = gen("random_30", ids=range(0, 87))     # n = 86 -> round(25.8) = 26 dropped
    assert p.meta["total_dropped_frames"] == 26
    p = gen("random_50", ids=range(0, 88))     # n = 87 -> round(43.5) = 44
    assert p.meta["total_dropped_frames"] == 44
    b = gen("burst_30", ids=range(0, 87))      # 86*0.3/4 = 6.45 -> 6 bursts = 24 frames
    assert b.meta["number_of_bursts"] == 6 and b.meta["total_dropped_frames"] == 24
    assert b.meta["actual_drop_rate"] == pytest.approx(24 / 86)
    b = gen("burst_50", ids=range(0, 87))      # 10.75 -> 11 bursts
    assert b.meta["number_of_bursts"] == 11


@pytest.mark.parametrize("cond", CONDS)
def test_deterministic_and_seed_sensitive(cond):
    a, b = gen(cond, 0), gen(cond, 0)
    assert a.meta == b.meta and a.frames.equals(b.frames)
    assert gen(cond, 1).meta["dropped_original_frame_ids"] != a.meta["dropped_original_frame_ids"]
    assert gen(cond, 0, track=63).meta["dropped_original_frame_ids"] != a.meta["dropped_original_frame_ids"]


def test_exact_positions_are_pinned():
    # Regression guard: the sha256 ranking must never change silently (M4 depends on it).
    assert fd.hash_order("k", 6) == fd.hash_order("k", 6)
    p = gen("burst_30", 0, ids=range(339, 426))
    assert [x["start"] for x in p.meta["bursts"]] == [345, 360, 378, 389, 399, 416]
    assert gen("random_30", 0, ids=range(0, 11)).meta["dropped_original_frame_ids"] == \
        sorted(1 + i for i in fd.hash_order(fd.rng_key("Scene01", "clone", 0, 86, "random_30", 0), 10)[:3])


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("cond", ["burst_30", "burst_50"])
def test_bursts_exact_length_and_separated(cond, seed):
    p = gen(cond, seed, ids=range(0, 60))
    runs = drop_runs(p.frames)
    assert [r for r in runs] == [(b["start"], b["end"]) for b in p.meta["bursts"]]  # runs == bursts: no merging
    assert all(e - s + 1 == 4 for s, e in runs)
    assert all(b[0] - a[1] >= 2 for a, b in zip(runs, runs[1:]))                  # >= 1 received between


def test_burst_infeasible_and_tiny():
    p = gen("burst_50", ids=range(0, 4))    # 3 post-prompt frames: no room for a 4-burst
    assert p.meta["number_of_bursts"] == 0 and p.meta["total_dropped_frames"] == 0
    assert p.meta["requested_number_of_bursts"] == 0 and p.meta["request_fits"] is True  # 3*0.5/4 rounds to 0
    p = gen("burst_50", ids=range(0, 2))    # one post-prompt frame
    assert p.meta["total_dropped_frames"] == 0 and p.received_ids == (0, 1)
    p = gen("burst_50", ids=range(0, 10))   # n = 9: 1.125 -> 1 burst
    assert p.meta["number_of_bursts"] == 1
    starts, k_req, fits = fd.burst_drop_positions(9, 1.0, "x")  # wants 2 bursts (2.25), only 9 >= 4+1+4 fits 2
    assert len(starts) == 2 and fits and starts[1] - starts[0] >= 5
    starts, k_req, fits = fd.burst_drop_positions(8, 1.0, "x")  # wants 2, only 1 fits (needs 9)
    assert len(starts) == 1 and k_req == 2 and not fits
    with pytest.raises(ValueError):
        gen("burst_30", ids=[5, 6, 8])


def test_consecutive_burst_boundary():
    # Exactly-fitting layout: bursts separated by exactly one received frame.
    starts, _, _ = fd.burst_drop_positions(9, 1.0, "x")
    assert starts == [0, 5]


# ----------------------------------------------------------------------------- manifest I/O
def test_manifest_roundtrip_and_no_silent_overwrite(tmp_path):
    p = gen("burst_50", 2)
    j, c = fd.save(p, tmp_path)
    q = fd.load(j)
    assert q.meta == p.meta and q.frames.equals(p.frames)
    assert fd.save(p, tmp_path) == (j, c)                 # identical re-save is a no-op
    before = (j.read_bytes(), c.read_bytes())
    other = gen("burst_50", 2, ids=range(100, 186))       # same name, different clip -> refuse
    with pytest.raises(fd.ManifestError):
        fd.save(other, tmp_path)
    assert (j.read_bytes(), c.read_bytes()) == before
    meta = json.loads(j.read_text())
    assert {"scene", "variation", "camera_id", "track_id", "condition", "seed", "prompt_original_frame_id",
            "original_start_frame", "original_end_frame", "requested_drop_rate", "actual_drop_rate",
            "burst_length", "number_of_bursts", "total_original_frames", "total_received_frames",
            "total_dropped_frames"} <= set(meta)
    assert pd.read_csv(c).columns.tolist() == fd.FRAME_COLUMNS


def test_tampered_manifest_rejected(tmp_path):
    j, c = fd.save(gen("random_30"), tmp_path)
    c.write_text(c.read_text().replace("True,False,", "False,True,", 1))
    with pytest.raises(fd.ManifestError, match="sha256"):
        fd.load(j)


def test_validate_catches_broken_patterns():
    p = gen("burst_30")
    bad = fd.DropPattern(dict(p.meta), p.frames.copy())
    bad.frames.loc[0, ["is_received", "is_dropped"]] = [False, True]
    with pytest.raises(fd.ManifestError):
        fd.validate(bad)
    bad = fd.DropPattern({**p.meta, "bursts": [{**p.meta["bursts"][0], "end": p.meta["bursts"][0]["end"] + 1, "length": 5}]
                          + p.meta["bursts"][1:]}, p.frames)
    with pytest.raises(fd.ManifestError):
        fd.validate(bad)


# ----------------------------------------------------------------------------- clips
Row = namedtuple("Row", "track_id prompt_frame last_frame")


def test_received_only_clip_and_symlinks(vkitti_root, tmp_path):
    ds = VKittiDataset(vkitti_root)
    p = fd.generate("Scene01", "clone", 0, 0, [0, 1, 2], "burst_50", "burst", 1.0, 0)  # n=2: no burst fits
    assert clip_from_pattern(p).original_frame_ids == (0, 1, 2)
    p = fd.generate("Scene01", "clone", 0, 0, [0, 1, 2], "random_50", "random", 0.5, 0)
    clip = clip_from_pattern(p)
    assert len(clip.original_frame_ids) == 2 and clip.original_frame_ids[0] == 0 and clip.variant == "random_50_seed0"
    d = write_clip_folder(ds, clip, tmp_path)
    assert d == tmp_path / "random_50_seed0" / "Scene01_track000"
    for i, f in enumerate(clip.original_frame_ids):
        assert (d / f"{i:05d}.jpg").resolve() == ds.rgb_path(f).resolve()
    dropped = p.meta["dropped_original_frame_ids"][0]
    assert ds.rgb_path(dropped).resolve() not in {x.resolve() for x in d.glob("*.jpg")}
    verify_clip_folder(ds, clip, d)
    # Rewriting into a folder that has stale frames from a longer clip leaves no leftovers.
    full = make_clip("Scene01", Row(0, 0, 2))
    write_clip_folder(ds, Clip("Scene01", 0, full.original_frame_ids, variant="random_50_seed0"), tmp_path)
    write_clip_folder(ds, clip, tmp_path)
    assert len(list(d.glob("*.jpg"))) == 2
    # M2 layout unchanged for the no-drop clip.
    assert write_clip_folder(ds, full, tmp_path) == tmp_path / "Scene01_track000"


def test_clip_rejects_bad_orders():
    with pytest.raises(ValueError):
        Clip("Scene01", 0, (3, 2))
    with pytest.raises(ValueError):
        Clip("Scene01", 0, (3, 3))
    with pytest.raises(ValueError):
        Clip("Scene01", 0, ())


# ----------------------------------------------------------------------------- paired evaluation
def metric_table(ids, j, prompt, gt_empty=None):
    j = np.asarray(j, float)
    return pd.DataFrame({"original_frame_id": ids, "is_prompt": [i == prompt for i in ids],
                         "gt_empty": gt_empty or [False] * len(ids), "J": j, "F": j, "JF": j, "F_strict": j})


def test_paired_uses_identical_frames_and_excludes_prompt():
    nd = metric_table(range(130, 137), [0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], 130)
    dr = metric_table([130, 131, 133, 135, 136], [0.95, 0.85, 0.6, 0.3, 0.2], 130)
    pf = paired_frames(nd, dr)
    assert pf.original_frame_id.tolist() == [131, 133, 135, 136]
    assert pf.J_no_drop_matched.tolist() == [0.9, 0.7, 0.5, 0.4]
    s = paired_summary(pf)
    assert s["matched_frames"] == 4
    assert s["J_no_drop_matched"] == pytest.approx(0.625) and s["J_drop"] == pytest.approx(0.4875)
    assert s["delta_J"] == pytest.approx(0.4875 - 0.625)       # drop - no_drop: negative = worse


def test_paired_guards():
    nd = metric_table(range(130, 135), [1] * 5, 130)
    with pytest.raises(ValueError):
        paired_frames(nd, metric_table([130, 140], [1, 1], 130))       # frame missing in no-drop
    with pytest.raises(ValueError):
        paired_frames(nd, metric_table([131, 132], [1, 1], 131))       # different prompt
    with pytest.raises(ValueError):
        paired_frames(nd, metric_table([130, 131], [1, 1], 130, [False, True]))  # different GT


def test_paired_with_empty_gt_frames_follows_m2_protocol():
    from baseline.metrics import j_measure
    z = np.zeros((4, 4), bool)
    j_empty_ok = j_measure(z, z)                    # both empty -> 1 (M2/DAVIS)
    nd = metric_table([0, 1, 2], [1, j_empty_ok, 1], 0, [False, True, False])
    dr = metric_table([0, 1, 2], [1, 0.0, 1], 0, [False, True, False])  # false positive on empty GT
    pf = paired_frames(nd, dr)
    assert pf.gt_empty.tolist() == [True, False] and pf.delta_J.tolist() == [-1.0, 0.0]


# ----------------------------------------------------------------------------- recovery events
def events_for(drop_ids, ids, j):
    frames = pd.DataFrame({"original_frame_id": ids, "is_dropped": [f in drop_ids for f in ids]})
    jb = {f: v for f, v in zip([f for f in ids if f not in drop_ids], j)}
    return recovery_events(frames, jb)


def test_recovery_classes():
    ids = list(range(100, 125))
    # bursts 102-105, 110-113, 118-121; received after: 106-109 (4), 114-117 (4), 122-124 (3)
    drops = set(range(102, 106)) | set(range(110, 114)) | set(range(118, 122))
    recv = [f for f in ids if f not in drops]          # 100,101,106..109,114..117,122..124
    j = dict(zip(recv, [1, 1, 0.9, 0.8, 0.9, 0.9, 0.2, 0.3, 0.6, 0.7, 0.1, 0.1, 0.1]))
    ev = recovery_events(pd.DataFrame({"original_frame_id": ids, "is_dropped": [f in drops for f in ids]}), j)
    assert ev["class"].tolist() == ["intact_at_first_frame", "recovered_later", "unrecovered_end"]
    assert ev.degraded_after_drop.tolist() == [False, True, True]
    assert ev.recovery_onset_received_k.tolist()[1] == 3 and ev.recovery_onset_original_frame[1] == 116
    assert ev.recovery_onset_original_gap[1] == 116 - 113
    assert ev.first_received_after.tolist() == [106, 114, 122]
    assert ev.censored_by.tolist() == ["next_drop", "next_drop", "clip_end"]
    assert ev.confirmed_within_n.iloc[1] is True          # confirmed at received frame 4 <= 5
    assert ev.confirmed_within_n.tolist()[2] is None      # 3-frame window < 5: undeterminable
    s = recovery_summary(ev)
    assert s["degraded_runs"] == 2 and s["degraded_recovered"] == 1 and s["degraded_censored"] == 1
    assert s["degraded_within_5_denominator"] == 0


def test_recovery_censoring_edges():
    ids = list(range(0, 12))
    # run 1-4, one received (5), run 6-9, received 10, 11
    drops = set(range(1, 5)) | set(range(6, 10))
    j = {0: 1, 5: 0.2, 10: 0.3, 11: 0.2}
    ev = recovery_events(pd.DataFrame({"original_frame_id": ids, "is_dropped": [f in drops for f in ids]}), j)
    assert ev["class"].tolist() == ["insufficient_observation", "unrecovered_end"]
    assert ev.lost_after_drop.tolist() == [False, False]
    # A run that reaches the clip end has no observation at all.
    ev = recovery_events(pd.DataFrame({"original_frame_id": [0, 1, 2], "is_dropped": [False, True, True]}), {0: 1})
    assert ev["class"].tolist() == ["no_observation"] and ev.window_length.tolist() == [0]


def test_recovery_structure_without_j_and_pre_drop_flag():
    ids = list(range(10))
    frames = pd.DataFrame({"original_frame_id": ids, "is_dropped": [3 <= f <= 6 for f in ids]})
    ev = recovery_events(frames)
    assert ev.window_frames.tolist() == [[7, 8, 9]] and ev.pre_drop_frame.tolist() == [2]
    ev = recovery_events(frames, {0: 1, 1: 1, 2: 0.05, 7: 0.9, 8: 0.9, 9: 0.9})
    assert ev.pre_drop_J.tolist() == [0.05]              # already lost before the drop
    assert ev["class"].tolist() == ["intact_at_first_frame"]


# ----------------------------------------------------------------------------- real manifests
REAL = __import__("pathlib").Path(__file__).resolve().parents[1] / "outputs" / "milestone3" / "manifests"


@pytest.mark.realdata
@pytest.mark.skipif(not REAL.is_dir(), reason="M3 manifests not generated")
def test_saved_real_manifests_validate_and_reproduce():
    js = sorted(REAL.glob("*/*.json"))
    assert len(js) == 96
    for j in js:
        fd.load(j)   # sha256 + invariants + regeneration
    tracks = {p.name for p in REAL.iterdir()}
    assert tracks == {"Scene01_track090", "Scene01_track086", "Scene01_track063", "Scene01_track054",
                      "Scene06_track012", "Scene06_track009", "Scene06_track011", "Scene06_track013"}
