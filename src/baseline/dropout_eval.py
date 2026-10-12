"""Paired no-drop vs dropout evaluation and burst-recovery events (Milestone 3, used by M4).

Paired: both runs are scored on the SAME original frame ids = frames received in the dropout run,
prompt excluded. delta = dropout - matched_no_drop (negative = degradation).

Recovery (mentor plan §10.3-4): after a run of dropped frames, recovery is first confirmed when
J >= j_threshold on two consecutive RECEIVED frames. The observation window is the received frames
after the run, up to the next dropped run or the end of the clip. Time coordinates: received-frame
count and original-frame difference only (no seconds: FPS unverified, see MILESTONE_1 §4).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["J", "F", "JF", "F_strict"]


# ----------------------------------------------------------------------------- paired evaluation
def paired_frames(no_drop: pd.DataFrame, drop: pd.DataFrame) -> pd.DataFrame:
    """Per-frame table on the dropout run's received non-prompt frames.

    Both inputs are per-frame metric tables for ONE track (columns original_frame_id, is_prompt,
    gt_empty and METRICS), e.g. baseline.evaluation.evaluate output.
    """
    d = drop[~drop.is_prompt]
    nd = no_drop.set_index("original_frame_id")
    missing = sorted(set(d.original_frame_id) - set(nd.index))
    if missing:
        raise ValueError(f"no-drop run lacks frames {missing[:10]}")
    if drop.is_prompt.sum() != 1 or no_drop.is_prompt.sum() != 1 or \
            drop[drop.is_prompt].original_frame_id.iloc[0] != no_drop[no_drop.is_prompt].original_frame_id.iloc[0]:
        raise ValueError("runs must share exactly one, identical prompt frame")
    out = d[["original_frame_id", "gt_empty", *METRICS]].rename(columns={m: f"{m}_drop" for m in METRICS})
    out = out.join(nd[METRICS].rename(columns={m: f"{m}_no_drop_matched" for m in METRICS}), on="original_frame_id")
    gt_nd = nd.loc[out.original_frame_id, "gt_empty"].to_numpy()
    if not np.array_equal(gt_nd, out.gt_empty.to_numpy()):
        raise ValueError("GT emptiness differs between runs: different track or data")
    for m in METRICS:
        out[f"delta_{m}"] = out[f"{m}_drop"] - out[f"{m}_no_drop_matched"]
    return out.reset_index(drop=True)


def paired_summary(pf: pd.DataFrame) -> dict:
    """Track-level means over the matched frames (the unit for later statistics)."""
    s = {"matched_frames": int(len(pf))}
    for m in METRICS:
        s[f"{m}_no_drop_matched"] = float(pf[f"{m}_no_drop_matched"].mean())
        s[f"{m}_drop"] = float(pf[f"{m}_drop"].mean())
        s[f"delta_{m}"] = s[f"{m}_drop"] - s[f"{m}_no_drop_matched"]
    return s


def check_m2_reference(m2_dir: Path, pattern, sam2_cfg: dict, eval_cfg: dict, m2_eval_cfg: dict) -> pd.DataFrame:
    """Validate that stored M2 predictions are a valid no-drop reference for this pattern; return its rows."""
    m2_dir = Path(m2_dir)
    env = json.loads((m2_dir / "reports" / "environment.json").read_text())
    if env["checkpoint_sha256"] != sam2_cfg["checkpoint_sha256"] or env["sam2_upstream_commit"] != sam2_cfg["upstream_commit"]:
        raise ValueError("M2 reference used a different checkpoint or SAM 2 commit")
    if env["config"] != sam2_cfg["config"] or env["apply_postprocessing"] != sam2_cfg["apply_postprocessing"]:
        raise ValueError("M2 reference used a different model config")
    if {k: m2_eval_cfg[k] for k in ("boundary_th", "boundary_th_strict", "lost_j")} != \
            {k: eval_cfg[k] for k in ("boundary_th", "boundary_th_strict", "lost_j")}:
        raise ValueError("evaluation settings differ from the M2 reference")
    m = pattern.meta
    mp = pd.read_csv(m2_dir / "reports" / "video_index_mapping.csv")
    mp = mp[(mp.scene == m["scene"]) & (mp.track_id == m["track_id"])]
    if mp.original_frame_id.tolist() != list(range(m["original_start_frame"], m["original_end_frame"] + 1)) or \
            mp.video_idx.tolist() != list(range(len(mp))):
        raise ValueError("M2 clip frames differ from the manifest's original range")
    pf = pd.read_csv(m2_dir / "reports" / "per_frame_metrics.csv")
    pf = pf[(pf.scene == m["scene"]) & (pf.track_id == m["track_id"])]
    if pf[pf.is_prompt].original_frame_id.tolist() != [m["prompt_original_frame_id"]]:
        raise ValueError("M2 prompt frame differs")
    return pf.reset_index(drop=True)


# ----------------------------------------------------------------------------- recovery events
def drop_runs(frames: pd.DataFrame) -> list[tuple[int, int]]:
    """Maximal runs of consecutive dropped original frames, as (start, end) original ids."""
    runs, start = [], None
    for f, dropped in zip(frames.original_frame_id, frames.is_dropped):
        if dropped and start is None:
            start = f
        if not dropped and start is not None:
            runs.append((start, prev))
            start = None
        prev = f
    if start is not None:
        runs.append((start, prev))
    return runs


def recovery_events(frames: pd.DataFrame, j_by_frame: dict | None = None, j_threshold: float = 0.5,
                    lost_j: float = 0.1, within: int = 5) -> pd.DataFrame:
    """One row per dropped run (= per burst in burst conditions).

    frames: manifest frame table. j_by_frame: {original_frame_id: J} for received frames of the
    dropout run (None -> structure only, for preparing M4). Classes (j known):
      no_observation           run reaches the clip end: nothing received afterwards
      insufficient_observation 1 received frame before the next run/end: the 2-frame rule cannot fire
      intact_at_first_frame    J>=thr on the 1st and 2nd frames after the run: never inaccurate,
                               NOT counted as a recovery
      recovered_later          first confirmed pair starts at received frame k>=2
      unrecovered_next_burst   no pair before the next dropped run (censored by it)
      unrecovered_end          no pair before the clip end (censored by the end)
    degraded_after_drop = J of the first received frame < thr. pre_drop_J flags runs that start
    while tracking is already poor (degradation not attributable to the drop).
    """
    f = frames.reset_index(drop=True)
    pos = {fid: i for i, fid in enumerate(f.original_frame_id)}
    runs = drop_runs(f)
    rows = []
    for r, (start, end) in enumerate(runs):
        before = f.iloc[: pos[start]]
        before = before[~before.is_dropped]
        nxt = runs[r + 1][0] if r + 1 < len(runs) else None
        after = f.iloc[pos[end] + 1:]
        if nxt is not None:
            after = after[after.original_frame_id < nxt]
        win = after.original_frame_id.tolist()  # all received by construction
        row = {"run_index": r, "run_start": start, "run_end": end, "run_length": end - start + 1,
               "pre_drop_frame": int(before.original_frame_id.iloc[-1]) if len(before) else None,
               "first_received_after": win[0] if win else None,
               "window_frames": win, "window_length": len(win),
               "censored_by": "next_drop" if nxt is not None else "clip_end",
               "within_n_determinable": len(win) >= within}
        if j_by_frame is not None:
            js = [j_by_frame[x] for x in win]
            k = next((i + 1 for i in range(len(js) - 1) if js[i] >= j_threshold and js[i + 1] >= j_threshold), None)
            if not win:
                cls = "no_observation"
            elif k == 1:
                cls = "intact_at_first_frame"
            elif k:
                cls = "recovered_later"
            elif len(win) < 2:
                cls = "insufficient_observation"
            else:
                cls = "unrecovered_next_burst" if nxt is not None else "unrecovered_end"
            row.update({
                "window_J": js, "pre_drop_J": j_by_frame.get(row["pre_drop_frame"]),
                "first_J": js[0] if js else None,
                "degraded_after_drop": bool(js and js[0] < j_threshold),
                "lost_after_drop": bool(js and js[0] < lost_j),
                "class": cls,
                "recovery_onset_received_k": k,              # 1-based received frames after the run
                "recovery_confirmed_received_k": k + 1 if k else None,
                "recovery_onset_original_frame": win[k - 1] if k else None,
                "recovery_onset_original_gap": win[k - 1] - end if k else None,   # original frames after run end
                # Plan-literal 2-frame criterion confirmed within the first `within` received frames.
                # True also for intact runs -- use class / degraded_after_drop before calling it "recovery".
                "confirmed_within_n": (k + 1 <= within) if k else (False if len(win) >= within else None),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def recovery_summary(ev: pd.DataFrame, within: int = 5) -> dict:
    """Counts by class; recovery rates only over runs that degraded (J<thr at the first frame)."""
    if ev.empty or "class" not in ev:
        return {"runs": int(len(ev))}
    deg = ev[ev.degraded_after_drop]
    det = deg[deg.within_n_determinable]
    lit = ev[ev.within_n_determinable]
    return {
        "runs": int(len(ev)), "by_class": ev["class"].value_counts().to_dict(),
        "degraded_runs": int(len(deg)), "lost_runs": int(ev.lost_after_drop.sum()),
        "degraded_recovered": int((deg["class"] == "recovered_later").sum()),
        "degraded_censored": int(deg["class"].isin(["unrecovered_next_burst", "unrecovered_end",
                                                     "insufficient_observation"]).sum()),
        f"degraded_recovered_within_{within}_rate": (float(det.confirmed_within_n.astype(bool).mean())
                                                    if len(det) else None),
        f"degraded_within_{within}_denominator": int(len(det)),
        # Plan §10.4 literal: every run with a >= `within`-frame window, intact runs included.
        f"plan_literal_within_{within}_rate": float(lit.confirmed_within_n.astype(bool).mean()) if len(lit) else None,
        f"plan_literal_within_{within}_denominator": int(len(lit)),
    }
