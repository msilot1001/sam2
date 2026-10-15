"""M10: oracle-driven bottleneck decomposition of SGGL (CUDA FP32, frozen SAM 2.1 Hiera-T).

    uv run python scripts/m10.py prereg                       # freeze arms/subsets/metrics (before any M10 output)
    M9_DATA=<root> uv run python scripts/m10.py work          # worker: claim (arm, unit) tasks until none are left

Every arm is the SGGL rollout (scripts/m8.py run_arm, centre/side/output/memory as in M9) with ONE kind of
GT-informed intervention switched on. GT is used only as the arm spec says; the output mask is always a SAM 2
prediction (or empty), never a GT mask. Knobs (default = SGGL):
  center : sggl | gt          L crop centre for frame t = GT box centre of frame t (non-causal; held while GT absent)
  side   : fixed | gt         L crop side for frame t = M6 crop-side rule (vcam.crop_side) applied to GT of frame t
  output : sggl | select | correct
           select  = GT-best (by J) of the fixed pool {G x3, L x3 raw decoder hypotheses, empty}; ties keep SGGL
           correct = perfect target-region detector with executable actions only: keep the SGGL output if correct;
                     else switch to the other view's mask if that is correct; else abstain (empty) if the output is
                     wrong-region (precision < 0.5, incl. any output while GT is absent); else keep (partial target)
  admit  : all | gt | region | presence | random
           gt       = a view's frame-t prediction enters its memory iff it is correct in that view
           region   = blocked only if the view's mask is wrong-region (non-empty, precision < 0.5 in that view)
           presence = blocked only when GT is absent in the view and the view's mask is non-empty
           random   = per unit-view, the same NUMBER of blocked frames as arm OR7, at seeded random frames
  memory : native | gt | cand | cond0
           gt    = frame t's memory of each view re-encoded from the GT mask (SAM 2 mask-input path), after output t
           cand  = re-encoded from the view's own GT-best raw hypothesis (candidate-constrained)
           cond0 = only the frame-0 prompt memory is ever read
"correct" (view v, mask m): GT inside v's region empty and m empty, or J(m, GT in v) >= 0.5.
Workers claim tasks by atomic lock files, so any number of workers on any number of GPUs can share one queue.
"""
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

from baseline.dropout_runner import atomic_csv, atomic_json
from memint import controlled
from spatial import sggl, vcam

_s = importlib.util.spec_from_file_location("m7", Path(__file__).with_name("m7.py"))
m7 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m7)
m55 = m7.m55
OUT = Path("outputs/milestone10")
RUNS = Path(os.environ.get("M10_RUNS_DIR", OUT / "runs"))
PREREG = Path(os.environ.get("M10_PREREG", OUT / "preregistration.json"))
LOCKS = Path(os.environ.get("M10_LOCKS", OUT / "locks"))    # outside the synced runs dir
TAU = 0.5

ARMS = {   # single oracles (phase 2), in queue priority order; combinations are added by amendment after phase 2
    "OR0": {},
    "OR7": {"admit": "gt"},
    "OR7W": {"admit": "region"},
    "OR8": {"memory": "gt"},
    "OR4": {"center": "gt", "side": "gt"},
    "OR5M": {"output": "select", "memory": "cand"},
    "OR6": {"output": "correct", "admit": "gt"},
    "OR2": {"center": "gt"},
    "OR3": {"side": "gt"},
    "OR1M": {"admit": "presence"},
    "OR5S": {"output": "select"},
    "OR8C0": {"memory": "cond0"},
    "OR7R": {"admit": "random"},
}
DEFAULT = {"center": "sggl", "side": "fixed", "output": "sggl", "admit": "all", "memory": "native"}
SETS = "MOSEV2:dev,VOST:dev,LVOSV2:dev"
PLAN = {
    "subset": "M9 frozen manifests, dev split only (MOSEv2 62 / VOST 31 / LVOS v2 16 units; LVOS first 300 frames). "
              "conf split is not used for any oracle in phase 2; it may be used once for a fixed list of arms "
              "(declared by amendment before launch) as a replication. PUMaVOS C sealed.",
    "baseline": "OR0 = SGGL rerun on the SAME pod with the extended logging; parity vs M9 rows reported.",
    "offline_oracles_from_OR0_rows": {
        "OR1off": "output := empty on GT-absent frames, unchanged elsewhere",
        "OR5off": "framewise hindsight best of the 7-member pool (by J; J&F of that member)",
        "OR6off": "perfect target-region detector on the logged G/L masks (same rule as output=correct)",
        "note": "framewise hindsight; no rollout, no claim about future frames"},
    "primary": "per dataset, video-equal-weighted mean of (arm - OR0) J&F (unit means -> video means), dev split",
    "ci": "95% video-cluster bootstrap (10k resamples of videos, video-equal weights); LOVO min/max; top-2 video "
          "share of net gain; sign counts. Datasets reported separately; pooled only as a secondary number.",
    "secondary": "dJ, dF, unit-weighted dJ&F, tiny (area0 < 0.001) dJ&F, wrong-region loss, drift episodes "
                 "(onset/duration/recovery), worst-video regression, latency/VRAM",
    "recovered_loss": "dJ&F / (1 - OR0 J&F) per dataset (fraction of the baseline's missing J&F)",
    "stopping": "an arm is not expanded beyond dev if its dev effect is |dJ&F| < 0.005 on all three datasets, or if "
                "a leak check fails (output equals a GT mask). Negative results are not grounds to stop. "
                "Combinations (phase 3) are declared by amendment after phase 2, chosen by single-oracle headroom.",
    "machine": "all phase-2 arms on one RunPod RTX 4090 pod; comparisons only within that pod",
}


def spec(name):
    return {**DEFAULT, **json.loads(PREREG.read_text())["arms"][name]}


# ----------------------------------------------------------------------------- SAM 2 hooks
def hook_decoder(pred):
    """Keep the raw decoder output of the last call (3 multimask logits BEFORE the no-object suppression)."""
    box = {}
    pred.sam_mask_decoder.register_forward_hook(lambda m, a, o: box.update(lr=o[0].detach(), iou=o[1].detach()))
    return box


@torch.inference_mode()
def rewrite_memory(pred, st, od, t, m_view):
    """Replace frame t's memory entry by SAM 2's own encoding of mask `m_view` (view-res bool) as a mask input
    (the add_new_mask path: memory features, object pointer and score all from the mask; no memory read)."""
    m = torch.from_numpy(np.ascontiguousarray(m_view)).float().cuda()[None, None]
    s = pred.image_size
    mi = Fn.interpolate(m, size=(s, s), mode="bilinear", align_corners=False, antialias=True) >= 0.5
    planner, pred._memint_planner = pred._memint_planner, None
    try:
        out, _ = pred._run_single_frame_inference(inference_state=st, output_dict=od, frame_idx=t, batch_size=1,
                                                  is_init_cond_frame=False, point_inputs=None, mask_inputs=mi,
                                                  reverse=False, run_mem_encoder=True)
    finally:
        pred._memint_planner = planner
    od["non_cond_frame_outputs"][t] = out


def cands(lr, box, hw, g):
    """Raw hypotheses (1x3x256x256 logits) of one view -> list of frame-res bool masks (crop views pasted)."""
    if box is None:
        up = Fn.interpolate(lr.float(), size=tuple(hw), mode="bilinear", align_corners=False)[0] > 0
        return [x.cpu().numpy() for x in up]
    y0, y1, x0, x1 = box
    up = Fn.interpolate(lr.float(), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0] > 0
    return [vcam.paste(x.cpu().numpy(), box, hw) for x in up]


def jac(m, g):
    u = np.logical_or(m, g).sum()
    return 1.0 if u == 0 else float(np.logical_and(m, g).sum() / u)


def region(g, box):
    """GT restricted to a view's region (frame coords)."""
    if box is None:
        return g
    y0, y1, x0, x1 = box
    r = np.zeros_like(g)
    r[y0:y1, x0:x1] = g[y0:y1, x0:x1]
    return r


def prec(m, g):
    k = m.sum()
    return 1.0 if k == 0 else float(np.logical_and(m, g).sum() / k)


def correct(m, gv):
    return jac(m, gv) >= TAU      # jac(empty, empty) = 1


def gt_side(g, hw):
    """M6 crop-side rule on the GT of the frame being predicted (None if absent)."""
    b = vcam.bbox(g)
    if b is None:
        return None
    s = max(b[1] - b[0], b[3] - b[2]) / vcam.TARGET
    return max(s, min(hw) / vcam.MAX_ZOOM)


def random_blocks(u, n):
    """OR7R: per view, the number of frames OR7 blocked on this unit, at seeded uniform random frames."""
    o = pd.read_csv(RUNS / "OR7" / f"{m7.stem(u)}.csv")
    rng = np.random.default_rng(int(hashlib.sha256(u.name.encode()).hexdigest()[:8], 16))
    return {v: set(rng.choice(np.arange(1, n), size=int((~o[f"adm_{v}"]).sum()), replace=False).tolist()) for v in "GL"}


# ----------------------------------------------------------------------------- rollout
def run_unit(pred, dec, s, sp, u, name, A, n=None):
    n = n or len(u.frames)
    g0 = m7.gt(u, 0)
    hw = g0.shape
    d = sggl.Dual(pred, m7.paths(u), g0)
    rblock = random_blocks(u, n) if A["admit"] == "random" else None
    other = getattr(m7, "other", None)
    rows = []
    for t in range(1, n):
        g = m7.gt(u, t)
        if A["center"] == "gt" and g.any():        # non-causal: GT of the frame being predicted
            d.set_center(vcam.center_of(g))
        if A["side"] == "gt" and g.any():
            d.set_side("L", gt_side(g, hw))
        t0 = time.perf_counter()
        V, C = {}, {}
        for v in "GL":
            planner = sggl._cond_only if A["memory"] == "cond0" else None
            V[v] = d.view(v, t, planner=planner)
            C[v] = cands(dec["lr"], V[v].box, hw, g)
        out, ok = sggl.a2x(V["G"], V["L"], hw)
        mv = {"G": V["G"].mask, "L": vcam.paste(V["L"].mask, V["L"].box, hw)}
        gv = {"G": g, "L": region(g, V["L"].box)}
        ok_v = {v: correct(mv[v], gv[v]) for v in "GL"}
        pool = C["G"] + C["L"] + [np.zeros(hw, bool)]
        pj = [jac(m, g) for m in pool]
        j_sggl = jac(out, g)
        best = int(np.argmax(pj))
        src = "L" if ok else "G"
        if A["output"] == "select" and pj[best] > j_sggl + 1e-9:
            out, src = pool[best], ("G", "L", "E")[min(best // 3, 2)]
        elif A["output"] == "correct" and not correct(out, g):
            alt = "G" if ok else "L"
            if correct(mv[alt], g):
                out, src = mv[alt], alt
            elif prec(out, g) < TAU:
                out, src = np.zeros(hw, bool), "E"
        adm, rew = {}, {}
        for v in "GL":
            st = d.S[v]
            if A["admit"] == "gt":
                adm[v] = ok_v[v]
            elif A["admit"] == "region":
                adm[v] = prec(mv[v], gv[v]) >= TAU
            elif A["admit"] == "presence":
                adm[v] = gv[v].any() or not mv[v].any()
            elif A["admit"] == "random":
                adm[v] = t not in rblock[v]
            else:
                adm[v] = True
            if not adm[v]:
                st.excluded.add(t)
            rew[v] = False
            box = V[v].box
            crop = (lambda m: m) if box is None else (lambda m: m[box[0]:box[1], box[2]:box[3]])
            if A["memory"] == "gt":
                rewrite_memory(pred, st.st, st.od, t, crop(gv[v]))
                rew[v] = True
            elif A["memory"] == "cand":
                cj = [jac(m, gv[v]) for m in C[v]] + [jac(np.zeros(hw, bool), gv[v])]
                k = int(np.argmax(cj))
                if cj[k] > jac(mv[v], gv[v]) + 1e-9:
                    rewrite_memory(pred, st.st, st.od, t, crop(C[v][k] if k < 3 else np.zeros(hw, bool)))
                    rew[v] = True
        torch.cuda.synchronize()
        lat = time.perf_counter() - t0
        y0, y1, x0, x1 = V["L"].box
        oth = other(u, t) if other is not None else np.zeros(hw, bool)
        bj = pool[best]
        rows.append({"set": s, "split": sp, "unit": u.name, "video": u.video, "arm": name, "video_idx": t,
                     "acc": ok, "src": src, "latency_s": lat, "score_G": V["G"].score, "score_L": V["L"].score,
                     "iou_G": V["G"].iou, "iou_L": V["L"].iou, "gt_empty": not g.any(), "area0": float(g0.mean()),
                     **m55.jf(out, g), "JF_G": m55.jf(mv["G"], g)["JF"], "JF_L": m55.jf(mv["L"], g)["JF"],
                     "J_G": jac(mv["G"], g), "J_L": jac(mv["L"], g), "Jv_L": jac(mv["L"], gv["L"]),
                     "ok_G": ok_v["G"], "ok_L": ok_v["L"], "adm_G": adm["G"], "adm_L": adm["L"],
                     "rew_G": rew["G"], "rew_L": rew["L"],
                     **{f"cJ{k}": pj[k] for k in range(7)}, "best": best, "JF_best": m55.jf(bj, g)["JF"],
                     "crop_y0": y0, "crop_x0": x0, "crop_side": y1 - y0, "gt_area": int(g.sum()),
                     "gt_in_crop": float(gv["L"].sum() / g.sum()) if g.any() else np.nan,
                     "out_area": int(out.sum()), "out_inter_gt": int((out & g).sum()),
                     "out_inter_other": int((out & oth).sum()), "other_area": int(oth.sum()),
                     "G_area": int(mv["G"].sum()), "G_inter_gt": int((mv["G"] & g).sum()),
                     "L_area": int(mv["L"].sum()), "L_inter_gt": int((mv["L"] & g).sum())})
        if A["center"] != "gt":
            d.set_center(vcam.center_of(out))    # SGGL: centre follows the output (held while it is empty)
    peak = torch.cuda.max_memory_allocated()
    d.close()
    return pd.DataFrame(rows).assign(peak_vram_bytes=peak)


# ----------------------------------------------------------------------------- queue
def prereg():
    assert not PREREG.exists(), "preregistration is frozen"
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    atomic_json({"created_kst": time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)),
                 "created_before_any_m10_output": True, "arms": ARMS, "default": DEFAULT, "sets": SETS, "tau_J": TAU, "plan": PLAN,
                 "doc": __doc__}, PREREG)
    print("froze", PREREG)


def tasks(arms, sets):
    us = m7.units(sets)
    us.sort(key=lambda x: -len(x[2].frames))                       # longest first: short tail
    return [(a, s, sp, u) for a in arms for s, sp, u in us]


def lock(f):
    return LOCKS / f.parent.name / f"{f.stem}.lock"


def claim(f):
    lock(f).parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(lock(f), os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def work(arms=None, sets=SETS):
    arms = (arms or ",".join(json.loads(PREREG.read_text())["arms"])).split(",")
    pred = m55.build_predictor()
    controlled.install(pred, None, {})
    dec = hook_decoder(pred)
    env = {"machine": os.environ.get("M8_MACHINE", socket.gethostname()), "torch": torch.__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "fill_hole_area": pred.fill_hole_area,
           "code": json.loads(Path("CODE_SYNC.json").read_text()) if Path("CODE_SYNC.json").exists() else "working tree"}
    left = tasks(arms, sets)
    while left:
        waiting = []
        for a, s, sp, u in left:
            dd = RUNS / a
            dd.mkdir(parents=True, exist_ok=True)
            f = dd / f"{m7.stem(u)}.csv"
            if f.exists() or lock(f).exists():
                continue
            A = spec(a)
            if A["admit"] == "random" and not (RUNS / "OR7" / f"{m7.stem(u)}.csv").exists():
                waiting.append((a, s, sp, u))                      # needs OR7's blocked counts first
                continue
            if not claim(f):
                continue
            if not (dd / f"ENV_{env['machine']}.json").exists():
                atomic_json({**env, "arm": A}, dd / f"ENV_{env['machine']}.json")
            torch.cuda.reset_peak_memory_stats()
            w0 = time.time()
            df = run_unit(pred, dec, s, sp, u, a, A)
            atomic_csv(df.assign(wall_s=time.time() - w0, machine=env["machine"]), f)
            lock(f).unlink()
            print(a, u.name, len(df), round(df.JF.mean(), 4), f"{len(df) / (time.time() - w0):.2f} fps", flush=True)
        left = waiting
        if left:
            time.sleep(30)


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
