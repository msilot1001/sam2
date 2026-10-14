"""M7 stage-0/1 CPU replay over the recorded SGGL trajectories (outputs/milestone7/rec).

    uv run python scripts/m7_replay.py frames      # per-frame metrics of every policy + oracles -> rec_replay/*.csv
    uv run python scripts/m7_replay.py summary     # per-unit table, pilot rule, oracles, taxonomy -> replay_report.json

REPLAY caveat: the crop trajectory is SGGL's; a policy that changes the output would move the crop in a live run.
Output-changing candidates that pass screening are re-run causally (scripts/m7.py run).
"""
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.dropout_runner import atomic_csv, atomic_json
from baseline.metrics import f_measure, j_measure
from spatial import mask_fusion, policies, sggl, vcam

_s = importlib.util.spec_from_file_location("m7", Path(__file__).with_name("m7.py"))
m7 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m7)
REC, RR = m7.REC, m7.OUT / "rec_replay"


def _view_mask(lg256, side):
    return torch.nn.functional.interpolate(torch.as_tensor(lg256, dtype=torch.float32)[None, None], size=(side, side),
                                           mode="bilinear", align_corners=False)[0, 0].numpy() > 0


def observations(df, Z, hw):
    """Yield per-frame observation dicts (the same fields a live run builds from sggl.View objects)."""
    side, side2 = int(df.side.iloc[0]), int(df.side2.iloc[0])
    for i, r in enumerate(df.itertuples()):
        box, box2 = tuple(int(x) for x in Z["box"][i]), tuple(int(x) for x in Z["box2"][i])
        o = {"t": r.video_idx, "hw": hw, "box": box, "sG": r.score_G, "sL": r.score_L, "sL2": r.score_L2,
             "sLp": r.score_Lp, "sLt": r.score_Lt, "touchL": bool(r.touch_L), "simL": Z["simL"][i].astype(np.float32)}
        lG = torch.as_tensor(Z["G"][i], dtype=torch.float32)[None, None]
        o["lG"] = sggl.full_to_frame(lG, hw)
        o["mG"] = o["lG"] > 0
        for k, b, s in (("L", box, side), ("L2", box2, side2), ("Lp", box, side), ("Lt", box, side)):
            lg = torch.as_tensor(Z[k][i], dtype=torch.float32)[None, None]
            o[f"l{k}"] = sggl.crop_to_frame(lg, b, hw)
            mv = _view_mask(Z[k][i], s)
            o[f"m{k}"] = vcam.paste(mv, b, hw)
            if k == "L":
                o["mL_view"] = mv
        yield o


def _metrics(m, g):
    return j_measure(m, g), f_measure(m, g)


def replay_unit(f):
    torch.set_num_threads(1)
    out = RR / f.name
    if out.exists():
        return
    df = pd.read_csv(f)
    Z = np.load(f.with_suffix(".npz"))
    u = m7.m6._UNITS().get(df.unit.iloc[0]) if df.set.iloc[0] == "DEV" else None
    if u is None:
        from rfeas.units import davis_units, vkitti_units
        pool = davis_units("train")[0] if df.set.iloc[0] == "REP" else vkitti_units()[0]
        u = next(x for x in pool if x.name == df.unit.iloc[0])
    g0 = m7.gt(u, 0)
    pols = [P(g0) for P in policies.ALL]
    rows = []
    for o in observations(df, Z, g0.shape):
        t = o["t"]
        g = m7.gt(u, t)
        r = {"unit": df.unit.iloc[0], "video_idx": t, "area0": float(g0.mean())}
        cache = {}
        for p in pols:
            m, acc = p(o)
            key = m.tobytes()
            if key not in cache:
                cache[key] = _metrics(m, g)
            r[f"J|{p.name}"], r[f"F|{p.name}"] = cache[key]
            r[f"acc|{p.name}"] = acc
            for k, v in p.log.items():
                r[f"{k}|{p.name}"] = v
            p.log = {}
        # oracles / bounds (offline, GT)
        for k in ("G", "L", "L2", "Lp", "Lt"):
            r[f"J_{k}"], r[f"F_{k}"] = _metrics(o[f"m{k}"], g)
        bl = mask_fusion.blend(o["lL"], o["lG"])
        r["J_blend"], r["F_blend"] = _metrics(bl, g)
        r["J_union"], r["F_union"] = _metrics(o["mL"] | o["mG"], g)
        r["J_inter"], r["F_inter"] = _metrics(o["mL"] & o["mG"], g)
        co = mask_fusion.component_oracle(o["mL"], o["mG"], g)
        r["J_comp"], r["F_comp"] = _metrics(co, g)
        r["IoU_LG"] = policies._iou(o["mL"], o["mG"])
        r["IoU_LL2"] = policies._iou(o["mL"], o["mL2"])
        r["g_obs"] = pols[0].h.g_observable(o["hw"])
        r["lost"], r["reacq"] = pols[0].h.lost, pols[0].h.reacq
        rows.append(r)
    atomic_csv(pd.DataFrame(rows), out)
    print(f.name, flush=True)


def frames(n_proc="8"):
    RR.mkdir(parents=True, exist_ok=True)
    fs = sorted(f for f in REC.glob("*.csv"))
    with Pool(int(n_proc)) as p:
        list(p.imap_unordered(replay_unit, fs))




# ----------------------------------------------------------------------------- analysis
def _boot(df, col, n=2000):
    from rfeas.risk import cluster_bootstrap
    return cluster_bootstrap(df, "video", lambda d: float(d[col].mean()), n=n)


def load_frames():
    """Per-frame table: recorded columns + replay columns (JF = (J+F)/2 per candidate)."""
    import warnings
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    parts = []
    for f in sorted(RR.glob("*.csv")):
        a, b = pd.read_csv(f), pd.read_csv(REC / f.name)
        keep = [c for c in b.columns if c not in a.columns or c in ("unit", "video_idx")]
        parts.append(b[keep].merge(a, on=["unit", "video_idx"]))
    F = pd.concat(parts, ignore_index=True)
    for c in [c for c in F.columns if c.startswith("J|")]:
        F["JF|" + c[2:]] = (F[c] + F["F|" + c[2:]]) / 2
    for k in ("G", "L", "L2", "Lp", "Lt", "blend", "union", "inter", "comp"):
        F[f"JF_{k}"] = (F[f"J_{k}"] + F[f"F_{k}"]) / 2
    return F.copy()


def units_table(F, cols):
    """Per-unit means of the given per-frame columns, with set/video/size flags."""
    U = F.groupby(["set", "unit", "video"]).agg(area0=("area0", "first"), n=("video_idx", "size"),
                                                 **{c: (c, "mean") for c in cols}).reset_index()
    U["small"], U["tiny"] = U.area0 < 0.01, U.area0 < 0.001
    return U


def pilot(U, col, base="JF|SGGL", label=None):
    """Pre-registered pilot rule on pooled small crop units (DEV+REP), per-unit deltas vs SGGL."""
    S = U[U.small].copy()
    S["d"] = S[col] - S[base]
    vid = S.groupby("video").d.mean()
    lovo = [S[S.video != v].d.mean() for v in vid.index]
    no_miami = S[S.video != "davis/miami-surf"].d.mean()
    ci = _boot(S, "d")
    r = {"n_units": int(len(S)), "n_videos": int(S.video.nunique()), "dJF_vs_SGGL": float(S.d.mean()), "CI_video": ci,
         "videos_up_0.01": int((vid > 0.01).sum()), "videos_down_0.01": int((vid < -0.01).sum()),
         "units_up_0.01": int((S.d > 0.01).sum()), "units_down_0.01": int((S.d < -0.01).sum()),
         "lovo_min": float(min(lovo)), "lovo_max": float(max(lovo)), "without_miami": float(no_miami),
         "worst_unit": [S.loc[S.d.idxmin(), "unit"], float(S.d.min())], "best_unit": [S.loc[S.d.idxmax(), "unit"], float(S.d.max())],
         "dJF_vs_P5": float((S[col] - S["JF_G"]).mean())}
    adv = (r["dJF_vs_SGGL"] >= 0.005 and r["videos_up_0.01"] >= 3 and r["lovo_min"] > 0 and r["without_miami"] > 0
           and r["worst_unit"][1] >= -0.05)
    rej = r["dJF_vs_SGGL"] <= 0 or (r["videos_up_0.01"] <= 2 and r["lovo_min"] <= 0)
    r["verdict"] = "ADVANCE" if adv else ("REJECT" if rej else "INCONCLUSIVE")
    return r


def taxonomy(F):
    """Offline (GT) attribution of disagreement / error frames to the M7 failure mechanisms 1-12 (multi-label +
    primary label by priority). Frames considered: presence disagreement, |J_L - J_G| > 0.3, or SGGL J < 0.5 with
    GT present, or SGGL output non-empty while GT empty."""
    G_abs = (F.score_G <= 0) | (F.area_G == 0)
    L_pres = (F.score_L > 0) & (F.area_L > 0)
    gtp = ~F.gt_empty
    ev = ((F.score_G > 0) != (F.score_L > 0)) | ((F.J_L - F.J_G).abs() > 0.3) | (gtp & (F["J|SGGL"] < 0.5)) \
        | (F.gt_empty & (F["J|SGGL"] < 1))
    E = F[ev].copy()
    g_abs, l_pres, gp = G_abs[ev], L_pres[ev], gtp[ev]
    lab = pd.DataFrame(index=E.index)
    lab["1_G_absent_resolution"] = gp & g_abs & (E.J_L >= 0.5) & ((E.score_Gp <= 0) | (E.J_Gp < 0.5)) & ((E.score_Lb <= 0) | (E.IoU_LLb < 0.3))
    lab["2_G_absent_state"] = gp & g_abs & (E.score_Gp > 0) & (E.J_Gp >= 0.5)
    lab["3_L_false_reacq"] = l_pres & (E.J_L < 0.1) & (E.gt_empty | (E.oth_L > 0.5)) & (E.lost | E.reacq)
    lab["4_correct_L_vetoed"] = gp & (E.J_L >= 0.5) & (E.score_G <= 0) & E["acc|SGGL"] & ~E["acc|SGGL-V"]
    lab["5_complementary"] = gp & (E.J_L >= 0.3) & (E.J_G >= 0.3) & (E[["J_blend", "J_union", "J_inter"]].max(axis=1) > E[["J_L", "J_G"]].max(axis=1) + 0.05)
    lab["6_different_targets"] = gp & (E.score_G > 0) & l_pres & (E.IoU_LG < 0.1) & ((E.oth_L > 0.5) | (E.oth_G > 0.5))
    lab["7_both_wrong"] = gp & (E.J_L < 0.3) & (E.J_G < 0.3)
    lab["8_crop_escape"] = gp & (E.gt_in_crop < 0.9) & (E.J_L < E.J_G)
    lab["9_memory_feedback"] = l_pres & (E.J_L < 0.3) & (E.J_Lp >= 0.5)
    lab["10_boundary_thin"] = gp & (E.IoU_LG >= 0.5) & (E["J|SGGL"] < 0.8)
    lab["11_annotation_absence"] = E.gt_empty & l_pres & (E.score_G > 0) & (E.IoU_LG > 0.5)
    lab["3b_L_wrong_object_tracking"] = l_pres & gp & (E.J_L < 0.1) & (E.oth_L > 0.5) & ~(E.lost | E.reacq)
    order = ["11_annotation_absence", "3_L_false_reacq", "3b_L_wrong_object_tracking", "6_different_targets",
             "9_memory_feedback", "2_G_absent_state", "1_G_absent_resolution", "4_correct_L_vetoed", "8_crop_escape",
             "7_both_wrong", "5_complementary", "10_boundary_thin"]
    prim = lab[order].idxmax(axis=1).where(lab[order].any(axis=1), "12_unresolved")
    E["primary"] = prim
    E["loss_SGGL"] = (1 - E["JF|SGGL"])
    out = {}
    for k in order + ["12_unresolved"]:
        sub = E[E.primary == k]
        out[k] = {"frames_primary": int(len(sub)), "frames_any": int(lab[k].sum()) if k in lab else int(len(sub)),
                  "units": int(sub.unit.nunique()), "videos": int(sub.video.nunique()),
                  "SGGL_JF_loss_sum": float(sub.loss_SGGL.sum()),
                  "top_videos": sub.groupby("video").size().sort_values(ascending=False).head(4).to_dict()}
    return out, E


def summary():
    F = load_frames()
    pols = [c[3:] for c in F.columns if c.startswith("JF|")]
    cols = [f"JF|{p}" for p in pols] + ["JF_G", "JF_L", "JF_L2", "JF_Lp", "JF_Lt", "JF_blend", "JF_union", "JF_inter", "JF_comp"]
    F["or_sel"] = F[["JF_G", "JF_L"]].max(axis=1)
    F["or_sel5"] = F[["JF_G", "JF_L", "JF_L2", "JF_Lp", "JF_Lt"]].max(axis=1)
    F["or_fuse"] = F[["JF_G", "JF_L", "JF_blend", "JF_union", "JF_inter"]].max(axis=1)
    F["or_comp"] = F[["or_sel", "JF_comp"]].max(axis=1)
    U = units_table(F, cols + ["or_sel", "or_sel5", "or_fuse", "or_comp", "JF"])
    rep = {"replay_fidelity": {"max_abs_J_diff_SGGL_replay_vs_live": float((F["J|SGGL"] - F.J).abs().max()),
                               "mean_JF_diff": float((F["JF|SGGL"] - F.JF).mean())},
           "counts": {s: {"units": int((U.set == s).sum()), "videos": int(U[U.set == s].video.nunique()),
                          "small_units": int(((U.set == s) & U.small).sum()), "tiny_units": int(((U.set == s) & U.tiny).sum()),
                          "frames": int((F.set == s).sum())} for s in U.set.unique()}}
    rep["pilots"] = {p: pilot(U, f"JF|{p}") for p in pols if p != "SGGL"}
    rep["pilots_by_set"] = {p: {s: float((U[(U.set == s) & U.small][f"JF|{p}"] - U[(U.set == s) & U.small]["JF|SGGL"]).mean())
                                for s in ("DEV", "REP")} for p in pols}
    S = U[U.small]
    rep["oracles_small_vs_SGGL"] = {k: {"mean": float((S[k] - S["JF|SGGL"]).mean()), "CI": _boot(S.assign(d=S[k] - S["JF|SGGL"]), "d")}
                                    for k in ("or_sel", "or_sel5", "or_fuse", "or_comp", "JF_blend", "JF_L2", "JF_Lp", "JF_Lt")}
    rep["SGGL_vs_P5_small"] = {"mean": float((S["JF|SGGL"] - S.JF_G).mean()), "CI": _boot(S.assign(d=S["JF|SGGL"] - S.JF_G), "d")}
    tx, E = taxonomy(F)
    rep["taxonomy"] = tx
    rep["event_frame_fraction"] = float(F.event.mean())
    # M1 2x2 on frames where L is present (correct = J >= 0.5 with GT present, or empty output when GT empty)
    def corr(J, area):
        return np.where(F.gt_empty, area == 0, J >= 0.5)
    cL, cLp = corr(F.J_L, F.area_L), corr(F.J_Lp, F.area_Lp)
    sel = (F.score_L > 0).to_numpy()
    rep["M1_2x2_L_present"] = {"both_correct": int((cL & cLp & sel).sum()), "L_correct_Lp_wrong": int((cL & ~cLp & sel).sum()),
                               "L_wrong_Lp_correct": int((~cL & cLp & sel).sum()), "both_wrong": int((~cL & ~cLp & sel).sum()),
                               "units_L_wrong_Lp_correct": int(F[~cL & cLp & sel].unit.nunique()),
                               "videos_L_wrong_Lp_correct": int(F[~cL & cLp & sel].video.nunique())}
    rep["M1_disagreement_vs_time"] = F.assign(dis=F.IoU_LLp < 0.5, tb=pd.cut(F.video_idx, [0, 10, 30, 60, 1000])).groupby("tb", observed=True).dis.mean().rename(index=str).to_dict()
    # R1: resolution probe at event frames: does degrading L to G's pixel density kill correct vs false candidates differently?
    R = F[F.score_Lb.notna() & (F.score_L > 0)]
    R = R.assign(Lgood=np.where(R.gt_empty, R.area_L == 0, R.J_L >= 0.5), lb_lost=(R.score_Lb <= 0) | (R.IoU_LLb < 0.3))
    rep["R1_resolution_probe"] = R.groupby(["event", "Lgood"]).agg(n=("lb_lost", "size"), frac_lost_at_G_res=("lb_lost", "mean"),
                                                                    units=("unit", "nunique")).reset_index().to_dict("records")
    rep["R2_L2"] = {"small_L2_vs_SGGL": rep["oracles_small_vs_SGGL"]["JF_L2"]}
    atomic_json(rep, m7.OUT / "replay_report.json")
    atomic_csv(U, m7.OUT / "replay_per_unit.csv")
    atomic_csv(E, m7.OUT / "taxonomy_frames.csv")
    print(json.dumps({k: v for k, v in rep.items() if k in ("replay_fidelity", "counts", "oracles_small_vs_SGGL", "SGGL_vs_P5_small")}, indent=1, default=str))
    for p, r in rep["pilots"].items():
        print(f"{p:14s} {r['dJF_vs_SGGL']:+.4f} {np.round(r['CI_video'], 4)} up{r['videos_up_0.01']} dn{r['videos_down_0.01']} "
              f"lovo[{r['lovo_min']:+.4f},{r['lovo_max']:+.4f}] noMiami {r['without_miami']:+.4f} worst {r['worst_unit'][1]:+.3f} {r['verdict']}")


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
