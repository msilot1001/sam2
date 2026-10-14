"""Regenerate every M7 decision report from raw per-frame CSVs (CPU only).

    uv run python scripts/m7_reports.py all
writes outputs/milestone7/{causal_pilots_report.json, teval_report.json, m3r_report.json, m3r_per_unit.csv,
vk_report.json, vk_per_unit.csv, runs/M3-vel-r/PROVENANCE.json}.

Machine matching: every arm is compared with the SGGL baseline produced on the same machine.
  Vast RTX 5090 : runs/SGGL (DEV+REP), rec_vk (VK)
  Ubuntu RTX 2080: rec (stage 0, DEV+REP)
M3-vel-r ran units 0-39 (unit order of m7.units("DEV,REP")) on Ubuntu and 40-63 on Vast (PROVENANCE.json).
"""
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.dropout_runner import atomic_csv, atomic_json

_s = importlib.util.spec_from_file_location("r", Path(__file__).with_name("m7_replay.py"))
r = importlib.util.module_from_spec(_s)
_s.loader.exec_module(r)
m7 = r.m7
O = m7.OUT
VAST_TAIL_START = 40


def _agg(pattern, name):
    fs = [f for f in glob.glob(pattern) if "vkitti" not in Path(f).name]
    return pd.concat([pd.read_csv(f) for f in fs]).groupby(["set", "unit", "video"]).agg(
        **{name: ("JF", "mean"), f"G_{name}": ("JF_G", "mean")}).reset_index()


def _area():
    return pd.read_csv(O / "replay_per_unit.csv")[["unit", "area0"]]


def provenance():
    us = m7.units("DEV,REP")
    prov = {m7.stem(u): ("vast-rtx5090" if k >= VAST_TAIL_START else "ubuntu-rtx2080") for k, (_, _, u) in enumerate(us)}
    (O / "runs/M3-vel-r/PROVENANCE.json").write_text(json.dumps(prov, indent=0))
    return prov


def pilots():
    B = _agg(str(O / "runs/SGGL/*.csv"), "JF|SGGL").merge(_area(), on="unit")
    out = {}
    for a in ("F1-blend.7", "M3-vel", "M1-always"):
        U = B.merge(_agg(str(O / f"runs/{a}/*.csv"), f"JF|{a}"), on=["set", "unit", "video"])
        assert (U[f"G_JF|{a}"] - U["G_JF|SGGL"]).abs().max() == 0, "baseline not machine-matched"
        U["JF_G"], U["small"] = U["G_JF|SGGL"], U.area0 < 0.01
        out[a] = r.pilot(U, f"JF|{a}")
    atomic_json(out, O / "causal_pilots_report.json")
    return out


def m3r():
    prov = provenance()
    rv = _agg(str(O / "runs/M3-vel-r/*.csv"), "JF|M3-vel-r")
    ub = _agg(str(O / "rec/*.csv"), "SGGL_u")
    vs = _agg(str(O / "runs/SGGL/*.csv"), "SGGL_v")
    m3 = _agg(str(O / "runs/M3-vel/*.csv"), "M3")[["unit", "M3"]]
    U = rv.merge(ub[["unit", "SGGL_u", "G_SGGL_u"]], on="unit").merge(vs[["unit", "SGGL_v", "G_SGGL_v"]], on="unit") \
        .merge(m3, on="unit").merge(_area(), on="unit")
    U["vast"] = U.unit.map(lambda u: prov[u.replace("/", "_")].startswith("vast"))
    U["JF|SGGL"] = np.where(U.vast, U.SGGL_v, U.SGGL_u)
    U["JF_G"] = np.where(U.vast, U.G_SGGL_v, U.G_SGGL_u)
    assert (U["G_JF|M3-vel-r"] - U.JF_G).abs().max() == 0, "baseline not machine-matched"
    U["small"] = U.area0 < 0.01
    x = r.pilot(U, "JF|M3-vel-r")
    S = U[U.small]
    d = S["JF|M3-vel-r"] - S.M3
    x["minus_M3vel"] = {"mean": float(d.mean()), "CI": r._boot(S.assign(d=d), "d"), "caveat": "M3-vel ran on Vast; ubuntu units cross-machine"}
    atomic_json(x, O / "m3r_report.json")
    atomic_csv(U, O / "m3r_per_unit.csv")
    return x


def teval():
    T = pd.concat([pd.read_csv(f) for f in glob.glob(str(O / "teval/*.csv"))])
    H = ["T0", "T1", "T2", "T3", "T4"]
    agg = {f"JF|{h}": (f"JF|{h}", "mean") for h in H + ["SGGL"]}
    agg.update(area0=("area0", "first"), JF_G=("JF_G", "mean"))
    U = T.groupby(["set", "unit", "video"]).agg(**agg).reset_index()
    U["small"] = U.area0 < 0.01
    res = {h: r.pilot(U, f"JF|{h}") for h in H}
    S = U[U.small]
    for a, b in (("T1", "T0"), ("T2", "T1"), ("T3", "T2"), ("T4", "T3"), ("T2", "T0")):
        d = S[f"JF|{a}"] - S[f"JF|{b}"]
        res[f"{a}-{b}"] = {"mean": float(d.mean()), "CI": r._boot(S.assign(d=d), "d")}
    atomic_json(res, O / "teval_report.json")
    return res


def vk():
    from rfeas.risk import cluster_bootstrap
    R = pd.concat([pd.read_csv(f) for f in glob.glob(str(O / "rec_vk/*.csv"))], ignore_index=True)
    R["veto"] = R.a2x_crop & ~R.a2v_crop          # frames where SGGL-V's global veto overrides SGGL's crop choice
    U = R.groupby(["unit", "video"]).agg(SGGL=("JF", "mean"), P5=("JF_G", "mean"), J_SGGL=("J", "mean"), J_P5=("J_G", "mean"),
                                         J_B0=("J_B0", "mean"), n=("JF", "size"),
                                         veto_frames=("veto", "sum")).reset_index()
    for a in ("SGGL-V", "M3-vel"):
        C = pd.concat([pd.read_csv(f) for f in glob.glob(str(O / f"runs/{a}/vkitti*.csv"))])
        U = U.merge(C.groupby("unit").agg(**{a: ("JF", "mean"), f"P5_{a}": ("JF_G", "mean")}).reset_index(), on="unit")
        assert (U[f"P5_{a}"] - U.P5).abs().max() < 1e-9, "VK arm not on the same machine as rec_vk"
    out = {"units": len(U), "scenes": int(U.video.nunique()), "frames": int(U.n.sum()),
           "note": "VKITTI2 Scene01/06 were development scenes in M2-M4 (fresh only for SGGL); effective n = 2 scenes",
           "veto_frames_total": int(U.veto_frames.sum()), "veto_units": int((U.veto_frames > 0).sum())}
    for k, (c, b) in {"SGGL-P5": ("SGGL", "P5"), "SGGL-V-SGGL": ("SGGL-V", "SGGL"), "M3-vel-SGGL": ("M3-vel", "SGGL"),
                      "J:SGGL-B0": ("J_SGGL", "J_B0"), "J:P5-B0": ("J_P5", "J_B0")}.items():
        d = U[c] - U[b]
        out[k] = {"mean": float(d.mean()), "CI_by_unit_descriptive": cluster_bootstrap(U.assign(d=d), "unit", lambda x: float(x.d.mean()), n=2000),
                  "by_scene": U.assign(d=d).groupby("video").d.mean().round(4).to_dict(),
                  "up_0.01": int((d > 0.01).sum()), "down_0.01": int((d < -0.01).sum()), "worst": [U.loc[d.idxmin(), "unit"], float(d.min())]}
    atomic_json(out, O / "vk_report.json")
    atomic_csv(U, O / "vk_per_unit.csv")
    return out


def all():  # noqa: A001
    for f in (pilots, m3r, teval, vk):
        res = f()
        print(f.__name__, json.dumps({k: (v.get("dJF_vs_SGGL", v.get("mean")) if isinstance(v, dict) else v) for k, v in res.items()},
                                     default=str)[:600])


if __name__ == "__main__":
    globals()[sys.argv[1]]()
