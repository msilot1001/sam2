"""Analysis of round-2 memory write-back (A3, A3s) vs logged A2x / B1 / A1 (pre-registered rules)."""
import glob, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from baseline.dropout_runner import atomic_json, atomic_csv
_s = importlib.util.spec_from_file_location("m6", Path(__file__).with_name("m6.py")); m6 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m6)
OUT = Path("outputs/milestone61/trackA")
sm = {r["unit"]: r["small"] for r in json.load(open("outputs/milestone6/preregistration.json"))["rows"]}
R = Path("outputs/milestone6/runs")
rows = []
for arm in ("A2d", "A3", "A3s"):
    for f in sorted((OUT / arm).glob("*.csv")):
        X = pd.read_csv(f)
        B = pd.read_csv(R / "B1" / f.name).set_index("video_idx"); A = pd.read_csv(R / "A1" / f.name).set_index("video_idx")
        Z = pd.read_csv(R / "A2x" / f.name).set_index("video_idx")
        X["JF_B1"] = X.video_idx.map(B.JF); X["JF_A1"] = X.video_idx.map(A.JF); X["JF_A2x"] = X.video_idx.map(Z.JF)
        X["or_new"] = X[["JF_G", "JF_L"]].max(axis=1); X["or_old"] = X[["JF_B1", "JF_A1"]].max(axis=1)
        rows.append(X)
D = pd.concat(rows, ignore_index=True)
D["late"] = D.video_idx >= 21
U = D.groupby(["arm", "set", "unit", "video"]).agg(JF=("JF", "mean"), JF_G=("JF_G", "mean"), JF_L=("JF_L", "mean"), B1=("JF_B1", "mean"),
                                                  A2x=("JF_A2x", "mean"), or_new=("or_new", "mean"), or_old=("or_old", "mean"),
                                                  lat=("latency_s", "median"), vram=("peak_vram_bytes", "max"),
                                                  wroteG=("wrote", lambda w: float((w == "G").mean()))).reset_index()
L = D[D.late].groupby(["arm", "unit"]).agg(JF21=("JF", "mean"), B121=("JF_B1", "mean"), A2x21=("JF_A2x", "mean")).reset_index()
U = U.merge(L, on=["arm", "unit"], how="left")
U["small"] = U.unit.map(sm)
U["d"] = U.JF - U.B1; U["d_vs_A2x"] = U.JF - U.A2x; U["dG"] = U.JF_G - U.B1; U["or_gain"] = U.or_new - U.or_old
atomic_csv(U, OUT / "a3_per_unit.csv")
rep = {}
for arm, g in U.groupby("arm"):
    G = g[g.set.isin(m6.GATE_SETS)]
    s = G[G.small]
    rep[arm] = {"units": int(len(g)), "small_gate_units_applied": int(len(s)),
                "small_dJF_over15": float(s.d.sum() / 15), "A2x_small_over15": float(s.A2x.sub(s.B1).sum() / 15),
                "small_vs_A2x_over15": float(s.d_vs_A2x.sum() / 15), "CI_vs_A2x_applied": m6._boot(s, "d_vs_A2x"),
                "units_below_A2x_0.01": g[g.d_vs_A2x < -0.01][["unit", "d_vs_A2x"]].round(4).to_dict("records"),
                "units_above_A2x_0.01": int((g.d_vs_A2x > 0.01).sum()),
                "global_tracker_G_vs_B1_small_over15": float(s.dG.sum() / 15),
                "oracle_gain_new_vs_old_small_over15": float(s.or_gain.sum() / 15),
                "late21_vs_A2x_small": float((s.JF21 - s.A2x21).mean()),
                "all_sets_vs_A2x": {k: float(v.d_vs_A2x.mean()) for k, v in g.groupby("set")},
                "latency_median_s": float(g.lat.median()), "peak_vram_GB": float(g.vram.max() / 1e9),
                "write_rate_G": float(g.wroteG.mean())}
    r = rep[arm]
    r["success"] = bool(r["small_vs_A2x_over15"] >= 0.01 and not r["units_below_A2x_0.01"] and r["oracle_gain_new_vs_old_small_over15"] >= 0.01)
atomic_json(rep, OUT / "a3_report.json")
print(json.dumps(rep, indent=1, default=float))
print(U[U.arm != "A2d"][["arm", "set", "unit", "small", "B1", "A2x", "JF", "d_vs_A2x", "dG"]].round(4).to_string())
