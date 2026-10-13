"""Round-4 replication of A2x (A2d) and A3 on DAVIS-train crop objects vs their B1 logs (pre-registered)."""
import glob, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from baseline.dropout_runner import atomic_csv, atomic_json
_s = importlib.util.spec_from_file_location("m6", Path(__file__).with_name("m6.py")); m6 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m6)
OUT = Path("outputs/milestone61/replication")
B = pd.concat([pd.read_csv(f) for f in glob.glob("outputs/milestone61/trackD/runs/B1/*.csv")])
ub = B.groupby(["unit", "video"]).JF.mean().rename("JF_B1").reset_index()
rows = []
for arm in ("A2d", "A3"):
    fs = sorted(glob.glob(str(OUT / arm / "*.csv")))
    if fs:
        X = pd.concat([pd.read_csv(f) for f in fs])
        rows.append(X.groupby("unit").agg(JF=("JF", "mean"), JF_G=("JF_G", "mean")).rename(columns=lambda c: f"{c}_{arm}"))
U = ub.set_index("unit").join(rows, how="inner").reset_index()
from rfeas.units import davis_units
af = {u.name: float(m6.gt(u, 0).mean()) for u in davis_units("train")[0] if u.name in set(U.unit)}
U["area_frac"] = U.unit.map(af); U["small"] = U.area_frac < 0.01; U["tiny"] = U.area_frac < 0.001
for a in ("A2d", "A3"):
    if f"JF_{a}" in U:
        U[f"d_{a}"] = U[f"JF_{a}"] - U.JF_B1
if "JF_A3" in U:
    U["A3_minus_A2d"] = U.JF_A3 - U.JF_A2d
    U["G_A3_vs_B1"] = U.JF_G_A3 - U.JF_B1
atomic_csv(U, OUT / "replication_per_unit.csv")
rep = {"objects": int(len(U)), "videos": int(U.video.nunique()), "small": int(U.small.sum()), "tiny": int(U.tiny.sum())}
for a in ("A2d", "A3"):
    if f"d_{a}" not in U:
        continue
    for name, g in (("all", U), ("small", U[U.small]), ("tiny", U[U.tiny]), ("nonsmall", U[~U.small])):
        if len(g):
            rep[f"{a}_{name}"] = {"n": int(len(g)), "videos": int(g.video.nunique()), "dJF": float(g[f"d_{a}"].mean()),
                                  "CI": m6._boot(g, f"d_{a}") if g.video.nunique() > 1 else None,
                                  "up_0.01": int((g[f"d_{a}"] > 0.01).sum()), "down_0.01": int((g[f"d_{a}"] < -0.01).sum())}
if "A3_minus_A2d" in U:
    for name, g in (("all", U), ("small", U[U.small])):
        rep[f"A3_minus_A2d_{name}"] = {"mean": float(g.A3_minus_A2d.mean()), "CI": m6._boot(g, "A3_minus_A2d"),
                                       "worse_0.01": g[g.A3_minus_A2d < -0.01][["unit", "A3_minus_A2d"]].round(4).to_dict("records")}
    rep["G_A3_vs_B1_small"] = float(U[U.small].G_A3_vs_B1.mean())
s = rep.get("A2d_small")
if s:
    rep["replicates_A2x"] = bool(s["dJF"] > 0 and s["CI"][0] > 0 and s["down_0.01"] <= 0.1 * s["n"])
if "A3_minus_A2d_all" in rep:
    rep["A3_nonnegative"] = rep["A3_minus_A2d_all"]["mean"] >= 0
atomic_json(rep, OUT / "replication_report.json")
print(json.dumps(rep, indent=1, default=float))
print(U.sort_values("area_frac")[["unit", "area_frac", "JF_B1", *[c for c in U if c.startswith("d_") or c == "A3_minus_A2d"]]].round(4).to_string())
