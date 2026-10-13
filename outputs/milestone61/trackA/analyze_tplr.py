"""Track A analysis: T1 (TPLR) vs B1/A1/A2x and the candidate-oracle value test (pre-registered)."""
import glob, json
import numpy as np, pandas as pd
import importlib.util
s = importlib.util.spec_from_file_location("m6", "scripts/m6.py"); m6 = importlib.util.module_from_spec(s); s.loader.exec_module(m6)
sm = {r["unit"]: r["small"] for r in json.load(open("outputs/milestone6/preregistration.json"))["rows"]}
R = "outputs/milestone6/runs"
rows = []
for f in sorted(glob.glob("outputs/milestone61/trackA/T1/*.csv")):
    T = pd.read_csv(f); stem = f.split("/")[-1]
    B = pd.read_csv(f"{R}/B1/{stem}").set_index("video_idx"); A = pd.read_csv(f"{R}/A1/{stem}").set_index("video_idx")
    X = pd.read_csv(f"{R}/A2x/{stem}").set_index("video_idx")
    for r in T.itertuples():
        rows.append({"set": r.set, "unit": r.unit, "video": r.video, "t": r.video_idx, "JF_T": r.JF, "JF_G": B.JF[r.video_idx],
                     "JF_L": A.JF[r.video_idx], "JF_X": X.JF[r.video_idx], "used": r.used_local, "lat": r.latency_s})
D = pd.DataFrame(rows)
D["or2"] = D[["JF_G", "JF_L"]].max(axis=1); D["or3"] = D[["JF_G", "JF_L", "JF_T"]].max(axis=1)
U = D.groupby(["set", "unit", "video"]).mean(numeric_only=True).reset_index(); U["small"] = U.unit.map(sm)
for c in ("JF_T", "JF_L", "JF_X", "or2", "or3"): U["d_" + c] = U[c] - U.JF_G
U.to_csv("outputs/milestone61/trackA/tplr_per_unit.csv", index=False)
G = U[U.set.isin(m6.GATE_SETS)]; g = G[G.small]
rep = {"units_applied": int(len(U)), "gate_small_units_applied": int(len(g)),
       "oracle_value_small_mean_over_applied": float((g.or3 - g.or2).mean()),
       "oracle_value_small_mean_over_15": float((g.or3 - g.or2).sum() / 15),
       "T1_small_dJF_over_15": float(g.d_JF_T.sum() / 15), "A2x_small_dJF_over_15": float(g.d_JF_X.sum() / 15),
       "T1_nonsmall_applied": float(G[~G.small].d_JF_T.mean()),
       "T1_up_down_small": [int((g.d_JF_T > 0.01).sum()), int((g.d_JF_T < -0.01).sum())],
       "all_sets_T1_vs_B1": {k: float(v.d_JF_T.mean()) for k, v in U.groupby("set")},
       "latency_local_median_s": float(D[D.used].lat.median())}
rep["adds_information"] = rep["oracle_value_small_mean_over_15"] >= 0.01
rep["deployable_success"] = rep["T1_small_dJF_over_15"] >= rep["A2x_small_dJF_over_15"] + 0.01 and rep["T1_nonsmall_applied"] >= -0.005
json.dump(rep, open("outputs/milestone61/trackA/tplr_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1)); print(U[["set", "unit", "small", "JF_G", "d_JF_T", "d_JF_L", "d_JF_X", "d_or3"]].round(4).to_string())
