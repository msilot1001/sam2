"""Round-5 analysis: A2v (cross-view presence veto) vs A2x on dev (21 crop units) and DAVIS-train replication (43)."""
import glob, importlib.util, json
from pathlib import Path
import pandas as pd
from baseline.dropout_runner import atomic_csv, atomic_json
_s = importlib.util.spec_from_file_location("m6", Path(__file__).with_name("m6.py")); m6 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m6)
O = Path("outputs/milestone61")
sm = {r["unit"]: r["small"] for r in json.load(open("outputs/milestone6/preregistration.json"))["rows"]}
rep = {}
def load(pat):
    fs = sorted(glob.glob(pat))
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else None
# dev
V = load(str(O / "trackA/A2v/*.csv"))
if V is not None:
    X = pd.concat([pd.read_csv(f) for f in glob.glob("outputs/milestone6/runs/A2x/*.csv")])
    B = pd.concat([pd.read_csv(f) for f in glob.glob("outputs/milestone6/runs/B1/*.csv")])
    U = V.groupby(["set", "unit", "video"]).JF.mean().rename("A2v").reset_index()
    U = U.merge(X.groupby("unit").JF.mean().rename("A2x"), on="unit").merge(B.groupby("unit").JF.mean().rename("B1"), on="unit")
    U["small"] = U.unit.map(sm); U["d_vs_A2x"] = U.A2v - U.A2x; U["d"] = U.A2v - U.B1
    atomic_csv(U, O / "trackA/a2v_dev_per_unit.csv")
    g = U[U.set.isin(m6.GATE_SETS) & U.small]
    rep["dev"] = {"units": int(len(U)), "small_gate_applied": int(len(g)), "A2v_small_over15": float(g.d.sum() / 15),
                  "A2x_small_over15": float((g.A2x - g.B1).sum() / 15), "A2v_minus_A2x_small_over15": float(g.d_vs_A2x.sum() / 15),
                  "CI_applied": m6._boot(g, "d_vs_A2x"),
                  "units_changed_0.005": U[U.d_vs_A2x.abs() > 0.005][["unit", "d_vs_A2x"]].round(4).to_dict("records")}
    rep["dev"]["noninferior"] = rep["dev"]["A2v_minus_A2x_small_over15"] >= -0.003
# replication
R = load(str(O / "replication/A2v/*.csv"))
if R is not None:
    P = pd.read_csv(O / "replication/replication_per_unit.csv")
    U2 = P.merge(R.groupby("unit").JF.mean().rename("JF_A2v"), on="unit", how="inner")
    U2["d_A2v"] = U2.JF_A2v - U2.JF_B1; U2["A2v_minus_A2d"] = U2.JF_A2v - U2.JF_A2d
    atomic_csv(U2, O / "replication/a2v_per_unit.csv")
    s = U2[U2.small]
    rep["replication"] = {"units": int(len(U2)), "small_n": int(len(s)), "videos": int(s.video.nunique()),
                          "A2v_small": float(s.d_A2v.mean()), "CI": m6._boot(s, "d_A2v"),
                          "A2d_small": float(s.d_A2d.mean()), "A2v_minus_A2d": float(s.A2v_minus_A2d.mean()),
                          "CI_diff": m6._boot(s, "A2v_minus_A2d"),
                          "up_down": [int((s.d_A2v > 0.01).sum()), int((s.d_A2v < -0.01).sum())],
                          "miami_surf_obj5": float(U2[U2.unit.str.contains("miami-surf__obj5")].d_A2v.iloc[0]) if U2.unit.str.contains("miami-surf__obj5").any() else None,
                          "worse_than_A2d_0.01": U2[U2.A2v_minus_A2d < -0.01][["unit", "A2v_minus_A2d"]].round(4).to_dict("records")}
    rep["replication"]["catastrophe_removed"] = rep["replication"]["miami_surf_obj5"] is not None and rep["replication"]["miami_surf_obj5"] >= -0.05
atomic_json(rep, O / "veto_report.json")
print(json.dumps(rep, indent=1, default=float))
