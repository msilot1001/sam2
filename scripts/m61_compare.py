"""M6.1 common benchmark: B0 official / B1 P5 / B2 A2x / B3 A2c / Track-B head (output-only) on every dev set.

Unit = object; per-unit means over all evaluated frames (prompt frame excluded); deltas paired vs B1; CIs = video-
clustered bootstrap. Sources: M5.5 proto csv (B0 on DAVIS val / SegTrack), M5.6 runs (B0 on PUMaVOS), M6 runs (B1, A2x,
A2c: units without crop are identical to B1 by construction), M6.1 trackB/video (head).
"""
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.dropout_runner import atomic_csv, atomic_json

_s = importlib.util.spec_from_file_location("m6", Path(__file__).with_name("m6.py"))
m6 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m6)
OUT = Path("outputs/milestone61")
M = ["J", "F", "JF", "F_strict"]


def frames():
    B1 = m6._runs("B1")[["set", "unit", "video", "video_idx", *M]].assign(arm="B1")
    sets = dict(zip(B1.unit, B1.set))
    old = [pd.read_csv(f).query("arm == 'official'") for d in ("val", "segtrack") for f in sorted(glob.glob(f"outputs/milestone55/proto/{d}/*.csv"))]
    old += [pd.read_csv(f).query("arm == 'B0_official'") for f in sorted(glob.glob("outputs/milestone56/runs/*.csv"))]
    B0 = pd.concat(old)[["unit", "video_idx", *M]].assign(arm="B0")
    B0 = B0[B0.unit.isin(sets)]
    parts = [B1, B0.assign(set=B0.unit.map(sets), video=B0.unit.map(dict(zip(B1.unit, B1.video))))]
    for arm in ("A2x", "A2c"):
        X = m6._runs(arm)[["unit", "video_idx", *M]]
        Y = B1.drop(columns=M).merge(X, on=["unit", "video_idx"], how="left")
        for c in M:
            Y[c] = Y[c].fillna(pd.Series(B1[c].to_numpy(), index=Y.index))
        parts.append(Y.assign(arm=arm))
    hv = sorted(glob.glob(str(OUT / "trackB/video/*.csv")))
    if hv:
        H = pd.concat([pd.read_csv(f) for f in hv])
        H = H.rename(columns={f"{c}_head": c for c in M})[["set", "unit", "video", "video_idx", *M]].assign(arm="HEAD")
        parts.append(H)
    return pd.concat(parts, ignore_index=True)


def main():
    F = frames()
    af = {}
    for f in ("outputs/milestone55/manifest.json", "outputs/milestone56/manifest.json", "outputs/milestone55/manifest_segtrack.json"):
        af.update({r["unit"]: r["area_frac"] for r in json.loads(Path(f).read_text())["rows"]})
    F["late"] = F.video_idx >= 21
    U = F.groupby(["set", "unit", "video", "arm"])[M].mean().reset_index()
    L = F[F.late].groupby(["unit", "arm"]).JF.mean().rename("JF21").reset_index()
    U = U.merge(L, on=["unit", "arm"], how="left")
    W = U.pivot_table(index=["set", "unit", "video"], columns="arm", values=M + ["JF21"]).reset_index()
    W.columns = ["_".join(c).strip("_") for c in W.columns]
    W["area_frac"] = W.unit.map(af)
    W["size"] = np.where(W.area_frac < 0.001, "tiny<0.1%", np.where(W.area_frac < 0.01, "small0.1-1%", "ordinary"))
    arms = [a for a in ("B0", "A2x", "A2c", "HEAD") if f"JF_{a}" in W]
    for a in arms:
        for c in M + ["JF21"]:
            W[f"d{c}_{a}"] = W[f"{c}_{a}"] - W[f"{c}_B1"]
    atomic_csv(W, OUT / "comparison_per_unit.csv")
    rep = {}
    groups = {"gate_set(DAVIS_val+PUMaVOS_B)": W[W.set.isin(m6.GATE_SETS)], "all_dev": W}
    groups.update({s: g for s, g in W.groupby("set")})
    for gname, g in groups.items():
        rep[gname] = {"units": int(len(g)), "videos": int(g.video.nunique()), "JF_B1": float(g.JF_B1.mean())}
        for a in arms:
            sub = g.dropna(subset=[f"dJF_{a}"])
            rep[gname][a] = {"units": int(len(sub)), "dJF": float(sub[f"dJF_{a}"].mean()), "CI": m6._boot(sub, f"dJF_{a}"),
                             "dJ": float(sub[f"dJ_{a}"].mean()), "dF": float(sub[f"dF_{a}"].mean()),
                             "dF_strict": float(sub[f"dF_strict_{a}"].mean()), "dJF21": float(sub[f"dJF21_{a}"].mean()),
                             "up_0.01": int((sub[f"dJF_{a}"] > 0.01).sum()), "down_0.01": int((sub[f"dJF_{a}"] < -0.01).sum())}
            for sz, h in sub.groupby("size"):
                rep[gname][a][f"size:{sz}"] = {"units": int(len(h)), "dJF": float(h[f"dJF_{a}"].mean())}
    atomic_json(rep, OUT / "comparison_report.json")
    print(json.dumps(rep, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
