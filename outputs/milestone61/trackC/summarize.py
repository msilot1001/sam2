import json, pandas as pd, numpy as np
E = pd.read_csv("events.csv"); C = pd.read_csv("carryover_oracle.csv"); FA = pd.read_csv("frames.csv")
MAIN = ["DAVIS_val", "PUMaVOS_A", "PUMaVOS_B"]
vis = FA[FA.set.isin(MAIN) & ~FA.gt_empty]
TOT = (1 - vis.J).sum(); TOTx = (1 - vis.J[~vis.artifact]).sum()
print(f"total err main visible: {TOT:.1f} over {len(vis)} frames (non-artifact {TOTx:.1f})")
def summ(e):
    return pd.Series(dict(events=len(e), units=e.unit.nunique(), videos=e.video.nunique(),
        frames=sum(len(json.loads(x)) for x in e.frames), err_vis=e.err_vis.sum(), err_empty=e.err_empty.sum(),
        share=e[e.set.isin(MAIN)].err_vis.sum() / TOT))
for scope, Es in [("ALL sets", E), ("MAIN", E[E.set.isin(MAIN)])]:
    print("==", scope); print(E.groupby(["cat"]).apply(lambda d: d).pipe(lambda _: Es.groupby(["cat", "persistent"]).apply(summ)).round(3).to_string())
    print(Es.groupby("cat").apply(summ).round(3).to_string())
for c, d in E.groupby("cat"):
    print("--", c); print(d.sort_values("err_vis", ascending=False).head(8)[["unit", "start", "end", "length", "persistent", "err_vis", "err_empty"]].to_string())
def union_share(cats, main=True):
    fr = set()
    for r in E[E.cat.isin(cats)].itertuples():
        fr |= {(r.unit, t) for t in json.loads(r.frames)}
    k = vis.set_index(["unit", "t"]); m = k.index.isin(list(fr))
    return (1 - k.J[m]).sum() / TOT, m.sum()
print("union all cats", union_share(list(E.cat.unique())))
for c in E.cat.unique(): print("union", c, union_share([c]))
print("== carry-over"); 
for scope, d in [("ALL", C), ("MAIN", C[C.set.isin(MAIN)]), ("MAIN visible", C[C.set.isin(MAIN) & ~C.gt_empty])]:
    print(scope, len(d), "mean gain %.3f median %.3f" % (d.gain.mean(), d.gain.median()), "gain>=.15:", (d.gain >= .15).sum(),
          "videos(gain>=.15):", d[d.gain >= .15].video.nunique(), "units", d[d.gain >= .15].unit.nunique(),
          "sum pos gain %.2f" % d.gain.clip(lower=0).sum(), "neg:", (d.gain < 0).sum())
print(C[C.gain >= .15].groupby("video").size().sort_values(ascending=False).to_string())
print(C.sort_values("gain", ascending=False).head(10).round(3).to_string())
