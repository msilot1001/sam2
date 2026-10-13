"""Track C (M6.1): retrospective GT-based temporal-error analysis of B1. CPU only, read-only on inputs."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage

ROOT = Path("/home/sojak/sam2"); OUT = ROOT / "outputs/milestone61/trackC"
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("m6", ROOT / "scripts/m6.py"); m6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m6)
jm = m6.j_measure
GUITAR, PIMPLES = "pumavos__pumavos/guitar__obj1", "pumavos__pumavos/pimples__obj1"
MAIN = {"DAVIS_val", "PUMaVOS_A", "PUMaVOS_B"}
K = 3  # transient = resolves within <=3 frames


def artifact(unit, t, gt_empty):
    return (unit == PIMPLES and t >= 74) or (unit == GUITAR and gt_empty)


def runs(flags):
    """[(start, end_exclusive)] of consecutive True."""
    out, s = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and s is None: s = i
        if not f and s is not None: out.append((s, i)); s = None
    return out


ev, carry, drift, frames_all = [], [], [], []
for f in sorted((ROOT / "outputs/milestone6/runs/B1").glob("*.csv")):
    g = pd.read_csv(f).sort_values("video_idx").reset_index(drop=True)
    s, unit, video = g.set[0], g.unit[0], g.video[0]
    u = m6._UNITS()[unit]
    n = len(u.frames); G = [m6.gt(u, t) for t in range(n)]; H, W = G[0].shape
    P = np.unpackbits(np.load(f.with_suffix(".npz"))["B1"])[:(n - 1) * H * W].reshape(n - 1, H, W).astype(bool)
    P = np.concatenate([G[:1], P])  # index = video_idx; t=0 is the prompt (GT) frame
    assert list(g.video_idx) == list(range(1, n))
    J = np.r_[1.0, g.J.values]; empty = np.r_[False, g.gt_empty.values]
    art = np.array([artifact(unit, t, empty[t]) for t in range(n)])
    pa, ga = P.reshape(n, -1).sum(1), np.array([x.sum() for x in G])
    for t in range(1, n):
        frames_all.append(dict(set=s, unit=unit, video=video, t=t, J=J[t], gt_empty=empty[t], artifact=art[t]))

    def add(cat, t0, t1, persistent, **kw):
        fr = [t for t in range(t0, t1) if not art[t]]
        if not fr: return
        ev.append(dict(set=s, unit=unit, video=video, cat=cat, start=t0, end=t1 - 1, length=t1 - t0,
                       persistent=persistent, err_vis=float(sum(1 - J[t] for t in fr if not empty[t])),
                       err_empty=float(sum(1 - J[t] for t in fr if empty[t])), frames=json.dumps(fr), **kw))

    # (a) component disappearance despite continued visibility
    lab = [ndimage.label(x) for x in G]
    t = 1
    while t < n:
        hit = None
        if not art[t] and not art[t - 1]:
            Lp, kp = lab[t - 1]; Lc, kc = lab[t]
            for c in range(1, kc + 1):
                comp = Lc == c
                if comp.sum() < 50: continue
                ov = np.bincount(Lp[comp], minlength=kp + 1); ov[0] = 0
                if ov.max() == 0: continue
                prev = Lp == ov.argmax()
                if prev.sum() < 50: continue
                if (P[t - 1] & prev).sum() / prev.sum() >= 0.5 and (P[t] & comp).sum() / comp.sum() < 0.1:
                    hit = comp; break
        if hit is None: t += 1; continue
        comp_gain = [jm(P[t] | hit, G[t]) - J[t]]   # J gain if the lost component alone were restored
        cur, k = hit, t + 1   # follow the GT component until it is >=50% covered again or leaves GT
        rec = None
        while k < n:
            Lk, _ = lab[k]; ov = np.bincount(Lk[cur], minlength=Lk.max() + 1); ov[0] = 0
            if ov.max() == 0: rec = "gt_gone"; break
            cur = Lk == ov.argmax()
            if (P[k] & cur).sum() / cur.sum() >= 0.5: rec = "recovered"; break
            comp_gain.append(jm(P[k] | cur, G[k]) - J[k])
            k += 1
        add("a_component_loss", t, k, persistent=not (rec and k - t <= K), resolution=rec or "video_end",
            comp_px=int(hit.sum()), comp_gain=float(sum(g_ for j, g_ in zip(range(t, k), comp_gain) if not art[j])))
        t = k if k > t else t + 1
    # (b) sudden collapse: pred area drops >50% while GT area changes <20%, and J drops
    t = 2
    while t < n:
        if (pa[t - 1] > 0 and ga[t - 1] > 0 and ga[t] > 0 and pa[t] < 0.5 * pa[t - 1]
                and abs(ga[t] / ga[t - 1] - 1) < 0.2 and J[t] < J[t - 1] and not art[t]):
            ref = J[t - 1]; k = t
            while k < n and J[k] < ref - 0.1: k += 1
            k = max(k, t + 1)
            add("b_collapse", t, k, persistent=k - t > K, J_before=ref, J_at=J[t], resolution="recovered" if k < n else "video_end")
            t = k
        else: t += 1
    # (c) wrong-object capture
    for a, b in runs(np.r_[False, g.other_obj_frac.values >= 0.3]):
        add("c_wrong_object", a, b, persistent=b - a > K)
    # (c2) capture of an unannotated region: >=30% of predicted pixels in components disjoint from GT (GT visible)
    def disj(t):
        L, k = ndimage.label(P[t]); hitl = np.unique(L[G[t]])
        return (np.isin(L, hitl, invert=True) & P[t]).sum() / pa[t]
    dj = np.array([False] + [bool(ga[t] > 0 and pa[t] > 0 and disj(t) >= 0.3) for t in range(1, n)])
    for a, b in runs(dj):
        add("c2_disjoint_region", a, b, persistent=b - a > K)
    # (d) false presence when GT empty
    for a, b in runs(empty & (pa > 0) & ~art):
        add("d_false_presence", a, b, persistent=b - a > K)
    # (f) drift
    v = [t for t in range(1, n) if not empty[t] and not art[t]]
    e, l = [t for t in v if t <= 20], [t for t in v if t > 20]
    if len(e) >= 10 and len(l) >= 10:
        je, jl = J[e].mean(), J[l].mean()
        drift.append(dict(set=s, unit=unit, video=video, J_1_20=je, J_21p=jl, drop=je - jl, n_late=len(l),
                          err_late=float((1 - J[l]).sum()), excess_err_late=float(np.clip(je - J[l], 0, None).sum())))
        if je - jl > 0.1:
            add("f_drift", 21, n, persistent=True, J_1_20=je, J_21p=jl)
    # Task 2: carry-over oracle
    for t in range(2, n):
        if J[t] < J[t - 1] - 0.2 and not art[t] and not art[t - 1]:
            jc = jm(P[t - 1], G[t])
            carry.append(dict(set=s, unit=unit, video=video, t=t, gt_empty=empty[t], J_prev=J[t - 1], J_t=J[t],
                              J_carry=jc, gain=jc - J[t]))
    print(unit, flush=True)

# (e) failed reappearance from M6 table: failed = J_first < 0.5; transient = delay <= 3
R = pd.read_csv(ROOT / "outputs/milestone6/reports/reappear_events.csv")
FA = pd.DataFrame(frames_all)
for r in R.itertuples():
    if r.type == "annotation_defined" or (r.unit == PIMPLES and r.reappear >= 74) or r.J_first >= 0.5: continue
    end = r.reappear + (int(r.delay) if pd.notna(r.delay) else 10**6)
    q = FA[(FA.unit == r.unit) & (FA.t >= r.reappear) & (FA.t < end) & ~FA.artifact]
    ev.append(dict(set=r.set, unit=r.unit, video=r.video, cat="e_failed_reappear", start=r.reappear,
                   end=int(q.t.max()), length=len(q), persistent=not (pd.notna(r.delay) and r.delay <= K),
                   err_vis=float((1 - q.J[~q.gt_empty]).sum()), err_empty=float((1 - q.J[q.gt_empty]).sum()),
                   frames=json.dumps(q.t.tolist()), resolution=r.type))

E = pd.DataFrame(ev); C = pd.DataFrame(carry)
E.to_csv(OUT / "events.csv", index=False); C.to_csv(OUT / "carryover_oracle.csv", index=False)
pd.DataFrame(drift).to_csv(OUT / "drift_per_unit.csv", index=False); FA.to_csv(OUT / "frames.csv", index=False)
