"""M8 figures (GPU for the re-run of a few frames; GT is drawn only in the last, offline column).

    uv run python scripts/m8_viz.py events      # viz/30_rescue_events.png: generator evidence on co-absence frames
    uv run python scripts/m8_viz.py timelines   # viz/31_causal_timelines.png: per-frame J&F SGGL vs rescue arms
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

_s = importlib.util.spec_from_file_location("m8", Path(__file__).with_name("m8.py"))
m8 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m8)
R, sggl, vcam, motion = m8.R, m8.sggl, m8.vcam, m8.motion
VIZ = m8.OUT / "viz"
# (unit, frames) chosen from stage-1 tables: drone = rescue success, miami-surf = distractor false positive (GT empty)
EVENTS = [("davis__davis/drone__obj1", (30,)), ("davis__davis/miami-surf__obj6", None)]


def _img(frame):
    x = (frame[0].cpu() * R.STD + R.MEAN).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def events():
    pred = m8.m55.build_predictor()
    m8.controlled.install(pred, None, {})
    flow = motion.Flow()
    units = {u.name: (s, u) for s, x, u in m8.m7.units("DEV,REP")}
    rows = []
    for name, ts in EVENTS:
        s, u = units[name]
        if ts is None:                                  # first co-absence frame with GT empty where F_cpfr8_L fired
            c = pd.read_csv(next(p for d in ("diag", "runpod/diag") for p in [m8.OUT / d / f"{m8.m7.stem(u)}.cand.csv"] if p.exists()))
            c = c[(c.gen == "F_cpfr8_L") & (c.score > 0) & c.gt_empty]
            ts = (int(c.video_idx.iloc[0]),)
        g0 = m8.m7.gt(u, 0)
        hw = g0.shape
        ps = m8.m7.paths(u)
        d = sggl.Dual(pred, ps, g0)
        refs = m8.Refs(d, g0)
        trk = motion.Track(vcam.bbox(g0))
        prev = R.frame_tensor(ps[0]).cuda()
        for t in range(1, max(ts) + 1):
            G, L = d.view("G", t), d.view("L", t)
            out, ok = sggl.a2x(G, L, hw)
            frame = R.frame_tensor(ps[t]).cuda()
            trk.step_flow(flow(prev, frame), hw)
            if t in ts:
                with torch.inference_mode():
                    cands, maps, pk = m8.generators(pred, d, t, hw, frame, refs, trk, d._iou,
                                                    only={"A_cos#0", "B_att#0", "F_oclr8_L", "F_cpfr8_L"})
                H, W = hw
                fy, fx = trk.flow_center()
                a = R.occupancy_alpha(trk.size(), 8)
                views = {k: R.view_image(frame, R.axis_map(H, fy, a, k == "cpfr"), R.axis_map(W, fx, a, k == "cpfr"))
                         for k in ("oclr", "cpfr")}
                rows.append((u.name, t, _img(frame), {k: v.float().cpu().numpy() for k, v in maps.items()},
                             {n: (np.nan_to_num(lg, nan=-1) > 0, sc) for n, lg, sc, _, _ in cands},
                             {k: _img(v) for k, v in views.items()}, m8.m7.gt(u, t), G.score, L.score))
            trk.observe(t, vcam.bbox(out))
            d.set_center(vcam.center_of(out))
            prev = frame
        d.close()
    VIZ.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(len(rows), 7, figsize=(28, 4.2 * len(rows)))
    ax = np.atleast_2d(ax)
    for i, (name, t, img, maps, cands, views, g, sG, sL) in enumerate(rows):
        ax[i, 0].imshow(img)
        ax[i, 0].set_title(f"{name.split('__')[-2]}/{name.split('__')[-1]} t={t}\nSGGL: G score {sG:.1f}, L score {sL:.1f} (both absent)")
        for j, k in enumerate(("A_cos", "B_att")):
            ax[i, 1 + j].imshow(maps[k], cmap="magma")
            ax[i, 1 + j].set_title(f"{k} heatmap (stride 16)")
        for j, k in enumerate(("oclr", "cpfr")):
            ax[i, 3 + j].imshow(views[k])
            ax[i, 3 + j].set_title(f"F_{k}8 view (input to L memory)")
        ov = img.copy()
        for n, col in (("A_cos#0", (1, 0, 0)), ("F_oclr8_L", (0, 1, 0)), ("F_cpfr8_L", (0, 0, 1))):
            m, sc = cands[n]
            ov[m] = 0.5 * ov[m] + 0.5 * np.array(col)
        ax[i, 5].imshow(ov)
        ax[i, 5].set_title("candidates: red A_cos#0 (score %.1f)\ngreen OCLR8_L (%.1f), blue CPFR8_L (%.1f)" % (
            cands["A_cos#0"][1], cands["F_oclr8_L"][1], cands["F_cpfr8_L"][1]))
        gi = img.copy()
        gi[g] = 0.4 * gi[g] + 0.6 * np.array((1, 1, 0))
        ax[i, 6].imshow(gi)
        ax[i, 6].set_title("offline GT (analysis only)" + (" — EMPTY" if not g.any() else ""))
        for a_ in ax[i]:
            a_.axis("off")
    fig.tight_layout()
    fig.savefig(VIZ / "30_rescue_events.png", dpi=70)
    print("wrote", VIZ / "30_rescue_events.png")


def timelines(*arms):
    arms = arms or ("FLOWVEL", "M3vel", "RESC_OCLR")
    units = ["davis__davis_drone__obj1", "davis__davis_kite-surf__obj2", "davis__davis_skate-park__obj2", "davis__davis_scooter-board__obj2", "davis__davis_lab-coat__obj2"]
    fig, ax = plt.subplots(len(units), 1, figsize=(14, 3 * len(units)))
    for i, st in enumerate(units):
        b = pd.read_csv(m8.OUT / "runpod/runs/SGGL" / f"{st}.csv")
        ax[i].plot(b.video_idx, b.JF, "k-", lw=2, label="SGGL")
        for a in arms:
            f = m8.OUT / "runpod/runs" / a / f"{st}.csv"
            if f.exists():
                x = pd.read_csv(f)
                ax[i].plot(x.video_idx, x.JF, lw=1, label=a)
                r = x[x.rescued.astype(bool)]
                ax[i].scatter(r.video_idx, r.JF, s=12)
        ax[i].fill_between(b.video_idx, 0, 1, where=b.gt_empty, color="0.9", label="GT empty")
        ax[i].set_title(st)
        ax[i].legend(fontsize=7, ncol=5)
    fig.tight_layout()
    VIZ.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIZ / "31_causal_timelines.png", dpi=70)
    print("wrote", VIZ / "31_causal_timelines.png")


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
