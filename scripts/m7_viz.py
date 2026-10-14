"""M7 event figures from the stage-0 recordings (CPU). Successes AND failures; GT drawn only for offline inspection.

    uv run python scripts/m7_viz.py            # writes outputs/milestone7/viz/*.png
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

_s = importlib.util.spec_from_file_location("r", Path(__file__).with_name("m7_replay.py"))
r = importlib.util.module_from_spec(_s)
_s.loader.exec_module(r)
m7 = r.m7
from spatial import policies  # noqa: E402

# (unit, frame, why) -- chosen from the audit categories; includes failures of the incumbent and of candidates
EVENTS = [("davis__davis/miami-surf__obj5", 30, "false local re-acquisition of a look-alike; G also fires at t=31"),
          ("davis__davis/miami-surf__obj5", 40, "SGGL tracks the distractor; SGGL-V vetoes (G absent)"),
          ("davis__davis/dog-gooses__obj5", 82, "correct local re-acquisition without G support (veto cost)"),
          ("davis__davis/drone__obj1", 40, "both views wrong: target left the crop, G absent"),
          ("davis__davis/kite-surf__obj2", 30, "G absent from tracking state (G-protected finds it)"),
          ("davis__davis/skate-park__obj2", 40, "both absent; 2x crop state L2 finds it"),
          ("davis__davis/lab-coat__obj1", 20, "complementary masks (blend > both)"),
          ("davis__davis/dog-gooses__obj5", 40, "fusion failure: G spreads into background")]
POL = {"SGGL": policies.Policy, "SGGL-V": policies.SGGLV, "F1-blend.7": policies.F1BlendL, "F3-asym": policies.F3Asym}


def _unit(name):
    from rfeas.units import davis_units
    return m7.m6._UNITS().get(name) or next(x for x in davis_units("train")[0] if x.name == name)


def overlay(ax, rgb, m, color, title, box=None):
    ax.imshow(rgb)
    if m is not None and m.any():
        c = np.zeros((*m.shape, 4))
        c[m] = (*color, 0.55)
        ax.imshow(c)
    if box is not None:
        y0, y1, x0, x1 = box
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "y-", lw=1)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def figure(unit, t_show, why, out):
    df = pd.read_csv(m7.REC / f"{unit.replace('/', '_')}.csv")
    Z = np.load(m7.REC / f"{unit.replace('/', '_')}.npz")
    u = _unit(unit)
    g0 = m7.gt(u, 0)
    pols = {k: P(g0) for k, P in POL.items()}
    for o in r.observations(df, Z, g0.shape):
        res = {k: p(o) for k, p in pols.items()}
        if o["t"] == t_show:
            break
    row = df[df.video_idx == t_show].iloc[0]
    rgb = np.asarray(Image.open(m7.paths(u)[t_show]).convert("RGB"))
    rgb0 = np.asarray(Image.open(m7.paths(u)[0]).convert("RGB"))
    g = m7.gt(u, t_show)
    y0, y1, x0, x1 = o["box"]
    pad = (y1 - y0) // 2
    Y0, Y1, X0, X1 = max(0, y0 - pad), min(g.shape[0], y1 + pad), max(0, x0 - pad), min(g.shape[1], x1 + pad)
    cr = lambda a: a[Y0:Y1, X0:X1]
    bx = (y0 - Y0, y1 - Y0, x0 - X0, x1 - X0)
    sg, f1 = res["SGGL"][0], res["F1-blend.7"][0]
    panels = [(rgb0, g0, (0, 1, 0), "frame 0 prompt (reference)", None, False),
              (cr(rgb), cr(o["mG"]), (0, 0.4, 1), f"G  s={o['sG']:.2f}", bx, True),
              (cr(rgb), cr(o["mL"]), (1, 0.5, 0), f"L  s={o['sL']:.2f} touch={o['touchL']}", bx, True),
              (cr(rgb), cr(o["mL2"]), (0.6, 0, 1), f"L2 (2x)  s={o['sL2']:.2f}", bx, True),
              (cr(rgb), cr(o["mLp"]), (1, 0, 1), f"L protected-mem s={o['sLp']:.2f}", bx, True),
              (cr(rgb), cr(sg), (1, 0, 0), f"SGGL J={row.J:.2f}", bx, True),
              (cr(rgb), cr(res["SGGL-V"][0]), (1, 0, 0), "SGGL-V", bx, True),
              (cr(rgb), cr(f1 & ~sg) | cr(sg & ~f1), (1, 1, 0), "F1-blend.7 edits (yellow)", bx, True),
              (cr(rgb), cr(g), (0, 1, 0), "GT (offline only)", bx, True)]
    fig, axs = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.6))
    for ax, (im, m, c, ttl, b, _) in zip(axs, panels):
        overlay(ax, im, m, c, ttl, b)
    fig.suptitle(f"{unit} t={t_show}: {why}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def timelines():
    """Per-frame J&F of SGGL vs M3-vel (both causal, Vast, machine-matched) for the largest gains and the worst loss."""
    d = m7.OUT / "viz"
    d.mkdir(parents=True, exist_ok=True)
    us = ["davis__davis/skate-park__obj2", "davis__davis/sheep__obj4", "davis__davis/lab-coat__obj2"]
    fig, axs = plt.subplots(len(us), 1, figsize=(8, 2.2 * len(us)), sharex=False)
    for ax, u in zip(axs, us):
        st = u.replace("/", "_")
        a = pd.read_csv(m7.OUT / "runs/SGGL" / f"{st}.csv")
        b = pd.read_csv(m7.OUT / "runs/M3-vel" / f"{st}.csv")
        ax.plot(a.video_idx, a.JF, "k-", lw=1, label=f"SGGL {a.JF.mean():.3f}")
        ax.plot(b.video_idx, b.JF, "r-", lw=1, label=f"M3-vel {b.JF.mean():.3f}")
        ax.plot(a.video_idx, a.JF_G, color="0.6", lw=0.8, label=f"P5 {a.JF_G.mean():.3f}")
        ax.set_title(u, fontsize=8)
        ax.set_ylabel("J&F")
        ax.legend(fontsize=7, loc="lower left")
    axs[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(d / "20_M3vel_timelines.png", dpi=110)
    plt.close(fig)
    print("ok timelines")


def vk_example():
    """One VKITTI frame where SGGL differs most from P5 (largest positive) and the worst VK unit (SGGL < P5)."""
    from vkitti.dataset import track_mask  # noqa: F401
    d = m7.OUT / "viz"
    R = pd.concat([pd.read_csv(f) for f in sorted((m7.OUT / "rec_vk").glob("*.csv"))], ignore_index=True)
    R["d"] = R.JF - R.JF_G
    picks = [R.loc[R.d.idxmax()], R.loc[R[R.unit == "vkitti__vkitti/Scene01__obj57"].d.idxmin()]]
    from rfeas.units import vkitti_units
    us = {u.name: u for u in vkitti_units()[0]}
    fig, axs = plt.subplots(len(picks), 3, figsize=(12, 2.4 * len(picks)))
    for row, p in zip(axs, picks):
        u, t = us[p.unit], int(p.video_idx)
        Z = np.load(m7.OUT / "rec_vk" / f"{p.unit.replace('/', '_')}.npz")
        df = pd.read_csv(m7.OUT / "rec_vk" / f"{p.unit.replace('/', '_')}.csv")
        i = int(np.nonzero(df.video_idx.to_numpy() == t)[0][0])
        g = m7.gt(u, t)
        o = next(x for k, x in enumerate(r.observations(df.iloc[i:i + 1].reset_index(drop=True),
                                                         {k: (v[i:i + 1] if v.ndim and len(v) == len(df) else v) for k, v in Z.items()}, g.shape)))
        rgb = np.asarray(Image.open(m7.paths(u)[t]).convert("RGB"))
        sg = o["mL"] if p.a2x_crop else o["mG"]
        overlay(row[0], rgb, o["mG"], (0, 0.4, 1), f"P5 / G  J&F={p.JF_G:.2f}", o["box"])
        overlay(row[1], rgb, sg, (1, 0, 0), f"SGGL ({'crop' if p.a2x_crop else 'global'})  J&F={p.JF:.2f}", o["box"])
        overlay(row[2], rgb, g, (0, 1, 0), "GT (offline only)", o["box"])
        row[0].set_ylabel(f"{p.unit.split('__', 1)[1]} t={t}", fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "21_VK_examples.png", dpi=110)
    plt.close(fig)
    print("ok vk")


def main():
    d = m7.OUT / "viz"
    d.mkdir(parents=True, exist_ok=True)
    for i, (u, t, why) in enumerate(EVENTS):
        try:
            figure(u, t, why, d / f"{i:02d}_{u.split('__')[1].replace('/', '_')}_{u.split('__')[2]}_t{t}.png")
            print("ok", u, t)
        except Exception as e:                       # a figure must never abort the others
            print("skip", u, t, e)


if __name__ == "__main__":
    globals()[sys.argv[1]]() if len(sys.argv) > 1 else main()
