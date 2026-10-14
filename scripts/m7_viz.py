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
    main()
