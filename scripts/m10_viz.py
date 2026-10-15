"""M10 failure-episode figures (CPU, PIL only). Needs `m10.py rerun` outputs (masks) and the frames.

    M9_DATA=<root> python scripts/m10_viz.py <viz_dir> <out_dir> <tag> <unit> <arm> [<arm> ...]

One PNG per unit: rows = GT, then each arm; columns = 8 frames around the first SGGL (OR0) loss episode.
Overlay: arm mask red, GT outline green, L crop box yellow. Caption: frame, J&F, output source, memory writes
(G/L admitted: upper case = written, '-' = blocked, '*' = memory re-encoded). Bottom strip: per-frame J&F curves.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

_s = importlib.util.spec_from_file_location("m7", Path(__file__).with_name("m7.py"))
m7 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m7)
W = 300
COLS = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


def unit_by_name(name):
    for s, sp, u in m7.units(name.split("__")[0].upper()):
        if u.name == name:
            return u
    raise KeyError(name)


def outline(g):
    e = g.copy()
    e[1:-1, 1:-1] &= ~(g[:-2, 1:-1] & g[2:, 1:-1] & g[1:-1, :-2] & g[1:-1, 2:])
    return e


def frames_of(df):
    """8 frames: 2 before the first loss onset (J < 0.5 with GT present), the onset, then spread to the end."""
    bad = (~df.gt_empty.astype(bool)) & (df.J < 0.5)
    i0 = int(np.argmax(bad.to_numpy())) if bad.any() else 0
    idx = sorted(set([max(0, i0 - 2), max(0, i0 - 1), i0] + np.linspace(i0 + 1, len(df) - 1, 5).astype(int).tolist()))
    return [i for i in idx if i < len(df)][:8]


def cell(img, m, g, box, cap):
    a = np.array(img).astype(np.float32)
    if m is not None:
        a[m] = 0.45 * a[m] + 0.55 * np.array([230, 40, 40])
    a[outline(g)] = [40, 230, 40]
    im = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(im)
    if box is not None:
        y0, x0, s = box
        d.rectangle([x0, y0, x0 + s, y0 + s], outline=(255, 220, 0), width=max(2, im.width // 300))
    h = int(im.height * W / im.width)
    im = im.resize((W, h))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 14], fill=(0, 0, 0))
    d.text((3, 1), cap, fill=(255, 255, 255))
    return im


def main(vdir, odir, tag, name, *arms):
    vdir, odir = Path(vdir), Path(odir)
    u = unit_by_name(name)
    stem = m7.stem(u)
    rows = {a: pd.read_csv(vdir / a / f"{stem}.csv") for a in arms}
    masks = {a: np.load(vdir / a / f"{stem}.npz") for a in arms}
    hw = tuple(masks[arms[0]]["hw"])
    pick = frames_of(rows[arms[0]])
    paths = m7.paths(u)
    tiles = []
    for r, a in enumerate(("GT",) + arms):
        line = []
        for i in pick:
            t = i + 1                                             # rows start at video_idx 1
            img = Image.open(paths[t]).convert("RGB")
            g = m7.gt(u, t)
            if a == "GT":
                line.append(cell(img, g, g, None, f"t={t} GT area={int(g.sum())}"))
                continue
            x = rows[a].iloc[i]
            m = np.unpackbits(masks[a]["out"][i])[: hw[0] * hw[1]].reshape(hw).astype(bool)
            wr = "".join((v if x[f"adm_{v}"] else "-") + ("*" if x[f"rew_{v}"] else "") for v in "GL")
            line.append(cell(img, m, g, (int(x.crop_y0), int(x.crop_x0), int(x.crop_side)),
                             f"{a} JF={x.JF:.2f} src={x.src} mem={wr}"))
        tiles.append(line)
    h = max(t.height for line in tiles for t in line)
    canvas = Image.new("RGB", (W * len(pick), h * len(tiles) + 160), (255, 255, 255))
    for r, line in enumerate(tiles):
        for c, t in enumerate(line):
            canvas.paste(t, (c * W, r * h))
    d = ImageDraw.Draw(canvas)
    y0, Wt = h * len(tiles) + 10, W * len(pick)
    d.rectangle([0, y0, Wt - 1, y0 + 140], outline=(180, 180, 180))
    for k, a in enumerate(arms):
        jf = rows[a].JF.to_numpy()
        xs = np.linspace(5, Wt - 5, len(jf))
        d.line(list(zip(xs, y0 + 135 - 130 * jf)), fill=COLS[k % len(COLS)], width=2)
        d.text((8 + 110 * k, y0 + 2), a, fill=COLS[k % len(COLS)])
    for i in pick:
        x = 5 + (Wt - 10) * i / max(len(rows[arms[0]]) - 1, 1)
        d.line([(x, y0), (x, y0 + 140)], fill=(200, 200, 200))
    out = odir / tag / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
