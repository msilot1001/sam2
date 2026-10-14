"""GT-only per-frame geometry for the M9 failure taxonomy (run where the dataset lives).

    M9_DATA=<root> uv run python scripts/m9_gtbox.py <dataset>   -> outputs/milestone9/gtbox/<dataset>.csv
Columns: unit, video_idx, gt_long (GT bbox long side, px), prompt_area (px), crop_side (SGGL fixed crop side, px).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rfeas import xdata  # noqa: E402
from spatial import vcam  # noqa: E402


def main(ds):
    rows = []
    for _, u in xdata.units(ds):
        g0 = xdata.gt(u, 0)
        side = vcam.crop_side(g0)
        pa = int(g0.sum())
        for t in range(1, len(u.frames)):
            g = xdata.gt(u, t)
            ys, xs = np.nonzero(g)
            gl = int(max(np.ptp(ys), np.ptp(xs)) + 1) if len(ys) else 0
            rows.append({"unit": u.name, "video_idx": t, "gt_long": gl, "prompt_area": pa, "crop_side": side})
    out = Path("outputs/milestone9/gtbox")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{ds}.csv", index=False)
    print(ds, len(rows))


if __name__ == "__main__":
    main(sys.argv[1])
