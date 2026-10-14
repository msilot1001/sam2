"""M9 cheap-motion check: does phase correlation reproduce the RAFT camera-motion estimate FLOWVEL/CAMCARRY used?

    M9_DATA=<root> uv run python scripts/m9_motion_agree.py <dataset> [max_units]
For consecutive frame pairs of each unit: RAFT-small camera motion (median flow outside 2x the GT box of frame t-1;
GT is used only to exclude the target for this offline estimator comparison) vs global phase correlation.
Writes outputs/milestone9/reports/motion_agree_<dataset>.csv (dx/dy of both, endpoint error, latency of both).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rfeas import xdata  # noqa: E402
from spatial import motion, rescue as R, vcam  # noqa: E402


def main(ds, max_units="40"):
    flow = motion.Flow()
    rows = []
    for _, u in xdata.units(ds)[: int(max_units)]:
        ps = xdata.paths(u)
        prev = R.frame_tensor(ps[0]).cuda()
        for t in range(1, len(ps)):
            cur = R.frame_tensor(ps[t]).cuda()
            g = xdata.gt(u, t - 1)
            box = vcam.bbox(g)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fl = flow(prev, cur)
            cam = motion.camera(fl, box)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            pc = motion.phase_shift(prev, cur)
            t2 = time.perf_counter()
            rows.append({"unit": u.name, "video": u.video, "t": t, "raft_dx": cam[0], "raft_dy": cam[1],
                         "pc_dx": pc[0], "pc_dy": pc[1], "epe": float(np.hypot(*(cam - pc))),
                         "raft_mag": float(np.hypot(*cam)), "lat_raft": t1 - t0, "lat_pc": t2 - t1})
            prev = cur
        print(u.name, len(rows), flush=True)
    out = Path("outputs/milestone9/reports")
    out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / f"motion_agree_{ds}.csv", index=False)
    print(d[["epe", "raft_mag", "lat_raft", "lat_pc"]].describe().round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
