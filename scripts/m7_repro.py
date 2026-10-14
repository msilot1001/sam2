"""Cross-machine reproducibility check of the M7 stage-0 recording path (functional check, NOT an experiment).

    python scripts/m7_repro.py run <tag> <unit,unit,...> [nofill]   # re-record units into outputs/milestone7/repro/<tag>/
    python scripts/m7_repro.py compare <tag>                       # compare with outputs/milestone7/rec (RTX 2080 originals)

Never writes into outputs/milestone7/rec. `nofill` sets predictor.fill_hole_area = 0, i.e. the behaviour of an
environment where sam2._C is missing (the RTX 2080 machine: hole filling is skipped with a warning).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_s = importlib.util.spec_from_file_location("m7", Path(__file__).with_name("m7.py"))
m7 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m7)
R = m7.OUT / "repro"


def _unit(name):
    from rfeas.units import davis_units
    return m7.m6._UNITS().get(name) or next(x for x in davis_units("train")[0] if x.name == name)


def run(tag, names, nofill=""):
    from memint import controlled
    d = R / tag
    d.mkdir(parents=True, exist_ok=True)
    pred = m7.m55.build_predictor()
    controlled.install(pred, None, {})
    if nofill:
        pred.fill_hole_area = 0
    try:
        import sam2._C  # noqa: F401
        has_c = True
    except Exception:
        has_c = False
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "sam2_C": has_c,
           "fill_hole_area": pred.fill_hole_area, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
           "tf32_cudnn": torch.backends.cudnn.allow_tf32}
    (d / "env.json").write_text(json.dumps(env, indent=1))
    for name in names.split(","):
        u = _unit(name)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        df, arr = m7.record_unit(pred, "DEV" if name in m7.m6._UNITS() else "REP", u)
        df["wall_s"] = time.time() - t0
        np.savez_compressed(d / f"{m7.stem(u)}.npz", **arr)
        df.to_csv(d / f"{m7.stem(u)}.csv", index=False)
        print(tag, name, len(df), round(df.JF.mean(), 4), f"{time.time() - t0:.0f}s", flush=True)


def compare(tag):
    d = R / tag
    rows = []
    for f in sorted(d.glob("*.csv")):
        a, b = pd.read_csv(m7.REC / f.name), pd.read_csv(f)
        za, zb = np.load(m7.REC / f.with_suffix(".npz").name), np.load(f.with_suffix(".npz"))
        n = min(len(a), len(b))
        r = {"unit": a.unit[0], "frames": n, "JF_ref": float(a.JF[:n].mean()), "JF_new": float(b.JF[:n].mean()),
             "max_abs_dJF_frame": float((a.JF[:n] - b.JF[:n]).abs().max()),
             "frames_JF_differs": int(((a.JF[:n] - b.JF[:n]).abs() > 1e-9).sum()),
             "first_diff_frame": int(a.video_idx[:n][(a.JF[:n] - b.JF[:n]).abs() > 1e-9].min()) if ((a.JF[:n] - b.JF[:n]).abs() > 1e-9).any() else None,
             "max_abs_dscore_G": float((a.score_G[:n] - b.score_G[:n]).abs().max()),
             "max_abs_dscore_L": float((a.score_L[:n] - b.score_L[:n]).abs().max()),
             "crop_decisions_differ": int((a.a2x_crop[:n] != b.a2x_crop[:n]).sum()),
             "boxes_differ": int((za["box"][:n] != zb["box"][:n]).any(1).sum())}
        for k in ("G", "L", "Lp", "L2"):
            x, y = za[k][:n].astype(np.float32), zb[k][:n].astype(np.float32)
            r[f"max_abs_dlogit_{k}"] = float(np.abs(x - y).max())
            r[f"sign_flips_{k}"] = int(((x > 0) != (y > 0)).sum())
        r["prod_latency_ref_s"] = float(a.lat_prod_s.median())
        r["prod_latency_new_s"] = float(b.lat_prod_s.median())
        r["peak_vram_ref_GB"] = float(a.peak_vram_bytes[0] / 2**30)
        r["peak_vram_new_GB"] = float(b.peak_vram_bytes[0] / 2**30)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(d / "compare.csv", index=False)
    print(out.T.to_string())


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
