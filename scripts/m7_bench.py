"""Throughput profiling of the M7 causal rollout (engineering only; outputs are discarded, nothing under runs/).

    python scripts/m7_bench.py profile <unit> [n_frames]     # per-stage wall time of run_policy (SGGL)
Thread count is taken from OMP_NUM_THREADS / torch defaults of the calling environment.
"""
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

_s = importlib.util.spec_from_file_location("m7", Path(__file__).with_name("m7.py"))
m7 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m7)


def _wrap(obj, name, acc, key):
    orig = getattr(obj, name)

    def f(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            acc[key] += time.perf_counter() - t0
    setattr(obj, name, f)
    return orig


def profile(unit, n="40"):
    from memint import controlled, engine
    from spatial import policies, sggl, vcam
    pred = m7.m55.build_predictor()
    controlled.install(pred, None, {})
    u = m7.m6._UNITS()[unit]
    m7.run_policy(pred, "DEV", u, policies.Policy, n=4)                    # warm-up (cudnn, caches)
    acc = defaultdict(float)
    _wrap(engine, "predict", acc, "sam2_predict(gpu+sync)")
    _wrap(vcam.LazyCropFrames, "__getitem__", acc, "jpeg_decode_crop_resize")
    _wrap(m7.m55, "jf", acc, "J/F_eval")
    _wrap(m7, "gt", acc, "gt_load")
    _wrap(sggl, "full_to_frame", acc, "upsample_G_frame_res")
    _wrap(sggl, "crop_to_frame", acc, "upsample_L_frame_res")
    _wrap(policies.Policy, "__call__", acc, "policy(+identity)")
    _wrap(m7, "_cos_map", acc, "identity_cos_map")
    t0, c0 = time.perf_counter(), time.process_time()
    df = m7.run_policy(pred, "DEV", u, policies.Policy, n=int(n))
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    acc = dict(acc)
    # engine.predict internally reads frames (decode) -> report decode separately and subtract from predict
    acc["sam2_predict_excl_decode"] = acc["sam2_predict(gpu+sync)"] - acc["jpeg_decode_crop_resize"]
    out = {"unit": unit, "frames": len(df), "threads": torch.get_num_threads(), "wall_s": wall, "proc_cpu_s": cpu,
           "s_per_frame": wall / len(df), "stages_s": {k: round(v, 3) for k, v in acc.items()},
           "JF_mean": float(df.JF.mean())}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
