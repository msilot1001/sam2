"""Causal memory-write gates (CHECK 2). gate(s_t, history) -> True means 'skip writing frame t'.

Inputs are only the frame-t signals and earlier signals; no GT, no future frames. Thresholds are
the pre-registered values in configs/rfeas.yaml.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .risk import FEATURES, Logistic, causal_features


def presence_gate(s, hist):
    return s["obj_score"] < 0.0


def iou_gate(s, hist):
    return s["pred_iou"] < 0.7


def consistency_gate(s, hist):
    c = s["consistency_iou_prev"]
    return bool(np.isfinite(c) and c < 0.5)


def load_risk_model(path: Path):
    d = json.loads(Path(path).read_text())
    return {"model": Logistic.from_dict(d["model"]), "threshold": d["threshold"]}


def risk_scores(model, signals: pd.DataFrame) -> np.ndarray:
    f = causal_features(signals)
    return model["model"].predict(f[FEATURES].to_numpy())


def make_risk_gate(model):
    def gate(s, hist):
        df = pd.DataFrame(hist + [s])
        return bool(risk_scores(model, df)[-1] >= model["threshold"])
    return gate


def random_gate(fraction: float, key: str):
    def gate(s, hist):
        u = int.from_bytes(hashlib.sha256(f"{key}|{s['video_idx']}".encode()).digest()[:8], "big") / 2**64
        return u < fraction
    return gate


def skip_exactly(t: int):
    return lambda s, hist: s["video_idx"] == t
