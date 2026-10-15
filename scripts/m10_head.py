"""M10 amend10: learned selection head (offline, dev gate). Spec = outputs/milestone10/preregistration_amend10.json.

    uv run python scripts/m10_head.py smoke [dev_dir]                 # load + features + untrained forward; no fit, no gate
    uv run python scripts/m10_head.py fit <train_dir> [<train_dir2> ...]   # CV, tau*, refit, dev gate -> reports + head/
    uv run python scripts/m10_head.py selftest                        # synthetic data invariants

Data = OR0K rows (scripts/m10.py run_unit, extra="k"): one CSV + <stem>.tok.npz per unit. Training = LVOS v2 train units,
dev = outputs/milestone10/runpod/runs/OR0K. Dev rows are never used for fitting, standardisation or tau.
Training rows = hypotheses of frames in scope (out_area > 0), the rule's own domain (gt_empty frames kept: their cJ = 0
teaches the head not to switch there). Dev scores come from the refit on all training rows (saved under head/).
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

A = importlib.util.module_from_spec(s := importlib.util.spec_from_file_location("m10_analysis", Path(__file__).with_name("m10_analysis.py")))
s.loader.exec_module(A)

OUT = Path("outputs/milestone10")
HEAD = OUT / "head"
DEV_DIR = A.RUNS / "OR0K"
HYPER = {"opt": "Adam", "lr": 1e-3, "weight_decay": 1e-4, "epochs": 30, "batch": 512, "seed": 0, "loss": "MSE to cJ",
         "hidden": {"b": [64, 64], "c": [128, 128]}, "folds": 5, "fold_seed": 0, "null_shuffle_seed": 0, "bootstrap_seed": 10}
DEVICE = os.environ.get("M10_HEAD_DEVICE", "cpu")
SCALAR = ["piou", "is_L", "scale", "larea", "tiou", "cdist", "cdist_missing", "xiou", "cos", "cos_missing", "ptrcos", "stab", "mlogit", "vscore"]
MODELS = ("R_piou", "a", "b", "c", "null_b", "null_c")
CANDS = {"H1": ("a", "b", "c"), "H2": ("b", "c")}
NULL_OF = {"a": "null_b", "b": "null_b", "c": "null_c"}
RANK = {"a": 0, "b": 1, "c": 2}
PREC, MIN_SW, MOS_MIN, SIMPLER = 0.70, 20, -0.005, 0.005


# ----------------------------------------------------------------------------- data
def load_runs(dirs):
    """CSV rows + aligned tokens of every unit in dirs -> (df, tok [N,6,256], itok [N,2,256])."""
    dfs, toks, itoks = [], [], []
    for d in dirs:
        for f in sorted(Path(d).glob("*.csv")):
            if f.stat().st_size <= 1:
                continue
            x = pd.read_csv(f)
            z = np.load(f.with_name(f.stem + ".tok.npz"))
            assert len(x) == len(z["tok"]) == len(z["itok"]), f"{f}: rows {len(x)} tok {len(z['tok'])} itok {len(z['itok'])}"
            assert z["tok"].shape[1:] == (6, 256) and z["itok"].shape[1:] == (2, 256), f.name
            dfs.append(x)
            toks.append(z["tok"])
            itoks.append(z["itok"])
    assert dfs, f"no CSV in {dirs}"
    df = pd.concat(dfs, ignore_index=True)
    df["ds"] = df.set.str.split(":").str[0].str.lower()
    for c in ("gt_empty", "acc"):
        df[c] = df[c].astype(bool)
    df["state"] = A.state(df)
    df["active"] = df.out_area > 0
    return df, np.concatenate(toks).astype(np.float32), np.concatenate(itoks).astype(np.float32)


def feats(df):
    """Scalar inputs per hypothesis -> [N, 6, len(SCALAR)] (raw; standardised later with training statistics)."""
    n = len(df)
    X = np.zeros((n, 6, len(SCALAR)), np.float32)
    scale = (df.crop_side / df.frame_long).to_numpy(np.float32)
    for k in range(6):
        L = k >= 3
        cd, co = df[f"cdist{k}"].to_numpy(np.float32), df[f"cos{k}"].to_numpy(np.float32)
        cols = [df[f"piou{k}"], np.full(n, float(L)), scale if L else np.ones(n), np.log1p(df[f"harea{k}"]), df[f"tiou{k}"],
                np.nan_to_num(cd), np.isnan(cd), df[f"xiou{k}"], np.nan_to_num(co), np.isnan(co),
                df[f"ptrcos{k}"], df[f"stab{k}"], df[f"mlogit{k}"], df.score_L if L else df.score_G]
        X[:, k] = np.stack([np.asarray(c, np.float32) for c in cols], 1)
    assert np.isfinite(X).all(), "non-finite scalar feature"
    return X


def inputs(df, tok, itok, model):
    """Model input tensor [N, 6, F]: scalars (b) or scalars + mask token + the view's IoU token (c)."""
    X = feats(df)
    if model.endswith("c"):
        X = np.concatenate([X, tok, itok[:, [0, 0, 0, 1, 1, 1]]], 2)
    return X


def labels(df):
    return df[[f"cJ{k}" for k in range(6)]].to_numpy(np.float32)


def piou(df):
    return df[[f"piou{k}" for k in range(6)]].to_numpy(np.float32)


def shuffled(y, seed=HYPER["null_shuffle_seed"]):
    """Null labels: cJ permuted among the 6 hypotheses of each frame."""
    idx = np.random.default_rng(seed).permuted(np.tile(np.arange(6), (len(y), 1)), axis=1)
    return np.take_along_axis(y, idx, 1)


# ----------------------------------------------------------------------------- models
class Head:
    """Fitted scorer: 'a' = per-view affine on piou; 'b'/'c' = MLP on standardised inputs. score(X [N,6,F]) -> [N,6]."""

    def __init__(self, model, mu=None, sd=None, net=None, ab=None):
        self.model, self.mu, self.sd, self.net, self.ab = model, mu, sd, net, ab

    @staticmethod
    def fit(model, X, y, seed=HYPER["seed"]):
        if model == "a":
            ab = np.zeros((2, 2), np.float32)                        # [view, (alpha, beta)]
            for v, ks in enumerate((slice(0, 3), slice(3, 6))):
                p, t = X[:, ks, 0].ravel().astype(np.float64), y[:, ks].ravel().astype(np.float64)
                ab[v] = np.linalg.lstsq(np.c_[p, np.ones_like(p)], t, rcond=None)[0]
            return Head(model, ab=ab)
        F = X.shape[2]
        Z = X.reshape(-1, F)
        mu, sd = Z.mean(0), Z.std(0) + 1e-6
        h = HYPER["hidden"][model[-1]]
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(F, h[0]), nn.ReLU(), nn.Linear(h[0], h[1]), nn.ReLU(), nn.Linear(h[1], 1)).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=HYPER["lr"], weight_decay=HYPER["weight_decay"])
        Zt = torch.as_tensor((Z - mu) / sd, device=DEVICE)
        yt = torch.as_tensor(y.reshape(-1, 1), device=DEVICE)
        g = torch.Generator().manual_seed(seed)
        for _ in range(HYPER["epochs"]):
            for i in torch.randperm(len(Zt), generator=g).split(HYPER["batch"]):
                opt.zero_grad()
                nn.functional.mse_loss(net(Zt[i]), yt[i]).backward()
                opt.step()
        return Head(model, mu, sd, net.eval())

    def score(self, X):
        if self.model == "a":
            v = np.array([0, 0, 0, 1, 1, 1])
            return (self.ab[v, 0] * X[:, :, 0] + self.ab[v, 1]).astype(np.float32)
        Z = torch.as_tensor((X.reshape(-1, X.shape[2]) - self.mu) / self.sd, device=DEVICE)
        with torch.no_grad():
            return torch.cat([self.net(z) for z in Z.split(8192)]).cpu().numpy().reshape(X.shape[:2])

    def save(self, p):
        torch.save({"model": self.model, "mu": self.mu, "sd": self.sd, "ab": self.ab, "hyper": HYPER, "scalar": SCALAR,
                    "state_dict": None if self.net is None else self.net.state_dict()}, p)

    @staticmethod
    def load(p):
        d = torch.load(p, weights_only=False)
        h = Head(d["model"], d["mu"], d["sd"], ab=d["ab"])
        if d["state_dict"] is not None:
            F = d["state_dict"]["0.weight"].shape[1]
            hd = HYPER["hidden"][d["model"][-1]]
            h.net = nn.Sequential(nn.Linear(F, hd[0]), nn.ReLU(), nn.Linear(hd[0], hd[1]), nn.ReLU(), nn.Linear(hd[1], 1))
            h.net.load_state_dict(d["state_dict"])
            h.net.to(DEVICE).eval()
        return h


def score(rows_df, tok, itok, model="b"):
    """Frozen dev-time head for online arms (m10.py): rows_df = OR0K-style frame rows -> scores [N, 6]."""
    if model == "R_piou":
        return piou(rows_df)
    return Head.load(HEAD / f"{model}.pt").score(inputs(rows_df, tok, itok, model))


def folds(videos, seed=HYPER["fold_seed"], n=HYPER["folds"]):
    """Fold id per row, assigned by VIDEO (every video in exactly one fold)."""
    vids = np.random.default_rng(seed).permutation(np.array(sorted(set(videos))))
    return pd.Series(videos).map({v: i % n for i, v in enumerate(vids)}).to_numpy()


def cv_scores(model, X, y, fold):
    """Out-of-fold scores [N,6] (every row scored by the model that did not see its video)."""
    S = np.full(y.shape, np.nan, np.float32)
    for f in np.unique(fold):
        S[fold == f] = Head.fit(model, X[fold != f], y[fold != f]).score(X[fold == f])
    assert np.isfinite(S).all()
    return S


# ----------------------------------------------------------------------------- rules and gate
def choose(S, P, acc, hyp):
    """k_sggl (SAM 2 pick in the SGGL view), rule pick and score margin per frame. H1 = other view's own pick;
    H2 = argmax score within the SGGL view."""
    r = np.arange(len(S))
    v0 = np.where(acc, 3, 0)
    own = np.stack([P[r, v0 + i] for i in range(3)], 1)
    k_sg = v0 + own.argmax(1)
    if hyp == "H1":
        v1 = 3 - v0
        pick = v1 + np.stack([P[r, v1 + i] for i in range(3)], 1).argmax(1)
    else:
        pick = v0 + np.stack([S[r, v0 + i] for i in range(3)], 1).argmax(1)
    return k_sg, pick, S[r, pick] - S[r, k_sg]


def frame_table(df, S, hyp):
    """Per-frame choice table in the amend5 gate_eval layout (GT only scores the choice)."""
    F = df[["ds", "video", "unit", "video_idx", "gt_empty", "acc", "active", "state"]].copy().reset_index(drop=True)
    Jc = labels(df)
    k_sg, pick, margin = choose(S, piou(df), F.acc.to_numpy(), hyp)
    r = np.arange(len(F))
    F["k_sggl"], F["pick"], F["margin"] = k_sg, pick, margin
    F["J_sggl"], F["J_pick"] = Jc[r, k_sg], Jc[r, pick]
    return F


def curve(F, grid, pooled=None):
    """Switch count / precision at every tau on grid (pooled over the given datasets, default all rows)."""
    d = F[F.active & (F.pick != F.k_sggl)]
    if pooled is not None:
        d = d[d.ds.isin(pooled)]
    out = []
    for tau in grid:
        sw = d[d.margin >= tau]
        out.append({"tau": float(tau), "switches": int(len(sw)), "precision": float((sw.J_pick > sw.J_sggl).mean()) if len(sw) else None})
    return out


def tau_star(F_oof):
    """Grid {0} + quantiles 0.05..0.95 of positive OOF margins; smallest tau with OOF precision >= 0.70 and >= 20 switches."""
    d = F_oof[F_oof.active & (F_oof.pick != F_oof.k_sggl)]
    pos = d.margin[d.margin > 0].to_numpy()
    grid = [0.0] + (sorted(np.quantile(pos, np.arange(0.05, 1.0, 0.05)).tolist()) if len(pos) else [])
    c = curve(F_oof, grid)
    ok = [x for x in c if x["precision"] is not None and x["precision"] >= PREC and x["switches"] >= MIN_SW]
    return (ok[0]["tau"] if ok else None), grid, c


def pooled_precision(F, tau, dsets=("mosev2", "vost")):
    sw = F[F.active & (F.pick != F.k_sggl) & (F.margin >= tau) & F.ds.isin(dsets)]
    return (float((sw.J_pick > sw.J_sggl).mean()) if len(sw) else None), int(len(sw))


def gate_one(F_oof, F_dev, rng):
    """tau* from training OOF; dev evaluation (amend5 gate_eval per dataset + pooled MOSEv2/VOST precision + curves)."""
    ts, grid, c_tr = tau_star(F_oof)
    r = {"tau_star": ts, "grid": grid, "curve_train_oof": c_tr, "curve_dev": curve(F_dev, grid, ("mosev2", "vost")),
         "dev_at_tau0": A.gate_eval(F_dev, 0.0, rng)}
    if ts is not None:
        r["train_oof_precision"] = next(x["precision"] for x in c_tr if x["tau"] == ts)
        r["dev"] = A.gate_eval(F_dev, ts, rng)
        r["dev_precision_pooled"], r["dev_switches_pooled"] = pooled_precision(F_dev, ts)
    return r


def passes(r, rn):
    """Criteria (1)-(4) of the dev gate; rn = the null head's result (no tau* -> null makes no switches: dJ 0, precision 0)."""
    if r.get("tau_star") is None:
        return False, {"tau_star": False}
    e = r["dev"]
    nd = {ds: 0.0 for ds in A.DSETS}
    npr = 0.0
    if rn.get("tau_star") is not None:
        nd = {ds: rn["dev"][ds]["dJ_present"] for ds in A.DSETS if ds in rn["dev"]}
        npr = rn["dev_precision_pooled"] or 0.0
    c = {"tau_star": True,
         "1_pooled_precision": (r["dev_precision_pooled"] or 0.0) >= PREC,
         "2_ci_lower_vost_lvos": all(ds in e and e[ds]["ci"][0] > 0 for ds in ("vost", "lvosv2")),
         "3_mosev2_point": "mosev2" in e and e["mosev2"]["dJ_present"] >= MOS_MIN,
         "4_beats_null": all(ds in e and e[ds]["dJ_present"] > nd.get(ds, 0.0) for ds in ("vost", "lvosv2"))
                         and (r["dev_precision_pooled"] or 0.0) > npr}
    return all(c.values()), c


def winner(res, hyp):
    """Larger VOST+LVOS dJ sum; the simpler model unless the larger one is ahead by > 0.005 on both."""
    ps = [m for m in CANDS[hyp] if res[m][hyp]["pass"]]
    if not ps:
        return None
    dj = lambda m, ds: res[m][hyp]["dev"][ds]["dJ_present"]
    tot = lambda m: dj(m, "vost") + dj(m, "lvosv2")
    best = max(ps, key=tot)
    simpler = [m for m in ps if RANK[m] < RANK[best]]
    if simpler:
        alt = max(simpler, key=tot)
        if not all(dj(best, ds) - dj(alt, ds) > SIMPLER for ds in ("vost", "lvosv2")):
            best = alt
    return best


# ----------------------------------------------------------------------------- pipeline
def run(tr, dev, save_dir=None):
    """Fit every model (CV -> OOF scores; refit -> dev scores), gate H1/H2. tr/dev = (df, tok, itok)."""
    dtr, dde = tr[0], dev[0]
    scope = dtr.active.to_numpy()
    fold = folds(dtr.video.to_numpy())
    y = labels(dtr)
    y_null = shuffled(y)
    rng = np.random.default_rng(HYPER["bootstrap_seed"])
    res, oof, dev_s, fit_info = {}, {}, {}, {}
    for m in MODELS:
        if m == "R_piou":
            oof[m], dev_s[m] = piou(dtr), piou(dde)
        else:
            X, Xd = inputs(*tr, m), inputs(*dev, m)
            yy = y_null if m.startswith("null") else y
            S = np.full(y.shape, np.nan, np.float32)
            S[scope] = cv_scores(m, X[scope], yy[scope], fold[scope])
            S[~scope] = 0.0                                   # out of scope: never used by any rule
            h = Head.fit(m, X[scope], yy[scope])
            oof[m], dev_s[m] = S, h.score(Xd)
            fit_info[m] = {"oof_mse_in_scope": float(((S[scope] - yy[scope]) ** 2).mean()),
                           "refit_mse_in_scope": float(((h.score(X[scope]) - yy[scope]) ** 2).mean()),
                           **({"alpha_beta_GL": h.ab.tolist()} if m == "a" else {})}
            if save_dir:
                h.save(Path(save_dir) / f"{m}.pt")
        res[m] = {}
        for hyp in ("H1", "H2"):
            r = gate_one(frame_table(dtr, oof[m], hyp), frame_table(dde, dev_s[m], hyp), rng)
            res[m][hyp] = r
            print(m, hyp, "tau*", None if r["tau_star"] is None else round(r["tau_star"], 4),
                  {ds: (round(x["dJ_present"], 4), [round(c, 4) for c in x["ci"]], x["switches"], x["precision"] and round(x["precision"], 3))
                   for ds, x in r.get("dev", {}).items()}, flush=True)
    for m in MODELS:
        for hyp in ("H1", "H2"):
            r = res[m][hyp]
            if m in CANDS[hyp]:
                r["pass"], r["criteria"] = passes(r, res[NULL_OF[m]][hyp])
            else:
                r["pass"] = False
    chosen = {hyp: winner(res, hyp) for hyp in CANDS}
    print("passing", {h: [m for m in CANDS[h] if res[m][h]["pass"]] for h in CANDS}, "chosen", chosen)
    if save_dir:
        np.savez_compressed(Path(save_dir) / "scores.npz", dev_unit=dde.unit.to_numpy(), dev_video_idx=dde.video_idx.to_numpy(),
                            train_unit=dtr.unit.to_numpy(), train_video_idx=dtr.video_idx.to_numpy(), train_fold=fold,
                            **{f"dev_{m}": dev_s[m] for m in MODELS}, **{f"oof_{m}": oof[m] for m in MODELS})
    return {"models": res, "chosen": chosen, "fit": fit_info, "folds": {str(v): int(f) for v, f in zip(dtr.video, fold)}}


def counts(df):
    return {"frames": len(df), "units": int(df.unit.nunique()), "videos": int(df.video.nunique()), "in_scope": int(df.active.sum()),
            "gt_present": int((~df.gt_empty).sum()), "per_ds": {ds: int(g.unit.nunique()) for ds, g in df.groupby("ds")}}


def fit(*train_dirs):
    assert train_dirs, "fit <train_dir> [...]"
    tr, dev = load_runs(train_dirs), load_runs([DEV_DIR])
    assert not set(tr[0].video) & set(dev[0].video), "training/dev videos overlap"
    HEAD.mkdir(parents=True, exist_ok=True)
    r = run(tr, dev, HEAD)
    out = {"amend": "preregistration_amend10.json", "hyper": HYPER, "device": DEVICE, "scalar_inputs": SCALAR,
           "training_rows": "hypotheses of frames in scope (out_area > 0); gt_empty frames kept",
           "null_without_tau_star": "treated as no switches (dJ 0, precision 0) in criterion (4)",
           "train": {"dirs": [str(d) for d in train_dirs], **counts(tr[0])}, "dev": {"dir": str(DEV_DIR), **counts(dev[0])}, **r}
    A.save(out, "head_offline.json")
    md(out)
    return out


def md(res):
    f3 = lambda x: "—" if x is None else f"{x:+.3f}"
    names = {"mosev2": "MOSEv2", "vost": "VOST", "lvosv2": "LVOS v2"}
    L = ["# M10 amend10: 학습 선택 헤드 오프라인 게이트 (dev)", "",
         f"- 학습: {res['train']['dirs']} ({res['train']['units']} 단위 / {res['train']['videos']} 영상 / {res['train']['frames']} 프레임, scope {res['train']['in_scope']}). "
         f"영상 단위 5-fold OOF 점수로만 tau* 선택. dev 점수 = 전체 학습 행 재적합 모델.",
         f"- dev: {res['dev']['per_ds']} 단위. dJ = 전환 프레임에서 cJ(pick) − cJ(k_sggl), 그 외 0; GT 존재 프레임 평균, 단위→영상→영상 동일 가중; 영상 bootstrap 95% CI (10k).",
         "- 통과 = (1) MOSEv2+VOST 합산 정밀도 ≥ 0.70, (2) VOST·LVOS CI 하한 > 0, (3) MOSEv2 점추정 ≥ −0.005, (4) null 헤드보다 우세. tau* 없음 = 실패.",
         f"- 하이퍼파라미터 고정: {res['hyper']}", ""]
    for hyp, title in (("H1", "H1: 뷰 보정 (G vs L, 각 뷰의 SAM 2 선택 사이 전환)"), ("H2", "H2: 뷰 내 선택 (SGGL 출력 뷰의 3 가설)")):
        L += [f"## {title}", "", "| 모델 | tau* | 학습 OOF 정밀도 | dev 합산 정밀도 (MOSEv2+VOST) | MOSEv2 dJ [CI] | VOST dJ [CI] | LVOS v2 dJ [CI] | 전환 수 (M/V/L) | 통과 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            r = res["models"][m][hyp]
            if r["tau_star"] is None:
                L.append(f"| {m} | — | — | — | — | — | — | — | {'실패 (tau* 없음)' if m in CANDS[hyp] else '—'} |")
                continue
            e = r["dev"]
            cell = lambda ds: f"{f3(e[ds]['dJ_present'])} [{f3(e[ds]['ci'][0])}, {f3(e[ds]['ci'][1])}]" if ds in e else "—"
            sw = "/".join(str(e[ds]["switches"]) if ds in e else "—" for ds in A.DSETS)
            st = ("통과" if r["pass"] else "실패") if m in CANDS[hyp] else "(비교용)"
            L.append(f"| {m} | {r['tau_star']:.3f} | {r['train_oof_precision']:.3f} | {'—' if r['dev_precision_pooled'] is None else f"{r['dev_precision_pooled']:.3f}"} "
                     f"| {cell('mosev2')} | {cell('vost')} | {cell('lvosv2')} | {sw} | {st} |")
        L += ["", f"선택: **{res['chosen'][hyp] or '없음'}**", ""]
    L += ["## 이탈/정상 프레임 dJ (tau*)", "", "| 모델 | 가설 | 데이터셋 | 이탈 프레임 dJ | 정상 프레임 dJ | 정밀도 |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        for hyp in ("H1", "H2"):
            for ds, e in res["models"][m][hyp].get("dev", {}).items():
                L.append(f"| {m} | {hyp} | {names[ds]} | {f3(e['dJ_drift_frames'])} | {f3(e['dJ_correct_frames'])} | {f3(e['precision'])} |")
    L += ["", "정밀도–tau 곡선 (학습 OOF, dev)과 fold 배정은 outputs/milestone10/reports/head_offline.json. 재적합 모델: outputs/milestone10/head/<model>.pt."]
    Path("docs/m10/HEAD_RESULTS.md").write_text("\n".join(L) + "\n")


def smoke(dev_dir=str(DEV_DIR)):
    """Load + feature build + shape asserts + one untrained forward pass. No fitting, no gate."""
    df, tok, itok = load_runs([dev_dir])
    print(counts(df), "tok", tok.shape, "itok", itok.shape)
    for m in ("b", "c"):
        X = inputs(df, tok, itok, m)
        assert X.shape == (len(df), 6, len(SCALAR) + (512 if m == "c" else 0)), X.shape
        h = Head(m, X.reshape(-1, X.shape[2]).mean(0), X.reshape(-1, X.shape[2]).std(0) + 1e-6)
        F, hd = X.shape[2], HYPER["hidden"][m]
        torch.manual_seed(0)
        h.net = nn.Sequential(nn.Linear(F, hd[0]), nn.ReLU(), nn.Linear(hd[0], hd[1]), nn.ReLU(), nn.Linear(hd[1], 1)).to(DEVICE).eval()
        S = h.score(X)
        assert S.shape == (len(df), 6) and np.isfinite(S).all()
        print(m, "features", X.shape, "scores", S.shape)
    for hyp in ("H1", "H2"):
        F = frame_table(df, piou(df), hyp)
        print(hyp, "R_piou disagreements in scope", int((F.active & (F.pick != F.k_sggl)).sum()))


# ----------------------------------------------------------------------------- selftest
def synth(n_videos, frames, rng, set_="LVOSV2:train", prefix="v"):
    rows, n = [], n_videos * frames
    for v in range(n_videos):
        for t in range(1, frames + 1):
            rows.append({"set": set_, "unit": f"{prefix}{v}__obj1", "video": f"{prefix}{v}", "video_idx": t})
    df = pd.DataFrame(rows)
    df["acc"] = rng.random(n) < 0.5
    df["out_area"] = rng.integers(0, 50, n)
    df["gt_empty"] = rng.random(n) < 0.1
    df["J"] = rng.random(n)
    df["out_inter_gt"], df["out_inter_other"] = rng.integers(0, 20, n), rng.integers(0, 5, n)
    df["score_G"], df["score_L"] = rng.normal(5, 3, n), rng.normal(5, 3, n)
    df["crop_side"], df["frame_long"] = rng.integers(100, 500, n), 1000
    for k in range(6):
        df[f"piou{k}"] = rng.random(n)
        df[f"harea{k}"] = rng.integers(0, 5000, n)
        for f in ("tiou", "xiou", "ptrcos", "stab"):
            df[f"{f}{k}"] = rng.random(n)
        df[f"cdist{k}"] = np.where(rng.random(n) < 0.1, np.nan, rng.random(n))
        df[f"cos{k}"] = np.where(rng.random(n) < 0.1, np.nan, rng.random(n))
        df[f"mlogit{k}"] = rng.normal(0, 3, n)
        df[f"cJ{k}"] = np.clip(0.5 * df[f"piou{k}"] + rng.normal(0, 0.2, n), 0, 1)
        df[f"vJ{k}"] = df[f"cJ{k}"]
    df["cJ6"] = 0.0
    df["ds"] = df.set.str.split(":").str[0].str.lower()
    df["state"], df["active"] = A.state(df), df.out_area > 0
    return df, rng.normal(0, 1, (n, 6, 256)).astype(np.float32), rng.normal(0, 1, (n, 2, 256)).astype(np.float32)


def selftest():
    rng = np.random.default_rng(1)
    tr = synth(10, 40, rng)
    fold = folds(tr[0].video.to_numpy())
    assert (pd.Series(fold).groupby(tr[0].video.to_numpy()).nunique() == 1).all(), "a video spans folds"
    assert len(np.unique(fold)) == HYPER["folds"]
    F = frame_table(tr[0], piou(tr[0]), "H2")
    assert int((F.active & (F.pick != F.k_sggl)).sum()) == 0, "H2 with R_piou must never switch"
    devs = [synth(4, 30, np.random.default_rng(s), "VOST:dev", f"d{s}") for s in (2, 3)]
    for d in devs:                                            # three datasets so the gate code paths run
        d[0]["ds"] = np.array(A.DSETS)[np.arange(len(d[0])) % 3]
    ts = [{(m, h): r[h]["tau_star"] for m, r in run(tr, d)["models"].items() for h in ("H1", "H2")} for d in devs]
    assert ts[0] == ts[1], "tau* depends on dev rows"
    assert ts[0][("R_piou", "H2")] is None
    print("selftest ok", ts[0])


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
