"""M8 rescue-candidate rankers (fit on REP only; causal features only).

  R1  train-free : accept iff score > 0, then rank by appearance margin cos_fg - cos_bg
  R2  logistic   : per-candidate logistic regression on scalar features (independent scoring)
  R3  attention  : set ranker. Candidate tokens self-attend (competition between proposals) and cross-attend to the
                   frame-0 fg/bg reference tokens (identity conditioning); an extra learned "none" token lets it abstain.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

SCALARS = ["score", "iou", "cos_fg", "cos_bg", "margin", "log_area", "dist_flow"]


def scalars(df):
    x = df.assign(margin=df.cos_fg - df.cos_bg, log_area=np.log(df.area_ratio.clip(1e-3, 1e3)),
                  dist_flow=df.dist_flow.fillna(5.0).clip(0, 5), score=df.score.clip(-20, 20),
                  cos_fg=df.cos_fg.fillna(0), cos_bg=df.cos_bg.fillna(0), iou=df.iou.fillna(0))[SCALARS]
    x["margin"] = x["margin"].fillna(0)
    x["log_area"] = x["log_area"].fillna(np.log(1e-3))
    return x.to_numpy(np.float32)


def r1(df):
    return np.where(df.score.to_numpy() > 0, (df.cos_fg - df.cos_bg).fillna(-1).to_numpy(), -np.inf)


class LR:
    """L2-regularised logistic regression (torch LBFGS), standardised inputs."""

    def fit(self, X, y, l2=1e-2):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = torch.from_numpy((X - self.mu) / self.sd)
        w = torch.zeros(X.shape[1] + 1, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200)
        yt = torch.from_numpy(y.astype(np.float32))

        def closure():
            opt.zero_grad()
            loss = Fn.binary_cross_entropy_with_logits(Z @ w[:-1] + w[-1], yt) + l2 * (w[:-1] ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.w = w.detach().numpy()
        return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.w[:-1] + self.w[-1]

    def state(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist()}

    @classmethod
    def load(cls, s):
        o = cls()
        o.mu, o.sd, o.w = (np.asarray(s[k], np.float32) for k in ("mu", "sd", "w"))
        return o


class SetRanker(nn.Module):
    """R3. Inputs: cand B x K x (256 + S) (pooled feature ++ standardised scalars), gen ids B x K, ref B x R x 256 (frame-0
    fg + bg tokens), ref_type B x R (0 fg / 1 bg), pad mask B x K. Output: B x (K + 1) logits (last = none)."""

    def __init__(self, n_gen, s=len(SCALARS), d=64, heads=4):
        super().__init__()
        self.inp = nn.Linear(256 + s, d)
        self.gen = nn.Embedding(n_gen, d)
        self.ref = nn.Linear(256, d)
        self.rtype = nn.Embedding(2, d)
        self.none = nn.Parameter(torch.zeros(1, 1, d))
        self.self_att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def forward(self, cand, gen, ref, rtype, pad):
        x = self.inp(cand) + self.gen(gen)
        x = torch.cat([x, self.none.expand(len(x), 1, -1)], 1)
        pad = torch.cat([pad, torch.zeros(len(x), 1, dtype=torch.bool, device=pad.device)], 1)
        h = self.n1(x)
        x = x + self.self_att(h, h, h, key_padding_mask=pad)[0]
        r = self.ref(ref) + self.rtype(rtype)
        x = x + self.cross(self.n2(x), r, r)[0]
        x = x + self.ff(x)
        return self.out(x)[..., 0].masked_fill(pad, -1e4)
