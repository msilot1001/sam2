"""T family heads: learned residual correction of the LOCAL view's logits from frozen SAM 2 features (M7).

Variants (one class, flags):
  T0  no-transfer control: local features + local logits only
  T1  + aligned global features/logits (G sampled on the crop grid, sggl.sample_global_in_crop) -> simple CNN fusion
  T2  T1 + frame-0 target tokens: fixed token-similarity maps per view + token cross-attention FiLM (target_attention)
  T3  T2 + abstaining transfer: a learned per-cell gate alpha in [0,1] multiplies every global-derived channel
  T4  T3 + identity constraint: alpha additionally x sigmoid(a (cos_fg_G - cos_bg_G) + b) (G cells unlike the target
      transfer nothing)
All variants are exactly the identity (output == local logits) at initialisation (zero-init last layers; tested).
Inputs (batched, crop grid): memL/bbL Bx256x64x64, memG/bbG Bx256x32x32, lL/lG Bx1x256x256, sL/sG B, tok Bx4x256.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from spatial.target_attention import TokenAttention, token_maps

VARIANTS = ("T0", "T1", "T2", "T3", "T4")


def _block(cin, c, dil):
    return nn.Sequential(nn.Conv2d(cin, c, 3, padding=dil, dilation=dil), nn.GroupNorm(8, c), nn.GELU())


class FusionHead(nn.Module):
    def __init__(self, variant="T1", c=64):
        super().__init__()
        assert variant in VARIANTS
        self.v = variant
        self.useG = variant != "T0"
        self.tok = variant in ("T2", "T3", "T4")
        self.gate = variant in ("T3", "T4")
        self.idc = variant == "T4"
        self.pL = nn.Conv2d(256, 48, 1)
        cin = 48 + 1
        if self.useG:
            self.pG = nn.Conv2d(256, 48, 1)
            cin += 48 + 1
        if self.tok:
            self.att = TokenAttention(film_ch=c)
            cin += 4
        if self.gate:
            self.alpha = nn.Conv2d(4 + 1, 1, 3, padding=1)
            nn.init.zeros_(self.alpha.weight)
            nn.init.constant_(self.alpha.bias, 2.0)          # start mostly open (sigmoid(2) = 0.88)
        if self.idc:
            self.ida = nn.Parameter(torch.tensor(5.0))
            self.idb = nn.Parameter(torch.tensor(0.0))
        self.b1, self.b2, self.b3 = _block(cin, c, 1), _block(c, c, 2), _block(c, c, 4)
        self.out = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        lL = x["lL"]
        lL64 = Fn.adaptive_avg_pool2d(lL, 64) / 10
        feats = [self.pL(x["memL"].float()), lL64]
        alpha = None
        if self.useG:
            memG = Fn.interpolate(x["memG"].float(), size=64, mode="bilinear", align_corners=False)
            lG64 = Fn.adaptive_avg_pool2d(x["lG"], 64) / 10
            gch = [self.pG(memG), lG64]
            if self.tok:
                bbG = Fn.interpolate(x["bbG"].float(), size=64, mode="bilinear", align_corners=False)
                mL = token_maps(x["tok"][:, :2].float(), x["bbL"].float())
                mG = token_maps(x["tok"][:, 2:].float(), bbG)
            if self.gate:
                sG = x["sG"].float().view(-1, 1, 1, 1).expand(-1, 1, 64, 64) / 10
                alpha = torch.sigmoid(self.alpha(torch.cat([lG64, lG64.abs(), sG, mG], 1)))
                if self.idc:
                    alpha = alpha * torch.sigmoid(self.ida * (mG[:, :1] - mG[:, 1:2]) + self.idb)
                gch = [g * alpha for g in gch]
                mG = mG * alpha
            feats += gch
            if self.tok:
                feats += [mL, mG]
        h = self.b1(torch.cat(feats, 1))
        if self.tok:
            g, b, _, _ = self.att(x["tok"].float(), x["bbL"].float(), x["memL"].float(),
                                  Fn.interpolate(x["bbG"].float(), size=32), x["memG"].float())
            h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
        h = self.b3(self.b2(h))
        delta = Fn.interpolate(self.out(h) * 10, size=lL.shape[-2:], mode="bilinear", align_corners=False)
        self.last_alpha = alpha
        return lL + delta


def loss_fn(logit256, soft256):
    """Soft-target BCE + dice at the 256 logit grid (targets = area-resampled crop GT)."""
    bce = Fn.binary_cross_entropy_with_logits(logit256, soft256)
    p = torch.sigmoid(logit256)
    dice = 1 - (2 * (p * soft256).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + soft256.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
