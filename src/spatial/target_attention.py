"""T2-T4: target-token cross-attention over separately processed global/local SAM 2 features (M7).

Reference tokens come ONLY from frame 0 (prompt mask): fg/bg mask-pooled stride-16 backbone features in the local
view (fg_L, bg_L) and in the global view (fg_G, bg_G). At frame t the tokens query each view separately:
    attn_v = softmax_x( (Wq tok) . (Wk bb_v(x)) / sqrt(d) )       v in {L, G}, bounded: K=4 tokens x 64*64 (or 32*32) keys
    read_v = sum_x attn_v(x) Wv mem_v(x)
Evidence: cos-similarity maps of the fixed tokens to each view's backbone features (fg - bg contrast), plus the
attention readouts. Readouts produce FiLM (gamma, beta) for the fused local feature map; the maps are extra input
channels. View identity is kept by separate projections; geometry by sampling G on the crop grid (sggl).
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class TokenAttention(nn.Module):
    def __init__(self, c=256, d=64, n_tok=4, film_ch=64):
        super().__init__()
        self.q = nn.Linear(c, d)
        self.kL, self.kG = nn.Conv2d(c, d, 1), nn.Conv2d(c, d, 1)
        self.vL, self.vG = nn.Conv2d(c, d, 1), nn.Conv2d(c, d, 1)
        self.film = nn.Sequential(nn.Linear(2 * n_tok * d + n_tok * d, 128), nn.GELU(), nn.Linear(128, 2 * film_ch))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.d = d

    def read(self, q, k, v):
        att = torch.softmax(torch.einsum("bnd,bdx->bnx", q, k.flatten(2)) / self.d ** 0.5, dim=-1)
        return torch.einsum("bnx,bdx->bnd", att, v.flatten(2)), att

    def forward(self, tok, bbL, memL, bbG, memG):
        """tok B x K x C (fixed frame-0 tokens); bb*/mem* B x C x h x w. Returns gamma, beta (B x film_ch), attn maps."""
        q = self.q(tok)
        rL, aL = self.read(q, self.kL(bbL), self.vL(memL))
        rG, aG = self.read(q, self.kG(bbG), self.vG(memG))
        z = torch.cat([rL.flatten(1), rG.flatten(1), (rL - rG).flatten(1)], 1)
        g, b = self.film(z).chunk(2, 1)
        return g, b, aL, aG


def token_maps(tok, bb):
    """Fixed cosine maps of each token to each cell (B x K x h x w)."""
    return torch.einsum("bkc,bchw->bkhw", Fn.normalize(tok, dim=-1), Fn.normalize(bb, dim=1))
