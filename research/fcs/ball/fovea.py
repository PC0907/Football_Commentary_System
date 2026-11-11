"""Fovea network: finds the ball in a native-resolution crop around the tracker's prediction.

Input (C_in = 12 at full configuration), crop S×S (S = 160):
  9  RGB of frames t-2, t-1, t           (causal temporal stack — TrackNet idea)
  2  |I_t − I_{t-1}|, |I_t − I_{t-2}|     (grey; motion / magnocellular channel)
  1  prior heatmap N(prediction, σ_pred)  (the tracker's Bayesian prediction as an input)
Outputs at stride 2 (80×80): heatmap logit, sub-pixel offset (2), box size (2, log px), blur axis (2):
  b = |v| · (cos 2θ, sin 2θ) — a streak has an orientation but no sign; the tracker resolves the sign.

Architectures
  'unet'  TrackNet-like encoder–decoder CNN.
  'kpa'   Kalman-Prior-Attention transformer: conv stem to stride 8 → tokens; self-attention logits get an additive
          bias  β_h · log N(p_j | μ_pred, σ_pred²)  (prior over KEY positions: every query is nudged to look where the
          ball is expected — Bayesian prior inside attention) and  γ_h · m_j  (motion energy of key token j — magno
          gating). Decoded back to stride 2 by a light CNN head.
Ablation flags: frames (1 or 3), use_diff, use_prior.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

S = 160
STRIDE = 2


def conv(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True))


class Heads(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.hm = nn.Conv2d(c, 1, 1); self.off = nn.Conv2d(c, 2, 1); self.size = nn.Conv2d(c, 2, 1); self.blur = nn.Conv2d(c, 2, 1)
        nn.init.constant_(self.hm.bias, -4.0)

    def forward(self, x):
        return dict(hm=self.hm(x), off=self.off(x), size=self.size(x), blur=self.blur(x))


class FoveaUNet(nn.Module):
    def __init__(self, c_in=12, w=(32, 64, 128, 192)):
        super().__init__()
        self.e0 = nn.Sequential(conv(c_in, w[0]), conv(w[0], w[0]))            # 160
        self.e1 = nn.Sequential(conv(w[0], w[1], 2), conv(w[1], w[1]))          # 80
        self.e2 = nn.Sequential(conv(w[1], w[2], 2), conv(w[2], w[2]))          # 40
        self.e3 = nn.Sequential(conv(w[2], w[3], 2), conv(w[3], w[3]))          # 20
        self.d2 = nn.Sequential(conv(w[3] + w[2], w[2]), conv(w[2], w[2]))
        self.d1 = nn.Sequential(conv(w[2] + w[1], w[1]), conv(w[1], w[1]))      # 80 (stride 2)
        self.heads = Heads(w[1])

    def forward(self, x, prior_mu=None, prior_sigma=None):
        e0 = self.e0(x); e1 = self.e1(e0); e2 = self.e2(e1); e3 = self.e3(e2)
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        return self.heads(d1)


class KPABlock(nn.Module):
    """Transformer block whose attention logits carry a Gaussian-prior bias and a motion bias over keys."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads; self.dk = d // heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.beta = nn.Parameter(torch.full((heads,), 0.5)); self.gamma = nn.Parameter(torch.zeros(heads))

    def forward(self, x, logprior, motion):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)                       # B,h,N,N
        att = att + self.beta.view(1, -1, 1, 1) * logprior[:, None, None, :] + self.gamma.view(1, -1, 1, 1) * motion[:, None, None, :]
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(y)
        return x + self.mlp(self.n2(x))


class FoveaKPA(nn.Module):
    def __init__(self, c_in=12, d=160, depth=4, heads=4):
        super().__init__()
        self.stem = nn.Sequential(conv(c_in, 32), conv(32, 64, 2), conv(64, 64))   # stride 2 (80)
        self.down = nn.Sequential(conv(64, 96, 2), conv(96, d, 2))                  # stride 8 (20×20 tokens)
        g = S // 8
        self.pos = nn.Parameter(torch.zeros(1, g * g, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([KPABlock(d, heads) for _ in range(depth)])
        self.up = nn.Sequential(conv(d + 64, 64), conv(64, 64))
        self.heads = Heads(64)
        yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        self.register_buffer("tok_xy", torch.stack([xx, yy], -1).reshape(-1, 2).float() * 8 + 4)   # token centres, px

    def forward(self, x, prior_mu, prior_sigma, motion_map=None):
        s2 = self.stem(x); t = self.down(s2); B, D, g, _ = t.shape
        tok = t.flatten(2).transpose(1, 2) + self.pos
        d2 = ((self.tok_xy[None] - prior_mu[:, None, :]) ** 2).sum(-1)               # B,N
        logprior = -0.5 * d2 / (prior_sigma[:, None] ** 2 + 16.0)
        if motion_map is None:
            motion = torch.zeros_like(logprior)
        else:
            m = F.adaptive_avg_pool2d(motion_map, g).flatten(1)
            motion = (m - m.mean(1, keepdim=True)) / (m.std(1, keepdim=True) + 1e-3)
        for b in self.blocks:
            tok = b(tok, logprior, motion)
        t = tok.transpose(1, 2).reshape(B, D, g, g)
        u = self.up(torch.cat([F.interpolate(t, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        return self.heads(u)


def build(arch="unet", frames=3, use_diff=True, use_prior=True):
    c_in = 3 * frames + (2 if use_diff and frames == 3 else 0) + (1 if use_prior else 0)
    return (FoveaUNet(c_in) if arch == "unet" else FoveaKPA(c_in)), c_in


def make_input(frames_rgb, prior_xy, prior_sigma, frames=3, use_diff=True, use_prior=True):
    """frames_rgb: B×3×3×S×S float in [0,1] (t-2, t-1, t). prior_xy: B×2 (px in crop). prior_sigma: B."""
    B = frames_rgb.shape[0]
    chans = [frames_rgb[:, 3 - frames:].reshape(B, 3 * frames, S, S)]
    motion = None
    if frames == 3:
        g = frames_rgb.mean(2)                                                      # B×3×S×S grey
        motion = torch.stack([(g[:, 2] - g[:, 1]).abs(), (g[:, 2] - g[:, 0]).abs()], 1)
        if use_diff:
            chans.append(motion * 4.0)
    if use_prior:
        yy, xx = torch.meshgrid(torch.arange(S, device=frames_rgb.device), torch.arange(S, device=frames_rgb.device), indexing="ij")
        d2 = (xx[None] - prior_xy[:, 0, None, None]) ** 2 + (yy[None] - prior_xy[:, 1, None, None]) ** 2
        chans.append(torch.exp(-0.5 * d2 / (prior_sigma[:, None, None] ** 2))[:, None])
    mm = None if motion is None else torch.minimum(motion[:, :1], motion[:, 1:])     # three-frame difference
    return torch.cat(chans, 1), mm


def focal_loss(logit, target, alpha=2, beta=4):
    """CenterNet focal loss on a Gaussian-splatted target."""
    p = logit.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    lp = -((1 - p) ** alpha) * torch.log(p) * pos
    ln = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    return (lp.sum() + ln.sum()) / pos.sum().clamp(min=1.0)


def decode(out, k=3, thr=0.05):
    """Top-k peaks → list per image of (cx, cy, w, h, score, blur_len, blur_angle) in crop px."""
    hm = out["hm"].sigmoid(); B, _, H, W = hm.shape
    peak = (hm == F.max_pool2d(hm, 5, 1, 2)) & (hm > thr)
    res = []
    for b in range(B):
        ys, xs = torch.nonzero(peak[b, 0], as_tuple=True)
        sc = hm[b, 0, ys, xs]; o = torch.argsort(sc, descending=True)[:k]; cur = []
        for i in o.tolist():
            y, x = int(ys[i]), int(xs[i])
            ox, oy = out["off"][b, :, y, x].tolist(); lw, lh = out["size"][b, :, y, x].tolist()
            bx, by = out["blur"][b, :, y, x].tolist()
            cur.append(((x + ox) * STRIDE, (y + oy) * STRIDE, math.exp(lw), math.exp(lh), float(sc[i]),
                        math.hypot(bx, by), 0.5 * math.atan2(by, bx)))
        res.append(cur)
    return res
