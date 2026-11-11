"""PitchGeo: direct regression of pitch-marking geometry.

Outputs, for each of the 26 marking classes (fixed class queries, no set matching):
  presence logit
  line  : unit normal n = (nx, ny) and offset rho  —  n · X = rho   (straight markings)
  ellipse: centre (cx, cy), semi-axes (a, b), orientation (cos 2phi, sin 2phi)  (circles)
  log_sigma: per-class aleatoric scale of the residuals (Laplace likelihood) — the network
             reports how sure it is about each marking; used as weights by the MAP camera
             solver and propagated to pitch-position covariances downstream
  + a dense per-class line mask at 1/4 resolution (auxiliary; K1 variant)

Coordinates: isotropic normalised image coordinates X = x * (W/H), Y = y with x, y in
[0, 1]; rho, centres and axes are in these units (1 unit = image height).

Backbone: DINOv2 (Apache-2.0) via Hugging Face transformers, fine-tuned with a lower LR.
"""
from __future__ import annotations

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import C


def sine_pos_2d(h, w, dim, device):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    y = y.float() / max(h - 1, 1); x = x.float() / max(w - 1, 1)
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, device=device).float() / d))
    out = [torch.sin(x[..., None] * omega * math.pi * 2), torch.cos(x[..., None] * omega * math.pi * 2),
           torch.sin(y[..., None] * omega * math.pi * 2), torch.cos(y[..., None] * omega * math.pi * 2)]
    return torch.cat(out, -1).reshape(h * w, -1)


class PitchGeo(nn.Module):
    def __init__(self, backbone: str = "facebook/dinov2-small", d: int = 256, n_dec: int = 3,
                 patch: int = 14):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(backbone)
        hid = self.backbone.config.hidden_size
        self.patch = patch
        self.proj = nn.Linear(hid, d)
        self.queries = nn.Embedding(C, d)
        layer = nn.TransformerDecoderLayer(d, 8, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_dec)
        self.head_presence = nn.Linear(d, 1)
        self.head_logsig = nn.Linear(d, 1)
        nn.init.zeros_(self.head_logsig.weight)          # start at sigma ≈ 50 px @540:
        nn.init.constant_(self.head_logsig.bias, 0.6)    # -4 + 3 tanh(0.6) ≈ log(50/540)
        self.head_line = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        self.head_ellipse = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 6))
        self.dense = nn.Sequential(nn.Conv2d(hid, d, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.Conv2d(d, C, 1))

    def forward(self, x):
        B, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        out = self.backbone(pixel_values=x).last_hidden_state      # [B, 1 + h*w, hid]
        tok = out[:, 1:1 + h * w]
        mem = self.proj(tok) + sine_pos_2d(h, w, self.proj.out_features, x.device)[None]
        q = self.queries.weight[None].expand(B, -1, -1)
        z = self.decoder(q, mem)                                     # [B, C, d]
        presence = self.head_presence(z).squeeze(-1)
        # sigma in image heights; softly bounded to [e^-7, e^-1] ≈ [0.5 px, 200 px] @540 (v1 used a hard
        # clamp at e^-4 ≈ 10 px, which saturated with zero gradient — fixed)
        log_sigma = -4.0 + 3.0 * torch.tanh(self.head_logsig(z).squeeze(-1))
        ln = self.head_line(z)
        n = F.normalize(ln[..., :2], dim=-1)
        aspect = W / H
        rho = ln[..., 2] + 0.5 * (n[..., 0] * aspect + n[..., 1])    # bias: lines through centre
        el = self.head_ellipse(z)
        centre = torch.stack([torch.sigmoid(el[..., 0]) * 3 * aspect - aspect,
                              torch.sigmoid(el[..., 1]) * 3 - 1], -1)  # allow off-image centres
        axes = torch.exp(el[..., 2:4].clamp(-6, 3))
        orient = F.normalize(el[..., 4:6], dim=-1)
        fmap = tok.transpose(1, 2).reshape(B, -1, h, w)
        dense = F.interpolate(self.dense(fmap), size=(H // 4, W // 4), mode="bilinear", align_corners=False)
        return dict(presence=presence, log_sigma=log_sigma, n=n, rho=rho, centre=centre, axes=axes, orient=orient,
                    dense=dense, aspect=aspect)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers + losses
# ──────────────────────────────────────────────────────────────────────────────

def ellipse_conic(centre, axes, orient):
    """Conic matrix (…, 3, 3) of the ellipse in isotropic coords."""
    c2, s2 = orient[..., 0], orient[..., 1]
    phi = 0.5 * torch.atan2(s2, c2)
    c, s = torch.cos(phi), torch.sin(phi)
    a2, b2 = axes[..., 0] ** 2 + 1e-8, axes[..., 1] ** 2 + 1e-8
    A = c * c / a2 + s * s / b2
    Bq = 2 * c * s * (1 / a2 - 1 / b2)
    Cq = s * s / a2 + c * c / b2
    x0, y0 = centre[..., 0], centre[..., 1]
    D = -2 * A * x0 - Bq * y0
    E = -2 * Cq * y0 - Bq * x0
    Fq = A * x0 * x0 + Bq * x0 * y0 + Cq * y0 * y0 - 1
    M = torch.stack([torch.stack([A, Bq / 2, D / 2], -1),
                     torch.stack([Bq / 2, Cq, E / 2], -1),
                     torch.stack([D / 2, E / 2, Fq], -1)], -2)
    return M


def sampson_distance(conic, P):
    """conic (B, C, 3, 3); P (B, C, N, 2) -> (B, C, N) first-order geometric distance."""
    ph = torch.cat([P, torch.ones_like(P[..., :1])], -1)
    v = torch.einsum("bcij,bcnj->bcni", conic, ph)
    alg = (ph * v).sum(-1)
    grad = 2 * v[..., :2]
    return alg.abs() / (grad.norm(dim=-1) + 1e-6)


def laplace_nll(d, log_sigma):
    """Negative log-likelihood of residual distances d under Laplace(0, sigma) (per point)."""
    return d / torch.exp(log_sigma) + log_sigma


def _mirror_batch(batch):
    from .data import MIRROR_IDX
    m = dict(batch)
    for k in ("present", "pts", "pmask", "dense"):
        m[k] = batch[k][:, MIRROR_IDX]
    return m


def pitchgeo_loss(out, batch, line_idx, circle_idx, mirror_invariant=True, **kw):
    """Mirror-invariant wrapper: per image, min over {GT labels, centrally mirrored labels}."""
    if not mirror_invariant:
        return _pitchgeo_loss(out, batch, line_idx, circle_idx, **kw)
    l1, st1, per1 = _pitchgeo_loss(out, batch, line_idx, circle_idx, per_image=True, **kw)
    l2, st2, per2 = _pitchgeo_loss(out, _mirror_batch(batch), line_idx, circle_idx, per_image=True, **kw)
    use2 = per2 < per1
    total = torch.where(use2, per2, per1).mean()
    st = {k: float(np.where(use2.cpu().numpy(), 1, 0).mean() * st2[k] + (1 - use2.float().mean().item()) * st1[k])
          for k in st1}
    st["mirror_frac"] = float(use2.float().mean())
    return total, st


def _pitchgeo_loss(out, batch, line_idx, circle_idx, w_line=1.0, w_ell=1.0, w_dense=1.0, huber=0.01,
                   probabilistic=True, per_image=False):
    present = batch["present"]
    pmask = batch["pmask"]
    pts = batch["pts"].clone()
    pts[..., 0] = pts[..., 0] * out["aspect"]                      # isotropic coordinates
    l_pres_i = F.binary_cross_entropy_with_logits(out["presence"], present, reduction="none").mean(1)
    l_pres = l_pres_i.mean()
    # straight markings: point-to-line distance of every annotated point
    li = line_idx
    n, rho = out["n"][:, li], out["rho"][:, li]
    d_line = ((pts[:, li] * n[:, :, None, :]).sum(-1) - rho[..., None]).abs()
    m = pmask[:, li] * present[:, li, None]
    if probabilistic:
        nll = laplace_nll(d_line, out["log_sigma"][:, li, None])
    else:
        nll = 20 * F.huber_loss(d_line, torch.zeros_like(d_line), reduction="none", delta=huber)
    l_line = (nll * m).sum() / m.sum().clamp(min=1)
    l_line_i = (nll * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)
    # circles: Sampson distance of the annotated arc points to the predicted ellipse
    ci = circle_idx
    conic = ellipse_conic(out["centre"][:, ci], out["axes"][:, ci], out["orient"][:, ci])
    d_ell = sampson_distance(conic, pts[:, ci])
    me = pmask[:, ci] * present[:, ci, None]
    if probabilistic:
        nll_e = laplace_nll(d_ell, out["log_sigma"][:, ci, None])
    else:
        nll_e = 20 * F.huber_loss(d_ell, torch.zeros_like(d_ell), reduction="none", delta=huber)
    l_ell = (nll_e * me).sum() / me.sum().clamp(min=1)
    l_ell_i = (nll_e * me).sum((1, 2)) / me.sum((1, 2)).clamp(min=1)
    dense_t = batch["dense"]
    l_dense_i = F.binary_cross_entropy_with_logits(out["dense"], dense_t, reduction="none",
                                                   pos_weight=torch.tensor(20.0, device=dense_t.device)).mean((1, 2, 3))
    l_dense = l_dense_i.mean()
    total = l_pres + w_line * l_line + w_ell * l_ell + w_dense * l_dense
    with torch.no_grad():
        H = 1.0   # distances are in units of image height
        stats = dict(pres=l_pres.item(), line=l_line.item(), ell=l_ell.item(), dense=l_dense.item(),
                     line_px540=float((d_line * m).sum() / m.sum().clamp(min=1)) * 540,
                     ell_px540=float((d_ell * me).sum() / me.sum().clamp(min=1)) * 540,
                     sigma_px540=float((torch.exp(out["log_sigma"]) * present).sum() / present.sum().clamp(min=1)) * 540)
    if per_image:
        per = l_pres_i + w_line * l_line_i + w_ell * l_ell_i + w_dense * l_dense_i
        return total, stats, per
    return total, stats
