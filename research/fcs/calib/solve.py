"""MAP camera estimation from PitchGeo outputs (variants K2 / K3).

Observation model. For every class c judged present, the network gives an image line
(n_c, rho_c) — or an ellipse conic for circles — and a scale sigma_c. A camera theta =
(log f, rotation vector, centre C) projects 3-D points X sampled along marking c; each
projected point should lie on the predicted line / ellipse. Residuals are point-to-line
distances (Sampson distances for ellipses) divided by sigma_c, i.e. the Laplace likelihood
of the network, approximated with a soft-L1 robust least-squares loss. Because the
predicted lines are infinite, *every* sampled point of the marking constrains the camera,
including the parts outside the image — the solver never needs extremities.

Priors (K3, Gaussian, as extra residuals): camera height log-normal around 20 m, camera on
the near side (y > 34 m) at 20–120 m from the touchline, |x| < 60 m, roll ≈ 0.
Mirror ambiguity: the same observations are also explained by the centrally mirrored
labelling; both hypotheses are solved and the one with the higher posterior (lower MAP
objective) is kept, its margin being reported as the hypothesis log-odds.
Uncertainty: Laplace approximation  Σ_theta ≈ (Jᵀ J)^-1 at the optimum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from .pitch import ALL_CLASSES, CIRCLE_CLASSES, GROUND_LINES, LINES, mirror_name
from .sn_metric import sample_marking
from ..geometry.projection import Camera

SAMPLES = {n: sample_marking(n, step_m=2.0) for n in ALL_CLASSES}


@dataclass
class Observations:
    """Per-class predictions in PIXELS of the evaluation image."""
    lines: Dict[str, tuple]       # name -> (n (2,), rho, sigma_px)
    ellipses: Dict[str, tuple]    # name -> (conic 3x3 in pixel coords, sigma_px)
    width: int
    height: int
    points: Dict[str, tuple] = None   # name -> (pixel points (N, 2), sigma_px): point-on-marking evidence


def from_network(out: dict, b: int, width: int, height: int, p_thr: float = 0.5) -> Observations:
    """Convert network outputs (isotropic coords, 1 unit = image height) to pixel units."""
    import torch
    pres = torch.sigmoid(out["presence"][b]).cpu().numpy()
    n = out["n"][b].cpu().numpy(); rho = out["rho"][b].cpu().numpy()
    sig = np.exp(out["log_sigma"][b].cpu().numpy()) * height
    from .model import ellipse_conic
    conics = ellipse_conic(out["centre"][b], out["axes"][b], out["orient"][b]).cpu().numpy()
    S = np.diag([1 / height, 1 / height, 1.0])      # pixel -> isotropic
    lines, ells = {}, {}
    for i, name in enumerate(ALL_CLASSES):
        if pres[i] < p_thr:
            continue
        if name in CIRCLE_CLASSES:
            ells[name] = (S.T @ conics[i] @ S, float(sig[i]))
        else:
            lines[name] = (n[i], float(rho[i] * height), float(sig[i]))
    return Observations(lines, ells, width, height)


DENSE = {n: sample_marking(n, step_m=0.5) for n in ALL_CLASSES}


def _project(theta, X, w, h):
    """Project 3-D points; returns (pixels, depth). theta[7] (optional) = k1 distortion."""
    f = np.exp(theta[0]); R, _ = cv2.Rodrigues(theta[1:4]); C = theta[4:7]
    Xc = (X - C) @ R.T
    z = Xc[:, 2]
    zs = np.where(z > 1e-3, z, 1e-3)
    xn = Xc[:, :2] / zs[:, None]
    if len(theta) > 7:
        xn = xn * (1 + theta[7] * np.sum(xn ** 2, 1, keepdims=True))
    return xn * f + np.array([w / 2, h / 2]), z


def _point_to_polyline(Pts, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    ap = Pts[:, None, :] - a[None]
    t = np.clip(np.einsum("nsk,sk->ns", ap, ab) / np.maximum(np.einsum("sk,sk->s", ab, ab), 1e-12), 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.min(np.linalg.norm(Pts[:, None, :] - proj, axis=-1), axis=1)


def _residuals(theta, obs: Observations, mirrored: bool, priors: bool):
    w, h = obs.width, obs.height
    res = []
    for name, (n, rho, s) in obs.lines.items():
        X = SAMPLES[mirror_name(name) if mirrored else name]
        u, z = _project(theta, X, w, h)
        r = (u @ n - rho) / max(s, 0.5)
        res.append(np.where(z > 1e-3, r, 50.0) / np.sqrt(len(X)))
    for name, (Cq, s) in obs.ellipses.items():
        X = SAMPLES[mirror_name(name) if mirrored else name]
        u, z = _project(theta, X, w, h)
        uh = np.c_[u, np.ones(len(X))]
        v = uh @ Cq.T
        d = np.abs(np.einsum("ij,ij->i", uh, v)) / (2 * np.linalg.norm(v[:, :2], axis=1) + 1e-9)
        res.append(np.where(z > 1e-3, d / max(s, 0.5), 50.0) / np.sqrt(len(X)))
    for name, (Pts, s) in (obs.points or {}).items():
        X = DENSE[mirror_name(name) if mirrored else name]
        u, z = _project(theta, X, w, h)
        front = z > 1e-3
        if front.sum() < 2:
            res.append(np.full(len(Pts), 50.0)); continue
        res.append(_point_to_polyline(Pts, u[front]) / max(s, 0.5) / np.sqrt(len(Pts)))
    if len(theta) > 7:
        res.append([theta[7] / 0.1])                                     # mild prior on distortion
    if priors:
        R, _ = cv2.Rodrigues(theta[1:4]); C = theta[4:7]
        h = -C[2]
        res.append([(np.log(max(h, 0.1)) - np.log(20.0)) / 0.5,       # height ~ logN(20 m, 0.5)
                    max(0.0, 40.0 - C[1]) / 5.0,                        # behind the near touchline
                    max(0.0, abs(C[0]) - 60.0) / 5.0,
                    _roll(R) / np.deg2rad(3.0)])                         # cameramen avoid roll
    return np.concatenate([np.atleast_1d(np.asarray(r, float)) for r in res])


def _roll(R):
    # camera x-axis should be horizontal: its z (vertical) component ~ 0
    return float(np.arcsin(np.clip(R[0, 2], -1, 1)))


def _line_dlt_G(obs: Observations, mirrored: bool):
    """Pitch->image homography G from ground-line correspondences: a pitch line L and its
    image l satisfy L ∝ Gᵀ l, i.e. L × (Gᵀ l) = 0 — two linear equations per line.
    Points-mode observations are turned into lines by total least squares first."""
    pairs = []
    items = dict(obs.lines)
    for name, (P, s) in (obs.points or {}).items():
        if name in LINES and len(P) >= 2:
            m = P.mean(0); _, _, Vt = np.linalg.svd(P - m); n = Vt[-1]
            items.setdefault(name, (n, float(n @ m), s))
    Sx = np.diag([1 / obs.height, 1 / obs.height, 1.0])           # pixel normalisation
    for name, (n, rho, _) in items.items():
        mname = mirror_name(name) if mirrored else name
        if mname not in GROUND_LINES:
            continue
        a, b = LINES[mname]
        Lp = np.cross(np.r_[a[:2], 1.0], np.r_[b[:2], 1.0])     # pitch line (homogeneous)
        li = np.array([n[0], n[1], -rho]) @ np.linalg.inv(Sx)    # image line in normalised px
        pairs.append((Lp / np.linalg.norm(Lp), li / np.linalg.norm(li)))
    if len(pairs) < 4:
        return None
    A = []
    for Lp, l in pairs:
        # (Gᵀ l)_k = sum_i G[i,k] l_i ; cross(Lp, Gᵀl) = 0
        M = np.zeros((3, 9))
        for k in range(3):
            for i in range(3):
                M[k, 3 * i + k] = l[i]
        cx = np.array([[0, -Lp[2], Lp[1]], [Lp[2], 0, -Lp[0]], [-Lp[1], Lp[0], 0]])
        A.append((cx @ M)[:2])
    A = np.vstack(A)
    _, sv, Vt = np.linalg.svd(A)
    Gn = Vt[-1].reshape(3, 3)
    G = np.linalg.inv(Sx) @ Gn                                     # back to pixel units
    if not np.all(np.isfinite(G)) or abs(np.linalg.det(G)) < 1e-12:
        return None
    return G


def _init_from_G(G, w, h):
    from ..geometry.projection import _camera_closed_form
    try:
        cam = _camera_closed_form(np.linalg.inv(G), w, h)
    except Exception:
        return None
    if cam is None:
        return None
    rv, _ = cv2.Rodrigues(cam.R)
    return np.r_[np.log(cam.K[0, 0]), rv.ravel(), cam.center]


def _inits(obs: Observations, mirrored: bool = False):
    out = []
    G = _line_dlt_G(obs, mirrored)
    if G is not None:
        for Gs in (G, -G):
            p = _init_from_G(Gs, obs.width, obs.height)
            if p is not None:
                out.append(p)
    for f in (0.8, 1.5, 3.0, 6.0):
        for cx in (-25.0, 0.0, 25.0):
            for look_x in (-40.0, 0.0, 40.0):
                C = np.array([cx, 34 + 45.0, -18.0])
                tgt = np.array([look_x, 0.0, 0.0])
                z = tgt - C; z /= np.linalg.norm(z)
                x = np.cross(np.array([0, 0, 1.0]), z); x /= np.linalg.norm(x)
                y = np.cross(z, x)
                R = np.stack([x, y, z])
                if np.linalg.det(R) < 0:
                    R[0] *= -1
                rv, _ = cv2.Rodrigues(R)
                out.append(np.r_[np.log(f * obs.width), rv.ravel(), C])
    return out


@dataclass
class Solution:
    camera: Optional[Camera]
    cost: float
    mirrored: bool
    log_odds_vs_mirror: float
    cov: Optional[np.ndarray]


def solve_camera(obs: Observations, priors: bool = True, min_classes: int = 3,
                 distortion: bool = False) -> Solution:
    n_obs = len(obs.lines) + 2 * len(obs.ellipses) + len(obs.points or {})
    if n_obs < min_classes:
        return Solution(None, np.inf, False, 0.0, None)
    best = {}
    for mirrored in (False, True):
        cands = []
        for p0 in _inits(obs, mirrored):
            if distortion:
                p0 = np.r_[p0, 0.0]
            # cheap screening: a few iterations from every init, then refine the best few
            try:
                r = least_squares(_residuals, p0, args=(obs, mirrored, priors), loss="soft_l1",
                                  f_scale=1.0, max_nfev=30)
                cands.append((r.cost, r.x))
            except Exception:
                continue
        cands.sort(key=lambda c: c[0])
        top = None
        for _, x0 in cands[:4]:
            r = least_squares(_residuals, x0, args=(obs, mirrored, priors), loss="soft_l1",
                              f_scale=1.0, max_nfev=300)
            if top is None or r.cost < top.cost:
                top = r
        if top is not None:
            best[mirrored] = top
    if not best:
        return Solution(None, np.inf, False, 0.0, None)
    m = min(best, key=lambda k: best[k].cost)
    r = best[m]
    other = best.get(not m)
    log_odds = float(other.cost - r.cost) if other is not None else np.inf
    f = np.exp(r.x[0]); R, _ = cv2.Rodrigues(r.x[1:4]); C = r.x[4:7]
    cam = Camera(np.array([[f, 0, obs.width / 2], [0, f, obs.height / 2], [0, 0, 1.0]]), R, -R @ C,
                 k1=float(r.x[7]) if len(r.x) > 7 else 0.0)
    try:
        J = r.jac
        cov = np.linalg.pinv(J.T @ J)
    except Exception:
        cov = None
    if cam.center[2] > 0:
        return Solution(None, r.cost, m, log_odds, cov)
    return Solution(cam, float(r.cost), m, log_odds, cov)
