"""Estimation primitives shared by all trackers.

Everything here is plain linear-Gaussian machinery written out explicitly so that the
trackers built on top of it can say exactly which noise model they use:

* ``LinearKF``            — predict / update with arbitrary F, Q, H, R (per call).
* ``project_speed``       — state-constraint projection (Simon 2010, "Kalman filtering
                            with state constraints"): clamp |v| <= v_max by projecting
                            the mean and shrinking the covariance along the velocity axis.
* ``rts_smooth``          — Rauch–Tung–Striebel fixed-interval smoother over a stored
                            filter history (offline mode: commentary is generated after
                            the clip is processed, so we can afford to look ahead).
* ``cv_model`` / ``ca_model`` — discrete white-noise-acceleration (CV) and
                            white-noise-jerk (CA) models for an n-D position.
* ``imm_mix`` / ``imm_combine`` — Interacting Multiple Model bookkeeping
                            (Blom & Bar-Shalom 1988) for models with *different state
                            dimensions*, via explicit embedding matrices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

CHI2_2D_99 = 9.21     # chi-square(2) 99% gate
CHI2_2D_999 = 13.82   # chi-square(2) 99.9% gate


# ──────────────────────────────────────────────────────────────────────────────
# Motion models
# ──────────────────────────────────────────────────────────────────────────────

def cv_model(dt: float, sigma_a: float, ndim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Constant velocity, state [p(ndim), v(ndim)], white-noise acceleration sigma_a."""
    F = np.eye(2 * ndim)
    F[:ndim, ndim:] = dt * np.eye(ndim)
    q = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * sigma_a ** 2
    Q = np.kron(q, np.eye(ndim))
    return F, Q


def ca_model(dt: float, sigma_j: float, ndim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Constant acceleration, state [p, v, a], white-noise jerk sigma_j."""
    F = np.eye(3 * ndim)
    F[:ndim, ndim:2 * ndim] = dt * np.eye(ndim)
    F[:ndim, 2 * ndim:] = 0.5 * dt ** 2 * np.eye(ndim)
    F[ndim:2 * ndim, 2 * ndim:] = dt * np.eye(ndim)
    q = np.array([
        [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
        [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
        [dt ** 3 / 6, dt ** 2 / 2, dt],
    ]) * sigma_j ** 2
    return F, np.kron(q, np.eye(ndim))


# ──────────────────────────────────────────────────────────────────────────────
# Kalman filter
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Gaussian:
    x: np.ndarray
    P: np.ndarray

    def copy(self) -> "Gaussian":
        return Gaussian(self.x.copy(), self.P.copy())


def kf_predict(g: Gaussian, F: np.ndarray, Q: np.ndarray, u: Optional[np.ndarray] = None) -> Gaussian:
    x = F @ g.x if u is None else F @ g.x + u
    P = F @ g.P @ F.T + Q
    return Gaussian(x, 0.5 * (P + P.T))


def innovation(g: Gaussian, H: np.ndarray, R: np.ndarray, z: np.ndarray,
               h_of_x: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (innovation y, innovation covariance S). ``h_of_x`` lets EKF callers
    pass the nonlinear predicted measurement while ``H`` is its Jacobian."""
    zhat = H @ g.x if h_of_x is None else h_of_x
    y = z - zhat
    S = H @ g.P @ H.T + R
    return y, 0.5 * (S + S.T)


def kf_update(g: Gaussian, H: np.ndarray, R: np.ndarray, z: np.ndarray,
              h_of_x: Optional[np.ndarray] = None) -> Tuple[Gaussian, float]:
    """Joseph-form update. Returns (posterior, log-likelihood of z)."""
    y, S = innovation(g, H, R, z, h_of_x)
    Sinv = np.linalg.inv(S)
    K = g.P @ H.T @ Sinv
    x = g.x + K @ y
    I_KH = np.eye(len(g.x)) - K @ H
    P = I_KH @ g.P @ I_KH.T + K @ R @ K.T
    return Gaussian(x, 0.5 * (P + P.T)), gaussian_loglik(y, S, Sinv)


def gaussian_loglik(y: np.ndarray, S: np.ndarray, Sinv: Optional[np.ndarray] = None) -> float:
    if Sinv is None:
        Sinv = np.linalg.inv(S)
    sign, logdet = np.linalg.slogdet(S)
    return float(-0.5 * (y @ Sinv @ y + logdet + len(y) * np.log(2 * np.pi)))


def mahalanobis2(y: np.ndarray, S: np.ndarray) -> float:
    return float(y @ np.linalg.solve(S, y))


# ──────────────────────────────────────────────────────────────────────────────
# State constraints (physiology / physics)
# ──────────────────────────────────────────────────────────────────────────────

def project_speed(g: Gaussian, vel_idx: Sequence[int], v_max: float) -> Gaussian:
    """Project the state onto {|v| <= v_max} (estimate projection, Simon & Chia 2002).

    The mean is radially clamped. The covariance is projected with the Jacobian of the
    clamp so that the filter does not stay over-confident about an infeasible velocity
    direction component: along the velocity direction the variance is shrunk
    (we have just learned |v| cannot be larger), orthogonal components are kept.
    """
    vi = list(vel_idx)
    v = g.x[vi]
    s = float(np.linalg.norm(v))
    if s <= v_max or s == 0.0:
        return g
    x = g.x.copy()
    x[vi] = v * (v_max / s)
    n = v / s
    # Jacobian of v -> v_max * v / |v| :  (v_max/s) (I - n n^T)
    Jv = (v_max / s) * (np.eye(len(vi)) - np.outer(n, n))
    J = np.eye(len(x))
    J[np.ix_(vi, vi)] = Jv
    # keep a small floor along n so P stays positive definite
    P = J @ g.P @ J.T
    floor = 1e-3 * np.trace(g.P[np.ix_(vi, vi)]) / len(vi)
    P[np.ix_(vi, vi)] += floor * np.outer(n, n)
    return Gaussian(x, 0.5 * (P + P.T))


# ──────────────────────────────────────────────────────────────────────────────
# RTS smoother
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterStep:
    """One time step of a stored filter history."""
    prior: Gaussian            # x_{k|k-1}
    post: Gaussian             # x_{k|k}
    F: np.ndarray              # transition used to go k -> k+1 (filled at next predict)


def rts_smooth(steps: List[FilterStep]) -> List[Gaussian]:
    """Fixed-interval RTS smoother. ``steps[k].F`` must be the transition from k to k+1
    (the last one is unused)."""
    n = len(steps)
    if n == 0:
        return []
    out: List[Gaussian] = [None] * n  # type: ignore
    out[-1] = steps[-1].post.copy()
    for k in range(n - 2, -1, -1):
        post, nxt_prior, F = steps[k].post, steps[k + 1].prior, steps[k].F
        C = post.P @ F.T @ np.linalg.pinv(nxt_prior.P)
        x = post.x + C @ (out[k + 1].x - nxt_prior.x)
        P = post.P + C @ (out[k + 1].P - nxt_prior.P) @ C.T
        out[k] = Gaussian(x, 0.5 * (P + P.T))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# IMM with heterogeneous state spaces
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IMMState:
    modes: List[Gaussian]
    mu: np.ndarray                       # mode probabilities
    history: list = field(default_factory=list)


def imm_mix(state: IMMState, Pi: np.ndarray, to_common: List[np.ndarray],
            from_common: List[np.ndarray]) -> Tuple[List[Gaussian], np.ndarray]:
    """Mixing step for modes with different state dimensions.

    Every mode j has an embedding ``to_common[j]`` (C x n_j) into a common state space
    and a projection ``from_common[j]`` (n_j x C) back. Missing components in the
    common space for a mode are filled from mode-specific defaults by the caller
    (here: they get the mixed value of the modes that do have them).
    Returns the mixed initial conditions per mode and the predicted mode probabilities.
    """
    M = len(state.modes)
    c = Pi.T @ state.mu                          # predicted mode probs
    c = np.maximum(c, 1e-12)
    w = (Pi * state.mu[:, None]) / c[None, :]    # w[i, j] = P(i at k-1 | j at k)
    common = [Gaussian(to_common[i] @ g.x, to_common[i] @ g.P @ to_common[i].T)
              for i, g in enumerate(state.modes)]
    mixed = []
    for j in range(M):
        xj = sum(w[i, j] * common[i].x for i in range(M))
        Pj = sum(w[i, j] * (common[i].P + np.outer(common[i].x - xj, common[i].x - xj))
                 for i in range(M))
        mixed.append(Gaussian(from_common[j] @ xj, from_common[j] @ Pj @ from_common[j].T))
    return mixed, c


def imm_probs(c: np.ndarray, logliks: np.ndarray) -> np.ndarray:
    m = np.max(logliks)
    p = c * np.exp(logliks - m)
    s = p.sum()
    return p / s if s > 0 else np.full_like(c, 1.0 / len(c))
