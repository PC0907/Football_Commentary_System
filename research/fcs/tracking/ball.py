"""Ball trackers B0..B3.

B0  highest-scoring ball detection per frame (no temporal model).
B1  image-plane constant-velocity Kalman filter with ego-motion compensation, chi2
    gating, coasting through missed detections.
B2  physics IMM in the pitch frame with *heterogeneous* modes
      rolling   : [X, Y, VX, VY] on the ground, rolling-resistance deceleration
      airborne  : [X, Y, Z, VX, VY, VZ], gravity + quadratic aerodynamic drag
    observed through the full camera recovered from the pitch homography. All mode
    likelihoods are evaluated in the *same* (pixel) measurement space so that mode
    probabilities are comparable.
B3  B2 + a third, *possessed*, mode and player detections as a second sensor:
      - context-dependent transition matrix: the prior probability of entering
        possession grows as the ball approaches the nearest player's feet;
      - in possessed mode the nearest player's foot position is fused as a
        pseudo-measurement, so during occlusion by the player's body the estimate
        follows the player instead of drifting ballistically.

Pitch frame: X 0..105, Y 0..68 (far -> near), Z into the ground (height = -Z).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..filters import Gaussian, cv_model, kf_predict
from ..geometry.camera import apply_h, h_jacobian
from ..geometry.projection import G_ACC, Camera, camera_from_homography

BALL_R = 0.11
ROLL, AIR, POSS = 0, 1, 2


@dataclass
class BallConfig:
    variant: str = "B3"
    fps: float = 25.0
    min_score: float = 0.05          # candidates considered
    init_score: float = 0.35         # to (re)initialise
    coast: int = 12                  # frames an un-updated estimate is still reported
    max_miss: int = 50               # frames before the track is abandoned
    gate: float = 13.8 * 2           # chi2(2) gate on pixel innovation
    score_weight: float = 1.0        # weight of log(score) in candidate selection
    px_sigma: float = 2.5            # detection centre noise (pixels) ...
    px_sigma_rel: float = 0.15       # ... + relative to box size
    # physics
    roll_decel: float = 0.4          # 1/s, linearised rolling resistance
    drag_k: float = 0.013            # 1/m, quadratic drag  (0.5 rho Cd A / m)
    sigma_roll: float = 3.0          # m/s^2 process noise
    sigma_air: float = 1.5
    sigma_poss: float = 5.0
    restitution: float = 0.6
    # IMM
    p_stay: float = 0.95
    homography_fallback: bool = True  # B2/B3 without a 3-D camera: use H (ground plane) instead of dropping the frame
    air_R_inflate: float = 25.0      # AIR-mode pixel variance factor when height is unobservable (H only)
    poss_radius: float = 1.2         # m, scale of the possession context
    poss_meas_sigma: float = 0.6     # m, ball-to-foot offset in possession
    img_w: int = 1920
    img_h: int = 1080


# ──────────────────────────────────────────────────────────────────────────────
# B0 / B1
# ──────────────────────────────────────────────────────────────────────────────

class BallTrackerB0:
    def __init__(self, cfg: BallConfig):
        self.cfg = cfg

    def step(self, frame, balls, **_) -> Optional[tuple]:
        if len(balls) == 0:
            return None
        b = balls[np.argmax(balls[:, 4])]
        return (frame, *b[:4], float(b[4])) if b[4] >= self.cfg.init_score else None


class BallTrackerB1:
    """Image-plane CV KF on the ball centre with CMC."""

    def __init__(self, cfg: BallConfig):
        self.cfg = cfg
        self.g: Optional[Gaussian] = None
        self.size = np.array([15.0, 15.0])
        self.miss = 0

    def _R(self, b):
        s = self.cfg.px_sigma + self.cfg.px_sigma_rel * float(np.mean(b[2:4]))
        return s * s * np.eye(2)

    def step(self, frame, balls, M=None, **_) -> Optional[tuple]:
        cfg = self.cfg
        if self.g is not None:
            if M is not None:
                c = apply_h(M, self.g.x[None, :2])[0]
                J = h_jacobian(M, self.g.x[:2])
                A = np.eye(4); A[:2, :2] = J; A[2:, 2:] = J
                x = A @ self.g.x; x[:2] = c
                self.g = Gaussian(x, A @ self.g.P @ A.T)
            F, Q = cv_model(1.0, 3.0, 2)
            self.g = kf_predict(self.g, F, Q)
        cand = balls[balls[:, 4] >= cfg.min_score] if len(balls) else balls
        chosen = None
        if self.g is not None and len(cand):
            best = -np.inf
            for b in cand:
                z = b[:2] + b[2:4] / 2
                S = self.g.P[:2, :2] + self._R(b)
                y = z - self.g.x[:2]
                d2 = y @ np.linalg.solve(S, y)
                if d2 > cfg.gate:
                    continue
                sc = -0.5 * d2 - 0.5 * np.log(np.linalg.det(S)) + cfg.score_weight * np.log(b[4])
                if sc > best:
                    best, chosen = sc, b
        if chosen is None and len(cand) and (self.g is None or self.miss > cfg.max_miss):
            b = cand[np.argmax(cand[:, 4])]
            if b[4] >= cfg.init_score:
                z = b[:2] + b[2:4] / 2
                self.g = Gaussian(np.r_[z, 0, 0], np.diag([25, 25, 400, 400.0]))
                self.size = b[2:4].copy(); self.miss = 0
                return (frame, *b[:4], float(b[4]))
        if chosen is not None:
            z = chosen[:2] + chosen[2:4] / 2
            Hm = np.eye(2, 4); R = self._R(chosen)
            S = Hm @ self.g.P @ Hm.T + R
            K = self.g.P @ Hm.T @ np.linalg.inv(S)
            self.g = Gaussian(self.g.x + K @ (z - self.g.x[:2]), (np.eye(4) - K @ Hm) @ self.g.P)
            self.size = 0.7 * self.size + 0.3 * chosen[2:4]; self.miss = 0
            return (frame, *chosen[:4], float(chosen[4]))
        if self.g is None:
            return None
        self.miss += 1
        return self._coast(frame)

    def _coast(self, frame):
        if self.miss > self.cfg.coast:
            return None
        c = self.g.x[:2]
        if not (0 <= c[0] < self.cfg.img_w and 0 <= c[1] < self.cfg.img_h):
            return None
        return (frame, c[0] - self.size[0] / 2, c[1] - self.size[1] / 2, *self.size, 0.3)


# ──────────────────────────────────────────────────────────────────────────────
# B2 / B3: physics IMM
# ──────────────────────────────────────────────────────────────────────────────

class HomographyProj:
    """Pitch->image projection from the image->pitch homography alone (exact on the ground plane).
    Used when no physically valid 3-D camera can be recovered from H (most broadcast frames):
    height is then unobservable, so the projection ignores z (zero Jacobian column) and the AIR
    mode's measurement noise is inflated by the caller."""

    def __init__(self, H_img2pitch: np.ndarray):
        self.Hinv = np.linalg.inv(H_img2pitch)

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return apply_h(self.Hinv, X[:, :2])

    def project_jacobian(self, X: np.ndarray) -> np.ndarray:
        J = np.zeros((2, 3)); J[:, :2] = h_jacobian(self.Hinv, np.asarray(X)[:2])
        return J


_EMB4 = np.zeros((6, 4)); _EMB4[0, 0] = _EMB4[1, 1] = _EMB4[3, 2] = _EMB4[4, 3] = 1.0   # 4D -> 6D
_PROJ4 = _EMB4.T


class BallTrackerIMM:
    def __init__(self, cfg: BallConfig):
        self.cfg = cfg
        self.modes: Optional[List[Gaussian]] = None
        self.mu = None
        self.use_poss = cfg.variant == "B3"
        self.n = 3 if self.use_poss else 2
        self.size = np.array([15.0, 15.0])
        self.miss = 0
        self.last_px: Optional[np.ndarray] = None
        self.mode_log: List[Tuple[int, np.ndarray]] = []

    # ---- dynamics ---------------------------------------------------------
    def _predict_roll(self, g: Gaussian, sigma: float) -> Gaussian:
        dt = 1.0 / self.cfg.fps
        F, Q = cv_model(dt, sigma, 2)
        F = F.copy(); F[2:, 2:] *= (1 - self.cfg.roll_decel * dt)
        return kf_predict(g, F, Q)

    def _predict_air(self, g: Gaussian) -> Gaussian:
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        p, v = g.x[:3], g.x[3:]
        s = float(np.linalg.norm(v)) + 1e-6
        a = np.array([0, 0, G_ACC]) - cfg.drag_k * s * v
        x = np.r_[p + v * dt + 0.5 * a * dt * dt, v + a * dt]
        Jv = np.eye(3) - cfg.drag_k * dt * (s * np.eye(3) + np.outer(v, v) / s)
        F = np.eye(6); F[:3, 3:] = dt * np.eye(3); F[3:, 3:] = Jv
        _, Q = cv_model(dt, cfg.sigma_air, 3)
        P = F @ g.P @ F.T + Q
        if x[2] > 0:                      # hit the ground: bounce
            x[2] = 0.0
            x[5] = -cfg.restitution * x[5]
        return Gaussian(x, 0.5 * (P + P.T))

    # ---- measurement in pixel space -----------------------------------------
    def _h(self, cam: Camera, X3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = cam.project(X3)[0]
        return u, cam.project_jacobian(X3)

    def _meas(self, j: int, g: Gaussian, cam: Camera):
        """Returns (predicted pixel, Jacobian wrt mode state)."""
        if j == AIR:
            u, J3 = self._h(cam, g.x[:3])
            Hj = np.zeros((2, 6)); Hj[:, :3] = J3
        else:
            u, J3 = self._h(cam, np.r_[g.x[:2], -BALL_R])
            Hj = np.zeros((2, 4)); Hj[:, :2] = J3[:, :2]
        return u, Hj

    def _R(self, b):
        s = self.cfg.px_sigma + self.cfg.px_sigma_rel * float(np.mean(b[2:4]))
        return s * s * np.eye(2)

    # ---- IMM bookkeeping ------------------------------------------------------
    def _to6(self, j, g: Gaussian) -> Gaussian:
        if j == AIR:
            return g
        P = _EMB4 @ g.P @ _EMB4.T
        P[2, 2] += 1e-4; P[5, 5] += 1e-4
        return Gaussian(_EMB4 @ g.x, P)

    def _from6(self, j, g: Gaussian) -> Gaussian:
        if j == AIR:
            P = g.P.copy()
            P[2, 2] = max(P[2, 2], 0.25); P[5, 5] = max(P[5, 5], 4.0)
            x = g.x.copy(); x[2] = min(x[2], 0.0)
            return Gaussian(x, P)
        return Gaussian(_PROJ4 @ g.x, _PROJ4 @ g.P @ _PROJ4.T)

    def _tpm(self, d_player: Optional[float]) -> np.ndarray:
        cfg = self.cfg
        n = self.n
        Pi = np.full((n, n), (1 - cfg.p_stay) / (n - 1))
        np.fill_diagonal(Pi, cfg.p_stay)
        if self.use_poss:
            # context: probability of entering possession given distance to nearest foot
            q = 0.25 * np.exp(-(d_player / cfg.poss_radius) ** 2) if d_player is not None else 0.0
            q_leave = 0.02 + 0.2 * (1 - np.exp(-(d_player / cfg.poss_radius) ** 2)) if d_player is not None else 0.3
            for i in (ROLL, AIR):
                Pi[i, POSS] = q
                Pi[i, [k for k in range(n) if k not in (i, POSS)]] = (1 - cfg.p_stay) / 2
                Pi[i, i] = 1 - Pi[i].sum() + Pi[i, i]
            Pi[POSS] = [q_leave * 0.7, q_leave * 0.3, 1 - q_leave]
        return Pi / Pi.sum(1, keepdims=True)

    def _init(self, b, H):
        z = b[:2] + b[2:4] / 2
        X = apply_h(H, z[None])[0]
        g4 = Gaussian(np.r_[X, 0, 0], np.diag([1.0, 1.0, 100.0, 100.0]))
        g6 = Gaussian(np.r_[X, 0, 0, 0, 0], np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 25.0]))
        self.modes = [g4.copy(), g6] + ([g4.copy()] if self.use_poss else [])
        self.mu = np.array([0.6, 0.2, 0.2][:self.n]); self.mu /= self.mu.sum()
        self.size = b[2:4].copy(); self.miss = 0; self.last_px = z

    # ---- main step ----------------------------------------------------------------
    def step(self, frame, balls, H=None, players=None, **_) -> Optional[tuple]:
        cfg = self.cfg
        cam = camera_from_homography(H, cfg.img_w, cfg.img_h) if H is not None else None
        if cam is None and H is not None and cfg.homography_fallback:
            cam = HomographyProj(H)        # graceful degradation: ground-plane measurement model
        cand = balls[balls[:, 4] >= cfg.min_score] if len(balls) else balls
        feet = None
        if H is not None and players is not None and len(players):
            feet = apply_h(H, np.column_stack([players[:, 0] + players[:, 2] / 2, players[:, 1] + players[:, 3]]))

        if self.modes is None or cam is None:
            if cam is not None and len(cand) and cand[:, 4].max() >= cfg.init_score:
                b = cand[np.argmax(cand[:, 4])]
                self._init(b, H)
                return (frame, *b[:4], float(b[4]))
            return None

        # --- IMM mixing + prediction
        Xc = sum(self.mu[j] * self._to6(j, g).x for j, g in enumerate(self.modes))
        d_player = float(np.min(np.linalg.norm(feet - Xc[:2], axis=1))) if feet is not None else None
        Pi = self._tpm(d_player)
        c = np.maximum(Pi.T @ self.mu, 1e-12)
        w = (Pi * self.mu[:, None]) / c[None, :]
        six = [self._to6(i, g) for i, g in enumerate(self.modes)]
        pred = []
        for j in range(self.n):
            xm = sum(w[i, j] * six[i].x for i in range(self.n))
            Pm = sum(w[i, j] * (six[i].P + np.outer(six[i].x - xm, six[i].x - xm)) for i in range(self.n))
            g = self._from6(j, Gaussian(xm, Pm))
            if j == ROLL:
                g = self._predict_roll(g, cfg.sigma_roll)
            elif j == AIR:
                g = self._predict_air(g)
            else:
                g = self._predict_roll(g, cfg.sigma_poss)
                if feet is not None:      # fuse the nearest player's feet (player tracks as a sensor)
                    k = int(np.argmin(np.linalg.norm(feet - g.x[:2], axis=1)))
                    if np.linalg.norm(feet[k] - g.x[:2]) < 3 * cfg.poss_radius:
                        Hm = np.eye(2, 4); R = cfg.poss_meas_sigma ** 2 * np.eye(2)
                        S = Hm @ g.P @ Hm.T + R
                        K = g.P @ Hm.T @ np.linalg.inv(S)
                        g = Gaussian(g.x + K @ (feet[k] - g.x[:2]), (np.eye(4) - K @ Hm) @ g.P)
            pred.append(g)

        # --- candidate selection by mixture likelihood in pixel space
        best, chosen, chosen_upd = -np.inf, None, None
        for b in cand:
            z = b[:2] + b[2:4] / 2
            R = self._R(b)
            lls, upd, d2min = [], [], np.inf
            for j, g in enumerate(pred):
                u, Hj = self._meas(j, g, cam)
                y = z - u
                Rj = R * self.cfg.air_R_inflate if (j == AIR and isinstance(cam, HomographyProj)) else R
                S = Hj @ g.P @ Hj.T + Rj
                Si = np.linalg.inv(S)
                d2 = float(y @ Si @ y); d2min = min(d2min, d2)
                lls.append(-0.5 * d2 - 0.5 * np.log(np.linalg.det(S)) - np.log(2 * np.pi))
                K = g.P @ Hj.T @ Si
                Pn = (np.eye(len(g.x)) - K @ Hj) @ g.P
                upd.append(Gaussian(g.x + K @ y, 0.5 * (Pn + Pn.T)))
            if d2min > cfg.gate:
                continue
            lls = np.array(lls)
            mix = np.log(np.sum(c * np.exp(lls - lls.max()))) + lls.max()
            sc = mix + cfg.score_weight * np.log(b[4])
            if sc > best:
                best, chosen, chosen_upd, chosen_ll = sc, b, upd, lls

        if chosen is not None:
            self.modes = chosen_upd
            p = c * np.exp(chosen_ll - chosen_ll.max())
            self.mu = p / p.sum()
            self.size = 0.7 * self.size + 0.3 * chosen[2:4]
            self.miss = 0
            self.last_px = chosen[:2] + chosen[2:4] / 2
            self.mode_log.append((frame, self.mu.copy()))
            return (frame, *chosen[:4], float(chosen[4]))

        # no gated candidate: keep the prediction; re-initialise on a confident far detection
        self.modes, self.mu = pred, c / c.sum()
        self.miss += 1
        if len(cand) and self.miss > 3 and cand[:, 4].max() >= max(cfg.init_score, 0.5):
            b = cand[np.argmax(cand[:, 4])]
            self._init(b, H)
            return (frame, *b[:4], float(b[4]))
        if self.miss > cfg.max_miss:
            self.modes = None
            return None
        if self.miss > cfg.coast:
            return None
        Xc = sum(self.mu[j] * self._to6(j, g).x for j, g in enumerate(self.modes))
        u = cam.project(Xc[:3] if self.mu[AIR] > 0.5 else np.r_[Xc[:2], -BALL_R])[0]
        if not (0 <= u[0] < cfg.img_w and 0 <= u[1] < cfg.img_h):
            return None
        return (frame, u[0] - self.size[0] / 2, u[1] - self.size[1] / 2, *self.size, 0.3)


def make_ball_tracker(cfg: BallConfig):
    return {"B0": BallTrackerB0, "B1": BallTrackerB1}.get(cfg.variant, BallTrackerIMM)(cfg)
