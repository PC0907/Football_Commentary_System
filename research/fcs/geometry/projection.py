"""Full pinhole camera from a pitch homography (needed for the airborne-ball model).

Pitch frame: X along the length (0..105), Y across (0 = far touchline .. 68 = near),
Z = X x Y, which points *into* the ground. A point at height h above the pitch is
(X, Y, -h). Gravity therefore accelerates Z positively: Z'' = +g.

Given G = H^{-1} (pitch -> image) = K [r1 r2 t] up to scale, with K = diag-ish
(f, f, principal point at the image centre), the two orthonormality constraints on
r1, r2 each give an estimate of f^2; we combine the well-conditioned ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

G_ACC = 9.81


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    k1: float = 0.0          # radial distortion on normalised coords: x_d = x (1 + k1 r^2)

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.column_stack([self.R, self.t])

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if self.k1 == 0.0:
            x = np.column_stack([X, np.ones(len(X))]) @ self.P.T
            return x[:, :2] / x[:, 2:3]
        Xc = X @ self.R.T + self.t
        xn = Xc[:, :2] / Xc[:, 2:3]
        xn = xn * (1 + self.k1 * np.sum(xn ** 2, 1, keepdims=True))
        return xn * self.K[0, 0] + self.K[:2, 2]

    def depth(self, X: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X) @ self.R.T + self.t)[:, 2]

    def project_jacobian(self, X: np.ndarray) -> np.ndarray:
        """2x3 Jacobian of the projection at 3-D point X."""
        P = self.P
        x = P @ np.r_[X, 1.0]
        J = np.empty((2, 3))
        for i in range(2):
            J[i] = (P[i, :3] * x[2] - x[i] * P[2, :3]) / x[2] ** 2
        return J


def camera_from_homography(H_img2pitch: np.ndarray, width: int, height: int,
                           refine: bool = True, max_resid_px: float = 8.0) -> Optional[Camera]:
    """Closed-form initialisation (below) + nonlinear refinement.

    The closed form estimates f from the orthonormality constraints and then forces R to
    be a rotation; when H is not exactly consistent with a square-pixel, central-principal-
    point camera this projection destroys agreement with H (median ~300 px on SNMOT).
    The refinement fits (f, rotation, translation) — 7 parameters — so that the camera
    reproduces H on a grid of pitch points visible in the image, from several focal-length
    initialisations; the camera is returned with its RMS residual in ``cam.resid_px`` and
    rejected if the residual exceeds ``max_resid_px``.
    """
    cam0 = _camera_closed_form(H_img2pitch, width, height)
    if not refine:
        return cam0
    return _refine_camera(H_img2pitch, width, height, cam0, max_resid_px)


def _refine_camera(H, width, height, cam0, max_resid_px):
    import cv2
    from scipy.optimize import least_squares
    try:
        G = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    # pitch grid points that project inside (a margin around) the image
    xs, ys = np.meshgrid(np.linspace(0, 105, 15), np.linspace(0, 68, 11))
    Pw = np.c_[xs.ravel(), ys.ravel()]
    ph = np.c_[Pw, np.ones(len(Pw))] @ G.T
    ok = ph[:, 2] > 1e-9
    U = np.full((len(Pw), 2), np.nan)
    U[ok] = ph[ok, :2] / ph[ok, 2:3]
    vis = ok & (U[:, 0] > -0.25 * width) & (U[:, 0] < 1.25 * width) & (U[:, 1] > -0.25 * height) & (U[:, 1] < 1.25 * height)
    if vis.sum() < 6:
        return None
    Pw, U = Pw[vis], U[vis]
    X3 = np.c_[Pw, np.zeros(len(Pw))]
    cx, cy = width / 2, height / 2

    def unpack(p):
        f = np.exp(p[0]); R, _ = cv2.Rodrigues(p[1:4]); return f, R, p[4:7]

    def resid(p):
        f, R, t = unpack(p)
        Xc = X3 @ R.T + t
        z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
        return np.r_[f * Xc[:, 0] / z + cx - U[:, 0], f * Xc[:, 1] / z + cy - U[:, 1]]

    inits = []
    if cam0 is not None:
        rv, _ = cv2.Rodrigues(cam0.R)
        inits.append(np.r_[np.log(cam0.K[0, 0]), rv.ravel(), cam0.t])
    # generic inits: camera on the near side, above the pitch, looking at the visible centroid
    centroid = np.r_[Pw.mean(0), 0.0]
    for f in (0.8, 1.5, 3.0, 6.0):
        for dist, hgt in ((45.0, 18.0), (70.0, 25.0)):
            C = np.array([52.5, 68 + dist, -hgt])
            z = centroid - C; z /= np.linalg.norm(z)
            x = np.cross(np.array([0, 0, 1.0]), z); x /= np.linalg.norm(x)   # image x roughly along +X
            y = np.cross(z, x)
            R = np.stack([x, y, z])
            if np.linalg.det(R) < 0:
                R[0] *= -1
            rv, _ = cv2.Rodrigues(R)
            inits.append(np.r_[np.log(f * width), rv.ravel(), -R @ C])
    best = None
    for p0 in inits:
        try:
            r = least_squares(resid, p0, method="lm", max_nfev=400)
        except Exception:
            continue
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        if best is None or rms < best[0]:
            best = (rms, r.x)
    if best is None or best[0] > max_resid_px:
        return None
    f, R, t = unpack(best[1])
    cam = Camera(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), R, t)
    if cam.center[2] > 0:            # below the ground: not a physical solution
        return None
    cam.resid_px = best[0]           # type: ignore[attr-defined]
    return cam


def _camera_closed_form(H_img2pitch: np.ndarray, width: int, height: int) -> Optional[Camera]:
    try:
        G = np.linalg.inv(H_img2pitch)
    except np.linalg.LinAlgError:
        return None
    cx, cy = width / 2.0, height / 2.0
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    g = T @ G                         # principal point moved to origin
    g1, g2 = g[:, 0], g[:, 1]
    ests = []
    den1 = g1[2] * g2[2]
    if abs(den1) > 1e-12:
        f2 = -(g1[0] * g2[0] + g1[1] * g2[1]) / den1
        if f2 > 0:
            ests.append((f2, abs(den1)))
    den2 = g1[2] ** 2 - g2[2] ** 2
    if abs(den2) > 1e-12:
        f2 = -((g1[0] ** 2 + g1[1] ** 2) - (g2[0] ** 2 + g2[1] ** 2)) / den2
        if f2 > 0:
            ests.append((f2, abs(den2)))
    if not ests:
        return None
    f2 = sum(e * w for e, w in ests) / sum(w for _, w in ests)
    f = float(np.sqrt(f2))
    if not (0.2 * width < f < 30 * width):
        return None
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    A = np.linalg.inv(K) @ G
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    cam = Camera(K, R, t)
    if cam.center[2] > 0:             # camera below the ground: flip the sign ambiguity
        cam = Camera(K, np.column_stack([-R[:, 0], -R[:, 1], R[:, 2]]), -t)
        # re-orthonormalise the flipped basis to keep det(R) = +1
        Rf = np.column_stack([-r1, -r2, np.cross(-r1, -r2)])
        U, _, Vt = np.linalg.svd(Rf)
        cam = Camera(K, U @ Vt, -t)
        if cam.center[2] > 0:
            return None
    return cam
