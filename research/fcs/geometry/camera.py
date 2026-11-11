"""Camera geometry for broadcast football.

Physical premise used throughout: the main broadcast camera sits at a fixed position and
only pans / tilts / zooms. For a purely rotating (and zooming) camera, the image-to-image
mapping between any two frames is *exactly* a homography, M = K_t R K_{t-1}^{-1}
(the "infinite homography"). Therefore

  * ego-motion between frames is a 3x3 homography estimated from background flow, and
  * the image->pitch homography propagates as   H_t = H_{t-1} M_t^{-1}.

This gives us two independent "sensors" for the image->pitch calibration of frame t:
  (a) a *keypoint sensor*  — field keypoints detected in frame t (absolute, noisy,
      sometimes unavailable), and
  (b) an *ego-motion sensor* — the previous calibration pushed through background flow
      (relative, precise frame-to-frame, but drifts).
`CalibrationFilter` fuses them (calibration variant C2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0


# ──────────────────────────────────────────────────────────────────────────────
# Homography helpers
# ──────────────────────────────────────────────────────────────────────────────

def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    ph = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def h_jacobian(H: np.ndarray, u: np.ndarray) -> np.ndarray:
    """2x2 Jacobian d(H(u))/du at image point u — used to propagate pixel noise into
    pitch metres (heteroscedastic measurement covariance)."""
    x, y = float(u[0]), float(u[1])
    a = H @ np.array([x, y, 1.0])
    w = a[2]
    J = np.empty((2, 2))
    for i in range(2):
        J[i, 0] = (H[i, 0] * w - a[i] * H[2, 0]) / w ** 2
        J[i, 1] = (H[i, 1] * w - a[i] * H[2, 1]) / w ** 2
    return J


def normalize_h(H: np.ndarray) -> np.ndarray:
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def well_conditioned(H: Optional[np.ndarray], max_cond: float = 1e9) -> bool:
    """Reject degenerate image->pitch maps (singular / non-finite)."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    try:
        return np.linalg.cond(H) < max_cond
    except np.linalg.LinAlgError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Ego-motion (camera motion compensation)
# ──────────────────────────────────────────────────────────────────────────────

class EgoMotion:
    """Frame-to-frame homography from sparse background optical flow.

    Foreground (boxes) is masked out, and so is the top of the frame (crowd/billboards
    move with the camera too, so they are fine; the mask only removes players).
    """

    def __init__(self, max_corners: int = 600, downscale: float = 0.5):
        self.prev: Optional[np.ndarray] = None
        self.max_corners = max_corners
        self.s = downscale

    def __call__(self, frame_bgr: np.ndarray, boxes_tlwh: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """Returns (M, n_inliers): M maps pixel coords of the previous frame to this
        frame (identity on the first call or on failure)."""
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
        mask = np.full(g.shape, 255, np.uint8)
        if boxes_tlwh is not None:
            for x, y, w, h in np.asarray(boxes_tlwh)[:, :4] * self.s:
                cv2.rectangle(mask, (int(x - 2), int(y - 2)), (int(x + w + 2), int(y + h + 2)), 0, -1)
        M, n = np.eye(3), 0
        if self.prev is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev[0], self.max_corners, 0.01, 8, mask=self.prev[1])
            if p0 is not None and len(p0) >= 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev[0], g, p0, None, winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1
                if ok.sum() >= 8:
                    Hs, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 1.0)
                    if Hs is not None:
                        S = np.diag([self.s, self.s, 1.0])
                        M = np.linalg.inv(S) @ Hs @ S
                        n = int(inl.sum())
        self.prev = (g, mask)
        return normalize_h(M), n


# ──────────────────────────────────────────────────────────────────────────────
# Calibration fusion (keypoint sensor ⊕ ego-motion sensor)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Calib:
    H: Optional[np.ndarray]       # image -> pitch metres
    sigma_m: float                # isotropic 1-sigma calibration uncertainty (metres) at pitch scale
    source: str                   # "keypoints" | "propagated" | "fused" | "none"


def _grid(width: int, height: int, n: int = 7) -> np.ndarray:
    xs = np.linspace(0.05 * width, 0.95 * width, n)
    ys = np.linspace(0.35 * height, 0.95 * height, n)   # lower part of frame = pitch
    return np.array([(x, y) for y in ys for x in xs], float)


def _on_pitch(P: np.ndarray, margin: float = 10.0) -> np.ndarray:
    return ((P[:, 0] > -margin) & (P[:, 0] < PITCH_LENGTH + margin) &
            (P[:, 1] > -margin) & (P[:, 1] < PITCH_WIDTH + margin))


class CalibrationFilter:
    """Recursive fusion of keypoint calibrations with ego-motion propagation.

    The calibration is represented by where a fixed grid of image points lands on the
    pitch. Each sensor yields such a point set with an isotropic uncertainty; the fused
    point set is the inverse-variance weighted mean, and H is refit to it. Ego-motion
    propagation inflates the variance by ``drift_m`` per frame (random-walk drift), so a
    frame with no keypoints is *predicted*, not dropped, and a single bad keypoint fit is
    damped by the propagated estimate.

    A keypoint calibration that disagrees with the prediction by more than ``gate``
    standard deviations is treated as an outlier (e.g. left/right box confusion) unless
    the filter has already drifted too far, in which case it re-initialises.
    """

    def __init__(self, width: int, height: int, drift_m: float = 0.15, gate: float = 4.0,
                 max_sigma: float = 6.0, kp_floor: float = 0.0):
        self.grid = _grid(width, height)
        self.drift = drift_m
        self.gate = gate
        self.max_sigma = max_sigma
        self.kp_floor = kp_floor          # minimum 1-sigma (m) assigned to a keypoint fit
        self.state: Optional[Calib] = None
        self.last_pred: Optional[Calib] = None

    def step(self, M_prev_to_cur: np.ndarray, H_kp: Optional[np.ndarray], sigma_kp: float) -> Calib:
        pred = None
        if self.state is not None and self.state.H is not None:
            Hp = normalize_h(self.state.H @ np.linalg.inv(M_prev_to_cur)) \
                if well_conditioned(M_prev_to_cur) else self.state.H
            sp = float(np.hypot(self.state.sigma_m, self.drift))
            pred = Calib(Hp, sp, "propagated")
        self.last_pred = pred
        meas = (Calib(H_kp, max(sigma_kp, self.kp_floor), "keypoints")
                if well_conditioned(H_kp) else None)

        if pred is None and meas is None:
            self.state = Calib(None, np.inf, "none")
        elif pred is None or (pred.sigma_m > self.max_sigma and meas is not None):
            self.state = meas
        elif meas is None:
            self.state = pred
        else:
            Pp, Pm = apply_h(pred.H, self.grid), apply_h(meas.H, self.grid)
            ok = _on_pitch(Pp) & _on_pitch(Pm)
            if ok.sum() < 6:
                self.state = meas
            else:
                d = np.median(np.linalg.norm(Pp[ok] - Pm[ok], axis=1))
                s_tot = np.hypot(pred.sigma_m, meas.sigma_m)
                if d > self.gate * s_tot:
                    # inconsistent: trust the propagated estimate unless it is stale
                    self.state = pred
                else:
                    wp, wm = 1 / pred.sigma_m ** 2, 1 / meas.sigma_m ** 2
                    Pf = (wp * Pp[ok] + wm * Pm[ok]) / (wp + wm)
                    Hf, _ = cv2.findHomography(self.grid[ok], Pf, 0)
                    sf = (wp + wm) ** -0.5
                    self.state = Calib(normalize_h(Hf), sf, "fused") if well_conditioned(Hf) else meas
        return self.state


def fuse_calibs(a: Optional[Calib], b: Optional[Calib], grid: np.ndarray) -> Calib:
    """Inverse-variance fusion of two calibration estimates on the image grid."""
    if a is None or a.H is None:
        return b if b is not None else Calib(None, np.inf, "none")
    if b is None or b.H is None:
        return a
    Pa, Pb = apply_h(a.H, grid), apply_h(b.H, grid)
    ok = _on_pitch(Pa) & _on_pitch(Pb)
    if ok.sum() < 6:
        return a if a.sigma_m <= b.sigma_m else b
    wa, wb = 1 / a.sigma_m ** 2, 1 / b.sigma_m ** 2
    Hf, _ = cv2.findHomography(grid[ok], (wa * Pa[ok] + wb * Pb[ok]) / (wa + wb), 0)
    if not well_conditioned(Hf):
        return a if a.sigma_m <= b.sigma_m else b
    return Calib(normalize_h(Hf), (wa + wb) ** -0.5, "smoothed")


def smooth_calibrations(H_kp: list, s_kp: list, M: np.ndarray, width: int, height: int,
                        **kw) -> list:
    """Offline two-filter calibration smoother (variant C2-smooth).

    Forward filter: frames 1..T with ego-motion M_t (t-1 -> t).
    Backward filter: frames T..1; going from t+1 to t the image map is M_{t+1}^{-1},
    so the "prev->cur" transform passed to the filter is inv(M_{t+1}).
    The smoothed estimate at t fuses the forward *posterior* with the backward
    *prediction* (which has not yet seen frame t's keypoints, avoiding double counting).
    Index 0 is unused (frames are 1-based), matching the SNMOT caches.
    """
    T = len(H_kp) - 1
    fw = CalibrationFilter(width, height, **kw)
    bw = CalibrationFilter(width, height, **kw)
    fpost = [None] * (T + 1)
    bpred = [None] * (T + 2)
    for t in range(1, T + 1):
        fpost[t] = fw.step(M[t], H_kp[t], s_kp[t])
    for t in range(T, 0, -1):
        Mi = np.linalg.inv(M[t + 1]) if t + 1 <= T else np.eye(3)
        bw.step(Mi, H_kp[t], s_kp[t])
        bpred[t] = bw.last_pred
    return [None] + [fuse_calibs(fpost[t], bpred[t], fw.grid) for t in range(1, T + 1)]
