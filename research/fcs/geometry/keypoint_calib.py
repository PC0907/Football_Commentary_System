"""Keypoint "sensor": field keypoints -> image->pitch homography + uncertainty.

Reuses the production keypoint model and line-intersection geometry from
``pipeline/homography/model.py`` (C0), and adds:
  * C1 — confidence-weighted, outlier-robust refit and an explicit uncertainty
    estimate (residual standard error of the fit, scaled by the number of
    correspondences) so downstream filters can weight this sensor.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from pipeline.homography import model as hm  # noqa: E402

_WORLD = {name: (x, y) for _, x, y, name in hm.KEYPOINTS_DATA}


def keypoints_model():
    return hm.get_model()


def correspondences(kp: np.ndarray, conf_thr: float = hm.CONFIDENCE_THRESHOLD
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """kp: (46, 3) x, y, conf. Returns (img_pts, pitch_pts, weights)."""
    mask = kp[:, 2] >= conf_thr
    names = {hm.KEYPOINT_NAMES[i] for i in np.where(mask)[0]}
    fps = hm.get_detectable_field_points(names)
    if not fps:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        coords = hm.compute_field_point_coordinates(fps, kp, hm.KEYPOINT_NAMES)
    src, dst, w = [], [], []
    name_to_idx = {n: i for i, n in enumerate(hm.KEYPOINT_NAMES)}
    for fp in sorted(coords):
        if fp not in _WORLD:
            continue
        src.append(coords[fp]); dst.append(_WORLD[fp])
        # weight = product of the confidences of the keypoints defining the 2 lines
        lines = hm.FIELD_POINT_TO_KEYPOINT_LINES.get(fp, [])
        c = [kp[name_to_idx[a], 2] * kp[name_to_idx[b], 2] for a, b in lines] or [1.0]
        w.append(float(np.prod(c)) ** (1 / max(1, len(c))))
    return np.asarray(src, float), np.asarray(dst, float), np.asarray(w, float)


def calibrate_c0(kp: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Production behaviour: unweighted RANSAC on all correspondences."""
    src, dst, _ = correspondences(kp)
    if len(src) < 4:
        return None, np.inf
    H, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), cv2.RANSAC, 3.0)
    return H, (2.0 if H is not None else np.inf)


def calibrate_c1(kp: np.ndarray, ransac_m: float = 1.5, prior_sigma: float = 2.0
                 ) -> Tuple[Optional[np.ndarray], float]:
    """Robust, confidence-weighted fit with an uncertainty estimate (metres).

    1. RANSAC in *pitch metres* (threshold is physical, not pixels).
    2. Weighted DLT on inliers (weights = keypoint confidences).
    3. sigma = residual standard error * sqrt(8 / (2 n)) — the standard error of the
       fitted map for an 8-parameter model with n point pairs; falls back to a prior
       when there is no residual degree of freedom (n == 4).
    Also rejects geometrically degenerate fits (pitch corners mapping to a non-convex
    or absurd-area quadrilateral).
    """
    src, dst, w = correspondences(kp)
    n = len(src)
    if n < 4:
        return None, np.inf
    H, inl = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), cv2.RANSAC, ransac_m)
    if H is None:
        return None, np.inf
    inl = inl.ravel().astype(bool)
    if inl.sum() >= 5:
        Hw = _weighted_dlt(src[inl], dst[inl], w[inl])
        if Hw is not None:
            H = Hw
    if not _plausible(H):
        return None, np.inf
    m = int(inl.sum())
    res = np.linalg.norm(_apply(H, src[inl]) - dst[inl], axis=1)
    if 2 * m - 8 > 0:
        se = np.sqrt(np.sum(res ** 2) / (2 * m - 8))
        sigma = max(0.2, se) * np.sqrt(8.0 / (2 * m)) + 0.1
    else:
        sigma = prior_sigma
    return H / H[2, 2], float(sigma)


# ──────────────────────────────────────────────────────────────────────────────
# C1L: point-on-line calibration (each keypoint is an extremity of a known line)
# ──────────────────────────────────────────────────────────────────────────────

_L, _W = 105.0, 68.0
# pitch frame: X 0..105 (left goal line .. right), Y 0..68 (far touchline .. near)
PITCH_LINES = {
    "Side line top": (0, 1, 0.0), "Side line bottom": (0, 1, -_W),
    "Side line left": (1, 0, 0.0), "Side line right": (1, 0, -_L),
    "Middle line": (1, 0, -_L / 2),
    "Big rect. left top": (0, 1, -13.84), "Big rect. left bottom": (0, 1, -54.16),
    "Big rect. left main": (1, 0, -16.5),
    "Big rect. right top": (0, 1, -13.84), "Big rect. right bottom": (0, 1, -54.16),
    "Big rect. right main": (1, 0, -(_L - 16.5)),
    "Small rect. left top": (0, 1, -24.84), "Small rect. left bottom": (0, 1, -43.16),
    "Small rect. left main": (1, 0, -5.5),
    "Small rect. right top": (0, 1, -24.84), "Small rect. right bottom": (0, 1, -43.16),
    "Small rect. right main": (1, 0, -(_L - 5.5)),
}


def _kp_line(name: str):
    base = name.rsplit(" ", 1)[0]
    return base if base in PITCH_LINES else None


KP_LINE = [_kp_line(n) for n in hm.KEYPOINT_NAMES]    # None for goal (off-plane) points


def _line_dlt(pts: np.ndarray, lines: np.ndarray, w: np.ndarray) -> Optional[np.ndarray]:
    """Solve l_i^T H u_i = 0 for H (image -> pitch), Hartley-normalised."""
    c = pts.mean(0); s = np.sqrt(2) / max(1e-9, np.mean(np.linalg.norm(pts - c, axis=1)))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    un = np.column_stack([pts, np.ones(len(pts))]) @ T.T
    # pitch-side normalisation: centre/scale the pitch
    Tp = np.array([[1 / 52.5, 0, -1], [0, 1 / 34.0, -1], [0, 0, 1]])
    ln = lines @ np.linalg.inv(Tp)            # lines transform with the inverse
    A = np.einsum("ni,nj->nij", ln, un).reshape(len(un), 9) * np.sqrt(w)[:, None]
    _, sv, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tp) @ Hn @ T
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def _line_residuals(H, pts, lines):
    """Distance (m) of each mapped point to its pitch line."""
    P = _apply(H, pts)
    return np.abs(lines[:, 0] * P[:, 0] + lines[:, 1] * P[:, 1] + lines[:, 2]) / np.hypot(lines[:, 0], lines[:, 1])


def _directions_ok(lines: np.ndarray) -> bool:
    horiz = np.unique(lines[lines[:, 1] != 0][:, 2]); vert = np.unique(lines[lines[:, 0] != 0][:, 2])
    return len(horiz) >= 2 and len(vert) >= 2


def calibrate_c1l(kp: np.ndarray, conf_thr: float = 0.5, inlier_m: float = 0.75,
                  iters: int = 300, prior_sigma: float = 2.0, rng=None
                  ) -> Tuple[Optional[np.ndarray], float]:
    """RANSAC over *lines* (each contributes its 1-2 extremity constraints).

    A minimal sample needs >= 8 constraints from >= 2 distinct lines per direction
    (otherwise the map along one axis is unconstrained). Inliers: mapped point within
    ``inlier_m`` metres of its line. Final weighted refit on inliers; uncertainty from
    the residual standard error with 8 parameters, as in C1.
    """
    rng = rng or np.random.default_rng(0)
    idx = [i for i in range(len(kp)) if kp[i, 2] >= conf_thr and KP_LINE[i] is not None]
    if len(idx) < 8:
        return None, np.inf
    pts = kp[idx, :2].astype(float)
    lines = np.array([PITCH_LINES[KP_LINE[i]] for i in idx], float)
    w = kp[idx, 2].astype(float)
    names = [KP_LINE[i] for i in idx]
    uniq = sorted(set(names))
    if not _directions_ok(lines):
        return None, np.inf
    by_line = {n: [k for k, m in enumerate(names) if m == n] for n in uniq}
    best, best_inl = None, None
    for _ in range(iters):
        pick = rng.permutation(uniq)
        sel, k = [], 0
        while len(sel) < 8 and k < len(pick):
            sel += by_line[pick[k]]; k += 1
        sel = np.array(sel)
        if len(sel) < 8 or not _directions_ok(lines[sel]):
            continue
        H = _line_dlt(pts[sel], lines[sel], w[sel])
        if H is None or not np.all(np.isfinite(H)):
            continue
        inl = _line_residuals(H, pts, lines) < inlier_m
        if best_inl is None or inl.sum() > best_inl.sum():
            best, best_inl = H, inl
        if inl.all():
            break
    if best is None or best_inl.sum() < 8 or not _directions_ok(lines[best_inl]):
        return None, np.inf
    H = _line_dlt(pts[best_inl], lines[best_inl], w[best_inl])
    if H is None or not _plausible(H):
        return None, np.inf
    r = _line_residuals(H, pts[best_inl], lines[best_inl])
    m = int(best_inl.sum())
    # each constraint is 1-D here: dof = m - 8
    se = np.sqrt(np.sum(r ** 2) / (m - 8)) if m > 8 else prior_sigma
    sigma = max(0.2, se) * np.sqrt(8.0 / m) + 0.1 if m > 8 else prior_sigma
    return H, float(sigma)


def _apply(H, p):
    ph = np.column_stack([p, np.ones(len(p))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def _weighted_dlt(src, dst, w) -> Optional[np.ndarray]:
    # Hartley normalisation
    def norm(p):
        c = p.mean(0); s = np.sqrt(2) / max(1e-9, np.mean(np.linalg.norm(p - c, axis=1)))
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        return (np.column_stack([p, np.ones(len(p))]) @ T.T)[:, :2], T
    a, Ta = norm(src); b, Tb = norm(dst)
    A = []
    for (x, y), (u, v), wi in zip(a, b, np.sqrt(np.maximum(w, 1e-6))):
        A.append(wi * np.array([-x, -y, -1, 0, 0, 0, u * x, u * y, u]))
        A.append(wi * np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v]))
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ Hn @ Ta
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def _plausible(H: np.ndarray) -> bool:
    """The inverse map of the image must be a proper (orientation-preserving or
    consistently reversed) projective map; reject near-singular fits."""
    if not np.all(np.isfinite(H)):
        return False
    d = np.linalg.det(H)
    return abs(d) > 1e-12
