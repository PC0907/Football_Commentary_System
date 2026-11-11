"""Configurable multi-object tracker for players, goalkeepers and referees.

One engine, several modelling choices, so that variants T0..T4 differ
*only* in the choices below and an ablation is meaningful:

  motion        'none'  : no motion model, IoU on last box               (T0, repo baseline)
                'image' : image-plane CV Kalman on (cx,cy,w,h)            (T1)
                'pitch' : pitch-plane CV Kalman on (X,Y,VX,VY) in metres  (T2+)
  cmc           warp the image-plane state by the frame-to-frame ego-motion homography
  het_R         measurement covariance propagated from pixels to metres through the
                homography Jacobian at the foot point (+ calibration variance)
  v_max         physiological speed limit (m/s): state-constraint projection after
                every update, and a *reachability gate* for lost tracks
                (|z - x_last| <= v_max * dt_lost + k sigma)
  assoc         'iou'   : 1 - IoU
                'mahal' : pitch-plane Mahalanobis (image IoU when no calibration)
                'fusion': negative log-likelihood fusion of independent sensors —
                          kinematic NLL (pitch or image) + appearance log-likelihood
                          ratio (calibrated on a tuning split)
  stitch        offline physics-feasible tracklet stitching (T4)

Detections: array rows [x, y, w, h, score, cls].  Classes: 0 player, 1 GK, 2 referee,
3 ball, 4 other.  Ball rows are ignored here (see ball.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..filters import CHI2_2D_999, Gaussian, cv_model, kf_predict, kf_update, project_speed
from ..geometry.camera import apply_h, h_jacobian

BIG = 1e6


@dataclass
class TrackerConfig:
    motion: str = "image"            # none | image | pitch
    cmc: bool = False
    het_R: bool = False
    v_max: Optional[float] = None    # m/s
    assoc: str = "iou"               # iou | mahal | fusion
    pitch_stage: str = "lost"        # 'lost': pitch kinematics only to re-identify lost
                                     #  tracks (time-scale-aware fusion); 'all': every track
    appearance: bool = False
    stitch: bool = False
    # generic
    fps: float = 25.0
    max_age: int = 75                # frames a lost track is kept (3 s)
    min_score_new: float = 0.5
    # score-aware association for real detectors (no effect with oracle boxes, score = 1)
    det_high: float = 0.0            # dets >= det_high take part in all stages (and may start tracks)
    det_low: float = 0.0             # det_low <= score < det_high: only extend existing tracks (IoU stage)
    min_track_len: int = 1           # offline: drop tracklets with fewer detections than this
    iou_gate: float = 0.2            # min IoU to match in IoU stages
    second_iou_stage: bool = True
    # pitch-plane model
    sigma_a: float = 4.0             # m/s^2 white-noise acceleration (players)
    foot_px_sigma: Tuple[float, float] = (0.15, 0.06)   # x: *w, y: *h  (pixel noise model)
    iso_R_m: float = 0.6             # used when het_R is False
    reach_k: float = 3.0
    # fusion
    app_weight: float = 1.0
    app_lr: Optional[Callable[[np.ndarray], np.ndarray]] = None   # rho -> log LR
    # stitching
    stitch_max_gap: int = 125        # frames
    stitch_v_max: float = 9.0
    stitch_app_min_llr: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Appearance sensor: torso colour histogram with grass suppressed
# ──────────────────────────────────────────────────────────────────────────────

def torso_hist(img: np.ndarray, box: np.ndarray, bins=(12, 4, 4)) -> Optional[np.ndarray]:
    import cv2
    x, y, w, h = box[:4]
    H, W = img.shape[:2]
    x0, x1 = int(max(0, x + 0.2 * w)), int(min(W, x + 0.8 * w))
    y0, y1 = int(max(0, y + 0.15 * h)), int(min(H, y + 0.6 * h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (30, 40, 40), (85, 255, 255))
    mask = cv2.bitwise_not(grass)
    if mask.sum() < 255 * 6:
        mask = None
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, list(bins), [0, 180, 0, 256, 0, 256]).ravel()
    s = hist.sum()
    return hist / s if s > 0 else None


def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.sqrt(a * b)))


# ──────────────────────────────────────────────────────────────────────────────
# Track
# ──────────────────────────────────────────────────────────────────────────────

class Track:
    _next = 1

    def __init__(self, det: np.ndarray, frame: int, cfg: TrackerConfig, z_pitch, R_pitch, hist):
        self.id = Track._next; Track._next += 1
        self.cls = int(det[5])
        self.box = det[:4].copy()
        self.cfg = cfg
        self.hits = 1
        self.last_frame = frame
        self.start_frame = frame
        self.hist = hist
        self.history: List[Tuple[int, np.ndarray, float, Optional[np.ndarray]]] = []  # frame, box, score, pitch
        # image KF on (cx, cy, w, h)
        z = _box_to_z(self.box)
        F, _ = cv_model(1.0, 1.0, 4)
        self.img = Gaussian(np.r_[z, np.zeros(4)], np.diag(np.r_[_img_std(z) ** 2, (10 * _img_vstd(z)) ** 2]))
        # pitch KF on (X, Y, VX, VY)
        self.pitch: Optional[Gaussian] = None
        self.pitch_last: Optional[Gaussian] = None
        if z_pitch is not None:
            self.pitch = Gaussian(np.r_[z_pitch, 0.0, 0.0], np.diag([R_pitch[0, 0], R_pitch[1, 1], 9.0, 9.0]))
            self.pitch_last = self.pitch.copy()
        self._record(frame, det[4])

    # -------------------------------------------------------------------------
    def predict(self, M: Optional[np.ndarray]):
        cfg = self.cfg
        if cfg.motion in ("image", "pitch"):
            if cfg.cmc and M is not None:
                self.img = _warp_img_state(self.img, M)
            F, _ = cv_model(1.0, 1.0, 4)
            z = self.img.x[:4]
            Q = np.diag(np.r_[_img_std(z) ** 2, _img_vstd(z) ** 2])
            self.img = kf_predict(self.img, F, Q)
        if cfg.motion == "pitch" and self.pitch is not None:
            F, Q = cv_model(1.0 / cfg.fps, cfg.sigma_a, 2)
            self.pitch = kf_predict(self.pitch, F, Q)
            if cfg.v_max:
                self.pitch = project_speed(self.pitch, [2, 3], cfg.v_max)

    def predicted_box(self) -> np.ndarray:
        if self.cfg.motion == "none":
            return self.box
        return _z_to_box(self.img.x[:4])

    def update(self, det: np.ndarray, frame: int, z_pitch, R_pitch, hist):
        cfg = self.cfg
        self.box = det[:4].copy()
        z = _box_to_z(self.box)
        if cfg.motion in ("image", "pitch"):
            Hm = np.eye(4, 8)
            self.img, _ = kf_update(self.img, Hm, np.diag(_img_std(z) ** 2), z)
        if cfg.motion == "pitch" and z_pitch is not None:
            if self.pitch is None:
                self.pitch = Gaussian(np.r_[z_pitch, 0, 0], np.diag([R_pitch[0, 0], R_pitch[1, 1], 9.0, 9.0]))
            else:
                self.pitch, _ = kf_update(self.pitch, np.eye(2, 4), R_pitch, z_pitch)
                if cfg.v_max:
                    self.pitch = project_speed(self.pitch, [2, 3], cfg.v_max)
            self.pitch_last = self.pitch.copy()
        if hist is not None:
            self.hist = hist if self.hist is None else 0.9 * self.hist + 0.1 * hist
        self.hits += 1
        self.last_frame = frame
        self._record(frame, det[4])

    def _record(self, frame, score):
        p = None if self.pitch is None else self.pitch.x[:2].copy()
        self.history.append((frame, self.box.copy(), float(score), p))


def _box_to_z(b):
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]], float)


def _z_to_box(z):
    return np.array([z[0] - z[2] / 2, z[1] - z[3] / 2, z[2], z[3]], float)


def _img_std(z):
    h = max(z[3], 1.0)
    return np.array([h / 20, h / 20, h / 20, h / 20])


def _img_vstd(z):
    h = max(z[3], 1.0)
    return np.array([h / 160, h / 160, h / 160, h / 160])


def _warp_img_state(g: Gaussian, M: np.ndarray) -> Gaussian:
    """Push the image-plane box state through the ego-motion homography M (BoT-SORT
    style, but with the full homography linearised at the box centre)."""
    c = g.x[:2]
    c2 = apply_h(M, c[None])[0]
    J = h_jacobian(M, c)
    s = float(np.sqrt(abs(np.linalg.det(J))))       # local zoom factor
    A = np.eye(8)
    A[0:2, 0:2] = J; A[4:6, 4:6] = J
    A[2, 2] = A[3, 3] = A[6, 6] = A[7, 7] = s
    x = A @ g.x
    x[0:2] = c2
    return Gaussian(x, A @ g.P @ A.T)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0], a[:, 1]; ax2, ay2 = ax1 + a[:, 2], ay1 + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]; bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    ua = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(ua, 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# Tracker
# ──────────────────────────────────────────────────────────────────────────────

class MultiTracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self.finished: List[Track] = []
        Track._next = 1

    # measurement model in the pitch plane --------------------------------------
    def _pitch_meas(self, dets: np.ndarray, H: Optional[np.ndarray], sigma_cal: float):
        if H is None or len(dets) == 0:
            return [None] * len(dets), [None] * len(dets)
        cfg = self.cfg
        feet = np.column_stack([dets[:, 0] + dets[:, 2] / 2, dets[:, 1] + dets[:, 3]])
        Z = apply_h(H, feet)
        Rs = []
        for u, d in zip(feet, dets):
            if cfg.het_R:
                J = h_jacobian(H, u)
                Rpx = np.diag([(cfg.foot_px_sigma[0] * d[2]) ** 2, (cfg.foot_px_sigma[1] * d[3]) ** 2])
                R = J @ Rpx @ J.T + (sigma_cal ** 2) * np.eye(2)
            else:
                R = (cfg.iso_R_m ** 2) * np.eye(2)
            Rs.append(R)
        return list(Z), Rs

    def _kin_cost(self, tracks: List[Track], Z, Rs, frame) -> np.ndarray:
        """Pitch-plane kinematic negative log-likelihood with chi2 + reachability gates."""
        cfg = self.cfg
        C = np.full((len(tracks), len(Z)), BIG)
        for i, t in enumerate(tracks):
            if t.pitch is None:
                continue
            P = t.pitch.P[:2, :2]; x = t.pitch.x[:2]
            dt = (frame - t.last_frame) / cfg.fps
            for j, (z, R) in enumerate(zip(Z, Rs)):
                if z is None:
                    continue
                S = P + R
                y = z - x
                d2 = float(y @ np.linalg.solve(S, y))
                if cfg.v_max and t.pitch_last is not None and dt > 1.5 / cfg.fps:
                    # physical reachability from the last confirmed position
                    reach = cfg.v_max * dt + cfg.reach_k * np.sqrt(np.trace(R) / 2 + np.trace(t.pitch_last.P[:2, :2]) / 2)
                    if np.linalg.norm(z - t.pitch_last.x[:2]) > reach:
                        continue
                if d2 > CHI2_2D_999 * (1 + 2 * (dt > 0.5)):
                    continue
                C[i, j] = 0.5 * d2 + 0.5 * np.log(np.linalg.det(S))
        return C

    def _app_cost(self, tracks, hists) -> np.ndarray:
        C = np.zeros((len(tracks), len(hists)))
        if self.cfg.app_lr is None:
            return C
        for i, t in enumerate(tracks):
            for j, h in enumerate(hists):
                if t.hist is None or h is None:
                    continue
                rho = bhattacharyya(t.hist, h)
                C[i, j] = -float(self.cfg.app_lr(np.array([rho]))[0])
        return C

    def step(self, frame: int, dets: np.ndarray, M: Optional[np.ndarray] = None,
             H: Optional[np.ndarray] = None, sigma_cal: float = 1.0, img: Optional[np.ndarray] = None):
        cfg = self.cfg
        dets = dets[dets[:, 5] != 3] if len(dets) else dets        # no ball here
        for t in self.tracks:
            t.predict(M)
        use_pitch = cfg.motion == "pitch" and H is not None
        Z, Rs = self._pitch_meas(dets, H if use_pitch else None, sigma_cal)
        hists = [torso_hist(img, d) if (cfg.appearance and img is not None) else None for d in dets]

        unmatched_t = list(range(len(self.tracks)))
        sc = dets[:, 4] if len(dets) else np.zeros(0)
        unmatched_d = [j for j in range(len(dets)) if sc[j] >= cfg.det_high]
        low_d = [j for j in range(len(dets)) if cfg.det_low <= sc[j] < cfg.det_high]
        matches: List[Tuple[int, int]] = []

        def solve(ti, di, C, max_cost):
            if not ti or not di:
                return [], ti, di
            sub = C[np.ix_(ti, di)]
            # class gating
            for a, i in enumerate(ti):
                for b, j in enumerate(di):
                    if self.tracks[i].cls != int(dets[j, 5]) and not ({self.tracks[i].cls, int(dets[j, 5])} <= {0, 1}):
                        sub[a, b] = BIG
            r, c = linear_sum_assignment(sub)
            m, used_t, used_d = [], set(), set()
            for a, b in zip(r, c):
                if sub[a, b] < max_cost:
                    m.append((ti[a], di[b])); used_t.add(ti[a]); used_d.add(di[b])
            return m, [i for i in ti if i not in used_t], [j for j in di if j not in used_d]

        tb = np.array([t.predicted_box() for t in self.tracks]) if self.tracks else np.zeros((0, 4))
        IoU = _iou_matrix(tb, dets[:, :4]) if len(dets) else np.zeros((len(self.tracks), 0))

        A = self._app_cost(self.tracks, hists) if (cfg.assoc == "fusion" and cfg.appearance) else None

        if cfg.pitch_stage == "all" and use_pitch and cfg.assoc != "iou":
            # (ablation) pitch-plane kinematics for every track, every frame
            C1 = self._kin_cost(self.tracks, Z, Rs, frame)
            if A is not None:
                C1 = np.where(C1 < BIG, C1 + cfg.app_weight * A, BIG)
            m, unmatched_t, unmatched_d = solve(unmatched_t, unmatched_d, C1, BIG / 2)
            matches += m
        else:
            # Stage 1 — short horizon: tracks seen in the previous frame, image-plane
            # IoU after CMC-compensated prediction (precise frame-to-frame sensor)
            if cfg.pitch_stage == "lost" and cfg.assoc != "iou":
                recent = [i for i in unmatched_t if frame - self.tracks[i].last_frame <= 1]
            else:
                recent = list(unmatched_t)
            C1 = 1.0 - IoU
            if A is not None:
                C1 = C1 + cfg.app_weight * 0.1 * A
            m, rest_t, unmatched_d = solve(recent, unmatched_d, C1, 1.0 - cfg.iou_gate)
            matches += m
            unmatched_t = [i for i in unmatched_t if i not in recent] + rest_t
            # Stage 2 — long horizon: lost tracks re-identified in the pitch plane with
            # physics (reachability gate, speed-limited prediction, metric Mahalanobis)
            if use_pitch and cfg.assoc != "iou" and unmatched_t and unmatched_d:
                C2 = self._kin_cost(self.tracks, Z, Rs, frame)
                if A is not None:
                    C2 = np.where(C2 < BIG, C2 + cfg.app_weight * A, BIG)
                m, unmatched_t, unmatched_d = solve(unmatched_t, unmatched_d, C2, BIG / 2)
                matches += m

        # Stage 2: IoU recovery for what the primary stage could not explain
        if cfg.second_iou_stage and unmatched_t and unmatched_d and cfg.assoc != "iou":
            m, unmatched_t, unmatched_d = solve(unmatched_t, unmatched_d, 1.0 - IoU, 1.0 - cfg.iou_gate)
            matches += m

        # Low-confidence detections may only extend tracks the confident ones left unexplained
        if low_d and unmatched_t:
            m, unmatched_t, _ = solve(unmatched_t, low_d, 1.0 - IoU, 1.0 - max(cfg.iou_gate, 0.5))
            matches += m

        for i, j in matches:
            self.tracks[i].update(dets[j], frame, Z[j], Rs[j], hists[j])
        for j in unmatched_d:
            if dets[j, 4] >= cfg.min_score_new:
                self.tracks.append(Track(dets[j], frame, cfg, Z[j], Rs[j], hists[j]))
        alive = []
        for t in self.tracks:
            if frame - t.last_frame > cfg.max_age:
                self.finished.append(t)
            else:
                alive.append(t)
        self.tracks = alive
        return [(frame, t.id, *t.box, 1.0) for t in self.tracks if t.last_frame == frame]

    # -------------------------------------------------------------------------
    def all_tracks(self) -> List[Track]:
        return self.finished + self.tracks

    def results(self) -> List[tuple]:
        tracks = self.all_tracks()
        id_map = stitch_tracklets(tracks, self.cfg) if self.cfg.stitch else {t.id: t.id for t in tracks}
        rows = []
        for t in tracks:
            if len(t.history) < self.cfg.min_track_len:
                continue
            for fr, box, s, _ in t.history:
                rows.append((fr, id_map[t.id], *box, s))
        return rows


# ──────────────────────────────────────────────────────────────────────────────
# Offline physics-feasible stitching (T4)
# ──────────────────────────────────────────────────────────────────────────────

def stitch_tracklets(tracks: List[Track], cfg: TrackerConfig) -> Dict[int, int]:
    """Join tracklet A -> B (A ends before B starts) when the implied pitch-plane speed
    across the gap is physically feasible and appearance agrees. Global min-cost
    one-to-one assignment over all (end, start) pairs; chains are resolved by
    union-find."""
    ends = [t for t in tracks if t.history]
    n = len(ends)
    C = np.full((n, n), BIG)
    for i, a in enumerate(ends):
        fa, _, _, pa = a.history[-1]
        for j, b in enumerate(ends):
            if i == j or a.cls != b.cls and not ({a.cls, b.cls} <= {0, 1}):
                continue
            fb, _, _, pb = b.history[0]
            gap = fb - fa
            if gap <= 0 or gap > cfg.stitch_max_gap:
                continue
            # temporal overlap check is implicit (gap > 0); also require no frame overlap
            if pa is None or pb is None:
                continue
            dt = gap / cfg.fps
            v = np.linalg.norm(pb - pa) / dt
            if v > cfg.stitch_v_max:
                continue
            llr = 0.0
            if cfg.app_lr is not None and a.hist is not None and b.hist is not None:
                llr = float(cfg.app_lr(np.array([bhattacharyya(a.hist, b.hist)]))[0])
                if llr < cfg.stitch_app_min_llr:
                    continue
            C[i, j] = v / cfg.stitch_v_max + 0.02 * gap / cfg.fps - 0.2 * llr
    parent = {t.id: t.id for t in tracks}
    if n:
        r, c = linear_sum_assignment(C)
        for i, j in zip(r, c):
            if C[i, j] < BIG / 2:
                parent[ends[j].id] = ends[i].id

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    return {t.id: find(t.id) for t in tracks}
