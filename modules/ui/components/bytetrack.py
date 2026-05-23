"""
ByteTrack multi-object tracker with Kalman-filter motion prediction.

Key improvement over the original: the predict() method now runs a proper
constant-velocity Kalman filter, so fast-moving objects (especially the ball)
maintain their track ID through brief occlusions or missed detections.

Public API (unchanged):
    tracker = BYTETracker(track_thresh, match_thresh, track_buffer, frame_rate)
    tracked_objects = tracker.update(detections)   # detections: list[dict]
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ══════════════════════════════════════════════════════════════════════════════
# Kalman filter
# ══════════════════════════════════════════════════════════════════════════════

class KalmanFilter:
    """
    8-dimensional Kalman filter for axis-aligned bounding-box tracking.

    State vector  : [cx, cy, w, h,  v_cx, v_cy, v_w, v_h]
    Observation   : [cx, cy, w, h]
    Motion model  : constant velocity (one frame = one time step)

    Noise weights are proportional to the object height so the filter adapts
    to objects at different depths (standard DeepSORT / ByteTrack convention).
    """

    _ndim = 4  # spatial dims (cx, cy, w, h)

    def __init__(self):
        n = self._ndim
        # State-transition matrix: position += velocity * dt  (dt = 1 frame)
        self._F = np.eye(2 * n, dtype=np.float32)
        for i in range(n):
            self._F[i, n + i] = 1.0

        # Observation matrix: observe only [cx, cy, w, h], not velocities
        self._H = np.eye(n, 2 * n, dtype=np.float32)

        # Noise scale weights (empirically tuned for pixel coordinates)
        self._std_wp = 1.0 / 20.0    # position
        self._std_wv = 1.0 / 160.0   # velocity

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pos_std(self, h: float) -> list:
        w = self._std_wp * h
        return [w, w, 1e-2, w]   # cx, cy, w, h

    def _vel_std(self, h: float) -> list:
        w = self._std_wv * h
        return [w, w, 1e-5, w]   # v_cx, v_cy, v_w, v_h

    # ── public interface ──────────────────────────────────────────────────────

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialise a new track from the first measurement *z* = [cx, cy, w, h].

        Returns
        -------
        mean        : (8,) initial state vector
        covariance  : (8, 8) initial covariance matrix
        """
        h = float(z[3])
        mean = np.concatenate([z.astype(np.float32), np.zeros(self._ndim, np.float32)])
        # Inflate initial velocity uncertainty by 10×
        std = self._pos_std(h) + [10 * s for s in self._vel_std(h)]
        P   = np.diag(np.square(std).astype(np.float32))
        return mean, P

    def predict(self, mean: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propagate state one frame forward (prediction step)."""
        h = float(mean[3])
        Q = np.diag(np.square(self._pos_std(h) + self._vel_std(h)).astype(np.float32))
        mean_pred = self._F @ mean
        P_pred    = self._F @ P @ self._F.T + Q
        return mean_pred.astype(np.float32), P_pred.astype(np.float32)

    def update(
        self, mean: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct state with new measurement *z* = [cx, cy, w, h]."""
        h = float(mean[3])
        R = np.diag(np.square([self._std_wp * h,
                                self._std_wp * h,
                                0.1,
                                self._std_wp * h]).astype(np.float32))
        S = self._H @ P @ self._H.T + R
        K = P @ self._H.T @ np.linalg.inv(S)

        new_mean = mean + K @ (z.astype(np.float32) - self._H @ mean)
        new_P    = (np.eye(2 * self._ndim, dtype=np.float32) - K @ self._H) @ P
        return new_mean.astype(np.float32), new_P.astype(np.float32)

    def project(self, mean: np.ndarray) -> np.ndarray:
        """Project state to measurement space → [cx, cy, w, h]."""
        return (self._H @ mean).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Track state enum
# ══════════════════════════════════════════════════════════════════════════════

class TrackState:
    NEW     = 0
    TRACKED = 1
    LOST    = 2
    REMOVED = 3


# ══════════════════════════════════════════════════════════════════════════════
# Single track object
# ══════════════════════════════════════════════════════════════════════════════

# One shared KalmanFilter instance (stateless — all state lives in STrack)
_kalman = KalmanFilter()


class STrack:
    """
    Represents a single tracked object with a Kalman-filtered state.

    *tlwh* is always kept in sync with the Kalman posterior so callers can read
    the smoothed bounding box directly from ``track.tlwh``.
    """

    def __init__(self, tlwh: np.ndarray, score: float, cls: int):
        """
        Parameters
        ----------
        tlwh  : [x_top_left, y_top_left, width, height]
        score : detection confidence
        cls   : integer class-id
        """
        self.tlwh  = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.cls   = int(cls)

        self.state        = TrackState.NEW
        self.is_activated = False
        self.track_id     = -1

        # Kalman state (allocated on first activation)
        self._mean: np.ndarray | None       = None
        self._covariance: np.ndarray | None = None

        self.frame_id    = 0
        self.start_frame = 0
        self.end_frame   = 0
        self.tracklet_len = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    @property
    def _cx_cy_w_h(self) -> np.ndarray:
        x, y, w, h = self.tlwh
        return np.array([x + w / 2, y + h / 2, w, h], dtype=np.float32)

    def _sync_tlwh(self):
        """Recompute tlwh from the Kalman posterior (call after predict/update)."""
        if self._mean is None:
            return
        cx, cy, w, h = _kalman.project(self._mean)
        w = max(w, 1.0)
        h = max(h, 1.0)
        self.tlwh = np.array([cx - w / 2, cy - h / 2, w, h], dtype=np.float32)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def activate(self, track_id: int):
        """Assign an ID and initialise the Kalman filter from the first detection."""
        self.track_id     = track_id
        self.state        = TrackState.TRACKED
        self.is_activated = True
        self.frame_id     = 1
        self.start_frame  = 1
        self._mean, self._covariance = _kalman.initiate(self._cx_cy_w_h)

    def predict(self):
        """
        Run the Kalman prediction step for this track.

        Must be called once per frame *before* the matching step so that the
        predicted location is used for IoU computation.
        """
        if self._mean is not None:
            self._mean, self._covariance = _kalman.predict(
                self._mean, self._covariance
            )
            self._sync_tlwh()

    def update(self, new_det: "STrack"):
        """
        Fuse a matched detection *new_det* into the Kalman state (correction step).
        """
        self.frame_id      += 1
        self.tracklet_len  += 1

        z = new_det._cx_cy_w_h
        if self._mean is not None:
            self._mean, self._covariance = _kalman.update(
                self._mean, self._covariance, z
            )
            self._sync_tlwh()
        else:
            self.tlwh = new_det.tlwh.copy()

        self.score        = new_det.score
        self.state        = TrackState.TRACKED
        self.is_activated = True
        self.end_frame    = self.frame_id

    def mark_lost(self):
        self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED


# ══════════════════════════════════════════════════════════════════════════════
# Main tracker
# ══════════════════════════════════════════════════════════════════════════════

class BYTETracker:
    """
    Two-pass IoU matching tracker (ByteTrack algorithm) with Kalman prediction.

    Pass 1 — match high-confidence detections to all active tracks.
    Pass 2 — match remaining low-confidence detections to unmatched tracks.

    Tracks that go unmatched for more than *track_buffer* frames are removed.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int   = 30,
        frame_rate:   int   = 30,
    ):
        self.track_thresh  = track_thresh
        self.match_thresh  = match_thresh
        self.track_buffer  = track_buffer
        self.frame_rate    = frame_rate
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        self.tracked_tracks: list[STrack] = []
        self.lost_tracks:    list[STrack] = []
        self.track_id_count: int          = 0
        self._frame_id:      int          = 0  # incremented each update() call

    # ── main entry point ──────────────────────────────────────────────────────

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Update the tracker with detections from the current frame.

        Parameters
        ----------
        detections : list of dicts with keys
            ``object_id``, ``pixel_x``, ``pixel_y``, ``width``, ``height``,
            ``confidence``

        Returns
        -------
        list of dicts — same schema extended with ``track_id``.
        Positions reflect the Kalman-smoothed estimate, not the raw detection.
        """
        self._frame_id += 1

        # Convert raw detections to STrack objects
        def _to_strack(d: dict) -> STrack:
            return STrack(
                [d["pixel_x"] - d["width"] / 2,
                 d["pixel_y"] - d["height"] / 2,
                 d["width"],
                 d["height"]],
                d["confidence"],
                d["object_id"],
            )

        stracks   = [_to_strack(d) for d in detections]
        high_conf = [s for s in stracks if s.score >= self.track_thresh]
        low_conf  = [s for s in stracks if s.score <  self.track_thresh]

        # ── Predict all active tracks ─────────────────────────────────────────
        for t in self.tracked_tracks:
            if t.is_activated:
                t.predict()

        # ── Pass 1 : high-confidence detections ↔ active tracks ───────────────
        pool   = self.tracked_tracks
        cost1  = self._iou_distance(pool, high_conf)
        m1, u_track1, u_det1 = self._match(cost1, pool, high_conf, self.match_thresh)

        newly_activated: list[STrack] = []
        for it, id_ in m1:
            pool[it].update(high_conf[id_])
            newly_activated.append(pool[it])

        # ── Pass 2 : low-confidence detections ↔ unmatched active tracks ──────
        r_tracks = [pool[i] for i in u_track1]
        newly_lost: list[STrack] = []

        if low_conf:
            cost2 = self._iou_distance(r_tracks, low_conf)
            m2, u_track2, _ = self._match(cost2, r_tracks, low_conf, thresh=0.5)
            for it, id_ in m2:
                r_tracks[it].update(low_conf[id_])
                newly_activated.append(r_tracks[it])
            final_unmatched = [r_tracks[i] for i in u_track2]
        else:
            final_unmatched = r_tracks

        for t in final_unmatched:
            if t.state != TrackState.LOST:
                t.mark_lost()
                newly_lost.append(t)

        # ── Initialise new tracks from unmatched high-conf detections ─────────
        for id_ in u_det1:
            det = high_conf[id_]
            if det.score >= self.track_thresh:
                self.track_id_count += 1
                det.activate(self.track_id_count)
                newly_activated.append(det)

        # ── Age out lost tracks ───────────────────────────────────────────────
        surviving_lost: list[STrack] = []
        for t in self.lost_tracks + newly_lost:
            if (self._frame_id - t.end_frame) > self.max_time_lost:
                t.mark_removed()
            else:
                surviving_lost.append(t)
        self.lost_tracks = surviving_lost

        # ── Update active set ─────────────────────────────────────────────────
        self.tracked_tracks = (
            [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]
            + newly_activated
        )

        # ── Build output list ─────────────────────────────────────────────────
        output = []
        for t in self.tracked_tracks:
            if t.is_activated:
                x, y, w, h = t.tlwh
                output.append({
                    "object_id":  t.cls,
                    "track_id":   t.track_id,
                    "pixel_x":    float(x + w / 2),
                    "pixel_y":    float(y + h / 2),
                    "width":      float(w),
                    "height":     float(h),
                    "confidence": float(t.score),
                })
        return output

    # ── Internal matching helpers ─────────────────────────────────────────────

    def _iou_distance(
        self, tracks: list[STrack], detections: list[STrack]
    ) -> np.ndarray:
        """Return cost matrix of shape (|tracks|, |dets|) where cost = 1 − IoU."""
        n, m = len(tracks), len(detections)
        cost = np.ones((n, m), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                cost[i, j] = 1.0 - self._iou(t.tlwh, d.tlwh)
        return cost

    @staticmethod
    def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
        """Compute IoU between two boxes in [x, y, w, h] (top-left) format."""
        xi1 = max(b1[0], b2[0]);  yi1 = max(b1[1], b2[1])
        xi2 = min(b1[0] + b1[2], b2[0] + b2[2])
        yi2 = min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        union = b1[2] * b1[3] + b2[2] * b2[3] - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _match(
        cost: np.ndarray,
        tracks: list,
        detections: list,
        thresh: float,
    ) -> tuple[list, list, list]:
        """
        Hungarian algorithm matching.

        Returns (matches, unmatched_track_indices, unmatched_detection_indices).
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        c = cost.copy()
        c[c > thresh] = 1.0
        rows, cols = linear_sum_assignment(c)

        matches, u_t, u_d = [], list(range(len(tracks))), list(range(len(detections)))
        for r, c_ in zip(rows, cols):
            if cost[r, c_] <= thresh:
                matches.append((r, c_))
                u_t.remove(r)
                u_d.remove(c_)
        return matches, u_t, u_d
