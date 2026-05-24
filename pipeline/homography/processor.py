"""
HomographyProcessor — wraps homography.py so the VideoProcessor pipeline can
call a single `.process(frame, detections)` method and get back world-coordinate
positions plus a confidence score.

Uses the cached YOLO keypoint model (best.pt in this directory) via the
`get_model()` singleton in homography.py to avoid reloading on every frame.
"""

import logging
from typing import List, Dict, Tuple, Any

import numpy as np

from .model import (
    get_model,
    KEYPOINT_NAMES,
    CONFIDENCE_THRESHOLD,
    KEYPOINTS_DATA,
    get_detectable_field_points,
    compute_field_point_coordinates,
    calculate_homography_matrix,
    transform_object_positions,
)

log = logging.getLogger(__name__)


class HomographyProcessor:
    """
    Thin wrapper around homography.py for the UI pipeline.

    Keeps the last valid homography matrix so that frames where keypoint
    detection is poor (cloudy pitch view, camera pan) can still produce
    approximate minimap positions.
    """

    # EMA weight applied to each new raw H matrix.
    # Low alpha = smoother but slightly lagged; 0.2 is a good default for 30 fps.
    _H_ALPHA: float = 0.2

    def __init__(self):
        # Cached / EMA-smoothed homography matrix
        self._H:        np.ndarray | None = None   # raw last-good H (fallback)
        self._H_smooth: np.ndarray | None = None   # EMA-smoothed H (used for transforms)
        self.last_confidence: float = 0.0
        # Maps track_id → YOLO class (object_id) for the current frame
        self._class_map: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run keypoint detection on *frame*, compute the homography, and map
        each detection's foot-point to real-world pitch coordinates (metres).

        Parameters
        ----------
        frame      : BGR numpy frame from the video.
        detections : List of detection dicts.  Each must contain at least
                     one of the following coordinate keys:
                       pixel_x / pixel_y  (preferred, foot-point centre)
                       center_x / center_y (fallback)
                     and an identifier key: track_id or object_id.

        Returns
        -------
        field_frame      : Copy of the input frame (homography is used only
                           for coordinate mapping, not for warping the image).
        field_positions  : List of dicts {object_id, world_x_meters,
                           world_y_meters}.  Empty on failure.
        """
        object_positions = self._build_object_positions(detections)

        try:
            transformed, confidence = self._run_homography(frame, object_positions)
            self.last_confidence = confidence
            self._restore_class_ids(transformed)
            return frame.copy(), transformed

        except ValueError as exc:
            # Not enough keypoints this frame — try the cached matrix
            log.debug("Homography skipped (not enough keypoints): %s", exc)
            if self._H is not None and object_positions:
                try:
                    transformed = transform_object_positions(object_positions, self._H)
                    self._restore_class_ids(transformed)
                    self.last_confidence = max(0.0, self.last_confidence - 0.05)
                    return frame.copy(), transformed
                except Exception as exc2:
                    log.debug("Cached homography also failed: %s", exc2)
            self.last_confidence = 0.0
            return frame.copy(), []

        except Exception as exc:
            log.error("HomographyProcessor error: %s", exc, exc_info=True)
            self.last_confidence = 0.0
            return frame.copy(), []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_object_positions(
        self, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert detection dicts to the format expected by homography.py.

        homography.py uses ``object_id`` as a unique identifier per object;
        we use ``track_id`` for that role so each tracked object keeps a stable
        identity across frames.  The true YOLO class (0=Player-L … 4=Ball) is
        saved in ``self._class_map`` and restored after the transform so that
        field_positions carry both ``track_id`` and the correct ``object_id``
        (YOLO class) for minimap colour lookup.
        """
        positions = []
        self._class_map = {}
        for i, d in enumerate(detections):
            track_id   = int(d.get("track_id", i))
            yolo_class = int(d.get("object_id", -1))
            self._class_map[track_id] = yolo_class
            positions.append(
                {
                    "object_id": track_id,   # homography uses this as a unique ID
                    "pixel_x":   float(d.get("pixel_x", d.get("center_x", 0))),
                    "pixel_y":   float(d.get("pixel_y", d.get("center_y", 0))),
                }
            )
        return positions

    def _restore_class_ids(self, transformed: List[Dict[str, Any]]) -> None:
        """
        After transform, ``object_id`` still holds the track_id value.
        Rename it to ``track_id`` and restore the real YOLO class as
        ``object_id`` so downstream consumers (minimap, event detector) see
        the correct values.
        """
        for pos in transformed:
            tid = pos.get("object_id", -1)
            pos["track_id"]  = tid
            pos["object_id"] = self._class_map.get(tid, -1)

    def _run_homography(
        self,
        frame: np.ndarray,
        object_positions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Detect keypoints, compute H, transform positions.

        Returns (transformed_positions, confidence_score).
        Raises ValueError if fewer than 4 correspondences are found.
        """
        model = get_model()
        results = model(frame, verbose=False)[0]

        # keypoints tensor: shape (N_kpts, 3) — x, y, confidence
        kpts = results.keypoints.data.cpu().numpy().reshape(-1, 3)

        # Filter by confidence threshold
        high_conf_mask = kpts[:, 2] >= CONFIDENCE_THRESHOLD
        detected_names = {KEYPOINT_NAMES[i] for i in np.where(high_conf_mask)[0]}

        detectable = get_detectable_field_points(detected_names)
        field_coords = compute_field_point_coordinates(detectable, kpts, KEYPOINT_NAMES)

        # Raises ValueError when < 4 correspondences
        H_raw = calculate_homography_matrix(field_coords, KEYPOINTS_DATA)
        if H_raw is None:
            raise ValueError("findHomography returned None — collinear or insufficient points")

        self._H = H_raw  # keep raw copy for fallback

        # ── EMA-smooth H to kill per-frame RANSAC jitter ─────────────────────
        if self._H_smooth is None:
            self._H_smooth = H_raw.copy()
        else:
            self._H_smooth = (
                self._H_ALPHA * H_raw + (1.0 - self._H_ALPHA) * self._H_smooth
            )
        H = self._H_smooth   # use smoothed matrix for all downstream work

        # Confidence: mix of keypoint ratio, mean confidence, and reprojection quality
        num_detected = int(np.sum(high_conf_mask))
        kpt_ratio = num_detected / len(KEYPOINT_NAMES)
        mean_conf = float(np.mean(kpts[high_conf_mask, 2])) if np.any(high_conf_mask) else 0.0

        # Reprojection error (uses smoothed H so reprojection improves over time)
        world_map = {name: (x, y) for _, x, y, name in KEYPOINTS_DATA}
        src, dst = [], []
        for fp_name, (px, py) in field_coords.items():
            if fp_name in world_map:
                src.append([px, py])
                dst.append(list(world_map[fp_name]))
        if src:
            src_arr = np.array(src, dtype=np.float32)
            dst_arr = np.array(dst, dtype=np.float32)
            src_h = np.column_stack([src_arr, np.ones(len(src_arr))])
            proj = (H @ src_h.T).T
            proj = proj[:, :2] / proj[:, 2:]
            err = float(np.mean(np.linalg.norm(proj - dst_arr, axis=1)))
            error_score = max(0.0, 1.0 - err / 5.0)
        else:
            error_score = 0.0

        confidence = 0.4 * kpt_ratio + 0.4 * mean_conf + 0.2 * error_score

        transformed = transform_object_positions(object_positions, H)
        return transformed, confidence
