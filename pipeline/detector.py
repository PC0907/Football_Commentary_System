"""
YOLO-based object detector for the Football Commentary UI pipeline.

Kit-colour helpers and rendering utilities live in utils/detection_utils.py —
this module imports from there to avoid duplication with 2Dview.py.
"""

import logging
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

log = logging.getLogger(__name__)

from .bytetrack import BYTETracker
from .utils import (
    LABELS, BOX_COLORS, TRACK_COLORS,
    get_grass_hsv, get_players_boxes,
    get_kits_colors, get_kits_classifier, classify_kits, get_left_team_label,
    draw_tracks,
)


class ObjectDetector:
    """
    YOLO-based object detector for the UI pipeline.

    The model emits raw class ids that represent generic player/GK/ball roles.
    On the first call to ``detect()`` we fit a 2-cluster KMeans on player
    kit colours so that subsequent frames can reliably label detections as
    Team-A (object_id 0), Team-B (object_id 1), GK-A (2), GK-B (3), or
    Ball (4) — the canonical ids expected by the minimap, event detector, and
    stats accumulator.

    Raw model class mapping (standalone script convention):
      0 → outfield player (any team) → re-labelled to 0 or 1 by kit colour
      1 → goalkeeper (any team)      → re-labelled to 2 or 3 by field half
      2 → ball                       → → 4
      3 → main referee               → → 5
      4 → side referee               → → 6
      5 → staff                      → → 7
    """

    _RAW_SHIFT = 2   # classes ≥ 2 are shifted up by _RAW_SHIFT (ball → 4 etc.)

    def __init__(self, model_path=None, conf_threshold: float = 0.5, device=None):
        self.model_path = (
            Path(model_path).resolve()
            if model_path
            else Path(__file__).resolve().parents[1] / "models" / "best_object.pt"
        )
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = self._load_model()

        # ── Kit-colour team classifier (initialised on first frame) ───────────
        self._kits_clf   = None        # fitted KMeans(n_clusters=2)
        self._left_label = 0           # KMeans cluster index that is Team-A
        self._grass_hsv  = None        # cached grass colour for masking

    def _load_model(self) -> YOLO:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Object detection model not found: {self.model_path}"
            )
        model = YOLO(str(self.model_path))
        if self.device:
            model.to(self.device)
        return model

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLO on *frame* and return canonical detection dicts.

        Keys: object_id, label, confidence, x1, y1, x2, y2,
              center_x, center_y, pixel_x, pixel_y, width, height.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        boxes  = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        cls_arr  = boxes.cls.cpu().numpy()
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()

        frame_w = frame.shape[1]

        # ── Build raw detection list & collect player crops ───────────────────
        raw: list[dict] = []
        player_crops: list[np.ndarray] = []
        player_indices: list[int] = []    # indices in raw[] that are outfield players

        for i in range(len(cls_arr)):
            raw_cls = int(cls_arr[i])
            x1, y1, x2, y2 = map(int, xyxy_arr[i])
            w  = x2 - x1
            h  = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            raw.append({
                "raw_cls": raw_cls,
                "confidence": float(conf_arr[i]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy, "w": w, "h": h,
            })
            if raw_cls == 0:   # outfield player
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    player_crops.append(crop)
                    player_indices.append(i)

        # ── Initialise kit-colour classifier on first frame ───────────────────
        if self._kits_clf is None and len(player_crops) >= 2:
            try:
                self._grass_hsv = get_grass_hsv(frame)
                kit_colors = get_kits_colors(player_crops, self._grass_hsv)
                self._kits_clf  = get_kits_classifier(kit_colors)
                # Which KMeans cluster sits on the left side of the frame?
                # get_left_team_label returns the cluster id for the left team
                # We need YOLO box objects — pass the boxes for player_indices
                player_boxes_subset = [boxes[j] for j in player_indices]
                self._left_label = get_left_team_label(
                    player_boxes_subset, kit_colors, self._kits_clf
                )
            except Exception as exc:
                log.warning("Kit-colour classifier init failed: %s", exc)

        # ── Assign per-player team via kit colour (if classifier is ready) ────
        kit_team: dict[int, int] = {}   # raw index → team (0 = left/Team-A, 1 = right/Team-B)
        if self._kits_clf is not None and player_crops:
            try:
                kit_colors = get_kits_colors(player_crops, self._grass_hsv)
                labels = self._kits_clf.predict(kit_colors)
                for seq_i, raw_i in enumerate(player_indices):
                    kit_team[raw_i] = int(labels[seq_i])
            except Exception as exc:
                log.debug("Kit-colour prediction failed: %s", exc)

        # ── Build canonical detection list ────────────────────────────────────
        detections: list[dict] = []
        for i, r in enumerate(raw):
            raw_cls = r["raw_cls"]
            cx, cy  = r["cx"], r["cy"]

            if raw_cls == 0:       # outfield player → team by kit colour
                team = kit_team.get(i, 0)  # fallback to Team-A if classifier not ready
                oid  = 0 if team == self._left_label else 1
            elif raw_cls == 1:     # goalkeeper → team by field half
                oid  = 2 if cx < 0.5 * frame_w else 3
            else:                  # ball (2→4), ref (3→5), side-ref (4→6), staff (5→7)
                oid = raw_cls + self._RAW_SHIFT

            label = LABELS[oid] if oid < len(LABELS) else str(oid)
            detections.append({
                "id":         oid,
                "object_id":  oid,
                "label":      label,
                "confidence": r["confidence"],
                "x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"],
                "center_x":   cx,   "center_y":  cy,
                "pixel_x":    cx,   "pixel_y":   cy,
                "width":      r["w"], "height":  r["h"],
            })
        return detections


# ── Standalone video-processing script ───────────────────────────────────────
# Run this file directly to produce a tracked output video without the UI.

def process_video(video_path: str, model_path: str | None = None,
                  output_dir: str = "./output") -> bool:
    """
    Detect and track objects in *video_path*, saving an annotated output video.

    This function is for CLI/debug use only; the UI pipeline uses
    VideoProcessor (processor.py) instead.
    """
    try:
        if model_path is None:
            model_path = str(Path(__file__).resolve().parents[1] / "models" / "best_object.pt")

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: could not open {video_path}")
            return False

        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name  = os.path.basename(video_path)
        out_path    = os.path.join(output_dir,
                                   f"{os.path.splitext(video_name)[0]}_tracked_out.mp4")

        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (width, height))
        if not out.isOpened():
            print(f"Error: could not create {out_path}")
            return False

        model         = YOLO(model_path)
        tracker       = BYTETracker(track_thresh=0.5, match_thresh=0.8,
                                     track_buffer=30, frame_rate=int(fps))
        traj_history  = {}
        kits_clf      = None
        left_label    = 0
        grass_hsv     = None
        frame_id      = 0

        print(f"Processing {total} frames from {video_path}")
        pbar = tqdm(total=total, desc="Tracking")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            pbar.update(1)

            try:
                result = model(frame, conf=0.5, verbose=False)[0]
                players_imgs, players_boxes = get_players_boxes(result)
                if not players_imgs:
                    out.write(frame)
                    continue

                if frame_id == 1:
                    grass_hsv = get_grass_hsv(frame)
                    kits_colors_init = get_kits_colors(players_imgs, grass_hsv)
                    kits_clf  = get_kits_classifier(kits_colors_init)
                    left_label = get_left_team_label(players_boxes, kits_colors_init, kits_clf)

                # Build per-frame detection list for the tracker
                boxes_r = result.boxes
                cls_arr = boxes_r.cls.cpu().numpy()
                xyxy_arr = boxes_r.xyxy.cpu().numpy()
                conf_arr = boxes_r.conf.cpu().numpy()

                frame_width = frame.shape[1]
                detections = []
                for i in range(len(cls_arr)):
                    lbl = int(cls_arr[i])
                    x1, y1, x2, y2 = map(int, xyxy_arr[i])
                    w = x2 - x1;  h = y2 - y1
                    cx = (x1 + x2) / 2.0;  cy = (y1 + y2) / 2.0

                    if lbl == 0:
                        # Distinguish team by kit colour
                        crop = frame[y1:y2, x1:x2]
                        kit  = get_kits_colors([crop], grass_hsv)
                        team = classify_kits(kits_clf, kit)[0]
                        lbl  = 0 if team == left_label else 1
                    elif lbl == 1:
                        # GK: assign left (2) or right (3) by field half
                        lbl = 2 if cx < 0.5 * frame_width else 3
                    else:
                        lbl += 2  # shift referee / staff IDs

                    detections.append({
                        "object_id":  lbl,
                        "pixel_x":    cx,
                        "pixel_y":    cy,
                        "width":      w,
                        "height":     h,
                        "confidence": float(conf_arr[i]),
                    })

                tracked = tracker.update(detections)
                annotated = draw_tracks(frame.copy(), tracked, traj_history)

                # HUD overlay
                cv2.putText(annotated, f"Tracks: {len(tracked)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"Frame: {frame_id}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                out.write(annotated)

            except Exception as exc:
                print(f"\nError on frame {frame_id}: {exc}")
                out.write(frame)
                continue

        pbar.close()
        print(f"\nDone — {frame_id} frames.  Output: {out_path}")
        return True

    except Exception as exc:
        print(f"\nFatal error in process_video: {exc}")
        return False

    finally:
        cap.release() if "cap" in dir() else None
        out.release() if "out" in dir() else None
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    _video = sys.argv[1] if len(sys.argv) > 1 else "/home/fawwaz/Downloads/footballVideos/match_video_022.mp4"
    _model = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / "best_object.pt")
    if not process_video(_video, _model):
        sys.exit(1)
