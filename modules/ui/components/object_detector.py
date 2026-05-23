"""
YOLO-based object detector for the Football Commentary UI pipeline.

Kit-colour helpers and rendering utilities live in utils/detection_utils.py —
this module imports from there to avoid duplication with 2Dview.py.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from .bytetrack import BYTETracker
from ..utils.detection_utils import (
    LABELS, BOX_COLORS, TRACK_COLORS,
    get_grass_hsv, get_players_boxes,
    get_kits_colors, get_kits_classifier, classify_kits, get_left_team_label,
    draw_tracks,
)


class ObjectDetector:
    """Simple YOLO-based object detector for the UI pipeline."""

    def __init__(self, model_path=None, conf_threshold: float = 0.5, device=None):
        self.model_path = (
            Path(model_path).resolve()
            if model_path
            else Path(__file__).resolve().parent / "best_object.pt"
        )
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Object detection model not found: {self.model_path}"
            )
        model = YOLO(str(self.model_path))
        if self.device:
            model.to(self.device)
        return model

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLO on *frame* and return a list of detection dicts.

        Each dict has keys: id, object_id, label, confidence,
        x1, y1, x2, y2, center_x, center_y, width, height.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        boxes  = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        cls  = boxes.cls.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        detections = []
        for i in range(len(cls)):
            label = int(cls[i])
            x1, y1, x2, y2 = map(int, xyxy[i])
            w  = x2 - x1
            h  = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append({
                "id":         label,
                "object_id":  label,
                "label":      LABELS[label] if label < len(LABELS) else str(label),
                "confidence": float(conf[i]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "center_x": cx,  "center_y": cy,
                "width":    w,   "height":   h,
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
            model_path = str(Path(__file__).resolve().parent / "best_object.pt")

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
