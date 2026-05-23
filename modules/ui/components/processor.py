"""
VideoProcessor — background QThread that drives the full analysis pipeline:

  frame → object detection → tracking → homography → event detection
        → commentary generation → rendering → write output + emit signals

New signals (beyond the original three):
  frame_ready          object  — rendered BGR numpy frame (for live preview)
  minimap_updated      list    — world-coordinate positions for the minimap
  commentary_generated str     — a single formatted commentary line
  homography_confidence float  — 0-1 quality score from HomographyProcessor
"""

from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import os
import logging
import json
from pathlib import Path

from .homography_processor import HomographyProcessor
from .object_detector import ObjectDetector
from .tracker import Tracker
from .renderer import Renderer


log = logging.getLogger(__name__)


# ── Stub components (used when no real implementation is injected) ─────────────

class DummyJerseyDetector:
    def detect(self, crop):
        return None


class DummyEventDetector:
    def detect(self, frame_buffer):
        return []


class DummyCommentaryGenerator:
    def generate(self, events):
        return []


# ── Main processor thread ─────────────────────────────────────────────────────

class VideoProcessor(QThread):
    """QThread that processes a football video through the full analysis pipeline."""

    # ── Signals ───────────────────────────────────────────────────────────────
    progress_updated      = pyqtSignal(int, str)   # (percent, status_message)
    processing_complete   = pyqtSignal(str, bool)  # (output_path, success)
    stats_updated         = pyqtSignal(dict)        # snapshot of match_stats
    frame_ready           = pyqtSignal(object)      # rendered BGR numpy frame
    minimap_updated       = pyqtSignal(list)        # [{object_id, world_x_meters, world_y_meters}]
    commentary_generated  = pyqtSignal(str)         # "MM:SS — commentary text"
    homography_confidence = pyqtSignal(float)       # 0.0 – 1.0

    def __init__(
        self,
        input_path: str,
        team_data: dict,
        object_model_path: str | None = None,
        homography_processor: HomographyProcessor | None = None,
        tracker: Tracker | None = None,
        jersey_detector=None,
        event_detector=None,
        commentary_generator=None,
    ):
        super().__init__()
        self.input_path = input_path
        self.team_data  = team_data
        self.canceled   = False

        # Sub-components
        self.object_detector       = ObjectDetector(object_model_path)
        self.homography_processor  = homography_processor or HomographyProcessor()
        self.tracker               = tracker or Tracker(frame_rate=30)
        self.jersey_detector       = jersey_detector or DummyJerseyDetector()
        self.event_detector        = event_detector or DummyEventDetector()
        self.commentary_generator  = commentary_generator or DummyCommentaryGenerator()
        self.renderer              = Renderer()

        # State
        self.players     = {}
        self.match_stats = {
            "possession": {"team_a": 0, "team_b": 0},
            "shots":      {"team_a": 0, "team_b": 0},
            "score":      {"team_a": 0, "team_b": 0},
            "passes":     {"team_a": 0, "team_b": 0},
            "fouls":      {"team_a": 0, "team_b": 0},
        }

        # Output
        self.output_dir  = Path.cwd() / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = str(
            self.output_dir / f"{Path(self.input_path).stem}_processed.mp4"
        )

    # ── QThread entry point ───────────────────────────────────────────────────

    def run(self):
        try:
            self._process_video()
            self.processing_complete.emit(self.output_path, True)
        except Exception as exc:
            log.error("Processing error: %s", exc, exc_info=True)
            self.progress_updated.emit(0, f"Error: {exc}")
            self.processing_complete.emit("", False)

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _process_video(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.input_path}")

        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError(f"Cannot open output writer: {self.output_path}")

        frame_number = 0

        try:
            while True:
                if self.canceled:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_number / fps if fps > 0 else 0.0

                # ── Detection ─────────────────────────────────────────────────
                detections = self.object_detector.detect(frame)

                # ── Homography ────────────────────────────────────────────────
                _, field_positions = self.homography_processor.process(frame, detections)

                # ── Tracking ──────────────────────────────────────────────────
                tracked_objects = self.tracker.update(detections)

                # ── Event detection & commentary ──────────────────────────────
                frame_snapshot = {
                    "timestamp":      timestamp,
                    "tracking":       tracked_objects,
                    "field_positions": field_positions,
                }
                events    = self.event_detector.detect([frame_snapshot])
                commentary = self.commentary_generator.generate(events)

                # ── Build tracking_data for the renderer ──────────────────────
                tracking_data = {"players": {}, "ball": None}
                for obj in tracked_objects:
                    x1 = int(obj["pixel_x"] - obj["width"]  / 2)
                    y1 = int(obj["pixel_y"] - obj["height"] / 2)
                    x2 = x1 + int(obj["width"])
                    y2 = y1 + int(obj["height"])
                    tracking_data["players"][obj["track_id"]] = [x1, y1, x2, y2]
                    if obj["object_id"] == 4:   # Ball
                        tracking_data["ball"] = (x1, y1, obj["width"], obj["height"])

                metadata = {
                    "frame_number":   frame_number,
                    "timestamp":      timestamp,
                    "total_frames":   frame_count,
                    "fps":            fps,
                    "width":          width,
                    "height":         height,
                    "teams":          self.team_data,
                    "players":        self.players,
                    "match_stats":    self.match_stats,
                    "tracking":       tracking_data,
                    "jersey_numbers": {},
                    "field_positions": field_positions,
                    "events":         events,
                    "commentary":     commentary,
                    "score":          self.match_stats["score"],
                }

                rendered_frame = self.renderer.render(frame, metadata)
                out.write(rendered_frame)

                # ── Emit signals ───────────────────────────────────────────────
                # Live frame: every frame (inference is the bottleneck, not signals)
                self.frame_ready.emit(rendered_frame)

                # Minimap: every 3 frames to keep UI responsive
                if field_positions and frame_number % 3 == 0:
                    self.minimap_updated.emit(field_positions)
                    self.homography_confidence.emit(
                        self.homography_processor.last_confidence
                    )

                # Commentary lines
                if isinstance(commentary, list):
                    for line in commentary:
                        if line:
                            ts = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
                            self.commentary_generated.emit(f"{ts}  —  {line}")
                elif isinstance(commentary, str) and commentary:
                    ts = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
                    self.commentary_generated.emit(f"{ts}  —  {commentary}")

                # Progress + stats: every 30 frames
                if frame_number % 30 == 0 or frame_number == frame_count - 1:
                    pct = int((frame_number / frame_count) * 100) if frame_count > 0 else 0
                    self.progress_updated.emit(
                        pct, f"Frame {frame_number + 1} / {frame_count}"
                    )
                    self._emit_stats()

                frame_number += 1

        finally:
            cap.release()
            out.release()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _emit_stats(self):
        self.stats_updated.emit(
            {"players": {}, "match": self.match_stats}
        )

    def _save_player_stats(self, stats_path: str) -> str:
        with open(stats_path, "w") as f:
            json.dump({"players": {}, "match": self.match_stats}, f, indent=2)
        return stats_path
