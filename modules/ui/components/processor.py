"""
VideoProcessor — background QThread driving the full analysis pipeline.

Correct data-flow order (was previously wrong — homography ran before tracking):

  Frame
    │
    ▼ ObjectDetector.detect()            → raw detections (pixel coords)
    │
    ▼ Tracker.update(detections)         → tracked_objects (stable track_ids)
    │
    ▼ HomographyProcessor.process()      → field_positions (world metres)
       [uses tracked_objects so minimap dots have stable IDs]
    │
    ▼ EventDetector.detect(buffer)       → events (canonical schema)
       [operates on world-coordinate data]
    │
    ├──▶ stats accumulation (_update_stats)
    │
    ▼ CommentaryGenerator.submit(event)  → async; results polled via get_nowait()
    │
    ▼ Renderer.render()                  → annotated frame written to output video

Signals
-------
progress_updated      (int, str)   — (percent, status message)  every 30 frames
processing_complete   (str, bool)  — (output_path, success)     on finish/error
stats_updated         (dict)       — match stats snapshot        every 30 frames
frame_ready           (object)     — BGR numpy frame             every frame
minimap_updated       (list)       — world positions list        every 3 frames
homography_confidence (float)      — 0-1 quality score           every 3 frames
commentary_generated  (str)        — "MM:SS — text"              on new line
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from PyQt6.QtCore import QThread, pyqtSignal

from .event_detector       import BaseEventDetector, EventDetectorFactory
from .commentary_generator import BaseCommentaryGenerator, CommentaryGeneratorFactory
from .homography_processor import HomographyProcessor
from .object_detector      import ObjectDetector
from .renderer             import Renderer
from .tracker              import Tracker

log = logging.getLogger(__name__)


# ── Object-id constants (matches detection_utils.LABELS) ──────────────────────
_OID_PLAYER_L = 0
_OID_PLAYER_R = 1
_OID_GK_L     = 2
_OID_GK_R     = 3
_OID_BALL     = 4

# Ball is "possessed" if nearest player is within this radius (metres)
_POSSESSION_RADIUS_M = 2.5


# ═══════════════════════════════════════════════════════════════════════════════

class VideoProcessor(QThread):
    """QThread that processes a football video through the full analysis pipeline."""

    # ── Signals ───────────────────────────────────────────────────────────────
    progress_updated      = pyqtSignal(int, str)
    processing_complete   = pyqtSignal(str, bool)
    stats_updated         = pyqtSignal(dict)
    frame_ready           = pyqtSignal(object)      # BGR numpy frame
    minimap_updated       = pyqtSignal(list)         # world-position dicts
    homography_confidence = pyqtSignal(float)
    commentary_generated  = pyqtSignal(str)          # "MM:SS — text"

    def __init__(
        self,
        input_path: str,
        team_data: dict,
        object_model_path: Optional[str] = None,
        homography_processor: Optional[HomographyProcessor] = None,
        tracker: Optional[Tracker] = None,
        event_detector: Optional[BaseEventDetector] = None,
        commentary_generator: Optional[BaseCommentaryGenerator] = None,
        event_detector_config: Optional[Dict[str, Any]] = None,
        commentary_generator_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.team_data  = team_data
        self.canceled   = False

        # ── Sub-components ────────────────────────────────────────────────────
        self.object_detector      = ObjectDetector(object_model_path)
        self.homography_processor = homography_processor or HomographyProcessor()
        self.tracker              = tracker or Tracker(frame_rate=30)
        self.renderer             = Renderer()

        # Event detector — default rule_based, overrideable via config dict
        if event_detector is not None:
            self.event_detector = event_detector
        else:
            cfg = event_detector_config or {"type": "rule_based"}
            self.event_detector = EventDetectorFactory.create(cfg)

        # Commentary generator — default Phi (falls back to template automatically)
        if commentary_generator is not None:
            self.commentary_generator = commentary_generator
        else:
            cfg = commentary_generator_config or {"type": "phi"}
            self.commentary_generator = CommentaryGeneratorFactory.create(cfg)

        # ── Match state ───────────────────────────────────────────────────────
        self.match_stats: Dict[str, Any] = {
            "possession": {"team_a": 0, "team_b": 0},   # raw frame counts
            "shots":      {"team_a": 0, "team_b": 0},
            "score":      {"team_a": 0, "team_b": 0},
            "passes":     {"team_a": 0, "team_b": 0},
            "fouls":      {"team_a": 0, "team_b": 0},
        }

        # Rolling frame buffer fed to the event detector
        self._event_buffer: List[Dict] = []
        self._event_window: int = 90      # 3 seconds at 30 fps

        # ── Output ────────────────────────────────────────────────────────────
        self.output_dir  = Path.cwd() / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = str(
            self.output_dir / f"{Path(self.input_path).stem}_processed.mp4"
        )

    # ── QThread entry ─────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self._process_video()
            self.processing_complete.emit(self.output_path, True)
        except Exception as exc:
            log.error("VideoProcessor error: %s", exc, exc_info=True)
            self.progress_updated.emit(0, f"Error: {exc}")
            self.processing_complete.emit("", False)
        finally:
            self.commentary_generator.shutdown()

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _process_video(self) -> None:
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.input_path}")

        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Update event detector with the real fps
        self.event_detector.set_fps(fps)

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

                # ─────────────────────────────────────────────────────────────
                # 1. Object detection
                # ─────────────────────────────────────────────────────────────
                detections = self.object_detector.detect(frame)

                # ─────────────────────────────────────────────────────────────
                # 2. Tracking  [MUST precede homography — stable IDs needed]
                # ─────────────────────────────────────────────────────────────
                tracked_objects = self.tracker.update(detections)

                # ─────────────────────────────────────────────────────────────
                # 3. Homography  [uses tracked_objects → stable minimap dots]
                # ─────────────────────────────────────────────────────────────
                _, field_positions = self.homography_processor.process(
                    frame, tracked_objects
                )

                # ─────────────────────────────────────────────────────────────
                # 4. Event detection  (rolling buffer)
                # ─────────────────────────────────────────────────────────────
                snapshot: Dict[str, Any] = {
                    "timestamp":       timestamp,
                    "tracking":        tracked_objects,
                    "field_positions": field_positions,
                    "fps":             fps,
                }
                self._event_buffer.append(snapshot)
                if len(self._event_buffer) > self._event_window:
                    self._event_buffer.pop(0)

                events = self.event_detector.detect(list(self._event_buffer))

                # ─────────────────────────────────────────────────────────────
                # 5. Commentary  (submit async; poll results)
                # ─────────────────────────────────────────────────────────────
                for event in events:
                    self.commentary_generator.submit(event)

                ts_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
                for _ in range(10):   # drain up to 10 pending lines per frame
                    line = self.commentary_generator.get_nowait()
                    if line is None:
                        break
                    self.commentary_generated.emit(f"{ts_str}  —  {line}")

                # ─────────────────────────────────────────────────────────────
                # 6. Stats accumulation
                # ─────────────────────────────────────────────────────────────
                self._update_stats(events, field_positions)

                # ─────────────────────────────────────────────────────────────
                # 7. Build renderer metadata + render
                # ─────────────────────────────────────────────────────────────
                tracking_data = self._build_tracking_data(tracked_objects)
                metadata = {
                    "frame_number":    frame_number,
                    "timestamp":       timestamp,
                    "total_frames":    frame_count,
                    "fps":             fps,
                    "width":           width,
                    "height":          height,
                    "teams":           self.team_data,
                    "match_stats":     self.match_stats,
                    "tracking":        tracking_data,
                    "jersey_numbers":  {},
                    "field_positions": field_positions,
                    "events":          events,
                    "score":           self.match_stats["score"],
                }
                rendered_frame = self.renderer.render(frame, metadata)
                out.write(rendered_frame)

                # ─────────────────────────────────────────────────────────────
                # 8. Emit UI signals
                # ─────────────────────────────────────────────────────────────
                self.frame_ready.emit(rendered_frame)

                if frame_number % 3 == 0:
                    if field_positions:
                        self.minimap_updated.emit(field_positions)
                    self.homography_confidence.emit(
                        self.homography_processor.last_confidence
                    )

                if frame_number % 30 == 0 or frame_number == frame_count - 1:
                    pct = int((frame_number / frame_count) * 100) if frame_count else 0
                    self.progress_updated.emit(
                        pct, f"Frame {frame_number + 1} / {frame_count}"
                    )
                    self._emit_stats()

                frame_number += 1

        finally:
            cap.release()
            out.release()

    # ── Stats accumulation ────────────────────────────────────────────────────

    def _update_stats(
        self,
        events: List[Dict[str, Any]],
        field_positions: List[Dict[str, Any]],
    ) -> None:
        """Update match_stats from detected events and possession estimate."""

        # ── Possession: nearest team to ball owns this frame ──────────────────
        ball = next(
            (fp for fp in field_positions if fp.get("object_id") == _OID_BALL), None
        )
        if ball:
            bx = ball.get("world_x_meters", 0.0)
            by = ball.get("world_y_meters", 0.0)
            best_dist  = float("inf")
            best_team  = None
            for fp in field_positions:
                oid = fp.get("object_id", -1)
                if oid not in (_OID_PLAYER_L, _OID_GK_L,
                               _OID_PLAYER_R, _OID_GK_R):
                    continue
                d = ((fp.get("world_x_meters", 0) - bx) ** 2
                     + (fp.get("world_y_meters", 0) - by) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_team = (
                        "team_a" if oid in (_OID_PLAYER_L, _OID_GK_L) else "team_b"
                    )
            if best_team and best_dist <= _POSSESSION_RADIUS_M:
                self.match_stats["possession"][best_team] += 1

        # ── Event-based stats ─────────────────────────────────────────────────
        for e in events:
            etype = (e.get("type") or "").lower()
            team  = e.get("team", "")
            if not team:
                continue

            if etype == "shot":
                self.match_stats["shots"][team] = (
                    self.match_stats["shots"].get(team, 0) + 1
                )
            elif etype == "goal":
                self.match_stats["score"][team] = (
                    self.match_stats["score"].get(team, 0) + 1
                )
                self.match_stats["shots"][team] = (
                    self.match_stats["shots"].get(team, 0) + 1
                )
            elif etype in ("pass",):
                self.match_stats["passes"][team] = (
                    self.match_stats["passes"].get(team, 0) + 1
                )
            elif etype in ("foul", "penalty"):
                # The fouling team committed the foul
                self.match_stats["fouls"][team] = (
                    self.match_stats["fouls"].get(team, 0) + 1
                )

    def _emit_stats(self) -> None:
        """Emit match_stats with possession converted to percentages."""
        raw = self.match_stats["possession"]
        total = raw.get("team_a", 0) + raw.get("team_b", 0)
        poss_pct = {
            "team_a": int(100 * raw.get("team_a", 0) / total) if total else 50,
            "team_b": int(100 * raw.get("team_b", 0) / total) if total else 50,
        }
        self.stats_updated.emit(
            {
                "players": {},
                "match": {
                    **self.match_stats,
                    "possession": poss_pct,   # override raw counts with %
                },
            }
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_tracking_data(tracked_objects: List[Dict]) -> Dict:
        data: Dict[str, Any] = {"players": {}, "ball": None}
        for obj in tracked_objects:
            x1 = int(obj["pixel_x"] - obj["width"]  / 2)
            y1 = int(obj["pixel_y"] - obj["height"] / 2)
            x2 = x1 + int(obj["width"])
            y2 = y1 + int(obj["height"])
            data["players"][obj["track_id"]] = [x1, y1, x2, y2]
            if obj["object_id"] == _OID_BALL:
                data["ball"] = (x1, y1, obj["width"], obj["height"])
        return data

    def _save_player_stats(self, stats_path: str) -> str:
        with open(stats_path, "w") as f:
            json.dump({"players": {}, "match": self.match_stats}, f, indent=2)
        return stats_path
