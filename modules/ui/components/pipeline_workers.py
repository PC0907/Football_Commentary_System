"""
Pipeline worker classes for the Football Commentary processing chain.

Correct data-flow order
-----------------------
  Frame
    │
    ▼ DetectionWorker
  detections  (pixel coords, class labels)
    │
    ▼ TrackingWorker
  tracked_objects  (pixel coords + stable track_ids from ByteTrack)
    │
    ▼ HomographyWorker
  field_positions  (world metres per tracked object)
    │
    ├──▶ JerseyWorker   → jersey_numbers  (optional, runs in parallel)
    │
    ▼ EventWorker
  events  (pass / shot / goal / corner …)
    │
    ▼ RenderWorker
  rendered frame

Each worker exposes a ``process(data)`` method that transforms its input tuple
into an output tuple.  Workers communicate via queues in a threaded setup,
but the same ``process()`` can be called synchronously from VideoProcessor.

Previous bug: TrackingWorker was calling
    self.tracker.update(frame, field_positions)
which has the wrong argument types AND the wrong position in the chain
(tracking should precede homography, not follow it).
"""

import logging
import time
from collections import deque
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Base worker
# ══════════════════════════════════════════════════════════════════════════════

class PipelineWorker:
    """Abstract base class for pipeline stage workers."""

    def __init__(self, input_queue, output_queue):
        self.input_queue  = input_queue
        self.output_queue = output_queue
        self.running      = True

    def stop(self):
        self.running = False

    def process(self, data: Any) -> Any:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Object detection
# ══════════════════════════════════════════════════════════════════════════════

class DetectionWorker(PipelineWorker):
    """
    Runs the YOLO object detector on each frame.

    Input  : (frame: np.ndarray, metadata: dict)
    Output : (frame, detections: list[dict], metadata)

    ``detections`` format (list of dicts):
        object_id, pixel_x, pixel_y, width, height, confidence
    """

    def __init__(self, input_queue, output_queue, object_detector):
        super().__init__(input_queue, output_queue)
        self.object_detector = object_detector

    def process(self, data):
        frame, metadata = data
        if frame is None:
            return None, [], metadata

        detections = self.object_detector.detect(frame)

        # Normalise to the tracker-compatible dict format
        # (ObjectDetector returns 'center_x'/'center_y', tracker wants 'pixel_x'/'pixel_y')
        normalised = []
        for d in detections:
            normalised.append({
                "object_id":  d.get("object_id", d.get("id", 0)),
                "pixel_x":    d.get("pixel_x",   d.get("center_x", 0.0)),
                "pixel_y":    d.get("pixel_y",   d.get("center_y", 0.0)),
                "width":      d.get("width",  d.get("x2", 0) - d.get("x1", 0)),
                "height":     d.get("height", d.get("y2", 0) - d.get("y1", 0)),
                "confidence": d.get("confidence", 0.0),
            })

        return frame, normalised, metadata


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Multi-object tracking
# ══════════════════════════════════════════════════════════════════════════════

class TrackingWorker(PipelineWorker):
    """
    Feeds detections into ByteTrack and assigns stable track IDs.

    Input  : (frame, detections: list[dict], metadata)
    Output : (frame, tracked_objects: list[dict], metadata)

    ``tracked_objects`` adds ``track_id`` to each detection dict and contains
    Kalman-smoothed positions.

    NOTE: Tracking operates on *pixel* coordinates only.  World-coordinate
    conversion (homography) happens in the next stage.
    """

    def __init__(self, input_queue, output_queue, tracker):
        super().__init__(input_queue, output_queue)
        self.tracker = tracker
        self.logger  = logging.getLogger(__name__)

    def process(self, data):
        frame, detections, metadata = data
        if frame is None:
            return None, [], metadata

        try:
            tracked_objects = self.tracker.update(detections)
        except Exception as exc:
            self.logger.error(f"TrackingWorker: tracker.update() failed — {exc}")
            tracked_objects = []

        return frame, tracked_objects, metadata


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Homography
# ══════════════════════════════════════════════════════════════════════════════

class HomographyWorker(PipelineWorker):
    """
    Transforms tracked object pixel positions to real-world field coordinates.

    Input  : (frame, tracked_objects: list[dict], metadata)
    Output : (frame, tracked_objects, field_positions: list[dict], metadata)

    ``field_positions`` is a list of dicts with:
        object_id, track_id, world_x_meters, world_y_meters

    tracked_objects are also augmented in-place with 'world_x' / 'world_y'
    keys so downstream workers can access both pixel and field coords from
    a single list.
    """

    def __init__(self, input_queue, output_queue, homography_processor):
        super().__init__(input_queue, output_queue)
        self.homography_processor = homography_processor
        self.logger = logging.getLogger(__name__)

    def process(self, data):
        frame, tracked_objects, metadata = data
        if frame is None:
            return None, [], [], metadata

        try:
            _, field_positions = self.homography_processor.process(frame, tracked_objects)
        except Exception as exc:
            self.logger.error(f"HomographyWorker: process() failed — {exc}")
            field_positions = []

        # Annotate tracked_objects with world coordinates for downstream consumers
        pos_by_idx = {i: p for i, p in enumerate(field_positions)}
        for i, obj in enumerate(tracked_objects):
            if i in pos_by_idx:
                obj["world_x"] = pos_by_idx[i].get("world_x_meters", 0.0)
                obj["world_y"] = pos_by_idx[i].get("world_y_meters", 0.0)

        return frame, tracked_objects, field_positions, metadata


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4a — Jersey number detection (optional, can run in parallel)
# ══════════════════════════════════════════════════════════════════════════════

class JerseyWorker(PipelineWorker):
    """
    Crops each tracked player and runs jersey-number OCR.

    Input  : (frame, tracked_objects: list[dict], metadata)
    Output : (jersey_numbers: dict[track_id → int | None], metadata)

    Only objects with object_id in {0, 1, 2, 3} (players / GKs) are cropped.
    Results are cached per track_id and only refreshed every N frames to save
    compute.
    """

    _PLAYER_CLASS_IDS = {0, 1, 2, 3}
    _REFRESH_EVERY    = 15  # frames between re-detections per track

    def __init__(self, input_queue, output_queue, jersey_detector):
        super().__init__(input_queue, output_queue)
        self.jersey_detector = jersey_detector
        self._cache: dict[int, int | None] = {}   # track_id → jersey number
        self._last_frame: dict[int, int]   = {}   # track_id → frame when last detected

    def process(self, data):
        frame, tracked_objects, metadata = data
        if frame is None:
            return {}, metadata

        frame_number = metadata.get("frame_number", 0)
        h, w = frame.shape[:2]
        jersey_numbers: dict[int, int | None] = {}

        for obj in tracked_objects:
            if obj.get("object_id") not in self._PLAYER_CLASS_IDS:
                continue

            tid = obj["track_id"]
            last_detected = self._last_frame.get(tid, -self._REFRESH_EVERY)

            # Return cached result if recently detected
            if tid in self._cache and (frame_number - last_detected) < self._REFRESH_EVERY:
                jersey_numbers[tid] = self._cache[tid]
                continue

            # Crop player region
            cx, cy     = obj["pixel_x"], obj["pixel_y"]
            bw, bh     = obj["width"],   obj["height"]
            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))
            x2 = min(w, int(cx + bw / 2))
            y2 = min(h, int(cy + bh / 2))
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                jersey_numbers[tid] = self._cache.get(tid)
                continue

            try:
                result = self.jersey_detector.detect(crop)
            except Exception:
                result = None

            self._cache[tid]      = result
            self._last_frame[tid] = frame_number
            jersey_numbers[tid]   = result

        return jersey_numbers, metadata


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4b — Event detection
# ══════════════════════════════════════════════════════════════════════════════

class EventWorker(PipelineWorker):
    """
    Detects football events (pass, shot, goal, corner, free-kick, foul)
    from a rolling buffer of world-coordinate tracking data.

    Input  : (frame, tracked_objects: list[dict], field_positions: list[dict], metadata)
    Output : (events: list[dict], metadata)

    The worker maintains an internal frame buffer (default 30 frames) so the
    event detector can look back in time without the caller managing state.

    NOTE: event_detector.detect() receives a list of frame snapshots.
    Each snapshot has keys: timestamp, tracking, field_positions.
    The real FootballEventDetector (2D_event_detector.py) operates on world
    coordinates, so field_positions must be non-empty for events to fire.
    """

    def __init__(self, input_queue, output_queue, event_detector, buffer_size: int = 30):
        super().__init__(input_queue, output_queue)
        self.event_detector = event_detector
        self.frame_buffer: deque = deque(maxlen=buffer_size)

    def process(self, data):
        frame, tracked_objects, field_positions, metadata = data
        if frame is None:
            return [], metadata

        self.frame_buffer.append({
            "timestamp":      metadata.get("timestamp", 0.0),
            "tracking":       tracked_objects,
            "field_positions": field_positions,
        })

        events = self.event_detector.detect(list(self.frame_buffer))
        return events, metadata


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Rendering
# ══════════════════════════════════════════════════════════════════════════════

class RenderWorker(PipelineWorker):
    """
    Composites all overlays onto the frame (bounding boxes, score, commentary…).

    Input  : (frame, metadata: dict)
    Output : (rendered_frame, metadata)

    ``metadata`` is expected to contain all processing results merged into a
    single dict — the VideoProcessor (processor.py) is responsible for building
    this dict before calling RenderWorker.
    """

    def __init__(self, input_queue, output_queue, renderer):
        super().__init__(input_queue, output_queue)
        self.renderer = renderer

    def process(self, data):
        frame, metadata = data
        if frame is None:
            return None, metadata

        rendered_frame = self.renderer.render(frame, metadata)
        return rendered_frame, metadata
