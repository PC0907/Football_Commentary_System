import logging
from .bytetrack import BYTETracker

class Tracker:
    """Wrapper for BYTETracker that exposes a stable API for the UI pipeline."""

    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30):
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate
        )
        self.logger = logging.getLogger(__name__)

    def update(self, detections):
        """Update the tracker with the latest detections."""
        try:
            return self.tracker.update(detections)
        except Exception as e:
            self.logger.error(f"Tracker update failed: {e}")
            return []
