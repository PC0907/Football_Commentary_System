from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import os
import logging
from pathlib import Path
import json

from .homography_processor import HomographyProcessor
from .object_detector import ObjectDetector
from .tracker import Tracker
from .renderer import Renderer


class DummyJerseyDetector:
    def detect(self, crop):
        return None


class DummyEventDetector:
    def detect(self, frame_buffer):
        return []


class DummyCommentaryGenerator:
    def generate(self, events):
        return []


class VideoProcessor(QThread):
    """Thread for processing football videos without blocking the UI."""

    progress_updated = pyqtSignal(int, str)
    processing_complete = pyqtSignal(str, bool)
    stats_updated = pyqtSignal(dict)

    def __init__(
        self,
        input_path,
        team_data,
        object_model_path=None,
        homography_processor=None,
        tracker=None,
        jersey_detector=None,
        event_detector=None,
        commentary_generator=None,
    ):
        super().__init__()
        self.input_path = input_path
        self.team_data = team_data
        self.output_path = None
        self.canceled = False

        self.object_detector = ObjectDetector(object_model_path)
        self.homography_processor = homography_processor or HomographyProcessor()
        self.tracker = tracker or Tracker(frame_rate=30)
        self.jersey_detector = jersey_detector or DummyJerseyDetector()
        self.event_detector = event_detector or DummyEventDetector()
        self.commentary_generator = commentary_generator or DummyCommentaryGenerator()
        self.renderer = Renderer()

        self.players = {}
        self.match_stats = {
            'possession': {'team_a': 0, 'team_b': 0},
            'shots': {'team_a': 0, 'team_b': 0},
            'score': {'team_a': 0, 'team_b': 0},
            'passes': {'team_a': 0, 'team_b': 0},
            'fouls': {'team_a': 0, 'team_b': 0}
        }

        self.output_dir = Path.cwd() / 'output'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = str(self.output_dir / f"{Path(self.input_path).stem}_processed.mp4")

    def run(self):
        try:
            self._process_video()
            self.processing_complete.emit(self.output_path, True)
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            self.progress_updated.emit(0, f"Error: {str(e)}")
            self.processing_complete.emit('', False)

    def _process_video(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise Exception('Failed to open video file')

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise Exception(f'Failed to open output video writer: {self.output_path}')

        frame_number = 0
        while True:
            if self.canceled:
                break

            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps if fps > 0 else 0.0
            detections = self.object_detector.detect(frame)
            _, field_positions = self.homography_processor.process(frame, detections)
            tracked_objects = self.tracker.update(detections)

            # TODO: Pass real field_positions + tracked_objects once HomographyProcessor
            # is wired to the real homography.py implementation.
            # For now the event detector and commentary generator receive stubs.
            frame_snapshot = {
                'timestamp': timestamp,
                'tracking': tracked_objects,
                'field_positions': field_positions,
            }
            events = self.event_detector.detect([frame_snapshot])
            commentary = self.commentary_generator.generate(events)

            tracking_data = {'players': {}, 'ball': None}
            for obj in tracked_objects:
                x1 = int(obj['pixel_x'] - obj['width'] / 2)
                y1 = int(obj['pixel_y'] - obj['height'] / 2)
                x2 = x1 + int(obj['width'])
                y2 = y1 + int(obj['height'])
                tracking_data['players'][obj['track_id']] = [x1, y1, x2, y2]
                if obj['object_id'] == 4:
                    tracking_data['ball'] = (x1, y1, obj['width'], obj['height'])

            metadata = {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'total_frames': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'teams': self.team_data,
                'players': self.players,
                'match_stats': self.match_stats,
                'tracking': tracking_data,
                'jersey_numbers': {},
                'field_positions': field_positions,
                'events': events,
                'commentary': commentary,
                'score': self.match_stats['score']
            }

            rendered_frame = self.renderer.render(frame, metadata)
            out.write(rendered_frame)

            if frame_number % 30 == 0 or frame_number == frame_count - 1:
                progress = int((frame_number / frame_count) * 100) if frame_count > 0 else 0
                self.progress_updated.emit(progress, f"Processing frame {frame_number + 1}/{frame_count}")
                self._emit_stats_update()

            frame_number += 1

        cap.release()
        out.release()

    def _save_player_stats(self, stats_path):
        stats_dict = {
            'players': {},
            'match': self.match_stats
        }
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        return stats_path

    def _emit_stats_update(self):
        stats_snapshot = {
            'players': {},
            'match': self.match_stats
        }
        self.stats_updated.emit(stats_snapshot)
