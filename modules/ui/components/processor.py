from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import tempfile
import logging
import time
import queue
import threading
from pathlib import Path
import json
import concurrent.futures

from .homography_processor import HomographyProcessor
from .object_detector import ObjectDetector
from .pipeline_workers import (
    DetectionWorker, HomographyWorker, TrackingWorker,
    JerseyWorker, EventWorker, RenderWorker
)
from .renderer import Renderer

class VideoProcessor(QThread):
    """Thread for processing football videos without blocking the UI"""
    progress_updated = pyqtSignal(int, str)
    processing_complete = pyqtSignal(str, bool)
    stats_updated = pyqtSignal(dict)
    
    def __init__(self, input_path, team_data, object_model_path="./best_object.pt", homography_processor=None, 
                 tracker=None, jersey_detector=None, event_detector=None, commentary_generator=None):
        super().__init__()
        self.input_path = input_path
        self.team_data = team_data
        self.output_path = None
        self.canceled = False
        
        # Initialize processing modules
        self.object_detector = ObjectDetector(object_model_path) if object_model_path else DummyObjectDetector()
        self.homography_processor = homography_processor or HomographyProcessor()
        self.tracker = tracker or DummyTracker()
        self.jersey_detector = jersey_detector or DummyJerseyDetector()
        self.event_detector = event_detector or DummyEventDetector()
        self.commentary_generator = commentary_generator or DummyCommentaryGenerator()
        self.renderer = Renderer()
        
        # Initialize queues
        self.detection_queue = queue.Queue(maxsize=10)
        self.homography_queue = queue.Queue(maxsize=10)
        self.tracking_queue = queue.Queue(maxsize=10)
        self.jersey_queue = queue.Queue(maxsize=20)
        self.event_queue = queue.Queue(maxsize=10)
        self.render_queue = queue.Queue(maxsize=10)
        
        # Initialize workers
        self.workers = [
            DetectionWorker(self.detection_queue, self.homography_queue, self.object_detector),
            HomographyWorker(self.homography_queue, self.tracking_queue, self.homography_processor),
            TrackingWorker(self.tracking_queue, self.jersey_queue, self.tracker),
            JerseyWorker(self.jersey_queue, self.event_queue, self.jersey_detector),
            EventWorker(self.event_queue, self.render_queue, self.event_detector),
            RenderWorker(self.render_queue, None, self.renderer)
        ]
        
        # Initialize state
        self.players = {}
        self.match_stats = {
            'possession': {'team_a': 0, 'team_b': 0},
            'shots': {'team_a': 0, 'team_b': 0},
            'score': {'team_a': 0, 'team_b': 0},
            'passes': {'team_a': 0, 'team_b': 0},
            'fouls': {'team_a': 0, 'team_b': 0}
        }
        
    def run(self):
        """Main processing function that runs in a separate thread"""
        try:
            # Create a temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                self.output_path = f"{temp_dir}/processed_video.mp4"
                stats_path = f"{temp_dir}/player_stats.json"
                
                # Start worker threads
                worker_threads = []
                for worker in self.workers:
                    thread = threading.Thread(target=self._worker_loop, args=(worker,))
                    thread.daemon = True
                    thread.start()
                    worker_threads.append(thread)
                
                # Process video
                self._process_video()
                
                # Stop workers
                for worker in self.workers:
                    worker.stop()
                
                # Wait for worker threads to complete
                for thread in worker_threads:
                    thread.join()
                
                # Save final statistics
                self._save_player_stats(stats_path)
                
                if not self.canceled:
                    self.processing_complete.emit(self.output_path, True)
                else:
                    self.processing_complete.emit("", False)
                    
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            self.progress_updated.emit(0, f"Error: {str(e)}")
            self.processing_complete.emit("", False)
    
    def _worker_loop(self, worker):
        """Worker thread loop"""
        while worker.running:
            try:
                # Get data from input queue
                data = worker.input_queue.get(timeout=1.0)
                if data is None:
                    break
                
                # Process data
                result = worker.process(data)
                
                # Put result in output queue if not None
                if result is not None and worker.output_queue is not None:
                    worker.output_queue.put(result)
                
                worker.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in worker {worker.__class__.__name__}: {str(e)}")
                continue
    
    def _process_video(self):
        """Process the video using the pipeline architecture"""
        try:
            # Load the video
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                raise Exception("Failed to open video file")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create video writer for output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            # Process each frame
            for frame_number in range(frame_count):
                if self.canceled:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Create frame metadata
                timestamp = frame_number / fps
                metadata = {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'total_frames': frame_count,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'teams': self.team_data,
                    'players': self.players,
                    'match_stats': self.match_stats
                }
                
                # Put frame in detection queue
                self.detection_queue.put((frame.copy(), metadata))
                
                # Update progress
                if frame_number % 30 == 0:
                    progress = int((frame_number / frame_count) * 100)
                    self.progress_updated.emit(progress, f"Processing frame {frame_number + 1}/{frame_count}")
                    
                    # Emit stats update
                    self._emit_stats_update()
                
                # Try to get rendered frame
                try:
                    rendered_frame, _ = self.render_queue.get_nowait()
                    out.write(rendered_frame)
                except queue.Empty:
                    continue
            
            # Release resources
            cap.release()
            out.release()
            
        except Exception as e:
            logging.error(f"Error processing video: {str(e)}")
            raise
    
    def _save_player_stats(self, stats_path):
        """Save player statistics to JSON file"""
        stats_dict = {
            'players': {pid: stats.to_dict() for pid, stats in self.players.items()},
            'match': self.match_stats
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
            
        return stats_path
    
    def _emit_stats_update(self):
        """Emit current player and match statistics to update the UI"""
        stats_snapshot = {
            'players': {pid: player.to_dict() for pid, player in self.players.items()},
            'match': self.match_stats
        }
        self.stats_updated.emit(stats_snapshot)