import queue
import threading
import logging
import time
from typing import Dict, Any
from collections import deque
import concurrent.futures

class PipelineWorker:
    """Base class for pipeline workers"""
    def __init__(self, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        
    def stop(self):
        """Stop the worker"""
        self.running = False
        
    def process(self, data):
        """Process input data - to be implemented by subclasses"""
        raise NotImplementedError

class DetectionWorker(PipelineWorker):
    """Worker for object detection"""
    def __init__(self, input_queue, output_queue, object_detector):
        super().__init__(input_queue, output_queue)
        self.object_detector = object_detector
        
    def process(self, data):
        """Process frame through object detection"""
        frame, metadata = data
        if frame is None:
            return None, None
            
        detections = self.object_detector.detect(frame)
        return frame, detections, metadata

class HomographyWorker(PipelineWorker):
    """Worker for homography processing"""
    def __init__(self, input_queue, output_queue, homography_processor):
        super().__init__(input_queue, output_queue)
        self.homography_processor = homography_processor
        
    def process(self, data):
        """Process frame through homography"""
        frame, detections, metadata = data
        if frame is None:
            return None, None, None
            
        field_frame, field_positions = self.homography_processor.process(frame, detections)
        return frame, field_frame, field_positions, metadata

class TrackingWorker(PipelineWorker):
    """Worker for player tracking"""
    def __init__(self, input_queue, output_queue, tracker):
        super().__init__(input_queue, output_queue)
        self.tracker = tracker
        self.players = {}
        
    def process(self, data):
        """Process frame through tracking"""
        frame, field_frame, field_positions, metadata = data
        if frame is None:
            return None, None, None, None, None
            
        tracking_results = self.tracker.update(frame, field_positions)
        return frame, field_frame, tracking_results, field_positions, metadata

class JerseyWorker(PipelineWorker):
    """Worker for jersey number detection"""
    def __init__(self, input_queue, output_queue, jersey_detector):
        super().__init__(input_queue, output_queue)
        self.jersey_detector = jersey_detector
        self.jersey_requests = {}
        
    def process(self, data):
        """Process player crops through jersey detection"""
        player_crops, metadata = data
        if player_crops is None:
            return None, None
            
        results = {}
        for player_id, crop in player_crops.items():
            results[player_id] = self.jersey_detector.detect(crop)
            
        return results, metadata

class EventWorker(PipelineWorker):
    """Worker for event detection"""
    def __init__(self, input_queue, output_queue, event_detector):
        super().__init__(input_queue, output_queue)
        self.event_detector = event_detector
        self.frame_buffer = deque(maxlen=30)
        
    def process(self, data):
        """Process frame through event detection"""
        frame, field_frame, tracking_results, field_positions, metadata = data
        if frame is None:
            return None, None
            
        # Add frame data to buffer
        self.frame_buffer.append({
            'timestamp': metadata['timestamp'],
            'tracking': tracking_results,
            'field_positions': field_positions
        })
        
        # Detect events
        events = self.event_detector.detect(self.frame_buffer)
        return events, metadata

class RenderWorker(PipelineWorker):
    """Worker for final frame rendering"""
    def __init__(self, input_queue, output_queue, renderer):
        super().__init__(input_queue, output_queue)
        self.renderer = renderer
        
    def process(self, data):
        """Process frame through rendering"""
        frame, metadata = data
        if frame is None:
            return None, None
            
        rendered_frame = self.renderer.render(frame, metadata)
        return rendered_frame, metadata 