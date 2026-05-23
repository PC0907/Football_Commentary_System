# ⚠️  SUPERSEDED — canonical version is at:
#     modules/ui/components/tracker.py  (wraps modules/ui/components/bytetrack.py)
# This copy is kept only for historical reference of the AFCS prototype.

import cv2
import numpy as np
from typing import List, Dict, Tuple

# Assuming bytetrack.py is in the same directory.  If not, adjust the import.
try:
    from bytetrack import BYTETracker  # Import the BYTETracker class
except ImportError:
    raise ImportError("ByteTrack is required. Please install ByteTrack and ensure it's in your PYTHONPATH.")
import logging

class Tracker:
    """
    Class for performing multi-object tracking using ByteTrack.
    """

    def __init__(self, track_thresh=0.5, match_thresh=0.8, frame_rate=30): # Changed default track_thresh to 0.5 to match bytetrack.py
        """
        Initializes the Tracker.

        Args:
            track_thresh (float, optional):  Detection confidence threshold for tracking.  Defaults to 0.25.
            match_thresh (float, optional):  Matching threshold for associating detections with tracks. Defaults to 0.8.
            frame_rate (int, optional): The frame rate of the video being processed. Defaults to 30.
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.tracker = BYTETracker(
            track_thresh=self.track_thresh,
            match_thresh=self.match_thresh,
            frame_rate=self.frame_rate
        )
        self.tracked_objects: Dict[int, dict] = {}  # Store tracked objects by track ID.
        self.lost_frames: Dict[int, int] = {} #Keeps track of how many frames a track has been lost.
        self.frame_id = 0 # Added frame counter

    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Updates the tracker with new detections and returns the tracked objects.

        Args:
            detections (List[Dict]): A list of detections from the object detector.  Each detection is a dictionary
                with keys: 'x1', 'y1', 'x2', 'y2', 'class_id', 'confidence', 'label'.
            frame (numpy.ndarray): The current video frame.  Not used by this BYTETracker version, but kept for compatibility.

        Returns:
            List[Dict]: A list of tracked objects.  Each tracked object is a dictionary with the following keys:
                - 'track_id' (int): The unique track ID.
                - 'x1' (int): x-coordinate of the top-left corner of the bounding box.
                - 'y1' (int): y-coordinate of the top-left corner of the bounding box.
                - 'x2' (int): x-coordinate of the bottom-right corner of the bounding box.
                - 'y2' (int): y-coordinate of the bottom-right corner of the bounding box.
                - 'class_id' (int): The class ID of the object.
                - 'confidence' (float):  The detection confidence.
                - 'label' (str): The label (name) of the object.
                - 'trajectory' (List[Tuple[int, int]]):  The past positions of the object.
                - 'age' (int): The number of frames the track has been active.
                - 'avg_speed' (Tuple[float, float]): The average speed of the object in (x, y) directions.
                - 'active' (bool):  Indicates if the object is currently active.
        """
        self.frame_id += 1 # Increment frame counter.
        # Convert detections to the format expected by BYTETracker.
        byte_detections = []
        for det in detections:
            byte_detections.append({  # Construct dict in the format BYTETracker expects
                'object_id': det['class_id'],
                'pixel_x': (det['x1'] + det['x2']) / 2,
                'pixel_y': (det['y1'] + det['y2']) / 2,
                'width': det['x2'] - det['x1'],
                'height': det['y2'] - det['y1'],
                'confidence': det['confidence'],
            })

        # Update the tracker with the new detections.
        tracked_objects = self.tracker.update(byte_detections)

        # Process the output from BYTETracker into a list of dictionaries.
        output_tracks = []
        for track in tracked_objects:
            track_id = track['track_id'] # Changed from track.track_id to track['track_id']
            x1 = int(track['pixel_x'] - track['width'] / 2) #added these lines
            y1 = int(track['pixel_y'] - track['height'] / 2)
            x2 = int(track['pixel_x'] + track['width'] / 2)
            y2 = int(track['pixel_y'] + track['height'] / 2)
            class_id = int(track['object_id']) # Changed from track.cls to track['object_id']
            confidence = track['confidence']
            label = detections[0]['label'] if detections else "Unknown"

            if track_id not in self.tracked_objects:
                self.tracked_objects[track_id] = {
                    'track_id': track_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'class_id': class_id,
                    'label': label,
                    'confidence': confidence,
                    'trajectory': [(int(track['pixel_x']), int(track['pixel_y']))],  # Store center point
                    'age': 1,
                    'avg_speed': (0, 0),
                    'active': True
                }
            else:
                # update tracked object
                self.tracked_objects[track_id]['x1'] = x1
                self.tracked_objects[track_id]['y1'] = y1
                self.tracked_objects[track_id]['x2'] = x2
                self.tracked_objects[track_id]['y2'] = y2
                self.tracked_objects[track_id]['class_id'] = class_id
                self.tracked_objects[track_id]['label'] = label
                self.tracked_objects[track_id]['confidence'] = confidence
                self.tracked_objects[track_id]['trajectory'].append((int(track['pixel_x']), int(track['pixel_y'])))
                self.tracked_objects[track_id]['age'] += 1
                self.tracked_objects[track_id]['active'] = True
                if len(self.tracked_objects[track_id]['trajectory']) > 2:
                    x1_prev, y1_prev = self.tracked_objects[track_id]['trajectory'][-2]
                    x2_prev, y2_prev = self.tracked_objects[track_id]['trajectory'][-1]
                    dt = 1 / self.frame_rate
                    vx = (x2_prev - x1_prev) / dt
                    vy = (y2_prev - y1_prev) / dt
                    self.tracked_objects[track_id]['avg_speed'] = (vx, vy)
            output_tracks.append(self.tracked_objects[track_id])
            self.lost_frames[track_id] = 0
        
        # Handle lost tracks.
        for track_id in list(self.lost_frames.keys()): # Iterate over a copy of the keys.
            self.lost_frames[track_id] += 1
            if track_id in self.tracked_objects:
                continue #if it is still being tracked don't do the rest
            if self.lost_frames[track_id] > 15:
                self.lost_frames.pop(track_id)
                if track_id in self.tracked_objects:
                    del self.tracked_objects[track_id]
            elif track_id in self.tracked_objects:
                self.tracked_objects[track_id]['age'] += 1
                self.tracked_objects[track_id]['active'] = False
                output_tracks.append(self.tracked_objects[track_id])
        return output_tracks

    def get_tracked_objects(self) -> Dict[int, dict]:
        """
        Returns all currently tracked objects.

        Returns:
            Dict[int, dict]: A dictionary mapping track IDs to tracked object information.
        """
        return self.tracked_objects
