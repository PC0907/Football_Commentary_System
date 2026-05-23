# First, let's create a ByteTrack implementation file called bytetrack.py

import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class BYTETracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30):
        self.tracked_tracks = []
        self.lost_tracks = []
        self.track_id_count = 0
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
    
    def update(self, detections):
        """
        Main tracking function. Takes in detection results and returns tracked objects.
        
        Args:
            detections: list of dictionaries with keys:
                - object_id: class id
                - pixel_x, pixel_y: center coordinates
                - width, height: size of bounding box
                - confidence: detection confidence
                
        Returns:
            List of tracked objects with same format plus a track_id
        """
        # Convert detections to STrack format for internal processing
        stracks = []
        for det in detections:
            stracks.append(STrack(
                [det['pixel_x'], det['pixel_y'], det['width'], det['height']], 
                det['confidence'], 
                det['object_id']
            ))
        
        # Split current tracked_tracks into activated and unconfirmed by confidence
        activated_stracks = []
        refined_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Step 1: Get predicted locations from existing tracks
        for track in self.tracked_tracks:
            if not track.is_activated:
                # Skip unconfirmed tracks
                continue
            track.predict()
        
        # Step 2: First association with high score detection boxes
        high_score_detections = [d for d in stracks if d.score >= self.track_thresh]
        low_score_detections = [d for d in stracks if d.score < self.track_thresh]
        
        # Association for high-confidence detections
        track_pool = self.tracked_tracks
        dists = self._iou_distance(track_pool, high_score_detections)
        matches, u_track, u_detection = self._get_assignments(dists, track_pool, high_score_detections, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = track_pool[itracked]
            det = high_score_detections[idet]
            
            # Update matched tracks with new detections
            track.update(det)
            activated_stracks.append(track)
            
        # Step 3: Second association with remaining and low score detections
        if len(low_score_detections) > 0:
            # Get unmatched tracks from first association
            r_tracked_stracks = [track_pool[i] for i in u_track]
            
            # Associate with low score detections
            dists = self._iou_distance(r_tracked_stracks, low_score_detections)
            matches, u_track, u_detection = self._get_assignments(dists, r_tracked_stracks, low_score_detections, thresh=0.5)
            
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = low_score_detections[idet]
                
                # Update matched tracks with new detections
                track.update(det)
                activated_stracks.append(track)
                
            u_track_second = [r_tracked_stracks[i] for i in u_track]
        else:
            u_track_second = [track_pool[i] for i in u_track]
        
        # Mark unmatched tracks as lost
        for track in u_track_second:
            if not track.state == TrackState.LOST:
                track.mark_lost()
                lost_stracks.append(track)
        
        # Step 4: Init new tracks
        for i in u_detection:
            det = high_score_detections[i]
            if det.score >= self.track_thresh:
                # Create new track with high confidence detections
                new_track = STrack([det.tlwh[0], det.tlwh[1], det.tlwh[2], det.tlwh[3]], det.score, det.cls)
                new_track.activate(self.track_id_count + 1)
                self.track_id_count += 1
                activated_stracks.append(new_track)
        
        # Step 5: Update state
        # Update lost tracks
        for track in self.lost_tracks:
            if self._frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        
        # Remove the lost tracks that have been lost for too long
        self.lost_tracks = [t for t in self.lost_tracks if t not in removed_stracks]
        
        # Update tracked_tracks
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]
        self.tracked_tracks = self.tracked_tracks + activated_stracks
        
        # Convert STrack objects back to detection format with track_id
        tracked_objects = []
        for track in self.tracked_tracks:
            if track.is_activated:
                tlwh = track.tlwh
                tracked_objects.append({
                    'object_id': track.cls,
                    'track_id': track.track_id,
                    'pixel_x': tlwh[0] + tlwh[2]/2,  # center x
                    'pixel_y': tlwh[1] + tlwh[3]/2,  # center y
                    'width': tlwh[2],
                    'height': tlwh[3],
                    'confidence': track.score
                })
        
        return tracked_objects
    
    def _iou_distance(self, tracks, detections):
        """
        Computes IOU distance matrix between tracks and detections
        """
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = 1 - self._calculate_iou(track.tlwh, det.tlwh)
        
        return iou_matrix
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes
        box format: [x, y, width, height]
        """
        # Convert to [x1,y1,x2,y2] format
        b1_x1, b1_y1 = box1[0], box1[1]
        b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        b2_x1, b2_y1 = box2[0], box2[1]
        b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Get the intersection rectangle
        inter_rect_x1 = max(b1_x1, b2_x1)
        inter_rect_y1 = max(b1_y1, b2_y1)
        inter_rect_x2 = min(b1_x2, b2_x2)
        inter_rect_y2 = min(b1_y2, b2_y2)
        
        # Calculate intersection area
        width = max(0, inter_rect_x2 - inter_rect_x1)
        height = max(0, inter_rect_y2 - inter_rect_y1)
        inter_area = width * height
        
        # Calculate union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        # Calculate IoU
        if union_area == 0:
            return 0
        return inter_area / union_area
    
    def _get_assignments(self, dists, tracks, detections, thresh):
        """
        Using Hungarian algorithm to get assignments between tracks and detections
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Cap distance threshold
        dists[dists > thresh] = 1.0
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(dists)
        
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if dists[row, col] <= thresh:
                matches.append((row, col))
                if row in unmatched_tracks:
                    unmatched_tracks.remove(row)
                if col in unmatched_detections:
                    unmatched_detections.remove(col)
        
        return matches, unmatched_tracks, unmatched_detections


class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    """
    Single Track object for ByteTrack
    """
    def __init__(self, tlwh, score, cls):
        # Raw data: [x,y,w,h], confidence score, and class
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.cls = cls
        
        # State and history
        self.state = TrackState.NEW
        self.is_activated = False
        self.track_id = -1
        
        # Position and velocity for Kalman filter
        self.mean = np.zeros((8,))
        self.covariance = np.zeros((8, 8))
        
        # Simple position prediction (can be replaced with Kalman)
        self.mean[:4] = self._tlwh_to_xyah(self.tlwh)
        
        # Track history
        self.frame_id = 0
        self.start_frame = 0
        self.end_frame = 0
        self.tracklet_len = 0
        
    def predict(self):
        """
        Simple prediction step (can be enhanced with proper Kalman filter)
        """
        pass  # For simplicity, we skip motion prediction
        
    def activate(self, track_id):
        """
        Activate a track
        """
        self.track_id = track_id
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = 1
        self.start_frame = self.frame_id
        
    def update(self, new_track):
        """
        Update a track
        """
        self.frame_id += 1
        self.tracklet_len += 1
        
        # Update position
        self.tlwh = new_track.tlwh
        self.mean[:4] = self._tlwh_to_xyah(self.tlwh)
        self.score = new_track.score
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.end_frame = self.frame_id
        
    def mark_lost(self):
        """
        Mark this track as lost
        """
        self.state = TrackState.LOST
        
    def mark_removed(self):
        """
        Mark this track as removed
        """
        self.state = TrackState.REMOVED
        
    def _tlwh_to_xyah(self, tlwh):
        """
        Convert bounding box to format [center_x, center_y, aspect_ratio, height]
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret