import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Any

class HomographyProcessor:
    """Processes video frames to transform object positions from image coordinates to field coordinates"""
    
    def __init__(self):
        self.model = None  # Will be initialized on first use
        
    def process(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a frame and its detections to transform object positions to field coordinates.
        
        Args:
            frame: Input video frame
            detections: List of detected objects with their positions
            
        Returns:
            Tuple containing:
                - field_frame: Frame with field coordinates visualization
                - field_positions: List of transformed object positions in field coordinates
        """
        try:
            # Convert detections to the format expected by process_frame
            object_positions = []
            for det in detections:
                object_positions.append({
                    'object_id': det['id'],
                    'pixel_x': det['center_x'],
                    'pixel_y': det['center_y']
                })
            
            # Process frame through homography module
            transformed_positions, reprojection_error, confidence, _ = process_frame(
                frame, object_positions, 0  # Frame ID 0 for single frame processing
            )
            
            # Create visualization frame
            field_frame = frame.copy()
            
            # Draw transformed positions
            for pos in transformed_positions:
                x, y = int(pos['world_x_meters']), int(pos['world_y_meters'])
                cv2.circle(field_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    field_frame,
                    f"ID:{pos['object_id']}",
                    (x - 30, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # Add metrics
            cv2.putText(
                field_frame,
                f"Error: {reprojection_error:.2f}m",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                field_frame,
                f"Confidence: {confidence:.2%}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            return field_frame, transformed_positions
            
        except Exception as e:
            logging.error(f"Error in homography processing: {str(e)}")
            return frame, [] 