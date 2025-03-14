import os
import cv2
import numpy as np

class CropProcessor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.total_crops = 0
        os.makedirs(output_dir, exist_ok=True)
        
    def process_frame(self, frame, detections, frame_number):
        """
        Process a single frame to extract and save player crops
        
        Args:
            frame: Current video frame (numpy array)
            detections: Detection results from ObjectDetector
            frame_number: Current frame number for naming
        """
        # Get relevant detection data
        boxes = detections['boxes']
        players = detections['players'][1]  # players_boxes
        
        # Process each detected player
        for i, box in enumerate(players):
            self._save_crop(box, frame, frame_number, i)
            
    def _save_crop(self, box, frame, frame_number, player_idx):
        """Extract and save a single player crop"""
        # Get coordinates with padding
        x1, y1, x2, y2 = self._get_padded_coords(box, frame.shape)
        
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        
        # Generate filename
        filename = f"frame_{frame_number}_player_{player_idx}_{x1}_{y1}.jpg"
        output_path = os.path.join(self.output_dir, filename)
        
        # Save crop
        cv2.imwrite(output_path, crop)
        self.total_crops += 1
        
    def _get_padded_coords(self, box, frame_shape):
        """Calculate padded coordinates with bounds checking"""
        # Original coordinates (convert tensor to numpy if needed)
        if hasattr(box.xyxy[0], 'cpu'):
            coords = box.xyxy[0].cpu().numpy()
        else:
            coords = box.xyxy[0].numpy()
            
        x1, y1, x2, y2 = map(int, coords)
        
        # Add padding (5 pixels)
        padding = 5
        new_x1 = max(0, x1 - padding)
        new_y1 = max(0, y1 - padding)
        new_x2 = min(frame_shape[1], x2 + padding)
        new_y2 = min(frame_shape[0], y2 + padding)
        
        return new_x1, new_y1, new_x2, new_y2
