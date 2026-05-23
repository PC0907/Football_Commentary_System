import os
import cv2
import numpy as np
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from utils.visualization import draw_keypoints
from utils.file_utils import create_directory

class PoseEstimator:
    def __init__(self, config):
        self.config = config
        self.model = None
        self._initialize_output_dirs()
        
    def _initialize_output_dirs(self):
        """Create output directories for pose estimation results"""
        create_directory(self.config.KEYPOINTS_DIR)
        create_directory(self.config.TORSO_CROPS_DIR)
        create_directory(self.config.PROCESSED_TORSO_DIR)
        
    def initialize(self):
        """Initialize VitPose model"""
        register_all_modules()
        self.model = init_model(
            self.config.POSE_CONFIG,
            self.config.POSE_WEIGHTS,
            device=self.config.DEVICE
        )
        return self
        
    def process_crops(self, crop_paths):
        """Process all legible crops through VitPose"""
        results = []
        for crop_path in crop_paths:
            result = self._process_single_crop(crop_path)
            if result:
                results.append(result)
        return results
        
    def _process_single_crop(self, crop_path):
        """Process a single crop through VitPose pipeline"""
        try:
            # Load image
            img = cv2.imread(crop_path)
            if img is None:
                return None
                
            # Get base filename
            base_name = os.path.basename(crop_path).split('.')[0]
            
            # Run pose estimation
            bbox = [[0, 0, img.shape[1], img.shape[0]]]
            pose_result = inference_topdown(self.model, img, bbox)
            
            # Extract keypoints
            keypoints = pose_result[0].pred_instances.keypoints[0]
            scores = pose_result[0].pred_instances.keypoint_scores[0]
            
            # Visualize keypoints
            keypoint_img = draw_keypoints(img, keypoints, scores)
            keypoint_path = os.path.join(
                self.config.KEYPOINTS_DIR,
                f"{base_name}_keypoints.jpg"
            )
            cv2.imwrite(keypoint_path, keypoint_img)
            
            # Extract torso
            torso_original, torso_processed = self._extract_torso(img, keypoints)
            if torso_original is not None:
                # Save original torso
                torso_path = os.path.join(
                    self.config.TORSO_CROPS_DIR,
                    f"{base_name}_torso.jpg"
                )
                cv2.imwrite(torso_path, torso_original)
                
                # Save processed torso
                processed_path = os.path.join(
                    self.config.PROCESSED_TORSO_DIR,
                    f"{base_name}_torso_processed.jpg"
                )
                cv2.imwrite(processed_path, torso_processed)
                
            return {
                'image_path': crop_path,
                'keypoints': keypoints,
                'scores': scores,
                'torso_path': torso_path,
                'processed_path': processed_path
            }
            
        except Exception as e:
            print(f"Error processing {crop_path}: {str(e)}")
            return None
            
    def _extract_torso(self, img, keypoints):
        """Extract and process torso region"""
        # Get relevant keypoints
        left_shoulder = keypoints[5][:2].astype(int)
        right_shoulder = keypoints[6][:2].astype(int)
        left_hip = keypoints[11][:2].astype(int)
        right_hip = keypoints[12][:2].astype(int)
        
        # Calculate bounds with padding
        padding = 5
        x1 = max(0, int(min(left_shoulder[0], left_hip[0])) - padding)
        y1 = max(0, int(min(left_shoulder[1], right_shoulder[1])) - padding)
        x2 = min(img.shape[1], int(max(right_shoulder[0], right_hip[0])) + padding)
        y2 = min(img.shape[0], int(max(left_hip[1], right_hip[1])) + padding)
        
        # Extract original crop
        torso_original = img[y1:y2, x1:x2]
        if torso_original.size == 0:
            return None, None
            
        # Process crop
        torso_processed = self._process_torso(torso_original)
        
        return torso_original, torso_processed
        
    def _process_torso(self, torso):
        """Enhance torso crop"""
        # Resize
        torso = cv2.resize(torso, (0,0), fx=2, fy=2)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        torso = cv2.filter2D(torso, -1, kernel)
        
        # Enhance contrast
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
