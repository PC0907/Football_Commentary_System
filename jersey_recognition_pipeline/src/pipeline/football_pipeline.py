import cv2
import os
from .object_detector import ObjectDetector
from .crop_processor import CropProcessor
from .classifier import LegibilityClassifier
from .pose_estimator import PoseEstimator
from utils.file_utils import create_directory

class FootballAnalysisPipeline:
    def __init__(self, config):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Configuration object containing paths and parameters
        """
        self.config = config
        self.object_detector = None
        self.crop_processor = None
        self.classifier = None
        self.pose_estimator = None

    def initialize(self):
        """Initialize all pipeline components"""
        # Initialize object detector
        self.object_detector = ObjectDetector(
            self.config.DETECTION_WEIGHTS
        )
        
        # Initialize crop processor
        self.crop_processor = CropProcessor(
            self.config.CROP_DIR
        )
        
        # Initialize classifier
        self.classifier = LegibilityClassifier(self.config)
        self.classifier.load_model()
        
        # Initialize pose estimator
        self.pose_estimator = PoseEstimator(self.config)
        self.pose_estimator.initialize()

    def process_video(self, video_path):
        """
        Process video through all pipeline stages
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Dictionary containing final results
        """
        # Create output directories
        create_directory(self.config.CROP_DIR)
        create_directory(self.config.LEGIBLE_CROPS_DIR)
        
        # Process video frames
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
                
            # Stage 1: Object detection
            detections = self.object_detector.detect(frame)
            
            # Stage 2: Crop extraction
            self.crop_processor.process_frame(
                frame=frame,
                detections=detections,
                frame_number=frame_count
            )
            
        cap.release()
        
        # Stage 3: Legibility filtering
        legible_crops = self.classifier.filter_crops(
            self.config.CROP_DIR
        )
        
        # Stage 4: Pose estimation
        pose_results = self.pose_estimator.process_crops(
            legible_crops
        )
        
        return {
            'crops_dir': self.config.CROP_DIR,
            'legible_crops': legible_crops,
            'pose_results': pose_results,
            'torso_dir': self.config.TORSO_CROPS_DIR
        }
