import os
import argparse
from pathlib import Path
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argus
from scipy.ndimage import maximum_filter, gaussian_filter1d
import sys
sys.path.insert(0, '/content/ball-action-spotting') #For colab environments
# Import your model architecture
from src.predictors import MultiDimStackerPredictor
from src.utils import post_processing
from src.frames import get_frames_processor

# Constants
VIDEO_FPS = 25.0  # From your constants.py
RESOLUTION = "720p"
INDEX_SAVE_ZONE = 1
TTA = False

# Action classes
CLASSES = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside", 
    "Shots on target", "Shots off target", "Clearance", 
    "Ball out of play", "Throw-in", "Foul", 
    "Indirect free-kick", "Direct free-kick", "Corner", "Card"
]
CARD_CLASSES = ["Yellow card", "Red card", "Yellow->red card"]
CLASS2TARGET = {cls: trg for trg, cls in enumerate(CLASSES)}

# Post-processing parameters
POSTPROCESS_PARAMS = {
    "gauss_sigma": 3.0,
    "height": 0.2,
    "distance": 15,
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to the pretrained model")
    parser.add_argument("--video_path", required=True, type=str, help="Path to the video file")
    parser.add_argument("--output_path", required=True, type=str, help="Path to save the prediction JSON")
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID to use")
    parser.add_argument("--half", default=1, type=int, help="Half of the game (1 or 2)")
    parser.add_argument("--game_name", default=None, type=str, help="Game name for the output JSON")
    return parser.parse_args()

def get_video_info(video_path):
    """Get video information using OpenCV"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height
    }

class OpenCVFrameFetcher:
    """Frame fetcher using OpenCV instead of NvDec"""
    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_index = -1
        
    def fetch_frame(self):
        """Fetch next frame from video"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.current_index += 1
        
        # Convert BGR to grayscale (single channel)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert to tensor
        frame = torch.from_numpy(frame).to(torch.float32)
        
        # Add batch and time dimensions: [H, W] -> [1, 1, H, W]
        frame = frame.unsqueeze(0).unsqueeze(0)
        
        return frame
    
    def close(self):
        """Close video capture"""
        if self.cap.isOpened():
            self.cap.release()
            
def post_processing(frame_indexes, frame_predictions, gauss_sigma=3.0, height=0.25, distance=15):
    """Post-process raw predictions to get action detections"""
    frame_predictions = gaussian_filter1d(frame_predictions, sigma=gauss_sigma)
    peaks = []
    confidences = []
    
    for i in range(1, len(frame_predictions) - 1):
        if (frame_predictions[i] > frame_predictions[i-1] and 
            frame_predictions[i] > frame_predictions[i+1] and 
            frame_predictions[i] > height):
            
            # Check if it's the highest peak in the vicinity
            start_idx = max(0, i - distance)
            end_idx = min(len(frame_predictions), i + distance + 1)
            if frame_predictions[i] == np.max(frame_predictions[start_idx:end_idx]):
                peaks.append(frame_indexes[i])
                confidences.append(float(frame_predictions[i]))
    
    return peaks, confidences

def get_raw_predictions(predictor, video_path, frame_count):
    """Process video frames to get raw predictions"""
    frame_fetcher = OpenCVFrameFetcher(video_path)
    
    indexes_generator = predictor.indexes_generator
    min_frame_index = indexes_generator.clip_index(0, frame_count, INDEX_SAVE_ZONE)
    max_frame_index = indexes_generator.clip_index(frame_count, frame_count, INDEX_SAVE_ZONE)
    
    frame_index2prediction = dict()
    predictor.reset_buffers()
    
    with tqdm(total=frame_count) as t:
        while True:
            frame = frame_fetcher.fetch_frame()
            if frame is None:
                break
                
            frame_index = frame_fetcher.current_index
            
            # Now frame should have shape [1, 1, C, H, W]
            # We need to extract just the [C, H, W] part for the predictor
            frame = frame[0, 0]  # This should give us [C, H, W]
            
            prediction, predict_index = predictor.predict(frame, frame_index)
            
            if predict_index < min_frame_index:
                t.update()
                continue
                
            if prediction is not None:
                frame_index2prediction[predict_index] = prediction.cpu().numpy()
                
            t.update()
            
            if predict_index == max_frame_index:
                break
    
    frame_fetcher.close()
    predictor.reset_buffers()
    
    frame_indexes = sorted(frame_index2prediction.keys())
    if not frame_indexes:
        raise ValueError("No predictions were generated. Check frame processing and model compatibility.")
        
    raw_predictions = np.stack([frame_index2prediction[i] for i in frame_indexes], axis=0)
    
    return frame_indexes, raw_predictions
    
def raw_predictions_to_actions(frame_indexes, raw_predictions):
    """Convert raw predictions to action detections"""
    class2actions = dict()
    for cls, cls_index in CLASS2TARGET.items():
        class2actions[cls] = post_processing(
            frame_indexes, 
            raw_predictions[:, cls_index], 
            **POSTPROCESS_PARAMS
        )
        print(f"Predicted {len(class2actions[cls][0])} {cls} actions")
    
    return class2actions

def prepare_spotting_results(class_actions, half, game_name, output_path):
    """Prepare spotting results in JSON format"""
    results_spotting = {
        "UrlLocal": game_name,
        "predictions": list(),
    }

    for cls, (frame_indexes, confidences) in class_actions.items():
        cls = "Yellow card" if cls == "Card" else cls
        for frame_index, confidence in zip(frame_indexes, confidences):
            position = round(frame_index / VIDEO_FPS * 1000)
            seconds = int(frame_index / VIDEO_FPS)
            prediction = {
                "gameTime": f"{half} - {seconds // 60:02}:{seconds % 60:02}",
                "label": cls,
                "position": str(position),
                "half": str(half),
                "confidence": str(confidence),
            }
            results_spotting["predictions"].append(prediction)
    
    results_spotting["predictions"] = sorted(
        results_spotting["predictions"],
        key=lambda pred: int(pred["position"])
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as outfile:
        json.dump(results_spotting, outfile, indent=4)
    
    print("Spotting results saved to", output_path)
    return results_spotting

def main():
    args = parse_arguments()
    
    # Load model
    print(f"Loading model from {args.model_path}")
    predictor = MultiDimStackerPredictor(args.model_path, device=f"cuda:{args.gpu_id}", tta=TTA)
    
    # Get video info
    video_path = Path(args.video_path)
    video_info = get_video_info(video_path)
    print("Video info:", video_info)
    
    # Set game name if not provided
    game_name = args.game_name if args.game_name else video_path.stem
    
    # Get predictions
    print(f"Processing video: {video_path}")
    frame_indexes, raw_predictions = get_raw_predictions(
        predictor, video_path, video_info["frame_count"]
    )
    
    # Convert raw predictions to actions
    class_actions = raw_predictions_to_actions(frame_indexes, raw_predictions)
    
    # Prepare results
    results = prepare_spotting_results(
        class_actions, args.half, game_name, args.output_path
    )
    
    print(f"Completed processing. Detected {len(results['predictions'])} actions.")

if __name__ == "__main__":
    main()
