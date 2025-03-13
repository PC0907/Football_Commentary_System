# main.py
import argparse
from pathlib import Path
from configs.paths import PathConfig
from pipeline.football_pipeline import FootballAnalysisPipeline

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Jersey Number Recognition Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="results", help="Output directory path")
    args = parser.parse_args()

    # Initialize configuration
    config = PathConfig()
    
    # Configure paths
    config.DETECTION_WEIGHTS = "models/detection/last.pt"
    config.CLASSIFIER_WEIGHTS = "models/legibility/legibility_resnet34_soccer_20240215.pth"
    config.POSE_CONFIG = "models/vitpose/configs/rtmpose-l_8xb256-420e_coco-256x192.py"
    config.POSE_WEIGHTS = "models/vitpose/vitpose-h.pth"
    
    # Configure output directories
    config.CROP_DIR = str(Path(args.output) / "crops")
    config.LEGIBLE_CROPS_DIR = str(Path(args.output) / "legible_crops")
    config.TORSO_CROPS_DIR = str(Path(args.output) / "torso_crops")
    
    try:
        # Initialize and run pipeline
        pipeline = FootballAnalysisPipeline(config)
        pipeline.initialize()
        results = pipeline.process_video(args.input)
        
        # Print results summary
        print("\nProcessing complete!")
        print(f"Input video: {args.input}")
        print(f"Total legible crops: {len(results['legible_crops']}")
        print(f"Torso crops saved to: {results['torso_dir']}")
        print(f"All results saved to: {args.output}")
        
    except Exception as e:
        print(f"\nError processing video: {str(e)}")
        raise

if __name__ == "__main__":
    main()
