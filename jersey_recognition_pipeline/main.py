# main.py
import argparse
from pathlib import Path
from configs.paths import PathConfig
from src.pipeline.football_pipeline import FootballAnalysisPipeline
from utils.file_utils import create_directory

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Jersey Number Recognition Pipeline")
    parser.add_argument("--input", type=str, required=True, 
                       help="Path to input video file")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory path")
    args = parser.parse_args()

    # Initialize configuration
    config = PathConfig()
    config.OUTPUT_DIR = Path(args.output)
    
    try:
        # Create output directories
        create_directory(config.CROP_DIR)
        create_directory(config.LEGIBLE_CROPS_DIR)
        create_directory(config.TORSO_DIR)
        
        # Initialize pipeline
        pipeline = FootballAnalysisPipeline(config)
        pipeline.initialize()
        
        # Process video
        results = pipeline.process_video(args.input)
        
        # Print results
        print("\nProcessing complete!")
        print(f"Input video: {args.input}")
        print(f"Total legible crops: {len(results['legible_crops'])}")
        print(f"Torso crops saved to: {config.TORSO_DIR}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
