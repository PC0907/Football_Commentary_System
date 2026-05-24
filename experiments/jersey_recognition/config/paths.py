from pathlib import Path

class PathConfig:
    # Base directories
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Model paths
    DETECTION_WEIGHTS = MODELS_DIR / "detection/last.pt"
    CLASSIFIER_WEIGHTS = MODELS_DIR / "legibility/legibility_resnet34_soccer_20240215.pth"
    POSE_WEIGHTS = MODELS_DIR / "vitpose/vitpose-h.pth"
    POSE_CONFIG = MODELS_DIR / "vitpose/configs/rtmpose-l_8xb256-420e_coco-256x192.py"

    # Output directories (created automatically)
    OUTPUT_DIR = PROJECT_ROOT / "results"
    CROP_DIR = OUTPUT_DIR / "extracted_crops"
    LEGIBLE_CROPS_DIR = OUTPUT_DIR / "filtered_crops"
    TORSO_DIR = OUTPUT_DIR / "torso_regions"
