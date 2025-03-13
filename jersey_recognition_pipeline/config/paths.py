class PathConfig:
    # Directory paths
    CROP_DIR = "./outputs/crops"
    LEGIBLE_CROPS_DIR = "./outputs/legible_crops"
    KEYPOINTS_DIR = "./outputs/keypoints"
    TORSO_CROPS_DIR = "./outputs/torso_crops"
    PROCESSED_TORSO_DIR = "./outputs/processed_torso"
    
    # Model weights
    DETECTION_WEIGHTS = "./weights/last.pt"
    CLASSIFIER_WEIGHTS = "./models/legibility_resnet34_soccer_20240215.pth"
    POSE_WEIGHTS = "./ViTPose/checkpoints/vitpose-h.pth"
    POSE_CONFIG = "rtmpose-l_8xb256-420e_coco-256x192.py"
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
