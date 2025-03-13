# setup_models.py
import os
import gdown
import requests
from pathlib import Path

# Base directories
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# 1. Download Object Detection Model
# --------------------------
DETECTION_DIR = MODEL_DIR / "detection"
DETECTION_DIR.mkdir(exist_ok=True)

yolo_url = "https://your-yolo-weights-url/last.pt"  # Replace with your actual URL
yolo_path = DETECTION_DIR / "last.pt"

if not yolo_path.exists():
    print("Downloading YOLO weights...")
    gdown.download(yolo_url, str(yolo_path), fuzzy=True)
    
# --------------------------
# 2. Download Legibility Classifier
# --------------------------
LEGIBILITY_DIR = MODEL_DIR / "legibility"
LEGIBILITY_DIR.mkdir(exist_ok=True)

legibility_url = "https://drive.google.com/uc?id=18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw"
legibility_path = LEGIBILITY_DIR / "legibility_resnet34_soccer_20240215.pth"

if not legibility_path.exists():
    print("\nDownloading legibility classifier...")
    gdown.download(legibility_url, str(legibility_path), fuzzy=True)

# --------------------------
# 3. Download VitPose Models
# --------------------------
VITPOSE_DIR = MODEL_DIR / "vitpose"
VITPOSE_DIR.mkdir(parents=True, exist_ok=True)

# Model weights
vitpose_url = "https://drive.google.com/uc?id=1A3ftF118IcxMn_QONndR-8dPWpf7XzdV"
vitpose_path = VITPOSE_DIR / "vitpose-h.pth"

if not vitpose_path.exists():
    print("\nDownloading VitPose weights...")
    gdown.download(vitpose_url, str(vitpose_path), fuzzy=True)

# Config file (from MMPose)
config_url = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py"
config_path = VITPOSE_DIR / "configs" / "rtmpose-l_8xb256-420e_coco-256x192.py"

if not config_path.exists():
    print("\nDownloading VitPose config...")
    response = requests.get(config_url)
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        f.write(response.text)

print("\nAll models downloaded successfully!")
print(f"Models saved to: {MODEL_DIR}")
