# Object Detection — YOLO Training & Experiments

This directory contains training scripts, notebooks, and evaluation code for the
**YOLOv8 object detection model** that identifies players, goalkeepers, the ball,
referees, and staff in broadcast football footage.

The trained weights live at `models/player_ball_detector_yolov8.pt` (project root).
The production inference class is `pipeline/detector.py::ObjectDetector`.

---

## Purpose

Broadcast football video contains multiple object classes that move, occlude each other,
and change appearance dramatically (kit colours vary by team, lighting varies by stadium).
The goals of this component are:

1. Detect and box all relevant objects in every frame.
2. Classify each detection into 8 canonical classes (see below).
3. Distinguish the two teams via **kit-colour KMeans clustering** (not a separate model).

### Canonical class mapping

| ID | Label | Notes |
|----|-------|-------|
| 0 | Player-L (Team A) | Outfield player, left team |
| 1 | Player-R (Team B) | Outfield player, right team |
| 2 | GK-L | Goalkeeper, left half |
| 3 | GK-R | Goalkeeper, right half |
| 4 | Ball | |
| 5 | Main Ref | |
| 6 | Side Ref | |
| 7 | Staff | |

The raw YOLO model outputs only **6 classes** (outfield player, GK, ball, main ref,
side ref, staff) — team separation is done post-inference via kit colour. See
[How It Works](#how-it-works).

---

## Files

| File | Description |
|------|-------------|
| `Football_Object_Detection.ipynb` | Main training notebook. Downloads SoccerNet dataset, configures YOLOv8 training, evaluates mAP. |
| `PlayerAndBall.ipynb` | Earlier notebook focusing on player + ball detection only (2-class). |
| `main.py` | CLI inference script: runs the trained model on a video and saves annotated output. |
| `mainFawwaz.py` | Personal variant of `main.py` with additional debug overlays and local dataset paths. |
| `track.py` | Adds multi-object tracking (ByteTrack) on top of YOLO detections. |
| `track_kaggle.py` | Kaggle-adapted version of `track.py` for running in a Kaggle notebook environment. |
| `requirements.txt` | Python dependencies for this experiment (ultralytics, opencv-python, etc.). |
| `weights/best.pt` | Best weights checkpoint from training. |
| `weights/last.pt` | Last-epoch checkpoint. |
| `CustomCode/` | Custom detection scripts: `detect_ball.py`, `detect_ball2.py` — earlier ball-only experiments with YOLOv5. |
| `test_videos/` | Sample video clips used for evaluation. |
| `output/` | Annotated output videos from inference runs. |

---

## How It Works

### 1 — Dataset
Training data is from the **SoccerNet Object Tracking** dataset (broadcast camera, top-down
and lateral views). The dataset provides bounding boxes for players, goalkeepers, ball, and
referees across many European league matches.

### 2 — YOLO training
YOLOv8m (medium) was trained for 25 epochs on the SoccerNet dataset. The model was fine-tuned
from the official YOLOv8 pretrained weights (COCO ImageNet backbone). Key hyperparameters:
- Input resolution: 1280 × 720 (native broadcast resolution)
- Batch size: 8 (GPU memory limit)
- Confidence threshold at inference: **0.5**

### 3 — Kit-colour team separation (post-inference)

Because YOLO cannot know which team a player belongs to (that changes every match), team
assignment is done geometrically using **KMeans on HSV kit colours**:

1. **First frame**: extract all outfield-player bounding-box crops → remove grass background
   (mask HSV values near the field's grass colour) → compute average remaining pixel colour →
   cluster into 2 groups with KMeans(n_clusters=2).
2. **Left-team determination**: find the spatial average x-position of each KMeans cluster.
   The cluster with the smaller mean x sits on the left → labelled Team A.
3. **Every subsequent frame**: run KMeans prediction on each new player's crop colour and
   assign Team A or Team B accordingly.

GKs are separated by position: a GK detected in the left half of the frame → GK-L (class 2),
right half → GK-R (class 3). This works because goalkeepers stay near their own goal.

---

## Problems Faced

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Teams swapped mid-match | KMeans cluster order is arbitrary; left/right team can flip if camera pans | Re-compute `_left_label` on a per-frame basis — or cache it from frame 1 and keep stable |
| Ball missed at high speed | Motion blur + small bounding box below confidence threshold | Lower conf to 0.5; also consider tracking with Kalman filter to bridge gaps |
| Players merge into one box | Occlusion in dense formations | YOLO limitation; tracking helps assign IDs but doesn't fix detection |
| GK misclassified as outfield | Similar kit colour to teammates | GK is detected as a separate YOLO class (class 1) — already handled |
| Night game / floodlight glare | Training data lacks night-game examples | Re-train with augmented (low-light) data |
| Wrong team colour on first frame | Player crop is grass-heavy → wrong average colour | Tuned grass-mask HSV range; added minimum-crop-area guard |

---

## What Still Needs Fixing / Future Work

- [ ] **Team-swap robustness at camera pan**: if the camera pans 180°, the left/right
  assignment computed on frame 1 becomes wrong. Add a "Swap Teams" button in the UI (already
  exists) and expose this as a runtime toggle.
- [ ] **Re-train on larger, more diverse dataset**: the current model was trained on a
  relatively small SoccerNet subset. More stadiums, more leagues, night games, and VAR
  zoom shots would improve generalisation.
- [ ] **Confidence-aware suppression**: low-confidence detections (0.3–0.5) cause flickering.
  Consider a two-threshold system (high = keep, low = only keep if tracked previously).
- [ ] **Ball tracking continuity**: the ball disappears for several frames on fast shots.
  A dedicated ball-tracking head (Kalman + constant-velocity model) could bridge gaps.
- [ ] **Staff / side-ref removal**: staff members and side referees clutter the minimap.
  Either filter them out or assign them a distinct colour in the minimap.
