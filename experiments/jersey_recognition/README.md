# Jersey Recognition — Number OCR Pipeline

This directory contains a **modular pipeline for recognising jersey numbers** from
football video. It extracts player crops, filters for legibility, isolates the torso
region using pose keypoints, and runs an OCR classifier.

The goal is to automatically map `track_id → jersey_number → player_name` using the
team sheet loaded in the UI, enabling the commentary generator to refer to players by name.

---

## Purpose

Broadcast video often shows players from behind or at an angle where the number is not
readable. The pipeline therefore has multiple stages to ensure that only **high-quality,
legible torso crops** are passed to the OCR model:

```
Video frame
    │ object detection
    ▼
Player bounding box crops
    │ legibility classifier
    ▼  (reject blurry / back-facing / small)
Legible crops
    │ VitPose torso extraction
    ▼  (crop to shoulder–waist region)
Torso patches
    │ OCR classifier / digit recogniser
    ▼
Jersey number (int)
```

---

## Files and Directories

| Path | Description |
|------|-------------|
| `main.py` | Entry point. Accepts `--input` (video or crop folder) and `--output` dir. Runs full pipeline end-to-end. |
| `setup.py` | Installs the package in development mode (`pip install -e .`). |
| `verify_installation.py` | Smoke-test: imports every stage and prints a status report. |
| `requirements.txt` | All Python dependencies (ultralytics, vitpose, torchvision, pytesseract, etc.). |
| `src/pipeline/` | Pipeline stage modules (see below). |
| `src/video_processor.py` | Orchestrates the pipeline; reads frames from a video file. |
| `config/paths.py` | Centralised path constants (model paths, input/output dirs). |
| `utils/visualization.py` | Draws keypoints and bounding boxes for debug inspection. |
| `utils/file_utils.py` | Folder scanning, safe file writes, output directory management. |
| `Notebooks/` | Jupyter notebooks for stage-by-stage experimentation. |
| `sample_outputs/` | Example torso crops and recognised numbers from test runs. |
| `models/` | Pre-trained model weights (detection, legibility classifier, VitPose). |

### Pipeline stage modules (`src/pipeline/`)

| Module | Description |
|--------|-------------|
| `object_detector.py` | Runs the YOLO player detector; returns bounding boxes per frame. |
| `crop_processor.py` | Crops each bounding box from the frame; validates minimum size. |
| `classifier.py` | Binary legibility classifier: predicts whether a crop shows a readable number. |
| `torso_extraction.py` | Uses VitPose (pose estimation) to locate shoulder and hip keypoints → crops the torso region. |
| `football_pipeline.py` | Assembles the full multi-stage pipeline; manages frame buffers. |

---

## How the Pipeline Was Built

### Stage 1 — Crop extraction
Player crops from the object detector are far too large for OCR — they include legs, arms,
and background. A simple fixed-ratio top-third crop was tried first but failed on crouching
and sliding players.

### Stage 2 — Legibility classifier
A small CNN / MobileNet binary classifier trained on `experiments/jersey_annotator/` output
predicts whether the crop is frontal + legible. This filter removes ~60% of crops (players
seen from behind, too small, motion-blurred) before they reach the expensive VitPose model.

### Stage 3 — VitPose torso extraction
VitPose (a vision-transformer pose estimator) detects 17 body keypoints. The torso region
is defined as the bounding box between the left/right shoulder keypoints (top) and the
left/right hip keypoints (bottom). This crop contains the number reliably regardless of
player pose.

### Stage 4 — OCR
The torso patch is resized to a fixed size (128 × 64), contrast-enhanced (CLAHE), and passed
to a digit-classification model. Two approaches were tried:

- **Tesseract OCR**: fast, zero training required, but very inaccurate on jersey fonts
  (compressed, stylised, partially occluded).
- **Custom CNN digit classifier**: trained on jersey-annotated crops; significantly better
  accuracy on the in-domain data.

The custom CNN is currently the recommended path.

---

## Problems Faced

| Problem | Root Cause | Fix / Status |
|---------|-----------|--------------|
| Tesseract fails on jersey fonts | Font mismatch; jersey numbers use stylised glyphs | Replaced with custom CNN trained on labelled crops |
| VitPose too slow for real-time | Transformer inference ~80 ms/frame on CPU | Cache keypoints; run only on I-frames or at 5 fps |
| Legibility classifier overfits | Small training set; class imbalance | Added augmentation ×10 from `jersey_annotator`; balanced classes |
| Two-digit confusion (1 vs 11, 7 vs 17) | Crop edges cut off leading digit | Expand crop by 10% on each side before passing to OCR |
| VitPose shoulder KPs off-screen | Player near frame edge → partial body visible | Fall back to fixed-ratio top-third crop if pose fails |
| No ground truth for real-time eval | Hard to verify live | Evaluate offline on annotated test video with known jersey numbers |

---

## What Still Needs Fixing / Future Work

- [ ] **Integrate into the main pipeline**: `app/processor_thread.py` currently does not call
  any jersey recognition code. The integration point is after ByteTrack assigns `track_id`s —
  run jersey recognition at ~1 fps (every 30 frames) and update a `{track_id: jersey_number}`
  cache.
- [ ] **Map jersey number → player name**: once a jersey number is recognised, look it up in
  the team sheet (`data/teamsheets/efl.csv`) loaded in the UI and cache `track_id → name`.
- [ ] **Handle number not found**: if the jersey number isn't in the team sheet, fall back to
  `"Player #<number>"` rather than crashing.
- [ ] **Re-train legibility classifier on more data**: the current classifier was trained on
  a small batch; gather more diverse footage.
- [ ] **GPU acceleration**: the current pipeline runs on CPU. For real-time use, VitPose
  and the OCR model must run on GPU; profile and optimise the inference path.
- [ ] **Handle multi-digit grouping**: jersey numbers 10–99 must be read as a two-digit
  number, not two separate digits. Post-process OCR output to group adjacent digit detections.
