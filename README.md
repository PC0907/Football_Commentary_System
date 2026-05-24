# Automatic Football Commentary System

A computer-vision pipeline that ingests broadcast football footage and produces annotated video
with real-time object tracking, bird's-eye minimap, event detection, and auto-generated text
commentary — all surfaced through a themeable PyQt6 desktop UI.

---

## Table of Contents
- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Model Weights](#model-weights)
- [Module Status](#module-status)
- [Known Issues & TODO](#known-issues--todo)
- [Experiments](#experiments)

---

## What It Does

| Feature | Status |
|---------|--------|
| Upload a match video via GUI | ✅ Working |
| YOLO-based object detection (players, ball, GK, refs, staff) | ✅ Working |
| Kit-colour KMeans team assignment | ✅ Working |
| Multi-object tracking (ByteTrack) | ✅ Working |
| Bird's-eye minimap (homography → world coords) | ✅ Working |
| EMA-smoothed H matrix (eliminates minimap jitter) | ✅ Working |
| Rule-based event detection (pass/shot/goal/corner/foul) | ✅ Wired — thresholds need tuning |
| Template-based text commentary | ✅ Working |
| Live commentary panel in UI | ✅ Working |
| Player stats panel (player count, ball visibility) | ✅ Working |
| Team swap toggle (flip team colours in minimap) | ✅ Working |
| Themeable UI (Dark / Light / Blue / Green) | ✅ Working |
| Team sheet editor (manual entry + CSV import) | ✅ Working |
| TTS audio commentary | ⚠️ Standalone only — not wired to real-time pipeline |
| SRT subtitle export | ⚠️ Standalone only — not wired to real-time pipeline |
| Jersey number recognition | ⚠️ Standalone pipeline exists — not integrated into UI |
| LLM-based commentary (Phi / Gemma) | ⚠️ Stub in place — model not loaded |
| Per-player distance / heatmap stats | ⚠️ Prototype exists — not integrated |

---

## Architecture

```
Broadcast video
       │
       ▼
┌─────────────────────────────┐
│  ObjectDetector             │  pipeline/detector.py
│  YOLOv8 (8 classes)        │  player_ball_detector_yolov8.pt
│  + KMeans kit-colour teams  │
└────────────┬────────────────┘
             │ detections [object_id, bbox, confidence]
             ▼
┌─────────────────────────────┐
│  BYTETracker                │  pipeline/tracker.py
│  IoU-based Hungarian match  │  → stable track_ids across frames
└────────────┬────────────────┘
             │ tracked_objects [track_id, object_id, pixel_x/y]
             ▼
┌─────────────────────────────┐
│  HomographyProcessor        │  pipeline/homography/processor.py
│  YOLOv8 keypoints (46 pts)  │  field_keypoint_detector_yolov8.pt
│  Line intersections → H mat │
│  EMA-smoothed H (α=0.2)    │  → world_x_m, world_y_m per object
└────┬──────────────────┬─────┘
     │                  │
     ▼                  ▼
┌──────────┐    ┌────────────────────┐
│ Minimap  │    │ EventDetector      │  pipeline/events/detector.py
│ Widget   │    │ Rule-based         │  pipeline/events/football.py
│ (Qt)     │    │ pass/shot/goal/    │
└──────────┘    │ corner/foul        │
                └─────────┬──────────┘
                          │ events [type, timestamp, team, player_id]
                          ▼
                ┌────────────────────┐
                │ CommentaryGenerator│  pipeline/commentary/generator.py
                │ Template-based     │  → commentary text string
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  Qt UI             │  app/main.py
                │  Commentary panel  │  app/processor_thread.py
                │  Stats panel       │  app/widgets/
                └────────────────────┘
```

---

## Project Structure

```
Football_Commentary_System/
│
├── run.py                          # Entry point — python run.py
│
├── pipeline/                       # Pure Python, zero Qt dependency
│   ├── detector.py                 # YOLO object detector + KMeans team assignment
│   ├── tracker.py                  # BYTETracker wrapper
│   ├── bytetrack.py                # BYTETracker implementation
│   ├── renderer.py                 # OpenCV frame annotation (boxes, tracks)
│   ├── utils.py                    # Shared helpers (kit-colour, LABELS, draw_tracks)
│   ├── homography/
│   │   ├── model.py                # YOLO keypoint model, line intersection, findHomography
│   │   └── processor.py            # HomographyProcessor (EMA-smoothed H, class-map restore)
│   ├── events/
│   │   ├── football.py             # FootballEventDetector (rule-based, world coords)
│   │   └── detector.py             # BaseEventDetector ABC + factory + ensemble
│   └── commentary/
│       └── generator.py            # Template + LLM commentary generators + factory
│
├── app/                            # Qt layer — imports pipeline, never the reverse
│   ├── main.py                     # QMainWindow, theme, team-swap, stats panel
│   ├── processor_thread.py         # QThread orchestrator — calls pipeline in sequence
│   ├── themes.py                   # ThemeManager + _PALETTES (Dark/Light/Blue/Green)
│   └── widgets/
│       ├── video_player.py         # Dual video player widget
│       ├── minimap.py              # Bird's-eye minimap widget (QPainter + EMA positions)
│       └── team_sheet.py           # Team sheet dialog (CSV import, player name table)
│
├── models/
│   ├── player_ball_detector_yolov8.pt      # YOLOv8 8-class object detector
│   └── field_keypoint_detector_yolov8.pt   # YOLOv8 46-keypoint field detector
│
├── data/
│   ├── teamsheets/efl.csv          # EFL team-sheet template (jersey, name, position)
│   └── events/efl.json             # Sample events JSON for commentary testing
│
└── experiments/                    # Per-component research / training sandboxes
    ├── homography/                 # Keypoint detection, line intersection, H matrix
    ├── object_detection/           # YOLO training, kit-colour experiments
    ├── event_detection/            # Action spotting, rule-based threshold tuning
    ├── player_tracking/            # SORT / ByteTrack experiments
    ├── jersey_annotator/           # Annotation tool for jersey-number crops
    ├── jersey_recognition/         # OCR pipeline (VitPose torso + CNN classifier)
    ├── radar_view/                 # Standalone 2Dview.py (full pipeline in one file)
    ├── commentary/                 # Template generator + TTS audio standalone script
    └── homography_pipeline_tests/  # Integration tests, performance benchmarks
```

Each `experiments/` subdirectory has its own `README.md` documenting the files, approach,
problems faced, and what still needs work.

---

## Installation

```bash
git clone https://github.com/PC0907/Football_Commentary_System.git
cd Football_Commentary_System
pip install ultralytics opencv-python PyQt6 numpy scikit-learn tqdm
```

### Optional dependencies

| Package | Required for |
|---------|-------------|
| `pyttsx3` | TTS audio commentary (`experiments/commentary/audio_generator.py`) |
| `pydub` | Audio mixing for video export |
| `ffmpeg` (system) | Video + audio mux in standalone commentary script |
| `torch` | LLM commentary (`PhiCommentaryGenerator` — stub, not yet active) |

---

## Running the App

```bash
python run.py
```

Or as a module:

```bash
python -m app.main
```

### Workflow

1. **Upload Video** — pick a `.mp4` / `.mkv` broadcast clip.
2. **Team Sheet** — enter player names + jersey numbers (or import a CSV).
   CSV format: `jersey_number,name,position`
3. **Process Video** — runs the full pipeline; annotated output is written to `Output_Videos/`.
4. **Export Video** — save to any location.

### Standalone scripts

| Script | What it does |
|--------|--------------|
| `experiments/radar_view/2Dview.py` | Full detection + tracking + homography + radar overlay on a video |
| `experiments/commentary/audio_generator.py` | Generates SRT + TTS audio from a JSON events file |
| `experiments/homography_pipeline_tests/test_homography.py` | Tests homography on a single frame; prints reprojection error |
| `pipeline/detector.py` (run directly) | Tracked + annotated output video, no UI |

---

## Model Weights

| File | Architecture | Classes / Keypoints | Used for |
|------|-------------|---------------------|----------|
| `models/player_ball_detector_yolov8.pt` | YOLOv8m detection | 6 raw → 8 canonical | Object detection in every frame |
| `models/field_keypoint_detector_yolov8.pt` | YOLOv8 pose | 46 field keypoints | Homography H matrix computation |

Both models were trained on the **SoccerNet** dataset and fine-tuned on broadcast footage.

---

## Module Status

### Object Detector (`pipeline/detector.py`)
Wraps YOLOv8 with KMeans kit-colour assignment. Kit classifier is initialised on the first
frame that has ≥ 2 outfield players. GKs are assigned by field half. Canonical IDs (0–7)
are stable from frame 1 onwards. A "⇄ Swap Teams" button in the UI flips team colours in
the minimap when the camera perspective is reversed.

### ByteTracker (`pipeline/bytetrack.py`)
Custom IoU-based multi-object tracker. The `predict()` step is currently a **no-op** (no
Kalman filter) — the main weakness is ball tracking at high speed. The ball loses its
`track_id` on fast shots when the bounding box moves too far between frames.

### Homography (`pipeline/homography/`)
YOLO keypoint model detects up to 46 field-line endpoints; line intersections compute
real field-point pixel coordinates; RANSAC `findHomography` maps pixel → world metres.
The H matrix is **EMA-smoothed** (α = 0.2) to eliminate per-frame RANSAC jitter.
Confidence threshold is **0.5** (lowered from 0.8) to survive partial pitch visibility.

### Event Detector (`pipeline/events/`)
Rule-based detector operating on world-coordinate positions (metres). Detects: pass, shot,
goal, corner, free kick, foul. Event thresholds need tuning — shot speed is in m/frame
and must be normalised to m/s; foul proximity (1 m) is too tight for noisy homography.

### Commentary Generator (`pipeline/commentary/generator.py`)
Template-based generator with per-event phrase lists. Text commentary appears in the UI's
commentary panel in real time. A `PhiCommentaryGenerator` stub exists for LLM-based
commentary but the model is not yet loaded. TTS audio and SRT export are standalone only.

### Jersey Recognition (`experiments/jersey_recognition/`)
Multi-stage pipeline: legibility classifier → VitPose torso crop → CNN digit recogniser.
Works offline on player-crop folders. Not yet integrated into the real-time UI pipeline.

---

## Known Issues & TODO

### High-priority

1. **Ball tracking continuity**: the ball loses `track_id` on fast shots (no Kalman
   filter). Add a constant-velocity Kalman model to bridge occlusion gaps.

2. **Shot speed threshold**: `ball_speed > 5` in `pipeline/events/football.py` is in
   m/frame, not m/s. Normalise: `speed_ms = ball_speed * fps`; threshold should be ~15 m/s.

3. **Foul proximity**: 1 m between players is too tight given homography noise (~0.5 m
   typical reprojection error). Increase to 1.5–2 m and add a velocity-convergence check.

4. **Team swap mid-match**: if the camera pans 180°, Team A / Team B colours flip. The
   "⇄ Swap Teams" button corrects this manually; ideally detect the flip automatically.

### Medium-priority

5. **Jersey recognition integration**: map `track_id → jersey_number → player_name` and
   forward names to the commentary generator.

6. **TTS + SRT export**: wire commentary output to an SRT buffer; add an export button in
   the UI for a fully mixed video.

7. **Re-ID on track re-entry**: when a player exits and re-enters the frame, ByteTrack
   creates a new `track_id`. A re-ID embedding would restore the original ID.

8. **Synchronised video players**: the input and output video players share a common clock
   only loosely — true frame-lock would make scrubbing consistent.

### Low-priority

9. **LLM commentary**: `PhiCommentaryGenerator` stub is in place. Load a small on-device
   model (Phi-3-mini, Gemma-2B) and wire it through the `CommentaryGeneratorFactory`.

10. **Per-player heatmaps**: `experiments/homography_pipeline_tests/player_stats.py` has
    the accumulator prototype. Wire it into `_update_stats()` and render in a separate panel.

---

## Experiments

Each directory under `experiments/` is an independent research sandbox with its own full
`README.md`. Code here is developed and tested in isolation before being ported into the
production `pipeline/` layer.

| Directory | What's explored |
|-----------|----------------|
| `experiments/homography/` | Keypoint detection, Hough lines vs. YOLO, RANSAC tuning |
| `experiments/object_detection/` | YOLOv8 training, kit-colour KMeans |
| `experiments/event_detection/` | Rule-based thresholds, SoccerNet action spotting |
| `experiments/player_tracking/` | SORT vs. ByteTrack, Kalman filter options |
| `experiments/jersey_annotator/` | PyQt5 annotation GUI for jersey-number OCR data |
| `experiments/jersey_recognition/` | VitPose torso crop → CNN OCR pipeline |
| `experiments/radar_view/` | Standalone 2Dview.py (all pipeline stages in one file) |
| `experiments/commentary/` | Template generator + TTS + SRT + ffmpeg mixer |
| `experiments/homography_pipeline_tests/` | Integration tests, reprojection benchmarks |
