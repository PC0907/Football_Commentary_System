# Automatic Football Commentary System

A computer-vision pipeline that ingests broadcast football footage and produces annotated video with real-time object tracking, bird's-eye-view radar, event detection, and auto-generated commentary (text + audio).

---

## Table of Contents
- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Module Status](#module-status)
- [Known Issues & TODO](#known-issues--todo)
- [Contributing](#contributing)

---

## What It Does

| Feature | Status |
|---------|--------|
| Upload a match video via GUI | ✅ Working |
| YOLO-based object detection (players, ball, ref, GK, staff) | ✅ Working |
| Kit-colour team assignment (K-Means on HSV) | ✅ Working |
| Multi-object tracking (ByteTrack) | ✅ Working (see bugs below) |
| Homography → bird's-eye-view radar overlay | ✅ Working (standalone `2Dview.py`) |
| Homography in the **UI pipeline** | ⚠️ Stubbed — returns empty positions |
| 2D event detection (pass, shot, goal, corner, foul) | ⚠️ Stubbed — always returns [] |
| Jersey number recognition | ⚠️ Stubbed in UI — standalone pipeline exists |
| Template-based commentary from JSON events | ✅ Working (standalone `commentary.py`) |
| TTS audio commentary | ✅ Working (standalone `commentary.py`) |
| SRT subtitle overlay via ffmpeg | ✅ Working (standalone `commentary.py`) |
| Themeable PyQt6 desktop UI | ✅ Working |
| Team sheet editor (manual entry + CSV import) | ✅ Working |
| Player stats tracking (possession, passes, shots…) | ⚠️ Defined but not wired to UI |

---

## Architecture

```
Broadcast video
       │
       ▼
┌──────────────────┐
│  Object Detector │  YOLO (best_object.pt)
│  8 classes:      │  Player-L/R, GK-L/R, Ball,
│  + Kit KMeans    │  Main Ref, Side Ref, Staff
└────────┬─────────┘
         │ detections
         ▼
┌──────────────────┐
│   ByteTracker    │  IoU-based Hungarian matching
│   (bytetrack.py) │  → stable track_ids across frames
└────────┬─────────┘
         │ tracked_objects
         ▼
┌──────────────────┐    ┌────────────────────┐
│  Homography      │───▶│ 2D Radar View      │
│  (homography.py) │    │ (2Dview.py)        │
│  YOLO keypoints  │    │ Matplotlib pitch   │
│  → H matrix      │    │ overlay on video   │
└────────┬─────────┘    └────────────────────┘
         │ world coordinates (metres)
         ▼
┌──────────────────┐
│ Event Detector   │  Pass / Shot / Goal /
│ (2D_event_       │  Corner / Free-kick / Foul
│  detector.py)    │
└────────┬─────────┘
         │ events
         ▼
┌──────────────────┐    ┌────────────────────┐
│ Commentary Gen.  │───▶│ SRT subtitles      │
│ (commentary.py)  │    │ Audio (TTS)        │
│ JSON templates   │    │ ffmpeg overlay     │
└──────────────────┘    └────────────────────┘
```

The **UI pipeline** (`modules/ui/`) runs all of the above inside a `QThread` (`VideoProcessor`) so the GUI stays responsive during processing.

---

## Project Structure

```
Football_Commentary_System/
│
├── modules/ui/                         ← Main application
│   ├── main.py                         ← Entry point (PyQt6 app)
│   ├── themes.py                       ← Light/Dark/Blue/Green themes
│   ├── Input_Videos/                   ← Drop your .mkv/.mp4 here
│   ├── Output_Videos/                  ← Processed output saved here
│   ├── events_data/                    ← Per-video JSON event files
│   ├── teamsheets/                     ← Per-video player CSV files
│   └── components/
│       ├── processor.py                ← QThread pipeline orchestrator
│       ├── object_detector.py          ← YOLO wrapper + kit assignment
│       ├── bytetrack.py                ← ByteTrack multi-object tracker
│       ├── tracker.py                  ← Thin wrapper over BYTETracker
│       ├── homography_processor.py     ← UI adapter (⚠️ stubbed)
│       ├── homography.py               ← Full homography implementation ✅
│       ├── 2D_event_detector.py        ← Event logic (⚠️ not wired to UI)
│       ├── 2Dview.py                   ← Standalone radar-overlay script ✅
│       ├── commentary.py               ← SRT + TTS commentary generator ✅
│       ├── renderer.py                 ← OpenCV overlay renderer
│       ├── video_player.py             ← In-app video player widget
│       ├── team_sheet.py               ← Team sheet dialog (manual + CSV)
│       ├── pipeline_workers.py         ← Worker classes (not wired yet)
│       ├── best_object.pt              ← YOLO object detection weights
│       └── best.pt                     ← YOLO keypoint detection weights
│
├── jersey_recognition_pipeline/        ← Standalone jersey number OCR
│   ├── src/pipeline/
│   │   ├── football_pipeline.py
│   │   ├── object_detection.py
│   │   ├── torso_extraction.py
│   │   └── classifier.py
│   └── main.py
│
├── Homography/                         ← Research notebooks & tests
│   ├── homography.py
│   ├── line_detection.py
│   └── testHomography.py
│
├── Football-object-detection/          ← Earlier detection experiments
│   ├── track.py
│   └── CustomCode/detect_ball.py
│
├── event_detection/                    ← Action Spotting experiments
│   └── action_spotting/predictor_as.py
│
├── Player_Tracking/
│   └── testing_sort.py
│
├── Integrated_Gemini/AFCS/             ← Prototype integration (WIP)
│   ├── object_detection.py
│   ├── tracking.py
│   ├── homography.py
│   └── [several stubs — pipeline.py, commentary_generation.py, etc.]
│
├── JerseyImageAnnotator/               ← Tool for annotating jersey images
│   └── annotator.py
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd Football_Commentary_System

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. System dependencies

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 3. Model weights

The two YOLO model files are checked in under `modules/ui/components/`:
- `best_object.pt` — detects players, ball, referee, staff (8 classes)
- `best.pt` — detects field keypoints for homography (46 keypoints)

If they are missing (Git LFS not pulled), download them and place them in that directory.

---

## Running the App

```bash
cd modules/ui
python main.py
```

### Workflow

1. **Upload Video** — pick a `.mp4` / `.mkv` broadcast clip.
2. **Team Sheet** — enter player names + jersey numbers (or import a CSV).
   - CSV format: `jersey_number,name,position`
3. **Process Video** — runs object detection + tracking; writes annotated output to `Output_Videos/`.
4. **Export Video** — save the output to any location.

### Standalone scripts

| Script | What it does |
|--------|--------------|
| `modules/ui/components/2Dview.py` | Runs full detection + tracking + homography + radar overlay on a single video |
| `modules/ui/components/commentary.py` | Generates SRT + TTS audio from a JSON events file |
| `jersey_recognition_pipeline/main.py` | Runs jersey number OCR on a folder of player crops |
| `Homography/testHomography.py` | Tests the homography transform on a single frame |

---

## Module Status

### Object Detection (`object_detector.py`)
Wraps YOLO with KMeans kit-colour assignment to distinguish the two teams. The `ObjectDetector` class is clean. The module also contains a `process_video()` standalone function — this mixes concerns and should eventually be moved.

### ByteTrack (`bytetrack.py`)
Custom IoU-based multi-object tracker. No Kalman filter — the `predict()` step is a no-op, which means fast-moving objects (especially the ball) will lose their track ID on occlusion. Consider replacing with BoT-SORT or the official ByteTrack with Kalman.

### Homography (`homography.py`)
Detects 46 field keypoints with YOLO, computes line intersections to find field-point pixel coordinates, then uses RANSAC `findHomography` to map pixel → metres. Works well in `2Dview.py`. **Not yet wired into the UI `VideoProcessor`** — `homography_processor.py` is still a stub.

### Event Detection (`2D_event_detector.py`)
Full rule-based detector for passes, shots, goals, corners, free kicks, and fouls, operating on world-coordinate positions (metres). Depends on the homography output. **Not connected to the UI pipeline** — `DummyEventDetector` is used instead.

### Commentary (`commentary.py`)
Template-based generator. Reads a JSON events file (with `predictions` array), generates SRT subtitles, and optionally synthesises TTS audio and mixes it onto the video with ffmpeg. Works standalone. **Not connected to the real-time pipeline** — `DummyCommentaryGenerator` is used in the UI.

### Jersey Recognition (`jersey_recognition_pipeline/`)
Crops the torso region using pose keypoints, pre-processes the image, and runs an OCR classifier. Works as a standalone pipeline. **Not integrated into the UI.**

---

## Known Issues & TODO

### Bugs (fixed in this version)
- ✅ `BYTETracker` referenced `self._frame_id` but never initialised it → `AttributeError` on every call to track expiry
- ✅ `object_detector.py` `process_video()` referenced `reprojection_error` / `confidence` that were never in scope → `NameError`
- ✅ GK side-assignment compared `x1 < 0.5 * bbox_width` instead of `center_x < 0.5 * frame_width`
- ✅ `renderer.py` passed a `list` directly to `cv2.putText()` → `TypeError`
- ✅ `processor.py` always called `event_detector.detect([])` with a hardcoded empty list

### High-priority TODOs

1. **Wire homography into the UI pipeline**
   - In `processor.py`, replace `HomographyProcessor` (stub) with a call to `homography.process_frame()`.
   - This unblocks event detection and the radar view.

2. **Wire `FootballEventDetector` into the UI pipeline**
   - Replace `DummyEventDetector` in `processor.py` with `FootballEventDetector` from `2D_event_detector.py`.
   - Feed it `ball_pos` and `players_pos` in world coordinates (from homography).

3. **Wire commentary into the real-time pipeline**
   - `DummyCommentaryGenerator` should be replaced with a live generator that pulls templates from `commentary.py` based on detected events.

4. **Add Kalman filter to ByteTrack**
   - The current `predict()` is a no-op. Ball tracking is especially unreliable on fast shots.

5. **Integrate jersey recognition**
   - After tracking, crop each player's torso and run `jersey_recognition_pipeline` to get jersey numbers.
   - Map jersey numbers → player names via the loaded team sheet.

6. **Synchronise dual video players**
   - The input and output video players in the UI play independently; they should share a single timer.

7. **Remove code duplication**
   - Kit-colour detection functions are copy-pasted between `object_detector.py` and `2Dview.py` — extract to a shared `utils/kit_colors.py`.
   - `bytetrack.py` exists in three directories; keep one canonical copy.
   - `homography.py` exists in three directories; same issue.

8. **Fill in `Integrated_Gemini/AFCS/` stubs**
   - `pipeline.py`, `commentary_generation.py`, and `team_sheet.py` are empty files.

9. **Event detection thresholds need tuning**
   - Shot: `ball_speed > 5` (metres/frame at whatever FPS) — unit is ambiguous; needs normalising to m/s.
   - Possession radius `1.5 m` may be too tight for noisy homography output.
   - Foul proximity `1.0 m` between players is extremely tight.

10. **`pipeline_workers.py` interface mismatch**
    - `TrackingWorker.process()` calls `self.tracker.update(frame, field_positions)` but the tracker only accepts `(detections)`.

---

## Contributing

Branch from `main`. The active feature branches are:
- `homography` — homography research
- `object_detection_and_tracking` — detection & tracking work
- `UI` — front-end / Qt work

Open a PR against `main` when a feature is ready and all existing standalone scripts still run without errors.
