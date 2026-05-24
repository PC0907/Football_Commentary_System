# Radar View — 2D Pitch Overlay Script

This directory contains the **standalone radar-view script** that combines all pipeline
stages (object detection, team assignment, tracking, homography) into a single runnable
file and renders a bird's-eye pitch overlay alongside the broadcast video.

The script here is `2Dview.py`. This was the definitive integration test before each
component was split into the production `pipeline/` and `app/` packages.

---

## Purpose

The radar view (also called the minimap) shows a top-down schematic of the pitch with
coloured dots representing each player and the ball in real-world metric coordinates.
It is the most visual proof that the full pipeline — from pixel detections to world
positions — is working correctly.

`2Dview.py` runs the complete stack as a single standalone script so that each stage
can be inspected and debugged without launching the Qt UI.

---

## Files

| File | Description |
|------|-------------|
| `2Dview.py` | Full standalone pipeline script. Detects, tracks, applies homography, draws the radar overlay on each frame, and writes an annotated output video. Run with `python 2Dview.py [video_path] [model_path]`. |

---

## How It Works

```
Input video
    │
    ▼
Frame-by-frame loop
    ├── YOLO object detection (player_ball_detector_yolov8.pt)
    │       → bounding boxes, raw class IDs
    ├── Kit-colour KMeans team assignment (first frame)
    │       → object_id (0=Team-A, 1=Team-B, 2=GK-L, 3=GK-R, 4=Ball…)
    ├── BYTETracker multi-object tracking
    │       → stable track_ids across frames
    ├── Homography (field_keypoint_detector_yolov8.pt)
    │       → 46 keypoints → field point intersections → H matrix
    │       → pixel positions → (world_x_m, world_y_m)
    └── Radar overlay (matplotlib pitch diagram composited onto frame)
            → annotated frame written to output video
```

The radar panel shows:
- A line-drawn 105 × 68 m pitch (penalty boxes, centre circle, goal areas).
- **Blue dots** = Team A players.
- **Red dots** = Team B players.
- **Yellow dot** = Ball.
- **Cyan/Magenta dots** = Goalkeepers.

---

## How the Script Was Built

`2Dview.py` started as a single-file experiment to verify that homography worked end-to-end.
It was the first file to:

1. Wire up the YOLO keypoint model for homography.
2. Implement kit-colour KMeans team separation.
3. Render a matplotlib pitch figure and composite it onto the video frame using OpenCV.

When the Qt UI was built, the logic from this file was split into:
- `pipeline/detector.py` (YOLO + kit-colour)
- `pipeline/tracker.py` + `pipeline/bytetrack.py` (tracking)
- `pipeline/homography/model.py` + `pipeline/homography/processor.py` (H matrix)
- `app/widgets/minimap.py` (Qt minimap widget replacing matplotlib)

`2Dview.py` is kept here as a **reference implementation** — it is easier to diff against
than the split modules when debugging cross-module integration issues.

---

## Problems Faced

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Radar overlay flickers | Matplotlib `fig.canvas.draw()` is slow; redraw every frame | Cache the pitch background image; only redraw player positions each frame |
| Player dots jump every frame | RANSAC picks different H each frame | EMA smoothing on H matrix (α=0.2); see `pipeline/homography/processor.py` |
| Wrong team colours | KMeans cluster order changed between frames | Fixed: compute `_left_label` once on frame 1, cache for the full video |
| Output video large | Composite frame (1080p + matplotlib panel side-by-side) | Downscale composite to 720p for output |
| `from .bytetrack import BYTETracker` breaks standalone run | Relative import inside package | `2Dview.py` uses the pipeline package — must be run as `python -m experiments.radar_view.2Dview` or add the project root to `sys.path` first |

---

## What Still Needs Fixing / Future Work

- [ ] **Replace matplotlib with OpenCV drawing**: matplotlib compositing is 5× slower than
  pure OpenCV line/circle drawing. The Qt minimap widget already uses QPainter — port the
  same logic to an OpenCV overlay for the standalone script.
- [ ] **Fix standalone import path**: `2Dview.py` has relative imports that fail when run
  directly. Add a `sys.path.insert(0, project_root)` preamble or provide a shell wrapper.
- [ ] **Add team-swap flag**: `--swap-teams` CLI argument to flip Team A / Team B colours
  when the camera perspective is reversed.
- [ ] **EMA tuning**: the current α=0.2 trades off responsiveness vs. smoothness. Expose
  it as a `--ema-alpha` CLI argument for experimentation.
- [ ] **Write output to SRT**: detect events during the radar pass and write an SRT subtitle
  file as a byproduct, so `2Dview.py` can be used as a full offline commentary generator.
