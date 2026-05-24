# Homography Pipeline Tests — Integration & Performance Tests

This directory contains **integration test scripts** that verify the full homography
pipeline — from raw video frame to world-coordinate player positions — works correctly
and within acceptable performance bounds.

These tests sit between unit tests (which test individual functions) and end-to-end
UI tests (which require the full Qt app). They run without the Qt dependency.

---

## Purpose

The homography component is the most fragile part of the pipeline: it depends on YOLO
keypoint detection quality, line-intersection geometry, and RANSAC stability. Small
changes to the model, confidence threshold, or field-point definitions can silently
break position accuracy. These tests catch regressions before they reach the UI.

---

## Files

| File | Description |
|------|-------------|
| `test_homography.py` | Main integration test. Loads a known broadcast frame, runs the homography pipeline, and checks that the computed world positions are within expected bounds. Also benchmarks processing time per frame. |
| `player_stats.py` | `PlayerStats` class experiment: tracks per-player statistics (distance covered, possession time, heatmap) from a stream of world-coordinate positions. Used as a proof-of-concept for the stats accumulator. |
| `homography_test_results.jpg` | Reference output image saved from a passing test run. Shows the warped pitch with player positions overlaid. Useful for visual regression checking. |
| `output/` | Directory where test run outputs (images, JSON reports) are saved. |

---

## Running the Tests

```bash
# From the project root
cd experiments/homography_pipeline_tests

# Basic test on a single frame
python test_homography.py

# With custom image and model
python test_homography.py \
    --image /path/to/broadcast_frame.jpg \
    --model ../../models/field_keypoint_detector_yolov8.pt
```

Expected output:
```
Processing time: 0.142 seconds
Homography Metrics:
  Frame ID: 1
  Reprojection Error: 0.83 meters
  Confidence Score: 81.25%
Transformed positions: [...]
```

A reprojection error below **1.5 m** and confidence above **60 %** (≥ 4 matched field
points out of the maximum ~30 detectable) indicate a passing test.

---

## `player_stats.py` — Stats Accumulator Prototype

`PlayerStats` was the first prototype of per-player statistics accumulation. It tracks:

- **Distance covered** (metres): integrates Euclidean distance between consecutive
  world-coordinate positions.
- **Possession time** (frames/seconds): counts frames where the player is within 1.5 m
  of the ball.
- **Heatmap**: divides the pitch into a 20 × 13 grid and accumulates visits per cell.
- **Velocity**: computed from the last 30 stored positions.

This class informed the stats-accumulator logic now inside `app/processor_thread.py`'s
`_update_stats()` method.

---

## How the Tests Were Built

### Initial problem
When `HomographyProcessor` was first wired into `VideoProcessor`, world positions were
all returning `(0, 0)` — the H matrix was `None` because fewer than 4 field points were
matched. This test script was written to isolate that bug.

Findings:
- Default YOLO confidence of 0.8 left fewer than 4 surviving keypoints on most frames.
- Lowering to **0.5** resolved the issue.
- Field-point computation was silently skipping intersections where lines were parallel
  (degenerate case); added a determinant guard (`abs(det) < 1e-10 → skip`).

### Benchmark results (on test hardware)
| Stage | Time (ms) |
|-------|-----------|
| YOLO keypoint inference | 95–140 |
| Line-intersection geometry | < 1 |
| `findHomography` (RANSAC) | 2–5 |
| `transform_object_positions` | < 1 |
| **Total** | **~100–150 ms / frame** |

This is below the 33 ms frame budget for 30 fps but acceptable for the pipeline's
async thread (the UI renders the previous frame while the background thread processes
the next).

---

## Problems Faced

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| All positions returned `(0, 0)` | H matrix `None` due to < 4 matches | Lowered confidence threshold to 0.5 |
| `cv2.perspectiveTransform` shape error | Input array must be `(N, 1, 2)` not `(N, 2)` | Reshape with `.reshape(-1, 1, 2)` |
| `reprojection_error` / `confidence` undefined | Old function signature mismatch | Updated `test_homography.py` to use new `HomographyProcessor.process()` API |
| Parallel lines → division-by-zero | Degenerate case in line-intersection math | Added determinant guard |

---

## What Still Needs Fixing / Future Work

- [ ] **Automate with pytest**: convert `test_homography.py` to proper `pytest` test cases
  with assertions so CI can run them.
- [ ] **Add ground-truth positions**: annotate a set of test frames manually with known
  player positions (metres) and compare against pipeline output to compute absolute
  position error.
- [ ] **Stress test with challenging frames**: night games, heavy crowd, extreme zoom,
  camera on the short side of the pitch.
- [ ] **PlayerStats integration**: `player_stats.py` works standalone but is not yet called
  from `app/processor_thread.py`. Wire it up and expose per-player distance and heatmap in
  the stats panel.
- [ ] **Heatmap visualisation**: render the accumulated heatmap on a pitch diagram at the
  end of a session (export as PNG or display in a separate dialog in the UI).
