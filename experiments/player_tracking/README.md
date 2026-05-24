# Player Tracking — Multi-Object Tracking Experiments

This directory contains experiments for **multi-object tracking (MOT)** applied to
football players and the ball across video frames.

The production tracker is `pipeline/tracker.py` (wrapping `pipeline/bytetrack.py`).
The experiments here informed that implementation.

---

## Purpose

Object detection produces a new set of bounding boxes each frame, but has no memory of
previous frames — the same player gets a fresh detection every frame with no stable ID.
**Multi-object tracking** assigns a persistent `track_id` to each detected object across
frames so that downstream components (stats, event detection, minimap) can follow
individual players.

---

## Files

| File | Description |
|------|-------------|
| `testing_sort.py` | SORT (Simple Online and Realtime Tracking) implementation experiment. Tests IoU-based Hungarian matching on a sample video. Verified concept but SORT has no re-identification — IDs switch on occlusion. |
| `README.md` | This file. |

---

## Approach: ByteTrack

After evaluating SORT (see `testing_sort.py`), the pipeline moved to a custom
**BYTETracker** implementation (`pipeline/bytetrack.py`). ByteTrack improves over SORT by:

- Keeping **low-confidence detections** in a second candidate pool rather than discarding them.
  When a tracked object briefly falls below the detection threshold (e.g., ball occluded by a
  player), it stays alive in the low-confidence pool and can be re-associated when it
  reappears.
- Using **two-stage matching**: first match high-confidence detections to existing tracks
  (tight IoU), then try to match remaining tracks with low-confidence detections (looser IoU).

### Tracking pipeline summary

```
detections (list[dict])   ←── ObjectDetector.detect(frame)
        │
        ▼
BYTETracker.update(detections)
        │  Hungarian matching on IoU between predicted boxes
        │  and new detections; dead tracks removed after
        │  track_buffer frames of no match.
        ▼
tracked_objects (list[dict])
    { track_id, object_id, pixel_x, pixel_y, width, height,
      x1, y1, x2, y2, confidence }
```

`track_id` is a monotonically increasing integer assigned when a track is first created.
`object_id` is the canonical class (0–7) determined by kit colour, preserved from the
detection that initiated the track.

---

## Problems Faced

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| `AttributeError: BYTETracker has no _frame_id` | `__init__` missing `self._frame_id = 0` | Added in `pipeline/bytetrack.py` |
| Ball loses track on fast shots | No Kalman filter; `predict()` is a no-op → IoU = 0 on next frame | **Not fully fixed** — see Future Work |
| Players swap track IDs in dense clusters | IoU-based matching breaks when players overlap | Inherent SORT/ByteTrack limitation; re-ID models would help |
| Team label flips mid-match after re-ID | `object_id` from detection re-assigned inconsistently | Fixed: `_class_map` in `HomographyProcessor` caches `track_id → class` persistently |
| Track buffer too short → ID churn | `track_buffer=30` drops tracks after ~1 s off-screen | Increased to `track_buffer=60` for slower cameras |

---

## What Still Needs Fixing / Future Work

- [ ] **Add Kalman filter to the ball track**: model ball position as constant-velocity
  particle; the `predict()` step advances it one step per frame so IoU matching still works
  even when the ball is occluded for a few frames. This is the highest-priority tracking bug.
- [ ] **Re-identification for re-entering players**: when a player leaves the frame and
  returns, ByteTrack creates a new track ID. A re-ID embedding (e.g., fast-ReID or
  OSNet-extracted appearance features) would allow re-associating the old ID.
- [ ] **Evaluate BoT-SORT**: BoT-SORT (ByteTrack + camera motion compensation + re-ID)
  is the current state-of-the-art for sports tracking and would likely outperform the current
  custom implementation.
- [ ] **Separate ball tracker**: the ball behaves very differently from players (higher speed,
  smaller box, frequent occlusion). A dedicated single-object ball tracker (e.g., correlation
  filter or PF-based) run in parallel would be more reliable.
- [ ] **Kalman-based velocity estimation**: derive ball speed directly from tracker state
  rather than computing frame-to-frame position difference (noisy at low speeds).
