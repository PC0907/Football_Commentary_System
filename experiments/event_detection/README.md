# Event Detection — Action Spotting Experiments

This directory is the **research sandbox** for football event detection: identifying
discrete game events (pass, shot, goal, corner, free kick, foul, throw-in) from
video and/or player-position streams.

The production rule-based detector lives in `pipeline/events/` (`football.py` +
`detector.py`). The ML-based detection path (`MLEventDetector`) is currently a stub
pending the work described here.

---

## Purpose

Event detection is the bridge between low-level computer vision (tracking, positions)
and high-level commentary. Two fundamentally different approaches are explored here:

1. **Rule-based** (currently deployed): pure geometry on world-coordinate positions —
   ball speed, player proximity, ball crossing goal/corner lines, etc.
2. **ML-based action spotting** (under research): video classification models that
   recognise events from raw frame sequences.

---

## Sub-directories

### `action_spotting/`

Experiments based on the **SoccerNet Action Spotting** challenge.

| File | Description |
|------|-------------|
| `predictor_as.py` | Inference script using a pre-trained action-spotting model (E2E-Spot / NetVLAD variant). Takes a video clip and returns a JSON of spotted events with timestamps. |
| `README.md` | Setup guide for cloning the upstream `ball-action-spotting` repo and running predictions. |

**Approach**: the model ingests a sliding window of frames (typically 16–64 frames at 2 fps)
and outputs a probability vector over the action vocabulary. Event timestamps are detected
as peaks above a confidence threshold.

**Status**: inference confirmed working on sample clips. Not yet integrated into the
real-time pipeline (would require a frame-buffer + async inference thread).

---

### `ball_action_spotting/`

Experiments using the **`lRomul/ball-action-spotting`** model, which focuses specifically
on ball-contact events (kick, header, save) using ball trajectory.

| File | Description |
|------|-------------|
| `README.md` | Setup and run instructions (clone → install pytorch-argus → run `predictions.py`). |

**Approach**: uses `kornia` geometric transforms to warp frames relative to ball position,
feeding a per-ball-contact window to a ResNet/EfficientNet backbone. Returns event type
at each ball contact point.

**Status**: model runs on isolated clips. The issue is that it requires ball trajectory
as input — which means the homography + tracking pipeline must run first.

---

### `identify_actors.py`

Attempts to identify *which player* is responsible for a detected event (e.g., who passed
the ball) by finding the player closest to the ball at event time. Uses world-coordinate
positions from homography.

---

## The Rule-Based Approach (current production path)

Rather than ML action spotting, the production system uses **`pipeline/events/football.py`**,
which implements the following rules on world-coordinate positions (metres):

| Event | Rule |
|-------|------|
| **Pass** | Ball changes closest-player-team; new closest player different from previous. |
| **Shot** | Ball speed > threshold AND ball moving toward goal AND shooter in attacking half. |
| **Goal** | Ball crosses the goal-line within the goal-post width. |
| **Corner** | Ball exits the field near a corner flag AND last touch from defending team. |
| **Free kick** | Ball stationary near a foul location; possessing player changes. |
| **Foul** | Two players from different teams within 1 m; ball disputed. |

Thresholds and distances operate in **metres** (homography output), so they are
camera-independent.

---

## Problems Faced

| Problem | Root Cause | Fix / Status |
|---------|-----------|--------------|
| Shot threshold poorly calibrated | `ball_speed > 5` is in m/frame not m/s; depends on FPS | Normalise: `speed_ms = ball_speed * fps`; threshold in m/s |
| Pass fires every frame | Ball always has a closest player | Added `_possession_frames` hysteresis: require ≥ 3 frames of possession before counting pass |
| Goal not detected | Ball crosses goal line between frames (high speed) | Interpolate ball trajectory between consecutive frames |
| ML inference too slow for real-time | Action spotting models run at ~5 fps max on CPU | Would require GPU or offline post-processing mode |
| Actor identification wrong | Closest player ≠ ball-contact player | Needs velocity direction from tracking to determine who kicked the ball |

---

## What Still Needs Fixing / Future Work

- [ ] **Calibrate shot speed threshold**: use `speed_ms = ball_velocity_m_per_frame * fps`
  and set threshold at ~15 m/s (typical minimum shot speed).
- [ ] **Goal interpolation**: between frame N and N+1, linearly interpolate ball position
  and check if the path crosses the goal line.
- [ ] **Integrate ML action spotting in offline mode**: run `action_spotting/predictor_as.py`
  on a completed video export, then overlay event markers in a review UI.
- [ ] **Foul detection**: 1 m proximity is too tight for noisy homography. Increase to 1.5 m
  or require velocity direction convergence (players approaching each other).
- [ ] **Throw-in detection**: ball exits side line → nearest player of opposite team restarts
  from same location.
- [ ] **Corner kick confirmation**: require the ball to be stationary in the corner arc for ≥ 5
  frames before firing the corner event.
- [ ] **Actor disambiguation**: when two players are equidistant from the ball, use ball velocity
  direction to pick the kicker (ball travels away from the kicker's position).
