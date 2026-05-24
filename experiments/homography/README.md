# Homography — Field Projection Experiments

This directory is the **development sandbox** for the homography component: detecting field
keypoints from broadcast video, computing the pixel-to-world-coordinate transformation, and
projecting player positions onto a top-down pitch diagram.

The finished, production-ready version lives in [`pipeline/homography/`](../../pipeline/homography/).

---

## Purpose

A broadcast camera sees the pitch at an angle. To compute real-world positions (e.g., "the
ball is 23 m from the left goal"), we need a **homography matrix H** that maps any pixel
coordinate to a 2D world coordinate in metres.

The approach used here:
1. A YOLOv8 **keypoint model** (`best.pt`, a local copy of `field_keypoint_detector_yolov8.pt`)
   detects up to **46 labelled field keypoints** (line ends, penalty-box corners, centre circle,
   goal posts, etc.).
2. Each field point is defined as the **intersection of two line segments**. If both keypoints
   on each segment are detected with sufficient confidence, their intersection gives one precise
   world point.
3. With ≥ 4 matched pixel↔world point pairs, `cv2.findHomography` with RANSAC computes H.
4. H maps any `(pixel_x, pixel_y)` to `(world_x_m, world_y_m)` on a 105 × 68 m pitch.

---

## Files

| File | Description |
|------|-------------|
| `homography.py` | Core library: YOLO keypoint inference, line-intersection geometry, H matrix computation, `transform_object_positions()`. The direct ancestor of `pipeline/homography/model.py`. |
| `testHomography.py` | CLI test script: runs the full homography pipeline on a single input frame/video and visualises keypoints + warped pitch side-by-side. |
| `line_detection.py` | Earlier experiment: uses classical Hough-transform line detection instead of YOLO keypoints. Abandoned due to false positives from advertising boards and crowd. |
| `LineDetection.ipynb` | Notebook version of the Hough approach. Shows why line detection alone is insufficient. |
| `MultipleLineDetection.ipynb` | Extended Hough experiment with multi-scale filtering. |
| `Final_homography_testing_code.ipynb` | End-to-end notebook: loads a frame, runs the YOLO keypoint model, computes H, and renders the warped pitch with player dots. The definitive experiment before porting to `pipeline/`. |
| `field_point_coordinates.npy` | NumPy array of all 46 world-coordinate field points (metres) in the same order as `KEYPOINT_NAMES`. Used by `compute_field_point_coordinates()`. |
| `best.pt` | Local copy of the YOLO keypoint model used for these experiments. Production code uses `models/field_keypoint_detector_yolov8.pt` at the project root. |
| `00151.jpg`, `inputImage.png`, `selected_frame.jpg` | Sample broadcast frames used for testing. |
| `warped_field.jpg`, `warped_field_with_objects.jpg` | Output images showing the warped top-down pitch. |
| `output/` | Additional test output images/videos. |

---

## How the Component Was Built

### Stage 1 — Classical line detection (`line_detection.py`, `LineDetection.ipynb`)
The first attempt used Hough transforms on the green channel. Problems:
- Advertising boards, crowd, and stadium lights produce many spurious lines.
- Line segments from different pitch markings are visually indistinguishable.
- No reliable way to label *which* line segment maps to *which* world coordinate.
- **Result: abandoned.**

### Stage 2 — YOLO keypoint model
A YOLOv8-pose model was fine-tuned on annotated broadcast frames with **46 keypoint classes**
(one per labelled line endpoint / corner). Each keypoint's identity is known from the class
index, so no line-segment labelling is needed.

Key insight: most field points are not directly visible (e.g., penalty-spot centre), but they
are defined as the **intersection of two detected lines**. Each line is defined by two keypoints
(e.g., "Big rect. left top LEFT" and "Big rect. left top RIGHT"). If both keypoints are detected,
their line can be extended to find the intersection with another detected line.

`FIELD_POINT_TO_KEYPOINT_LINES` in `homography.py` maps each of ~30 computed field points to
the pair of keypoint-pairs that define its two bounding lines.

### Stage 3 — Confidence threshold tuning
With the default confidence of 0.8, fewer than 4 keypoints survived in many frames (partial
pitch visibility, blur, crowd overlap). Lowering to **0.5** dramatically increased keypoint
survival rate while keeping false positives acceptable.

### Stage 4 — RANSAC + EMA smoothing
`findHomography(..., cv2.RANSAC, 5.0)` rejects outlier correspondences. However, RANSAC picks
a different random inlier set each frame, causing the H matrix to jump. In the production version
(`pipeline/homography/processor.py`) an **exponential moving average** (α = 0.2) is applied to H:
```
H_smooth = 0.2 * H_new + 0.8 * H_prev
```
This was validated by comparing jitter in the minimap before/after.

---

## Problems Faced

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Fewer than 4 keypoints detected → H fails | Confidence threshold 0.8 too strict | Lowered to 0.5 |
| Minimap dots jitter every frame | RANSAC random inlier set changes H each frame | EMA smoothing on H (α=0.2) |
| Line detection confuses ad boards with pitch lines | No semantic labelling in Hough | Replaced with keypoint model |
| Field points outside camera FOV → fallback needed | Camera only shows part of pitch | Use previous H matrix if < 4 matches |
| `cv2.findHomography` returns `None` | < 4 correspondences | Guard with `if H is not None` |

---

## What Still Needs Fixing / Future Work

- [ ] **Camera pan/zoom robustness**: when the camera zooms in far, fewer than 4 keypoints
  may be visible. The EMA smoothing helps but a dedicated low-keypoint fallback (e.g., affine
  transform from 3 points) would be more robust.
- [ ] **Reprojection error metric**: `testHomography.py` computes reprojection error but it's
  not currently surfaced in the UI pipeline. Expose it as a confidence score to skip bad frames.
- [ ] **Re-train keypoint model** on more diverse broadcast footage (night games, different
  stadiums, severe weather, heavy zoom).
- [ ] **Fisheye / wide-angle correction**: some broadcast cameras introduce lens distortion.
  Pre-undistort frames before computing H.
- [ ] **3D homography (fundamental matrix)**: for very wide-angle or tilted shots, a planar
  homography breaks down. A projective + perspective model would be more accurate.
