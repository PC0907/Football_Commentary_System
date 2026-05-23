"""
Standalone script: detection + tracking + homography + radar overlay.

Run directly:
    python 2Dview.py  [video_path]  [model_path]

Kit-colour helpers and drawing utilities are imported from
utils/detection_utils.py (single source of truth).
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from .bytetrack import BYTETracker
from .homography import process_frame as homography_process_frame
from ..utils.detection_utils import (
    LABELS, TRACK_COLORS,
    get_grass_hsv, get_players_boxes,
    get_kits_colors, get_kits_classifier, classify_kits, get_left_team_label,
    draw_tracks,
)


# ── Pitch drawing ─────────────────────────────────────────────────────────────

#: Real-world keypoint positions (metres, origin at top-left corner).
_KEYPOINTS: dict[int, tuple[float, float]] = {
    kp[0]: (kp[1], kp[2]) for kp in [
        ( 1,  0.0,  0.0), ( 2, 52.5,  0.0), ( 3, 105.0,  0.0),
        ( 4,  0.0, 13.84), ( 5, 16.5, 13.84), ( 6,  88.5, 13.84), ( 7, 105.0, 13.84),
        ( 8,  0.0, 24.84), ( 9,  5.5, 24.84), (10,  99.5, 24.84), (11, 105.0, 24.84),
        (12, 52.5, 24.85), (13, 16.5, 26.69), (14,  88.5, 26.69),
        (15,  0.0, 30.34), (16, 105.0, 30.34),
        (17, 11.0, 34.0),  (18, 52.5, 34.0),  (19, 94.0, 34.0),
        (20,  0.0, 37.66), (21, 105.0, 37.66),
        (22, 16.5, 41.31), (23,  88.5, 41.31),
        (24, 52.5, 43.15), (25,  0.0, 43.16), (26,  5.5, 43.16),
        (27, 99.5, 43.16), (28, 105.0, 43.16),
        (29,  0.0, 54.16), (30, 16.5, 54.16), (31,  88.5, 54.16), (32, 105.0, 54.16),
        (33,  0.0, 68.0),  (34, 52.5, 68.0),  (35, 105.0, 68.0),
    ]
}

_CONNECTIONS = [
    (33, 35), (33, 1), (1, 3), (35, 3),
    (29, 30), (30, 5), (4, 5), (34, 2),
    (32, 31), (31, 6), (6, 7),
    (25, 26), (26, 9), (9, 8),
    (28, 27), (27, 10), (10, 11),
    (16, 21), (20, 15),
]


def draw_football_pitch(ax, color: str = "white", linewidth: float = 2.0):
    """Draw a standard football pitch on *ax* using pre-defined keypoints."""
    for a, b in _CONNECTIONS:
        xs = [_KEYPOINTS[a][0], _KEYPOINTS[b][0]]
        ys = [_KEYPOINTS[a][1], _KEYPOINTS[b][1]]
        ax.plot(xs, ys, color=color, linewidth=linewidth)

    # Centre circle
    cx, cy = _KEYPOINTS[18]
    ax.add_patch(Circle((cx, cy), 9.15, fill=False, color=color, linewidth=linewidth))

    # Penalty arcs (outside the box, facing outward)
    for kp_id, t1, t2 in [(17, -60, 60), (19, 120, 240)]:
        px, py = _KEYPOINTS[kp_id]
        ax.add_patch(Arc((px, py), 2 * 9.15, 2 * 9.15,
                         angle=0, theta1=t1, theta2=t2,
                         color=color, linewidth=linewidth))

    # Spots
    for kp_id in (17, 18, 19):
        ax.add_patch(Circle(_KEYPOINTS[kp_id], 0.5, color=color))

    ax.set_aspect("equal")
    ax.set_xticks([]);  ax.set_yticks([])
    ax.set_xlim(-5, 110);  ax.set_ylim(-5, 73)


# ── Radar view ────────────────────────────────────────────────────────────────

_OBJ_COLORS = {
    0: "#ff3333", 1: "#3333ff", 2: "#33ff33", 3: "#ffff33",
    4: "#ffffff", 5: "#ff33ff", 6: "#33ffff", 7: "#aaaaaa",
}
_OBJ_MARKERS = {0: "o", 1: "o", 2: "s", 3: "s", 4: "*", 5: "^", 6: "^", 7: "x"}


def create_radar_view(
    tracked_objects: list[dict],
    pitch_length: float = 105.0,
    pitch_width:  float =  68.0,
    radar_size:   tuple[int, int] = (640, 360),
) -> np.ndarray:
    """
    Render a bird's-eye radar view of tracked objects onto a pitch diagram.

    Parameters
    ----------
    tracked_objects : each dict must have ``world_x``, ``world_y``, ``object_id``.
    radar_size      : output image size in pixels (width, height).

    Returns
    -------
    BGR np.ndarray of shape (radar_size[1], radar_size[0], 3).
    """
    fig_w, fig_h = radar_size[0] / 100, radar_size[1] / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100, facecolor="#222222")
    ax.set_facecolor("#222222")
    draw_football_pitch(ax)

    half_l, half_w = pitch_length / 2, pitch_width / 2

    for obj in tracked_objects:
        if "world_x" not in obj or "world_y" not in obj:
            continue
        x = float(obj["world_x"])
        y = pitch_width - float(obj["world_y"])   # flip Y so pitch reads correctly
        if abs(x) > half_l * 2 + 10 or abs(y) > half_w * 2 + 10:
            continue
        oid = int(obj.get("object_id", 0))
        ax.plot(x, y,
                marker=_OBJ_MARKERS.get(oid, "o"),
                color=_OBJ_COLORS.get(oid, "#ffffff"),
                markersize=6 if oid == 4 else 4,
                markeredgecolor="white" if oid != 4 else None)

    plt.tight_layout(pad=0.1)
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def overlay_radar(frame: np.ndarray, radar_img: np.ndarray) -> np.ndarray:
    """Composite *radar_img* into the bottom-centre of *frame* in-place."""
    fh, fw  = frame.shape[:2]
    rh, rw  = radar_img.shape[:2]
    xo = fw // 2 - rw // 2
    yo = fh - rh - 20

    roi  = frame[yo:yo + rh, xo:xo + rw]
    gray = cv2.cvtColor(radar_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg  = cv2.bitwise_and(roi,       roi,       mask=mask_inv)
    fg  = cv2.bitwise_and(radar_img, radar_img, mask=mask)
    frame[yo:yo + rh, xo:xo + rw] = cv2.add(bg, fg)
    cv2.rectangle(frame, (xo - 10, yo - 10), (xo + rw + 10, yo + rh + 10),
                  (200, 200, 200), 2)
    return frame


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_video_with_radar(
    video_path: str,
    model_path: str,
    output_dir: str = "./output",
) -> bool:
    """
    Full pipeline: detection → tracking → homography → radar overlay.

    Saves annotated output video with a bird's-eye radar inset.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: could not open {video_path}")
            return False

        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name  = os.path.basename(video_path)
        out_path    = os.path.join(output_dir,
                                   f"{os.path.splitext(video_name)[0]}_tracked_radar_overlay.mp4")

        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (width, height))
        if not out.isOpened():
            print("Error: could not create output video")
            return False

        model         = YOLO(model_path)
        tracker       = BYTETracker(track_thresh=0.5, match_thresh=0.8,
                                     track_buffer=30, frame_rate=int(fps))
        traj_history  = {}
        kits_clf      = None
        left_label    = 0
        grass_hsv     = None
        frame_id      = 0
        radar_w, radar_h = 320, 180

        print(f"Processing {total} frames → {out_path}")
        pbar = tqdm(total=total, desc="Radar overlay")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            pbar.update(1)

            try:
                result = model(frame, conf=0.5, verbose=False)[0]
                players_imgs, players_boxes = get_players_boxes(result)
                if not players_imgs:
                    out.write(frame)
                    continue

                if frame_id == 1:
                    grass_hsv  = get_grass_hsv(frame)
                    kc_init    = get_kits_colors(players_imgs, grass_hsv)
                    kits_clf   = get_kits_classifier(kc_init)
                    left_label = get_left_team_label(players_boxes, kc_init, kits_clf)

                boxes_r  = result.boxes
                cls_arr  = boxes_r.cls.cpu().numpy()
                xyxy_arr = boxes_r.xyxy.cpu().numpy()
                conf_arr = boxes_r.conf.cpu().numpy()
                frame_width = frame.shape[1]

                detections = []
                for i in range(len(cls_arr)):
                    lbl = int(cls_arr[i])
                    x1, y1, x2, y2 = map(int, xyxy_arr[i])
                    w = x2 - x1;  h = y2 - y1
                    cx = (x1 + x2) / 2.0;  cy = (y1 + y2) / 2.0

                    if lbl == 0:
                        crop = frame[y1:y2, x1:x2]
                        kit  = get_kits_colors([crop], grass_hsv)
                        team = classify_kits(kits_clf, kit)[0]
                        lbl  = 0 if team == left_label else 1
                    elif lbl == 1:
                        lbl = 2 if cx < 0.5 * frame_width else 3
                    else:
                        lbl += 2

                    detections.append({
                        "object_id":  lbl,
                        "pixel_x":    cx,
                        "pixel_y":    cy,
                        "width":      float(w),
                        "height":     float(h),
                        "confidence": float(conf_arr[i]),
                    })

                tracked = tracker.update(detections)

                # Homography → world coordinates
                obj_pixels = [{"object_id": t["object_id"],
                               "pixel_x": t["pixel_x"],
                               "pixel_y": t["pixel_y"]} for t in tracked]
                transformed, reproj_error, confidence, _ = homography_process_frame(
                    frame, obj_pixels, frame_id
                )

                for i, t_obj in enumerate(tracked):
                    if i < len(transformed):
                        t_obj["world_x"] = transformed[i]["world_x_meters"]
                        t_obj["world_y"] = transformed[i]["world_y_meters"]

                # Annotate + radar
                annotated = draw_tracks(frame.copy(), tracked, traj_history)

                cv2.putText(annotated, f"Error: {reproj_error:.2f}m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"Confidence: {confidence:.1%}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"Tracks: {len(tracked)}",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                radar_img = create_radar_view(tracked, radar_size=(radar_w, radar_h))
                mins = int((frame_id / fps) // 60)
                secs = int((frame_id / fps) % 60)
                cv2.putText(radar_img, f"{mins:02d}:{secs:02d}",
                            (radar_w - 50, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)

                final = overlay_radar(annotated, radar_img)
                out.write(final)

            except Exception as exc:
                print(f"\nError on frame {frame_id}: {exc}")
                out.write(frame)
                continue

        pbar.close()
        print(f"\nDone — {frame_id} frames.  Output: {out_path}")
        return True

    except Exception as exc:
        print(f"\nFatal error: {exc}")
        return False

    finally:
        if "cap" in dir() and cap.isOpened():
            cap.release()
        if "out" in dir():
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    _video = sys.argv[1] if len(sys.argv) > 1 else "/home/fawwaz/Downloads/CityUtdR.mp4"
    _model = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / "best_object.pt")
    if not process_video_with_radar(_video, _model):
        sys.exit(1)
