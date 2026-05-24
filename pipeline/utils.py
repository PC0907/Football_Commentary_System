"""
Shared detection utilities used across the Football Commentary pipeline.

Previously these functions were copy-pasted between object_detector.py and
2Dview.py.  Keeping one authoritative copy here prevents divergence.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

# ── Class labels and colour maps ──────────────────────────────────────────────

#: Class labels produced by the 8-class YOLO object model.
#: Index corresponds to the integer class-id in YOLO outputs.
LABELS = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]

#: BGR bounding-box colours keyed by string class-id (for legacy callers).
BOX_COLORS: dict[str, tuple[int, int, int]] = {
    "0": (150,  50,  50),   # Player-L  — dark red
    "1": ( 37,  47, 150),   # Player-R  — dark blue
    "2": ( 41, 248, 165),   # GK-L      — green
    "3": (166, 196,  10),   # GK-R      — yellow-green
    "4": (155,  62, 157),   # Ball      — purple
    "5": (123, 174, 213),   # Main Ref  — light blue
    "6": (217,  89, 204),   # Side Ref  — pink
    "7": ( 22,  11,  15),   # Staff     — near-black
}

#: 20 visually distinctive BGR colours for track trajectory rendering.
TRACK_COLORS: list[tuple[int, int, int]] = [
    (255,   0,   0), (  0, 255,   0), (  0,   0, 255), (255, 255,   0),
    (255,   0, 255), (  0, 255, 255), (128,   0,   0), (  0, 128,   0),
    (  0,   0, 128), (128, 128,   0), (128,   0, 128), (  0, 128, 128),
    ( 64,   0,   0), (  0,  64,   0), (  0,   0,  64), ( 64,  64,   0),
    ( 64,   0,  64), (  0,  64,  64), (192,   0,   0), (  0, 192,   0),
]


# ── Grass / kit colour helpers ────────────────────────────────────────────────

def get_grass_color(img: np.ndarray) -> tuple[float, float, float]:
    """Return the mean BGR colour of grass pixels in *img* (HSV green mask)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([80, 255, 255]))
    return cv2.mean(img, mask=mask)[:3]


def get_grass_hsv(img: np.ndarray) -> np.ndarray:
    """Return a (1,1,3) HSV array of the grass colour in *img*."""
    grass_bgr = get_grass_color(img)
    return cv2.cvtColor(np.uint8([[list(grass_bgr)]]), cv2.COLOR_BGR2HSV)


def get_players_boxes(result):
    """
    Extract player (class-id 0) crops and boxes from a YOLO result object.

    Returns
    -------
    players_imgs : list of np.ndarray  — BGR crops
    players_boxes : list               — corresponding YOLO box objects
    """
    players_imgs, players_boxes = [], []
    try:
        boxes = result.boxes
        cls   = boxes.cls.cpu().numpy()
        xyxy  = boxes.xyxy.cpu().numpy()
        for i in range(len(cls)):
            if int(cls[i]) == 0:
                x1, y1, x2, y2 = map(int, xyxy[i])
                crop = result.orig_img[y1:y2, x1:x2]
                if crop.size > 0:
                    players_imgs.append(crop)
                    players_boxes.append(boxes[i])
    except Exception as exc:
        print(f"[detection_utils] get_players_boxes error: {exc}")
    return players_imgs, players_boxes


def get_kits_colors(
    player_crops: list[np.ndarray],
    grass_hsv: np.ndarray | None = None,
    frame: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Compute the dominant kit colour for each player crop.

    Masks out grass pixels and restricts to the upper-body region (top half of
    the crop) to avoid pitch reflections from trousers / boots.

    Parameters
    ----------
    player_crops : list of BGR crop arrays
    grass_hsv    : pre-computed grass HSV (1,1,3); computed from *frame* if None
    frame        : full frame used only when *grass_hsv* is None
    """
    if grass_hsv is None:
        if frame is None:
            raise ValueError("Provide either grass_hsv or frame to get_kits_colors")
        grass_hsv = get_grass_hsv(frame)

    hue = int(grass_hsv[0, 0, 0])
    lo  = np.array([max(0,   hue - 10), 40, 40])
    hi  = np.array([min(180, hue + 10), 255, 255])

    kits_colors = []
    for crop in player_crops:
        if crop.size == 0:
            kits_colors.append(np.zeros(3, dtype=np.float32))
            continue
        hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask  = cv2.bitwise_not(cv2.inRange(hsv, lo, hi))  # non-grass pixels
        # Restrict to upper half (shirt region)
        upper = np.zeros(crop.shape[:2], np.uint8)
        upper[: crop.shape[0] // 2, :] = 255
        mask  = cv2.bitwise_and(mask, upper)
        color = np.array(cv2.mean(crop, mask=mask)[:3], dtype=np.float32)
        kits_colors.append(color)
    return kits_colors


def get_kits_classifier(kits_colors: list[np.ndarray]) -> KMeans:
    """Fit a 2-cluster KMeans on kit colours to separate the two teams."""
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(kits_colors)
    return km


def classify_kits(classifier: KMeans, kits_colors: list[np.ndarray]) -> np.ndarray:
    """Return team labels (0 or 1) for each kit colour vector."""
    return classifier.predict(kits_colors)


def get_left_team_label(
    players_boxes,
    kits_colors: list[np.ndarray],
    kits_clf: KMeans,
) -> int:
    """
    Determine which team cluster corresponds to the left side of the frame.

    Returns 0 if team-0 players are on the left, 1 otherwise.
    The result is used to keep team labels consistent across frames.
    """
    team_0_xs, team_1_xs = [], []
    try:
        for i, box in enumerate(players_boxes):
            x1 = int(box.xyxy.cpu().numpy()[0][0])
            label = classify_kits(kits_clf, [kits_colors[i]])[0]
            (team_0_xs if label == 0 else team_1_xs).append(x1)
        if team_0_xs and team_1_xs and np.mean(team_0_xs) > np.mean(team_1_xs):
            return 1
    except Exception as exc:
        print(f"[detection_utils] get_left_team_label error: {exc}")
    return 0


# ── Rendering helpers ─────────────────────────────────────────────────────────

def draw_tracks(
    frame: np.ndarray,
    tracked_objects: list[dict],
    trajectory_history: dict,
    max_trajectory_points: int = 30,
) -> np.ndarray:
    """
    Draw track circles, labels, and trajectory tails on *frame* in-place.

    Parameters
    ----------
    tracked_objects   : list of dicts with track_id, object_id, pixel_x, pixel_y
    trajectory_history: mutable dict mapping track_id → list of (x, y) tuples
    max_trajectory_points: maximum history length per track (older points dropped)
    """
    for obj in tracked_objects:
        track_id = obj["track_id"]
        label_id = obj["object_id"]
        x, y = int(obj["pixel_x"]), int(obj["pixel_y"])

        color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
        cv2.circle(frame, (x, y), 5, color, -1)

        label_text = (
            f"{LABELS[label_id]} #{track_id}"
            if label_id < len(LABELS)
            else f"cls{label_id} #{track_id}"
        )
        cv2.putText(frame, label_text, (x - 30, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update and draw trajectory tail
        history = trajectory_history.setdefault(track_id, [])
        history.append((x, y))
        if len(history) > max_trajectory_points:
            trajectory_history[track_id] = history[-max_trajectory_points:]
            history = trajectory_history[track_id]

        for i in range(1, len(history)):
            cv2.line(frame, history[i - 1], history[i], color, 2)

    return frame
