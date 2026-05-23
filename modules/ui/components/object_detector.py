import cv2
import numpy as np
import sys
import time
import os
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.cluster import KMeans
from .homography import process_frame
from .bytetrack import BYTETracker  # Import the new ByteTrack class

# Global labels and box colors
labels = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]
box_colors = {
    "0": (150, 50, 50),
    "1": (37, 47, 150),
    "2": (41, 248, 165),
    "3": (166, 196, 10),
    "4": (155, 62, 157),
    "5": (123, 174, 213),
    "6": (217, 89, 204),
    "7": (22, 11, 15)
}

# Track colors (20 distinctive colors for track visualization)
track_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
    (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0)
]

def get_grass_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
    players_imgs, players_boxes = [], []
    try:
        boxes = result.boxes
        cls = boxes.cls.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        for i in range(len(cls)):
            if int(cls[i]) == 0:
                x1, y1, x2, y2 = map(int, xyxy[i])
                player_img = result.orig_img[y1:y2, x1:x2]
                players_imgs.append(player_img)
                players_boxes.append(boxes[i])
    except Exception as e:
        print(f"Error in get_players_boxes: {str(e)}")
        return [], []
    return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
    kits_colors = []
    if grass_hsv is None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    for player_img in players:
        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
        upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)

        upper_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_mask[:player_img.shape[0] // 2, :] = 255
        mask = cv2.bitwise_and(mask, upper_mask)

        kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])
        kits_colors.append(kit_color)
    return kits_colors

def get_kits_classifier(kits_colors):
    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(kits_colors)
    return kmeans

def classify_kits(classifier, kits_colors):
    return classifier.predict(kits_colors)

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    try:
        team_0, team_1 = [], []
        for i in range(len(players_boxes)):
            box = players_boxes[i]
            xyxy = box.xyxy.cpu().numpy()[0]
            x1 = int(xyxy[0])
            team = classify_kits(kits_clf, [kits_colors[i]])[0]
            if team == 0:
                team_0.append(x1)
            else:
                team_1.append(x1)
        if np.mean(team_0) > np.mean(team_1):
            return 1
    except Exception as e:
        print(f"Error in get_left_team_label: {str(e)}")
    return 0

def draw_tracks(frame, tracked_objects, trajectory_history, max_trajectory_points=30):
    """
    Draw tracks and their trajectories
    """
    for obj in tracked_objects:
        track_id = obj['track_id']
        label = obj["object_id"]
        x, y = int(obj['pixel_x']), int(obj['pixel_y'])
        
        # Draw current position
        color = track_colors[track_id % len(track_colors)]
        cv2.circle(frame, (x, y), 5, color, -1)
        
        # Add label with track ID
        cv2.putText(
            frame,
            f"{labels[label]} #{track_id}",
            (x - 30, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        # Update trajectory history
        if track_id not in trajectory_history:
            trajectory_history[track_id] = []
        
        trajectory_history[track_id].append((x, y))
        
        # Limit trajectory length
        if len(trajectory_history[track_id]) > max_trajectory_points:
            trajectory_history[track_id] = trajectory_history[track_id][-max_trajectory_points:]
        
        # Draw trajectory
        if len(trajectory_history[track_id]) > 1:
            for i in range(1, len(trajectory_history[track_id])):
                pt1 = trajectory_history[track_id][i-1]
                pt2 = trajectory_history[track_id][i]
                cv2.line(frame, pt1, pt2, color, 2)
    
    return frame

def process_video(video_path, model_path, output_dir='./output'):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_name = os.path.basename(video_path)
        out_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_tracked_out.mp4")
        
        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        if not output_video.isOpened():
            print(f"Error: Could not create output video file {out_path}")
            return False

        model = YOLO(model_path)
        kits_clf = None
        left_team_label = 0
        grass_hsv = None
        frame_id = 0
        
        # Initialize ByteTrack
        tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, 
                              track_buffer=30, frame_rate=fps)
        
        # Initialize trajectory history
        trajectory_history = {}

        print(f"Processing video: {video_path}")
        print(f"Output will be saved to: {out_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")

        # Create progress bar
        pbar = tqdm(total=total_frames, desc="Processing frames")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_id += 1
            pbar.update(1)

            try:
                result = model(frame, conf=0.5, verbose=False)[0]
                players_imgs, players_boxes = get_players_boxes(result)
                if not players_imgs:
                    continue

                kits_colors = get_kits_colors(players_imgs, grass_hsv, frame)

                if frame_id == 1:
                    kits_clf = get_kits_classifier(kits_colors)
                    left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
                    grass_color = get_grass_color(frame)
                    grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

                detections = []
                boxes = result.boxes
                cls = boxes.cls.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()

                for i in range(len(cls)):
                    label = int(cls[i])
                    confidence = float(conf[i])
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    if label == 0:
                        kit_color = get_kits_colors([frame[y1:y2, x1:x2]], grass_hsv)
                        team = classify_kits(kits_clf, kit_color)[0]
                        label = 0 if team == left_team_label else 1
                    elif label == 1:
                        label = 2 if x1 < 0.5 * width else 3
                    else:
                        label += 2

                    detections.append({
                        'object_id': label,
                        'pixel_x': center_x,
                        'pixel_y': center_y,
                        'width': width,
                        'height': height,
                        'confidence': confidence
                    })

                # Update tracker with new detections
                tracked_objects = tracker.update(detections)

                # Process frame with homography
                object_positions = [{
                    'object_id': obj['object_id'],
                    'pixel_x': obj['pixel_x'],
                    'pixel_y': obj['pixel_y']
                } for obj in tracked_objects]

                transformed_positions, reprojection_error, confidence, _ = process_frame(
                    frame, object_positions, frame_id
                )

                # Add real-world coordinates to tracked objects
                for i, obj in enumerate(tracked_objects):
                    if i < len(transformed_positions):
                        obj['world_x'] = transformed_positions[i]['world_x_meters']
                        obj['world_y'] = transformed_positions[i]['world_y_meters']

                # Create annotated frame with tracks
                annotated_frame = frame.copy()
                annotated_frame = draw_tracks(annotated_frame, tracked_objects, trajectory_history)

                # Add performance metrics
                cv2.putText(
                    annotated_frame,
                    f"Error: {reprojection_error:.2f}m",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    annotated_frame,
                    f"Confidence: {confidence:.2%}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    annotated_frame,
                    f"Tracks: {len(tracked_objects)}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                output_video.write(annotated_frame)

            except Exception as e:
                print(f"\nError processing frame {frame_id}: {str(e)}")
                continue

        pbar.close()
        print(f"\nSuccessfully processed {frame_id} frames")
        print(f"Output video saved to: {out_path}")
        return True

    except Exception as e:
        print(f"\nError in process_video: {str(e)}")
        return False

    finally:
        if 'cap' in locals():
            cap.release()
        if 'output_video' in locals():
            output_video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "/home/fawwaz/Downloads/footballVideos/match_video_022.mp4"
    model_path = "./best_object.pt"
    success = process_video(video_path, model_path)
    if not success:
        print("Video processing failed")