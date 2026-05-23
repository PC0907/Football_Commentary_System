import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc


def draw_football_pitch(ax, color='white', linewidth=2):
    """
    Draw a football pitch using the provided keypoints and connections
    """
    # Define the keypoints
    KEYPOINTS_DATA = [
        (1, 0, 0, "far_left_corner"),
        (2, 52.5, 0, "far_center_line_end"),
        (3, 105, 0, "far_right_corner"),
        (4, 0, 13.84, "left_outer_box_far_left"),
        (5, 16.5, 13.84, "left_outer_box_far_right"),
        (6, 88.5, 13.84, "right_outer_box_far_left"),
        (7, 105, 13.84, "right_outer_box_far_right"),
        (8, 0, 24.84, "left_inner_box_far_left"),
        (9, 5.5, 24.84, "left_inner_box_far_right"),
        (10, 99.5, 24.84, "right_inner_box_far_left"),
        (11, 105, 24.84, "right_inner_box_far_right"),
        (12, 52.5, 24.85, "center_circle_far_point"),
        (13, 16.5, 26.69, "left_arc_far_point"),
        (14, 88.5, 26.69, "right_arc_far_point"),
        (15, 0, 30.34, "left_goal_far_post"),
        (16, 105, 30.34, "right_goal_far_post"),
        (17, 11, 34, "left_penalty_spot"),
        (18, 52.5, 34, "center_circle_center"),
        (19, 94, 34, "right_penalty_spot"),
        (20, 0, 37.66, "left_goal_near_post"),
        (21, 105, 37.66, "right_goal_near_post"),
        (22, 16.5, 41.31, "left_arc_near_point"),
        (23, 88.5, 41.31, "right_arc_near_point"),
        (24, 52.5, 43.15, "center_circle_near_point"),
        (25, 0, 43.16, "left_inner_box_near_left"),
        (26, 5.5, 43.16, "left_inner_box_near_right"),
        (27, 99.5, 43.16, "right_inner_box_near_left"),
        (28, 105, 43.16, "right_inner_box_near_right"),
        (29, 0, 54.16, "left_outer_box_near_left"),
        (30, 16.5, 54.16, "left_outer_box_near_right"),
        (31, 88.5, 54.16, "right_outer_box_near_left"),
        (32, 105, 54.16, "right_outer_box_near_right"),
        (33, 0, 68, "near_left_corner"),
        (34, 52.5, 68, "near_center_line_end"),
        (35, 105, 68, "near_right_corner"),
    ]
    
    # Convert to dictionary for easy lookup
    keypoints = {}
    for k in KEYPOINTS_DATA:
        keypoints[k[0]] = (k[1], k[2])
    
    # Define connections
    CONNECTIONS = [
        (33, 35),
        (33, 1),
        (1, 3),
        (35, 3),
        (29, 30),
        (30, 5),
        (4, 5),
        (34, 2),
        (32, 31),
        (31, 6),
        (6, 7),
        (25, 26),
        (26, 9),
        (9, 8),
        (28, 27),
        (27, 10),
        (10, 11),
        (16, 21),
        (20, 15),
    ]
    
    # Draw connection lines
    for conn in CONNECTIONS:
        start_point = keypoints[conn[0]]
        end_point = keypoints[conn[1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                color=color, linewidth=linewidth)
    
    # Draw center circle
    center_x, center_y = keypoints[18]
    circle_radius = 9.15  # Standard radius of center circle
    center_circle = Circle((center_x, center_y), circle_radius, fill=False, 
                          color=color, linewidth=linewidth)
    ax.add_patch(center_circle)
    
    # Draw penalty arcs
    left_arc_center = keypoints[17]
    right_arc_center = keypoints[19]
    
    # Penalty arcs
    penalty_area_radius = 9.15
    
    # Left penalty arc (arc spanning ~120 degrees)
    left_penalty_arc = Arc(left_arc_center, 2*penalty_area_radius, 2*penalty_area_radius,
                         angle=0, theta1=-60, theta2=60, color=color, linewidth=linewidth)
    
    # Right penalty arc (arc spanning ~120 degrees)
    right_penalty_arc = Arc(right_arc_center, 2*penalty_area_radius, 2*penalty_area_radius,
                          angle=0, theta1=120, theta2=240, color=color, linewidth=linewidth)
    
    ax.add_patch(left_penalty_arc)
    ax.add_patch(right_penalty_arc)
    
    # Draw center spot and penalty spots
    center_spot = Circle(keypoints[18], 0.5, color=color)
    left_penalty_spot = Circle(keypoints[17], 0.5, color=color)
    right_penalty_spot = Circle(keypoints[19], 0.5, color=color)
    
    ax.add_patch(center_spot)
    ax.add_patch(left_penalty_spot)
    ax.add_patch(right_penalty_spot)
    
    # Set aspect ratio to equal and remove ticks
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set limits
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 73)


def create_radar_view(tracked_objects, pitch_length=105, pitch_width=68, radar_size=(640, 360)):
    """
    Create a radar view image from tracked objects with a specific size
    """
    # Create a figure with dark background and specified size
    fig_width, fig_height = radar_size[0]/100, radar_size[1]/100  # Convert pixels to inches (assuming 100 dpi)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100, facecolor='#222222')
    ax.set_facecolor('#222222')
    
    # Draw the football pitch
    draw_football_pitch(ax, color='#ffffff')
    
    # Define colors for different objects
    object_colors = {
        0: '#ff3333',  # Player-L (red)
        1: '#3333ff',  # Player-R (blue)
        2: '#33ff33',  # GK-L (green)
        3: '#ffff33',  # GK-R (yellow)
        4: '#ffffff',  # Ball (white)
        5: '#ff33ff',  # Main Ref (magenta)
        6: '#33ffff',  # Side Ref (cyan)
        7: '#aaaaaa'   # Staff (gray)
    }
    
    # Define marker styles for different objects
    object_markers = {
        0: 'o',  # Player-L
        1: 'o',  # Player-R
        2: 's',  # GK-L (square)
        3: 's',  # GK-R (square)
        4: '*',  # Ball (star)
        5: '^',  # Main Ref (triangle)
        6: '^',  # Side Ref (triangle)
        7: 'x'   # Staff (x)
    }
    
    half_pitch_length = pitch_length / 2
    half_pitch_width = pitch_width / 2
    
    # Inside create_radar_view function
# When plotting each tracked object, flip the y-coordinate
    for obj in tracked_objects:
        if 'world_x' in obj and 'world_y' in obj:
            # Convert from meters to pitch coordinates
            x = obj['world_x']  # Already centered
            y = pitch_width - obj['world_y']  # Flip the y-coordinate
            
            # Check if coordinates are within reasonable bounds
            if abs(x) <= half_pitch_length*2 + 10 and abs(y) <= half_pitch_width*2 + 10:
                object_id = obj['object_id']
                track_id = obj.get('track_id', 0)
                
                # Plot the object with appropriate color and marker
                ax.plot(x, y, 
                    marker=object_markers.get(object_id, 'o'),
                    color=object_colors.get(object_id, '#ffffff'),
                    markersize=6 if object_id == 4 else 4,
                    markeredgecolor='white' if object_id != 4 else None)
    # Remove legend for the mini radar
    plt.tight_layout(pad=0.1)
    
    # Convert to image
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Convert to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    return img


def process_video_with_radar(video_path, model_path, output_dir='./output'):
    """
    Process video with football tracking and create radar view overlay
    """
    try:
        # Import required modules here to avoid circular imports
        import cv2
        import numpy as np
        import sys
        import time
        import os
        from tqdm import tqdm
        from ultralytics import YOLO
        from sklearn.cluster import KMeans
        from homography import process_frame
        from bytetrack import BYTETracker
        
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
        out_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_tracked_radar_overlay.mp4")
        
        # Create output video writer for the combined video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        if not output_video.isOpened():
            print(f"Error: Could not create output video file")
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

        # Define radar size and position
        radar_width, radar_height = 320, 180  # Size of the radar view
        
        print(f"Processing video: {video_path}")
        print(f"Output will be saved to: {out_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")

        # Create progress bar
        pbar = tqdm(total=total_frames, desc="Processing frames")

        # Define labels and box colors (copied from original code)
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
                
                color = track_colors[track_id % len(track_colors)]
                cv2.circle(frame, (x, y), 5, color, -1)
                
                cv2.putText(
                    frame,
                    f"{labels[label]} #{track_id}",
                    (x - 30, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
                
                if track_id not in trajectory_history:
                    trajectory_history[track_id] = []
                
                trajectory_history[track_id].append((x, y))
                
                if len(trajectory_history[track_id]) > max_trajectory_points:
                    trajectory_history[track_id] = trajectory_history[track_id][-max_trajectory_points:]
                
                if len(trajectory_history[track_id]) > 1:
                    for i in range(1, len(trajectory_history[track_id])):
                        pt1 = trajectory_history[track_id][i-1]
                        pt2 = trajectory_history[track_id][i]
                        cv2.line(frame, pt1, pt2, color, 2)
            
            return frame

        def overlay_radar(frame, radar_img):
            """
            Overlay radar image on the bottom center of the frame
            """
            h, w = frame.shape[:2]
            r_h, r_w = radar_img.shape[:2]
            
            # Calculate position (bottom center)
            x_offset = w // 2 - r_w // 2
            y_offset = h - r_h - 20  # 20 pixels from bottom
            
            # Create a semi-transparent overlay background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_offset-10, y_offset-10), 
                          (x_offset+r_w+10, y_offset+r_h+10), (0, 0, 0), -1)
            
            # Apply the transparent overlay
            alpha = 0.
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
            
            # Apply the radar image
            roi = frame[y_offset:y_offset+r_h, x_offset:x_offset+r_w]
            
            # Create a mask for the radar (non-black pixels)
            gray = cv2.cvtColor(radar_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            
            # Black out the area of radar in ROI
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            
            # Take only region of radar from radar image
            img2_fg = cv2.bitwise_and(radar_img, radar_img, mask=mask)
            
            # Put radar in ROI and modify the frame
            dst = cv2.add(img1_bg, img2_fg)
            frame[y_offset:y_offset+r_h, x_offset:x_offset+r_w] = dst
            
            # Add a border around the radar
            cv2.rectangle(frame, (x_offset-10, y_offset-10), 
                          (x_offset+r_w+10, y_offset+r_h+10), (200, 200, 200), 2)
            
            return frame

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
                    # Use the bottom center of the bounding box instead of the center
                    center_x = (x1 + x2) / 2
                    center_y = (y1+y2) / 2  # Bottom of the bounding box

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

                # Create radar view with specific size
                radar_img = create_radar_view(tracked_objects, radar_size=(radar_width, radar_height))
                
                # Add game time to radar view
                minutes = int((frame_id / fps) // 60)
                seconds = int((frame_id / fps) % 60)
                cv2.putText(
                    radar_img,
                    f"{minutes:02d}:{seconds:02d}",
                    (radar_width - 50, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                # Overlay radar on annotated frame
                final_frame = overlay_radar(annotated_frame, radar_img)
                
                output_video.write(final_frame)

            except Exception as e:
                print(f"\nError processing frame {frame_id}: {str(e)}")
                continue

        pbar.close()
        print(f"\nSuccessfully processed {frame_id} frames")
        print(f"Output video saved to: {out_path}")
        return True

    except Exception as e:
        print(f"\nError in process_video_with_radar: {str(e)}")
        return False

    finally:
        if 'cap' in locals():
            cap.release()
        if 'output_video' in locals():
            output_video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "/home/fawwaz/Downloads/CityUtdR.mp4"
    model_path = "./best_object.pt"
    success = process_video_with_radar(video_path, model_path)
    if not success:
        print("Video processing failed")