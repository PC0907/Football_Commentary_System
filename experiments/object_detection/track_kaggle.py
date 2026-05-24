import cv2
import numpy as np
import sys
from ultralytics import YOLO
from sklearn.cluster import KMeans

def get_grass_color(img):
    """
    Finds the color of the grass in the background of the image

    Args:
        img: np.array object of shape (WxHx3) that represents the BGR value of the
        frame pixels .

    Returns:
        grass_color
            Tuple of the BGR value of the grass color in the image
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the mean value of the pixels that are not masked
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
    players_imgs = []
    players_boxes = []
    for box in result.boxes:
        label = int(box.cls.cpu().numpy()[0])  # Add .cpu() here
        if label == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Add .cpu() here
            player_img = result.orig_img[y1:y2, x1:x2]
            players_imgs.append(player_img)
            players_boxes.append(box)
    return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
    """
    Finds the kit colors of all the players in the current frame

    Args:
        players: List of np.array objects that contain the BGR values of the image
        portions that contain players.
        grass_hsv: tuple that contain the HSV color value of the grass color of
        the image background.

    Returns:
        kits_colors
            List of np arrays that contain the BGR values of the kits color of all
            the players in the current frame
    """
    kits_colors = []
    if grass_hsv is None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    for player_img in players:
        # Convert image to HSV color space
        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

        # Define range of green color in HSV
        lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
        upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Bitwise-AND mask and original image
        mask = cv2.bitwise_not(mask)
        upper_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
        mask = cv2.bitwise_and(mask, upper_mask)

        kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])

        kits_colors.append(kit_color)
    return kits_colors

def get_kits_classifier(kits_colors):
    kits_kmeans = KMeans(n_clusters=2, n_init='auto')  # Add n_init here
    kits_kmeans.fit(kits_colors)
    return kits_kmeans

def classify_kits(kits_classifer, kits_colors):
    """
    Classifies the player into one of the two teams according to the player's kit
    color

    Args:
        kits_classifer: sklearn.cluster.KMeans object that can classify the
        players kits into 2 teams according to their color.
        kits_colors: List of np.array objects that contain the BGR values of
        the colors of the kits of the players found in the current frame.

    Returns:
        team
            np.array object containing a single integer that carries the player's
            team number (0 or 1)
    """
    team = kits_classifer.predict(kits_colors)
    return team

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    """
    Finds the label of the team that is on the left of the screen

    Args:
        players_boxes: List of ultralytics.engine.results.Boxes objects that
        contain various information about the bounding boxes of the players found
        in the image.
        kits_colors: List of np.array objects that contain the BGR values of
        the colors of the kits of the players found in the current frame.
        kits_clf: sklearn.cluster.KMeans object that can classify the players kits
        into 2 teams according to their color.
    Returns:
        left_team_label
            Int that holds the number of the team that's on the left of the image
            either (0 or 1)
    """
    left_team_label = 0
    team_0 = []
    team_1 = []

    for i in range(len(players_boxes)):
        x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].cpu().numpy())

        team = classify_kits(kits_clf, [kits_colors[i]]).item()
        if team == 0:
            team_0.append(np.array([x1]))
        else:
            team_1.append(np.array([x1]))

    team_0 = np.array(team_0)
    team_1 = np.array(team_1)

    if np.average(team_0) - np.average(team_1) > 0:
        left_team_label = 1

    return left_team_label

def annotate_video(video_path, model):
    try:
        import os
        import cv2
        import numpy as np
        from sklearn.cluster import KMeans

        # Create output directory if it doesn't exist
        os.makedirs('./output', exist_ok=True)

        # Debug paths
        print(f"[DEBUG] Input video path: {os.path.abspath(video_path)}")
        print(f"[DEBUG] Output directory: {os.path.abspath('./output')}")

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.abspath(f'./output/{video_name}_out.mp4')
        print(f"[DEBUG] Output will be saved to: {output_path}")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not output_video.isOpened():
            raise RuntimeError("Failed to initialize video writer")

        # Tracking variables
        kits_clf = None
        left_team_label = 0
        grass_hsv = None
        frame_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Process frame with tracking
            results = model.track(
                frame,
                conf=0.5,
                persist=True,
                tracker="botsort.yaml",
                verbose=False
            )
            result = results[0]

            # Get players and kits
            players_imgs, players_boxes = get_players_boxes(result)
            kits_colors = get_kits_colors(players_imgs, grass_hsv, frame)

            # Initialize on first frame
            if frame_count == 0:
                kits_clf = get_kits_classifier(kits_colors)
                left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
                grass_color = get_grass_color(frame)
                grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

            annotated_frame = frame.copy()

            for box in result.boxes:
                # Move tensor to CPU before conversion
                orig_label = int(box.cls.cpu().numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                text = ""
                label = orig_label

                # Player detection
                if orig_label == 0:
                    # Get team classification
                    kit_color = get_kits_colors([frame[y1:y2, x1:x2]], grass_hsv)
                    team = classify_kits(kits_clf, kit_color)
                    label = 0 if team == left_team_label else 1
                    
                    # Get tracking ID
                    track_id = int(box.id.cpu().item()) if box.id is not None else -1
                    text = f"{labels[label]} {track_id}"

                # Goalkeeper handling
                elif orig_label == 1:
                    label = 2 if x1 < width // 2 else 3
                    text = labels[label]

                # Other objects
                else:
                    label = orig_label + 2
                    text = labels[label]

                # Draw annotations
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[str(label)], 2)
                cv2.putText(annotated_frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_colors[str(label)], 2)

            # Write frame
            output_video.write(annotated_frame)
            frame_count += 1
            
            # Progress update every 5%
            if frame_count % (total_frames // 20) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete ({frame_count}/{total_frames} frames)")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise
    finally:
        # Cleanup resources
        if 'cap' in locals():
            cap.release()
        if 'output_video' in locals():
            output_video.release()
        cv2.destroyAllWindows()

        # Final verification
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) // (1024 * 1024)
            print(f"\nSuccess! Output video created at: {output_path}")
            print(f"Final file size: {file_size} MB")
        else:
            print("\nFailed to create output video")
            
if __name__ == "__main__":
    labels = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]
    box_colors = {
        "0": (150, 50, 50),   # Player-L
        "1": (37, 47, 150),    # Player-R
        "2": (41, 248, 165),   # GK-L
        "3": (166, 196, 10),   # GK-R
        "4": (155, 62, 157),   # Ball
        "5": (123, 174, 213),  # Main Ref
        "6": (217, 89, 204),   # Side Ref
        "7": (22, 11, 15)      # Staff
    }
    model = YOLO("/kaggle/working/Football_Commentary_System/src/Football-object-detection/weights/last.pt")
    video_path = sys.argv[1]
    annotate_video(video_path, model)
