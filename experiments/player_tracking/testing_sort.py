import cv2
import numpy as np
import sys
from ultralytics import YOLO
from sklearn.cluster import KMeans
import supervision as sv
from trackers import DeepSORTFeatureExtractor, DeepSORTTracker

def get_grass_color(img):
    """
    Finds the color of the grass in the background of the image
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the mean value of the pixels that are not masked
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
    """
    Finds the images of the players in the frame and their bounding boxes.
    """
    players_imgs = []
    players_boxes = []
    for box in result.boxes:
        label = int(box.cls.numpy()[0])
        if label == 0:  # Person class
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            player_img = result.orig_img[y1: y2, x1: x2]
            players_imgs.append(player_img)
            players_boxes.append(box)
    return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
    """
    Finds the kit colors of all the players in the current frame
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
    """
    Creates a K-Means classifier that can classify the kits according to their BGR
    values into 2 different clusters each of them represents one of the teams
    """
    kits_kmeans = KMeans(n_clusters=2)
    kits_kmeans.fit(kits_colors)
    return kits_kmeans

def classify_kits(kits_classifier, kits_colors):
    """
    Classifies the player into one of the two teams according to the player's kit color
    """
    team = kits_classifier.predict(kits_colors)
    return team

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    """
    Finds the label of the team that is on the left of the screen
    """
    team_0 = []
    team_1 = []

    for i in range(len(players_boxes)):
        x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())

        team = classify_kits(kits_clf, [kits_colors[i]]).item()
        if team == 0:
            team_0.append(np.array([x1]))
        else:
            team_1.append(np.array([x1]))

    team_0 = np.array(team_0) if team_0 else np.array([[0]])
    team_1 = np.array(team_1) if team_1 else np.array([[0]])

    if np.average(team_0) - np.average(team_1) > 0:
        left_team_label = 1
    else:
        left_team_label = 0

    return left_team_label

def process_video(video_path, detection_model, output_path):
    """
    Process the video with integrated team detection and player tracking
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize DeepSORT tracker
    feature_extractor = DeepSORTFeatureExtractor.from_timm(
        model_name="mobilenetv4_conv_small.e1200_r224_in1k")
    tracker = DeepSORTTracker(feature_extractor=feature_extractor)

    # Initialize Supervision annotators
    team_a_color = sv.Color(150, 50, 50)  # Blue team color (BGR)
    team_b_color = sv.Color(37, 47, 150)  # Red team color (BGR)
    
    # Dictionary for storing team assignments
    player_teams = {}  # track_id -> team (0 or 1)

    # Initialize trace annotator with team colors
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=100
    )

    # Variables for team classification
    kits_clf = None
    left_team_label = 0
    grass_hsv = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Run YOLO on the frame
        result = detection_model(frame, conf=0.5, verbose=False)[0]
        
        # Get player detections
        players_imgs, players_boxes = get_players_boxes(result)
        
        # Convert YOLO detections to Supervision format
        detections = sv.Detections.from_yolov8(result)
        
        # First frame processing - determine team colors
        if frame_count == 1 and players_imgs:
            kits_colors = get_kits_colors(players_imgs, grass_hsv, frame)
            kits_clf = get_kits_classifier(kits_colors)
            left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
            grass_color = get_grass_color(frame)
            grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
        
        # Update tracker with new detections
        tracked_detections = tracker.update(detections, frame)
        
        # Classify players and update team assignments
        for i, (xyxy, _, _, track_id, _) in enumerate(tracked_detections):
            if track_id is not None:
                x1, y1, x2, y2 = map(int, xyxy)
                player_img = frame[y1:y2, x1:x2]
                
                # Skip small detections
                if player_img.shape[0] == 0 or player_img.shape[1] == 0:
                    continue
                    
                kit_color = get_kits_colors([player_img], grass_hsv)
                
                if len(kit_color) > 0:  # Ensure we got a color
                    team = classify_kits(kits_clf, kit_color).item()
                    player_teams[track_id] = team
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw traces first
        annotated_frame = trace_annotator.annotate(annotated_frame, tracked_detections)
        
        # Draw boxes and labels with team-appropriate colors
        for xyxy, _, _, track_id, _ in tracked_detections:
            x1, y1, x2, y2 = map(int, xyxy)
            
            team = player_teams.get(track_id, 0)  # Default to team 0 if not classified
            
            if team == left_team_label:
                color = team_a_color
                label = "Player-L"
            else:
                color = team_b_color
                label = "Player-R"
                
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color.as_bgr, 2)
            
            # Draw track ID
            cv2.putText(
                annotated_frame, 
                f"{label} {track_id}", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color.as_bgr, 
                2
            )
        
        # Write the annotated frame
        out.write(annotated_frame)
        
        # Display progress (optional)
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    # Load YOLO model
    model = YOLO("./weights/last.pt")  # Update with your model path
    
    # Process video
    input_video = sys.argv[1] if len(sys.argv) > 1 else "input_video.mp4"
    output_video = "./output/tracked_teams_output.mp4"
    
    process_video(input_video, model, output_video)
