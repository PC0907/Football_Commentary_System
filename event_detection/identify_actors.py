import cv2
import numpy as np
import sys
import json
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Functions from your original object detection code that remain unchanged
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
    masked_img = cv2.bitwise_and(img, img, mask=mask)
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
        if label == 0:
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
    Creates a K-Means classifier that can classify the kits accroding to their BGR
    values into 2 different clusters each of them represents one of the teams
    """
    kits_kmeans = KMeans(n_clusters=2)
    kits_kmeans.fit(kits_colors)
    return kits_kmeans

def classify_kits(kits_classifer, kits_colors):
    """
    Classifies the player into one of the two teams according to the player's kit
    color
    """
    team = kits_classifer.predict(kits_colors)
    return team

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    """
    Finds the label of the team that is on the left of the screen
    """
    left_team_label = 0
    team_0 = []
    team_1 = []

    for i in range(len(players_boxes)):
        x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())

        team = classify_kits(kits_clf, [kits_colors[i]]).item()
        if team==0:
            team_0.append(np.array([x1]))
        else:
            team_1.append(np.array([x1]))

    team_0 = np.array(team_0)
    team_1 = np.array(team_1)

    if np.average(team_0) - np.average(team_1) > 0:
        left_team_label = 1

    return left_team_label

# New functions for actor identification
def extract_frames_around_position(cap, position, fps, window_size=2):
    """
    Extracts frames around a specific position in the video
    """
    frames = []
    
    # Calculate frame range
    start_frame = max(0, position - int(window_size * fps))
    end_frame = position + int(window_size * fps)
    
    for frame_idx in range(start_frame, end_frame, max(1, int(fps/5))):  # Sample 5 frames per second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if success:
            frames.append(frame)
    
    return frames

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def process_frame_actors(result, kits_clf, left_team_label, grass_hsv, event_type):
    """
    Processes detected objects in a frame to identify potential actors
    """
    actors = []
    
    # Get ball position if available
    ball_position = None
    for box in result.boxes:
        label = int(box.cls.numpy()[0])
        if label == 4:  # Ball
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            ball_position = ((x1 + x2) // 2, (y1 + y2) // 2)
            break
    
    # Process all detected objects
    for box in result.boxes:
        label = int(box.cls.numpy()[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
        
        actor_info = {
            "bbox": [x1, y1, x2, y2],
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "confidence": float(box.conf.numpy()[0])
        }
        
        # Process player
        if label == 0:  # Player
            player_img = result.orig_img[y1:y2, x1:x2]
            kit_color = get_kits_colors([player_img], grass_hsv)
            team = classify_kits(kits_clf, kit_color).item()
            
            actor_info["type"] = "Player"
            actor_info["team"] = "Left" if team == left_team_label else "Right"
            
            # Calculate distance to ball if available
            if ball_position:
                ball_dist = calculate_distance(actor_info["center"], ball_position)
                actor_info["ball_distance"] = ball_dist
        
        # Process goalkeeper
        elif label == 1:  # Goalkeeper
            actor_info["type"] = "Goalkeeper"
            actor_info["team"] = "Left" if x1 < result.orig_img.shape[1] // 2 else "Right"
        
        # Process ball
        elif label == 4:  # Ball
            actor_info["type"] = "Ball"
        
        # Process referee
        elif label == 5 or label == 6:  # Referee
            actor_info["type"] = "Referee"
            actor_info["role"] = "Main" if label == 5 else "Side"
        
        else:
            continue
        
        actors.append(actor_info)
    
    return actors

def identify_key_actors(event_actors, event_type):
    """
    Identifies key actors involved in an event based on event type
    """
    # Group actors by type and team
    players_left = [a for a in event_actors if a.get("type") == "Player" and a.get("team") == "Left"]
    players_right = [a for a in event_actors if a.get("type") == "Player" and a.get("team") == "Right"]
    goalkeepers = [a for a in event_actors if a.get("type") == "Goalkeeper"]
    balls = [a for a in event_actors if a.get("type") == "Ball"]
    
    key_actors = []
    
    # Apply event-specific rules for selected events
    if event_type == "Kick-off":
        # For kick-off, find players near the center of the field and close to the ball
        center_field = (640, 360)  # Assuming 1280x720 video
        if balls:
            ball_pos = balls[0]["center"]
            # Get players from both teams near center
            all_players = players_left + players_right
            # Sort by distance to center
            center_players = sorted(all_players, 
                                   key=lambda p: calculate_distance(p["center"], center_field))
            key_actors = center_players[:2]  # Take closest two players
    
    elif event_type == "Throw-in":
        # For throw-in, find player near sideline with hands up
        sideline_players = []
        for player in players_left + players_right:
            x_pos = player["center"][0]
            # Check if player is near sideline (left or right edge)
            if x_pos < 100 or x_pos > 1180:  # Assuming 1280-wide frame
                sideline_players.append(player)
        
        if sideline_players:
            # Take player most likely to be throwing
            key_actors = [sideline_players[0]]
            
            # Also find potential receiver(s)
            if balls and sideline_players:
                thrower = sideline_players[0]
                thrower_team = thrower.get("team")
                
                # Find teammates near the throw-in location
                teammates = players_left if thrower_team == "Left" else players_right
                teammates = [p for p in teammates if p != thrower]
                
                # Sort by distance to thrower
                sorted_teammates = sorted(teammates, 
                                      key=lambda p: calculate_distance(p["center"], thrower["center"]))
                
                if sorted_teammates:
                    key_actors.append(sorted_teammates[0])  # Add closest teammate
    
    elif event_type == "Goal":
        # For goal, identify scorer, possible assister, and goalkeeper
        
        # First, determine which goal area the ball is in
        goal_side = None
        if balls:
            ball_x = balls[0]["center"][0]
            if ball_x < 200:  # Left goal
                goal_side = "Left"
            elif ball_x > 1080:  # Right goal
                goal_side = "Right"
        
        if goal_side:
            # The scoring team is opposite to the goal side
            scoring_team = "Right" if goal_side == "Left" else "Left"
            
            # Get players from scoring team near goal
            scoring_team_players = players_left if scoring_team == "Left" else players_right
            
            # Get goalkeeper from conceding team
            conceding_gk = [gk for gk in goalkeepers if gk.get("team") == goal_side]
            
            # Add nearest attacking player as scorer
            if scoring_team_players:
                # Sort by distance to goal
                goal_center = (100, 360) if goal_side == "Left" else (1180, 360)
                sorted_players = sorted(scoring_team_players,
                                      key=lambda p: calculate_distance(p["center"], goal_center))
                
                if sorted_players:
                    key_actors.append(sorted_players[0])  # Likely scorer
                    
                    # If there's a second attacker nearby, they might be the assister
                    if len(sorted_players) > 1:
                        key_actors.append(sorted_players[1])
            
            # Add goalkeeper
            if conceding_gk:
                key_actors.append(conceding_gk[0])
    
    # For other events, use a simple default approach
    else:
        # Default: get players closest to the ball
        if balls:
            ball_pos = balls[0]["center"]
            all_players = players_left + players_right
            # Sort by distance to ball
            sorted_players = sorted(all_players, 
                                   key=lambda p: calculate_distance(p["center"], ball_pos))
            key_actors = sorted_players[:2]  # Take closest two players
    
    # Format the output
    processed_actors = []
    for actor in key_actors:
        processed_actor = {
            "type": actor["type"],
            "team": actor.get("team"),
            "confidence": actor["confidence"]
        }
        processed_actors.append(processed_actor)
    
    return processed_actors

def process_events_with_actors(video_path, events_json, model):
    """
    Processes each detected event to identify actors (players) involved
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize kit classifier (will be set on first frame)
    kits_clf = None
    left_team_label = None
    grass_hsv = None
    
    enhanced_events = []
    
    for event in events_json["predictions"]:
        # Convert position to frame number
        frame_position = int(event["position"])
        
        # Extract frames around the event (e.g., ±2 seconds)
        frames_to_analyze = extract_frames_around_position(cap, frame_position, fps, window_size=2)
        
        # Process each frame to detect objects
        event_actors = []
        for frame in frames_to_analyze:
            # Run object detection
            result = model(frame, conf=0.5, verbose=False)[0]
            
            # Get player info
            players_imgs, players_boxes = get_players_boxes(result)
            kits_colors = get_kits_colors(players_imgs, grass_hsv, frame)
            
            # Initialize kit classifier if needed
            if kits_clf is None:
                kits_clf = get_kits_classifier(kits_colors)
                left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
                grass_color = get_grass_color(result.orig_img)
                grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
            
            # Process detected objects
            frame_actors = process_frame_actors(result, kits_clf, left_team_label, grass_hsv, event["label"])
            event_actors.extend(frame_actors)
        
        # Process collected actors to find most likely event participants
        key_actors = identify_key_actors(event_actors, event["label"])
        
        # Add actors to event
        enhanced_event = event.copy()
        enhanced_event["actors"] = key_actors
        enhanced_events.append(enhanced_event)
    
    return {"UrlLocal": events_json["UrlLocal"], "predictions": enhanced_events}

def annotate_video(video_path, model):
    """
    Loads the input video and runs the object detection algorithm on its frames, finally it annotates the frame with
    the appropriate labels
    """
    cap = cv2.VideoCapture(video_path)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_name = video_path.split('/')[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('./output/'+video_name.split('.')[0] + "_out.mp4",
                                   fourcc,
                                   30.0,
                                   (width, height))

    kits_clf = None
    left_team_label = 0
    grass_hsv = None

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if success:

            # Run YOLOv8 inference on the frame
            annotated_frame = cv2.resize(frame, (width, height))
            result = model(annotated_frame, conf=0.5, verbose=False)[0]

            # Get the players boxes and kit colors
            players_imgs, players_boxes = get_players_boxes(result)
            kits_colors = get_kits_colors(players_imgs, grass_hsv, annotated_frame)

            # Run on the first frame only
            if current_frame_idx == 1:
                kits_clf = get_kits_classifier(kits_colors)
                left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
                grass_color = get_grass_color(result.orig_img)
                grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

            for box in result.boxes:
                label = int(box.cls.numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())

                # If the box contains a player, find to which team he belongs
                if label == 0:
                    kit_color = get_kits_colors([result.orig_img[y1: y2, x1: x2]], grass_hsv)
                    team = classify_kits(kits_clf, kit_color)
                    if team == left_team_label:
                        label = 0
                    else:
                        label = 1

                # If the box contains a Goalkeeper, find to which team he belongs
                elif label == 1:
                    if x1 < 0.5 * width:
                        label = 2
                    else:
                        label = 3

                # Increase the label by 2 because of the two add labels "Player-L", "GK-L"
                else:
                    label = label + 2

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[str(label)], 2)
                cv2.putText(annotated_frame, labels[label], (x1 - 30, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            box_colors[str(label)], 2)

            # Write the annotated frame
            output_video.write(annotated_frame)

        else:
            # Break the loop if the end of the video is reached
            break

    cv2.destroyAllWindows()
    output_video.release()
    cap.release()

def process_video_with_events(video_path, events_json_path, output_json_path):
    """
    Main function to integrate event detection with actor identification
    """
    # Load the events JSON
    with open(events_json_path, 'r') as f:
        events_json = json.load(f)
    
    # Initialize model
    model = YOLO("./weights/last.pt")
    
    # Process events to identify actors
    enhanced_events = process_events_with_actors(video_path, events_json, model)
    
    # Save enhanced events
    with open(output_json_path, 'w') as f:
        json.dump(enhanced_events, f, indent=4)
    
    print(f"Enhanced events saved to {output_json_path}")

# Main execution
if __name__ == "__main__":
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
    
    # Check if we're running event processing or just object detection
    if len(sys.argv) == 4:
        # Running event processing
        video_path = sys.argv[1]
        events_json_path = sys.argv[2]
        output_json_path = sys.argv[3]
        process_video_with_events(video_path, events_json_path, output_json_path)
    elif len(sys.argv) == 2:
        # Running original object detection
        model = YOLO("./weights/last.pt")
        video_path = sys.argv[1]
        annotate_video(video_path, model)
    else:
        print("Usage for object detection: python script.py video_path")
        print("Usage for event actor identification: python script.py video_path events_json_path output_json_path")
