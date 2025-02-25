import cv2
import numpy as np
import sys
from ultralytics import YOLO
from sklearn.cluster import KMeans
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import time

# DeepSORT imports
from deep_sort_realtime.deepsort_tracker import DeepSort

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
  """
  Finds the images of the players in the frame and their bounding boxes.

  Args:
      result: ultralytics.engine.results.Results object that contains all the
      result of running the object detection algroithm on the frame

  Returns:
      players_imgs
          List of np.array objects that contain the BGR values of the cropped
          parts of the image that contains players.
      players_boxes
          List of ultralytics.engine.results.Boxes objects that contain various
          information about the bounding boxes of the players found in the image.
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
  """
  Creates a K-Means classifier that can classify the kits accroding to their BGR
  values into 2 different clusters each of them represents one of the teams

  Args:
      kits_colors: List of np.array objects that contain the BGR values of
      the colors of the kits of the players found in the current frame.

  Returns:
      kits_kmeans
          sklearn.cluster.KMeans object that can classify the players kits into
          2 teams according to their color..
  """
  kits_kmeans = KMeans(n_clusters=2)
  kits_kmeans.fit(kits_colors);
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

def annotate_video(video_path, model):
    """
    Loads the input video and runs the object detection algorithm on its frames, finally it annotates the frame with
    the appropriate labels and tracks players using DeepSORT

    Args:
        video_path: String the holds the path of the input video
        model: Object that represents the trained object detection model
    Returns:
    """
    # Initialize DeepSORT trackers - one for each team and referees
    
    
    # Create separate trackers for each team and others
    deepsort_left = DeepSort(
        max_age=30,        # Maximum frames to keep track of players after they disappear
        n_init=3,          # Number of frames to confirm a new track
        max_iou_distance=0.7,  # IOU threshold
        max_cosine_distance=0.2,  # Appearance feature threshold
        nn_budget=100      # Maximum appearance features to store
    )
    
    deepsort_right = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100
    )
    
    deepsort_others = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100
    )
    
    cap = cv2.VideoCapture(video_path)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_name = video_path.split('/')[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('./output/'+video_name.split('.')[0] + "_tracked.mp4",
                                   fourcc,
                                   30.0,
                                   (width, height))

    kits_clf = None
    left_team_label = 0
    grass_hsv = None

    # Tracking variables
    track_history_left = {}   # Store track history for left team
    track_history_right = {}  # Store track history for right team
    track_history_others = {} # Store track history for referees and staff
    
    frame_count = 0
    
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if success:
            frame_count += 1
            
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
            
            # Separate detections for each team and others
            left_team_dets = []
            right_team_dets = []
            other_dets = []
            
            # List to store class labels for each detection
            left_team_classes = []
            right_team_classes = []
            other_classes = []
            
            for box in result.boxes:
                label = int(box.cls.numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                conf = float(box.conf.numpy()[0])
                
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Skip if the detection is too small
                if bbox_width < 10 or bbox_height < 10:
                    continue
                
                # Format for DeepSORT: [x1, y1, x2, y2, confidence]
                bbox = [x1, y1, x2, y2, conf]
                
                # If the box contains a player, find to which team he belongs
                if label == 0:
                    kit_color = get_kits_colors([result.orig_img[y1: y2, x1: x2]], grass_hsv)
                    team = classify_kits(kits_clf, kit_color)
                    
                    if team == left_team_label:
                        left_team_dets.append(bbox)
                        left_team_classes.append(0)  # Player-L
                    else:
                        right_team_dets.append(bbox)
                        right_team_classes.append(1)  # Player-R
                
                # If the box contains a Goalkeeper
                elif label == 1:
                    if x1 < 0.5 * width:
                        left_team_dets.append(bbox)
                        left_team_classes.append(2)  # GK-L
                    else:
                        right_team_dets.append(bbox)
                        right_team_classes.append(3)  # GK-R
                
                # Other classes (ball, referees, staff)
                else:
                    other_dets.append(bbox)
                    other_classes.append(label + 2)  # Ball, Main Ref, Side Ref, Staff
            
            # Update DeepSORT trackers with detections
            if left_team_dets:
                left_team_dets = np.array(left_team_dets)
                formatted_dets = []
                for i, det in enumerate(left_team_dets):
                    x1, y1, x2, y2, conf = det
                    w, h = x2-x1, y2-y1
                    formatted_dets.append(([x1, y1, w, h], conf, str(left_team_classes[i])))
                
                track_left = deepsort_left.update_tracks(formatted_dets, frame=annotated_frame)
 
            else:
                track_left = []
                
            if right_team_dets:
                right_team_dets = np.array(right_team_dets)
                formatted_dets_right = []
                for i, det in enumerate(right_team_dets):
                    x1, y1, x2, y2, conf = det
                    w, h = x2-x1, y2-y1
                    formatted_dets_right.append(([x1, y1, w, h], conf, str(right_team_classes[i])))
                
                track_right = deepsort_right.update_tracks(formatted_dets_right, frame=annotated_frame)
            else:
                track_right = []
                
            if other_dets:
                other_dets = np.array(other_dets)
                formatted_dets_others = []
                for i, det in enumerate(other_dets):
                    x1, y1, x2, y2, conf = det
                    w, h = x2-x1, y2-y1
                    formatted_dets_others.append(([x1, y1, w, h], conf, str(other_classes[i])))
                
                track_others = deepsort_others.update_tracks(formatted_dets_others, frame=annotated_frame)
            else:
                track_others = []
            
            # Draw tracking results for left team (format: [x1, y1, x2, y2, track_id, class_id])
            for track in track_left:
                x1, y1, x2, y2, track_id, class_id = track
                
                # Convert to integers
                x1, y1, x2, y2, track_id, class_id = int(x1), int(y1), int(x2), int(y2), int(track_id), int(class_id)
                
                # Get label and color
                label_idx = class_id  # 0 for Player-L, 2 for GK-L
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[str(label_idx)], 2)
                
                # Draw ID label
                text = f"{labels[label_idx]}-{track_id}"
                cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                            box_colors[str(label_idx)], 2)
                
                # Update track history
                if track_id not in track_history_left:
                    track_history_left[track_id] = []
                
                # Calculate center position
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                track_history_left[track_id].append(center)
                
                # Draw tracking line (last 30 frames)
                if len(track_history_left[track_id]) > 1:
                    for i in range(1, min(30, len(track_history_left[track_id]))):
                        if i == 1:  # Draw thicker line for the most recent movement
                            thickness = 2
                        else:
                            thickness = max(1, 3 - i // 10)  # Thinner lines for older positions
                        
                        cv2.line(annotated_frame, 
                                track_history_left[track_id][-i], 
                                track_history_left[track_id][-i-1], 
                                box_colors[str(label_idx)], 
                                thickness)
            
            # Draw tracking results for right team
            for track in track_right:
                x1, y1, x2, y2, track_id, class_id = track
                
                # Convert to integers
                x1, y1, x2, y2, track_id, class_id = int(x1), int(y1), int(x2), int(y2), int(track_id), int(class_id)
                
                # Get label and color
                label_idx = class_id  # 1 for Player-R, 3 for GK-R
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[str(label_idx)], 2)
                
                # Draw ID label
                text = f"{labels[label_idx]}-{track_id}"
                cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                            box_colors[str(label_idx)], 2)
                
                # Update track history
                if track_id not in track_history_right:
                    track_history_right[track_id] = []
                
                # Calculate center position
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                track_history_right[track_id].append(center)
                
                # Draw tracking line (last 30 frames)
                if len(track_history_right[track_id]) > 1:
                    for i in range(1, min(30, len(track_history_right[track_id]))):
                        if i == 1:  # Draw thicker line for the most recent movement
                            thickness = 2
                        else:
                            thickness = max(1, 3 - i // 10)
                        
                        cv2.line(annotated_frame, 
                                track_history_right[track_id][-i], 
                                track_history_right[track_id][-i-1], 
                                box_colors[str(label_idx)], 
                                thickness)
            
            # Draw tracking results for others (ball, referees, staff)
            for track in track_others:
                x1, y1, x2, y2, track_id, class_id = track
                
                # Convert to integers
                x1, y1, x2, y2, track_id, class_id = int(x1), int(y1), int(x2), int(y2), int(track_id), int(class_id)
                
                # Get label and color
                label_idx = class_id  # 4 for Ball, 5 for Main Ref, etc.
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[str(label_idx)], 2)
                
                # Draw ID label (not for the ball)
                if label_idx != 4:  # Not a ball
                    text = f"{labels[label_idx]}-{track_id}"
                    cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                box_colors[str(label_idx)], 2)
                else:
                    cv2.putText(annotated_frame, labels[label_idx], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                box_colors[str(label_idx)], 2)
                
                # Update track history (not for the ball)
                if label_idx != 4:  # Not a ball
                    if track_id not in track_history_others:
                        track_history_others[track_id] = []
                    
                    # Calculate center position
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    track_history_others[track_id].append(center)
                    
                    # Draw tracking line (last 30 frames)
                    if len(track_history_others[track_id]) > 1:
                        for i in range(1, min(30, len(track_history_others[track_id]))):
                            if i == 1:
                                thickness = 2
                            else:
                                thickness = max(1, 3 - i // 10)
                            
                            cv2.line(annotated_frame, 
                                    track_history_others[track_id][-i], 
                                    track_history_others[track_id][-i-1], 
                                    box_colors[str(label_idx)], 
                                    thickness)
            
            # Add frame counter
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write the annotated frame
            output_video.write(annotated_frame)

        else:
            # Break the loop if the end of the video is reached
            break

    cv2.destroyAllWindows()
    output_video.release()
    cap.release()

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
    model = YOLO("/kaggle/working/Football-Object-Detection/weights/last.pt")
    video_path = sys.argv[1]
    annotate_video(video_path, model)
