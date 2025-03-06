# import cv2
# import numpy as np
# import sys
# from ultralytics import YOLO
# from sklearn.cluster import KMeans
# import supervision as sv


# def get_grass_color(img):
#     """
#     Finds the color of the grass in the background of the image

#     Args:
#         img: np.array object of shape (WxHx3) that represents the BGR value of the
#         frame pixels .

#     Returns:
#         grass_color
#             Tuple of the BGR value of the grass color in the image
#     """
#     # Convert image to HSV color space
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # Define range of green color in HSV
#     lower_green = np.array([30, 40, 40])
#     upper_green = np.array([80, 255, 255])

#     # Threshold the HSV image to get only green colors
#     mask = cv2.inRange(hsv, lower_green, upper_green)

#     # Calculate the mean value of the pixels that are not masked
#     masked_img = cv2.bitwise_and(img, img, mask=mask)
#     grass_color = cv2.mean(img, mask=mask)
#     return grass_color[:3]

# def get_players_boxes(result):
#   """
#   Finds the images of the players in the frame and their bounding boxes.

#   Args:
#       result: ultralytics.engine.results.Results object that contains all the
#       result of running the object detection algroithm on the frame

#   Returns:
#       players_imgs
#           List of np.array objects that contain the BGR values of the cropped
#           parts of the image that contains players.
#       players_boxes
#           List of ultralytics.engine.results.Boxes objects that contain various
#           information about the bounding boxes of the players found in the image.
#   """
#   players_imgs = []
#   players_boxes = []
#   for box in result.boxes:
#     label = int(box.cls.numpy()[0])
#     if label == 0:
#       x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
#       player_img = result.orig_img[y1: y2, x1: x2]
#       players_imgs.append(player_img)
#       players_boxes.append(box)
#   return players_imgs, players_boxes

# def get_kits_colors(players, grass_hsv=None, frame=None):
#   """
#   Finds the kit colors of all the players in the current frame

#   Args:
#       players: List of np.array objects that contain the BGR values of the image
#       portions that contain players.
#       grass_hsv: tuple that contain the HSV color value of the grass color of
#       the image background.

#   Returns:
#       kits_colors
#           List of np arrays that contain the BGR values of the kits color of all
#           the players in the current frame
#   """
#   kits_colors = []
#   if grass_hsv is None:
# 	  grass_color = get_grass_color(frame)
# 	  grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

#   for player_img in players:
#       # Convert image to HSV color space
#       hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

#       # Define range of green color in HSV
#       lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
#       upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])

#       # Threshold the HSV image to get only green colors
#       mask = cv2.inRange(hsv, lower_green, upper_green)

#       # Bitwise-AND mask and original image
#       mask = cv2.bitwise_not(mask)
#       upper_mask = np.zeros(player_img.shape[:2], np.uint8)
#       upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
#       mask = cv2.bitwise_and(mask, upper_mask)

#       kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])

#       kits_colors.append(kit_color)
#   return kits_colors

# def get_kits_classifier(kits_colors):
#   """
#   Creates a K-Means classifier that can classify the kits accroding to their BGR
#   values into 2 different clusters each of them represents one of the teams

#   Args:
#       kits_colors: List of np.array objects that contain the BGR values of
#       the colors of the kits of the players found in the current frame.

#   Returns:
#       kits_kmeans
#           sklearn.cluster.KMeans object that can classify the players kits into
#           2 teams according to their color..
#   """
#   kits_kmeans = KMeans(n_clusters=2)
#   kits_kmeans.fit(kits_colors);
#   return kits_kmeans

# def classify_kits(kits_classifer, kits_colors):
#   """
#   Classifies the player into one of the two teams according to the player's kit
#   color

#   Args:
#       kits_classifer: sklearn.cluster.KMeans object that can classify the
#       players kits into 2 teams according to their color.
#       kits_colors: List of np.array objects that contain the BGR values of
#       the colors of the kits of the players found in the current frame.

#   Returns:
#       team
#           np.array object containing a single integer that carries the player's
#           team number (0 or 1)
#   """
#   team = kits_classifer.predict(kits_colors)
#   return team

# def get_left_team_label(players_boxes, kits_colors, kits_clf):
#   """
#   Finds the label of the team that is on the left of the screen

#   Args:
#       players_boxes: List of ultralytics.engine.results.Boxes objects that
#       contain various information about the bounding boxes of the players found
#       in the image.
#       kits_colors: List of np.array objects that contain the BGR values of
#       the colors of the kits of the players found in the current frame.
#       kits_clf: sklearn.cluster.KMeans object that can classify the players kits
#       into 2 teams according to their color.
#   Returns:
#       left_team_label
#           Int that holds the number of the team that's on the left of the image
#           either (0 or 1)
#   """
#   left_team_label = 0
#   team_0 = []
#   team_1 = []

#   for i in range(len(players_boxes)):
#     x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())

#     team = classify_kits(kits_clf, [kits_colors[i]]).item()
#     if team==0:
#       team_0.append(np.array([x1]))
#     else:
#       team_1.append(np.array([x1]))

#   team_0 = np.array(team_0)
#   team_1 = np.array(team_1)

#   if np.average(team_0) - np.average(team_1) > 0:
#     left_team_label = 1

#   return left_team_label

# def annotate_video(video_path, model):
#     cap = cv2.VideoCapture(video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Initialize ByteTrack
#     tracker = sv.ByteTrack(
#         track_thresh=0.25,
#         track_buffer=30,
#         match_thresh=0.8,
#         frame_rate=30
#     )

#     # Initialize annotators
#     box_annotator = sv.BoxAnnotator()
#     label_annotator = sv.LabelAnnotator()

#     # Video writer setup
#     video_name = video_path.split('/')[-1]
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video = cv2.VideoWriter(f'./output/{video_name.split(".")[0]}_out.mp4',
#                                   fourcc, 30, (width, height))

#     # Team classification initialization
#     kits_clf = None
#     left_team_label = 0
#     grass_hsv = None

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break

#         # Run YOLOv8 inference
#         result = model(frame, conf=0.5, verbose=False)[0]
        
#         # Convert YOLO results to supervision format
#         detections = sv.Detections.from_ultralytics(result)
        
#         # Filter only players and goalkeepers (classes 0 and 1)
#         mask = np.isin(detections.class_id, [0, 1])
#         detections = detections[mask]

#         # Update tracker
#         detections = tracker.update_with_detections(detections)

#         # First frame initialization
#         if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:
#             grass_color = get_grass_color(frame)
#             grass_hsv = cv2.cvtColor(np.uint8([[grass_color]]), cv2.COLOR_BGR2HSV)[0][0]
            
#             # Get initial kits colors for classification
#             first_frame_kits = []
#             for xyxy, _, _, _, _ in detections:
#                 x1, y1, x2, y2 = map(int, xyxy)
#                 player_img = frame[y1:y2, x1:x2]
#                 kit_color = get_kits_colors([player_img], grass_hsv, frame)
#                 if kit_color:
#                     first_frame_kits.append(kit_color[0])
            
#             if len(first_frame_kits) >= 2:
#                 kits_clf = get_kits_classifier(first_frame_kits)
#                 left_team_label = get_left_team_label_from_detections(detections, first_frame_kits, kits_clf)

#         # Prepare annotations
#         labels = []
#         for i, (xyxy, _, _, _, track_id) in enumerate(detections):
#             x1, y1, x2, y2 = map(int, xyxy)
#             player_img = frame[y1:y2, x1:x2]
            
#             # Classify team
#             if detections.class_id[i] == 0:  # Player
#                 kit_color = get_kits_colors([player_img], grass_hsv, frame)
#                 if kits_clf and kit_color:
#                     team = classify_kits(kits_clf, kit_color)
#                     team_label = "L" if team == left_team_label else "R"
#                 else:
#                     team_label = "L"
#                 labels.append(f"Player-{team_label} {track_id}")
#             else:  # Goalkeeper
#                 team_label = "L" if x1 < width//2 else "R"
#                 labels.append(f"GK-{team_label} {track_id}")

#         # Annotate frame
#         annotated_frame = box_annotator.annotate(
#             scene=frame.copy(),
#             detections=detections,
#             labels=labels
#         )

#         output_video.write(annotated_frame)

#     # Cleanup
#     cap.release()
#     output_video.release()
#     cv2.destroyAllWindows()

# # Helper function for initial team classification
# def get_left_team_label_from_detections(detections, kits_colors, kits_clf):
#     team_positions = {0: [], 1: []}
#     for i, (xyxy, *_ in enumerate(detections):
#         x1 = xyxy[0]
#         team = classify_kits(kits_clf, [kits_colors[i]]).item()
#         team_positions[team].append(x1)
    
#     avg_0 = np.mean(team_positions[0]) if team_positions[0] else float('inf')
#     avg_1 = np.mean(team_positions[1]) if team_positions[1] else float('inf')
#     return 0 if avg_0 < avg_1 else 1


# if __name__ == "__main__":

#     labels = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]
#     box_colors = {
#         "0": (150, 50, 50),
#         "1": (37, 47, 150),
#         "2": (41, 248, 165),
#         "3": (166, 196, 10),
#         "4": (155, 62, 157),
#         "5": (123, 174, 213),
#         "6": (217, 89, 204),
#         "7": (22, 11, 15)
#     }
#     model = YOLO("./weights/last.pt")
#     video_path = sys.argv[1]
#     annotate_video(video_path, model)
import cv2
import numpy as np
import sys
from ultralytics import YOLO
from sklearn.cluster import KMeans
import supervision as sv

# Helper functions
def get_grass_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    grass_color = cv2.mean(img, mask=mask)[:3]
    return grass_color

def get_kits_colors(players, grass_hsv=None, frame=None):
    kits_colors = []
    if grass_hsv is None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[grass_color]]), cv2.COLOR_BGR2HSV)[0][0]
    
    for player_img in players:
        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([grass_hsv[0] - 10, 40, 40])
        upper_green = np.array([grass_hsv[0] + 10, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        upper_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
        mask = cv2.bitwise_and(mask, upper_mask)
        kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])
        kits_colors.append(kit_color)
    return kits_colors

def get_kits_classifier(kits_colors):
    kits_kmeans = KMeans(n_clusters=2)
    kits_kmeans.fit(kits_colors)
    return kits_kmeans

def classify_kits(kits_classifer, kits_colors):
    return kits_classifer.predict(kits_colors)

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    team_0_x = []
    team_1_x = []
    
    for i, box in enumerate(players_boxes):
        x1 = int(box.xyxy[0].numpy()[0])
        team = classify_kits(kits_clf, [kits_colors[i]])[0]
        (team_0_x if team == 0 else team_1_x).append(x1)
    
    avg_0 = np.mean(team_0_x) if team_0_x else float('inf')
    avg_1 = np.mean(team_1_x) if team_1_x else float('inf')
    return 0 if avg_0 < avg_1 else 1

# Main annotation function
def annotate_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize tracker and annotators
    tracker = sv.ByteTrack(
        track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30
    )
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Video writer setup
    video_name = video_path.split('/')[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(
        f'./output/{video_name.split(".")[0]}_out.mp4',
        fourcc, 30, (width, height)
    )

    # Initialization variables
    kits_clf = None
    left_team_label = 0
    grass_hsv = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference
        result = model(frame, conf=0.5, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter only players (0) and goalkeepers (1)
        player_mask = np.isin(detections.class_id, [0, 1])
        detections = detections[player_mask]
        
        # Update tracker
        detections = tracker.update_with_detections(detections)

        # First frame initialization
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:
            grass_color = get_grass_color(frame)
            grass_hsv = cv2.cvtColor(np.uint8([[grass_color]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Get initial players for team classification
            players = [
                frame[int(y1):int(y2), int(x1):int(x2)]
                for [x1, y1, x2, y2], *_ in detections
            ]
            kits_colors = get_kits_colors(players, grass_hsv, frame)
            
            if len(kits_colors) >= 2:
                kits_clf = get_kits_classifier(kits_colors)
                left_team_label = get_left_team_label(
                    [box for box in result.boxes if int(box.cls) in [0, 1]],
                    kits_colors,
                    kits_clf
                )

        # Prepare labels with tracking IDs and team info
        labels = []
        for i, ([x1, y1, x2, y2], _, _, _, track_id) in enumerate(detections):
            if detections.class_id[i] == 0:  # Player
                player_img = frame[int(y1):int(y2), int(x1):int(x2)]
                kit_color = get_kits_colors([player_img], grass_hsv, frame)
                team = classify_kits(kits_clf, kit_color)[0] if kits_clf else 0
                team_label = "L" if team == left_team_label else "R"
                labels.append(f"Player-{team_label} {track_id}")
            else:  # Goalkeeper
                team_label = "L" if x1 < width//2 else "R"
                labels.append(f"GK-{team_label} {track_id}")

        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        output_video.write(annotated_frame)

    # Cleanup
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    labels = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]
    box_colors = {
        "0": (150, 50, 50),    # Player-L
        "1": (37, 47, 150),    # Player-R
        "2": (41, 248, 165),   # GK-L
        "3": (166, 196, 10),   # GK-R
        "4": (155, 62, 157),   # Ball
        "5": (123, 174, 213),  # Main Ref
        "6": (217, 89, 204),   # Side Ref
        "7": (22, 11, 15)      # Staff
    }
    
    model = YOLO("./weights/last.pt")
    video_path = sys.argv[1]
    annotate_video(video_path, model)
