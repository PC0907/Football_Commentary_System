import cv2
import numpy as np
import sys
import time
from ultralytics import YOLO
from sklearn.cluster import KMeans

def get_grass_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
    players_imgs = []
    players_boxes = []
    for box in result.boxes:
        label = int(box.cls.numpy()[0])
        if label == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            player_img = result.orig_img[y1:y2, x1:x2]
            players_imgs.append(player_img)
            players_boxes.append(box)
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
        upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
        mask = cv2.bitwise_and(mask, upper_mask)
        kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])
        kits_colors.append(kit_color)
    return kits_colors

def get_kits_classifier(kits_colors):
    kits_kmeans = KMeans(n_clusters=2)
    kits_kmeans.fit(kits_colors)
    return kits_kmeans

def classify_kits(kits_classifier, kits_colors):
    team = kits_classifier.predict(kits_colors)
    return team

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    left_team_label = 0
    team_0 = []
    team_1 = []

    for i in range(len(players_boxes)):
        x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())
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
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_name = video_path.split('/')[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('./output/'+video_name.split('.')[0] + "_out.mp4",
                                   fourcc,
                                   fps,
                                   (width, height))

    kits_clf = None
    left_team_label = 0
    grass_hsv = None
    frame_count = 0

    start_time = time.time()  # Start timer

    while cap.isOpened():
        success, frame = cap.read()
        current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if success:
            annotated_frame = cv2.resize(frame, (width, height))
            result = model(annotated_frame, conf=0.5, verbose=False)[0]

            players_imgs, players_boxes = get_players_boxes(result)
            kits_colors = get_kits_colors(players_imgs, grass_hsv, annotated_frame)

            if current_frame_idx == 1:
                kits_clf = get_kits_classifier(kits_colors)
                left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
                grass_color = get_grass_color(result.orig_img)
                grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

            for box in result.boxes:
                label = int(box.cls.numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())

                if label == 0:
                    kit_color = get_kits_colors([result.orig_img[y1:y2, x1:x2]], grass_hsv)
                    team = classify_kits(kits_clf, kit_color)
                    if team == left_team_label:
                        label = 0
                    else:
                        label = 1
                elif label == 1:
                    if x1 < 0.5 * width:
                        label = 2
                    else:
                        label = 3
                else:
                    label = label + 2

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[str(label)], 2)
                cv2.putText(annotated_frame, labels[label], (x1 - 30, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            box_colors[str(label)], 2)

            output_video.write(annotated_frame)

            # Update progress
            frame_count += 1
            elapsed_time = time.time() - start_time
            remaining_time = (total_frames - frame_count) * (elapsed_time / frame_count)
            percent_complete = (frame_count / total_frames) * 100

            print(f"\rProcessing: {percent_complete:.2f}% completed, Estimated time remaining: {remaining_time:.2f} seconds", end='')

        else:
            break

    print()  # New line after processing completion
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
    model = YOLO("./weights/last.pt")
    video_path = sys.argv[1]

    start_time = time.time()
    annotate_video(video_path, model)
    end_time = time.time()

    processing_time = end_time - start_time
    print(f"\nTotal processing time for the video: {processing_time:.2f} seconds")

