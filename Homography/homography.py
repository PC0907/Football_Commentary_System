import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict

# Keypoint labels in order (index 0 to 45)
KEYPOINT_NAMES = [
    "Big rect. left bottom LEFT", "Big rect. left bottom RIGHT", "Big rect. left main DOWN", "Big rect. left main UP",
    "Big rect. left top LEFT", "Big rect. left top RIGHT", "Big rect. right bottom LEFT", "Big rect. right bottom RIGHT",
    "Big rect. right main DOWN", "Big rect. right main UP", "Big rect. right top LEFT", "Big rect. right top RIGHT",
    "Goal left crossbar LEFT", "Goal left crossbar RIGHT", "Goal left post left DOWN", "Goal left post left UP",
    "Goal left post right DOWN", "Goal left post right UP", "Goal right crossbar LEFT", "Goal right crossbar RIGHT",
    "Goal right post left DOWN", "Goal right post left UP", "Goal right post right DOWN", "Goal right post right UP",
    "Middle line UP", "Middle line DOWN", "Side line bottom LEFT", "Side line bottom RIGHT",
    "Side line left DOWN", "Side line left UP", "Side line right DOWN", "Side line right UP",
    "Side line top LEFT", "Side line top RIGHT", "Small rect. left bottom LEFT", "Small rect. left bottom RIGHT",
    "Small rect. left main DOWN", "Small rect. left main UP", "Small rect. left top LEFT", "Small rect. left top RIGHT",
    "Small rect. right bottom LEFT", "Small rect. right bottom RIGHT", "Small rect. right main DOWN", "Small rect. right main UP",
    "Small rect. right top LEFT", "Small rect. right top RIGHT"
]

# Hardcoded path for the model
MODEL_PATH = "./best.pt"  # Change this as per the actual location

# Field Points and their corresponding line pairs
FIELD_POINT_TO_KEYPOINT_LINES = {
    "left_outer_box_far_right": [
        ("Big rect. left top LEFT", "Big rect. left top RIGHT"),  # Line 1
        ("Big rect. left main DOWN", "Big rect. left main UP")         # Line 2
    ],
    "left_outer_box_far_left": [
        ("Big rect. left top LEFT", "Small rect. left top LEFT"),  # Line 1
        ("Big rect. left top LEFT", "Big rect. left top RIGHT")            # Line 2
    ],
    "left_outer_box_near_right": [
        ("Big rect. left bottom LEFT", "Big rect. left bottom RIGHT"),  # Line 1
        ("Big rect. left main DOWN", "Big rect. left main UP")         # Line 2
    ],
    "left_outer_box_near_left": [
        ("Big rect. left top LEFT", "Small rect. left top LEFT"),  # Line 1
        ("Big rect. left bottom LEFT", "Big rect. left bottom RIGHT")            # Line 2
    ],
    "left_inner_box_far_right": [
        ("Small rect. left top LEFT", "Small rect. left top RIGHT"),  # Line 1
        ("Small rect. left main DOWN", "Small rect. left main UP")         # Line 2
    ],
    "left_inner_box_far_left": [
        ("Big rect. left top LEFT", "Small rect. left top LEFT"),  # Line 1
        ("Small rect. left top LEFT", "Small rect. left top RIGHT")            # Line 2
    ],
    "left_inner_box_near_right": [
        ("Small rect. left bottom LEFT", "Small rect. left bottom RIGHT"),  # Line 1
        ("Small rect. left main DOWN", "Small rect. left main UP")         # Line 2
    ],
    "left_inner_box_near_left": [
        ("Big rect. left top LEFT", "Small rect. left top LEFT"),  # Line 1
        ("Small rect. left bottom LEFT", "Small rect. left bottom RIGHT")            # Line 2
    ],
    #
    "right_outer_box_far_left": [
        ("Big rect. right top LEFT", "Big rect. right top RIGHT"),  # Line 1
        ("Big rect. right main DOWN", "Big rect. right main UP")         # Line 2
    ],
    "right_outer_box_far_right": [
        ("Big rect. right top RIGHT", "Small rect. right top RIGHT"),  # Line 1
        ("Big rect. right top LEFT", "Big rect. right top RIGHT")            # Line 2
    ],
    "right_outer_box_near_left": [
        ("Big rect. right bottom LEFT", "Big rect. right bottom RIGHT"),  # Line 1
        ("Big rect. right main DOWN", "Big rect. right main UP")         # Line 2
    ],
    "right_outer_box_near_right": [
        ("Big rect. right top RIGHT", "Small rect. right top RIGHT"),  # Line 1
        ("Big rect. right bottom LEFT", "Big rect. right bottom RIGHT")           # Line 2
    ],
    "right_inner_box_far_left": [
        ("Small rect. right top LEFT", "Small rect. right top RIGHT"),  # Line 1
        ("Small rect. right main DOWN", "Small rect. right main UP")         # Line 2
    ],
    "right_inner_box_far_right": [
        ("Big rect. right top RIGHT", "Small rect. right top RIGHT"),  # Line 1
        ("Small rect. right top LEFT", "Small rect. right top RIGHT")            # Line 2
    ],
    "right_inner_box_near_left": [
        ("Small rect. right bottom LEFT", "Small rect. right bottom RIGHT"),  # Line 1
        ("Small rect. right main DOWN", "Small rect. right main UP")         # Line 2
    ],
    "right_inner_box_near_right": [
        ("Big rect. right top RIGHT", "Small rect. right top RIGHT"),  # Line 1
        ("Small rect. right bottom LEFT", "Small rect. right bottom RIGHT")           # Line 2
    ],
    "far_left_corner": [
        ("Side line top LEFT", "Side line top RIGHT"),  # Line 1
        ("Big rect. left top LEFT", "Small rect. left top LEFT"),  # Line 2
    ],
    "far_right_corner": [
        ("Side line top LEFT", "Side line top RIGHT"),  # Line 1
        ("Big rect. right top RIGHT", "Small rect. right top RIGHT"),  # Line 2
    ],
    "near_left_corner": [
        ("Side line bottom LEFT", "Side line bottom RIGHT"),  # Line 1
        ("Big rect. left top LEFT", "Small rect. left top LEFT"),  # Line 2
    ],
    "near_right_corner": [
        ("Side line bottom LEFT", "Side line bottom RIGHT"),  # Line 1
        ("Big rect. right top RIGHT", "Small rect. right top RIGHT"),  # Line 2
    ],
    "left_goal_far_post": [
        ("Goal left post right DOWN", "Goal left post right UP"),  # Line 1
        ("Big rect. left top LEFT", "Small rect. left top LEFT"),  # Line 2
    ],  # Line 2
    "left_goal_near_post": [
        ("Goal left post left DOWN", "Goal left post left UP"),  # Line 1
        ("Big rect. left top LEFT", "Small rect. left top LEFT"),  # Line 2
    ],
    "right_goal_far_post": [
        ("Goal right post left DOWN", "Goal right post left UP"),  # Line 1
        ("Big rect. right top RIGHT", "Small rect. right top RIGHT"),  # Line 2
    ],  # Line 2
    "right_goal_near_post": [
        ("Goal right post right DOWN", "Goal right post right UP"),  # Line 1
        ("Big rect. right top RIGHT", "Small rect. right top RIGHT"),  # Line 2
    ],
    "near_center_line_end": [
        ("Middle line UP", "Middle line DOWN"),  # Line 1
        ("Side line top LEFT"," Side line top RIGHT"),  # Line 2
    ],
    "far_center_line_end": [
        ("Middle line UP", "Middle line DOWN"),  # Line 1
        ("Side line bottom LEFT"," Side line bottom RIGHT"),  # Line 2
    ]    # Add other field points similarly
}

# Coordinates for keypoints in the field (in meters, origin at top-left)
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

# Function to calculate intersection of two lines given by points (x1, y1), (x2, y2)
def line_intersection(line1, line2):
    x1, y1 = line1[0], line1[1]
    x2, y2 = line1[2], line1[3]
    x3, y3 = line2[0], line2[1]
    x4, y4 = line2[2], line2[3]
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Lines are parallel
    
    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    
    return intersect_x, intersect_y

# Get the image points from the field point names using keypoint line mappings
def get_image_points(field_points):
    image_points = []
    for field_point, lines in field_points.items():
        line1_keypoints = lines[0]
        line2_keypoints = lines[1]
        
        # Get the keypoint indices for each line's points
        idx1 = KEYPOINT_NAMES.index(line1_keypoints[0])
        idx2 = KEYPOINT_NAMES.index(line1_keypoints[1])
        idx3 = KEYPOINT_NAMES.index(line2_keypoints[0])
        idx4 = KEYPOINT_NAMES.index(line2_keypoints[1])
        
        # Get the pixel coordinates for each keypoint
        kpt1 = keypoints[idx1]
        kpt2 = keypoints[idx2]
        kpt3 = keypoints[idx3]
        kpt4 = keypoints[idx4]
        
        # Calculate intersection of the lines
        img_point = line_intersection((kpt1[0], kpt1[1], kpt2[0], kpt2[1]),
                                      (kpt3[0], kpt3[1], kpt4[0], kpt4[1]))
        if img_point:
            image_points.append(img_point)
    
    return image_points

# Main function to run inference and calculate homography
def run_inference():
    model = YOLO(MODEL_PATH)
    img = cv2.imread("input_image.jpg")  # Change this path
    
    # Detect keypoints with YOLO
    results = model(img, verbose=False)[0]
    keypoints = results.keypoints.data.cpu().numpy().reshape(-1, 3)  # Get keypoints
    
    # Extract image points using intersection logic
    image_points = get_image_points(FIELD_POINT_TO_KEYPOINT_LINES)
    
    # Corresponding field coordinates for homography calculation
    world_points = [pt[1:3] for pt in KEYPOINTS_DATA]  # Extract the field coordinates (x, y)
    
    # Compute homography matrix
    homography, _ = cv2.findHomography(np.array(image_points), np.array(world_points))
    
    # Transform player and ball coordinates using homography
    transformed_coords = []
    for player_bbox in player_bboxes:  # Assuming player_bboxes are extracted
        player_center = np.array([player_bbox[0], player_bbox[1], 1]).reshape(1, 3).T
        transformed_point = np.dot(homography, player_center)
        transformed_point /= transformed_point[2]  # Homogeneous coordinates normalization
        transformed_coords.append(transformed_point[:2])
    
    return transformed_coords

if __name__ == "__main__":
    transformed_player_coords = run_inference()
    print(transformed_player_coords)