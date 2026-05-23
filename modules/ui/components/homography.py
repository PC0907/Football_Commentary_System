import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional


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
CONFIDENCE_THRESHOLD = 0.8

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
print(f"Debug: Loaded FIELD_POINT_TO_KEYPOINT_LINES with {len(FIELD_POINT_TO_KEYPOINT_LINES)} field points")
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

# Cache the YOLO model as a global variable to avoid reloading
_model = None

def get_model():
    """
    Returns a cached instance of the YOLO model.
    This avoids reloading the model on each frame, significantly improving performance.
    """
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model

def get_detectable_field_points(detected_keypoints: Set[str]) -> List[str]:
    """
    Identifies which field points can be detected based on available keypoints.
    
    A field point is considered detectable if all its required keypoints are present
    in the detected_keypoints set.
    
    Args:
        detected_keypoints: Set of keypoint names that were successfully detected
        
    Returns:
        List of field point names that can be detected with the available keypoints
    """
    detectable_points = []
    
    for field_point, lines in FIELD_POINT_TO_KEYPOINT_LINES.items():
        keypoints = set()
        for line in lines:
            keypoints.add(line[0])
            keypoints.add(line[1])
        
        if all(kpt in detected_keypoints for kpt in keypoints):
            detectable_points.append(field_point)
    
    return detectable_points

def compute_field_point_coordinates(
    field_points: List[str],
    keypoints: np.ndarray,
    keypoint_names: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Computes the pixel coordinates of field points using line intersections.
    
    For each field point, it finds the intersection of two lines defined by
    pairs of keypoints. The intersection point represents the field point's
    location in the image.
    
    Args:
        field_points: List of field point names to compute coordinates for
        keypoints: Array of keypoint coordinates (x, y, confidence)
        keypoint_names: List of keypoint names corresponding to the keypoints array
        
    Returns:
        Dictionary mapping field point names to their (x, y) pixel coordinates
    """
    field_point_coords = {}
    
    for field_point in field_points:
        lines = FIELD_POINT_TO_KEYPOINT_LINES[field_point]
        
        try:
            line1_start_idx = keypoint_names.index(lines[0][0])
            line1_end_idx = keypoint_names.index(lines[0][1])
            line2_start_idx = keypoint_names.index(lines[1][0])
            line2_end_idx = keypoint_names.index(lines[1][1])
            
            line1_start = keypoints[line1_start_idx]
            line1_end = keypoints[line1_end_idx]
            line2_start = keypoints[line2_start_idx]
            line2_end = keypoints[line2_end_idx]
            
            intersection = line_intersection(
                (line1_start[0], line1_start[1], line1_end[0], line1_end[1]),
                (line2_start[0], line2_start[1], line2_end[0], line2_end[1])
            )
            
            if intersection:
                field_point_coords[field_point] = intersection
                
        except (ValueError, IndexError):
            continue
    
    return field_point_coords

def line_intersection(line1: Tuple[float, float, float, float], 
                     line2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
    """
    Computes the intersection point of two lines.
    
    Each line is defined by two points (x1, y1, x2, y2).
    Returns None if the lines are parallel.
    
    Args:
        line1: First line coordinates (x1, y1, x2, y2)
        line2: Second line coordinates (x1, y1, x2, y2)
        
    Returns:
        (x, y) coordinates of intersection point, or None if lines are parallel
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None
    
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    
    return (x, y)

def calculate_homography_matrix(
    field_point_coords: Dict[str, Tuple[float, float]],
    keypoints_data: List[Tuple[int, float, float, str]]
) -> np.ndarray:
    """
    Computes the homography matrix that maps image coordinates to field coordinates.
    
    Uses RANSAC to robustly estimate the homography matrix from corresponding points
    in the image and real-world coordinates.
    
    Args:
        field_point_coords: Dictionary mapping field points to their pixel coordinates
        keypoints_data: List of tuples containing (id, x, y, field_point_name) for real-world coordinates
        
    Returns:
        3x3 homography matrix
        
    Raises:
        ValueError: If fewer than 4 point correspondences are available
    """
    src_points = []
    dst_points = []
    
    world_coords_map = {name: (x, y) for _, x, y, name in keypoints_data}
    
    for field_point_name in sorted(field_point_coords.keys()):
        if field_point_name in world_coords_map:
            pixel_x, pixel_y = field_point_coords[field_point_name]
            world_x, world_y = world_coords_map[field_point_name]
            
            src_points.append([pixel_x, pixel_y])
            dst_points.append([world_x, world_y])
    
    if len(src_points) < 4:
        raise ValueError(f"Insufficient points for homography calculation. Need at least 4 points, got {len(src_points)}")
    
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    H, _ = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return H

def transform_object_positions(
    object_positions: List[Dict[str, float]],
    homography_matrix: np.ndarray
) -> List[Dict[str, float]]:
    """
    Transforms object positions from image coordinates to field coordinates.
    
    Args:
        object_positions: List of dictionaries containing:
            - object_id: Identifier for the object
            - pixel_x: X coordinate in image
            - pixel_y: Y coordinate in image
        homography_matrix: 3x3 homography matrix
        
    Returns:
        List of dictionaries containing:
            - object_id: Original object identifier
            - world_x_meters: X coordinate on field in meters
            - world_y_meters: Y coordinate on field in meters
    """
    if not object_positions:
        return []
        
    # Vectorized transformation for better performance
    pixel_coords = np.array([[obj['pixel_x'], obj['pixel_y'], 1] for obj in object_positions]).T
    world_coords = np.dot(homography_matrix, pixel_coords)
    world_coords = world_coords / world_coords[2]  # Normalize
    
    # Create transformed positions using list comprehension
    transformed_positions = [
        {
            'object_id': obj['object_id'],
            'world_x_meters': world_coords[0, i],
            'world_y_meters': world_coords[1, i]
        }
        for i, obj in enumerate(object_positions)
    ]
    
    return transformed_positions

def process_frame(
    frame: np.ndarray,
    object_positions: List[Dict[str, float]],
    frame_id: int
) -> Tuple[List[Dict[str, float]], float, float, int]:
    """
    Main processing function that transforms object positions to field coordinates.
    
    This is the main interface to the homography module. It takes a frame, object positions,
    and frame ID, detects field keypoints, computes the homography, and returns the transformed
    object positions in real-world coordinates along with quality metrics.
    
    Args:
        frame: Input frame from the camera
        object_positions: List of dictionaries containing:
            - object_id: Identifier for the object
            - pixel_x: X coordinate in image
            - pixel_y: Y coordinate in image
        frame_id: Unique identifier for the frame
            
    Returns:
        Tuple containing:
            - transformed_positions: List of transformed object positions in field coordinates
            - reprojection_error: Mean error in meters for the homography transformation
            - confidence_score: Score between 0-1 indicating confidence in the transformation
            - frame_id: The input frame ID
            
    Example:
        >>> frame = cv2.imread("football_frame.jpg")
        >>> object_positions = [{"object_id": 1, "pixel_x": 100, "pixel_y": 200}]
        >>> frame_id = 123
        >>> transformed_positions, error, confidence, frame_id = process_frame(frame, object_positions, frame_id)
    """
    # Use cached model for better performance
    model = get_model()
    
    # Run YOLO inference
    results = model(frame, verbose=False)[0]
    keypoints = results.keypoints.data.cpu().numpy().reshape(-1, 3)
    
    # Vectorized confidence filtering for better performance
    high_confidence_mask = keypoints[:, 2] >= CONFIDENCE_THRESHOLD
    high_confidence_keypoints = {KEYPOINT_NAMES[i] for i in np.where(high_confidence_mask)[0]}
    
    # Get detectable field points
    detectable_points = get_detectable_field_points(high_confidence_keypoints)
    
    # Compute field point coordinates
    field_point_coords = compute_field_point_coordinates(
        detectable_points,
        keypoints,
        KEYPOINT_NAMES
    )
    
    # Calculate homography matrix
    homography_matrix = calculate_homography_matrix(field_point_coords, KEYPOINTS_DATA)
    
    # Calculate reprojection error and confidence
    src_points = []
    dst_points = []
    world_coords_map = {name: (x, y) for _, x, y, name in KEYPOINTS_DATA}
    
    # Vectorized point collection for better performance
    for field_point_name in sorted(field_point_coords.keys()):
        if field_point_name in world_coords_map:
            pixel_x, pixel_y = field_point_coords[field_point_name]
            world_x, world_y = world_coords_map[field_point_name]
            src_points.append([pixel_x, pixel_y])
            dst_points.append([world_x, world_y])
    
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    # Vectorized reprojection error calculation
    if len(src_points) > 0:
        src_points_homogeneous = np.column_stack((src_points, np.ones(len(src_points))))
        projected_points = np.dot(homography_matrix, src_points_homogeneous.T).T
        projected_points = projected_points[:, :2] / projected_points[:, 2:]
        reprojection_error = np.mean(np.linalg.norm(projected_points - dst_points, axis=1))
    else:
        reprojection_error = float('inf')
    
    # Optimized confidence score calculation
    num_detected = len(high_confidence_keypoints)
    keypoint_confidence = np.mean(keypoints[high_confidence_mask, 2]) if np.any(high_confidence_mask) else 0.0
    
    # Normalize metrics
    keypoint_ratio = num_detected / len(KEYPOINT_NAMES)
    error_score = max(0, 1 - (reprojection_error / 5.0))  # Assume 5m is max acceptable error
    
    # Weighted average for final confidence
    confidence_score = 0.4 * keypoint_ratio + 0.4 * keypoint_confidence + 0.2 * error_score
    
    # Transform object positions
    transformed_positions = transform_object_positions(object_positions, homography_matrix)
    
    return transformed_positions, reprojection_error, confidence_score, frame_id

def run_inference():
    """
    Main function to run inference and calculate field point coordinates.
    """
    print("\nDebug: Starting run_inference")
    
    try:
        print("Debug: Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        print("Debug: Model loaded successfully")
        
        print("Debug: Reading input image...")
        img = cv2.imread("/home/fawwaz/Pictures/Screenshot from 2025-02-18 14-40-52.png")
        if img is None:
            raise ValueError("Could not read input image")
        print(f"Debug: Image loaded with shape: {img.shape}")
        
        print("Debug: Running YOLO inference...")
        results = model(img, verbose=False)[0]
        print("Debug: Inference completed")
        
        print("Debug: Extracting keypoints...")
        keypoints = results.keypoints.data.cpu().numpy().reshape(-1, 3)
        print(f"Debug: Keypoints shape: {keypoints.shape}")
        print(f"Debug: First few keypoints: {keypoints[:5]}")
        
        
        # Filter keypoints based on confidence threshold
        print(f"Debug: Filtering keypoints with confidence threshold: {CONFIDENCE_THRESHOLD}")
        high_confidence_keypoints = set()
        for i, kpt in enumerate(keypoints):
            if kpt[2] >= 0.8:   #confidence threshold
                high_confidence_keypoints.add(KEYPOINT_NAMES[i])
        
        print(f"Debug: Number of high confidence keypoints: {len(high_confidence_keypoints)}")
        print(f"Debug: High confidence keypoints: {high_confidence_keypoints}")
        
        print("Debug: Finding detectable field points...")
        detectable_points = get_detectable_field_points(high_confidence_keypoints)
        
        print("Debug: Computing field point coordinates...")
        field_point_coords = compute_field_point_coordinates(
            detectable_points,
            keypoints,
            KEYPOINT_NAMES
        )
        
        
        print("Debug: Saving coordinates...")
        np.save("field_point_coordinates.npy", field_point_coords)
        print("Debug: Coordinates saved successfully")\

        #Calculate homography matrix
        homography_matrix = calculate_homography_matrix(field_point_coords, KEYPOINTS_DATA)
        
        # Warp image to top-down view
        warped_img = warp_image_to_top_down(img, homography_matrix)
        
        # Example object positions (replace with actual object detections)
        example_objects = [
            {'object_id': 1, 'pixel_x': 500, 'pixel_y': 300},
            {'object_id': 2, 'pixel_x': 600, 'pixel_y': 400},
            {'object_id': 3, 'pixel_x': 700, 'pixel_y': 500}
        ]
        
        # Transform and plot object positions
        transformed_positions = transform_and_plot_object_positions(
            example_objects,
            homography_matrix,
            warped_img
        )
        
        return field_point_coords, homography_matrix, transformed_positions
        
    except Exception as e:
        print(f"Debug: ✗ Error in run_inference: {str(e)}")
        raise

# Example usage:
if __name__ == "__main__":
    print("\nDebug: Starting main execution")
    try:
        field_point_coords, homography_matrix, transformed_positions = run_inference()
        
        print("\nDebug: Field Point Coordinates:")
        for point, coords in field_point_coords.items():
            print(f"Debug: {point}: ({coords[0]:.2f}, {coords[1]:.2f})")
            
    except Exception as e:
        print(f"Debug: ✗ Main execution failed: {str(e)}") 
