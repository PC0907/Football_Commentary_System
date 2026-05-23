import numpy as np
from typing import Dict, List, Tuple, Set
import cv2
from ultralytics import YOLO

print("Debug: Imported required libraries")

# Original dictionary from the notebook
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
    ],
    "left_goal_near_post": [
        ("Goal left post left DOWN", "Goal left post left UP"),  # Line 1
        ("Big rect. left top LEFT", "Small rect. left top LEFT"),  # Line 2
    ],
    "right_goal_far_post": [
        ("Goal right post left DOWN", "Goal right post left UP"),  # Line 1
        ("Big rect. right top RIGHT", "Small rect. right top RIGHT"),  # Line 2
    ],
    "right_goal_near_post": [
        ("Goal right post right DOWN", "Goal right post right UP"),  # Line 1
        ("Big rect. right top RIGHT", "Small rect. right top RIGHT"),  # Line 2
    ],
    "near_center_line_end": [
        ("Middle line UP", "Middle line DOWN"),  # Line 1
        ("Side line top LEFT", "Side line top RIGHT"),  # Line 2
    ],
    "far_center_line_end": [
        ("Middle line UP", "Middle line DOWN"),  # Line 1
        ("Side line bottom LEFT", "Side line bottom RIGHT"),  # Line 2
    ]
}

print(f"Debug: Loaded FIELD_POINT_TO_KEYPOINT_LINES with {len(FIELD_POINT_TO_KEYPOINT_LINES)} field points")

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

# Confidence threshold for keypoint detection
CONFIDENCE_THRESHOLD = 0.5

def get_detectable_field_points(detected_keypoints: Set[str]) -> List[str]:
    """
    Checks which field points have all their keypoints detected.
    """
    print(f"\nDebug: Starting get_detectable_field_points")
    print(f"Debug: Number of detected keypoints: {len(detected_keypoints)}")
    print(f"Debug: First few detected keypoints: {list(detected_keypoints)[:5]}")
    
    detectable_points = []
    
    for field_point, lines in FIELD_POINT_TO_KEYPOINT_LINES.items():
        # Get all unique keypoints for this field point
        keypoints = set()
        for line in lines:
            keypoints.add(line[0])
            keypoints.add(line[1])
        
        print(f"\nDebug: Checking field point: {field_point}")
        print(f"Debug: Required keypoints: {keypoints}")
        
        # Check if all keypoints are detected
        if all(kpt in detected_keypoints for kpt in keypoints):
            print(f"Debug: ✓ All keypoints detected for {field_point}")
            detectable_points.append(field_point)
        else:
            missing = [kpt for kpt in keypoints if kpt not in detected_keypoints]
            print(f"Debug: ✗ Missing keypoints for {field_point}: {missing}")
    
    print(f"\nDebug: Found {len(detectable_points)} detectable field points")
    print(f"Debug: Detectable points: {detectable_points}")
    return detectable_points

def compute_field_point_coordinates(
    field_points: List[str],
    keypoints: np.ndarray,
    keypoint_names: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Computes pixel coordinates for detectable field points using line intersections.
    """
    print(f"\nDebug: Starting compute_field_point_coordinates")
    print(f"Debug: Number of field points to process: {len(field_points)}")
    print(f"Debug: Keypoints array shape: {keypoints.shape}")
    print(f"Debug: Number of keypoint names: {len(keypoint_names)}")
    
    field_point_coords = {}
    
    for field_point in field_points:
        print(f"\nDebug: Processing field point: {field_point}")
        lines = FIELD_POINT_TO_KEYPOINT_LINES[field_point]
        print(f"Debug: Lines for {field_point}: {lines}")
        
        try:
            # Get coordinates for all four keypoints
            line1_start_idx = keypoint_names.index(lines[0][0])
            line1_end_idx = keypoint_names.index(lines[0][1])
            line2_start_idx = keypoint_names.index(lines[1][0])
            line2_end_idx = keypoint_names.index(lines[1][1])
            
            print(f"Debug: Keypoint indices - Line1: ({line1_start_idx}, {line1_end_idx}), Line2: ({line2_start_idx}, {line2_end_idx})")
            
            # Get keypoint coordinates and confidence scores
            line1_start = keypoints[line1_start_idx]
            line1_end = keypoints[line1_end_idx]
            line2_start = keypoints[line2_start_idx]
            line2_end = keypoints[line2_end_idx]
            
            # Check confidence scores
            if (line1_start[2] < CONFIDENCE_THRESHOLD or 
                line1_end[2] < CONFIDENCE_THRESHOLD or 
                line2_start[2] < CONFIDENCE_THRESHOLD or 
                line2_end[2] < CONFIDENCE_THRESHOLD):
                print(f"Debug: ✗ Low confidence for {field_point} - skipping")
                print(f"Debug: Confidence scores - Line1: ({line1_start[2]:.3f}, {line1_end[2]:.3f}), Line2: ({line2_start[2]:.3f}, {line2_end[2]:.3f})")
                continue
            
            print(f"Debug: Line1 coordinates - Start: {line1_start[:2]}, End: {line1_end[:2]}")
            print(f"Debug: Line2 coordinates - Start: {line2_start[:2]}, End: {line2_end[:2]}")
            
            # Compute intersection point
            intersection = line_intersection(
                (line1_start[0], line1_start[1], line1_end[0], line1_end[1]),
                (line2_start[0], line2_start[1], line2_end[0], line2_end[1])
            )
            
            if intersection:
                print(f"Debug: ✓ Found intersection for {field_point}: {intersection}")
                field_point_coords[field_point] = intersection
            else:
                print(f"Debug: ✗ No intersection found for {field_point} (lines are parallel)")
                
        except ValueError as e:
            print(f"Debug: ✗ Error processing {field_point}: {str(e)}")
        except IndexError as e:
            print(f"Debug: ✗ Index error for {field_point}: {str(e)}")
    
    print(f"\nDebug: Successfully computed coordinates for {len(field_point_coords)} field points")
    return field_point_coords

def line_intersection(line1: Tuple[float, float, float, float], 
                     line2: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Calculates the intersection point of two lines.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        print(f"Debug: Lines are parallel - denominator is 0")
        return None
    
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    
    print(f"Debug: Intersection point calculated: ({x:.2f}, {y:.2f})")
    return (x, y)

def plot_field_points_on_image(
    img: np.ndarray,
    field_point_coords: Dict[str, Tuple[float, float]],
    output_path: str = "computed_fieldpoints.jpg"
) -> None:
    """
    Plots the computed field points on the image and saves it.
    
    Args:
        img: Input image
        field_point_coords: Dictionary mapping field points to their coordinates
        output_path: Path to save the output image
    """
    print("\nDebug: Starting plot_field_points_on_image")
    print(f"Debug: Number of field points to plot: {len(field_point_coords)}")
    
    # Create a copy of the image to draw on
    img_with_points = img.copy()
    
    # Define colors for different types of points
    colors = {
        "left": (0, 0, 255),      # Red for left side points
        "right": (255, 0, 0),     # Blue for right side points
        "center": (0, 255, 0),    # Green for center points
        "goal": (255, 255, 0)     # Yellow for goal points
    }
    
    # Plot each field point
    for point_name, coords in field_point_coords.items():
        x, y = int(coords[0]), int(coords[1])
        
        # Determine point color based on name
        if "left" in point_name:
            color = colors["left"]
        elif "right" in point_name:
            color = colors["right"]
        elif "center" in point_name:
            color = colors["center"]
        elif "goal" in point_name:
            color = colors["goal"]
        else:
            color = (255, 255, 255)  # White for other points
        
        # Draw the point
        cv2.circle(img_with_points, (x, y), 5, color, -1)
        
        # Add point name
        cv2.putText(img_with_points, point_name, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        print(f"Debug: Plotted {point_name} at ({x}, {y})")
    
    # Save the image
    cv2.imwrite(output_path, img_with_points)
    print(f"Debug: Saved image with field points to {output_path}")

def calculate_homography_matrix(
    field_point_coords: Dict[str, Tuple[float, float]],
    keypoints_data: List[Tuple[int, float, float, str]]
) -> np.ndarray:
    """
    Calculates the homography matrix using field point coordinates and their corresponding world coordinates.
    Uses all available field points that have corresponding world coordinates.
    
    Args:
        field_point_coords: Dictionary mapping field points to their pixel coordinates
        keypoints_data: List of tuples containing (id, x, y, field_point_name)
        
    Returns:
        Homography matrix
    """
    print("\nDebug: Starting calculate_homography_matrix")
    print(f"Debug: Available field points: {len(field_point_coords)}")
    print(f"Debug: Available keypoints data: {len(keypoints_data)}")
    
    # Prepare source and destination points
    src_points = []
    dst_points = []
    used_points = []
    skipped_points = []
    
    # Create a mapping of field point names to their world coordinates
    world_coords_map = {name: (x, y) for _, x, y, name in keypoints_data}
    
    # Sort points to ensure consistent ordering
    field_points_sorted = sorted(field_point_coords.keys())
    
    # Check all field points
    for field_point_name in field_points_sorted:
        if field_point_name in world_coords_map:
            pixel_x, pixel_y = field_point_coords[field_point_name]
            world_x, world_y = world_coords_map[field_point_name]
            
            # Debug right-side elements specifically
            if "right" in field_point_name.lower():
                print(f"\nDebug: Processing right-side element: {field_point_name}")
                print(f"Debug:   Original pixel coordinates: ({pixel_x:.2f}, {pixel_y:.2f})")
                print(f"Debug:   Original world coordinates: ({world_x:.2f}, {world_y:.2f})")
            
            src_points.append([pixel_x, pixel_y])
            dst_points.append([world_x, world_y])
            used_points.append(field_point_name)
            print(f"Debug: ✓ Using {field_point_name}")
            print(f"Debug:   Pixel: ({pixel_x:.2f}, {pixel_y:.2f})")
            print(f"Debug:   World: ({world_x:.2f}, {world_y:.2f})")
        else:
            skipped_points.append(field_point_name)
            print(f"Debug: ✗ Skipping {field_point_name} - no world coordinates found")
    
    # Check for points in keypoints_data that weren't used
    unused_world_points = set(world_coords_map.keys()) - set(used_points)
    if unused_world_points:
        print("\nDebug: World coordinates available but not used:")
        for point in unused_world_points:
            print(f"Debug: ✗ {point} - no corresponding field point detected")
    
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    print(f"\nDebug: Using {len(src_points)} point pairs for homography calculation")
    print(f"Debug: Skipped {len(skipped_points)} points")
    print(f"Debug: All detected field points are being used in homography computation")
    
    if len(src_points) < 4:
        raise ValueError(f"Insufficient points for homography calculation. Need at least 4 points, got {len(src_points)}")
    
    # Calculate homography matrix with RANSAC
    H, mask = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    print("Debug: Homography matrix calculated successfully")
    print(f"Debug: Homography matrix:\n{H}")
    
    # Analyze which points were considered inliers by RANSAC
    inlier_mask = mask.ravel().tolist()
    inlier_points = [pt for pt, is_inlier in zip(used_points, inlier_mask) if is_inlier]
    outlier_points = [pt for pt, is_inlier in zip(used_points, inlier_mask) if not is_inlier]
    
    print(f"\nDebug: RANSAC inlier analysis:")
    print(f"Debug:   Total points: {len(used_points)}")
    print(f"Debug:   Inlier points: {len(inlier_points)}")
    print(f"Debug:   Outlier points: {len(outlier_points)}")
    
    if outlier_points:
        print("\nDebug: Points considered outliers by RANSAC:")
        for point in outlier_points:
            print(f"Debug:   ✗ {point}")
    
    # Calculate reprojection error
    if len(src_points) > 0:
        src_points_homogeneous = np.column_stack((src_points, np.ones(len(src_points))))
        projected_points = np.dot(H, src_points_homogeneous.T).T
        projected_points = projected_points[:, :2] / projected_points[:, 2:]
        errors = np.linalg.norm(projected_points - dst_points, axis=1)
        mean_error = np.mean(errors)
        print(f"\nDebug: Mean reprojection error: {mean_error:.2f} meters")
        
        # Print individual point errors
        print("\nDebug: Individual point reprojection errors:")
        for i, (point_name, error, is_inlier) in enumerate(zip(used_points, errors, inlier_mask)):
            status = "✓" if is_inlier else "✗"
            print(f"Debug:   {status} {point_name}: {error:.2f} meters")
    
    return H

def warp_image_to_top_down(
    img: np.ndarray,
    homography_matrix: np.ndarray,
    output_size: Tuple[int, int] = (1050, 680)  # Standard football field size in pixels
) -> np.ndarray:
    """
    Warps the input image to a top-down view using the homography matrix.
    Scales the world coordinates from meters to pixels.
    
    Args:
        img: Input image
        homography_matrix: Homography matrix
        output_size: Size of the output image in pixels (width, height)
        
    Returns:
        Warped top-down view of the image
    """
    print("\nDebug: Starting warp_image_to_top_down")
    print(f"Debug: Output size: {output_size}")
    
    # Create scaling matrix to convert from meters to pixels
    # Field dimensions in meters: 105m x 68m
    scale_x = output_size[0] / 105.0  # pixels per meter in x direction
    scale_y = output_size[1] / 68.0   # pixels per meter in y direction
    
    # Create scaling matrix
    scale_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    
    print(f"Debug: Scaling factors - X: {scale_x:.2f} px/m, Y: {scale_y:.2f} px/m")
    
    # Combine homography with scaling
    scaled_homography = np.dot(scale_matrix, homography_matrix)
    
    # Warp the image
    warped_img = cv2.warpPerspective(img, scaled_homography, output_size)
    print("Debug: Image warped successfully")
    
    # Save the warped image
    cv2.imwrite("warped_field.jpg", warped_img)
    print("Debug: Saved warped image to warped_field.jpg")
    
    return warped_img

def transform_and_plot_object_positions(
    object_positions: List[Dict[str, float]],
    homography_matrix: np.ndarray,
    warped_img: np.ndarray
) -> List[Dict[str, float]]:
    """
    Transforms object positions using homography matrix and plots them on the warped image.
    Scales the world coordinates from meters to pixels.
    
    Args:
        object_positions: List of dictionaries containing {object_id, pixel_x, pixel_y}
        homography_matrix: Homography matrix
        warped_img: Warped top-down view of the field
        
    Returns:
        List of transformed object positions
    """
    print("\nDebug: Starting transform_and_plot_object_positions")
    print(f"Debug: Number of objects to transform: {len(object_positions)}")
    
    # Get image dimensions for scaling
    height, width = warped_img.shape[:2]
    scale_x = width / 105.0  # pixels per meter in x direction
    scale_y = height / 68.0  # pixels per meter in y direction
    
    transformed_positions = []
    
    for obj in object_positions:
        # Convert pixel coordinates to homogeneous coordinates
        pixel_coords = np.array([[obj['pixel_x'], obj['pixel_y'], 1]]).T
        
        # Transform coordinates
        world_coords = np.dot(homography_matrix, pixel_coords)
        world_coords = world_coords / world_coords[2]  # Normalize
        
        # Get world coordinates in meters
        world_x_meters, world_y_meters = world_coords[0][0], world_coords[1][0]
        
        # Convert to pixels
        world_x_pixels = world_x_meters * scale_x
        world_y_pixels = world_y_meters * scale_y
        
        # Store transformed position (in both meters and pixels)
        transformed_pos = {
            'object_id': obj['object_id'],
            'world_x_meters': world_x_meters,
            'world_y_meters': world_y_meters,
            'world_x_pixels': world_x_pixels,
            'world_y_pixels': world_y_pixels
        }
        transformed_positions.append(transformed_pos)
        
        # Plot on warped image using pixel coordinates
        plot_x, plot_y = int(world_x_pixels), int(world_y_pixels)
        cv2.circle(warped_img, (plot_x, plot_y), 5, (0, 255, 0), -1)  # Green circle
        cv2.putText(warped_img, f"ID: {obj['object_id']}", (plot_x + 10, plot_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        print(f"Debug: Object {obj['object_id']}")
        print(f"Debug:   Pixel: ({obj['pixel_x']:.2f}, {obj['pixel_y']:.2f})")
        print(f"Debug:   World (meters): ({world_x_meters:.2f}, {world_y_meters:.2f})")
        print(f"Debug:   World (pixels): ({world_x_pixels:.2f}, {world_y_pixels:.2f})")
    
    # Save the image with plotted positions
    cv2.imwrite("warped_field_with_objects.jpg", warped_img)
    print("Debug: Saved warped image with object positions to warped_field_with_objects.jpg")
    
    return transformed_positions

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
        img = cv2.imread("selected_frame.jpg")
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
            if kpt[2] >= CONFIDENCE_THRESHOLD:
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
        print("Debug: Coordinates saved successfully")
        
        # Print all calculated field point positions
        print("\nDebug: All Calculated Field Point Positions:")
        for point, coords in field_point_coords.items():
            print(f"Debug: {point}: ({coords[0]:.2f}, {coords[1]:.2f})")
        
        # Plot and save the field points on the image
        plot_field_points_on_image(img, field_point_coords)
        
        # Calculate homography matrix
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

class PointLabeler:
    def __init__(self, image_path: str, homography_matrix: np.ndarray, 
                 color: Tuple[int, int, int] = (0, 255, 0)):
        """
        Interactive player position labeling tool.
        
        Args:
            image_path: Path to the image
            homography_matrix: Pre-computed homography matrix
            color: BGR color for points (default: green)
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        self.color = color
        self.points = []  # Player positions
        self.homography_matrix = homography_matrix
        self.window_name = "Player Position Labeler"
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point labeling."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Added player position: ({x}, {y})")
            self.update_display()
            
    def update_display(self):
        """Update the display with current points."""
        display = self.image.copy()
        
        # Draw points
        for i, point in enumerate(self.points):
            cv2.circle(display, point, 5, self.color, -1)
            cv2.putText(display, f"P-{i+1}", (point[0] + 10, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)
        
        # Show instructions
        cv2.putText(display, "Left click: Add player position", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Enter: Show transformed positions", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "ESC: Exit", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show current point count
        cv2.putText(display, f"Players labeled: {len(self.points)}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        
        cv2.imshow(self.window_name, display)
        
    def run(self):
        """Run the interactive labeling tool."""
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 13:  # Enter
                if len(self.points) > 0:
                    self.show_transformed_positions()
                else:
                    print("No player positions labeled yet")
        
        cv2.destroyAllWindows()
        return self.points
    
    def show_transformed_positions(self):
        """Show original and transformed positions using transform_and_plot_object_positions."""
        try:
            # Convert points to the format expected by transform_and_plot_object_positions
            object_positions = [
                {'object_id': i+1, 'pixel_x': x, 'pixel_y': y}
                for i, (x, y) in enumerate(self.points)
            ]
            
            # Create a copy of the image for the transformed view
            transformed_view = self.image.copy()
            
            # Warp the image to top-down view
            warped_img = warp_image_to_top_down(transformed_view, self.homography_matrix)
            
            # Use the existing function to transform and plot positions
            transformed_positions = transform_and_plot_object_positions(
                object_positions,
                self.homography_matrix,
                warped_img
            )
            
            # Create side-by-side display
            # Resize the original image to match the warped image height
            height, width = warped_img.shape[:2]
            original_resized = cv2.resize(self.image, (int(width * (self.image.shape[1]/self.image.shape[0])), height))
            
            # Create a combined display
            combined = np.hstack((original_resized, warped_img))
            
            # Add titles
            cv2.putText(combined, "Original View", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Top-Down View", (width + 10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Calculate scaling factor to fit 1920x1080 display
            target_width = 1920
            target_height = 1080
            scale = min(target_width / combined.shape[1], target_height / combined.shape[0])
            
            # Resize the combined image
            new_width = int(combined.shape[1] * scale)
            new_height = int(combined.shape[0] * scale)
            combined = cv2.resize(combined, (new_width, new_height))
            
            # Create a resizable window
            cv2.namedWindow("Transformed Positions", cv2.WINDOW_NORMAL)
            
            # Set window size
            cv2.resizeWindow("Transformed Positions", new_width, new_height)
            
            # Show results
            cv2.imshow("Transformed Positions", combined)
            cv2.waitKey(0)
            cv2.destroyWindow("Transformed Positions")
            
            # Print transformed positions
            print("\nTransformed Player Positions:")
            for pos in transformed_positions:
                print(f"Player {pos['object_id']}:")
                print(f"  World Coordinates (meters): ({pos['world_x_meters']:.2f}, {pos['world_y_meters']:.2f})")
                print(f"  Pixel Coordinates: ({pos['world_x_pixels']:.2f}, {pos['world_y_pixels']:.2f})")
            
        except Exception as e:
            print(f"Error in transforming positions: {str(e)}")
            raise  # Re-raise the exception to stop execution

def interactive_player_labeling(image_path: str, homography_matrix: np.ndarray):
    """
    Run the interactive player position labeling tool.
    
    Args:
        image_path: Path to the image
        homography_matrix: Pre-computed homography matrix
    """
    print("\nStarting interactive player position labeling")
    print("Instructions:")
    print("1. Left click to add player positions")
    print("2. Press ENTER to see transformed positions")
    print("3. Press ESC to exit")
    
    labeler = PointLabeler(image_path, homography_matrix)
    points = labeler.run()
    
    print("\nLabeled player positions:")
    for i, point in enumerate(points):
        print(f"  Player {i+1}: ({point[0]}, {point[1]})")
    
    return points

# Example usage in main:
if __name__ == "__main__":
    print("\nDebug: Starting main execution")
    try:
        # First get the homography matrix
        print("Debug: Running inference to get homography matrix...")
        field_point_coords, homography_matrix, transformed_positions = run_inference()
        
        if homography_matrix is None:
            raise ValueError("Failed to compute homography matrix")
            
        print("\nDebug: Homography matrix computed successfully")
        print("Debug: Starting interactive player labeling...")
        
        # Then run interactive player labeling
        player_positions = interactive_player_labeling("selected_frame.jpg", homography_matrix)
        
        print("\nDebug: Execution completed successfully")
            
    except Exception as e:
        print(f"Debug: ✗ Main execution failed: {str(e)}")
        raise  # Re-raise the exception to stop execution 