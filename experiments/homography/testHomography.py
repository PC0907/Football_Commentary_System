import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from ultralytics import YOLO
import argparse

# Import the homography transformation functions
from homography import run_homography_transformation, FIELD_POINT_TO_KEYPOINT_LINES, KEYPOINT_NAMES

# Field dimensions (in meters)
FIELD_WIDTH = 105  # Standard FIFA field width
FIELD_HEIGHT = 68  # Standard FIFA field height

def visualize_keypoints(frame, keypoints, confidence_threshold=0.3):
    """
    Visualize detected keypoints on the original frame
    
    Args:
        frame: Original frame
        keypoints: Detected keypoints array [N, 3] (x, y, confidence)
        confidence_threshold: Minimum confidence to display keypoint
    
    Returns:
        Frame with keypoints visualized
    """
    vis_frame = frame.copy()
    
    # Draw keypoints
    for i, kpt in enumerate(keypoints):
        x, y, conf = kpt
        if conf > confidence_threshold:
            # Draw circle for each keypoint
            cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # Add keypoint index
            cv2.putText(vis_frame, str(i), (int(x) + 5, int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return vis_frame

def draw_field_lines(field_img):
    """
    Draw soccer field lines on an empty field image
    
    Args:
        field_img: Empty field image
    
    Returns:
        Field image with lines drawn
    """
    # Field dimensions in pixels (scale the meters to pixels)
    scale_factor = 8  # 8 pixels per meter
    field_width_px = int(FIELD_WIDTH * scale_factor)
    field_height_px = int(FIELD_HEIGHT * scale_factor)
    
    # Create a white image
    field = np.ones((field_height_px, field_width_px, 3), dtype=np.uint8) * 50  # Dark green background
    
    # Draw green field
    cv2.rectangle(field, (0, 0), (field_width_px, field_height_px), (50, 120, 50), -1)
    
    # Draw white lines (all coordinates scaled by scale_factor)
    # Outer boundary
    cv2.rectangle(field, (0, 0), (field_width_px, field_height_px), (255, 255, 255), 2)
    
    # Center line
    center_x = field_width_px // 2
    cv2.line(field, (center_x, 0), (center_x, field_height_px), (255, 255, 255), 2)
    
    # Center circle
    center_y = field_height_px // 2
    radius = int(9.15 * scale_factor)  # 9.15m radius
    cv2.circle(field, (center_x, center_y), radius, (255, 255, 255), 2)
    cv2.circle(field, (center_x, center_y), 3, (255, 255, 255), -1)  # Center spot
    
    # Penalty areas
    # Left penalty area
    pen_area_width = int(16.5 * scale_factor)
    pen_area_height = int(40.3 * scale_factor)
    pen_area_y_start = (field_height_px - pen_area_height) // 2
    cv2.rectangle(field, (0, pen_area_y_start), (pen_area_width, pen_area_y_start + pen_area_height), (255, 255, 255), 2)
    
    # Right penalty area
    cv2.rectangle(field, (field_width_px - pen_area_width, pen_area_y_start), 
                 (field_width_px, pen_area_y_start + pen_area_height), (255, 255, 255), 2)
    
    # Goal areas
    goal_area_width = int(5.5 * scale_factor)
    goal_area_height = int(18.32 * scale_factor)
    goal_area_y_start = (field_height_px - goal_area_height) // 2
    
    # Left goal area
    cv2.rectangle(field, (0, goal_area_y_start), (goal_area_width, goal_area_y_start + goal_area_height), (255, 255, 255), 2)
    
    # Right goal area
    cv2.rectangle(field, (field_width_px - goal_area_width, goal_area_y_start), 
                 (field_width_px, goal_area_y_start + goal_area_height), (255, 255, 255), 2)
    
    # Penalty spots
    left_pen_spot_x = int(11 * scale_factor)
    right_pen_spot_x = field_width_px - left_pen_spot_x
    cv2.circle(field, (left_pen_spot_x, center_y), 3, (255, 255, 255), -1)
    cv2.circle(field, (right_pen_spot_x, center_y), 3, (255, 255, 255), -1)
    
    # Penalty arcs
    pen_arc_radius = radius
    # Left penalty arc
    cv2.ellipse(field, (left_pen_spot_x, center_y), (pen_arc_radius, pen_arc_radius), 
                0, 310, 50, (255, 255, 255), 2)
    
    # Right penalty arc
    cv2.ellipse(field, (right_pen_spot_x, center_y), (pen_arc_radius, pen_arc_radius), 
                0, 130, 230, (255, 255, 255), 2)
    
    # Corner arcs
    corner_radius = int(1 * scale_factor)
    # Top-left corner
    cv2.ellipse(field, (0, 0), (corner_radius, corner_radius), 0, 0, 90, (255, 255, 255), 2)
    # Top-right corner
    cv2.ellipse(field, (field_width_px, 0), (corner_radius, corner_radius), 0, 90, 180, (255, 255, 255), 2)
    # Bottom-left corner
    cv2.ellipse(field, (0, field_height_px), (corner_radius, corner_radius), 0, 270, 360, (255, 255, 255), 2)
    # Bottom-right corner
    cv2.ellipse(field, (field_width_px, field_height_px), (corner_radius, corner_radius), 0, 180, 270, (255, 255, 255), 2)
    
    return field

def draw_objects_on_field(field_img, player_coords, ball_coord=None, scale_factor=8):
    """
    Draw players and ball on the field image
    
    Args:
        field_img: Field image with lines
        player_coords: List of transformed player coordinates (in meters)
        ball_coord: Transformed ball coordinate (in meters)
        scale_factor: Scale factor to convert meters to pixels
        
    Returns:
        Field image with players and ball drawn
    """
    field_with_objects = field_img.copy()
    
    # Draw players
    for i, coord in enumerate(player_coords):
        x, y = coord
        # Convert to pixel coordinates
        px = int(x * scale_factor)
        py = int(y * scale_factor)
        
        # Draw player as circle
        cv2.circle(field_with_objects, (px, py), 5, (0, 0, 255), -1)
        
        # Add player number
        cv2.putText(field_with_objects, str(i+1), (px + 7, py), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw ball
    if ball_coord is not None:
        ball_x, ball_y = ball_coord
        ball_px = int(ball_x * scale_factor)
        ball_py = int(ball_y * scale_factor)
        
        # Draw ball as circle with different color
        cv2.circle(field_with_objects, (ball_px, ball_py), 4, (255, 165, 0), -1)
    
    return field_with_objects

def visualize_keypoint_lines(frame, keypoints):
    """
    Visualize the lines used for finding field points
    
    Args:
        frame: Original frame
        keypoints: Detected keypoints
        
    Returns:
        Frame with lines visualized
    """
    vis_frame = frame.copy()
    
    # Draw lines from the field point mapping
    for field_point, lines in FIELD_POINT_TO_KEYPOINT_LINES.items():
        for line_pair in lines:
            try:
                # Get the indices of the keypoints
                idx1 = KEYPOINT_NAMES.index(line_pair[0])
                idx2 = KEYPOINT_NAMES.index(line_pair[1])
                
                # Get the coordinates
                kpt1 = keypoints[idx1]
                kpt2 = keypoints[idx2]
                
                # Only draw if both keypoints are confident
                if kpt1[2] > 0.3 and kpt2[2] > 0.3:
                    p1 = (int(kpt1[0]), int(kpt1[1]))
                    p2 = (int(kpt2[0]), int(kpt2[1]))
                    
                    # Draw line
                    cv2.line(vis_frame, p1, p2, (0, 255, 255), 1)
            except (ValueError, IndexError):
                # Skip if keypoints not found
                continue
    
    return vis_frame

def create_normalized_bbox_input(frame, num_players=3):
    """
    Create test bounding boxes for players and ball
    
    Args:
        frame: Input frame
        num_players: Number of test players to generate
        
    Returns:
        tuple: (player_bboxes, ball_bbox)
    """
    height, width = frame.shape[:2]
    
    # Create some test player bounding boxes [x, y, w, h]
    player_bboxes = []
    for i in range(num_players):
        # Distribute players across the width of the frame
        x = int(width * (i + 1) / (num_players + 1))
        y = int(height * 0.6)  # Place in the bottom half
        w = int(width * 0.05)  # Width ~5% of frame width
        h = int(height * 0.15)  # Height ~15% of frame height
        player_bboxes.append([x, y, w, h])
    
    # Create ball bounding box
    ball_x = int(width * 0.5)
    ball_y = int(height * 0.4)
    ball_w = int(width * 0.02)
    ball_h = int(height * 0.02)
    ball_bbox = [ball_x, ball_y, ball_w, ball_h]
    
    # Normalize bounding boxes to [0, 1] if needed
    if False:  # Set to True if your model expects normalized coordinates
        for i in range(len(player_bboxes)):
            player_bboxes[i][0] /= width
            player_bboxes[i][1] /= height
            player_bboxes[i][2] /= width
            player_bboxes[i][3] /= height
            
        ball_bbox[0] /= width
        ball_bbox[1] /= height
        ball_bbox[2] /= width
        ball_bbox[3] /= height
    
    return player_bboxes, ball_bbox

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test soccer field homography transformation')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='./best.pt', help='Path to the YOLO model')
    parser.add_argument('--players', type=int, default=5, help='Number of test players to generate')
    parser.add_argument('--normalize', action='store_true', help='Use normalized coordinates')
    parser.add_argument('--output', type=str, default='output', help='Output directory for visualization')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found")
        return
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Load the model
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Read the image
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image {args.image}")
        return
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Run keypoint detection
    results = model(frame, verbose=False)[0]
    
    # Check if keypoints were detected
    if results.keypoints is None:
        print("No keypoints detected in the frame")
        return
    
    # Get keypoints data
    keypoints = results.keypoints.data.cpu().numpy()[0]
    
    # Create test bounding boxes
    player_bboxes, ball_bbox = create_normalized_bbox_input(frame, args.players)
    
    # If using normalized coordinates, denormalize them
    if args.normalize:
        # Denormalize for visualization purposes
        player_bboxes_vis = []
        for bbox in player_bboxes:
            player_bboxes_vis.append([
                int(bbox[0] * width),
                int(bbox[1] * height),
                int(bbox[2] * width),
                int(bbox[3] * height)
            ])
        
        ball_bbox_vis = [
            int(ball_bbox[0] * width),
            int(ball_bbox[1] * height),
            int(ball_bbox[2] * width),
            int(ball_bbox[3] * height)
        ]
    else:
        player_bboxes_vis = player_bboxes
        ball_bbox_vis = ball_bbox
    
    # Run the homography transformation
    result = run_homography_transformation(frame, player_bboxes, ball_bbox)
    
    if not result["success"]:
        print("Homography transformation failed")
        # Still visualize the keypoints to debug
        vis_keypoints = visualize_keypoints(frame, keypoints)
        vis_lines = visualize_keypoint_lines(frame, keypoints)
        
        # Save visualizations
        cv2.imwrite(os.path.join(args.output, "keypoints.jpg"), vis_keypoints)
        cv2.imwrite(os.path.join(args.output, "keypoint_lines.jpg"), vis_lines)
        return
    
    # Visualize keypoints
    vis_keypoints = visualize_keypoints(frame, keypoints)
    
    # Visualize keypoint lines
    vis_lines = visualize_keypoint_lines(frame, keypoints)
    
    # Visualize bounding boxes on original frame
    vis_bboxes = frame.copy()
    for bbox in player_bboxes_vis:
        x, y, w, h = bbox
        cv2.rectangle(vis_bboxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Add foot point (bottom center)
        foot_x, foot_y = int(x + w/2), int(y + h)
        cv2.circle(vis_bboxes, (foot_x, foot_y), 3, (0, 0, 255), -1)
    
    # Add ball
    x, y, w, h = ball_bbox_vis
    cv2.rectangle(vis_bboxes, (x, y), (x+w, y+h), (255, 165, 0), 2)
    ball_center_x, ball_center_y = int(x + w/2), int(y + h/2)
    cv2.circle(vis_bboxes, (ball_center_x, ball_center_y), 3, (255, 165, 0), -1)
    
    # Draw soccer field
    field_img = draw_field_lines(None)
    
    # Draw players and ball on field
    field_with_objects = draw_objects_on_field(field_img, result["players"], result["ball"])
    
    # Save all visualizations
    cv2.imwrite(os.path.join(args.output, "keypoints.jpg"), vis_keypoints)
    cv2.imwrite(os.path.join(args.output, "keypoint_lines.jpg"), vis_lines)
    cv2.imwrite(os.path.join(args.output, "bounding_boxes.jpg"), vis_bboxes)
    cv2.imwrite(os.path.join(args.output, "field_visualization.jpg"), field_with_objects)
    
    # Optionally create side-by-side visualization
    height, width = frame.shape[:2]
    field_height, field_width = field_with_objects.shape[:2]
    
    # Resize field to match frame height
    scale = height / field_height
    resized_field = cv2.resize(field_with_objects, (int(field_width * scale), height))
    
    # Create side-by-side image
    combined = np.hstack((vis_bboxes, resized_field))
    cv2.imwrite(os.path.join(args.output, "combined_visualization.jpg"), combined)
    
    print(f"Visualization saved to {args.output}")
    
    # Display homography matrix
    print("\nHomography Matrix:")
    print(result["homography_matrix"])
    
    # Display transformed coordinates
    print("\nTransformed Player Coordinates (meters):")
    for i, coord in enumerate(result["players"]):
        print(f"Player {i+1}: ({coord[0]:.2f}, {coord[1]:.2f})")
    
    if result["ball"] is not None:
        print("\nTransformed Ball Coordinate (meters):")
        print(f"Ball: ({result['ball'][0]:.2f}, {result['ball'][1]:.2f})")

if __name__ == "__main__":
    main()