import cv2
import numpy as np
from homography import process_frame
import time

def visualize_results():
    """
    Visualizes the results of the homography transformation.
    This function loads a frame, processes it through the homography module,
    and displays the results with performance metrics.
    """
    # Load frame
    frame = cv2.imread("/home/fawwaz/Pictures/Screenshot from 2025-02-18 14-41-07.png")
    if frame is None:
        print("Error: Could not load image")
        return

    # Define object positions (from your object detector)
    object_positions = [
        {"object_id": 1, "pixel_x": 100, "pixel_y": 200},  # Player 1
        {"object_id": 2, "pixel_x": 300, "pixel_y": 400},  # Player 2
        {"object_id": "ball", "pixel_x": 250, "pixel_y": 300}  # Ball
    ]

    # Time the processing for performance measurement
    start_time = time.time()
    
    # Process frame and get transformed positions
    frame_id = 1  # Example frame ID
    transformed_positions, reprojection_error, confidence, returned_frame_id = process_frame(frame, object_positions, frame_id)
    
    processing_time = time.time() - start_time
    print(f"\nProcessing time: {processing_time:.3f} seconds")

    # Print metrics
    print("\nHomography Metrics:")
    print(f"Frame ID: {returned_frame_id}")
    print(f"Reprojection Error: {reprojection_error:.2f} meters")
    print(f"Confidence Score: {confidence:.2%}")

    # Print transformed positions
    print("\nTransformed Positions:")
    for pos in transformed_positions:
        print(f"Object {pos['object_id']} is at ({pos['world_x_meters']:.2f}, {pos['world_y_meters']:.2f}) meters")

    # Pre-allocate text positions and labels for better performance
    text_positions = [
        (10, 30),   # Frame ID
        (10, 70),   # Error
        (10, 110)   # Confidence
    ]
    
    # Vectorized object position drawing for better performance
    positions = np.array([[pos['pixel_x'], pos['pixel_y']] for pos in object_positions], dtype=np.int32)
    for pos in positions:
        cv2.circle(frame, tuple(pos), 5, (0, 255, 0), -1)
    
    # Add text labels with pre-formatted strings
    texts = [
        f"Frame ID: {returned_frame_id}",
        f"Error: {reprojection_error:.2f}m",
        f"Confidence: {confidence:.2%}"
    ]
    
    # Draw all text in a single loop
    for (x, y), text in zip(text_positions, texts):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display results
    cv2.imshow("Homography Test Results", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save results
    cv2.imwrite("homography_test_results.jpg", frame)
    print("\nResults saved as 'homography_test_results.jpg'")

if __name__ == "__main__":
    visualize_results()
