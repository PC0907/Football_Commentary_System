import cv2
import numpy as np
import sys
from ultralytics import YOLO

def detect_ball(video_path, model):
    """Detects footballs in the provided video using the YOLO model."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Prepare to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    output_path = 'output_ball_detection.mp4'
    frame_width = 640
    frame_height = 480
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for processing
        resized_frame = cv2.resize(frame, (frame_width, frame_height))

        # Perform detection
        results = model(resized_frame, conf=0.3)  # Lowered confidence threshold for more detections

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            label = int(box.cls.numpy()[0])

            # Check if the detected object is a ball (assuming '0' is the class ID for the ball)
            if label == 0:  # Change this number based on your class mapping
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a rectangle around the ball
                cv2.putText(resized_frame, "Football", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the annotated frame to the output video
        out.write(resized_frame)

    cap.release()
    out.release()
    print(f"Output video saved as: {output_path}")

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # Load the YOLOv8 model
    video_path = sys.argv[1]  # Video path passed as a command-line argument
    detect_ball(video_path, model)

