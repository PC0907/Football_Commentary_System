import cv2
import sys
from ultralytics import YOLO

def detect_ball(video_path, model):
    """Detects the ball in the provided video using the YOLO model."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Prepare to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Set codec for MP4
    output_path = 'output_ball_detection.mp4'
    out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame, conf=0.5)  # Confidence threshold set to 0.5
        for box in results[0].boxes:
            # Extract the bounding box coordinates and label
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            label = int(box.cls.numpy()[0])

            # Check if the detected object is a ball (assuming '4' is the class ID for the ball)
            if label == 4:  # Change this number based on your class mapping
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a rectangle around the ball
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the annotated frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('Ball Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved as: {output_path}")

if __name__ == "__main__":
    model = YOLO("yolov5s.pt")  # Load the YOLO model
    video_path = sys.argv[1]  # Video path passed as a command-line argument
    detect_ball(video_path, model)

