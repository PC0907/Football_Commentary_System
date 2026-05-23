import cv2
import torch
from ultralytics import YOLO  # Importing YOLO

class ObjectDetector:
    """
    Class for performing object detection using the YOLO model.
    """

    def __init__(self, model_path="models/best_object.pt", use_gpu=True):
        """
        Initializes the ObjectDetector.

        Args:
            model_path (str): Path to the YOLO model file (e.g., 'best_object.pt').
            use_gpu (bool, optional): Whether to use GPU for inference. Defaults to False.
        """
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu' # Determine the device
        try:
            self.model = YOLO(model_path)  # Load the YOLO model
            self.model.to(self.device) # Move model to specified device
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def detect(self, frame):
        """
        Performs object detection on the input frame.

        Args:
            frame (numpy.ndarray): The input video frame (BGR format).

        Returns:
            list: A list of detections. Each detection is a dictionary containing:
                - 'x1' (int): x-coordinate of the top-left corner of the bounding box.
                - 'y1' (int): y-coordinate of the top-left corner of the bounding box.
                - 'x2' (int): x-coordinate of the bottom-right corner of the bounding box.
                - 'y2' (int): y-coordinate of the bottom-right corner of the bounding box.
                - 'class_id' (int): The class ID of the detected object.
                - 'confidence' (float): The confidence score of the detection.
                - 'label' (str): The class name of the detected object.
        """
        try:
            results = self.model(frame, verbose=False)  # Perform inference, suppress verbose output
            detections = []
            for result in results: #Iterate through the results
                boxes = result.boxes #get the bounding boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    xyxy = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    label = self.model.names[class_id]  # Get class name from ID

                    x1, y1, x2, y2 = map(int, xyxy)  # Convert to integers

                    detections.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'class_id': class_id,
                        'confidence': confidence,
                        'label': label
                    })
            return detections
        except Exception as e:
            print(f"Error in detect: {e}")
            return []  # Return an empty list in case of an error.  Important for pipeline stability.
        
    def close(self):
        """
        Releases any resources used by the ObjectDetector.  Specifically the model
        """
        del self.model
        torch.cuda.empty_cache() # release the gpu memory.
