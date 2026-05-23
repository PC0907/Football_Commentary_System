import cv2
import unittest
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from object_detection import ObjectDetector  # Import the ObjectDetector class

class TestObjectDetector(unittest.TestCase):
    """
    Test case for the ObjectDetector class.
    """

    def setUp(self):
        """
        Set up the test environment.  This method is called before each test.
        """
        #  Replace with the actual path to your YOLO model.  Use a valid path.
        self.model_path = "models/best_object.pt"  #  Path to a YOLO model (e.g., yolov8n.pt, yolov8s.pt).  Download if needed.
        self.image_path = "data/image.jpg"  # Path to a test image.  Use a valid image.
        # Create a dummy image if the image file does not exist
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                raise FileNotFoundError(f"Image file not found at {self.image_path}")
        except FileNotFoundError:
            print(f"Warning: Image file not found at {self.image_path}. Creating a dummy image.")
            self.image_path = "dummy_image.jpg"
            # Create a black image
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(self.image_path, dummy_img)


        self.detector = ObjectDetector(self.model_path, use_gpu=False)  # Initialize the detector

    def tearDown(self):
        """
        Clean up the test environment.  This method is called after each test.
        """
        self.detector.close() # release resources.

    def test_load_model(self):
        """
        Test that the model loads correctly.
        """
        self.assertIsInstance(self.detector.model, torch.nn.Module, "Model should be a PyTorch model.")

    def test_detect_objects(self):
        """
        Test that the detect method returns a list of detections.
        """
        frame = cv2.imread(self.image_path)
        detections = self.detector.detect(frame)
        self.assertIsInstance(detections, list, "detect() should return a list.")
        if detections:
            self.assertIsInstance(detections[0], dict, "Each detection should be a dictionary.")

    def test_detection_output_format(self):
        """
        Test the format of the detection output.
        """
        frame = cv2.imread(self.image_path)
        detections = self.detector.detect(frame)
        if detections:
            detection = detections[0]  # Get the first detection
            self.assertTrue(all(key in detection for key in ['x1', 'y1', 'x2', 'y2', 'class_id', 'confidence', 'label']),
                            "Detection dictionary should contain required keys.")
            self.assertIsInstance(detection['x1'], int, "x1 should be an integer.")
            self.assertIsInstance(detection['y1'], int, "y1 should be an integer.")
            self.assertIsInstance(detection['x2'], int, "x2 should be an integer.")
            self.assertIsInstance(detection['y2'], int, "y2 should be an integer.")
            self.assertIsInstance(detection['class_id'], int, "class_id should be an integer.")
            self.assertEqual(detection['confidence'].dtype, np.float32, "confidence should be a float.")
            self.assertIsInstance(detection['label'], str, "label should be a string.")

    def test_detect_no_objects(self):
        """
        Test that the detect method returns an empty list when no objects are detected.
        """
        # Create an empty frame (all black)
        empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = self.detector.detect(empty_frame)
        self.assertEqual(detections, [], "Should return an empty list when no objects are detected.")



if __name__ == '__main__':
    unittest.main()
