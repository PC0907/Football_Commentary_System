import unittest
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tracking import Tracker  # Import the Tracker class
import cv2


class TestTracker(unittest.TestCase):
    """
    Test case for the Tracker class.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        self.tracker = Tracker(frame_rate=30)  # Initialize the tracker
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8) # Create a dummy frame

    def test_initialization(self):
        """
        Test that the tracker is initialized correctly.
        """
        self.assertIsInstance(self.tracker, Tracker, "Tracker should be an instance of the Tracker class.")
        self.assertEqual(self.tracker.track_thresh, 0.25, "track_thresh should be 0.25")
        self.assertEqual(self.tracker.match_thresh, 0.8, "match_thresh should be 0.8")
        self.assertEqual(self.tracker.frame_rate, 30, "frame_rate should be 30")
        self.assertIsInstance(self.tracker.tracked_objects, dict, "tracked_objects should be a dictionary.")

    def test_update_with_one_detection(self):
        """
        Test that the tracker correctly tracks a single detection.
        """
        detections = [{
            'x1': 10, 'y1': 20, 'x2': 30, 'y2': 40, 'class_id': 0, 'confidence': 0.9, 'label': 'person'
        }]
        tracked_objects = self.tracker.update(detections, self.frame)
        self.assertEqual(len(tracked_objects), 1, "Should track one object.")
        track = tracked_objects[0]
        self.assertIsInstance(track, dict, "Tracked object should be a dictionary.")
        self.assertTrue(all(key in track for key in ['track_id', 'x1', 'y1', 'x2', 'y2', 'class_id', 'label', 'confidence', 'trajectory', 'age', 'avg_speed', 'active']),
                        "Tracked object should contain required keys.")
        self.assertEqual(track['x1'], 10, "x1 should match")
        self.assertEqual(track['y1'], 20, "y1 should match")
        self.assertEqual(track['x2'], 30, "x2 should match")
        self.assertEqual(track['y2'], 40, "y2 should match")
        self.assertEqual(track['class_id'], 0, "class_id should match")
        self.assertEqual(track['label'], 'person', "label should match")
        self.assertAlmostEqual(track['confidence'], 0.9, places=5, msg="confidence should match")
        self.assertEqual(track['trajectory'], [(20, 30)], "trajectory should contain the center point.")
        self.assertEqual(track['age'], 1, "age should be 1")
        self.assertEqual(track['avg_speed'], (0,0), "avg_speed should be (0,0)")
        self.assertTrue(track['active'], "Object should be active")
        self.assertIsInstance(track['track_id'], int, "track_id should be an integer.")

    def test_update_multiple_frames(self):
        """
        Test that the tracker correctly tracks an object over multiple frames.
        """
        detections1 = [{
            'x1': 10, 'y1': 20, 'x2': 30, 'y2': 40, 'class_id': 0, 'confidence': 0.9, 'label': 'person'
        }]
        tracked_objects1 = self.tracker.update(detections1, self.frame)
        track_id = tracked_objects1[0]['track_id']

        detections2 = [{
            'x1': 15, 'y1': 25, 'x2': 35, 'y2': 45, 'class_id': 0, 'confidence': 0.8, 'label': 'person'
        }]
        tracked_objects2 = self.tracker.update(detections2, self.frame)
        self.assertEqual(len(tracked_objects2), 1, "Should still track one object.")
        track2 = tracked_objects2[0]
        self.assertEqual(track2['track_id'], track_id, "Should have the same track ID.")
        self.assertEqual(track2['x1'], 15, "x1 should be updated")
        self.assertEqual(track2['y1'], 25, "y1 should be updated")
        self.assertEqual(track2['x2'], 35, "x2 should be updated")
        self.assertEqual(track2['y2'], 45, "y2 should be updated")
        self.assertAlmostEqual(track2['confidence'], 0.8, places=5, msg="confidence should be updated")
        self.assertEqual(len(track2['trajectory']), 2, "trajectory should have two points.")
        self.assertEqual(track2['trajectory'][0], (20, 30), "First point in trajectory")
        self.assertEqual(track2['trajectory'][1], (25, 35), "Second point in trajectory")
        self.assertEqual(track2['age'], 2, "age should be incremented")
        vx, vy = track2['avg_speed']
        self.assertAlmostEqual(vx, (25-20) * 30, places=2, msg = "vx check") #delta_x * frame_rate
        self.assertAlmostEqual(vy, (35-30) * 30, places=2, msg = "vy check")
        self.assertTrue(track2['active'], "Object should be active")

    def test_lost_track(self):
        """
        Test that the tracker handles a lost track correctly.
        """
        detections1 = [{
            'x1': 10, 'y1': 20, 'x2': 30, 'y2': 40, 'class_id': 0, 'confidence': 0.9, 'label': 'person'
        }]
        tracked_objects1 = self.tracker.update(detections1, self.frame)
        track_id = tracked_objects1[0]['track_id']

        # No detections in the next frame (lost track)
        tracked_objects2 = self.tracker.update([], self.frame)
        self.assertEqual(len(tracked_objects2), 1, "Should still have one track (temporarily).")
        track2 = tracked_objects2[0]
        self.assertEqual(track2['track_id'], track_id, "Should have the same track ID.")
        self.assertFalse(track2['active'], "Object should not be active")
        self.assertEqual(track2['age'], 2, "age should be incremented")

        # Check that the track is removed after a few lost frames (default is 15)
        for _ in range(14):
            tracked_objects = self.tracker.update([], self.frame)
        self.assertEqual(len(self.tracker.get_tracked_objects()), 1, "Track should still exist")
        tracked_objects_final = self.tracker.update([], self.frame)
        self.assertEqual(len(self.tracker.get_tracked_objects()), 0, "Track should be removed after 15 frames.")
        self.assertEqual(self.tracker.lost_frames.get(track_id), None, "lost_frames entry should be removed")

    def test_multiple_objects(self):
        """
        Test tracking of multiple objects.
        """
        detections1 = [
            {'x1': 10, 'y1': 20, 'x2': 30, 'y2': 40, 'class_id': 0, 'confidence': 0.9, 'label': 'person'},
            {'x1': 50, 'y1': 60, 'x2': 70, 'y2': 80, 'class_id': 1, 'confidence': 0.8, 'label': 'car'}
        ]
        tracked_objects1 = self.tracker.update(detections1, self.frame)
        self.assertEqual(len(tracked_objects1), 2, "Should track two objects.")
        track_ids = [obj['track_id'] for obj in tracked_objects1]
        self.assertNotEqual(track_ids[0], track_ids[1], "Objects should have different track IDs.")

        detections2 = [
            {'x1': 12, 'y1': 22, 'x2': 32, 'y2': 42, 'class_id': 0, 'confidence': 0.85, 'label': 'person'},
            {'x1': 55, 'y1': 65, 'x2': 75, 'y2': 85, 'class_id': 1, 'confidence': 0.75, 'label': 'car'}
        ]
        tracked_objects2 = self.tracker.update(detections2, self.frame)
        self.assertEqual(len(tracked_objects2), 2, "Should still track two objects.")
        tracked_ids_2 = [obj['track_id'] for obj in tracked_objects2]
        self.assertIn(tracked_ids_2[0], track_ids, "Should keep tracking the same objects.")
        self.assertIn(tracked_ids_2[1], track_ids, "Should keep tracking the same objects.")

if __name__ == '__main__':
    unittest.main()
