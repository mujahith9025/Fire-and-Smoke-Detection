"""
Unit Tests for Fire and Smoke Detector Engine & Grad-CAM Heatmap
"""

import sys
import os
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector.model_engine import detector
from detector.gradcam import make_gradcam_heatmap, overlay_gradcam
from detector.video_processor import video_processor

class TestDetectorEngine(unittest.TestCase):

    def test_detector_initialization(self):
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.model)
        self.assertEqual(len(detector.img_size), 2)

    def test_predict_synthetic_frame(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # Red channel full

        res = detector.predict_frame(frame)
        self.assertIn("label", res)
        self.assertIn("confidence", res)
        self.assertIn("is_fire", res)
        self.assertIsInstance(res["confidence"], float)

    def test_gradcam_heatmap_generation(self):
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        batch = (frame.astype("float32") / 255.0)[None, ...]

        heatmap = make_gradcam_heatmap(batch, detector.model)
        self.assertIsNotNone(heatmap)
        self.assertEqual(heatmap.ndim, 2)

        overlaid = overlay_gradcam(frame, heatmap)
        self.assertEqual(overlaid.shape, frame.shape)

    def test_video_processor_smoothing(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        annotated, stats = video_processor.process_frame(frame)

        self.assertIsNotNone(annotated)
        self.assertIn("label", stats)
        self.assertIn("fps", stats)

if __name__ == "__main__":
    unittest.main()
