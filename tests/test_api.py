"""
Integration Tests for Flask REST API Endpoints
"""

import sys
import os
import io
import unittest
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app

class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "2.0.0")

    def test_predict_image_api(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        img_bytes = io.BytesIO(buf.tobytes())

        data = {
            "file": (img_bytes, "test_image.jpg")
        }

        response = self.client.post(
            "/api/v1/predict/image",
            data=data,
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()
        self.assertEqual(res_data["status"], "success")
        self.assertIn("result", res_data)
        self.assertIn("gradcam_url", res_data["result"])

    def test_history_endpoint(self):
        response = self.client.get("/api/v1/history?label=all")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["history"], list)

if __name__ == "__main__":
    unittest.main()
