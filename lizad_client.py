"""
Lightweight HTTP client for lizad_server.py - deliberately has no torch/ML
dependencies, just requests/numpy/cv2, so app.py stays light even though the
actual model lives in a separate process.
"""

import base64

import numpy as np
import cv2
import requests


class LiZADClient:
    def __init__(self, host="localhost", port=8000):
        self.base_url = f"http://{host}:{port}"

    def health(self, timeout=2.0):
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=timeout)
            return resp.ok
        except requests.exceptions.RequestException:
            return False

    def run(self, frame_rgb, timeout=5.0):
        """Same interface as LiZADEngine.run() - returns (anomaly_map_2d, score)."""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _, jpg_bytes = cv2.imencode(".jpg", frame_bgr)
        image_b64 = base64.b64encode(jpg_bytes.tobytes()).decode("utf-8")

        resp = requests.post(f"{self.base_url}/infer", json={"image_b64": image_b64}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        png_bytes = base64.b64decode(data["anomaly_map_b64"])
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        normalized = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        anomaly_map = normalized.astype(np.float32) / 255.0

        return anomaly_map, data["score"]
