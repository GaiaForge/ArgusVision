"""
Shared HTTP client base for the trainable anomaly detectors (PatchCore,
EfficientAD).

Both expose the same server API - /health, /infer, /train, /train_status -
and differ only in which port they listen on, so the client is written once
here and subclassed with a default port. Deliberately has no torch/anomalib
dependency: the Gradio UI process imports these and must stay light.

LiZAD deliberately does NOT share this base - it returns an anomaly map
alongside the score and has nothing to train, so its client has a genuinely
different interface.
"""

import base64

import cv2
import requests


class AnomalyDetectorClient:
    DEFAULT_PORT = None  # subclasses set this

    def __init__(self, host="localhost", port=None):
        self.base_url = f"http://{host}:{port or self.DEFAULT_PORT}"

    def health(self, timeout=2.0):
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=timeout)
            return resp.ok
        except requests.exceptions.RequestException:
            return False

    def run(self, frame_rgb, timeout=30.0):
        """Returns a single anomaly score. The timeout is generous compared to
        LiZAD's - these predict paths spin up a Lightning trainer loop per
        call, which is slower than a plain forward pass."""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _, jpg_bytes = cv2.imencode(".jpg", frame_bgr)
        image_b64 = base64.b64encode(jpg_bytes.tobytes()).decode("utf-8")

        resp = requests.post(f"{self.base_url}/infer", json={"image_b64": image_b64}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["score"]

    def train(self, timeout=10.0):
        """Kicks off training and returns immediately - the server runs it on a
        background thread. Poll train_status() for progress."""
        resp = requests.post(f"{self.base_url}/train", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def evaluate(self, timeout=300.0):
        """Score the captured dataset and return the normal/defect gap.
        Long timeout: this runs a predict pass over the whole test split."""
        resp = requests.post(f"{self.base_url}/evaluate", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def train_status(self, timeout=5.0):
        """Returns the status dict, or None if the server is unreachable.
        Returning None rather than raising keeps the UI's polling loop simple -
        a stopped server is an expected state, not an error."""
        try:
            resp = requests.get(f"{self.base_url}/train_status", timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            return None
