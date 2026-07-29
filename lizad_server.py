"""
Small FastAPI service wrapping LiZADEngine - keeps the heavy PyTorch/model
loading in its own long-running process, separate from the Gradio UI process.
Run this once on the Jetson; app.py talks to it via LiZADClient (lizad_client.py).
"""

import base64
import os
from typing import Dict

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from lizad_engine import LiZADEngine

# Override with LIZAD_CHECKPOINT=checkpoints/trained_on_mvtec/model.pth for an
# A/B comparison against the other checkpoint - no code change needed.
CHECKPOINT_PATH = os.environ.get("LIZAD_CHECKPOINT", "checkpoints/trained_on_visa/model.pth")

app = FastAPI()
engine = LiZADEngine(CHECKPOINT_PATH, class_name="pcb")


class InferRequest(BaseModel):
    image_b64: str  # base64-encoded JPEG bytes


class InferResponse(BaseModel):
    anomaly_map_b64: str  # base64-encoded PNG, anomaly map scaled to 0-255 uint8
    score: float


class MultiInferResponse(BaseModel):
    scores: Dict[str, float]  # prompt-set label -> score
    best_label: str  # label with the highest score
    anomaly_map_b64: str  # anomaly map for best_label only, to avoid N heatmaps per call


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    frame_rgb = _decode_frame(req.image_b64)
    anomaly_map, score = engine.run(frame_rgb)
    return InferResponse(anomaly_map_b64=_encode_map(anomaly_map), score=score)


@app.post("/infer_multi", response_model=MultiInferResponse)
def infer_multi(req: InferRequest):
    frame_rgb = _decode_frame(req.image_b64)
    results = engine.run_multi(frame_rgb)

    scores = {label: score for label, (_, score) in results.items()}
    best_label = max(scores, key=scores.get)
    best_map = results[best_label][0]

    return MultiInferResponse(scores=scores, best_label=best_label, anomaly_map_b64=_encode_map(best_map))


def _decode_frame(image_b64):
    jpg_bytes = base64.b64decode(image_b64)
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _encode_map(anomaly_map):
    normalized = np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8)
    _, png_bytes = cv2.imencode(".png", normalized)
    return base64.b64encode(png_bytes.tobytes()).decode("utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
