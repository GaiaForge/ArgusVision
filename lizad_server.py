"""
Small FastAPI service wrapping LiZADEngine - keeps the heavy PyTorch/model
loading in its own long-running process, separate from the Gradio UI process.
Run this once on the Jetson; app.py talks to it via LiZADClient (lizad_client.py).
"""

import base64

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from lizad_engine import LiZADEngine

CHECKPOINT_PATH = "checkpoints/trained_on_visa/model.pth"

app = FastAPI()
engine = LiZADEngine(CHECKPOINT_PATH, class_name="pcb")


class InferRequest(BaseModel):
    image_b64: str  # base64-encoded JPEG bytes


class InferResponse(BaseModel):
    anomaly_map_b64: str  # base64-encoded PNG, anomaly map scaled to 0-255 uint8
    score: float


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    jpg_bytes = base64.b64decode(req.image_b64)
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    anomaly_map, score = engine.run(frame_rgb)

    normalized = np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8)
    _, png_bytes = cv2.imencode(".png", normalized)
    map_b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

    return InferResponse(anomaly_map_b64=map_b64, score=score)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
