"""
PatchCore inference server (port 8001) - reference-image anomaly detection.

Builds a "memory bank" of features from captured normal images and flags
anything that departs from it. Strong on structural defects; measured
cleanly separating missing components where LiZAD's zero-shot scoring was
marginal.

The serving/training implementation is shared with EfficientAD in
detector_server.py - see there for the API and the known-unvalidated parts.
Training is triggered from the UI (POST /train), not at startup.
"""

import uvicorn
from anomalib.models import Patchcore

from detector_server import build_app

CHECKPOINT_PATH = "checkpoints/patchcore/model.ckpt"

# The memory bank works with very few images, but below this the result is
# meaningless rather than merely rough.
MIN_TRAIN_IMAGES = 5

app = build_app(
    label="PatchCore",
    model_cls=Patchcore,
    checkpoint_path=CHECKPOINT_PATH,
    min_train_images=MIN_TRAIN_IMAGES,
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
