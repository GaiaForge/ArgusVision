"""
EfficientAD inference server (port 8002) - reference-image anomaly detection
aimed at LOGICAL anomalies.

EfficientAD (MVTec, WACV 2024) pairs a student-teacher network with an
autoencoder branch, and that second branch exists specifically to catch
logical anomalies - missing items, wrong position, wrong quantity - as
opposed to purely structural/texture defects. That is precisely the class
where LiZAD's zero-shot scoring proved weak on this project's boards, which
is why it's worth evaluating here.

Trains on normal images only (no defect labelling) and runs at
millisecond-level latency once trained.

The serving/training implementation is shared with PatchCore in
detector_server.py. Two differences are configured below:

- train_batch_size=1: anomalib raises a validation error for anything else.
- max_epochs: EfficientAD trains a real network rather than fitting a memory
  bank, so it needs actual epochs. 20 is a starting point for a small
  dataset, not a tuned value - raise it once there are more images.
"""

import uvicorn
from anomalib.models import EfficientAd

from detector_server import build_app

CHECKPOINT_PATH = "checkpoints/efficientad/model.ckpt"

# Higher than PatchCore's floor: this trains a network rather than fitting a
# memory bank, so it needs meaningfully more data before the result means
# anything. Even at this threshold expect "inconclusive" rather than a verdict.
MIN_TRAIN_IMAGES = 20

MAX_EPOCHS = 20

app = build_app(
    label="EfficientAD",
    model_cls=EfficientAd,
    checkpoint_path=CHECKPOINT_PATH,
    min_train_images=MIN_TRAIN_IMAGES,
    datamodule_kwargs={"train_batch_size": 1},
    engine_kwargs={"max_epochs": MAX_EPOCHS},
    show_epoch_progress=True,
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
