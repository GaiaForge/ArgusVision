"""
Shared FastAPI implementation for the trainable anomalib detectors.

PatchCore and EfficientAD need identical serving behaviour - load a
checkpoint at startup, score frames on /infer, retrain on /train from the
UI, report progress on /train_status - and differ only in model class,
checkpoint path, port, and a couple of fit parameters. Writing it once
means there is one live path to debug rather than two, which matters
because this path (anomalib's PredictDataset + Engine.predict per request)
has not yet been validated on real hardware.

NOTE: the anomalib Folder/Engine/PredictDataset calls here are a best-effort
reading of anomalib 2.x's API, validated so far only via the offline
experiment scripts. Expect to debug on first run, same as every other new
library integration in this project.
"""

import base64
import contextlib
import os
import tempfile

import numpy as np
import cv2
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from anomalib.data import PredictDataset
from anomalib.engine import Engine

from detector_data import build_datamodule, normal_path
from train_service import TrainJob, count_images, read_metadata, write_metadata


class _EpochProgress:
    """Lightning callback that publishes epoch progress to a TrainJob.

    Defined lazily inside a factory because importing lightning at module
    scope would pull it into every consumer; anomalib already depends on it
    server-side, but the import cost is real on a Jetson.
    """

    @staticmethod
    def make(report, lo=0.3, hi=0.85):
        from lightning.pytorch.callbacks import Callback

        class Progress(Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                total = trainer.max_epochs or 1
                done = trainer.current_epoch + 1
                report(lo + (hi - lo) * (done / total), f"Training epoch {done}/{total}...")

        return Progress()


@contextlib.contextmanager
def trusted_torch_load():
    """Load checkpoints with torch.load's pre-2.6 behaviour.

    PyTorch 2.6 changed torch.load's `weights_only` default from False to
    True. Lightning checkpoints contain anomalib enums (anomalib.PrecisionType
    and friends) that the restricted unpickler refuses, so loading fails with
    "Weights only load failed / Unsupported global".

    torch.serialization.add_safe_globals() is the officially suggested fix,
    but it means allowlisting each rejected class in turn - fix one and the
    next surfaces. These checkpoints are written by this same server, on this
    machine, from POST /train, so the provenance concern that motivated the
    default change doesn't apply: load them unrestricted instead.

    Scoped to the startup load deliberately - this patches a global, and the
    narrower the window the better.
    """
    original = torch.load

    def patched(*args, **kwargs):
        # Force, don't setdefault: Lightning passes weights_only=True
        # explicitly, so a default would never be consulted.
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


class InferRequest(BaseModel):
    image_b64: str  # base64-encoded JPEG bytes


class InferResponse(BaseModel):
    score: float


def build_app(
    *,
    label,
    model_cls,
    checkpoint_path,
    min_train_images,
    datamodule_kwargs=None,
    engine_kwargs=None,
    show_epoch_progress=False,
):
    """Returns a configured FastAPI app for one detector."""
    app = FastAPI()
    state = {"model": None}
    job = TrainJob()
    datamodule_kwargs = datamodule_kwargs or {}
    engine_kwargs = engine_kwargs or {}

    def do_train(report):
        n_images = count_images(normal_path())
        if n_images < min_train_images:
            raise ValueError(
                f"Only {n_images} normal images in {normal_path()} - {label} needs at least "
                f"{min_train_images}. Capture more in the Live Capture tab first."
            )

        report(0.1, f"Loading {n_images} normal images...")
        datamodule = build_datamodule(f"{label.lower()}_live", **datamodule_kwargs)

        report(0.3, f"Training {label}...")
        model = model_cls()
        callbacks = [_EpochProgress.make(report)] if show_epoch_progress else []
        engine = Engine(callbacks=callbacks, **engine_kwargs)
        engine.fit(datamodule=datamodule, model=model)

        report(0.9, "Saving checkpoint...")
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        engine.trainer.save_checkpoint(checkpoint_path)
        write_metadata(checkpoint_path, n_images)

        state["model"] = model
        return f"Trained on {n_images} normal images"

    def load_checkpoint_at_startup():
        if not os.path.isfile(checkpoint_path):
            job.set_idle_message(f"No model trained yet - click Train to build the {label} model")
            print(f"No checkpoint at {checkpoint_path} - use POST /train to build one.")
            return
        try:
            with trusted_torch_load():
                state["model"] = model_cls.load_from_checkpoint(checkpoint_path)
            meta = read_metadata(checkpoint_path)
            if meta:
                job.set_idle_message(
                    f"Trained on {meta['n_images']} normal images at {meta['trained_at']}"
                )
            else:
                job.set_idle_message("Checkpoint loaded (no training metadata found)")
            print(f"Loaded {label} checkpoint from {checkpoint_path}")
        except Exception as e:
            # A bad checkpoint must not stop the server booting - the UI can
            # still reach it and trigger a retrain, which is the actual fix.
            # Include the first line of the message, not just the exception
            # type: "UnpicklingError" alone sent us chasing the wrong cause.
            first_line = next((ln for ln in str(e).splitlines() if ln.strip()), "")
            job.set_idle_message(
                f"Could not load checkpoint - {type(e).__name__}: {first_line[:140]}"
            )
            print(f"WARNING: failed to load {checkpoint_path}: {e}")

    load_checkpoint_at_startup()

    @app.post("/infer", response_model=InferResponse)
    def infer(req: InferRequest):
        model = state["model"]
        if model is None:
            raise HTTPException(status_code=409, detail=f"No trained model - train {label} first")

        jpg_bytes = base64.b64decode(req.image_b64)
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "frame.jpg")
            cv2.imwrite(tmp_path, frame_bgr)
            dataset = PredictDataset(path=tmp_path)
            predictions = Engine().predict(model=model, dataset=dataset)

        score = 0.0
        for batch in predictions or []:
            if len(batch.pred_score) > 0:
                score = float(batch.pred_score[0])
                break
        return InferResponse(score=score)

    @app.post("/evaluate")
    def evaluate():
        """Score the captured dataset with the CURRENTLY LOADED model and
        report the normal/defect gap.

        Deliberately evaluates the deployed model rather than refitting (which
        is what the offline experiment scripts do) - the question this answers
        is "how well does the model I'm actually running separate this defect",
        which is what a threshold gets set from.
        """
        model = state["model"]
        if model is None:
            raise HTTPException(status_code=409, detail=f"No trained model - train {label} first")

        datamodule = build_datamodule(f"{label.lower()}_eval", **datamodule_kwargs)
        results = Engine().predict(model=model, datamodule=datamodule)

        normal_scores, defect_scores = [], []
        for batch in results or []:
            for gt_label, score in zip(batch.gt_label, batch.pred_score):
                (defect_scores if gt_label else normal_scores).append(float(score))

        if not normal_scores or not defect_scores:
            return {
                "ok": False,
                "message": (
                    f"Evaluation returned {len(normal_scores)} normal and "
                    f"{len(defect_scores)} defect scores - need both. Capture "
                    "images in each category first."
                ),
            }

        max_normal = max(normal_scores)
        min_defect = min(defect_scores)
        return {
            "ok": True,
            "max_normal": max_normal,
            "min_defect": min_defect,
            "gap": min_defect - max_normal,
            "n_normal": len(normal_scores),
            "n_defect": len(defect_scores),
            "suggested_threshold": round((max_normal + min_defect) / 2, 3),
        }

    @app.post("/train")
    def train():
        started, message = job.start(do_train)
        return {"started": started, "message": message}

    @app.get("/train_status")
    def train_status():
        status = job.status(checkpoint_path)
        # The UI needs the live count to spot a stale model; only the server
        # can see the data directory.
        status["current_n_images"] = count_images(normal_path())
        status["has_model"] = state["model"] is not None
        return status

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
