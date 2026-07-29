"""
Shared training-job plumbing for the model servers (patchcore_server.py,
efficientad_server.py).

Both trainable detectors need exactly the same thing: a POST /train that
returns immediately (EfficientAD trains for minutes - a synchronous request
would time out), a GET /train_status the UI can poll, and a record of what
the current checkpoint was trained on so the UI can tell the operator when
their model has gone stale relative to the images they've since captured.

That logic is identical for both detectors, so it lives here rather than
being written twice and drifting apart.
"""

import json
import os
import threading
import traceback
from datetime import datetime


def count_images(folder):
    """Number of files directly inside `folder`. 0 if it doesn't exist."""
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])


def metadata_path(checkpoint_path):
    return checkpoint_path + ".meta.json"


def write_metadata(checkpoint_path, n_images):
    """Record what this checkpoint was trained on, so a freshly-started
    server can report it without having to retrain to find out."""
    meta = {"trained_at": datetime.now().isoformat(timespec="seconds"), "n_images": n_images}
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    tmp = metadata_path(checkpoint_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, metadata_path(checkpoint_path))  # atomic; avoids torn reads
    return meta


def read_metadata(checkpoint_path):
    """Returns the metadata dict, or None if absent/unreadable. Never raises -
    a corrupt sidecar should degrade to 'unknown', not kill server startup."""
    path = metadata_path(checkpoint_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: could not read {path} ({e})")
        return None


class TrainJob:
    """Runs one training function at a time on a background thread and exposes
    a pollable status.

    Only one run is allowed at a time - a second /train while one is already
    in progress is rejected rather than queued, since two concurrent fits
    would race on the same checkpoint file.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = "idle"  # idle | training | done | error
        self._progress = 0.0
        self._message = "Not trained yet"
        self._thread = None

    def _report(self, progress, message):
        """Passed into the training function so it can publish progress."""
        with self._lock:
            self._progress = float(progress)
            self._message = message

    def set_idle_message(self, message):
        """Used at startup to describe an already-loaded checkpoint."""
        with self._lock:
            if self._state == "idle":
                self._message = message

    def mark_done(self, message):
        with self._lock:
            self._state = "done"
            self._progress = 1.0
            self._message = message

    @property
    def is_training(self):
        with self._lock:
            return self._state == "training"

    def start(self, fn):
        """fn(report) -> str  where report(progress, message) publishes progress
        and the returned string becomes the final status message.

        Returns (started: bool, message: str).
        """
        with self._lock:
            if self._state == "training":
                return False, "Training already in progress"
            self._state = "training"
            self._progress = 0.0
            self._message = "Starting..."

        def runner():
            try:
                final = fn(self._report)
                with self._lock:
                    self._state = "done"
                    self._progress = 1.0
                    self._message = final or "Training complete"
            except Exception as e:
                # Print the full traceback server-side (this is the only place
                # it would otherwise be swallowed), but hand the UI a short
                # message rather than a wall of stack frames.
                traceback.print_exc()
                with self._lock:
                    self._state = "error"
                    self._progress = 0.0
                    self._message = f"{type(e).__name__}: {e}"

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        return True, "Training started"

    def status(self, checkpoint_path=None):
        with self._lock:
            status = {
                "state": self._state,
                "progress": self._progress,
                "message": self._message,
            }
        meta = read_metadata(checkpoint_path) if checkpoint_path else None
        status["trained_at"] = meta.get("trained_at") if meta else None
        status["n_images"] = meta.get("n_images") if meta else None
        return status
