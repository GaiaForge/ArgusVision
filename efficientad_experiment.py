"""
Offline EfficientAD evaluation - the gate before wiring EfficientAD into the
live UI.

EfficientAD (MVTec, WACV 2024) pairs a student-teacher network with an
autoencoder branch specifically aimed at LOGICAL anomalies - missing items,
wrong position, wrong quantity - rather than only structural/texture
defects. That is exactly the class where LiZAD failed on this project's
boards (7 missing MOSFETs scored ~0.25 vs ~0.0, a margin too thin to trust),
so it is worth measuring directly.

Read the result carefully. With a small dataset an OVERLAP result more
likely means "not enough data to tell yet" than "wrong approach" -
EfficientAD trains a real network, unlike PatchCore's memory bank, so it is
substantially hungrier for images. The runner prints which case it thinks
applies.

The reporting format is shared via experiment.py so results are directly
comparable with patchcore_experiment.py.

Run after installing anomalib (see install_anomalib.sh):
    conda activate LiZAD
    python efficientad_experiment.py
"""

from anomalib.models import EfficientAd

from experiment import run_experiment

# anomalib rejects any train_batch_size other than 1 for EfficientAD.
# max_epochs is a starting point for a small dataset, not a tuned value.
if __name__ == "__main__":
    run_experiment(
        label="EfficientAD",
        model_cls=EfficientAd,
        min_normal=20,
        datamodule_kwargs={"train_batch_size": 1},
        engine_kwargs={"max_epochs": 20},
    )
