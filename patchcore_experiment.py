"""
Offline PatchCore evaluation - does PatchCore separate our captured normal
images from our captured defect images, and by how much?

PatchCore builds a memory bank directly from real captured normal images and
flags anything statistically different from it - no defect examples needed to
build the bank, and no reliance on a pretrained model's generic idea of
"damage" recognising an absence as anomalous (which is where LiZAD's
zero-shot scoring proved weak on this project's boards).

This is the offline counterpart to patchcore_server.py, which serves the
same model live. Use this to decide whether a detector is worth using at
all; use the server (and the UI's Train button) for day-to-day work.

The reporting format is shared via experiment.py so results are directly
comparable with efficientad_experiment.py.

Run after installing anomalib (see install_anomalib.sh):
    conda activate LiZAD
    python patchcore_experiment.py
"""

from anomalib.models import Patchcore

from experiment import run_experiment

if __name__ == "__main__":
    run_experiment(label="PatchCore", model_cls=Patchcore, min_normal=5)
