#!/bin/bash
# Installs anomalib WITHOUT letting it touch the already-working Jetson-
# specific torch build. anomalib's own pip extras ([cu126]/[cu130]) pull in
# generic x86/SBSA-targeted torch wheels - installing those would very likely
# repeat the exact sm_87-vs-SBSA architecture mismatch this project already
# spent days fixing (see full_setup.sh). Instead: install anomalib with
# --no-deps, then manually install its other (non-torch) dependencies,
# leaving the existing jp6/cu129 torch install untouched.
set -e

echo "=== Installing anomalib without its bundled torch/torchvision ==="
pip install anomalib --no-deps

echo ""
echo "=== Installing anomalib's other dependencies (torch/torchvision excluded on purpose) ==="
# First pass installed 'pytorch-lightning' by mistake - anomalib 2.6.0 actually
# requires the newer 'lightning' meta-package instead, plus several deps pip's
# resolver flagged as genuinely missing (not just version-mismatched):
# freia, imagecodecs, omegaconf, platformdirs, rich-argparse, and a
# specific jsonargparse version range. Deliberately NOT installing
# opencv-python-headless here even though anomalib lists it - this project
# already has plain opencv-python installed (used by app.py's camera
# pipeline), and installing the headless variant alongside it risks
# conflicts over the same 'cv2' module name for no benefit we need yet.
pip install lightning omegaconf platformdirs rich-argparse freia imagecodecs "jsonargparse[signatures]>=4.27.7,<4.47.0" torchmetrics scikit-learn scikit-image pillow matplotlib einops kornia timm rich docstring-parser

echo ""
echo "=== Verifying the existing torch/CUDA install was not touched ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device capability:', torch.cuda.get_device_capability())"

echo ""
echo "=== Verifying anomalib imports (including what patchcore_experiment.py actually uses) ==="
python -c "
import anomalib
print('anomalib version:', anomalib.__version__)
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
print('Folder, Engine, Patchcore all imported OK')
"

echo ""
echo "=== Done ==="
echo "If the CUDA check above still shows 'True' and (8, 7), the torch install"
echo "is intact. If anomalib failed to import, check the error above - some of"
echo "its dependencies may not have ARM64/aarch64 wheels and could need a"
echo "different install path. This has not been tested on Jetson/ARM64 before,"
echo "so expect to debug on first run, same as every other new library in"
echo "this project so far."
