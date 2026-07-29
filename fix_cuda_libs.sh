#!/bin/bash
# Run this from inside the 'LiZAD' conda environment on the Jetson.
# Installs the CUDA 12 runtime library packages PyTorch's jp6/cu129 build needs,
# and rewrites the conda activate.d script to dynamically find every
# nvidia/*/lib and nvpl/lib folder, rather than hardcoding specific paths that
# change every time a new library gets added.
set -e

echo "=== Installing CUDA 12 runtime libraries ==="

PACKAGES=(
    nvidia-cuda-runtime-cu12
    nvidia-cublas-cu12
    nvidia-cudnn-cu12
    nvidia-cufft-cu12
    nvidia-curand-cu12
    nvidia-cusolver-cu12
    nvidia-cusparse-cu12
    nvidia-nvtx-cu12
    nvidia-nccl-cu12
    nvidia-cuda-cupti-cu12
)

for pkg in "${PACKAGES[@]}"; do
    echo "--- $pkg ---"
    pip install "$pkg" || echo "WARNING: $pkg failed to install (may not have an ARM64 wheel) - continuing"
done

echo ""
echo "=== Installing libopenblas (system package, CPU-side linear algebra PyTorch also needs) ==="
sudo apt install -y libopenblas0 || echo "WARNING: libopenblas0 not found - run 'apt-cache search openblas' to find the right package name for this Ubuntu version"

ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"

# Uses $CONDA_PREFIX directly (guaranteed set by conda during activation)
# rather than shelling out to `python -c "import site; ..."`, which was found
# to fail silently when run from inside an activate.d hook (timing/PATH issue
# specific to that execution context, even though it works fine interactively).
cat > "$ACTIVATE_DIR/env_vars.sh" << 'EOF'
_NVIDIA_LIBS=$(find "$CONDA_PREFIX"/lib/python3.*/site-packages/nvidia -maxdepth 2 -type d -name "lib" 2>/dev/null | tr '\n' ':')
_NVPL_LIBS=$(find "$CONDA_PREFIX"/lib/python3.*/site-packages/nvpl -maxdepth 2 -type d -name "lib" 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${_NVIDIA_LIBS}${_NVPL_LIBS}${LD_LIBRARY_PATH}"
unset _NVIDIA_LIBS _NVPL_LIBS
EOF

echo ""
echo "=== Applying to current session ==="
source "$ACTIVATE_DIR/env_vars.sh"

echo ""
echo "=== Verifying ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device capability:', torch.cuda.get_device_capability())"
