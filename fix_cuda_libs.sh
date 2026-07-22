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
)

for pkg in "${PACKAGES[@]}"; do
    echo "--- $pkg ---"
    pip install "$pkg" || echo "WARNING: $pkg failed to install (may not have an ARM64 wheel) - continuing"
done

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"

cat > "$ACTIVATE_DIR/env_vars.sh" << 'EOF'
# Dynamically find every nvidia/*/lib and nvpl/lib folder in this environment's
# site-packages, rather than hardcoding specific library paths that change
# every time a new CUDA library gets added to fix a missing-.so error.
_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
if [ -n "$_SITE_PACKAGES" ]; then
    _NVIDIA_LIBS=$(find "$_SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name "lib" 2>/dev/null | tr '\n' ':')
    _NVPL_LIBS=$(find "$_SITE_PACKAGES/nvpl" -maxdepth 2 -type d -name "lib" 2>/dev/null | tr '\n' ':')
    export LD_LIBRARY_PATH="${_NVIDIA_LIBS}${_NVPL_LIBS}${LD_LIBRARY_PATH}"
fi
unset _SITE_PACKAGES _NVIDIA_LIBS _NVPL_LIBS
EOF

echo ""
echo "=== Applying to current session ==="
source "$ACTIVATE_DIR/env_vars.sh"

echo ""
echo "=== Verifying ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device capability:', torch.cuda.get_device_capability())"
