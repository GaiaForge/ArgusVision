#!/bin/bash
# Consolidated LiZAD environment rebuild - Python 3.10 (the only version the
# Jetson-specific PyTorch index actually publishes torch wheels for).
#
# IMPORTANT: this must be SOURCED, not executed, since it runs `conda create`
# and `conda activate` and needs those changes to persist in your actual
# shell afterward - not just inside a throwaway subprocess.
#   Run it as:  source full_setup.sh
#   NOT as:     bash full_setup.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: this script must be SOURCED, not executed directly."
    echo "Run it as: source full_setup.sh"
    return 1 2>/dev/null || exit 1
fi

set -e

echo "=== Removing old LiZAD environment (wrong Python version for this hardware) ==="
conda deactivate 2>/dev/null || true
conda env remove -n LiZAD -y 2>/dev/null || echo "(no existing LiZAD env to remove)"

echo ""
echo "=== Creating LiZAD environment with Python 3.10 ==="
conda create -n LiZAD python=3.10 -y
conda activate LiZAD

echo ""
echo "=== Installing Jetson-targeted PyTorch (jp6/cu129, cp310) ==="
pip install torch torchvision torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu129

echo ""
echo "=== Installing CUDA 12 runtime libraries ==="
CUDA_PACKAGES=(
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
for pkg in "${CUDA_PACKAGES[@]}"; do
    echo "--- $pkg ---"
    pip install "$pkg" || echo "WARNING: $pkg failed to install - continuing"
done

echo ""
echo "=== Installing NVPL (CPU-side LAPACK) ==="
pip install nvpl-lapack || echo "WARNING: nvpl-lapack failed to install - continuing"

echo ""
echo "=== Installing libopenblas (system package) ==="
sudo apt install -y libopenblas0 || echo "WARNING: libopenblas0 not found - run 'apt-cache search openblas' manually"

echo ""
echo "=== Writing dynamic LD_LIBRARY_PATH activation hook ==="
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat > "$ACTIVATE_DIR/env_vars.sh" << 'EOF'
_NVIDIA_LIBS=$(find "$CONDA_PREFIX"/lib/python3.*/site-packages/nvidia -maxdepth 2 -type d -name "lib" 2>/dev/null | tr '\n' ':')
_NVPL_LIBS=$(find "$CONDA_PREFIX"/lib/python3.*/site-packages/nvpl -maxdepth 2 -type d -name "lib" 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${_NVIDIA_LIBS}${_NVPL_LIBS}${LD_LIBRARY_PATH}"
unset _NVIDIA_LIBS _NVPL_LIBS
EOF
source "$ACTIVATE_DIR/env_vars.sh"

echo ""
echo "=== Installing mobileclip (from the already-cloned repo if present) ==="
if [ -d "$HOME/LiZAD/ml-mobileclip" ]; then
    pip install "$HOME/LiZAD/ml-mobileclip"
else
    git clone https://github.com/apple/ml-mobileclip.git "$HOME/LiZAD/ml-mobileclip"
    pip install "$HOME/LiZAD/ml-mobileclip"
fi

echo ""
echo "=== Installing transformers, open_clip_torch ==="
pip install transformers open_clip_torch

echo ""
echo "=== Installing UI/server dependencies ==="
pip install gradio fastapi uvicorn requests opencv-python numpy

echo ""
echo "=== Installing Arena SDK Python wheel ==="
ARENA_WHEEL=$(find "$HOME/Downloads" -iname "arena_api*.whl" 2>/dev/null | head -n 1)
if [ -n "$ARENA_WHEEL" ]; then
    pip install "$ARENA_WHEEL"
else
    echo "WARNING: no arena_api wheel found under ~/Downloads - install manually:"
    echo "  pip install /path/to/arena_api-*.whl"
fi

echo ""
echo "=== Verifying CUDA ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device capability:', torch.cuda.get_device_capability())"

echo ""
echo "=== Setup complete ==="
echo "Environment 'LiZAD' (Python 3.10) is active in this shell right now."
echo "To run: cd ~/LiZAD && python lizad_server.py   (separate terminal: python app.py)"
