#!/bin/bash
# Run this from inside the cloned ArgusVision repo directory on the Jetson.
# Copies the app files into ~/LiZAD and installs everything needed into the
# existing 'LiZAD' conda environment.
set -e

LIZAD_DIR="$HOME/LiZAD"
ARENA_WHEEL=$(ls "$HOME"/Downloads/ARENA_API-*.whl 2>/dev/null | head -n 1)

echo "=== ArgusVision setup ==="

if [ ! -d "$LIZAD_DIR" ]; then
    echo "ERROR: $LIZAD_DIR not found."
    echo "This script expects the LiZAD repo (github.com/intelligolabs/LiZAD) to"
    echo "already be cloned and set up there, from earlier in this project."
    exit 1
fi

echo "Copying app.py, lizad_engine.py, lizad_server.py, lizad_client.py into $LIZAD_DIR ..."
cp app.py lizad_engine.py lizad_server.py lizad_client.py "$LIZAD_DIR/"

echo "Activating conda environment 'LiZAD' ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate LiZAD

echo "Installing UI/server dependencies (gradio, fastapi, uvicorn, requests) ..."
pip install gradio fastapi uvicorn requests

if [ -n "$ARENA_WHEEL" ]; then
    echo "Installing Arena SDK wheel: $ARENA_WHEEL"
    pip install "$ARENA_WHEEL"
else
    echo ""
    echo "WARNING: no Arena SDK wheel found in ~/Downloads/ARENA_API-*.whl"
    echo "Install it manually once you locate it:"
    echo "  conda activate LiZAD && pip install /path/to/ARENA_API-*.whl"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run (two separate terminals):"
echo "  Terminal 1: cd $LIZAD_DIR && conda activate LiZAD && python lizad_server.py"
echo "  Terminal 2: cd $LIZAD_DIR && conda activate LiZAD && python app.py"
echo ""
echo "Then open http://<jetson-ip>:7860 in a browser."
