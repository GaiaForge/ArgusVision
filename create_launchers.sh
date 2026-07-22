#!/bin/bash
# Creates two desktop launcher icons on the Jetson: one to start the LiZAD
# FastAPI inference server, one to start the ArgusVision Gradio app. Each
# opens in its own terminal window (kept open after exit so crashes/errors
# stay visible) rather than running silently in the background.
set -e

LIZAD_DIR="$HOME/LiZAD"
DESKTOP_DIR="$HOME/Desktop"
APPS_DIR="$HOME/.local/share/applications"

if [ ! -d "$LIZAD_DIR" ]; then
    echo "ERROR: $LIZAD_DIR not found. Run setup.sh first to copy the app files there."
    exit 1
fi

mkdir -p "$DESKTOP_DIR" "$APPS_DIR"

echo "=== Writing launcher scripts ==="
cat > "$LIZAD_DIR/start_server.sh" << 'EOF'
#!/bin/bash
cd "$HOME/LiZAD" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate LiZAD
python lizad_server.py
echo ""
echo "--- lizad_server.py exited - press Enter to close ---"
read
EOF
chmod +x "$LIZAD_DIR/start_server.sh"

cat > "$LIZAD_DIR/start_app.sh" << 'EOF'
#!/bin/bash
cd "$HOME/LiZAD" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate LiZAD
python app.py
echo ""
echo "--- app.py exited - press Enter to close ---"
read
EOF
chmod +x "$LIZAD_DIR/start_app.sh"

echo "=== Writing desktop launcher icons ==="
for TARGET_DIR in "$DESKTOP_DIR" "$APPS_DIR"; do
    cat > "$TARGET_DIR/argusvision-server.desktop" << EOF
[Desktop Entry]
Type=Application
Name=ArgusVision Server
Comment=Start the LiZAD inference FastAPI server (port 8000)
Exec=gnome-terminal --title="ArgusVision Server" -- "$LIZAD_DIR/start_server.sh"
Icon=network-server
Terminal=false
Categories=Utility;
EOF

    cat > "$TARGET_DIR/argusvision-app.desktop" << EOF
[Desktop Entry]
Type=Application
Name=ArgusVision
Comment=Start the ArgusVision capture/inspection UI (port 7860)
Exec=gnome-terminal --title="ArgusVision" -- "$LIZAD_DIR/start_app.sh"
Icon=camera-photo
Terminal=false
Categories=Utility;
EOF

    chmod +x "$TARGET_DIR/argusvision-server.desktop" "$TARGET_DIR/argusvision-app.desktop"
done

echo "=== Marking launchers as trusted (GNOME/Nautilus) ==="
if command -v gio &> /dev/null; then
    gio set "$DESKTOP_DIR/argusvision-server.desktop" "metadata::trusted" true 2>/dev/null || true
    gio set "$DESKTOP_DIR/argusvision-app.desktop" "metadata::trusted" true 2>/dev/null || true
fi

echo ""
echo "=== Done ==="
echo "Two icons created (Desktop and Applications menu):"
echo "  - ArgusVision Server  -> starts lizad_server.py in its own terminal"
echo "  - ArgusVision         -> starts app.py in its own terminal"
echo ""
echo "If a Desktop icon shows 'Untrusted application launcher - execute"
echo "anyway?' on first double-click, right-click it and choose"
echo "'Allow Launching' once - only needed the first time."
echo ""
echo "If double-clicking does nothing at all, gnome-terminal may not be"
echo "installed on this image - run 'which gnome-terminal' to check, and"
echo "if missing, 'sudo apt install -y gnome-terminal' or swap the Exec"
echo "line in the .desktop files for whatever terminal emulator is present."
