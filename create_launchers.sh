#!/bin/bash
# Creates desktop launcher icons on the Jetson, one per long-running service:
#   - LiZAD inference server      (port 8000)
#   - PatchCore inference server  (port 8001)
#   - ArgusVision UI              (port 7860)
#
# Each launcher pulls the latest code from ~/ArgusVision and copies it into
# ~/LiZAD before starting (so double-clicking always runs the current
# version, no manual git pull/cp/kill-old-process dance needed), opens in
# its own terminal window (kept open after exit so crashes/errors stay
# visible), activates the LiZAD conda environment without depending on an
# interactive shell's PATH, waits for the service to actually come up, and
# opens it in the default web browser.
set -e

LIZAD_DIR="$HOME/LiZAD"
DESKTOP_DIR="$HOME/Desktop"
APPS_DIR="$HOME/.local/share/applications"

if [ ! -d "$LIZAD_DIR" ]; then
    echo "ERROR: $LIZAD_DIR not found. Run setup.sh first to copy the app files there."
    exit 1
fi

mkdir -p "$DESKTOP_DIR" "$APPS_DIR"

# write_start_script <output-name> <python-file> <port> <health-path> <browser-path>
# health-path is polled to detect readiness; browser-path is what actually
# gets opened (e.g. FastAPI's /docs page rather than raw /health JSON).
# A .desktop launcher runs in a bare non-interactive shell, so PATH does not
# include conda the way an interactive terminal does (that's set up via
# ~/.bashrc, which GUI launches never source). Source conda's own profile
# script directly from its known install location instead of depending on
# `conda` already being resolvable, and fall back to `conda info --base`
# only if that fixed path doesn't exist (e.g. a different conda install).
write_start_script() {
    local out_name="$1" py_file="$2" port="$3" health_path="$4" url_path="$5"
    cat > "$LIZAD_DIR/$out_name" << EOF
#!/bin/bash
REPO_DIR="\$HOME/ArgusVision"
LIZAD_DIR="\$HOME/LiZAD"

CONDA_SH="\$HOME/miniconda3/etc/profile.d/conda.sh"
if [ ! -f "\$CONDA_SH" ]; then
    CONDA_SH="\$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
source "\$CONDA_SH"
conda activate LiZAD

echo "=== Pulling latest ArgusVision code ==="
if [ -d "\$REPO_DIR" ]; then
    # fetch + reset rather than pull: the public mirror is force-pushed as a
    # fresh squashed history on every sync (see sync_public.sh), and git pull
    # cannot merge across an unrelated history - it would fail every time and
    # silently leave stale code running. reset --hard is safe here because
    # this clone is only a code-delivery mechanism; captured images and
    # checkpoints live in \$LIZAD_DIR, not in the repo.
    if git -C "\$REPO_DIR" fetch origin && git -C "\$REPO_DIR" reset --hard origin/main; then
        # Copy everything rather than an explicit list - that list already went
        # stale once (a new script was pulled but never copied, so it appeared
        # to be "missing" on the Jetson). Data and .git aren't matched by these
        # globs, and the generated start_*.sh live only in \$LIZAD_DIR.
        cp "\$REPO_DIR"/*.py "\$REPO_DIR"/*.sh "\$LIZAD_DIR/" 2>/dev/null
    else
        echo "WARNING: code update failed - launching with whatever is already in \$LIZAD_DIR"
    fi
else
    echo "WARNING: \$REPO_DIR not found - launching with whatever code is already in \$LIZAD_DIR"
fi

echo "=== Stopping anything already running on port $port ==="
fuser -k $port/tcp 2>/dev/null || true
sleep 1

cd "\$LIZAD_DIR" || exit 1
python $py_file &
SERVICE_PID=\$!

echo "Waiting for $py_file to come up on port $port..."
for i in \$(seq 1 60); do
    curl -s -o /dev/null http://localhost:$port$health_path && break
    sleep 1
done

if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:$port$url_path" &> /dev/null &
else
    echo "xdg-open not found - open http://localhost:$port$url_path manually"
fi

wait \$SERVICE_PID
echo ""
echo "--- $py_file exited - press Enter to close ---"
read
EOF
    chmod +x "$LIZAD_DIR/$out_name"
}

# write_desktop_entry <file-basename> <display-name> <comment> <icon> <start-script>
write_desktop_entry() {
    local basename="$1" display_name="$2" comment="$3" icon="$4" start_script="$5"
    local target
    for target in "$DESKTOP_DIR" "$APPS_DIR"; do
        cat > "$target/$basename.desktop" << EOF
[Desktop Entry]
Type=Application
Name=$display_name
Comment=$comment
Exec=gnome-terminal --title="$display_name" -- "$LIZAD_DIR/$start_script"
Icon=$icon
Terminal=false
Categories=Utility;
EOF
        chmod +x "$target/$basename.desktop"
    done
}

echo "=== Writing launcher scripts ==="
write_start_script "start_server.sh"      "lizad_server.py"       8000 "/health" "/docs"
write_start_script "start_patchcore.sh"   "patchcore_server.py"   8001 "/health" "/docs"
write_start_script "start_efficientad.sh" "efficientad_server.py" 8002 "/health" "/docs"
write_start_script "start_app.sh"         "app.py"                7860 ""        ""

echo "=== Writing desktop launcher icons ==="
write_desktop_entry "argusvision-server" "ArgusVision LiZAD Server" \
    "Start the LiZAD zero-shot inference server (port 8000)" \
    "network-server" "start_server.sh"
write_desktop_entry "argusvision-patchcore" "ArgusVision PatchCore Server" \
    "Start the PatchCore reference-based inference server (port 8001)" \
    "network-server" "start_patchcore.sh"
write_desktop_entry "argusvision-efficientad" "ArgusVision EfficientAD Server" \
    "Start the EfficientAD logical-anomaly inference server (port 8002)" \
    "network-server" "start_efficientad.sh"
write_desktop_entry "argusvision-app" "ArgusVision" \
    "Start the ArgusVision capture/inspection UI (port 7860)" \
    "camera-photo" "start_app.sh"

echo "=== Marking launchers as trusted (GNOME/Nautilus) ==="
if command -v gio &> /dev/null; then
    for entry in argusvision-server argusvision-patchcore argusvision-efficientad argusvision-app; do
        gio set "$DESKTOP_DIR/$entry.desktop" "metadata::trusted" true 2>/dev/null || true
    done
fi

echo ""
echo "=== Done ==="
echo "Four icons created (Desktop and Applications menu):"
echo "  - ArgusVision LiZAD Server        -> lizad_server.py, port 8000"
echo "  - ArgusVision PatchCore Server    -> patchcore_server.py, port 8001"
echo "  - ArgusVision EfficientAD Server  -> efficientad_server.py, port 8002"
echo "  - ArgusVision                     -> app.py, opens http://localhost:7860"
echo ""
echo "Start whichever detector server(s) you want to compare, then the UI."
echo "No server is required for Live Capture or Channel Check - only for"
echo "their respective detector tabs."
echo ""
echo "NOTE: PatchCore and EfficientAD are now trained from the UI (a Train"
echo "button on each tab), not at server startup. A freshly started server"
echo "with no checkpoint will say so in the UI and wait to be trained."
echo ""
echo "Each launcher auto-updates from ~/ArgusVision and kills any prior"
echo "instance on its port before starting - future code changes just need"
echo "'git push' on the dev machine, then double-click the icon on the Jetson."
echo ""
echo "If a Desktop icon shows 'Untrusted application launcher - execute"
echo "anyway?' on first double-click, right-click it and choose"
echo "'Allow Launching' once - only needed the first time."
echo ""
echo "If double-clicking does nothing at all, gnome-terminal may not be"
echo "installed on this image - run 'which gnome-terminal' to check, and"
echo "if missing, 'sudo apt install -y gnome-terminal' or swap the Exec"
echo "line in the .desktop files for whatever terminal emulator is present."
