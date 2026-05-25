#!/bin/bash
# ---------------------------------------------------------------------------
# CellMatchr — double-click launcher for macOS
# Stays in the CellMatchr root folder (next to run_cellmatchr.py).
# First time: right-click → Open to bypass Gatekeeper.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Set CellMatchr icon on this file (runs silently every launch) ────────────
ICON_PATH="$SCRIPT_DIR/CellMatchr_Logo.png"
LAUNCHER_PATH="$SCRIPT_DIR/launch_cellmatchr.command"
if [ -f "$ICON_PATH" ]; then
    /usr/bin/osascript 2>/dev/null -e "
        use framework \"AppKit\"
        set img to current application's NSImage's alloc()'s initWithContentsOfFile:\"$ICON_PATH\"
        current application's NSWorkspace's sharedWorkspace()'s setIcon:img forFile:\"$LAUNCHER_PATH\" options:0
    " || true
fi

# ── Find the cellmatchr env Python directly (no conda activate needed) ───────
# Using the env's Python binary directly is more reliable than conda activate
# in non-interactive shells.
PYTHON_BIN=""
for CONDA_BASE in \
    "$HOME/miniforge3" \
    "$HOME/miniconda3" \
    "$HOME/anaconda3" \
    "/opt/homebrew/Caskroom/miniforge/base" \
    "/opt/miniconda3" \
    "/opt/miniforge3" \
    "/opt/conda"; do
    if [ -f "$CONDA_BASE/envs/cellmatchr/bin/python" ]; then
        PYTHON_BIN="$CONDA_BASE/envs/cellmatchr/bin/python"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo ""
    echo "ERROR: Could not find the 'cellmatchr' conda environment."
    echo ""
    echo "Create it with:"
    echo "    conda env create -f \"$SCRIPT_DIR/environment.yml\""
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

echo "Starting CellMatchr..."
"$PYTHON_BIN" "$SCRIPT_DIR/run_cellmatchr.py"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "CellMatchr exited with error code $EXIT_CODE."
    read -p "Press Enter to close..."
fi
