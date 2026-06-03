#!/bin/bash
# ---------------------------------------------------------------------------
# CellMatchr — double-click launcher for macOS (shows a Terminal/log window).
# Lives in  <project>/scripts/  next to run_cellmatchr.py.
# First time: right-click → Open to bypass Gatekeeper.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                     # project root
cd "$ROOT_DIR"

# ── Set CellMatchr icon on this file (runs silently every launch) ────────────
ICON_PATH="$ROOT_DIR/CellMatchr_Logo.png"
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
# in non-interactive shells.  Search common conda/anaconda layouts and, when
# available, the actual base reported by conda itself.
PYTHON_BIN=""
CONDA_BASES=()
if command -v conda >/dev/null 2>&1; then
    _b="$(conda info --base 2>/dev/null)"
    [ -n "$_b" ] && CONDA_BASES+=("$_b")
fi
[ -n "$CONDA_EXE" ] && CONDA_BASES+=("$(dirname "$(dirname "$CONDA_EXE")")")
CONDA_BASES+=(
    "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge"
    "$HOME/opt/anaconda3" "$HOME/opt/miniconda3"
    "/opt/anaconda3" "/opt/miniconda3" "/opt/miniforge3" "/opt/mambaforge"
    "/opt/homebrew/anaconda3" "/opt/homebrew/Caskroom/miniforge/base"
    "/opt/conda" "/anaconda3" "/miniconda3" "/usr/local/anaconda3"
)
for CONDA_BASE in "${CONDA_BASES[@]}"; do
    if [ -n "$CONDA_BASE" ] && [ -x "$CONDA_BASE/envs/cellmatchr/bin/python" ]; then
        PYTHON_BIN="$CONDA_BASE/envs/cellmatchr/bin/python"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo ""
    echo "ERROR: Could not find the 'cellmatchr' conda environment."
    echo ""
    echo "Create it with:"
    echo "    conda env create -f \"$ROOT_DIR/environment.yml\""
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
