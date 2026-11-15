#!/usr/bin/env python3
"""
CellMatchr entry point.

Run from the project root:
    python scripts/run_cellmatchr.py

Or, after `pip install -e .`:
    cellmatchr
"""
import os
import sys

if sys.platform == "win32":
    # The environment also contains Qt 6 libraries; keep Napari on PyQt5.
    os.environ.setdefault("QT_API", "pyqt5")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Make the project root importable no matter where this script is launched from
# (it lives in scripts/, one level below the cellmatchr package).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cellmatchr.launcher import main

if __name__ == "__main__":
    main()
