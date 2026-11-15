"""Check the Windows environment before processing project data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


CHECKS = (
    ("GUI and Qt", "import tkinter; import napari; from PyQt5 import QtWidgets"),
    ("Image readers", "import slideio; import tifffile; import czifile; "
     "import readlif.reader; import nd2; import oiffile"),
    ("Segmentation pipeline", "import cellmatchr.segmentation"),
    ("Analysis pipeline", "import cellmatchr.analysis; import openpyxl"),
    ("Retraining pipeline", "import cellmatchr.retraining"),
)


def run_check(label: str, source: str, *, timeout: int = 120) -> bool:
    environment = os.environ.copy()
    environment["QT_API"] = "pyqt5"
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", source],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=environment, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"FAIL  {label}: timed out after {timeout} seconds")
        return False
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        print(f"FAIL  {label} (exit {result.returncode})")
        for line in detail[-10:]:
            print(f"      {line}")
        return False
    print(f"PASS  {label}" + (f": {result.stdout.strip()}" if result.stdout.strip() else ""))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the CellMatchR Windows environment")
    parser.add_argument("--gui", action="store_true",
                        help="Also open and close a hidden Napari window")
    parser.add_argument("--stardist", action="store_true",
                        help="Also require the optional StarDist/TensorFlow backend")
    args = parser.parse_args()

    if sys.platform != "win32":
        print("This preflight check is intended for native Windows Python.")
        return 2
    if sys.version_info[:2] != (3, 11):
        print("CellMatchR Windows preview requires Python 3.11.")
        return 1

    print(f"Python: {sys.executable}")
    ok = True
    for label, source in CHECKS:
        ok = run_check(label, source) and ok
    ok = run_check(
        "PyTorch device",
        "import torch; print(f'{torch.__version__} | ' + "
        "(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))",
    ) and ok
    if args.gui:
        ok = run_check(
            "Napari window",
            "import napari; viewer = napari.Viewer(show=False); viewer.close()",
        ) and ok
    if args.stardist:
        ok = run_check("StarDist", "import tensorflow; import stardist") and ok

    print("Ready for Windows image tests." if ok else
          "Some checks failed; keep the output above for troubleshooting.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
