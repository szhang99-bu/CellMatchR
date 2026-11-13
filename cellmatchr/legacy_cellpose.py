"""Prepare Cellpose 3 models without changing CellMatchR's Cellpose 4 install.

The cyto2 and published rat-Nissl weights use the older Cellpose architecture.
This small, shared runtime is created only when one of those models is selected.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen


LEGACY_VERSION = "3.1.1.2"
MODEL_FILES = {
    "cyto2": (
        "cyto2torch_0",
        "https://www.cellpose.org/models/cyto2torch_0",
        "0c2ff54f8cf38aeccceb119b72ae82285bbda2efb15cc493dde43de917d95d81",
    ),
    "rat_nissl_v2": (
        "rat_nissl_v2",
        "https://raw.githubusercontent.com/BlueBrain/layer-recognition/main/cellpose_model/"
        "cellpose_residual_on_style_on_concatenation_off_train_2022_01_11_16_14_20.764792",
        "b314d3c37b3df2340eee70792c4e566800a6bf88ee49453600bbea7a4c7fc16e",
    ),
}


def runtime_home() -> Path:
    override = os.environ.get("CELLMATCHR_LEGACY_CELLPOSE_HOME")
    return Path(override).expanduser() if override else Path.home() / "CellMatchR" / "Legacy_Cellpose"


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _python_path(home: Path) -> Path:
    return home / "runtime" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_runtime(variant: str) -> tuple[str, str]:
    """Return isolated Python and verified weights for a legacy model."""
    if variant not in MODEL_FILES:
        raise ValueError(f"Unknown legacy Cellpose model: {variant}")
    home = runtime_home()
    home.mkdir(parents=True, exist_ok=True)
    python = _python_path(home)
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                        str(home / "runtime")], check=True, timeout=300)
    check = subprocess.run(
        [str(python), "-c", "import importlib.metadata as m; "
         "print(m.version('cellpose'))"], capture_output=True, text=True)
    if check.returncode != 0 or check.stdout.strip() != LEGACY_VERSION:
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps",
                        f"cellpose=={LEGACY_VERSION}"], check=True, timeout=600)

    name, url, expected = MODEL_FILES[variant]
    model_dir = home / "models"
    model_dir.mkdir(exist_ok=True)
    model = model_dir / name
    if not model.is_file() or _digest(model) != expected:
        temporary = model.with_name(f".{name}.{os.getpid()}.download")
        try:
            with urlopen(url, timeout=120) as source, temporary.open("wb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
            if _digest(temporary) != expected:
                raise RuntimeError(f"Downloaded {variant} model did not match its published file")
            temporary.replace(model)
        finally:
            temporary.unlink(missing_ok=True)
    return str(python), str(model)


def segmentation_command(script: str, config_path: str, config: dict) -> tuple[list[str], dict]:
    """Use the isolated runtime for legacy Cellpose prediction and wizard runs."""
    variant = str(config.get("seg_cellpose_variant") or "cpsam").strip().lower()
    if (Path(script).name not in {"segmentation.py", "parameter_wizard_compute.py"}
            or str(config.get("seg_backend") or "").lower() != "cellpose"
            or variant == "cpsam"):
        return [sys.executable, script, "--config", config_path], os.environ.copy()
    python, model = ensure_runtime(variant)
    environment = os.environ.copy()
    environment["CELLMATCHR_LEGACY_CELLPOSE_MODEL_PATH"] = model
    return [python, script, "--config", config_path], environment
