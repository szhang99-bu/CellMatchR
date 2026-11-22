"""Download one built-in model into the cache shared by prediction and previews."""

import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from stat import S_ISREG
from tempfile import TemporaryDirectory
from zipfile import is_zipfile


DOWNLOADS_TITLE = "Additional Pretrained Model Downloads"
MICROSAM_BASE_MODELS = ("vit_t", "vit_b", "vit_l", "vit_h")
MICROSAM_MODELS = (
    "vit_t_lm", "vit_b_lm", "vit_l_lm",
    *MICROSAM_BASE_MODELS,
    "vit_b_histopathology", "vit_l_histopathology", "vit_h_histopathology",
    "vit_b_medical_imaging", "vit_t_em_organelles", "vit_b_em_organelles", "vit_l_em_organelles",
)
MICROSAM_MIRRORS = {
    "vit_t": "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt",
}
MOBILE_SAM_REVISION = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"

# Published transfer sizes, in decimal MB, including both weights when needed.
MODEL_WEIGHT_MB = {
    "vit_t": 41, "vit_b": 375, "vit_l": 1250, "vit_h": 2565,
    "vit_t_lm": 85, "vit_b_lm": 414, "vit_l_lm": 1294,
    "vit_t_em_organelles": 85, "vit_b_em_organelles": 419, "vit_l_em_organelles": 1294,
    "vit_b_histopathology": 419, "vit_l_histopathology": 1294, "vit_h_histopathology": 2609,
    "vit_b_medical_imaging": 772,
    "hvit_t": 156, "hvit_s": 184, "hvit_b": 324, "hvit_l": 898,
    "hvit_t_cells": 291, "hvit_s_cells": 347, "hvit_b_cells": 626, "hvit_l_cells": 1775,
    "hvit_s_em_mito": 347, "hvit_s_em_nuclei": 347,
    "cpsam": 1234, "cyto2": 27, "rat_nissl_v2": 27, "2D_versatile_fluo": 5,
}


def built_in_model_id(settings):
    return (settings.get("seg_model_type") or settings.get("seg_microsam2_model")
            or settings.get("seg_cellpose_variant") or "2D_versatile_fluo")


def model_choice_label(name, settings):
    if any(value for key, value in settings.items() if key.endswith("_model_path")):
        return name
    size = MODEL_WEIGHT_MB.get(built_in_model_id(settings))
    if size is None:
        return name
    amount = f"{size / 1000:.2f} GB" if size >= 1000 else f"{size} MB"
    return f"{name}  (≈{amount})"


def model_is_downloaded(settings):
    """Check local weight files without loading the model or starting a runtime."""
    model = built_in_model_id(settings)
    backend = settings["seg_backend"]
    try:
        if backend == "microsam":
            from pooch import os_cache

            cache = Path(os.environ.get("MICROSAM_CACHEDIR") or os_cache("micro_sam")).expanduser() / "models"
            files = [cache / model]
            if model not in MICROSAM_BASE_MODELS:
                files.append(cache / f"{model}_decoder")
        elif backend == "microsam2":
            from .microsam2 import BASE_MODELS, _runtime_home

            cache = _runtime_home() / "model-cache" / "models"
            files = [cache / model]
            if model not in BASE_MODELS:
                files.append(cache / f"{model}_decoder")
        elif backend == "cellpose" and model == "cpsam":
            cache = Path(os.environ.get("CELLPOSE_LOCAL_MODELS_PATH") or Path.home() / ".cellpose" / "models")
            files = [cache / model]
        elif backend == "cellpose":
            from .legacy_cellpose import MODEL_FILES, runtime_home

            files = [runtime_home() / "models" / MODEL_FILES[model][0]]
        elif backend == "stardist":
            cache = Path(os.environ.get("KERAS_HOME") or Path.home() / ".keras") / "models" / "StarDist2D"
            return is_zipfile(cache / model / f"{model}.zip")
        else:
            return False
        for file in files:
            info = file.stat()
            if not S_ISREG(info.st_mode) or info.st_size == 0:
                return False
        return True
    except (OSError, ImportError, KeyError):
        return False


def enable_mobile_sam(model):
    """Make the optional ViT-Tiny implementation available in this process."""
    if not model.startswith("vit_t"):
        return
    from micro_sam import util

    if util.VIT_T_SUPPORT:
        return
    from filelock import FileLock

    root = Path(os.environ.get("CELLMATCHR_MOBILE_SAM_HOME",
                               str(Path.home() / "CellMatchR" / "MobileSAM"))).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    target = root / MOBILE_SAM_REVISION[:12]
    with FileLock(str(target) + ".lock", timeout=600):
        if not target.is_dir():
            print("[MODEL] Installing MobileSAM support for the Tiny model…", flush=True)
            source = (f"https://codeload.github.com/ChaoningZhang/MobileSAM/"
                      f"tar.gz/{MOBILE_SAM_REVISION}")
            with TemporaryDirectory(dir=root) as staging:
                packages = Path(staging) / "packages"
                env = os.environ.copy()
                env.pop("PIP_CONSTRAINT", None)
                env.pop("PIP_BUILD_CONSTRAINT", None)
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--no-input", "--no-deps",
                         "--no-cache-dir", "--disable-pip-version-check",
                         "--target", str(packages), source],
                        check=True, env=env, timeout=300)
                except (OSError, subprocess.SubprocessError) as exc:
                    raise RuntimeError("Could not install the optional MobileSAM dependency.") from exc
                packages.replace(target)
    sys.path.insert(0, str(target))
    importlib.invalidate_caches()
    importlib.reload(util)
    if not util.VIT_T_SUPPORT:
        raise RuntimeError("MobileSAM was downloaded but could not be imported.")


def microsam_weights(model):
    from micro_sam import util

    registry = util.models()
    names = [key for key in (model, f"{model}_decoder") if key in registry.registry]
    if not names or names[0] != model:
        raise ValueError(f"Unknown Micro-SAM model: {model}")
    return registry, names


def fetch_microsam_weight(registry, name):
    import requests

    try:
        return registry.fetch(name, progressbar=True)
    except (requests.RequestException, ValueError) as exc:
        mirror = MICROSAM_MIRRORS.get(name)
        if not mirror:
            host = registry.get_url(name).split("/")[2]
            raise RuntimeError(f"Could not download or verify {name} from {host}. "
                               "Retry the download or install the verified checkpoint "
                               "in the Micro-SAM model cache.") from exc
        print(f"[MODEL] Downloading {name} from its official alternate source…", flush=True)
        # Pooch verifies the same registered hash regardless of which URL served the data.
        registry.urls[name] = mirror
        return registry.fetch(name, progressbar=True)


def import_microsam_weights(model, files):
    """Accept local model files only when they match the upstream registry."""
    from pooch.hashes import hash_matches

    registry, names = microsam_weights(model)
    registry.abspath.mkdir(parents=True, exist_ok=True)
    files = [Path(path) for path in files]
    verified = {}
    for name in names:
        expected = registry.registry[name]
        cached = registry.abspath / name
        if cached.is_file() and hash_matches(cached, expected):
            continue
        selected = next((file for file in files if hash_matches(file, expected)), None)
        if selected is None:
            raise ValueError(f"No selected file matches the published checksum for {name}.")
        verified[name] = selected
    with TemporaryDirectory(dir=registry.abspath) as staging:
        for name, source in verified.items():
            pending = Path(staging) / name
            shutil.copyfile(source, pending)
            if not hash_matches(pending, registry.registry[name]):
                raise ValueError(f"The selected {name} file changed during import.")
        for name in verified:
            (Path(staging) / name).replace(registry.abspath / name)
    print(f"[MODEL] Verified weights for {model} are ready.", flush=True)


def model_stage(log):
    """Read the last model status message from a running process's log."""
    log = Path(log)
    try:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 65536))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    percent = ""
    for line in reversed(lines):
        if not percent:
            match = re.match(r"\s*(\d+%)\|", line)
            if match:
                percent = f" ({match[1]})"
        if line.startswith("[MODEL] "):
            status = line.removeprefix("[MODEL] ").strip()
            return status + percent if status.startswith("Downloading") and "%" not in status else status
        if line.startswith("Downloading file "):
            match = re.match(r"Downloading file '(.*?)'", line)
            name = match.group(1) if match else ""
            role = "decoder (file 2 of 2)" if name.endswith("_decoder") else ""
            return f"Downloading model {role or 'weights'}{percent}. First use may take several minutes."
    return "Downloading model weights" + percent if percent else ""


def download(backend, model):
    if backend == "microsam":
        enable_mobile_sam(model)
        registry, names = microsam_weights(model)
        for name in names:
            print(f"[MODEL] Checking {name}. Missing weights download automatically.", flush=True)
            fetch_microsam_weight(registry, name)
    elif backend == "microsam2":
        from .microsam2 import download_weights
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        download_weights(model, device)
    elif backend == "stardist":
        from .stardist_runtime import StarDistRuntime

        with StarDistRuntime():
            pass
    elif backend == "cellpose":
        if model == "cpsam":
            from cellpose.models import cache_CPSAM_model_path

            print("[MODEL] Preparing Cellpose-SAM weights (first use downloads them).", flush=True)
            cache_CPSAM_model_path()
        else:
            from .legacy_cellpose import ensure_runtime

            print(f"[MODEL] Preparing the legacy Cellpose runtime and {model} weights.", flush=True)
            ensure_runtime(model)
    else:
        raise ValueError(f"Unknown model backend: {backend}")
    print(f"[MODEL] {model} is ready.", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--import-microsam":
        import_microsam_weights(sys.argv[2], sys.argv[3:])
    else:
        download(*sys.argv[1:])
