"""Run micro-SAM2 in its own Python environment."""

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


VERSION = "2.0.1"
DEFAULT_MODEL = "hvit_b_cells"
BASE_MODELS = ("hvit_t", "hvit_s", "hvit_b", "hvit_l")
MODELS = ("hvit_t_cells", "hvit_s_cells", DEFAULT_MODEL, "hvit_l_cells",
          "hvit_s_em_mito", "hvit_s_em_nuclei", *BASE_MODELS)
DISPLAY_NAME = "micro-SAM2"


def preprocessing_name(brightfield):
    return "native_rgb" if brightfield else "native_channel"


def _environment():
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT",
                "QT_API", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        env.pop(key, None)
    env["SAM2_BUILD_CUDA"] = "0"
    return env


def _runtime_home():
    override = os.environ.get("CELLMATCHR_MICROSAM2_HOME")
    return Path(override).expanduser() if override else Path.home() / "CellMatchR" / "MicroSAM2"


def _torch_channel(device):
    if sys.platform == "darwin":
        return ""
    if device == "cuda":
        import torch

        if torch.version.cuda:
            version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
            return "cu128" if version >= (12, 8) else "cu126"
    return "cpu"


def _prepare_runtime(home, device, env):
    channel = _torch_channel(device)
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    runtime = home / f"runtime-{VERSION}-py{python_version}-{channel or 'mac'}"
    python = _runtime_python(runtime)
    if not ((runtime / ".setup_complete").is_file() and python.is_file()):
        # The installer owns its lock independently of a cancelled GUI run.
        # A retry waits for it instead of writing to the same venv.
        try:
            subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()),
                 "--install", str(runtime), channel],
                env=env, stdin=subprocess.DEVNULL, check=True, timeout=7200)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"{DISPLAY_NAME} setup failed. Check the error above and "
                               "run segmentation again to retry.") from exc
    return python


def _runtime_python(runtime):
    return runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _install_runtime(runtime, channel):
    from filelock import FileLock

    runtime = Path(runtime)
    python = _runtime_python(runtime)
    ready = runtime / ".setup_complete"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(runtime) + ".lock", timeout=3600):
        if ready.is_file() and python.is_file():
            return
        ready.unlink(missing_ok=True)
        print(f"[MODEL] {DISPLAY_NAME}: preparing its Python environment. First use downloads "
              "the runtime and selected model weights.", flush=True)
        pip = [str(python), "-I", "-m", "pip", "install", "--no-input",
               "--disable-pip-version-check", "--no-cache-dir"]
        torch_packages = ["torch==2.10.0", "torchvision==0.25.0"]
        subprocess.run([sys.executable, "-I", "-m", "venv", str(runtime)],
                       check=True, timeout=300)
        if channel:
            subprocess.run(pip + torch_packages + [
                "--index-url", f"https://download.pytorch.org/whl/{channel}"],
                check=True, timeout=1800)
            # SAM2 lists Torch as a build dependency. Build with the Torch
            # wheel already selected for this machine.
            subprocess.run(pip + ["setuptools>=61", "wheel"], check=True, timeout=300)
            subprocess.run(pip + ["--no-build-isolation", "--no-deps", "sam2==1.1.0"],
                           check=True, timeout=300)
        subprocess.run(pip + [f"micro-sam=={VERSION}"] + torch_packages,
                       check=True, timeout=1800)
        ready.touch()


def _send(stream, message, array=None):
    if array is not None:
        import numpy as np

        array = np.ascontiguousarray(array)
        message = dict(message, shape=array.shape, dtype=array.dtype.str, nbytes=array.nbytes)
    stream.write(json.dumps(message).encode("utf-8") + b"\n")
    if array is not None:
        stream.write(memoryview(array).cast("B"))
    stream.flush()


def _receive(stream):
    line = stream.readline()
    if not line:
        raise EOFError("worker stream closed")
    message = json.loads(line)
    if "nbytes" not in message:
        return message, None

    import numpy as np

    payload = stream.read(message["nbytes"])
    if len(payload) != message["nbytes"]:
        raise EOFError("incomplete worker array")
    return message, np.frombuffer(payload, dtype=message["dtype"]).reshape(message["shape"])


def _worker_command(model, device, *, download_only=False):
    if model not in MODELS:
        raise ValueError(f"Unknown {DISPLAY_NAME} model: {model}")
    home = _runtime_home()
    env = _environment()
    env["MICROSAM_CACHEDIR"] = str(home / "model-cache")
    if device == "mps":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.7")
    python = _prepare_runtime(home, device, env)
    command = [str(python), "-I", str(Path(__file__).resolve())]
    if download_only:
        command.append("--download")
    return command + [model, device], env


def download_weights(model, device):
    command, env = _worker_command(model, device, download_only=True)
    subprocess.run(command, env=env, check=True)


def _download_weight_ranges(url, output_file, description):
    """Switch a slow large transfer to four byte ranges, keeping bytes already read."""
    from concurrent.futures import ThreadPoolExecutor
    import re
    from threading import Lock
    from time import monotonic
    from urllib.parse import urlsplit

    import requests

    if urlsplit(url).hostname != "owncloud.gwdg.de" or not isinstance(output_file, (str, os.PathLike)):
        return False
    timeout = (15, 45)
    with requests.get(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
                      timeout=timeout, stream=True) as probe:
        limits = re.fullmatch(r"bytes 0-0/(\d+)", probe.headers.get("Content-Range", ""))
        if probe.status_code != 206 or not limits or int(limits[1]) < 32_000_000:
            return False
        size = int(limits[1])
    received = 0
    phase_bytes = 0
    started = last_update = None
    lock = Lock()

    def transfer(first, stop, *, check_speed=False):
        nonlocal received, started, last_update
        range_header = f"{first}-{stop - 1}"
        headers = {"Range": f"bytes={range_header}", "Accept-Encoding": "identity"}
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as source:
            if (source.status_code != 206
                    or source.headers.get("Content-Range") != f"bytes {range_header}/{size}"):
                raise ValueError("The model host returned a different byte range.")
            if check_speed:
                started = last_update = monotonic()
            with open(output_file, "r+b", buffering=0) as target:
                target.seek(first)
                for chunk in source.iter_content(chunk_size=256 * 1024):
                    if target.tell() + len(chunk) > stop:
                        raise ValueError("Unexpected number of model bytes.")
                    if target.write(chunk) != len(chunk):
                        raise OSError("Could not write the complete model chunk.")
                    with lock:
                        received += len(chunk)
                        now = monotonic()
                        elapsed = now - started
                        speed = (received - phase_bytes) / max(elapsed, 0.01) / 1_000_000
                        if now - last_update >= 1 or received == size:
                            print(f"[MODEL] Downloading {description} — {100 * received // size}% "
                                  f"({speed:.2f} MB/s)", flush=True)
                            last_update = now
                        if check_speed and elapsed >= 6 and speed < 0.75 and received < size:
                            return False
                if target.tell() != stop:
                    raise ValueError("Incomplete model file.")
        return True

    Path(output_file).write_bytes(b"")
    try:
        if transfer(0, size, check_speed=True):
            return True
        reason = "the connection is slow"
    except requests.RequestException:
        if received == size:
            return True  # Pooch still checks the complete file against the published hash.
        reason = "the connection was interrupted"

    print(f"[MODEL] Downloading {description} — {reason}. Trying parallel connections "
          f"and keeping {received / 1_000_000:.1f} MB already received "
          f"({100 * received // size}%).", flush=True)
    with open(output_file, "r+b") as target:
        target.truncate(size)
    phase_bytes = received
    started = last_update = monotonic()
    remaining = size - phase_bytes
    connections = min(4, remaining)
    boundaries = [phase_bytes + remaining * part // connections for part in range(connections + 1)]
    with ThreadPoolExecutor(max_workers=connections) as pool:
        list(pool.map(transfer, boundaries[:-1], boundaries[1:]))
    return True


def _prepare_model_weights(model):
    import pooch
    import requests
    from filelock import FileLock
    from micro_sam.v2 import util
    from pooch.hashes import hash_matches

    if model in BASE_MODELS:
        registry = pooch.create(path=util.get_model_cache_directory(), base_url="",
                                registry={model: util.HASHES[model]}, urls={model: util.URLS[model]})
        names = [model]
    else:
        registry = util.models()
        names = [model, f"{model}_decoder"]
    registry.abspath.mkdir(parents=True, exist_ok=True)

    for index, name in enumerate(names, start=1):
        path = registry.abspath / name
        role = "decoder" if name.endswith("_decoder") else "encoder"
        description = f"{model} {role} (file {index} of {len(names)})"
        with FileLock(str(path) + ".lock", timeout=7200):
            if path.is_file() and hash_matches(path, registry.registry[name]):
                print(f"[MODEL] Using cached {description}.", flush=True)
                continue

            def download(url, output_file, owner, check_only=False):
                standard = pooch.HTTPDownloader(progressbar=True, timeout=(15, 45))
                if not check_only:
                    print(f"[MODEL] Downloading {description}…", flush=True)
                    try:
                        if _download_weight_ranges(url, output_file, description):
                            return
                    except (requests.RequestException, ValueError):
                        print(f"[MODEL] Downloading {description} over a single connection…", flush=True)
                return standard(url, output_file, owner, check_only=check_only)

            registry.fetch(name, downloader=download)


class MicroSam2:
    """Keep one model loaded for all tiles and channels in a run."""

    def __init__(self, model, device):
        if model not in MODELS:
            raise ValueError(f"Unknown {DISPLAY_NAME} model: {model}")
        self.model, self.device = model, device
        self.process = None

    def __enter__(self):
        print(f"[MODEL] Checking {DISPLAY_NAME} setup for {self.model}…", flush=True)
        command, env = _worker_command(self.model, self.device)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        try:
            self._response()
        except BaseException:
            self.__exit__()
            raise
        return self

    def _response(self):
        try:
            status, result = _receive(self.process.stdout)
        except (EOFError, OSError, ValueError) as exc:
            raise RuntimeError(f"{DISPLAY_NAME} stopped. See the run log for details.") from exc
        if "error" in status:
            raise RuntimeError(f"{DISPLAY_NAME}: {status['error']}")
        return result

    def predict(self, image):
        try:
            _send(self.process.stdin, {}, image)
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"{DISPLAY_NAME} stopped. See the run log for details.") from exc
        labels = self._response()
        if labels is None or labels.shape != image.shape[:2]:
            raise RuntimeError(f"{DISPLAY_NAME} returned an unexpected mask size")
        return labels

    def __exit__(self, *_):
        process = self.process
        if process is None:
            return
        self.process = None
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        finally:
            process.stdout.close()


def _worker(model, device):
    import numpy as np
    import torch

    protocol_out = sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        try:
            from micro_sam.v2.automatic_segmentation import (
                automatic_instance_segmentation, get_predictor_and_segmenter,
            )

            if ((device == "cuda" and not torch.cuda.is_available()) or
                    (device == "mps" and not torch.backends.mps.is_available())):
                device = "cpu"
            print(f"{DISPLAY_NAME}: using {device}", file=sys.stderr, flush=True)
            _prepare_model_weights(model)
            predictor, segmenter = get_predictor_and_segmenter(
                model_type=model, device=device)
            print(f"[MODEL] {DISPLAY_NAME} weights ready. Generating predictions…", flush=True)
        except Exception as exc:
            traceback.print_exc()
            _send(protocol_out, {"error": str(exc)})
            return
        _send(protocol_out, {"ready": True})
        while True:
            try:
                _, image = _receive(sys.stdin.buffer)
            except EOFError:
                return
            try:
                options = {} if model in BASE_MODELS else {"min_size": 0}
                labels = automatic_instance_segmentation(
                    predictor, segmenter, image, ndim=2, verbose=False, **options)
                _send(protocol_out, {}, np.asarray(labels, dtype=np.uint32))
            except Exception as exc:
                traceback.print_exc()
                _send(protocol_out, {"error": str(exc)})


if __name__ == "__main__":
    if sys.argv[1] == "--install":
        _install_runtime(*sys.argv[2:])
    elif sys.argv[1] == "--download":
        _prepare_model_weights(sys.argv[2])
    else:
        _worker(*sys.argv[1:])
