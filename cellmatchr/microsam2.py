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
MODELS = ("hvit_t_cells", "hvit_s_cells", DEFAULT_MODEL, "hvit_l_cells")
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
        print(f"{DISPLAY_NAME}: preparing its Python environment. First use downloads "
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


class MicroSam2:
    """Keep one model loaded for all tiles and channels in a run."""

    def __init__(self, model, device):
        if model not in MODELS:
            raise ValueError(f"Unknown {DISPLAY_NAME} model: {model}")
        self.model, self.device = model, device
        self.process = None

    def __enter__(self):
        home = _runtime_home()
        env = _environment()
        env["MICROSAM_CACHEDIR"] = str(home / "model-cache")
        if self.device == "mps":
            env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.7")
        python = _prepare_runtime(home, self.device, env)
        print(f"{DISPLAY_NAME}: loading {self.model}. New model weights download "
              "automatically on first use.", flush=True)
        self.process = subprocess.Popen(
            [str(python), "-I", str(Path(__file__).resolve()), self.model, self.device],
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
            predictor, segmenter = get_predictor_and_segmenter(
                model_type=model, segmentation_mode="ais", device=device)
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
                labels = automatic_instance_segmentation(
                    predictor, segmenter, image, ndim=2, verbose=False, min_size=0)
                _send(protocol_out, {}, np.asarray(labels, dtype=np.uint32))
            except Exception as exc:
                traceback.print_exc()
                _send(protocol_out, {"error": str(exc)})


if __name__ == "__main__":
    if sys.argv[1] == "--install":
        _install_runtime(*sys.argv[2:])
    else:
        _worker(*sys.argv[1:])
