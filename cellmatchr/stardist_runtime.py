"""Prepare StarDist on first use without changing the main environment."""

import os
from pathlib import Path
import subprocess
import sys
import traceback

if __package__:
    from .microsam2 import _environment, _receive, _runtime_python, _send
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from microsam2 import _environment, _receive, _runtime_python, _send


BUILTIN_MODEL = "2D_versatile_fluo"
RUNTIME_VERSION = "0.9.1"


def _runtime():
    from filelock import FileLock

    root = Path(os.environ.get("CELLMATCHR_STARDIST_HOME",
                               str(Path.home() / "CellMatchR" / "StarDist"))).expanduser()
    runtime = root / f"runtime-{RUNTIME_VERSION}-py{sys.version_info.major}.{sys.version_info.minor}"
    python = _runtime_python(runtime)
    completed = runtime / ".setup_complete"
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(runtime) + ".lock", timeout=3600):
        if completed.is_file() and python.is_file():
            return python
        print("[MODEL] Installing the optional StarDist runtime. First use may take several minutes.",
              flush=True)
        env = _environment()
        try:
            subprocess.run(
                [sys.executable, "-I", "-m", "venv", str(runtime)],
                env=env, check=True, timeout=300)
            subprocess.run(
                [str(python), "-I", "-m", "pip", "install", "--no-input",
                 "--no-cache-dir", "--disable-pip-version-check", "numpy==1.26.4",
                 f"stardist=={RUNTIME_VERSION}", "tensorflow==2.16.2",
                 "csbdeep>=0.8.1,<0.9"],
                env=env, check=True, timeout=1800)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Could not install the StarDist runtime. Check the "
                               "connection and open the model log for details.") from exc
        completed.touch()
    return python


def _load_stardist(StarDist2D, checkpoint):
    if checkpoint:
        path = Path(checkpoint).resolve()
        return StarDist2D(None, name=path.name, basedir=str(path.parent))
    print(f"[MODEL] Preparing StarDist weights ({BUILTIN_MODEL}). "
          "The first run downloads them.", flush=True)
    return StarDist2D.from_pretrained(BUILTIN_MODEL)


class StarDistRuntime:
    """Use an installed StarDist runtime or serve predictions from an isolated one."""

    def __init__(self, checkpoint=""):
        self.checkpoint = checkpoint
        self.process = None

    def __enter__(self):
        try:
            from stardist.models import StarDist2D
        except (ImportError, OSError):
            python = _runtime()
            self.process = subprocess.Popen(
                [str(python), "-I", "-u", str(Path(__file__).resolve()), self.checkpoint],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=_environment())
            try:
                self._result()
            except BaseException:
                self.__exit__()
                raise
        else:
            self.model = _load_stardist(StarDist2D, self.checkpoint)
        return self

    def _result(self):
        try:
            status, labels = _receive(self.process.stdout)
        except (OSError, ValueError, EOFError) as exc:
            raise RuntimeError("The StarDist runtime stopped. See the model log.") from exc
        if "error" in status:
            raise RuntimeError(f"StarDist: {status['error']}")
        return labels

    def predict_instances(self, image):
        if self.process is None:
            return self.model.predict_instances(image)
        _send(self.process.stdin, {}, image)
        return self._result(), None

    def __exit__(self, *_):
        if self.process is not None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()
            self.process = None


def _worker(checkpoint):
    from contextlib import redirect_stdout

    protocol = sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        try:
            import numpy as np
            from stardist.models import StarDist2D

            model = _load_stardist(StarDist2D, checkpoint)
            _send(protocol, {"ready": True})
            while True:
                try:
                    _, image = _receive(sys.stdin.buffer)
                except EOFError:
                    return
                labels, _ = model.predict_instances(image)
                _send(protocol, {}, np.asarray(labels, dtype=np.uint32))
        except Exception as exc:
            traceback.print_exc()
            _send(protocol, {"error": str(exc)})


if __name__ == "__main__":
    _worker(sys.argv[1])
