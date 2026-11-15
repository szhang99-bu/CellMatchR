"""Checks for Windows launch preparation that can run before a PC is ready."""

import os
import subprocess
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from cellmatchr.launcher import _popen_logged
from scripts.windows_preflight import run_check


class WindowsPreviewTests(unittest.TestCase):
    def test_captured_child_logs_use_utf8_on_windows(self):
        original_environment = {"PATH": "test-path"}
        with patch("cellmatchr.launcher.sys.platform", "win32"), \
             patch("cellmatchr.launcher.subprocess.Popen") as popen:
            _popen_logged(["python", "segmentation.py"], env=original_environment)
        kwargs = popen.call_args.kwargs
        self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(kwargs["env"]["PATH"], "test-path")
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")
        self.assertEqual(original_environment, {"PATH": "test-path"})

    def test_preflight_checks_use_separate_python_process(self):
        completed = subprocess.CompletedProcess([], 0, "ready\n", "")
        with redirect_stdout(StringIO()), \
             patch("scripts.windows_preflight.subprocess.run", return_value=completed) as run:
            self.assertTrue(run_check("Sample check", "print('ready')"))
        args, kwargs = run.call_args
        self.assertEqual(args[0][-2:], ["-c", "print('ready')"])
        self.assertEqual(kwargs["env"]["QT_API"], "pyqt5")
        self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertTrue(os.path.isdir(kwargs["cwd"]))

    def test_preflight_reports_import_failure(self):
        completed = subprocess.CompletedProcess([], 1, "", "ImportError: missing reader\n")
        output = StringIO()
        with redirect_stdout(output), \
             patch("scripts.windows_preflight.subprocess.run", return_value=completed):
            self.assertFalse(run_check("Image readers", "import missing_reader"))
        self.assertIn("ImportError: missing reader", output.getvalue())


if __name__ == "__main__":
    unittest.main()
