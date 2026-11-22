"""Distinguish one brightfield measurement from its RGB display components."""

import json
import os

if __package__:
    from .file_paths import filesystem_path
else:
    from file_paths import filesystem_path


def is_brightfield(settings):
    return (settings.get("input_type") == "vsi"
            and settings.get("slide_signal_type") == "brightfield")


def read_run_progress(run):
    if run:
        progress = os.path.join(run, "_run_progress.json")
        try:
            with open(filesystem_path(progress), encoding="utf-8") as source:
                record = json.load(source)
            if isinstance(record, dict):
                return record
        except (OSError, ValueError, TypeError):
            pass
    return {}


def run_is_brightfield(run, settings=None):
    identity = read_run_progress(run).get("run_identity")
    if isinstance(identity, dict) and identity:
        return identity.get("slide_signal_type") == "brightfield"
    return is_brightfield(settings or {})
