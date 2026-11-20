"""Portable project records for confirmed model comparisons."""

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

FOLDER = "Model_Wizard_Previews"


def capture(folder, comparison, chosen, signature, model_identity):
    """Keep the source patch, comparison cards and exact label masks."""
    from .model_wizard_compute import write_json

    folder.mkdir()
    original = "original.png"
    shutil.copyfile(comparison["original"], folder / original)
    shutil.copyfile(chosen["raw"], folder / "raw.tif")
    if chosen.get("source_rgb"):
        shutil.copyfile(chosen["source_rgb"], folder / "source-rgb.tif")

    cards = []
    for index, item in enumerate(comparison["results"], 1):
        card = {"name": item["name"]}
        if item.get("overlay"):
            card.update(count=item["count"], overlay=f"model-{index}.png",
                        labels=f"model-{index}-labels.tif", raw="raw.tif")
            for key in ("overlay", "labels"):
                shutil.copyfile(item[key], folder / card[key])
            if chosen.get("source_rgb"):
                card["source_rgb"] = "source-rgb.tif"
        else:
            card.update(error=item["error"], log=f"model-{index}.log")
            shutil.copyfile(item["log"], folder / card["log"])
        cards.append(card)

    preview = {key: comparison["preview"][key] for key in ("sample", "brightfield", "source_path", "scene")}
    data = {**comparison, "version": 1, "original": original, "results": cards, "preview": preview,
            "chosen": chosen["name"], "signature": signature, "model_identity": model_identity}
    write_json(folder / "comparison.json", data)


def publish(folder, project, run):
    """Stage a complete record before making it visible to the project."""
    parent = Path(project) / FOLDER
    parent.mkdir(exist_ok=True)
    target = parent / Path(run).name
    if target.exists():
        if (target / "comparison.json").is_file() and (
                (target / "comparison.json").read_bytes() == (folder / "comparison.json").read_bytes()):
            return target
        target = parent / f"{target.name}-{uuid4().hex[:8]}"
    with TemporaryDirectory(prefix=".save-", dir=parent) as pending:
        staged = Path(pending) / "preview"
        shutil.copytree(folder, staged)
        staged.rename(target)
    return target


def load(folder):
    """Resolve local artifacts relative to the archive, including after a move."""
    folder = folder.resolve()

    def member(name):
        path = (folder / name).resolve()
        if path.parent != folder or not path.is_file():
            raise ValueError(f"Missing or invalid comparison file: {name}")
        return str(path)

    data = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    if (data["version"] != 1 or not isinstance(data["models"], dict) or len(data["box"]) != 4
            or not isinstance(data["signature"], dict) or len(data["preview"]["source_path"]) != 3):
        raise ValueError("Unsupported model comparison record.")
    data["original"] = member(data["original"])
    for item in data["results"]:
        for key in ("overlay", "labels", "raw", "source_rgb", "log"):
            if item.get(key):
                item[key] = member(item[key])
    if not any(row.get("overlay") and row["name"] == data["chosen"] and row["name"] in data["models"]
               for row in data["results"]):
        raise ValueError("The selected prediction is missing.")
    data["folder"] = folder
    return data


def saved(project):
    if not project:
        return
    try:
        directories = [folder for folder in (Path(project) / FOLDER).iterdir()
                       if not folder.name.startswith(".") and folder.is_dir()]
        directories.sort(key=lambda folder: folder.stat().st_mtime_ns, reverse=True)
    except OSError:
        return
    for folder in directories:
        try:
            yield load(folder)
        except (OSError, ValueError, KeyError, TypeError):
            continue
