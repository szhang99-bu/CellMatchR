"""Train a new all-data Nissl Cellpose model after source-held-out review.

The original Cellpose-SAM and Nissl General v1 checkpoints stay untouched.
The selected six- or eleven-epoch checkpoint is copied into a new shared
model-library record so it can be chosen in any CellMatchR project.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from recall_experiment import DEFAULT_OUTPUT, train_fold
from train_full_cytodark import load_crops, sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, choices=(6, 11), required=True)
    parser.add_argument("--name", default="Nissl Recall v1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = args.output / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError("Finish all source-held-out folds before final training")
    cv = json.loads(summary.read_text())
    selected_setting = next(
        item["setting"] for item in cv["all_candidates"]
        if item["key"].startswith(f"epoch_{args.epoch}__"))
    items = sources()
    crops = load_crops(items)
    training_dir = args.output / "final_training"
    record_file = training_dir / "training.json"
    if record_file.is_file():
        trained = json.loads(record_file.read_text())
    else:
        print(f"Training from original cpsam on {len(items)} source fields / "
              f"{len(crops)} crops", flush=True)
        trained = train_fold(crops, training_dir, "nissl_recall_all_fields")
        record_file.write_text(json.dumps(trained, indent=2) + "\n")
    chosen = Path(trained[f"epoch_{args.epoch}"])
    if not chosen.is_file():
        raise FileNotFoundError(chosen)
    library = Path.home() / "CellMatchR/Models"
    library.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = library / f"cellpose_ch0_nissl_recall_v1_{timestamp}"
    folder.mkdir()
    target = folder / "models/trained_model"
    target.parent.mkdir()
    shutil.copy2(chosen, target)
    record = {
        "backend": "cellpose", "channel": 0, "name": args.name,
        "model_path": str(target), "model_relative_path": "models/trained_model",
        "starting_model_name": "cpsam", "base_model": "cpsam",
        "preprocess": "raw_percentile", "epochs": args.epoch,
        "recorded_epochs": args.epoch, "training_rounds": 1,
        "history_complete": True,
        "advanced": {"learning_rate": 1e-5, "batch_size": 1,
                     "sampling": trained["sampling"]},
        "suggested_inference": selected_setting,
        "source_dataset": "CytoDArk0 20x",
        "training_sources": [item["stem"] for item in items],
        "source_heldout_macaque": cv["crossfit_f3_macaque"],
        "source_heldout_all_tissues": cv["crossfit_f3_overall"],
        "cross_validation_summary": str(summary),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    (folder / "model.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Registered new shared model: {target}", flush=True)
    print("The original checkpoints were not modified.", flush=True)


if __name__ == "__main__":
    main()
