"""Install the recall-selected Nissl Micro-SAM LM model as an additional model."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cellmatchr.retraining import model_library_root

FOLDER = "microsam_ch0_nissl_lm_recall_v1_20261113"
WEIGHT_NAME = "nissl_microsam_lm_recall_v1.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    chosen = selection["best_trained"]
    threshold = float(chosen["foreground_threshold"])
    if not 0.0 <= threshold <= 1.0 or not args.weights.is_file():
        raise ValueError("Selected checkpoint or threshold is invalid")
    target = model_library_root() / FOLDER
    if target.exists():
        raise FileExistsError(f"Model already installed: {target}")
    target.mkdir(parents=True)
    try:
        shutil.copy2(args.weights, target / WEIGHT_NAME)
        epochs = int(chosen["checkpoint"].removeprefix("epoch-").removesuffix(".pt"))
        record = {
            "backend": "microsam", "name": "Nissl LM Recall v1", "channel": 0,
            "model_path": str(target / WEIGHT_NAME),
            "model_relative_path": WEIGHT_NAME,
            "starting_model_name": "vit_b_lm", "parent_model_path": "",
            "training_rounds": 1, "recorded_epochs": epochs,
            "epochs": epochs, "history_complete": True,
            "preprocess": "brightfield_invert",
            "suggested_inference": {"foreground_threshold": threshold},
            "advanced": {"learning_rate": 3e-6, "batch_size": 1,
                         "patch_size": 256, "freeze_encoder": True},
            "validation_source": selection["validation_source"],
            "validation_result": chosen["validation"],
            "comparison_source": selection["test_source"],
            "comparison_result": next(iter(selection["trained_test"].values()))["overall"],
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        (target / "model.json").write_text(json.dumps(record, indent=2) + "\n")
    except Exception:
        shutil.rmtree(target)
        raise
    print(target)


if __name__ == "__main__":
    main()
