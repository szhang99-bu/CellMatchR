"""Save macaque masks from fold models that never trained on their source field."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from cellpose.models import CellposeModel

from compare_macaque_methods import DEFAULT_OUTPUT, macaque_crops
from finetune_cytodark import score, summarize
from recall_experiment import DEFAULT_OUTPUT as RUN
from train_full_cytodark import fold_map, sources


def main() -> None:
    summary = json.loads((RUN / "summary.json").read_text())
    chosen = {entry["target_fold"]: entry["chosen"]
              for entry in summary["crossfit_f3"]}
    folds = fold_map(sources())
    crops = macaque_crops()
    output = DEFAULT_OUTPUT / "recall_candidate_held_out"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(4):
        held = [crop for crop in crops if folds[crop["source"]] == fold]
        key = chosen[fold]
        score_data = json.loads((RUN / f"fold_{fold}/scores/{key}.json").read_text())
        setting = score_data["setting"]
        epoch = key.split("__", 1)[0]
        training = json.loads((RUN / f"fold_{fold}/training.json").read_text())
        model = CellposeModel(gpu=torch.cuda.is_available(),
                              pretrained_model=training[epoch])
        for crop in held:
            mask = np.asarray(model.eval(
                crop["image"], diameter=None,
                cellprob_threshold=setting["cellprob_threshold"],
                flow_threshold=setting["flow_threshold"],
                min_size=setting["min_size"])[0]).astype(np.uint32)
            np.savez_compressed(output / f"{crop['name']}.npz", mask=mask)
            row = score(crop["mask"], mask)
            row.update(name=crop["name"], fold=fold, setting=key)
            rows.append(row)
            print(crop["name"], row["tp"], row["fn"], row["fp"], key,
                  flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    total = summarize(rows)
    p, r = total["precision"], total["recall"]
    total["f2"] = 5 * p * r / (4 * p + r) if 4 * p + r else 0.0
    (output / "scores.json").write_text(json.dumps({
        "method": "recall_candidate_held_out", "overall": total,
        "patches": rows, "selection": "F3 on other folds only"}, indent=2) + "\n")
    print("Overall", total, flush=True)


if __name__ == "__main__":
    main()
