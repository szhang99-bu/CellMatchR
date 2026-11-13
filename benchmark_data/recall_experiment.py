"""Source-grouped Nissl experiment that gives missed annotated cells priority.

Run on the lab workstation. Original checkpoints and production settings are
left intact. All evaluation crops come from source fields excluded from their
candidate model's training fold.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from finetune_cytodark import score, summarize
from train_full_cytodark import fold_map, load_crops, sources

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path.home() / "CellMatchR/Experiments/Nissl_Recall_20261113"
REFERENCE = HERE / "cytodark_full_results"
THRESHOLDS = (-3.0, -2.0, -1.0, 0.0)
FLOW_LIMITS = (0.4, 1.0)
MIN_SIZES = (0, 15)
EPOCHS = 11
SEED = 24


def reference_misses() -> dict[str, float]:
    """Use zero-shot baseline errors to emphasize hard training crops."""
    misses = {}
    for fold in range(4):
        record = json.loads((REFERENCE / f"fold_{fold}.json").read_text())
        for row in record["baseline"]["patches"]:
            misses[row["name"]] = row["fn"] / max(1, row["gt"])
    return misses


def training_probs(crops: list[dict]) -> np.ndarray:
    counts = defaultdict(int)
    for crop in crops:
        counts[crop["group"]] += 1
    misses = reference_misses()
    n = len(crops)
    weights = np.array([
        (0.2 / n + 0.8 / (len(counts) * counts[crop["group"]]))
        * (1.0 + 1.5 * misses[crop["name"]])
        for crop in crops
    ], dtype=np.float64)
    return weights / weights.sum()


def train_fold(crops: list[dict], folder: Path, name: str) -> dict:
    import torch
    from cellpose import models, train

    folder.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    net = models.CellposeModel(gpu=torch.cuda.is_available(),
                               pretrained_model="cpsam").net
    probs = training_probs(crops)
    path, losses, _ = train.train_seg(
        net, train_data=[crop["image"] for crop in crops],
        train_labels=[crop["mask"] for crop in crops], train_probs=probs,
        n_epochs=EPOCHS, learning_rate=1e-5, batch_size=1, bsize=256,
        min_train_masks=1, nimg_per_epoch=len(crops) + 1,
        save_path=str(folder), model_name=name, save_every=5,
        save_each=True)
    del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    final = Path(path)
    sixth = final.with_name(final.name + "_epoch_0005")
    if not sixth.is_file():
        raise FileNotFoundError(sixth)
    return {"epoch_6": str(sixth), "epoch_11": str(final),
            "losses": [float(value) for value in losses],
            "training_sources": sorted({crop["source"] for crop in crops}),
            "sampling": "80% tissue-balanced, 20% natural; 1+1.5*baseline_miss_fraction"}


def settings():
    for cellprob in THRESHOLDS:
        for flow in FLOW_LIMITS:
            for min_size in MIN_SIZES:
                yield (cellprob, flow, min_size)


def validation_settings():
    # The first sweep showed no recall change from the minimum-size setting;
    # retain the installed default while testing both flow limits in each fold.
    for cellprob in THRESHOLDS:
        for flow in FLOW_LIMITS:
            yield (cellprob, flow, 15)


def setting_key(setting: tuple[float, float, int]) -> str:
    cellprob, flow, min_size = setting
    return f"cellprob={cellprob:g}_flow={flow:g}_min={min_size}"


def evaluate(checkpoint: str, crops: list[dict], output: Path,
             tag: str, selected_settings=None) -> None:
    import torch
    from cellpose.models import CellposeModel

    output.mkdir(parents=True, exist_ok=True)
    model = CellposeModel(gpu=torch.cuda.is_available(),
                          pretrained_model=checkpoint)
    for setting in selected_settings or settings():
        key = setting_key(setting)
        target = output / f"{tag}__{key}.json"
        if target.is_file():
            continue
        cellprob, flow, min_size = setting
        rows = []
        for index, crop in enumerate(crops, 1):
            prediction = np.asarray(model.eval(
                crop["image"], diameter=None, flow_threshold=flow,
                cellprob_threshold=cellprob, min_size=min_size)[0])
            row = score(crop["mask"], prediction)
            row.update(name=crop["name"], source=crop["source"],
                       group=crop["group"])
            rows.append(row)
            if index % 40 == 0:
                print(f"{tag} {key}: {index}/{len(crops)}", flush=True)
        totals = summarize(rows)
        p, r = totals["precision"], totals["recall"]
        totals["f2"] = 5 * p * r / (4 * p + r) if 4 * p + r else 0.0
        target.write_text(json.dumps({"checkpoint": checkpoint,
                                      "setting": {"cellprob_threshold": cellprob,
                                                  "flow_threshold": flow,
                                                  "min_size": min_size},
                                      "overall": totals, "patches": rows},
                                     indent=2) + "\n")
        print(f"{tag} {key}: TP={totals['tp']} FN={totals['fn']} "
              f"FP={totals['fp']} recall={r:.4f} F2={totals['f2']:.4f}",
              flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def initial_sweep(output: Path) -> None:
    items = [item for item in sources()
             if item["group"] == "Macaque | Visual Cortex"]
    crops = load_crops(items)
    models = {"cpsam": "cpsam"}
    library = Path.home() / "CellMatchR/Models"
    for record in library.glob("cellpose_ch0_nissl_general_v1_*/model.json"):
        info = json.loads(record.read_text())
        models["nissl_general_v1"] = str(record.parent / info["model_relative_path"])
    for name, checkpoint in models.items():
        evaluate(checkpoint, crops, output / "initial_sweep", name)


def cross_validate(output: Path) -> None:
    items = sources()
    folds = fold_map(items)
    output.mkdir(parents=True, exist_ok=True)
    (output / "folds.json").write_text(json.dumps(folds, indent=2) + "\n")
    for fold in range(4):
        folder = output / f"fold_{fold}"
        record_path = folder / "training.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text())
        else:
            training = load_crops([item for item in items if folds[item["stem"]] != fold])
            print(f"Training recall candidate fold {fold}: {len(training)} crops",
                  flush=True)
            record = train_fold(training, folder, f"nissl_recall_fold_{fold}")
            record["held_out_sources"] = sorted(
                item["stem"] for item in items if folds[item["stem"]] == fold)
            record["completed_at"] = datetime.now().isoformat(timespec="seconds")
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            del training
            gc.collect()
        held = load_crops([item for item in items if folds[item["stem"]] == fold])
        for epoch in ("epoch_6", "epoch_11"):
            evaluate(record[epoch], held, folder / "scores", epoch,
                     validation_settings())
        del held
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("sweep", "cross_validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.action == "sweep":
        initial_sweep(args.output)
    else:
        cross_validate(args.output)


if __name__ == "__main__":
    main()
