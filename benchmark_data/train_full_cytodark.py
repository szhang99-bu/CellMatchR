"""Grouped validation and final Cellpose-SAM training on CytoDArk0 20x Nissl.

All 512 px crops of an original 1024 px field stay in one fold. The paired
40x views are excluded because they show the same fields. Training starts
from the unmodified ``cpsam`` checkpoint in every fold and for the final run.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from finetune_cytodark import score, summarize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CytoDArk0_full20x"
RESULTS = ROOT / "cytodark_full_results"
LR = 1e-5
EPOCHS = 11
SEED = 24


def sources() -> list[dict]:
    metadata = pd.read_excel(DATA / "metadata/cytoDArk0_metadata.xlsx",
                             sheet_name="cytoDArk0_20x_1024")
    result = []
    for row in metadata.itertuples(index=False):
        stem = str(row.ID)
        image = DATA / "image" / f"{stem}.png"
        mask = DATA / "label" / f"{stem}.tiff"
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Incomplete source pair: {stem}")
        result.append({"stem": stem, "group": f"{row.Species} | {row.Area}",
                       "image": image, "mask": mask})
    if len(result) != 69 or len({x["stem"] for x in result}) != 69:
        raise ValueError("Expected 69 distinct original 20x fields")
    return result


def fold_map(items: list[dict]) -> dict[str, int]:
    """Keep source fields together and spread each tissue group across folds."""
    rng = np.random.default_rng(SEED)
    groups = defaultdict(list)
    for item in items:
        groups[item["group"]].append(item["stem"])
    assignment = {}
    for group, stems in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        rng.shuffle(stems)
        # Offset smaller groups toward currently smaller folds.
        order = sorted(range(4), key=lambda fold: sum(v == fold for v in assignment.values()))
        for index, stem in enumerate(stems):
            assignment[stem] = order[index % 4]
    if len(assignment) != 69:
        raise AssertionError("Fold assignment lost a source field")
    return assignment


def normalize(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, [0.5, 99.8])
    if high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)
    return ((np.clip(gray, low, high) - low) / (high - low) * 255).astype(np.uint8)


def crop_source(item: dict):
    image = np.asarray(Image.open(item["image"]).convert("L"), dtype=np.float32)
    mask = tifffile.imread(item["mask"])
    if image.shape != (1024, 1024) or mask.shape != image.shape:
        raise ValueError(f"Unexpected shape in {item['stem']}: {image.shape}, {mask.shape}")
    for row in range(2):
        for col in range(2):
            sl = np.s_[row * 512:(row + 1) * 512, col * 512:(col + 1) * 512]
            yield f"{item['stem']}_{row}_{col}", normalize(image[sl]), mask[sl].astype(np.int32)


def load_crops(items: list[dict]) -> list[dict]:
    result = []
    for item in items:
        for name, image, mask in crop_source(item):
            result.append({"name": name, "source": item["stem"],
                           "group": item["group"], "image": image, "mask": mask})
    return result


def train_model(crops: list[dict], folder: Path, model_name: str,
                epochs: int = EPOCHS):
    import torch
    from cellpose import models, train

    folder.mkdir(parents=True, exist_ok=True)
    net = models.CellposeModel(gpu=torch.cuda.is_available() or torch.backends.mps.is_available(),
                               pretrained_model="cpsam").net
    group_counts = defaultdict(int)
    for crop in crops:
        group_counts[crop["group"]] += 1
    n_groups = len(group_counts)
    n = len(crops)
    # Half natural sampling, half equally weighted tissue groups. This uses
    # all images without letting the 53 dolphin fields wholly dominate.
    probs = np.array([0.5 / n + 0.5 / (n_groups * group_counts[c["group"]])
                      for c in crops], dtype=np.float64)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    path, losses, _ = train.train_seg(
        net, train_data=[c["image"] for c in crops],
        train_labels=[c["mask"] for c in crops], train_probs=probs,
        n_epochs=epochs, learning_rate=LR, batch_size=1, bsize=256,
        min_train_masks=1, nimg_per_epoch=n + 1, save_path=str(folder),
        model_name=model_name, save_every=5, save_each=True)
    del net
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return Path(path), [float(v) for v in losses]


def evaluate(checkpoint: str, crops: list[dict]) -> dict:
    import torch
    from cellpose.models import CellposeModel

    model = CellposeModel(gpu=torch.cuda.is_available() or torch.backends.mps.is_available(),
                          pretrained_model=checkpoint)
    rows = []
    for i, crop in enumerate(crops, 1):
        prediction = np.asarray(model.eval(crop["image"], diameter=None,
                                           flow_threshold=0.4,
                                           cellprob_threshold=0.0)[0])
        row = score(crop["mask"], prediction)
        row.update(name=crop["name"], source=crop["source"], group=crop["group"])
        rows.append(row)
        if i % 20 == 0 or i == len(crops):
            print(f"  scored {i}/{len(crops)} patches: {checkpoint}", flush=True)
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {"overall": summarize(rows),
            "by_group": {group: summarize([r for r in rows if r["group"] == group])
                         for group in sorted({r["group"] for r in rows})},
            "patches": rows}


def validate() -> None:
    items = sources()
    folds = fold_map(items)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "folds.json").write_text(json.dumps(folds, indent=2) + "\n")
    for fold in range(4):
        record = RESULTS / f"fold_{fold}.json"
        if record.is_file():
            print(f"Fold {fold} already scored; skipping", flush=True)
            continue
        train_items = [x for x in items if folds[x["stem"]] != fold]
        held_items = [x for x in items if folds[x["stem"]] == fold]
        train_crops, held_crops = load_crops(train_items), load_crops(held_items)
        print(f"Fold {fold}: train {len(train_items)} fields/{len(train_crops)} patches, "
              f"held out {len(held_items)} fields/{len(held_crops)} patches", flush=True)
        baseline = evaluate("cpsam", held_crops)
        folder = RESULTS / f"fold_{fold}_temporary"
        final_path, losses = train_model(train_crops, folder, f"nissl_fold_{fold}")
        six_path = final_path.with_name(final_path.name + "_epoch_0005")
        if not six_path.is_file():
            raise FileNotFoundError(six_path)
        six = evaluate(str(six_path), held_crops)
        eleven = evaluate(str(final_path), held_crops)
        record.write_text(json.dumps({"fold": fold,
                                      "training_sources": [x["stem"] for x in train_items],
                                      "held_out_sources": [x["stem"] for x in held_items],
                                      "losses": losses, "baseline": baseline,
                                      "epoch_6": six, "epoch_11": eleven}, indent=2) + "\n")
        print(f"Fold {fold} F1: base={baseline['overall']['f1']:.4f}, "
              f"6={six['overall']['f1']:.4f}, 11={eleven['overall']['f1']:.4f}", flush=True)
        shutil.rmtree(folder)
        del train_crops, held_crops
        gc.collect()


def summarize_validation() -> dict:
    results = [json.loads((RESULTS / f"fold_{i}.json").read_text()) for i in range(4)]
    summary = {}
    for kind in ("baseline", "epoch_6", "epoch_11"):
        patches = [p for result in results for p in result[kind]["patches"]]
        summary[kind] = {"overall": summarize(patches),
                         "by_group": {group: summarize([p for p in patches if p["group"] == group])
                                      for group in sorted({p["group"] for p in patches})},
                         "by_source": {source: summarize([p for p in patches if p["source"] == source])
                                       for source in sorted({p["source"] for p in patches})}}
    # A selected setting must improve the pooled out-of-fold score, while
    # avoiding needless extra training if the two settings are effectively tied.
    six, eleven = summary["epoch_6"]["overall"]["f1"], summary["epoch_11"]["overall"]["f1"]
    summary["selected_epochs"] = 11 if eleven > six + 0.002 else 6
    for kind in ("epoch_6", "epoch_11"):
        changes = [summary[kind]["by_source"][source]["f1"] -
                   summary["baseline"]["by_source"][source]["f1"]
                   for source in summary["baseline"]["by_source"]]
        summary[kind]["source_fields_improved"] = sum(change > 0 for change in changes)
        summary[kind]["source_fields_declined"] = sum(change < 0 for change in changes)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def final_train() -> Path:
    summary = summarize_validation()
    selected = summary["selected_epochs"]
    all_crops = load_crops(sources())
    print(f"Final training: all 69 original fields/{len(all_crops)} patches, "
          f"{selected} epochs, source cpsam", flush=True)
    library = Path.home() / "CellMatchR" / "Models"
    library.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = library / f"cellpose_ch0_nissl_general_v1_{stamp}"
    folder.mkdir()
    path, losses = train_model(all_crops, folder, "trained_model", selected)
    if selected == 11:
        path.with_name(path.name + "_epoch_0005").unlink(missing_ok=True)
    record = {"backend": "cellpose", "model_path": str(path),
              "model_relative_path": str(path.relative_to(folder)),
              "name": "Nissl General v1", "channel": 0, "source_project": "CytoDArk0",
              "starting_model_name": "Cellpose-SAM", "parent_model_path": "",
              "training_rounds": 1, "recorded_epochs": selected,
              "history_complete": True, "epochs": selected,
              "advanced": {"learning_rate": LR, "batch_size": 1, "patch_size": 256,
                           "sampling": "half natural, half tissue-group balanced"},
              "preprocess": "raw_percentile", "base_model": "cpsam",
              "training_samples": [x["stem"] for x in sources()],
              "validation": "4-fold original-field grouped out-of-fold; see summary.json",
              "completed_at": datetime.now().isoformat(timespec="seconds")}
    (folder / "model.json").write_text(json.dumps(record, indent=2) + "\n")
    (RESULTS / "final_model.json").write_text(json.dumps({"path": str(path),
             "selected_epochs": selected, "losses": losses}, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "summarize", "final", "all"))
    args = parser.parse_args()
    if args.action in ("validate", "all"):
        validate()
    if args.action in ("summarize", "all"):
        print(json.dumps(summarize_validation(), indent=2), flush=True)
    if args.action in ("final", "all"):
        print(f"Final model: {final_train()}", flush=True)


if __name__ == "__main__":
    main()
