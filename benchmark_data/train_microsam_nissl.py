"""Fine-tune the histopathology Micro-SAM model on RGB Nissl fields.

The macaque visual-cortex source 68 is kept completely out of training and
validation. Source 69 and two non-macaque fields are used for validation.
Original cached weights are never modified; all outputs go to a new experiment
folder in the shared lab workstation library.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from finetune_cytodark import score, summarize
from train_full_cytodark import DATA, sources

OUTPUT = Path.home() / "CellMatchR/Experiments/Nissl_MicroSAM_20261113"
TEST_SOURCE = "ID68_Visual_Cortex_Macaque"
VALIDATION_SOURCE = "ID69_Visual_Cortex_Macaque"


def partition() -> dict[str, list[dict]]:
    items = sources()
    other = [item for item in items if item["group"] != "Macaque | Visual Cortex"]
    # Reserve two sources outside macaque as an additional validation check.
    validation_sources = {VALIDATION_SOURCE}
    validation_sources.update(item["stem"] for item in other[:2])
    return {
        "train": [item for item in items
                  if item["stem"] != TEST_SOURCE
                  and item["stem"] not in validation_sources],
        "validation": [item for item in items if item["stem"] in validation_sources],
        "test": [item for item in items if item["stem"] == TEST_SOURCE],
    }


def make_loader(items: list[dict], *, training: bool):
    from micro_sam.training import default_sam_loader
    from torch_em.data.sampler import MinInstanceSampler

    return default_sam_loader(
        raw_paths=[str(item["image"]) for item in items], raw_key=None,
        label_paths=[str(item["mask"]) for item in items], label_key=None,
        patch_shape=(256, 256), with_segmentation_decoder=True,
        batch_size=1, num_workers=2 if training else 0,
        n_samples=max(12, 4 * len(items)) if training else max(8, 2 * len(items)),
        is_train=training, min_size=5,
        sampler=MinInstanceSampler(2, min_size=5),
        max_sampling_attempts=1000,
    )


def train(output: Path, epochs: int) -> None:
    import torch
    from micro_sam.training import train_sam
    from micro_sam.util import export_custom_sam_model

    if not torch.cuda.is_available():
        raise RuntimeError("This Micro-SAM experiment requires the lab GPU")
    splits = partition()
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "starting_model": "vit_b_histopathology",
        "test_sources": [item["stem"] for item in splits["test"]],
        "validation_sources": [item["stem"] for item in splits["validation"]],
        "training_sources": [item["stem"] for item in splits["train"]],
        "epochs": epochs, "learning_rate": 1e-5, "batch_size": 1,
        "patch_size": 256, "freeze_encoder": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output / "split.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Micro-SAM RGB training: {len(splits['train'])} source fields; "
          f"validation {len(splits['validation'])}; test {len(splits['test'])}",
          flush=True)
    train_sam(
        name="nissl_microsam_histopathology", model_type="vit_b_histopathology",
        train_loader=make_loader(splits["train"], training=True),
        val_loader=make_loader(splits["validation"], training=False),
        n_epochs=epochs, early_stopping=4, n_sub_iteration=2,
        n_objects_per_batch=4, with_segmentation_decoder=True,
        freeze=["image_encoder"], lr=1e-5,
        save_root=str(output), verify_n_labels_in_loader=4,
        save_every_kth_epoch=2,
    )
    best = output / "checkpoints/nissl_microsam_histopathology/best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    exported = output / "nissl_microsam_histopathology.pth"
    export_custom_sam_model(best, model_type="vit_b", save_path=exported,
                            with_segmentation_decoder=True)
    print(f"Exported candidate: {exported}", flush=True)


def evaluate(output: Path, *, baseline: bool = False) -> None:
    import torch
    from PIL import Image
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter

    checkpoint = output / "nissl_microsam_histopathology.pth"
    if not baseline and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    predictor, segmenter = get_predictor_and_segmenter(
        model_type="vit_b_histopathology" if baseline else "vit_b",
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint=None if baseline else str(checkpoint))
    source = partition()["test"][0]
    rgb = np.asarray(Image.open(source["image"]).convert("RGB"))
    import tifffile
    labels = tifffile.imread(source["mask"])
    all_rows = {size: [] for size in (0, 10, 20)}
    for row in range(2):
        for col in range(2):
            name = f"{source['stem']}_{row}_{col}"
            sl = np.s_[row*512:(row+1)*512, col*512:(col+1)*512]
            image = rgb[sl]
            truth = labels[sl]
            predictor.set_image(image)
            embedding = {"features": predictor.get_image_embedding(),
                         "original_size": predictor.original_size,
                         "input_size": predictor.input_size}
            segmenter.initialize(image=image, image_embeddings=embedding)
            for size in all_rows:
                result = segmenter.generate(min_size=size)
                if isinstance(result, np.ndarray):
                    prediction = result.astype(np.uint32)
                elif isinstance(result, list):
                    prediction = np.zeros(truth.shape, dtype=np.uint32)
                    for index, proposal in enumerate(result, 1):
                        prediction[proposal["segmentation"]] = index
                else:
                    raise TypeError(f"Unexpected Micro-SAM output: {type(result)}")
                row_score = score(truth, prediction)
                row_score["name"] = name
                all_rows[size].append(row_score)
                target = output / ("baseline_masks" if baseline else "heldout_masks") / f"min_{size}"
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / f"{name}.npz", mask=prediction)
                print(f"{name} min_size={size}: TP={row_score['tp']} "
                      f"FN={row_score['fn']} FP={row_score['fp']}", flush=True)
    results = {str(size): {"overall": summarize(rows), "patches": rows}
               for size, rows in all_rows.items()}
    score_file = "baseline_scores.json" if baseline else "heldout_scores.json"
    (output / score_file).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({size: data["overall"] for size, data in results.items()},
                     indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate", "evaluate_baseline"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.action == "train":
        train(args.output, args.epochs)
    else:
        evaluate(args.output, baseline=args.action == "evaluate_baseline")


if __name__ == "__main__":
    main()
