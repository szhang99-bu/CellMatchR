# Nissl recall experiment on the lab PC

## Goal

Prioritize separate masks for expert-annotated cells, even if that creates
additional candidates for later review. Keep original models and production
settings intact. The original Cellpose-SAM and Nissl General v1 checkpoints
must remain available as starting points for later training.

## Files

- Canonical training data: `/home/shiyuzhang/CellMatchR/Training_Data/CytoDArk0`
- Source data integrity: `manifest.sha256` in that folder, verified for 200 files
- Original Cellpose-SAM checkpoint:
  `/home/shiyuzhang/Downloads/CellMatchR-linux-env/model-cache/cellpose/cpsam`
- Existing Nissl General v1 model:
  `/home/shiyuzhang/CellMatchR/Models/cellpose_ch0_nissl_general_v1_20261113_095825`
- New experiment output:
  `/home/shiyuzhang/CellMatchR/Experiments/Nissl_Recall_20261113`
- Python runtime: `/home/shiyuzhang/Downloads/CellMatchR-linux-env/bin/python`
- Scripts: this repository's `benchmark_data` folder

The training data and model checkpoints are outside Git. No original model is
overwritten by this experiment. The 16 macaque crops come from four distinct
1024 × 1024 source images, IDs 66–69.

## Training and evaluation

`recall_experiment.py sweep` scored the original Cellpose-SAM and existing
Nissl General v1 models at four cell-probability thresholds, two flow limits,
and two minimum sizes on the 16 macaque crops. These Nissl General v1 scores
are training-image checks, not held-out validation.

`recall_experiment.py cross_validate` trains four new models, each from the
unmodified `cpsam` starting checkpoint. All crops from a source image stay in
one fold. The training schedule is 11 epochs at learning rate 1e-5, batch size
1 and 256-pixel training patches. Sampling is 80% balanced by tissue group and
20% natural frequency, multiplied by `1 + 1.5 ×` the fraction of expert cells
missed by the original model on that training crop. This changes which crops
are seen more often; it does not change the Cellpose loss or CellMatchR's
processing methods. Both six-epoch and 11-epoch checkpoints are retained.

Each fold scores both checkpoints at cell-probability thresholds −3, −2, −1,
and 0 and flow thresholds 0.4 and 1.0, with Cellpose `min_size=15`. Results
include one-to-one cell matching at IoU ≥ 0.5, TP, FP, FN, precision, recall,
F1, and F2. The threshold and checkpoint for any target fold can be chosen
using only the other folds, then evaluated on that held-out fold. The final
selection uses F3, which gives recall more weight than precision.

To resume a stopped run, rerun the same `cross_validate` command. Completed
fold training records and individual score files are skipped. A fold without a
`training.json` is retrained from `cpsam`.

## Comparison images

`compare_macaque_methods.py` saves masks and scores for Cellpose-SAM,
Nissl General v1, Cellpose 2, the published rat Nissl model, Micro-SAM,
StarDist, and marker-controlled watershed on the same macaque crops.
`tune_watershed.py` chooses its parameters on IDs 66, 67, and 69, leaving
ID 68 out of tuning. `render_macaque_comparison.py` overlays matched masks,
unmatched masks, and missed expert cells for visual review.

After Cellpose scoring, `train_microsam_nissl.py` fine-tunes the original
`vit_b_histopathology` Micro-SAM model on the original RGB stain fields.
Macaque source ID68 is held out for final testing; ID69 and two non-macaque
sources are validation fields. Training freezes the image encoder and saves a
new exported checkpoint in `~/CellMatchR/Experiments/Nissl_MicroSAM_20261113`.
The original cached Micro-SAM checkpoint stays untouched. Its test script
compares masks at minimum sizes 0, 10, and 20 on all four ID68 crops.

## Results and status

The four source-held-out folds are complete. Across 39,984 expert cells, the
cross-fitted F3 selection found 34,663 (86.69% recall) and missed 5,321, with
11,750 extra proposals. On the 16 macaque crops (843 expert cells), it found
766 (90.87% recall), missed 77, and made 201 extra proposals. The original
Cellpose-SAM default found about 89% of macaque expert cells. The new model
does not win on every crop; on ID68 quadrant 0,0 it found 51/60, whereas the
original found 54/60. The pretrained RGB Micro-SAM histopathology model found
780/843 (92.5%) with 319 extra proposals across the same 16 crops, making it
the strongest raw-recall pretrained model in this comparison. Nissl General v1
was trained on these fields and is not a held-out comparison.

Raw proposals across the same 16 macaque crops, matched one-to-one to expert
cells at IoU ≥ 0.5 (843 expert cells total):

| Model and input | Found | Missed | Extra |
| --- | ---: | ---: | ---: |
| Pretrained Micro-SAM histopathology, original RGB | 780 | 63 | 319 |
| New Cellpose recall candidate, source-held-out | 766 | 77 | 201 |
| Cellpose-SAM, permissive setting | 761 | 82 | 336 |
| Cellpose-SAM, default | 752 | 91 | 188 |
| StarDist fluorescence, inverted gray | 737 | 106 | 580 |
| Cellpose 2, inverted gray | 726 | 117 | 240 |
| Micro-SAM light microscopy, inverted gray | 717 | 126 | 447 |
| Cellpose 2, native gray | 704 | 139 | 427 |
| Rat Nissl v2, native gray | 704 | 139 | 824 |
| StarDist H&E, original RGB | 701 | 142 | 190 |
| Tuned watershed, native gray | 680 | 163 | 460 |

The older Nissl General v1 produced 771 found, 72 missed, and 216 extra with
a permissive setting, but it used these same source images during training, so
that row is intentionally excluded from the held-out ranking. Watershed was
tuned on three of the four source images and only ID68 was fully held out from
parameter selection. All counts are **before** CellMatchR's intensity wizard
filter; removing extra proposals afterward may also remove true cells.

The all-data Cellpose model is registered separately as **Nissl Recall v1** in
`~/CellMatchR/Models/cellpose_ch0_nissl_recall_v1_20261113_113939`. It starts
from the original `cpsam` weights, trains on all 69 fields, and keeps the
six-epoch and 11-epoch checkpoints in `final_training/models`. The registered
model uses the six-epoch checkpoint and records the tested prediction settings
of cell probability −2, flow threshold 1, and minimum size 15. Step 3 loads
the first two settings when the model is selected; its object-size filter
remains user controlled. This final all-data model has no independent held-out
score; the source-held-out folds above are the performance estimate.

A real ID68 RGB crop was processed through CellMatchR's brightfield
slide-scanner path with the registered model and produced a 512 × 512 channel-0
mask with 63 candidates. Its original RGB TIFF was unchanged and the temporary
test project was removed.

The Micro-SAM RGB fine-tuning job finished on the lab PC after the Cellpose
run. Its original model found 188/204 annotated cells (92.16% recall) on the
completely held-out ID68 field, with 81 extra proposals at minimum size 0.
The fine-tuned model selected by validation loss found only 179/204 (87.75%
recall), with 63 extra proposals. Minimum sizes 10 and 20 did not change
recall. Because missing cells are the primary cost, this candidate is retained
as a research checkpoint and was **not** registered as a selectable model or
substituted for the original. `select_microsam_recall.py` checked earlier
checkpoints and more permissive proposal settings on ID69, then scored its
choice on ID68. It selected epoch 2 with center and boundary thresholds 0.5,
foreground threshold 0.2, and minimum size 0. This recovered 185/204 cells
(90.69% recall) with 85 extra proposals: better than the fine-tuned default,
still worse than the original's 188/204 and 81 extra. The selected checkpoint
is retained in the experiment folder but is not registered for app use. This
follow-up is exploratory because the default-setting ID68 result was already
examined. New lab slides should supply independent held-out images.

Comparison figures are in the Cellpose experiment's `figures` directory.
`macaque_all_models` overlays human outlines and 11 candidate methods on the
same crop; `macaque_before_after` focuses on the original and recall-trained
Cellpose masks; `macaque_recall_tradeoff` plots misses against extra proposals.
The Micro-SAM experiment's `figures` directory contains before/after and
validation-selected recall overlays on the held-out macaque image.
Check `cross_validation.log`, `summary.json`, and `fold_*/scores` for the full
source-held-out calculations. Fold checkpoints are research artifacts and do
not replace the original model or become selected automatically.
