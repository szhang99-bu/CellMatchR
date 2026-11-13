# Micro-SAM light microscopy fine-tuning for Nissl cells

## Question and data split

The original `vit_b_lm` model found 55 of 60 expert cells on the pictured
ID68 macaque crop, with all five misses touching a 24-pixel crop-edge band.
This experiment asked whether gentle fine-tuning could improve whole-field
recall without losing that useful behavior.

The [CytoDArk0](CytoDArk0_subset/README.md) expert masks are stored on the
lab PC outside Git. All four 512 × 512 crops from a 1024 × 1024 source stay
in one split. Training used 65 source fields (260 crops), validation used
three source fields (12 crops), and macaque source ID68 (four crops) was
excluded from both. ID69 was the macaque validation field for checkpoint and
proposal-threshold selection. ID68 was used only for the final comparison,
although its original-model result had already been inspected by the user.
Treat the result as exploratory, not a blind estimate across new slides.

## Training and selection

Run `train_microsam_lm_nissl.py train` on the lab GPU. Each RGB source is
converted to grayscale, normalized to the 0.5–99.8 percentile range within
each crop, inverted, and repeated across three channels. The source image
shown to users remains in its original colors. The original `vit_b_lm`
checkpoint is read only. Training uses 256-pixel patches, batch size 1,
learning rate `3e-6`, six epochs, and a frozen image encoder. Each epoch's
checkpoint is saved separately. The model with the lowest validation loss is
also exported for comparison.

`select_microsam_lm_recall.py` scores each saved epoch and the original model
on ID69 at foreground thresholds 0.2, 0.3, 0.4, and 0.5. It selects the
combination that finds the most expert cells, with extra proposals no greater
than twice the original default on ID69. Ties favor fewer interior misses,
then F3 and fewer extra proposals. Object minimum size is zero during this
comparison. The selected trained model is epoch 4 at foreground threshold
0.2; the original default uses 0.5. Masks are matched to expert cells
one-to-one at IoU ≥ 0.5.

## Results

| Source and model | Found | Missed | Extra | Interior misses |
| --- | ---: | ---: | ---: | ---: |
| ID69 validation, original LM | 210/233 | 23 | 103 | 13 |
| ID69 validation, selected fine-tuned LM | 220/233 | 13 | 78 | 9 |
| ID68 comparison, original LM | 177/204 | 27 | 97 | 12 |
| ID68 comparison, selected fine-tuned LM | 189/204 | 15 | 84 | 5 |
| Pictured ID68 crop 0,0, original LM | 55/60 | 5 | 21 | 0 |
| Pictured ID68 crop 0,0, selected fine-tuned LM | 54/60 | 6 | 27 | 2 |
| Neighboring ID68 crop 0,1, original LM | 54/61 | 7 | 31 | 6 |
| Neighboring ID68 crop 0,1, selected fine-tuned LM | 59/61 | 2 | 24 | 1 |

The lowest-loss epoch at its default threshold found only 169/204 cells on
ID68. Selection by found cells, using validation data only, was essential
for the user's high-recall goal. The fine-tuned model improves the full ID68
source but is **not** better on every crop: the original still wins on the
particular crop that motivated the experiment.

The fine-tuned model is installed as **Nissl LM Recall v1** in the shared
`~/CellMatchR/Models` library as an additional model. Step 3 and the Parameter
Wizard use its recorded 0.2 foreground threshold when that model is selected;
other models retain their existing settings. The original `vit_b_lm` remains
available to select and fine-tune again. The trained model is not chosen
automatically for a project. App object-size and intensity filters remain
user controlled, so changing them can change these raw-proposal counts.

The full lab experiment, checkpoints, score JSON, and masks are in
`~/CellMatchR/Experiments/Nissl_MicroSAM_LM_20261113` on the lab PC.
`render_microsam_lm_heldout.py` makes before/after PNG and PDF figures for
ID68 crops 0,0 and 0,1. A new independent slide set is needed before claiming
general performance beyond this dataset.
