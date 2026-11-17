# CellMatchR user guide

CellMatchR turns microscopy images into cell masks, lets you correct and review
them, produces measurements, and can fine-tune models from reviewed examples.
The [README](../README.md) is the short installation guide; this page explains
the full workflow and answers common questions.

## Contents

- [Install, launch, and update](#install-launch-and-update)
- [Choose the right input](#choose-the-right-input)
- [Step 1: Create or open a project](#step-1-create-or-open-a-project)
- [Step 2: Select images and regions](#step-2-select-images-and-regions)
- [Step 3: Segment cells](#step-3-segment-cells)
- [Step 4: Correct and validate predictions](#step-4-correct-and-validate-predictions)
- [Step 5: Analyze results](#step-5-analyze-results)
- [Step 6: Run the pipeline](#step-6-run-the-pipeline)
- [Step 7: Retrain a model](#step-7-retrain-a-model)
- [Understand files and outputs](#understand-files-and-outputs)
- [Frequently asked questions](#frequently-asked-questions)
- [Troubleshooting by symptom](#troubleshooting-by-symptom)
- [Command-line reference](#command-line-reference)

## Install, launch, and update

### Where to type the commands

| Computer | Open a command window |
|---|---|
| macOS | Press **⌘ Space**, type **Terminal**, then press Return. |
| Windows | Open **Start**, type **PowerShell**, and open it. |
| Linux | Open the **Terminal** app from your applications menu. |

Type each line in a code block into that window and press Return or Enter.
The command window shows where you are and prints results or errors. You do
not type the word `bash` above a code block.

### Get an updateable copy of CellMatchR

The first command to run in your command window is:

```bash
git --version
```

If it prints a version number, Git is ready. If Git is not found,
[install Git](https://git-scm.com/downloads), reopen the command window, and
run `git --version` again. Then move to your Desktop using the command for
your system:

**macOS or Linux Terminal:**

```bash
cd ~/Desktop
```

**Windows PowerShell:**

```powershell
cd ([Environment]::GetFolderPath('Desktop'))
```

The Windows command finds your actual Desktop location, including a Desktop
redirected to OneDrive. If `cd ~/Desktop` says the folder does not exist on
Linux, open your Desktop in the file manager and choose **Open in Terminal**
there.

The clone command below creates the app folder on your **Desktop by default**.
To put it somewhere else, replace the Desktop `cd` command above with the
path to an existing folder. For example, use `cd ~/Documents` on macOS/Linux
or `cd "D:\Apps"` in Windows PowerShell. Keep quotes around a path that
contains spaces. Now clone the repository and enter its folder:

```bash
git clone https://github.com/szhang99-bu/CellMatchR.git CellMatchR-app
cd CellMatchR-app
```

`git clone` creates **CellMatchR-app** on your Desktop by default, or in the
other folder you chose; the last `cd` enters it. You can open that folder in
Finder or File Explorer. The app folder has a different location from the
shared model library at `~/CellMatchR/Models`, so trained models stay outside the Git
checkout. Cloning retains Git history, allowing later updates with `git pull`.

**Already have an app folder?** Do not clone it again. Open a command
window and enter that existing folder instead:

- **macOS:** Type `cd ` including the space, drag the CellMatchR app folder from
  Finder onto the Terminal window, then press Return. Finder inserts its full
  path, including spaces in folder names.
- **Windows:** Open the CellMatchR app folder in File Explorer. Click the address
  bar, type `powershell`, and press Enter. PowerShell opens in that folder.
- **Linux:** Use your file manager's **Open in Terminal** option for the
  app folder, or run `cd /full/path/to/CellMatchR-app` in Terminal.

To confirm you are in the right folder, type `pwd`, then `ls environment.yml`.
The first command prints your current location; the second should list the
`environment.yml` file. If you see *file not found*, enter the CellMatchR
app folder and try again. Run environment commands only after this check.

**Code → Download ZIP** works if Git is unavailable, but a ZIP cannot use
`git pull`; every upgrade requires downloading and replacing application
files. Extract the ZIP first, enter the extracted folder as above, and keep
project output folders separate from the application folder when possible.

### Windows setup

The Windows setup script installs a Conda-compatible micromamba environment
for CellMatchR. You do **not** need to install Conda or Miniforge separately
for this Windows workflow.

1. Open the `CellMatchR-app` folder on your Desktop (or at the location you
   chose) in File Explorer.
2. Double-click `setup-windows.cmd` and wait for it to report completion. It
   creates the app environment and a desktop shortcut.
3. Double-click **Launch CellMatchR (Windows)**. `run-windows.cmd` in the
   app folder is another way to start it.

### macOS setup

1. In **Terminal**, type `conda --version`. A version number means Conda is
   available. Existing Miniforge, Miniconda, or Anaconda installations can be
   used if this command works. If it says `command not found`, check whether
   you already installed Conda but have not initialized it for Terminal; see
   your Conda installer's shell setup instructions before installing a second
   copy.
2. If Conda is absent, install [Miniforge's macOS package](https://github.com/conda-forge/miniforge#macos-pkg-installers).
   In **Apple menu → About This Mac**, check whether the chip is Apple Silicon
   or Intel and choose the corresponding `.pkg`. During installation, select
   shell initialization. Close Terminal, open it again, and check
   `conda --version` once more. Continue only when it prints a version.
3. Enter the app folder if needed, then type `ls environment.yml` to
   confirm you are beside the file. Create the app environment:

   ```bash
   conda env create -f environment.yml
   ```

   This creates an environment named `cellmatchr`. If you have already set it
   up, `conda env list` shows it; you can skip creating it again. To update an
   existing environment after a code update, use the command below under
   [Update an existing Git installation](#update-an-existing-git-installation).
4. Open `CellMatchR-app` on your Desktop (or at the location you chose) in
   Finder and open the app matching
   your macOS version:
   `CellMatchR (macOS 27+).app` or
   `CellMatchR (macOS 26 and earlier).app`. If your checkout is elsewhere,
   navigate to that folder in Finder instead. Keep the `.app` inside the
   app folder because it launches the code stored there.

### Linux setup

1. In Terminal, type `conda --version`. If it prints a version, continue. If
   Conda is absent, follow the [Miniforge Linux installation instructions](https://github.com/conda-forge/miniforge#unix-like-platforms-macos-linux--wsl),
   close and reopen Terminal, then check `conda --version` again.
2. Enter the app folder and confirm `ls environment.yml` finds the
   file. Run:

   ```bash
   PIP_CONSTRAINT="$PWD/scripts/linux-pip-constraints.txt" \
     conda env create --prefix "$PWD/../CellMatchR-linux-env" --file environment.yml
   bash scripts/install_linux_launcher.sh
   ```

3. Open **CellMatchR (Linux)** from the applications menu, or run
   `bash run-linux.sh` from the app folder to see a terminal log.

### Update an existing Git installation

Close CellMatchR and any running analysis or training first. In the app
folder, run `git status --short`. If it prints nothing, run
`git pull --ff-only` to update the code. If it lists files, check whether
they are local app edits or project data you placed in the app folder. Keep a
copy of anything important and resolve those changes before pulling; Git may
stop rather than overwrite them. Project data and the shared model library
are separate from the repository when you follow the setup above.

After an update that changes dependencies, rerun `setup-windows.cmd` on Windows.
On macOS run `conda env update -f environment.yml --prune`. On Linux run:

```bash
PIP_CONSTRAINT="$PWD/scripts/linux-pip-constraints.txt" \
  conda env update --prefix "$PWD/../CellMatchR-linux-env" --file environment.yml --prune
```

If you moved the checkout, recreate the Windows shortcut with
`.\setup-windows.cmd -ShortcutOnly` from PowerShell, or rerun
`bash scripts/install_linux_launcher.sh` on Linux.

## Choose the right input

Choose the acquisition type in **Step 2 → Input & Paths**. The two modes can
both encounter a `.lif` file; the file extension alone does not identify the
acquisition. CellMatchR checks its contents and skips a LIF in the wrong mode.

| Data | Mode | What happens |
|---|---|---|
| Zeiss `.czi` | **Confocal** | Read directly, one acquisition per sample. |
| Leica confocal `.lif` | **Confocal** | Read directly; stitched slide-scanner LIFs are skipped here. |
| `.tif` / `.tiff` microscopy images | **Confocal** | Read directly. |
| Nikon `.nd2`, Olympus `.oib` / `.oif` | **Confocal** | Available as beta readers; check a test image first. |
| Olympus slide-scanner `.vsi` | **Slide scanner** | Draw and export named, full-resolution regions. Keep its companion folder beside the VSI. |
| Stitched Leica slide-scanner `.lif` | **Slide scanner** | Draw and export regions; confocal LIFs are skipped here. |
| Large TIFF or BigTIFF slide | **Slide scanner** | Draw and export regions, including from an already converted scan. |
| Olympus `.oir` | Neither | Convert to TIFF in Fiji/ImageJ or another reader before using CellMatchR. |

For a VSI named `sample.vsi`, preserve the matching `_sample_` companion folder
next to it. Missing companion data can prevent the converter from opening the
scan. Keep the original acquisition files until you have checked exported
crops and masks.

TIFF previews use a stored image pyramid when available. For TIFFs with
explicit time and Z axes, the preview uses the first time point and a maximum
intensity projection through Z. Some tiled, compressed TIFFs need Zarr 3 in
the CellMatchR environment. The **Confocal** input browser can scan files
directly in the chosen root or one level of subfolders; **Skip subfolders** in
Step 3 limits the run to files directly in the root.

For slide scans, select **Laser fluorescence** when cells are bright labeled
signals, or **Brightfield stain** for dark stained cells such as Nissl. The
brightfield option adapts model input and the Parameter Wizard internally;
the image you see stays at its original stain brightness. Brightfield is
treated as one combined cell signal, **channel 0**.

## Step 1: Create or open a project

1. Choose **New Project**, enter a name, and select a project output folder.
2. Click **Create Project Folder**. CellMatchR creates
   `cellmatchr_project.json` and output subfolders there.
3. For an existing project, choose **Open Existing Project** and select its
   `cellmatchr_project.json` file. Check the input folder and selected run
   before starting another stage, especially after moving a project.

The project file stores paths and settings; masks, tables, and crops live in
the project output folder. You can open an earlier project with the current
app. If a project has been moved, CellMatchR resolves the output folder from
the selected project file and can find outputs stored there; external raw
image paths may still need to be selected again in Step 2.

## Step 2: Select images and regions

### Confocal or direct-read images

Select **Confocal**, browse to the input root, and click **Confirm Input
Folder**. A common layout is one acquisition per subfolder:

```text
input-root/
  sample-01/sample-01.czi
  sample-02/sample-02.czi
```

Inspect the file list before proceeding. If a `.lif` is marked as a
slide-scanner acquisition, choose **Slide scanner** for that file instead.
You can also use TIFF files as direct input.

### Slide-scanner images

1. Select **Slide scanner**, then **Laser fluorescence** or **Brightfield
   stain**.
2. Click **Launch Slide Scanner Converter** and choose the VSI, stitched LIF,
   or TIFF slide.
3. Draw and name the regions you need. Save and export them as full-resolution
   TIFF crops and matching region masks.
4. Return to Step 3. The exported TIFF folder is selected automatically.

The normal export location is `VSI_Converted_TIFF/` inside the project. If you
convert the same or another scan later, inspect the output choice: existing
exports can be replaced or kept in a numbered folder. Step 3 lets you choose
among multiple converted outputs. The optional output and input folder
overrides in Step 2 are for data already stored elsewhere.

## Step 3: Segment cells

### Select channels and a model

Click **Load Preview** to inspect channels; click the preview to toggle a
channel for prediction. You can also type zero-based IDs in **Channels to
predict**, separated by commas, such as `0,2`. For brightfield slide scans,
use `0` because the RGB stain is combined into one cell signal.

| Backend | Models shown in the app | Use it when |
|---|---|---|
| **Micro-SAM** | Built-in model types such as `vit_b_lm`, plus saved models | You want SAM-based masks or have a compatible fine-tuned checkpoint. |
| **Cellpose** | Cellpose-SAM (`cpsam`), Cellpose 2 (`cyto2`), Rat Nissl v2, plus saved models | You want Cellpose behavior or to compare its Nissl-specific option. |
| **StarDist 2D** | Built-in `2D_versatile_fluo`, plus saved models | Cells are reasonably star-convex and StarDist dependencies are installed. |

Select the **Segmentation backend** and then **Model for this run**. Saved
models in the shared library appear in the dropdown; an external checkpoint
can be browsed separately. Cellpose 2 and Rat Nissl v2 use a separate legacy
runtime, which is prepared on first use. Rat Nissl v2 came from juvenile rat
cresyl-violet images; test it on your own tissue before relying on it.
Changing backend or model does not delete other segmentation runs.

### Adjust sensitivity and filtering

Start with a **Test run** on the first file. For Cellpose, the main controls
are:

| Control | Starting value | If real cells are missed | If background is detected |
|---|---:|---|---|
| Cell diameter | `0` (automatic) | Set an appropriate typical diameter if scale looks wrong. | Check whether the size matches the cells. |
| Cell-probability threshold | `0.0` | Lower it to allow dimmer cells; expect more false positives. | Raise it for stricter detections. |
| Flow threshold | `0.4` | Raise it to retain more irregular masks. | Lower it to reject more poorly shaped masks. |

**Object Filtering** applies after prediction. A filter value of `0` disables
that filter. Available bounds cover minimum and maximum object area in pixels,
maximum major/minor axis ratio, and raw signal. Fluorescence uses a **minimum
mean raw intensity**; brightfield uses a **maximum mean raw brightness** so
darker stained cells can be kept. Per-channel filter overrides are available
for fluorescence. Be careful with strict filters: an object removed here will
not appear for later validation or analysis.

The **Parameter Wizard** segments the first sample without object filters,
then lets you pick example cells to derive size, shape, and signal thresholds.
For brightfield it shows the original stain brightness. You can leave a bound
unset to keep that filter off. Use a representative image and inspect its
results before processing a full batch.

**Preprocessing mode** changes model input. For brightfield, inversion is
handled in the backend as needed; the visible image is not inverted. **Tiling**
controls tile height, width, overlap halo, and QC overlay downsampling. Smaller
tiles may help memory-limited machines but can change boundary behavior; test
an image before a large run.

### Run and compare

Click **Run Segmentation**. **Test run** processes only the first file and
saves per-tile debug PNGs. If an interrupted run is detected, **Continue a
previous run** appears and skips completed files. New runs can be kept in
separate, named or numbered segmentation folders. In Steps 4 and 5, select
the exact run you want to review or analyze, especially when comparing
models. Each run's settings are shown beside its selector.

## Step 4: Correct and validate predictions

Step 4 is optional for basic analysis but required for reviewed training data.
Select the **segmentation run** and **channel to validate**. Set channel display
names and colors if helpful; the validator can display channels that were not
segmented. Work on one selected channel at a time.

### Correct masks in Napari

Open **Review Segmented Images** before making validation decisions. This
window shows raw images, predicted objects, and drawing layers.

| Task | Action |
|---|---|
| Add a missed cell | Click the large **ADD MISSED CELLS** button on the right and paint it. Each disconnected painted area becomes a separate object. For touching cells, use different label colors. |
| Remove a false prediction | Click the large red **REMOVE PREDICTED OBJECTS** button on the right, then paint any part of the prediction. This switches to a separate removal layer; the whole predicted object is removed when saved. |
| Undo a stroke | Press **Ctrl+Z** before saving. |
| Paint or navigate | Press **P** for paint mode; right-drag to pan; use `[` and `]` to change brush size. |
| Save changes | Click **Save & Next**. **Skip** leaves the image unchanged; **Previous** returns; **Finish** ends the session. |

On a Mac using an iPad as a Sidecar display, Apple Pencil draws or erases, one
finger pans, and Pencil double-tap switches tools. This behavior is enabled
only for Sidecar; normal Napari mouse controls remain available. **Test mode**
in Step 4 processes one image and saves to a `DEBUG_` file so you can inspect
the result before changing production masks. Napari's ordinary eraser on the
**Add ROIs here** layer reshapes an unsaved addition; use the red removal
button to reject an original model prediction. On the first normal save, the
editor keeps an untouched copy of the prediction beside the active mask.
The active mask is updated so existing analysis and older app versions still
read the corrected cells. Added objects are listed in
`_manual_additions_chN.tsv`; removed model predictions are listed with reason
`ARTIFACT` in `_manual_deletions_chN.tsv`. Those records let retraining tell
missed cells from incorrect predictions. A project edited before this feature
cannot recover an original prediction that was already overwritten; its next
edit preserves the mask as it exists then. Editing a mask after validation
may flag only affected objects for revalidation; check the validation status
before analysis or retraining.

### Validate objects

Click **Open Validation Window**. For each object, select a decision and save.
The default keys are **Left = YES**, **Right = NO**, **Up = MAYBE**,
**Shift = ARTIFACT**, and **Down = Save & Next**. You can customize decision
buttons and keys, up to six options; Shift and Down stay reserved. Use notes
and **Go Back** if you need to revise an earlier object. Decisions save as TSV
files alongside the prediction files.

For training, **YES** and **NO** are both cell labels and are targets by
default. **ARTIFACT** alone marks an incorrect prediction; **MAYBE** must be
resolved. You can choose one or several cell labels in Step 7 for a specialized
model. If the goal is high recall, add genuinely
missed cells in the mask editor rather than trying to label them in a list of
existing predictions: a missing prediction has no object to validate.
For an older project where **NO** was used to mean an incorrect prediction,
review those decisions before retraining. Mark incorrect objects ARTIFACT, or
select only the cell labels you want as training targets. Existing analysis
and segmentation outputs are not changed by retraining.

## Step 5: Analyze results

Select the segmentation run to analyze. The selection is shared with Step 4,
so check it again after switching between model runs. The **ROI mask folder**
is optional: leave it blank for whole-image analysis. For directly read
images, use matching full-size `.tif` masks or Fiji `.roi` files. For converted
slide scans, select the masks from the same converter run as the image crops.

The **Validation Status** panel shows review progress. When validation is
complete across the selected channel, analysis excludes objects marked
**ARTIFACT**. It does not automatically remove objects marked **NO**; use
**REMOVE PREDICTED OBJECTS** in Step 4 if a false prediction must disappear from
the measured mask. **Apply validation filter on partially-validated files**
is an explicit choice when only some objects have been reviewed; inspect
results carefully if using partial decisions.

For colocalization, turn on **Run colocalization analysis between two
channels**, choose the **cell of interest** and **protein marker** channels,
and set the IoU threshold from `0.0` to `1.0`. This requires two segmented
channels. The ROI inclusion rule is **centroid** or **any_pixel**. If you need
only object measurements from one channel, leave colocalization off.

Analysis writes per-image object CSV tables and
`combined_analysis_summary.xlsx`. The workbook includes counts, object size,
intensity, and configured colocalization summaries. An image missing a selected
channel is skipped and recorded rather than ending the whole run; its summary
row identifies the missing channel. The completion line reports how many
images were analyzed, skipped for missing channels, and failed. See
`analysis_errors.log` in the analysis output folder for file-specific details.

If results already exist for that run, CellMatchR asks **Replace analysis
results?**. **No** cancels and preserves them. **Yes** backs up that run's
previous generated results under `.analysis_backups/` in the project, then
runs the analysis. Other segmentation runs are not replaced. When reopening
a project after an interrupted replacement, the app checks unfinished backups
and attempts recovery.

## Step 6: Run the pipeline

Step 6 summarizes the current configuration and offers buttons for each
stage. **Run All** segments if needed and then analyzes the selected run.
Validation is interactive and opens separately from its own button; **Run
All** does not silently approve objects. If an analysis output exists, the
same replace-or-cancel question appears. Check the selected backend, channels,
and input folder in the summary before starting a long run.

## Step 7: Retrain a model

Use completed human validation from Step 4. Choose the reviewed segmentation
run, one channel, the model family to train, a starting model, and a **new model
name**. Corrected masks provide the examples: confirmed cell labels,
including YES and NO, become targets by default; ARTIFACT is background.
Resolve MAYBE objects first. The masks can
have been created with **any** model family. For example, reviewed Micro-SAM
masks can teach a Cellpose model.

**Initialize weights from** selects the starting model *within* the model
family being trained. It is not a save location and does not determine which
reviewed masks are used. Choose a built-in model or a saved fine-tuned model in
the dropdown. **Show advanced settings** contains an option to browse for an
external checkpoint. The named result is saved separately in the shared
library at `~/CellMatchR/Models`; it does not replace the built-in or earlier
model. You can select it for a later training round or in Step 3 for any
project. **Select the new model for this project's next segmentation runs**
also makes it the current choice after training.

![Step 7 starting model and new model name controls, circled in red](images/step7-model-setup.png)

*The red circles mark the starting weights and the name of the separate new model.*

### What each model and data control does

| Control | What to choose and what happens |
|---|---|
| **Reviewed masks from** | Select the current project's segmentation run whose masks you corrected in Step 4. This supplies training labels; it does not choose the model whose weights will be changed. |
| **Channel to train** | Choose one channel found in that run. Step 7 trains one channel at a time; train channels 0, 1, and 2 separately if you need all three. The menu updates when the reviewed run changes. |
| **Cell labels to train** | Leave every box unchecked (the default) to use **all confirmed cells**. Check one or more labels to make only those cells training targets. Other valid cell types become background in this specialized model, so it may stop detecting them. ARTIFACT always remains background and cannot be selected as a cell label. |
| **Model family to train** | Pick **Micro-SAM**, **Cellpose**, or **StarDist**. This controls the training software, compatible starting weights, and later Step 3 model choice. Reviewed masks can come from a different family. |
| **Initialize weights from** | Choose a built-in model or a previously saved fine-tuned model **of the selected family**. Continuing a fine-tuned model creates a *new* model; the prior checkpoint stays available. Use **Other starting model** under advanced settings only for a compatible checkpoint that is not listed. |
| **Save new model as** | Give the result a recognizable name. The suggested name includes family, channel, and date; you can replace it. The field accepts 1–60 characters without slashes, backslashes, tabs, or line breaks. Each run creates a separate folder, even if you reuse a display name. |
| **Open Model Library** | Open `~/CellMatchR/Models` in Finder or your file manager. You do not need to choose a save path: this shared library is used by every project on this computer. |
| **Select new model for this project's next segmentation runs** | After successful training, make the result the current Step 3 choice in this project. The model also appears in the model dropdown for other projects. The original model and earlier segmentation outputs remain available. |

To compare models fairly, segment the same held-out image with each and
compare both with its human-reviewed mask.

### Build the training image library

The current project's selected run and channel are included automatically.
To include another project, click **Add another project**, browse to its
`cellmatchr_project.json`, and choose its reviewed segmentation run and
channel. Repeat for as many projects as you need. The project path is taken
from the selected file's folder, which also works for projects moved from
another computer. Other projects are read as sources; training results are
written to the **current** project. Click **Refresh image library** after
changing the sources.

The library shows a color overlay of saved image channels without mask
outlines. Each completed image's card lists approximate original proposals,
reviewed ARTIFACT/erased predictions, training target cells, and the count
for each reviewed cell label (for example, YES and NO). The label counts show
all confirmed cell types in that image, even when you select only some as
training targets. Uncheck an image to exclude it
from this run. An image appears only after review is complete and every
current object has a resolved decision. The log lists skipped images and why.
The listed counts help spot an unusual sample before starting; inspect the
mask in Step 4 when a count is surprising.

**Refresh image library** after changing the selected run or channel, adding
a project, or changing review data. The gallery may show fewer images than a
segmentation run contains: only usable, fully reviewed images are offered.
With selected cell labels, an image containing none of those labels is
automatically left out. Unchecking an image excludes it from this training
run; it does not delete the image or its validation. If some images still
need verification, **Start Retraining** asks whether to train only on the
finished images. Choose **No** to finish review first; choosing **Yes** does
not change unfinished images. When choosing a whole-image holdout, click the
**Choose held-out image from library** button and then that image's preview.
Keep at least one other included image for training.

![Step 7 image library and Add another project control, circled in red](images/step7-image-library.png)

*Use **Add another project** to bring in more reviewed data. Uncheck an image
to leave it out of this run. The Mac screenshot uses real VSI slide scanner
crops with illustrative review labels; the displayed counts are examples, not
measured model accuracy.*

### Choose a validation approach

You must choose one of these before **Start Retraining** becomes available:

| Choice | What stays out of training | When to use it |
|---|---|---|
| **Reserve one whole image** (recommended) | Click **Choose held-out image from library**, then click that image's preview. | At least two reviewed images are available. This gives an independent before/after check. |
| **Reserve a small patch** | A randomly chosen patch around marked ARTIFACT objects; its pixels and overlapping cell labels are excluded from training. | Only a small dataset is available, but it has reviewed artifacts still visible in the mask. The default target is 8 incorrect predictions; change the approximate target beside the selection button if needed. The report records the random seed, exact patch location, and actual count, which can be lower when few artifacts are available. If every false prediction was erased, choose a whole-image holdout or the in-sample option instead. |
| **Use all images for training** | Nothing. The report evaluates an image used for training and clearly marks the result **in-sample**. | Very small datasets where an independent holdout would leave too little training data. Do not interpret this score as performance on new images. |

![Step 7 validation approach choices and held-out image button, circled in red](images/step7-validation-choice.png)

*The red circles mark the three choices and the button for picking a whole
image to hold out. Training stays unavailable until a choice is complete.
The screenshot shows a manually entered patch target of 3; the current
default is 8.*

Choose a holdout that resembles the images you intend to process later.
An independent holdout stays untouched by training; reserve more projects or
images for a stronger future comparison when possible.

The **Target incorrect predictions per patch (approx.)** field applies only
to the small-patch option. Its default is **8**, and it accepts **1–50**.
The app chooses a fresh random location around reviewed incorrect objects on
each run. Eight is an approximate target, not a required count: nearby
objects and image boundaries affect the actual patch. The report records the
random seed, patch coordinates, and actual incorrect-object count so you can
inspect exactly what was reserved. If the patch would leave no usable
training cells, choose a whole-image holdout or include more reviewed data.

**Start Retraining** becomes available when the selected images and
validation choice are valid. Progress appears in **Retraining Log**. A
training run can take substantially longer than the short trial preset;
leave the app open until it reports completion or an error.

After training, Step 7 reveals the training curve and reviewed reference,
before, and after segmentation images below the log. The captions compare
recall, missed cells, and false positives against the reviewed mask at an
object IoU threshold of 0.5. Use **Open report folder** to inspect the full
saved report in the current project's `Training_Reports` folder. The model
remains in the shared model library even if report generation fails; the log
and report folder explain that failure. Compare missed cells first when recall
is your priority, then inspect false positives and actual mask shapes.
For a specialized model, the report uses only selected labels as its reference;
predictions on other real cell types can therefore appear as off-target
detections in its score. Those cells are still present in the source review
and are never relabeled ARTIFACT.

### Training presets

| Preset | Epochs and learning rate | Training sample focus | Good first check |
|---|---|---|---|
| **High recall** | 20; `1e-5` | Repeat samples with human-added missed cells. | Fewer missed cells on the holdout. |
| **Default training** | 20; `1e-5` | Use each reviewed sample once. | Compare with the starting model. |
| **High precision** | 20; `1e-5` | Repeat samples with ARTIFACT or erased predictions. | Check that recall has not fallen. |
| **Gentle follow-up** | 8; `3e-6` | No extra repetition. | Small, stable change from a saved model. |
| **Recall follow-up** | 8; `3e-6` | Repeat missed-cell samples. | Better held-out recall. |
| **Precision follow-up** | 8; `3e-6` | Repeat samples with ARTIFACT or erased predictions. | Fewer false positives without losing true cells. |
| **Short trial** | 5; `1e-5` | No extra repetition. | Check data and training setup before a longer run. |

The learning rates shown apply to Micro-SAM and Cellpose. For StarDist, the
standard presets leave the pretrained model's learning rate in place, while
the three follow-up presets set `1e-4`. Selecting a preset fills **Training
epochs**, **Learning rate**, and **Training sample focus**; changing presets
or model family can overwrite manual values in those fields. It does not
change the separate validation choice or the segmentation detection
thresholds. Choose the preset first, then make any advanced adjustments.

Presets are starting points, not performance guarantees. Compare predictions
against held-out human annotations before adopting a new model. If missed
true cells are especially costly, begin with **High recall** or **Recall
follow-up**, use permissive prediction settings, and avoid filtering real
cells away. You can then remove false positives through validation and
appropriate analysis filters.

### Advanced settings and devices

Open **Show advanced settings** only when you need to adjust the recipe.
These values affect training, not the masks you have already reviewed.

| Setting | Meaning and practical effect |
|---|---|
| **Training epochs** | Number of passes through the training routine, from **1–500**. More epochs take longer and may help or overfit. The preset sets this first. A **Short trial** is a way to check the setup before committing to a longer run. |
| **Preprocessing** | Image transformation used to prepare training inputs and recorded with the new model for later segmentation. Keep it consistent with the images and segmentation runs you intend to use. The initial value follows the current project's Step 3 preprocessing choice. See the mode table below. |
| **Learning rate** | Size of each weight update. Larger values can change a model faster but may destabilize training; smaller values usually need more time. Enter a value greater than 0 and at most 1, such as `1e-5`. The preset fills this for Cellpose and Micro-SAM. A blank StarDist field uses that model's own default. |
| **Batch size** | Number of image patches presented together for a training update. Leave blank for the backend default, or enter **1–64**. Larger batches may use more GPU or unified memory. Cellpose and Micro-SAM default to **1**; StarDist uses its pretrained configuration. |
| **Patch size (px)** | Width and height of square training crops. Leave blank for the model default; otherwise enter **64–2048** in multiples of **16**. Smaller patches may fit in less memory but give less context. Cellpose uses a fixed **256-pixel** patch, so this control is hidden for Cellpose. Micro-SAM defaults to **256**; StarDist uses its pretrained configuration. This is a *training crop* and is different from the small patch reserved for validation. |
| **Training sample focus** | `none` uses every training sample once. `missed_cells` repeats samples containing manually added target cells; `rejected_cells` repeats samples containing ARTIFACT or erased model predictions. Repetition applies only to training examples, never to held-out validation data. It can shift emphasis but does not guarantee a higher recall or precision. |
| **Other starting model** / **Browse** | Select a compatible checkpoint outside the dropdown: a **file** for Cellpose or Micro-SAM, a **model folder** for StarDist. This becomes the starting weight source, not the output location. The dropdown is easier for built-in and previously saved models. |
| **StarDist distance loss** | **Model default** keeps the selected model's distance loss. `mae` and `mse` are alternate losses for predicted object distances. This control appears only for StarDist; Cellpose and Micro-SAM use their built-in training losses. |
| **Micro-SAM: keep image encoder frozen** | On by default. It trains the remaining trainable parts while keeping the image encoder weights fixed. Turning it off also updates the encoder, which can use more memory and may need more varied examples. This control appears only for Micro-SAM. |
| **Model Manager** | Opens the saved-model list and its training history; see [Manage saved models](#manage-saved-models). |

| Preprocessing mode | What it does | Typical reason to choose it |
|---|---|---|
| `local_bg_subtract` | Removes a blurred local background, clips negative values, then rescales intensity. | Fluorescence with uneven background; stronger background removal. |
| `local_bg_subtract_mild` | Uses a broader background estimate before rescaling. | A gentler local background correction. |
| `raw_percentile` | Rescales the original intensity range without local background subtraction. | Preserve image structure when background removal harms it. |
| `tile_clahe_mild` | Rescales intensity, then applies gentle local contrast enhancement. | Low contrast images needing modest local enhancement. |
| `tile_clahe` | Uses stronger local contrast enhancement. | Test only when the mild option still leaves cells hard to see; it can also amplify texture. |
| `brightfield_invert` | Rescales and internally inverts grayscale intensity for brightfield stains. | Dark stained cells on a pale slide-scanner background. The displayed image remains in its original colors; inversion is for model input. |

The same preprocessing mode is stored with the trained model. For an
apples-to-apples comparison, use the same reviewed examples and validation
image when changing one advanced setting. Training presets do not set an
inference sensitivity threshold; inspect held-out recall and false positives
before changing downstream object filters.

Training is memory intensive. **At least 16 GB of GPU memory or unified Mac
memory is recommended** for a practical retraining workflow; the exact need
depends on model, image size, patch size, and batch size. Smaller machines may
need smaller patches and batches, or a CPU run that takes much longer. Test a
short run first rather than assuming a model will fit.

| Backend | Native Windows training |
|---|---|
| Cellpose-SAM | NVIDIA GPU if PyTorch detects CUDA; otherwise CPU. |
| Micro-SAM | NVIDIA GPU if PyTorch detects CUDA; otherwise CPU. |
| StarDist | CPU with optional TensorFlow 2.16.2 setup below. |

The Windows workflow was smoke-tested from the CLI and GUI: Cellpose-SAM and
Micro-SAM used an RTX 3090. For StarDist GPU training, use a suitable Linux
or WSL2 TensorFlow environment. Native Windows CUDA support ended with
TensorFlow 2.10; see the [TensorFlow installation guide](https://www.tensorflow.org/install/pip).

For optional native Windows StarDist, run this in PowerShell from the
CellMatchR checkout after `setup-windows.cmd`:

```powershell
$runtime = "$env:SystemDrive\CellMatchR"
$mamba = "$runtime\Library\bin\micromamba.exe"
$prefix = "$runtime\CellMatchR-win-env"
$env:MAMBA_ROOT_PREFIX = $runtime
$stardist = "$runtime\stardist"
& $mamba --no-rc run --prefix $prefix python -m pip install --no-deps --target $stardist -r .\scripts\windows-stardist-requirements.txt
$env:CELLMATCHR_WINDOWS_STARDIST = $stardist
.\run-windows.cmd
```

Choose **StarDist** in that session. Start a fresh PowerShell session for the
default Cellpose/Micro-SAM runtime. For StarDist CLI commands, set
`$env:PYTHONPATH = $stardist`.

### Manage saved models

Click **Open Model Library** to see the files. Under **Show advanced
settings → Model Manager**, inspect each model's name, backend, channel,
creation date, training rounds, and recorded epochs. Older model records may
show incomplete history. Deletion requires two confirmations: answer the
warning and then type the exact model name. Deleting a model does not remove
models trained from it, but projects that use the deleted checkpoint will
need another model selected.

## Understand files and outputs

| Project item | Purpose |
|---|---|
| `cellmatchr_project.json` | Project settings and paths; choose this file to reopen the project. |
| `VSI_Converted_TIFF/` (and numbered variants) | Full-resolution slide regions and associated ROI masks. |
| `Segmentation/` (and run variants) | Predicted masks, channel images, QC material, and run information. |
| `_prediction_baseline_chN_*.tif` inside an edited sample | Untouched prediction copied before the first edit made by this version of the editor. It is kept outside the active mask search pattern. |
| `_manual_additions_chN.tsv`, `_manual_deletions_chN.tsv` inside an edited sample | IDs of missed cells you added and model predictions you removed as ARTIFACT. `N` is the channel number. |
| `Validation/` | Validation-stage project output; per-channel decisions are saved as TSV beside prediction files. |
| `Analysis/` (and run variants) | Object tables, combined workbook, and run report. |
| `.analysis_backups/` | Earlier generated analysis results preserved when you approve replacement. |
| `~/CellMatchR/Models/` | Shared fine-tuned model library, available across projects. |

The project may contain multiple conversion, segmentation, and analysis runs.
Use the selectors in Steps 3–5 to keep source crops, masks, human decisions,
and analysis results paired. Do not delete a run until you have confirmed
that no later result depends on it. The exact files inside a run vary by
backend and chosen options.

## Frequently asked questions

### Installation and updates

**Why clone instead of downloading the ZIP?** A clone can be updated with
`git pull`; a ZIP has no Git history and must be downloaded again for each
version. Both can launch the app after platform setup.

**How do I know whether Conda is already installed?** On macOS or Linux,
open Terminal and run `conda --version`. A version number means the command
is available. On Windows, `setup-windows.cmd` installs CellMatchR's own
Conda-compatible runtime, so a separate Conda installation is unnecessary.

**Conda is installed, but Terminal says `command not found`.** Close and
reopen Terminal. If the message remains, Conda may not be initialized for
your shell; use your existing distribution's shell initialization instructions.
Avoid installing a second Conda distribution until you have checked this.

**`environment.yml` was not found. Where should I run the command?** In the
CellMatchR application folder, not in your image or project folder. Run
`pwd` to see your location and `ls environment.yml` to confirm the file is
there. On macOS you can type `cd `, drag the folder from Finder into Terminal,
and press Return.

**Will `git pull` update my existing projects and models?** It updates tracked
application files in the checkout. Projects stored elsewhere and the shared
model library are not part of that checkout. Check `git status` before pulling
if you edited application files locally.

**I moved the CellMatchR folder and the launcher stopped working.** Keep the
macOS `.app` inside the checkout. On Windows recreate the shortcut with
`.\setup-windows.cmd -ShortcutOnly`; on Linux rerun
`bash scripts/install_linux_launcher.sh`.

**Can I open an older project?** Choose its `cellmatchr_project.json` in
Step 1. Verify the image input, selected segmentation run, ROI mask folder,
and analysis settings before starting a new run, particularly if folders
have moved or the project was created on another computer.

### Images and regions

**Which LIF mode should I use?** Choose **Confocal** for a confocal Leica
acquisition and **Slide scanner** for a stitched slide. The app checks the
file contents and flags or skips a LIF in the wrong mode.

**Why is a VSI missing its image data?** Check that its matching `_name_`
companion folder is beside the `.vsi` and has not been renamed or separated.

**Do I need to crop a large scan first?** Use the Slide Scanner Converter to
draw named regions. It exports full-resolution TIFFs and matching ROI masks.
An existing TIFF/BigTIFF slide can go through the same region workflow.

**Why are my Nissl cells treated as a single channel?** Brightfield RGB is
combined into one cell signal, channel `0`, for prediction. The original
color image remains visible for review.

**Why did a file not appear in segmentation?** Check the Step 2 file preview,
file format, LIF classification, input folder, and **Skip subfolders** setting.
The normal scan is the root and one level of subfolders, not an unlimited
recursive search.

### Predictions, correction, and validation

**How do I compare models on the same data?** Run each backend or model on the
same input, keep its separate segmentation output, and select each run in
Steps 4–5. Compare masks on the same human-annotated images and track missed
cells as well as false positives.

**A real cell is missing completely. Can the intensity wizard recover it?**
No. Object filters act on masks that the model already proposed. Lowering a
strict prediction threshold may reveal missed cells; add persistent misses
in **Review Segmented Images** and use them for retraining.

**Why did a correct cell disappear after filtering?** Check minimum size,
maximum size, shape ratio, and intensity/brightness bounds. Set a suspect
filter to `0` to disable it, then rerun a test image. For Cellpose, also check
cell-probability and flow thresholds.

**Can I delete only part of a bad prediction?** **REMOVE PREDICTED OBJECTS** marks
the entire touched predicted object for removal on save. Use **Ctrl+Z** to
undo a stroke before saving. This prevents a partial fragment from silently
remaining as a cell.

**Why does validation say some objects need review again?** A corrected mask
can change object identity after validation. Refresh validation status and
review the affected objects rather than assuming the earlier decision still
matches the edited mask.

**Do I have to validate before analysis?** Step 4 is optional for basic
analysis. Validation is needed to exclude ARTIFACT objects through the
validation filter or to create reviewed training targets. Check the Step 5
status panel if review is partial.

**Does marking a prediction NO remove it from analysis or training?** No.
YES and NO are both cell labels and are training targets by default. The
analysis validation filter removes ARTIFACT objects. To remove a false
prediction from the masks used in later analysis, use **REMOVE PREDICTED OBJECTS**
in the Step 4 editor and save the mask. To train a model for only one type of
cell, select that label in Step 7; other cell types then become background in
that specialized training run.

### Analysis and training

**Why is the completion line shown when a file failed or lacked a channel?**
The batch can finish while individual files are skipped or fail. The line
reports analyzed, skipped, and error counts; inspect `analysis_errors.log`
and the workbook status rows before using the result.

**What happens when I rerun analysis?** You choose **Yes** to back up and
replace generated results for that selected run, or **No** to cancel. Look
inside `.analysis_backups/` for earlier results.

**Why is colocalization unavailable?** It needs two segmented channels.
Select both channels in Step 3 and choose their masks in Step 5.

**Where is my fine-tuned model?** The default shared folder is
`~/CellMatchR/Models`. Step 7 has **Open Model Library** and a Model Manager.
Saved models appear in the starting-model and future-segmentation dropdowns.

**Can I continue training a fine-tuned model?** Yes. Select that saved model
under **Initialize weights from**, give the new result its own name, and
review held-out predictions after training. The previous checkpoint remains.

**Can I combine reviewed images from several projects?** Yes. Use **Add another
project** in Step 7 for each additional `cellmatchr_project.json`, then choose
its run and channel. The image library shows which project each sample came
from. Uncheck any sample that should not enter this training run.

**Why is an image absent from the training library?** Finish its Step 4 review,
resolve MAYBE decisions, and check that its raw channel and mask exist in the
selected segmentation run. Click **Refresh image library** and read the
skipped-sample reasons in the log.

**Can I train while some images are still awaiting review?** Yes. When you
start retraining, CellMatchR asks whether to use only images whose review is
complete. **Yes** excludes unfinished images from that training run; **No**
cancels so you can finish review. Unfinished decisions are not treated as
negative examples.

**What if I leave all cell labels unchecked?** That is the default: every
confirmed cell label is a training target. To make a model for specific cell
types, check one or more labels. Samples with none of the chosen labels are
skipped, and unchosen cell types become background only in the training copy
of the masks. The original masks and review decisions stay intact.

**Can I train all selected channels together?** Step 7 trains one channel at
a time. Select another channel and create another named model if needed.

**Which preset should I choose?** Start with **Default training** for a
baseline. If missed true cells matter most, compare **High recall** with the
baseline. Use **High precision** only after checking that it does not lose
real cells. A **Short trial** checks that data and dependencies work.

**How do I know whether a new model is better?** Compare it with the original
model on held-out, human-annotated images. Count true cells found, missed
true cells, and false detections; also inspect mask outlines. Training-set
examples alone cannot establish generalization.

## Troubleshooting by symptom

| Symptom | Check and next action |
|---|---|
| App does not open on macOS | Open the correct `.app` for your macOS version from inside the checkout. Launch `scripts/launch_cellmatchr.command` for a terminal log, or activate `cellmatchr` and run `python scripts/run_cellmatchr.py`. |
| macOS ZIP app reports permission denied | In the checkout, run the `chmod` command below once, then reopen the app. |
| Windows shortcut is missing after moving the checkout | Run `.\setup-windows.cmd -ShortcutOnly` from PowerShell in the checkout. |
| Windows GPU setup is unsuitable or unavailable | Run `.\setup-windows.cmd -Device CPU` for a CPU runtime; expect slower training and prediction. |
| Linux menu entry opens an old checkout | Rerun `bash scripts/install_linux_launcher.sh` from the current checkout; use `bash run-linux.sh` to inspect startup messages. |
| Channel preview is empty | Confirm the Step 2 input folder, inspect the file list, and check whether a LIF belongs to the other mode. |
| A converted slide run is not the one being segmented | In Step 3 expand **Use a different converted output** and select the intended folder. |
| Masks look sparse or contain many false detections | Test a representative image. Adjust model/Cellpose sensitivity, then inspect object filtering and preprocessing. Compare against human annotations. |
| Run stopped partway through a batch | Reopen Step 3 and use **Continue a previous run** if shown. Inspect the run log for the sample that stopped it. |
| Analysis has blank or missing-channel rows | Verify both selected channels exist for each sample; see `analysis_errors.log`. |
| Training cannot start | Finish validation, resolve MAYBE decisions, choose one of the three validation approaches, and check that the starting model and backend dependencies are installed. A whole-image holdout needs two included images; the patch or in-sample choice can use one. |
| Training runs out of memory | Reduce batch or patch size where supported, select a smaller model, or move training to a machine with more GPU/unified memory. |

For the macOS ZIP permission issue, run:

```bash
chmod +x 'CellMatchR (macOS 26 and earlier).app/Contents/MacOS/CellMatchR' \
  'CellMatchR (macOS 27+).app/Contents/MacOS/applet' scripts/launch_cellmatchr.command
```

If a Windows runtime needs a different short path, set
`CELLMATCHR_WINDOWS_RUNTIME` in both the setup and launch sessions. On Linux,
set `CELLMATCHR_LINUX_ENV` in the desktop or terminal environment to use a
custom environment prefix.

When reporting a reproducible issue, include your platform, file format,
chosen backend/model, the relevant step's log, and a small example image if
it can be shared. Remove identifying or sensitive data before sharing a
project configuration or raw image.

## Command-line reference

The app is the simplest way to choose runs and review images. The same
segmentation, analysis, and retraining modules can also be run in a terminal
or by a coding assistant. Open a terminal as described in
[Where to type the commands](#where-to-type-the-commands), enter the cloned
`CellMatchR-app` folder, and confirm that `environment.yml` is there. Paths
below are examples; replace them with the full paths on your computer.

**macOS:** Run `conda activate cellmatchr` after entering the app folder.
**Linux:** Run `conda activate "$PWD/../CellMatchR-linux-env"` if you used the
default Linux install path. For the same library and model-cache setup as
the Linux app launcher, then run:

```bash
export QT_API=pyqt5
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MICROSAM_CACHEDIR="${MICROSAM_CACHEDIR:-$CONDA_PREFIX/model-cache/micro-sam}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-$CONDA_PREFIX/model-cache/cellpose}"
```

On macOS or Linux, verify the Python environment and inspect the options:

```bash
pwd
ls environment.yml
python -c "import sys, cellmatchr; print(sys.executable); print(cellmatchr.__file__)"
python -m cellmatchr.segmentation --help
python -m cellmatchr.analysis --help
python -m cellmatchr.retraining --help
```

The printed package path should be inside this checkout. If it points to a
different installation, enter the intended app folder and select its
environment before running a job. `--help` is safe: it shows options without
processing images.

**Windows PowerShell:** Enter the cloned app folder first. Windows setup puts
the environment in `C:\CellMatchR` by default; use the installed runtime for
each CLI command:

```powershell
$runtime = "$env:SystemDrive\CellMatchR"
$mamba = "$runtime\Library\bin\micromamba.exe"
$prefix = "$runtime\CellMatchR-win-env"
$env:MAMBA_ROOT_PREFIX = $runtime
& $mamba --no-rc run --prefix $prefix python -c "import sys, cellmatchr; print(sys.executable); print(cellmatchr.__file__)"
& $mamba --no-rc run --prefix $prefix python -m cellmatchr.retraining --help
```

For segmentation and analysis, `--config` is the only direct CLI option. It
points to a JSON file with the same settings the app saves in a project's
`cellmatchr_project.json`. The default `~/.cellmatchr/config.json` reflects
the **last** app session, so do not assume it points to the project you intend
to process. Make a separate copy of your project config for CLI work and
inspect these fields before running:

| Module | Settings to check in the copied config |
|---|---|
| Segmentation | `input_type`, `input_root_folder`, `output_root_folder` (the chosen segmentation run folder), `seg_channels` (such as `"0,1"`), `seg_backend`, `seg_model_type` for Micro-SAM or the appropriate `seg_custom_model_path`, `seg_cellpose_model_path`, or `seg_stardist_model_path`, `seg_preprocess_mode`, `seg_resume`, `seg_dry_run`, and `seg_test_run`. |
| Analysis | `ana_microsam_output_root` (the segmentation run to read), `output_root_folder` (the analysis folder to write), `ana_red_channel`, `ana_pv_channel`, and `apply_partial_validation`. |

On macOS/Linux, one way to prepare a config is:

```bash
cp "/absolute/path/to/MyProject/cellmatchr_project.json" "/absolute/path/to/MyProject/cli-config.json"
```

Open `cli-config.json` in a text editor, check the paths and settings above,
and set `"seg_dry_run": true` and `"seg_test_run": false` for a first
segmentation check. Then run:

```bash
python -m cellmatchr.segmentation --config "/absolute/path/to/MyProject/cli-config.json"
```

The dry run lists the planned samples and channels without making masks.
When the plan is correct, set `"seg_dry_run": false` in the copied config
and run the command again. Use a new output folder for a new segmentation
run; `seg_resume` is for continuing the same run. To analyze that run, set
the analysis paths and channels in the copied config, then run:

```bash
python -m cellmatchr.analysis --config "/absolute/path/to/MyProject/cli-config.json"
```

Analysis writes the combined workbook and `analysis_errors.log` to its
output folder. A CLI run does not show the app's **Replace analysis results?**
dialog, so choose an empty analysis folder or preserve the existing results
before rerunning. On Windows, make the copy and run the same module through
the installed runtime, for example:

```powershell
Copy-Item "C:\path\to\MyProject\cellmatchr_project.json" "C:\path\to\MyProject\cli-config.json"
# Check and edit cli-config.json before running this command.
& $mamba --no-rc run --prefix $prefix python -m cellmatchr.segmentation --config "C:\path\to\MyProject\cli-config.json"
```

Retraining takes arguments directly instead of `--config`. This short trial
trains Micro-SAM from the built-in `vit_b_lm` weights using channel 1 of an
existing `Segmentation` run. It needs at least two fully reviewed images for
the automatic holdout; replace the project path, run name, and channel with
your own:

```bash
python -m cellmatchr.retraining \
  --project "/absolute/path/to/MyProject" \
  --seg-run "Segmentation" \
  --channel 1 \
  --backend microsam \
  --base-model vit_b_lm \
  --model-name "All reviewed cells trial" \
  --epochs 5 \
  --validation-mode automatic
```

With **no `--cell-label` option**, every confirmed cell label, including YES
and NO, is a target. ARTIFACT is background; resolve MAYBE first. For a
specialized model, add `--cell-label YES` or repeat the option, for example
`--cell-label YES --cell-label NO`. Only the named labels are targets; other
valid cell types become background in the temporary training masks. Images
with no selected target label are skipped. The source masks and review files
are not changed. The training report records the selected labels.

If any image in the selected run still needs review, the CLI stops before
training. Add `--allow-completed-only` **only if** you want to train on the
fully reviewed images and skip the unfinished ones; the app asks this as a
Yes/No question. For a single usable image, use `--validation-mode training_image`
for an in-sample check, but that score does not measure performance on new
images. See [Choose a validation approach](#choose-a-validation-approach)
before judging a model. Training saves a **new** model under
`~/CellMatchR/Models` and a report under the project's `Training_Reports`;
the starting model remains available. On Windows, use the same retraining
arguments after the PowerShell command prefix shown above, written on one
line or with PowerShell line continuations.

For a coding assistant, provide the app checkout, project folder, exact
segmentation run, channel, model family, selected cell labels (if any), and
whether completed images only are acceptable. Ask it to inspect `--help` and
the copied config's input/output paths before launching a long job. The GUI
launchers also set model-cache paths; if a CLI result differs from the GUI,
compare the Python environment, model path, preprocessing, and cache paths.
