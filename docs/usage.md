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

In your command window, run `git --version`. If it prints a version number,
Git is ready. If Git is not found, [install Git](https://git-scm.com/downloads)
and reopen the command window. Move to your Desktop using the command for
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
there. Now clone the repository and enter its folder:

```bash
git clone https://github.com/szhang99-bu/CellMatchR.git CellMatchR-app
cd CellMatchR-app
```

`git clone` creates **CellMatchR-app** on your Desktop; the last `cd` enters
it. You can open the folder directly from the Desktop in Finder or File
Explorer. The app folder has a different location from the shared model
library at `~/CellMatchR/Models`, so trained models stay outside the Git
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

1. On your Desktop, open the `CellMatchR-app` folder in File Explorer.
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
4. On your Desktop, open `CellMatchR-app` in Finder and open the app matching
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

Close CellMatchR and any running analysis or training first. In the checkout,
run `git status` to see whether you have local edits, then `git pull` to update
the code. Keep any local edits you want before pulling; Git may ask you to
resolve them. Project data and the shared model library are separate from the
repository when you follow the setup above.

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
| Add a missed cell | Select **Add ROIs here** and paint it. Each disconnected painted area becomes a separate object. For touching cells, use different label colors. |
| Remove a false prediction | Select **Erase whole object** and paint any part of that object. The complete predicted object is removed when saved. |
| Undo a stroke | Press **Ctrl+Z** before saving. |
| Paint or navigate | Press **P** for paint mode; right-drag to pan; use `[` and `]` to change brush size. |
| Save changes | Click **Save & Next**. **Skip** leaves the image unchanged; **Previous** returns; **Finish** ends the session. |

On a Mac using an iPad as a Sidecar display, Apple Pencil draws or erases, one
finger pans, and Pencil double-tap switches tools. This behavior is enabled
only for Sidecar; normal Napari mouse controls remain available. **Test mode**
in Step 4 processes one image and saves to a `DEBUG_` file so you can inspect
the result before changing production masks. Editing a mask after validation
may flag only affected objects for revalidation; check the validation status
before analysis or retraining.

### Validate objects

Click **Open Validation Window**. For each object, select a decision and save.
The default keys are **Left = YES**, **Right = NO**, **Up = MAYBE**,
**Shift = ARTIFACT**, and **Down = Save & Next**. You can customize decision
buttons and keys, up to six options; Shift and Down stay reserved. Use notes
and **Go Back** if you need to revise an earlier object. Decisions save as TSV
files alongside the prediction files.

For training, **YES** is a cell target, **NO** and **ARTIFACT** are background,
and **MAYBE** must be resolved. If the goal is high recall, add genuinely
missed cells in the mask editor rather than trying to label them in a list of
existing predictions: a missing prediction has no object to validate.

## Step 5: Analyze results

Select the segmentation run to analyze. The selection is shared with Step 4,
so check it again after switching between model runs. The **ROI mask folder**
is optional: leave it blank for whole-image analysis. For directly read
images, use matching full-size `.tif` masks or Fiji `.roi` files. For converted
slide scans, select the masks from the same converter run as the image crops.

The **Validation Status** panel shows review progress. When validation is
complete across the selected channel, analysis excludes objects marked
**ARTIFACT**. It does not automatically remove objects marked **NO**; use
**Erase whole object** in Step 4 if a false prediction must disappear from
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
run, one channel, the backend, a starting model, and a **new model name**.
Training uses corrected masks and decisions: YES objects become targets;
NO/ARTIFACT objects are background. Resolve MAYBE objects first. Keep at least
two reviewed samples so a separate validation set is possible.

**Starting model to fine-tune** means the model whose weights training begins
from; it is not a save location. The resulting named model is saved separately
in the shared library at `~/CellMatchR/Models` and does not replace the built-in
or earlier model. A saved model can be selected as the starting model for a
later training round and in Step 3 for future projects. **Select for future
segmentation** sets the new model as the current segmentation choice; models
remain available in the dropdown even when that option is left off.

### Training presets

| Preset | What it changes | Good first check |
|---|---|---|
| **High recall** | Repeats training samples with human-added missed cells; 20 epochs, 25% validation. | Look for fewer missed cells on held-out images. |
| **Default training** | Uses each reviewed sample once; 20 epochs, 20% validation. | Compare with the starting model. |
| **High precision** | Repeats samples with rejected false predictions; 20 epochs, 25% validation. | Verify recall has not fallen. |
| **Gentle follow-up** | Continues a saved model with 8 epochs and a lower learning rate. | Check for a small, stable change. |
| **Recall follow-up** | Gentle continuation emphasizing missed-cell samples. | Check held-out recall. |
| **Precision follow-up** | Gentle continuation emphasizing rejected samples. | Check held-out false positives and recall. |
| **Short trial** | Runs 5 epochs with a 20% validation split. | Verify the data and training setup. |

Presets are starting points, not performance guarantees. Compare predictions
against held-out human annotations before adopting a new model. If missed
true cells are especially costly, begin with **High recall** or **Recall
follow-up**, use permissive prediction settings, and avoid filtering real
cells away. You can then remove false positives through validation and
appropriate analysis filters.

### Advanced settings and devices

**Show advanced settings** exposes learning rate, batch size, patch size
(where supported), validation percentage, sample focus, and an external
starting model file or folder. StarDist offers MAE or MSE distance loss;
Cellpose and Micro-SAM use their built-in losses. Micro-SAM can freeze or
unfreeze its image encoder; the current default keeps it frozen. StarDist
uses its own default learning rate unless a preset or override sets one.
Change one setting at a time and compare on the same held-out images.

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

**Can I delete only part of a bad prediction?** **Erase whole object** marks
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

**Does marking a prediction NO remove it from analysis?** No. NO is used as
background for retraining, but the analysis validation filter removes
ARTIFACT objects. To remove a false prediction from the masks used in later
analysis, use **Erase whole object** in the Step 4 editor and save the mask.

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
as **Starting model to fine-tune**, give the new result its own name, and
review held-out predictions after training. The previous checkpoint remains.

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
| Training cannot start | Finish validation, resolve MAYBE decisions, use at least two reviewed samples, and check that the chosen model file and backend dependencies are installed. |
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

From the checkout in an active CellMatchR environment, inspect the available
options before running a module directly:

```bash
python -m cellmatchr.segmentation --help
python -m cellmatchr.analysis --help
python -m cellmatchr.retraining --help
```

On Windows, use the installed runtime from PowerShell:

```powershell
$runtime = "$env:SystemDrive\CellMatchR"
$mamba = "$runtime\Library\bin\micromamba.exe"
$prefix = "$runtime\CellMatchR-win-env"
$env:MAMBA_ROOT_PREFIX = $runtime
& $mamba --no-rc run --prefix $prefix python -m cellmatchr.retraining --help
```

The GUI launchers set the model-cache paths used by the app. If a direct CLI
run differs from the GUI, compare the active environment and paths first.
