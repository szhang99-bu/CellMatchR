# CellMatchR

<img src="CellMatchr_Logo.png" width="120" alt="CellMatchR logo" />

Segment cells in microscopy images, review predictions, and measure
colocalization. CellMatchR supports Cellpose-SAM, Micro-SAM, micro-SAM2, and StarDist.

**New here?** The [full user guide](docs/usage.md) has detailed setup
instructions, every workflow step, and answers to common questions.

## Install and launch

Already installed CellMatchR? Go to [Upgrade CellMatchR](#upgrade-cellmatchr).

First, open a command window: **Terminal** on macOS (press **⌘ Space**, type
`Terminal`, press Return), **PowerShell** on Windows (search for it in Start),
or your Linux **Terminal** app. Type the commands below in that window, one
line at a time. **First, check whether Git is installed:**

```bash
git --version
```

If you see a version number, continue. If Git is not found,
[install Git](https://git-scm.com/downloads), reopen the command window, and
check again. Next, go to your Desktop using the line for your system.

**macOS or Linux:**

```bash
cd ~/Desktop
```

**Windows PowerShell:**

```powershell
cd ([Environment]::GetFolderPath('Desktop'))
```

If your Linux Desktop is elsewhere, right-click it in your file manager and
choose **Open in Terminal**.

The clone command below creates the app folder on your **Desktop by default**.
To put it somewhere else, replace the Desktop `cd` command above with the
path to an existing folder (for example, `cd ~/Documents` on macOS/Linux or
`cd "D:\Apps"` on Windows). Then clone CellMatchR and enter its folder:

```bash
git clone https://github.com/szhang99-bu/CellMatchR.git CellMatchR-app
cd CellMatchR-app
```

Cloning with Git allows later updates with `git pull`. If you use
**Code → Download ZIP**, extract it before setup and download a fresh ZIP for
each upgrade. [ZIP upgrade instructions](docs/usage.md#update-a-zip-installation)

### Windows

Use 64-bit Windows on an x64 Intel or AMD processor. Setup installs its own
Python environment, so you can start with just Git.

1. Open the `CellMatchR-app` folder on your Desktop (or at the location you
   chose) and double-click **`setup-windows.cmd`**. Wait for **Setup complete**.
2. In the same folder, double-click **Launch CellMatchR (Windows)**.
   You can also double-click **`run-windows.cmd`**.

### macOS

1. In **Terminal**, check for Conda by typing `conda --version`. A version
   number means it is installed. If Terminal says `command not found`, install
   [Miniforge for your Mac](https://github.com/conda-forge/miniforge#macos-pkg-installers),
   choose shell initialization during setup, then close and reopen Terminal.
   Run `conda --version` again.
2. Make sure Terminal is in the **CellMatchR-app folder**: `ls environment.yml`
   should print `environment.yml`. Then create the environment:

   ```bash
   conda env create -f environment.yml
   ```

   If the file is not found, type `cd ` (with a space), drag the app
   folder from Finder into Terminal, press Return, and retry.
3. Open `CellMatchR-app` on your Desktop (or at the location you chose) and
   double-click the app that matches your macOS version.
   - **macOS 27 or later** — `CellMatchR (macOS 27+).app`
   - **macOS 26 or earlier** — `CellMatchR (macOS 26 and earlier).app`

### Linux

1. In Terminal, run `conda --version`. If it is not found, install
   [Miniforge](https://github.com/conda-forge/miniforge#unix-like-platforms-macos-linux--wsl),
   choose shell initialization, reopen Terminal, and check again.
2. In Bash, from the app folder (`ls environment.yml` should find the file),
   create the environment and add CellMatchR to your app menu:

   ```bash
   PIP_CONSTRAINT="$PWD/scripts/linux-pip-constraints.txt" \
     conda env create --prefix "$PWD/../CellMatchR-linux-env" --file environment.yml &&
     bash scripts/install_linux_launcher.sh
   ```

3. Open **CellMatchR (Linux)** from your app menu. For a terminal launch, run
   `bash run-linux.sh`.

## Upgrade CellMatchR

**Every upgrade has two parts: pull the new code, then update the app
environment.** `git pull` does not install new Python packages or update
existing ones. Do both parts even if you have updated CellMatchR before.
[Upgrading a ZIP copy](docs/usage.md#update-a-zip-installation)

Let any running analysis or training finish, then close CellMatchR. Open the
same Terminal or PowerShell app used for installation. Go to your **existing**
CellMatchR-app folder; do not clone a second copy. These commands use the
default Desktop location. If you installed elsewhere or your Desktop is stored
in a different location, use the actual path to your app folder. On macOS,
you can type `cd ` (with a space), drag that folder from Finder into Terminal,
and press Return.

**macOS or Linux Terminal:**

```bash
cd ~/Desktop/CellMatchR-app
git status --short --branch
```

**Windows PowerShell:**

```powershell
cd (Join-Path ([Environment]::GetFolderPath('Desktop')) 'CellMatchR-app')
git status --short --branch
```

For the standard release, the first line should show the `main` branch.
If you see a different branch or files listed below the branch line, check the
[update guide](docs/usage.md#update-an-existing-git-installation) before pulling.
Then run this in the same window:

```bash
git pull --ff-only
```

**Next, refresh the environment using the updated setup files:**

- **Windows:** Double-click `setup-windows.cmd` in the updated app folder and
  wait for **Setup complete**. If you set up a custom runtime or chose CPU-only
  setup, [repeat those settings](docs/usage.md#windows-setup).
- **macOS:** In Terminal, still in the app folder, run
  `conda env update --name cellmatchr --file environment.yml --prune`.
- **Linux:** In Bash, still in the app folder, run:

  ```bash
  PIP_CONSTRAINT="$PWD/scripts/linux-pip-constraints.txt" \
    conda env update --prefix "${CELLMATCHR_LINUX_ENV:-$PWD/../CellMatchR-linux-env}" --file environment.yml --prune
  ```

  If your launcher uses a custom environment path, set `CELLMATCHR_LINUX_ENV`
  to that path before running the update.

Wait for the environment update to finish, then reopen CellMatchR. Keep
project outputs and shared trained models outside the app folder so they stay
available across upgrades. [Troubleshooting](docs/usage.md#troubleshooting-by-symptom)

## Run your first project

1. **Create a project** and choose an output folder.
2. **Select images.** Use **Confocal** to read images directly. Use **Slide scanner**
   to draw and name regions in a VSI, Leica slide-scanner LIF, TIFF, or BigTIFF.
3. **Segment** the channels you want to measure. Choose **micro-SAM2** to try
   the newer models. For Nissl, use **Brightfield stain** and start with
   `hvit_b_cells`. The first run downloads the selected model and sets up
   its separate environment. On an Intel Mac, choose Micro-SAM or Cellpose-SAM.
   [Model guide](docs/usage.md#step-3-segment-cells)
4. In **Review Segmentation**, use **Bulk Review** to correct whole images,
   then **Individual Object Review** for per-cell decisions. Press **H** in
   the bulk editor to toggle predictions, or hold **Z** in the ROI Verifier
   to hide outlines until you release it. [Review guide](docs/usage.md#step-4-correct-and-validate-predictions)
5. Run **Analysis** to create object tables and an Excel workbook. Follow
   progress in the log below the run controls.

For detailed instructions on file formats, model choices, retraining, and
troubleshooting, see the [user guide](docs/usage.md).

## Process several image folders

Set up segmentation and analysis on a representative project. Open **Large
Batch Processing** near the bottom of the sidebar, add image folders, and
select any folder-specific ROI masks. For slide scans, use TIFFs exported in
Step 2. Check your filters on representative images from each group when
staining or imaging settings differ.

Click **Run Batch** to segment and analyze each folder using the current
project’s settings. Each run saves separate results and a log under
`Large_Batch_Processing/` in that project. [Batch processing guide](docs/usage.md#large-batch-processing)

## Retrain a model

1. In **Step 4**, save your mask corrections: erase false predictions and draw
   missed cells. Individual cell verification is optional when training on all
   cells left in the corrected mask. Complete it if you want to train only
   selected cell labels such as YES or NO.
2. Open **Step 7: Retrain Models**. Choose the corrected run and one channel,
   the model family, starting weights, a name for the new model, and a training
   preset. Leave cell labels unselected to use all kept cells, or check multiple
   labels from completed cell reviews for a specialized model. You can add images
   from other projects and exclude images from this run.
3. Choose how to check the result. **Reserve one whole image** is recommended
   when you have at least two corrected images. Click **Start Retraining**, then
   compare the saved before/after report with the corrected masks.

The new model is saved separately in the shared CellMatchR model library and
can be selected for later training or segmentation in any project. See the
[detailed Step 7 guide](docs/usage.md#step-7-retrain-a-model) for every control,
advanced setting, validation choice, and result.
