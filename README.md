# CellMatchR

<img src="CellMatchr_Logo.png" width="120" alt="CellMatchR logo" />

Segment cells in microscopy images, review predictions, and measure
colocalization. CellMatchR supports Cellpose-SAM, Micro-SAM, and StarDist.

**New here?** The [full user guide](docs/usage.md) has detailed setup
instructions, every workflow step, and answers to common questions.

## Install and launch

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

The clone command below creates the app folder on your **Desktop by default**.
To put it somewhere else, replace the Desktop `cd` command above with the
path to an existing folder (for example, `cd ~/Documents` on macOS/Linux or
`cd "D:\Apps"` on Windows). Then clone CellMatchR and enter its folder:

```bash
git clone https://github.com/szhang99-bu/CellMatchR.git CellMatchR-app
cd CellMatchR-app
```

Cloning with Git allows later updates with `git pull`. **Code → Download ZIP**
also works, but it cannot use `git pull`; each upgrade requires another ZIP.
Already have the app folder? Open your command window **in that folder**
instead of cloning again. The [user guide](docs/usage.md#install-launch-and-update)
shows how to find it.

### Windows

1. Open the `CellMatchR-app` folder on your Desktop (or at the location you
   chose) and double-click **`setup-windows.cmd`**. This installs the app's
   own Conda-compatible environment; no separate Conda installation is needed.
2. Double-click **Launch CellMatchR (Windows)**, the shortcut with the logo.
   **`run-windows.cmd`** also launches the app.

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
   reopen Terminal, and check again.
2. From the app folder (`ls environment.yml` should find the file),
   create the environment and add CellMatchR to your app menu:

   ```bash
   PIP_CONSTRAINT="$PWD/scripts/linux-pip-constraints.txt" \
     conda env create --prefix "$PWD/../CellMatchR-linux-env" --file environment.yml
   bash scripts/install_linux_launcher.sh
   ```

3. Open **CellMatchR (Linux)** from your app menu. For a terminal launch, run
   `bash run-linux.sh`.

## Run your first project

1. **Create a project** and choose an output folder.
2. **Select images.** Use **Confocal** to read images directly. Use **Slide scanner**
   to draw and name regions in a VSI, Leica slide-scanner LIF, TIFF, or BigTIFF.
3. **Segment** the channels you want to measure.
4. **Review** the predictions and run **Analysis** to generate object tables
   and a colocalization workbook.
5. **Retrain** in Step 7 if you want to fine-tune a model on reviewed masks.

For detailed, step-by-step instructions, file formats, model choices, updating,
and troubleshooting, see the [user guide](docs/usage.md).

## Retrain a model

1. Finish **Step 4: Validation** for the images you want to use. Mark real
   cells YES and incorrect predictions NO or ARTIFACT; resolve every MAYBE.
2. Open **Step 7: Retrain Models**. Choose the reviewed run and one channel,
   the model family, starting weights, a name for the new model, and a training
   preset. You can add reviewed images from other projects and exclude images
   from this run.
3. Choose how to check the result. **Reserve one whole image** is recommended
   when you have at least two reviewed images. Click **Start Retraining**, then
   compare the saved before/after report with the human-reviewed masks.

The new model is saved separately in the shared CellMatchR model library and
can be selected for later training or segmentation in any project. See the
[detailed Step 7 guide](docs/usage.md#step-7-retrain-a-model) for every control,
advanced setting, validation choice, and result.
