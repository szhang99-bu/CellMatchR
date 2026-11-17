# CellMatchR

<img src="CellMatchr_Logo.png" width="120" alt="CellMatchR logo" />

Segment cells in microscopy images, review predictions, and measure
colocalization. CellMatchR supports Cellpose-SAM, Micro-SAM, and StarDist.

## Install and launch

Clone with Git so you can update later with `git pull`:

```bash
git clone https://github.com/szhang99-bu/CellMatchR.git
cd CellMatchR
```

If you cannot use Git, **Code → Download ZIP** also works, but the ZIP has no
update history: to upgrade, you must download and replace the files again.
Then follow the instructions for your platform below.

### Windows

1. Double-click **`setup-windows.cmd`** and wait for setup to finish.
2. Double-click **Launch CellMatchR (Windows)**, the shortcut with the logo.
   **`run-windows.cmd`** also launches the app.

### macOS

1. Install [Miniforge](https://github.com/conda-forge/miniforge#install) if you need Conda.
2. Create the CellMatchR environment.

   ```bash
   conda env create -f environment.yml
   ```

3. In Finder, double-click the app that matches your macOS version.
   - **macOS 27 or later** — `CellMatchR (macOS 27+).app`
   - **macOS 26 or earlier** — `CellMatchR (macOS 26 and earlier).app`

### Linux

1. Install [Miniforge](https://github.com/conda-forge/miniforge#install) if you need Conda.
2. Create the environment and add CellMatchR to your app menu.

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
