# CellMatchR user guide

For setup and the launcher for your platform, start with the [README](../README.md).

## Images and regions

- **Confocal** reads CZI, confocal LIF, TIFF, ND2, OIB, and OIF directly.
- **Slide scanner** opens Olympus VSI, stitched Leica LIF, TIFF, and BigTIFF in
  the region selector. Keep a VSI next to its matching `_name_` companion folder.
- For a large TIFF or an already-converted scan, open **Step 2 → Slide scanner →
  Launch Slide Scanner Converter**. Draw and name each ROI, then save and export.
  The converter writes full-resolution TIFF crops and matching ROI masks.
- TIFF previews use a stored pyramid when available. TIFFs with explicit Z and
  time axes use a maximum-intensity Z projection of the first time point.
  Tiled and compressed TIFF reads require Zarr 3 from the CellMatchR environment.
- For brightfield, choose **Brightfield stain** in Step 2. Use the Parameter
  Wizard to check thresholds on a representative region.

## Review, analysis, and retraining

Edit missed or incorrect masks in **Step 4 → Review Segmented Images**, then
finish the per-channel validation before using the reviewed masks for training.
**Step 5 → Analysis** produces object tables and an Excel workbook from two
segmented channels. A rerun backs up the previous results in `.analysis_backups`.

In **Step 7 → Retrain Models**, choose the segmentation run, channel, and base
model. Mark accepted cells **YES** and resolve **MAYBE** decisions before
training. Keep at least two reviewed samples so training and validation can use
different samples. Choose the saved model in Step 3 for a later run.

### Training devices

| Backend | Native Windows training |
|---------|-------------------------|
| Cellpose-SAM | NVIDIA GPU when PyTorch detects CUDA, otherwise CPU |
| Micro-SAM | NVIDIA GPU when PyTorch detects CUDA, otherwise CPU |
| StarDist | CPU with the optional TensorFlow 2.16.2 environment |

The Windows workflow was smoke-tested from the CLI and GUI. Cellpose-SAM and
Micro-SAM used an RTX 3090. StarDist GPU training uses a CUDA-enabled TensorFlow
environment on Linux or under WSL2. Native Windows CUDA support in TensorFlow
ended with 2.10. [TensorFlow installation guide](https://www.tensorflow.org/install/pip)

### Optional StarDist setup on Windows

From PowerShell in the CellMatchR folder, after running `setup-windows.cmd`:

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

Choose **StarDist** in this session. Start a fresh PowerShell session for the
default Micro-SAM and Cellpose runtime. For a StarDist CLI session, set
`$env:PYTHONPATH = $stardist`.

## Command-line entry points

Run from the CellMatchR folder in the active CellMatchR environment.

```bash
python -m cellmatchr.segmentation --help
python -m cellmatchr.analysis --help
python -m cellmatchr.retraining --help
```

On Windows, set `$runtime`, `$mamba`, `$prefix`, and `$env:MAMBA_ROOT_PREFIX`
as above. Use `& $mamba --no-rc run --prefix $prefix python -m cellmatchr.retraining --help`
to run a module in the installed environment. The GUI launchers set model-cache
paths automatically.

## Launcher troubleshooting

- **Windows:** Run `.\setup-windows.cmd -ShortcutOnly` from PowerShell to recreate
  the logo shortcut after moving the checkout. To set up a CPU-only runtime,
  run `.\setup-windows.cmd -Device CPU`. Set `CELLMATCHR_WINDOWS_RUNTIME` in both
  setup and launch sessions to use a different short runtime path.
- **macOS:** Keep the `.app` bundle in the CellMatchR folder. Use
  `scripts/launch_cellmatchr.command` for a Terminal log, or run
  `conda activate cellmatchr` followed by `python scripts/run_cellmatchr.py`.
  If ZIP extraction stripped executable permissions, run the following once
  in the CellMatchR folder.

  ```bash
  chmod +x 'CellMatchR (macOS 26 and earlier).app/Contents/MacOS/CellMatchR' \
    'CellMatchR (macOS 27+).app/Contents/MacOS/applet' scripts/launch_cellmatchr.command
  ```

- **Linux:** Run `bash scripts/install_linux_launcher.sh` again if the checkout
  moved. Run `bash run-linux.sh` in a terminal to see startup errors. For a
  custom environment path, set `CELLMATCHR_LINUX_ENV` in the environment of
  the terminal or desktop session that starts CellMatchR.
