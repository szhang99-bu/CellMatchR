# CellMatchR for Windows — installation preview

This branch prepares **native Windows 11, x86-64** installation from Git. It
uses the same project format and image-processing code as the Mac version. The
Windows launcher and environment are separate. The first installation uses
CPU PyTorch so the app can be checked before choosing an NVIDIA build.

The package list has been solved for `win-64` from macOS, and the platform
independent tests pass. **The installer, windows, image readers, segmentation,
analysis, validation, and training still need a real Windows run.**

## Install on the Windows PC

1. Install [Miniforge for Windows](https://github.com/conda-forge/miniforge#miniforge3)
   and [Git for Windows](https://git-scm.com/download/win). Open the **Miniforge
   Prompt** from the Start menu. Use that prompt for the commands below.
2. Clone the Windows branch into its own folder:

   ```bat
   cd /d "%USERPROFILE%"
   git clone --branch feature/windows-preview --single-branch https://github.com/szhang99-bu/CellMatchR.git CellMatchR-Windows
   cd CellMatchR-Windows
   ```

3. Create the separate Windows environment. The NumPy constraint also applies
   to the pip packages installed by conda:

   ```bat
   set "PIP_CONSTRAINT=%CD%\scripts\windows-pip-constraints.txt"
   conda env create -f environment-windows.yml
   set "PIP_CONSTRAINT="
   conda activate cellmatchr-windows
   python -m pip install --no-deps -e .
   ```

4. Check imports, the selected PyTorch device, and a hidden Napari window:

   ```bat
   python scripts\windows_preflight.py --gui
   ```

5. Double-click **`run-windows.cmd`** in the repository folder, or run it from
   the Miniforge Prompt. It finds the `cellmatchr-windows` environment and opens
   the normal CellMatchR wizard. Start with a small test project and one image.

If Miniforge is installed in a custom directory and double-clicking cannot find
the environment, run from an activated Miniforge Prompt, or set
`CELLMATCHR_WINDOWS_PYTHON` to that environment's full `python.exe` path.

The macOS `.app` launchers are for Mac only. They stay in the repository for
shared development but are not part of Windows installation.

## Optional NVIDIA GPU

The initial environment is CPU based. If `nvidia-smi` shows an NVIDIA GPU and
an appropriate driver, choose a Windows CUDA wheel for **PyTorch 2.10.0 and
torchvision 0.25.0** from the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/).
For example, if that machine supports CUDA 12.8:

```bat
conda activate cellmatchr-windows
set "PIP_CONSTRAINT=%CD%\scripts\windows-pip-constraints.txt"
python -m pip install --upgrade --force-reinstall "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cu128
set "PIP_CONSTRAINT="
python scripts\windows_preflight.py --gui
```

The preflight output should name the GPU rather than `CPU`. Choose the CUDA
wheel that matches the PC's driver; the `cu128` command is an example, not a
requirement for every PC.

## Optional StarDist backend

StarDist is not needed for the main Cellpose-SAM or Micro-SAM workflows. To
try its prediction and retraining options, install the optional packages in
the same Windows environment, then run the extra check:

```bat
conda activate cellmatchr-windows
set "PIP_CONSTRAINT=%CD%\scripts\windows-pip-constraints.txt"
python -m pip install "tensorflow==2.15.1" "stardist==0.9.1"
set "PIP_CONSTRAINT="
python scripts\windows_preflight.py --stardist
```

Native Windows TensorFlow versions after 2.10 do not support CUDA GPUs, so
this StarDist setup is **CPU based**. [TensorFlow's Windows guidance](https://www.tensorflow.org/install/pip)
describes the limitation.

## Monday test order

1. Run `windows_preflight.py --gui`; save its output if anything fails.
2. Open the wizard and create a new test project. Confirm paths point to the
   Windows drive, then load a small TIFF or slide scanner image.
3. Run segmentation with Cellpose-SAM and Micro-SAM; inspect mask TIFFs and
   the correction editor. Run validation and analysis through Excel export.
4. Test model retraining on reviewed examples. If installed, test StarDist and
   its training path separately.
5. Open a **copy** of an existing project if cross-machine compatibility is
   needed. The project folder is relocated automatically, but raw-input and
   external-model paths from the Mac may need reselection on Windows.

The Windows branch remains separate until this real-PC test is complete.
