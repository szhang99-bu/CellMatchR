# CellMatchR

<img src="CellMatchr_Logo.png" width="320" alt="CellMatchR Logo"/>

Neuronal confocal image segmentation, manual validation, and colocalization analysis pipeline.

Reads images directly from most major microscope brands, plus **Olympus `.vsi`**
slide-scanner files via the built-in VSI Converter.
Segmentation runs with either [Micro-SAM](https://github.com/computational-cell-analytics/micro-sam)
or [Cellpose-SAM](https://github.com/MouseLand/cellpose) (selectable per run).

**Supported input formats**

| Format | Brand | Status |
|--------|-------|--------|
| `.czi` | Zeiss | ✅ tested |
| `.lif` | Leica (confocal **and** slide scanner) | ✅ tested |
| `.tif` / `.tiff` | any | ✅ tested |
| `.nd2` | Nikon | 🧪 **beta** — not yet validated on a real file; use with caution |
| `.oib` / `.oif` | Olympus FluoView | 🧪 **beta** — not yet validated on a real file; use with caution |
| `.vsi` | Olympus slide scanner | ✅ tested (via VSI Converter) |
| `.oir` | Olympus FV3000 | ❌ not supported — convert to `.tif` in [Fiji/ImageJ](https://imagej.net/software/fiji/) first |

> Leica writes both confocal and slide-scanner data as `.lif`. CellMatchR detects
> the difference automatically: each confocal acquisition becomes its own sample,
> while a slide-scanner file uses its stitched/merged image.

### Using Leica LIF files

Choose the workflow by acquisition type, even when both files have the `.lif`
extension or the same filename in different folders:

1. For **confocal LIF**, choose **Confocal / direct read** in Step 2. Select a
   folder containing the files directly or in one level of subfolders. The file
   list identifies confocal and slide-scanner LIF files. A slide scan in this
   folder is skipped; each confocal acquisition becomes a separate sample.
2. For **slide-scanner LIF**, choose **Slide scanner** in Step 2 and open the
   LIF in the converter. It offers stitched images; draw the regions you want
   to analyze and export them. Step 3 segments those exported TIFFs, and
   Step 5 analyzes the resulting masks with the matching ROI mask folder. A
   confocal-only LIF is directed back to the confocal workflow.
3. If you use both workflows in one project, keep each segmentation run in its
   own folder. **Run All** asks whether to analyze an existing run or segment
   the current input as a new run. Choose the intended run for analysis. The
   matching converter ROI masks follow the selected segmentation run when its
   source is recorded. For an older run without a source record, confirm the
   ROI mask folder before analysis. Direct-read runs leave converter ROI masks
   out of analysis.

When two direct-read acquisitions would otherwise produce the same sample
name, CellMatchR adds a stable source-based suffix to the new sample folders.
Unique inputs keep their established names, so existing projects need no
renaming. If two slide scans also share a filename, use separate converter
output runs for them.
Large stitched slide scans may need substantial memory during ROI conversion.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1  — Create / Open Project                                │
│    • New project: choose an output folder, name it              │
│    • Open existing: select cellmatchr_project.json              │
├─────────────────────────────────────────────────────────────────┤
│  Step 2  — Input & Paths                                        │
│    • Direct read → CZI / LIF / ND2 / OIB-OIF / TIF              │
│    • Slide scanner → Olympus VSI or Leica LIF: draw ROIs in     │
│                    the converter, export full-res TIFFs         │
├─────────────────────────────────────────────────────────────────┤
│  Step 3  — Segmentation                                         │
│    • Tiled Micro-SAM or Cellpose-SAM on chosen channels         │
│    • Tune object filters with the Parameter Wizard if needed    │
│    • Configure preprocessing, filters, and resume behavior      │
├─────────────────────────────────────────────────────────────────┤
│  Step 4  — Validation  (optional)                               │
│    • Review masks and add missed cells in the ROI editor        │
│    • Name/colour channels and review each detected cell         │
│    • Per-channel TSV and completion marker for each sample      │
├─────────────────────────────────────────────────────────────────┤
│  Step 5  — Analysis  (optional)                                 │
│    • IoU-based colocalization between two channels              │
│    • Per-sample CSV tables and QC TIFF overlays                 │
│    • Combined Excel summary saved to Analysis                   │
├─────────────────────────────────────────────────────────────────┤
│  Step 6  — Run Pipeline                                         │
│    • Run pending segmentation and analysis steps together       │
│    • Validation remains an interactive step                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites — install **conda** and **git** (one-time)

CellMatchR is installed into a `conda` environment, and the easiest way to get
(and later update) the code is with `git`. Install these two free tools first —
if you already have them, skip ahead to step 1.

**conda** — a Python environment/package manager. Install **Miniforge**
(recommended — lightweight, conda-forge based) or Miniconda:

- Miniforge install guide: <https://github.com/conda-forge/miniforge#install>
- Miniconda install guide: <https://www.anaconda.com/docs/getting-started/miniconda/install/>
- New to conda? Getting-started guide:
  <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>

**git** — version control, used to download and update the code (needed for
Option B below):

- Download git: <https://git-scm.com/downloads>
- GitHub's "Set up Git" walkthrough:
  <https://docs.github.com/en/get-started/getting-started-with-git/set-up-git>

**Verify both are installed** — open a **new** Terminal and run:

```bash
conda --version     # e.g. conda 24.11.3
git --version       # e.g. git version 2.43.0
```

If you see a version number for each, you're ready. If you get
`conda: command not found` (or `git: command not found`), the install didn't
finish hooking into your shell — **quit and reopen Terminal**, and on macOS/Linux
make sure the installer's init step ran (e.g. `conda init "$(basename "$SHELL")"`
then reopen Terminal). Then re-run the checks above.

### 1. Get the code

#### Option A — Download ZIP *(no git required, easiest for most users but difficult for later updates)*

1. Go to **https://github.com/szhang99-bu/CellMatchR**
2. Click the green **`<> Code`** button at the top of the page → **Download ZIP**
3. Unzip it to wherever you want to keep the software, e.g. your Desktop or Documents
4. Rename the unzipped folder from `CellMatchR-main` → `CellMatchR`

#### Option B — Clone with git

Replace the destination path with wherever you want to save it:

```bash
# macOS / Linux example — save to your Documents folder:
git clone https://github.com/szhang99-bu/CellMatchR.git ~/Documents/CellMatchR
cd ~/Documents/CellMatchR

# or save to the Desktop:
git clone https://github.com/szhang99-bu/CellMatchR.git ~/Desktop/CellMatchR
cd ~/Desktop/CellMatchR
```

### 2. Create the conda environment

First, confirm conda is working and that you're inside the project folder (the
one containing `environment.yml`):

```bash
conda --version                 # should print a version (see Prerequisites if not)
cd /path/to/CellMatchR          # the folder from step 1
ls environment.yml              # should list the file
```

> If `conda --version` fails, finish the **Prerequisites** above (install
> Miniforge/Miniconda, then reopen Terminal). Miniforge is recommended because
> it defaults to the free **conda-forge** channel; with Miniconda you may be
> prompted to accept the Anaconda Terms of Service the first time — accept it,
> or run `conda config --add channels conda-forge`.

Then build the environment from `environment.yml`. It pins Python and the
versions of several key packages used in the working environment, while some
other packages have version ranges. This downloads a lot of packages, so it
can take **10–20+ minutes**:

```bash
conda env create -f environment.yml
conda activate cellmatchr
```

Confirm it worked — your prompt should now start with `(cellmatchr)`:

```bash
conda env list                  # 'cellmatchr' should appear in the list
python -c "import torch, micro_sam; print('env OK')"
```

> **GPU users:** The `environment.yml` installs a CPU-only PyTorch build.
> After creating the environment, upgrade to a GPU build following the
> [official PyTorch guide](https://pytorch.org/get-started/locally/) — e.g.
> for CUDA 12.1:
> ```bash
> conda activate cellmatchr
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Install CellMatchR itself

```bash
pip install -e .
```

---

## Updating an existing install (other computers)

If a computer already has a working `cellmatchr` environment from an earlier
version, you only need to (1) get the new code and (2) add the new microscope
readers — **you do not need to rebuild the conda environment.**

```bash
# 1. Get the latest code
cd /path/to/CellMatchR
git pull                      # (or re-download the ZIP and replace the folder)

# 2. Add the new file-format readers + Cellpose-SAM backend into the env.
#    --no-deps is IMPORTANT: it adds only these packages and will NOT touch
#    numpy or anything else already installed.
conda activate cellmatchr
pip install --no-deps "readlif==0.6.5" "nd2==0.11.3" "oiffile==2026.2.8"
pip install "cellpose==4.1.1"     # Cellpose-SAM backend (deps already present; won't move numpy/torch)
```

> ⚠️ Use the exact versions above. In particular **do not** let `readlif`
> install 0.6.6, and do not let `cellpose` pull a version needing `numpy>=2` —
> that would break the segmentation stack (this env must stay on `numpy 1.26.x`).
> The pinned versions above are all numpy-1.26-compatible.

**About the macOS apps:** `CellMatchR.app` (macOS 26 and earlier) and
`CellMatchR (macOS 27+).app` ship with the code, so a `git pull` updates them
automatically. If you got the code as a **ZIP** (which can drop the executable
bit), re-arm the launchers once after unzipping:

```bash
chmod +x "CellMatchR.app/Contents/MacOS/CellMatchR" "CellMatchR (macOS 27+).app/Contents/MacOS/applet"
```

---

## Launching CellMatchR

> ⚠️ **On macOS 27 (Golden Gate) or later, launch with `CellMatchR (macOS 27+).app`.**
> It exists so CellMatchR can launch on macOS 27+; the original `CellMatchR.app`
> no longer can (it fails with `Operation not permitted`). On macOS 26 (Tahoe)
> or earlier, keep using `CellMatchR.app`.

### Option A — double-click the app (macOS, recommended)

Pick the app that matches your macOS version. Both are in the project folder:

| Your macOS | Double-click |
|------------|--------------|
| **macOS 27 (Golden Gate) or later** | **`CellMatchR (macOS 27+).app`** |
| macOS 26 (Tahoe) or earlier | `CellMatchR.app` |

Either app opens a Terminal window (which shows the live log), finds your conda
installation, and runs the wizard.

> **Why two apps?** macOS 27 stopped letting an app borrow Terminal's permission
> to read the Desktop, which `CellMatchR.app` relies on. On macOS 27 it opens a
> Terminal window that fails with `Operation not permitted`.
> `CellMatchR (macOS 27+).app` is built differently to work with that change,
> and it won't open on older macOS. Rebuild it with
> `bash scripts/build_cellmatchr_app_macos27.sh`.

**First launch (one time):**

- macOS Gatekeeper blocks unsigned apps, so the first time **right-click the
  app → Open → Open** to approve it. After that, a normal double-click works.
- **macOS 27+ only:** macOS may ask to allow **"CellMatchR (macOS 27+)"** to
  access files in your **Desktop** folder. Click **Allow**.

**Important — keep the app in the project folder:**

- The app must stay **inside the CellMatchR folder** (at the project root); it
  locates the code relative to its own position.
- To launch it from the **Dock or Applications**, do **not** move it — instead
  make an **alias** (right-click → **Make Alias**) and put the *alias* there, or
  drag the app to the Dock while holding **⌥⌘** (which creates a Dock alias).
  An alias remembers the original location, so the app still finds the code.
- If `CellMatchR (macOS 27+).app` is moved out of the folder, it shows an alert
  explaining this instead of failing silently.

> Got the code as a ZIP? ZIP downloads can strip the executable bit, so run
> this once in the project folder (see *Updating an existing install* above):
>
> ```bash
> chmod +x "CellMatchR.app/Contents/MacOS/CellMatchR" "CellMatchR (macOS 27+).app/Contents/MacOS/applet"
> ```

### Option B — double-click the `.command` (macOS, shows a Terminal/log window)

Double-click **`scripts/launch_cellmatchr.command`** in Finder (first time:
right-click → **Open**). This does the same thing as the app but keeps a Terminal
window open showing the live log — handy for debugging. Keep it inside the
`scripts/` folder.

### Option C — terminal

```bash
conda activate cellmatchr
python scripts/run_cellmatchr.py
```

### Option D — entry-point (after `pip install -e .`)

```bash
conda activate cellmatchr
cellmatchr
```

---

## Project files

When you create or save a project, CellMatchR writes a
`cellmatchr_project.json` into the project output folder.  
To re-open the project later, go to **Step 1 → Open Existing Project** and
click **Browse…** to select that `.json` file.

Settings are also cached in `~/.cellmatchr/config.json` and restored
automatically on the next launch.

---

## Project output structure

The first segmentation run can use `Segmentation/`. If it already contains
masks, **Save as new run** proposes a folder such as
`Segmentation_cellpose_20261028_103000`; existing masks are kept. Choose the
segmentation run to analyze in Step 5 or Run Pipeline. Its combined results go
to `Analysis/` for the original run, or the matching `Analysis_<suffix>/` for a
named run. Mask filenames and older project folders do not need conversion.
If an unfinished older run has no recorded method settings, **Continue** asks
you to confirm that the current settings match; starting a new run is the safe
default. New runs also record which source files and Leica scenes they used;
**Continue** stops if that input plan changes. If you replace source image
content, start a new run, even when the filename stays the same.

If analysis results already exist for the selected run, a simple **Yes / No**
dialog asks whether to replace them. **No** cancels the analysis. **Yes** keeps
a recoverable copy of that run's previous generated results in
`.analysis_backups/` before writing new ones. Other runs and unrelated files
are left in place. Reopening the project recovers previous results if a
replacement was interrupted before the new workbook finished. Optional
folders and files appear only after the corresponding step is run.

```
<project_output_folder>/
├── cellmatchr_project.json             # saved project settings
├── VSI_Converted_TIFF/                 # converted slide-scanner ROIs, if used
├── ParameterWizard_preview/            # first-sample preview, if used
├── Segmentation/
│   ├── _run_progress.json              # segmentation progress/resume state
│   └── <sample_name>/
│       ├── *_split_chN.tif              # raw channel images
│       ├── *_predict_original_chN_*.tif # predicted masks
│       ├── *_QC_overlay_chN_*.png       # segmentation QC
│       ├── ROI Validation Results/
│       │   ├── roi_verification_results_chN.tsv
│       │   └── _validation_complete_chN # when that channel is fully reviewed
│       ├── .validation_history/        # original decisions before a mask edit
│       └── analysis_results/           # analysis writes beside each sample
│           ├── <sample_name>_summary.csv
│           ├── <sample_name>_channel_N_object_table.csv
│           └── <sample_name>_QC_overlay.tif
├── Segmentation_<method>_<run>/         # additional segmentation run, if used
├── Analysis/
│   ├── combined_analysis_summary.xlsx  # combined analysis output
│   ├── analysis_run_summary.json        # analyzed/skipped/error counts
│   └── analysis_errors.log              # per-sample skip and error details
├── Analysis_<method>_<run>/             # matching analysis run, if used
└── .analysis_backups/                   # previous results after Yes to replace
```

Analysis applies validation-based artifact filtering automatically only when
the chosen channel has been fully reviewed across all samples. Step 5 offers
an option to apply partial validation when you explicitly want that behavior.
If you edit a mask that already has validation decisions, the Validation step
shows **Review changed cells** and opens only the affected or still-unreviewed
cells. Decisions for unchanged cells are kept. Until that review is finished,
older versions of the app see the edited channel as incomplete; they can
revalidate it from the beginning. Once a mask has a change record, an external
replacement cannot reliably identify just the changed cells and requires a
full review. Changes made to an old project before its first tracked edit
cannot be reconstructed automatically.

If a sample is missing a required analysis channel, it is listed in the
combined Excel sheet as `skipped_missing_channel` with blank measurement cells.
Other per-sample errors are listed as `error`; analysis continues with the
remaining samples. The final log line reports how many samples were analyzed,
skipped, or had errors, and points to `analysis_errors.log`.

---

## Repository structure

```
CellMatchr/
├── CellMatchR.app/                # macOS launcher — macOS 26 (Tahoe) and earlier
├── CellMatchR (macOS 27+).app/    # macOS launcher — macOS 27 (Golden Gate) and later
├── CellMatchr_Logo.png            # application logo
├── README.md
├── CODE_REVIEW_ISSUES.md        # remaining review findings and proposed fixes
├── environment.yml                # conda environment (key versions pinned)
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── run_cellmatchr.py          # CLI entry point  (python scripts/run_cellmatchr.py)
│   ├── launch_cellmatchr.command  # macOS launcher with a Terminal/log window
│   ├── build_cellmatchr_app.sh            # rebuilds CellMatchR.app (macOS ≤ 26)
│   └── build_cellmatchr_app_macos27.sh    # rebuilds CellMatchR (macOS 27+).app
└── cellmatchr/
    ├── __init__.py
    ├── config.py                  # config load/save (~/.cellmatchr/config.json)
    ├── launcher.py                # tkinter step-by-step wizard
    ├── vsi_converter.py           # VSI/LIF slide-scanner ROI converter
    ├── lif_classify.py            # Leica LIF acquisition classification
    ├── segmentation.py            # Micro-SAM / Cellpose-SAM tiled segmentation
    ├── parameter_wizard_compute.py # preview segmentation for filter tuning
    ├── parameter_wizard.py        # interactive filter tuning
    ├── roi_editor.py              # add missed cells to segmentation masks
    ├── validation_state.py        # track mask changes needing review
    ├── validator.py               # manual ROI validation UI
    └── analysis.py                # IoU colocalization analysis
```

---

## Running individual scripts

Each tool can be run standalone with `--config` to override the config path:

```bash
python -m cellmatchr.vsi_converter   --config /path/to/config.json
python -m cellmatchr.segmentation    --config /path/to/config.json
python -m cellmatchr.validator       --config /path/to/config.json
python -m cellmatchr.analysis        --config /path/to/config.json
```
