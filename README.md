# CellMatchR

<img src="CellMatchr_Logo.png" width="320" alt="CellMatchR Logo"/>

Neuronal confocal image segmentation, manual validation, and colocalization analysis pipeline.

Supports **Zeiss CZI** and **Olympus VSI** slide-scanner images.  
Uses [Micro-SAM](https://github.com/computational-cell-analytics/micro-sam) (Segment Anything for microscopy) for automatic cell segmentation.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1  — Create / Open Project                                │
│    • New project: choose an output folder, name it              │
│    • Open existing: select cellmatchr_project.json              │
├─────────────────────────────────────────────────────────────────┤
│  Step 2  — Input & Paths                                        │
│    • CZI / TIF  →  point to root input folder                   │
│    • VSI        →  launch VSI Converter (draws ROIs,            │
│                    exports full-resolution TIFFs)                │
├─────────────────────────────────────────────────────────────────┤
│  Step 3  — Segmentation                                         │
│    • Tiled MicroSAM prediction on chosen channels               │
│    • Configurable model, preprocessing, size / shape filters    │
├─────────────────────────────────────────────────────────────────┤
│  Step 4  — Validation  (optional)                               │
│    • Name and colour each channel                               │
│    • Manual YES / NO / MAYBE / ARTIFACT review per cell ROI     │
│    • Results saved as TSV; ✔ shown only when all ROIs are done  │
├─────────────────────────────────────────────────────────────────┤
│  Step 5  — Analysis  (optional)                                 │
│    • IoU-based colocalization between two channels              │
│    • Per-ROI cell counts + colocalization percentages           │
│    • QC PNG overlays; combined summary saved to Analysis/        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

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

The simplest way is to use the provided `environment.yml`, which specifies all
required packages with flexible version constraints (works on macOS arm64 and
Linux x86-64):

```bash
conda env create -f environment.yml
conda activate cellmatchr
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

## Launching CellMatchR

### Option A — double-click (macOS)

Double-click **`launch_cellmatchr.command`** in Finder.  
The first time, right-click → **Open** to bypass Gatekeeper.

The script automatically:
1. Finds your conda installation (miniforge3, miniconda3, anaconda3, …)
2. Activates the `cellmatchr` environment
3. Launches the CellMatchR wizard from the correct directory

> The `.command` file must stay in the **CellMatchR root folder** (next to
> `run_cellmatchr.py`) — you can create an alias / shortcut anywhere else.

### Option B — terminal

```bash
conda activate cellmatchr
python run_cellmatchr.py
```

### Option C — entry-point (after `pip install -e .`)

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

```
<project_output_folder>/
├── cellmatchr_project.json          # saved project settings
├── VSI_Converted_TIFF/              # (VSI only) exported full-res TIFFs
├── Segmentation/                    # MicroSAM prediction masks + QC images
│   └── <sample_name>/
│       ├── *_predict_original_*.tif
│       ├── *_split_ch*.tif
│       └── ROI Validation Results/
│           ├── roi_verification_results.tsv
│           └── _validation_complete   (written when all ROIs are reviewed)
└── Analysis/                        # colocalization results
    ├── combined_analysis_summary.csv
    ├── combined_analysis_summary.xlsx
    └── <sample_name>/
        └── analysis_results/
            ├── *_summary.csv
            ├── *_cell_of_interest_object_table.csv
            └── *_QC_overlay.tif
```

---

## Repository structure

```
CellMatchr/
├── CellMatchr_Logo.png            # application logo
├── launch_cellmatchr.command      # macOS double-click launcher
├── run_cellmatchr.py              # CLI entry point
├── environment.yml                # conda environment (flexible versions)
├── pyproject.toml
├── requirements.txt
└── cellmatchr/
    ├── __init__.py
    ├── config.py                  # config load/save (~/.cellmatchr/config.json)
    ├── launcher.py                # tkinter step-by-step wizard
    ├── vsi_converter.py           # VSI → TIF converter (napari / Qt)
    ├── segmentation.py            # MicroSAM tiled segmentation
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
