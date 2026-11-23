"""Run segmentation and analysis for independent folders using a project template."""

from datetime import datetime
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import sys
import tempfile

from .config import PROJECT_CONFIG_FILE
from .file_paths import filesystem_path
from .legacy_cellpose import segmentation_command
from .roi_mask_matching import roi_mask_files
from .slide_signal import is_brightfield


BATCH_FOLDER = "Large_Batch_Processing"
INPUT_EXTENSIONS = (".czi", ".lif", ".nd2", ".oib", ".oif", ".tif", ".tiff")
MASK_FOLDER = "DONOTDELETE-MASK_FOR_VSI"


def batch_input_folder(folder, config):
    """Open a converter's image exports when its parent folder was selected."""
    path = Path(folder).expanduser().resolve()
    if (config.get("input_type") == "vsi" and (path / "exports").is_dir()
            and (path.name.startswith("VSI_Converted_TIFF") or (path / MASK_FOLDER).is_dir())):
        return path / "exports"
    return path


def _direct_only(folder, config):
    # Older converter layouts put the crop TIFFs beside mask and preview folders.
    return bool(config.get("seg_skip_subdirs") or (
        config.get("input_type") == "vsi" and (folder / MASK_FOLDER).is_dir()))


def _input_problem(folder, config, masks=None):
    if not folder.is_dir():
        return "Choose an existing image folder."
    try:
        entries = list(folder.iterdir())
        files = [p for p in entries if p.is_file()]
        if not _direct_only(folder, config):
            for sub in entries:
                if sub.is_dir():
                    files.extend(p for p in sub.iterdir() if p.is_file())
    except OSError as exc:
        return f"Could not read the folder. {exc}"
    images = [p for p in files if p.suffix.lower() in INPUT_EXTENSIONS]
    if config.get("input_type") == "vsi":
        if not images or any(p.suffix.lower() not in (".tif", ".tiff") for p in images):
            return "Choose exported slide TIFFs. Convert raw slide scans in Step 2 first."
    elif not images:
        return "No supported microscopy images were found."
    if any(p.name.lower().endswith(("_mask_overview.tif", "_mask_overview.tiff"))
           or masks and p.parent.resolve() == masks for p in images):
        return "The image folder also includes ROI masks. Choose the image exports folder."
    return ""


def batch_settings_issue(config):
    """Return the first incomplete project setting and its setup page."""
    project = str(config.get("project_output_folder") or "").strip()
    if not project or not (Path(project) / PROJECT_CONFIG_FILE).is_file():
        return 0, "Create or load the project whose settings you want to reuse."

    source = str(config.get("input_root_folder") or "").strip()
    if config.get("input_type") not in ("czi", "vsi") or not source:
        return 1, "Select the template project's input format and image folder."
    problem = _input_problem(batch_input_folder(source, config), config)
    if problem:
        return 1, f"Check the template project's input. {problem}"

    brightfield = is_brightfield(config)
    try:
        raw_channels = str(config.get("seg_channels") or "").split(",")
        channels = {int(ch.strip()) for ch in raw_channels if ch.strip()}
    except ValueError:
        channels = set()
    if not channels or min(channels) < 0 or (brightfield and channels != {0}):
        return 2, "Choose the channels to predict. A brightfield project uses channel 0."
    try:
        size_min = int(config["seg_min_object_size"])
        size_max = int(config["seg_max_object_size"])
        intensity_min = float(config["seg_raw_mean_min"])
        intensity_max = float(config["seg_raw_mean_max"])
        ratio = float(config["seg_max_ratio"])
        if (min(size_min, size_max) < 0 or (size_max and size_min > size_max)
                or not all(math.isfinite(n) and n >= 0 for n in (intensity_min, intensity_max, ratio))
                or (intensity_max and intensity_min > intensity_max)):
            raise ValueError
        if any(int(config[key]) <= 0 for key in ("seg_tile_h", "seg_tile_w", "seg_qc_downsample")):
            raise ValueError
        if any(int(config[key]) < 0 for key in ("seg_halo_h", "seg_halo_w")):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        return 2, "Check the object filters, tile sizes, and QC downsampling in Segmentation."

    backend = config.get("seg_backend")
    checkpoint_key = {"microsam": "seg_custom_model_path", "cellpose": "seg_cellpose_model_path",
                      "stardist": "seg_stardist_model_path", "microsam2": None}
    if backend not in checkpoint_key:
        return 2, "Choose a segmentation backend and model."
    key = checkpoint_key[backend]
    checkpoint = str(config.get(key) or "").strip() if key else ""
    if checkpoint and not Path(checkpoint).expanduser().exists():
        return 2, "Locate the trained model or choose a built-in model."

    try:
        primary = 0 if brightfield else int(config.get("ana_red_channel", ""))
        compare = len(channels) > 1 and not brightfield and config.get("ana_coloc_enabled")
        marker = int(config.get("ana_pv_channel", "")) if compare else None
        iou = float(config.get("ana_iou_threshold", "")) if compare else 0
        if (primary not in channels or (compare and (marker not in channels or marker == primary))
                or not (math.isfinite(iou) and 0 <= iou <= 1)):
            raise ValueError
    except (TypeError, ValueError):
        return 4, "Set the measurement channel and any colocalization settings in Analysis."

    subfolder = str(config.get("ana_subfolder_name") or "").strip()
    if (not subfolder or subfolder in (".", "..") or Path(subfolder).name != subfolder
            or PureWindowsPath(subfolder).name != subfolder):
        return 4, "Set a simple analysis subfolder name in Output options."
    if config.get("ana_roi_cell_rule") not in ("centroid", "any_pixel"):
        return 4, "Choose a cell inclusion rule in ROI options."
    return None


def automatic_batch_masks(folder):
    candidates = [folder / MASK_FOLDER]
    if folder.name == "exports":
        candidates.append(folder.parent / MASK_FOLDER)
    return next((path for path in candidates if roi_mask_files(path)), None)


def batch_target_problem(folder, masks, config):
    """Check a row without changing any files or reading microscopy pixels."""
    if masks is not None:
        try:
            if not roi_mask_files(masks):
                return "No TIFF or Fiji ROI masks were found in the selected mask folder."
        except OSError as exc:
            return f"Could not read the ROI mask folder. {exc}"
    return _input_problem(folder, config, masks)


def _write_json(path, data):
    temp = path.with_suffix(".tmp")
    with open(filesystem_path(temp), "w", encoding="utf-8") as target:
        json.dump(data, target, indent=2)
    os.replace(filesystem_path(temp), filesystem_path(path))


def _analysis_outcome(folder):
    with open(filesystem_path(folder / "analysis_run_summary.json"), encoding="utf-8") as source:
        report = json.load(source)
    if (not report.get("workbook_written") or not report.get("analyzed")
            or not (folder / "combined_analysis_summary.xlsx").is_file()):
        raise RuntimeError("Analysis did not produce measurements. Check this folder's analysis log.")
    message = (f"{report['analyzed']} analyzed, {report.get('skipped_missing_channel', 0)} skipped, "
               f"{report.get('errors', 0)} errors")
    return ("warnings" if report.get("status") == "complete_with_warnings" else "complete", message)


def run_batch(config, targets, folder, *, run_stage, stopped, on_update, log):
    """Process one folder at a time and retain errors and partial results."""
    report_path = folder / "batch_report.json"
    rows = [{"input": str(image_root), "roi_masks": str(masks or ""), "status": "queued"}
            for image_root, masks in targets]

    def status(index, value, message=""):
        row = rows[index]
        row.update(status=value, detail=message)
        _write_json(report_path, rows)
        on_update(index, value, message)

    _write_json(report_path, rows)
    scripts = Path(__file__).resolve().parent
    for index, (image_root, masks) in enumerate(targets):
        if stopped():
            break
        name = re.sub(r"[^\w.-]+", "_", image_root.name).strip("._")[:65] or "images"
        output = folder / f"{index + 1:03d}_{name}"
        seg = output / "Segmentation"
        ana = output / "Analysis"
        rows[index]["output"] = str(output)
        log(f"\nFolder {index + 1}/{len(targets)} · {image_root}")
        try:
            problem = batch_target_problem(image_root, masks, config)
            if problem:
                raise ValueError(problem)
            output.mkdir()
            job = dict(config, project_name=image_root.name, project_output_folder=str(output),
                       input_root_folder=str(image_root), vsi_output_dir="", seg_input_subfolder="Segmentation",
                       seg_test_run=False, seg_dry_run=False, seg_resume=False,
                       seg_skip_subdirs=_direct_only(image_root, config), val_main_folder="",
                       ana_enabled=True, ana_roi_mask_folder=str(masks or ""), ana_roi_mask_choices={},
                       ana_microsam_output_root=str(seg), output_root_folder=str(ana))
            if is_brightfield(job):
                job.update(ana_red_channel="0", ana_coloc_enabled=False)
            elif len({int(x.strip()) for x in job["seg_channels"].split(",") if x.strip()}) == 1:
                job["ana_coloc_enabled"] = False
            for key in ("seg_custom_model_path", "seg_cellpose_model_path", "seg_stardist_model_path"):
                if job.get(key):
                    job[key] = str(Path(job[key]).expanduser().resolve())
            _write_json(output / PROJECT_CONFIG_FILE, job)
            status(index, "segmenting")
            seg_config = dict(job, output_root_folder=str(seg))
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "segmentation.json"
                _write_json(path, seg_config)
                command, env = segmentation_command(str(scripts / "segmentation.py"), str(path), seg_config)
                if stopped():
                    status(index, "cancelled")
                    break
                result = run_stage(command, env)
            if stopped():
                status(index, "cancelled")
                break
            if result:
                raise RuntimeError(f"Segmentation exited with code {result}.")

            status(index, "analyzing")
            result = run_stage([sys.executable, str(scripts / "analysis.py"), "--config",
                                str(output / PROJECT_CONFIG_FILE), "--use-saved-reviews"], None)
            if stopped():
                status(index, "cancelled")
                break
            if result:
                raise RuntimeError(f"Analysis exited with code {result}.")
            outcome, message = _analysis_outcome(ana)
            status(index, outcome, message)
            log(f"{outcome.capitalize()} · {message}")
        except Exception as exc:
            status(index, "cancelled" if stopped() else "failed", str(exc))
            log(f"{'Cancelled' if stopped() else 'Error'} · {exc}")

    for index, row in enumerate(rows):
        if row["status"] == "queued":
            status(index, "not run")
    return rows


def create_batch_folder(base):
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=base))
