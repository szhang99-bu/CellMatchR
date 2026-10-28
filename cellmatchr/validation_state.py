"""Focused revalidation after mask edits, without changing legacy file formats.

Old projects without a change record retain their existing TSV/marker logic.
When a reviewed mask is edited, its validation folder is backed up under
``.validation_history`` beside the sample. The active TSV is reset to a
header-only file and its completion marker is hidden until review finishes,
so older application versions also treat the edited mask as unvalidated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import tifffile


VALIDATION_FOLDER = "ROI Validation Results"
TSV_COLUMNS = ("ROI_ID", "Decision", "Note")


def sidecar_path(project_folder: str, channel: int | None) -> str:
    suffix = f"ch{channel}" if channel is not None else "legacy"
    return os.path.join(project_folder, VALIDATION_FOLDER,
                        f"validation_changes_{suffix}.json")


def _suffix(channel: int | None) -> str:
    return f"ch{channel}" if channel is not None else "legacy"


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: str, mask_name: str | None = None) -> dict:
    info = os.stat(path)
    return {
        "mask_file": mask_name or os.path.basename(path),
        "mask_size": info.st_size,
        "mask_mtime_ns": info.st_mtime_ns,
        "mask_sha256": _hash_file(path),
    }


def _label_ids(mask) -> set[int]:
    return {int(v) for v in np.unique(mask) if v > 0}


def _all_ids(mask_path: str | None) -> list[int]:
    if not mask_path or not os.path.isfile(mask_path):
        return []
    try:
        return sorted(_label_ids(tifffile.imread(mask_path)))
    except (OSError, ValueError, tifffile.TiffFileError):
        return []


def _atomic_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".validation_changes_", suffix=".tmp",
                                dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(data, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _read_rows(path: str | None) -> dict[int, dict]:
    """Read the existing three-column TSV, keeping the last row per label."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source, delimiter="\t")
            fields = {name.strip().lower(): name for name in (reader.fieldnames or [])
                      if name is not None}
            if not {"roi_id", "decision"}.issubset(fields):
                return {}
            rows = {}
            for row in reader:
                try:
                    raw_id = row[fields["roi_id"]]
                    number = float(raw_id)
                    if not number.is_integer():
                        continue
                    roi_id = int(number)
                except (TypeError, ValueError):
                    continue
                decision = (row.get(fields["decision"]) or "").strip()
                if roi_id > 0 and decision:
                    rows[roi_id] = {
                        "ROI_ID": roi_id,
                        "Decision": decision,
                        "Note": row.get(fields.get("note", "")) or "",
                    }
            return rows
    except OSError:
        return {}


def _read_external_rows_strict(path: str) -> dict[int, dict]:
    """Reject every malformed older-app row instead of silently dropping it."""
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter="\t")
        fields = {name.strip().lower(): name for name in (reader.fieldnames or [])
                  if name is not None}
        if not {"roi_id", "decision"}.issubset(fields):
            raise ValueError("required ROI_ID/Decision columns are missing")
        rows = {}
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"row {line_number} has extra columns")
            try:
                number = float(row[fields["roi_id"]])
                if not number.is_integer() or number <= 0:
                    raise ValueError("ROI_ID must be a positive whole number")
                roi_id = int(number)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {line_number} has an invalid ROI_ID") from exc
            decision = (row.get(fields["decision"]) or "").strip()
            if not decision:
                raise ValueError(f"row {line_number} has no decision")
            if roi_id in rows:
                raise ValueError(f"row {line_number} repeats ROI_ID {roi_id}")
            rows[roi_id] = {
                "ROI_ID": roi_id,
                "Decision": decision,
                "Note": row.get(fields.get("note", "")) or "",
            }
        return rows


def _write_rows(path: str, rows: dict[int, dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".validation_tsv_", suffix=".tmp",
                                dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=TSV_COLUMNS,
                                    delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for roi_id in sorted(rows):
                writer.writerow(rows[roi_id])
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _standard_tsv(project_folder: str, channel: int | None) -> str | None:
    val_dir = os.path.join(project_folder, VALIDATION_FOLDER)
    if channel is not None:
        channel_path = os.path.join(val_dir,
                                    f"roi_verification_results_ch{channel}.tsv")
        if os.path.isfile(channel_path):
            return channel_path
    legacy_path = os.path.join(val_dir, "roi_verification_results.tsv")
    return legacy_path if os.path.isfile(legacy_path) else None


def _tsv_has_body(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as source:
        next(source, None)  # header
        return any(line.strip() for line in source)


def _external_review_paths(val_dir: str, channel: int | None,
                           record: dict) -> list[str]:
    paths = [os.path.join(val_dir, record["active_tsv_name"])]
    if channel is not None:
        channel_path = os.path.join(
            val_dir, f"roi_verification_results_ch{channel}.tsv"
        )
        if channel_path not in paths:
            paths.append(channel_path)
    legacy = os.path.join(val_dir, "roi_verification_results.tsv")
    if record.get("legacy_global_marker_was_present") and legacy not in paths:
        paths.append(legacy)
    return paths


def _external_review_present(val_dir: str, channel: int | None,
                             record: dict) -> bool:
    if any(_tsv_has_body(path) for path in _external_review_paths(
            val_dir, channel, record)):
        return True
    marker = (f"_validation_complete_ch{channel}" if channel is not None
              else "_validation_complete")
    return (os.path.exists(os.path.join(val_dir, marker))
            or os.path.exists(os.path.join(val_dir, "_validation_complete")))


def _multiple_predicted_channels(project_folder: str) -> bool:
    channels = set()
    try:
        names = os.listdir(project_folder)
    except OSError:
        return False
    for name in names:
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        match = re.search(r"_predict_original_ch(\d+)", name, re.IGNORECASE)
        if match and not name.lower().startswith("debug"):
            channels.add(int(match.group(1)))
    return len(channels) > 1


def _has_validation(project_folder: str, channel: int | None) -> bool:
    val_dir = os.path.join(project_folder, VALIDATION_FOLDER)
    if _standard_tsv(project_folder, channel):
        return True
    markers = ["_validation_complete"]
    if channel is not None:
        markers.append(f"_validation_complete_ch{channel}")
    return any(os.path.exists(os.path.join(val_dir, name)) for name in markers)


def _read_record(project_folder: str, channel: int | None) -> dict:
    with open(sidecar_path(project_folder, channel), encoding="utf-8") as source:
        data = json.load(source)
    if data.get("schema_version") != 1:
        raise ValueError("unsupported validation change record")
    return data


def _unknown(reason: str, mask_path: str | None) -> dict:
    return {"status": "unknown", "pending_ids": _all_ids(mask_path),
            "reason": reason}


def get_change_state(project_folder: str, channel: int | None,
                     mask_path: str | None = None,
                     verify_hash: bool = True) -> dict:
    """Get ``legacy``, ``pending``, ``current``, or ``unknown`` mask status.

    ``legacy`` means no sidecar and preserves old-project behavior. A checksum
    mismatch gives ``unknown`` and all current labels must be reviewed.
    ``verify_hash=False`` avoids full TIFF hashing in UI folder listings;
    analysis and actual review should use the default full check.
    """
    if not os.path.exists(sidecar_path(project_folder, channel)):
        return {"status": "legacy", "pending_ids": [], "reason": ""}
    try:
        record = _read_record(project_folder, channel)
        name = record["mask_file"]
        if mask_path is None:
            mask_path = os.path.join(project_folder, name)
        if os.path.basename(mask_path) != name:
            return _unknown("Selected mask differs from the reviewed mask.",
                            mask_path)
        if not os.path.isfile(mask_path):
            return _unknown("The reviewed mask is missing.", mask_path)
        info = os.stat(mask_path)
        same_stat = (info.st_size == record["mask_size"]
                     and info.st_mtime_ns == record["mask_mtime_ns"])
        if (verify_hash or not same_stat) and _hash_file(mask_path) != record["mask_sha256"]:
            return _unknown("Mask changed outside the review editor.", mask_path)
        pending = sorted({int(v) for v in record["pending_roi_ids"] if int(v) > 0})
        if not record.get("finalized", False) and _external_review_present(
                os.path.join(project_folder, VALIDATION_FOLDER), channel,
                record):
            return {
                "status": "unknown",
                "pending_ids": sorted({int(v) for v in record["present_roi_ids"]}),
                "reason": "Validation was edited in another app; open validation to import it.",
            }
        if record.get("unknown_change"):
            return {"status": "unknown", "pending_ids": pending,
                    "reason": "Mask changes require a full review."}
        status = "pending" if pending or not record.get("finalized", False) else "current"
        return {"status": status, "pending_ids": pending, "reason": ""}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _unknown(f"Validation change record could not be read: {exc}",
                        mask_path)


def get_validation_tsv_path(project_folder: str, channel: int | None) -> str | None:
    """Return the safe TSV for current analysis, including pending review.

    While pending, the effective TSV has unchanged old decisions plus completed
    focused decisions, but excludes every queued label. The active standard
    TSV is intentionally header-only for older application versions.
    """
    state = get_change_state(project_folder, channel, verify_hash=False)
    if state["status"] == "legacy" or state["status"] == "current":
        return _standard_tsv(project_folder, channel)
    try:
        record = _read_record(project_folder, channel)
        path = os.path.join(project_folder, VALIDATION_FOLDER,
                            record["effective_tsv_name"])
        return path if os.path.isfile(path) else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _effective_rows(record: dict, val_dir: str,
                    present: set[int]) -> dict[int, dict]:
    base_name = record.get("base_tsv_name")
    base = _read_rows(os.path.join(val_dir, base_name) if base_name else None)
    interim = _read_rows(os.path.join(val_dir, record["interim_tsv_name"]))
    base.update(interim)
    pending = {int(v) for v in record["pending_roi_ids"]}
    return {roi_id: row for roi_id, row in base.items()
            if roi_id in present and roi_id not in pending}


def save_edited_mask(mask_path: str, channel: int | None,
                     old_mask: np.ndarray, new_mask: np.ndarray) -> dict:
    """Atomically replace an edited mask after making older readers safe.

    A hidden snapshot preserves all original validation files. The active
    validation directory is staged with a header-only TSV and no relevant
    completion marker, then swapped into place before the mask is replaced.
    A failed mask replacement rolls the validation directory back.
    """
    if old_mask.shape != new_mask.shape:
        raise ValueError("Edited mask shape differs from the original mask")
    sample_dir = os.path.dirname(mask_path)
    old_state = get_change_state(sample_dir, channel, mask_path)
    if old_state["status"] == "unknown":
        adopted = _adopt_external_review(sample_dir, channel, mask_path)
        if adopted is not None:
            old_state = adopted
    track = old_state["status"] != "legacy" or _has_validation(sample_dir, channel)

    fd, temp_mask = tempfile.mkstemp(prefix=".edited_mask_", suffix=".tif",
                                     dir=sample_dir)
    os.close(fd)
    try:
        tifffile.imwrite(temp_mask, new_mask.astype(np.uint32), compression="lzw")
        if not track:
            os.replace(temp_mask, mask_path)
            return get_change_state(sample_dir, channel, mask_path)

        val_dir = os.path.join(sample_dir, VALIDATION_FOLDER)
        history = os.path.join(sample_dir, ".validation_history")
        os.makedirs(history, exist_ok=True)
        snapshot = os.path.join(history, uuid.uuid4().hex)
        stage = tempfile.mkdtemp(prefix=".validation_stage_", dir=sample_dir)
        try:
            if os.path.isdir(val_dir):
                shutil.copytree(val_dir, stage, dirs_exist_ok=True)
            try:
                previous = _read_record(sample_dir, channel)
            except (OSError, ValueError, TypeError, KeyError):
                previous = None

            source_tsv = _standard_tsv(sample_dir, channel)
            original_name = os.path.basename(source_tsv) if source_tsv else (
                f"roi_verification_results_ch{channel}.tsv"
                if channel is not None else "roi_verification_results.tsv"
            )
            active_name = (previous or {}).get("active_tsv_name", original_name)
            suffix = _suffix(channel)
            interim_name = f".validation_new_reviews_{suffix}.tsv"
            effective_name = f".validation_effective_{suffix}.tsv"
            base_name = f".validation_baseline_{suffix}.tsv"
            if previous:
                interim_name = previous.get("interim_tsv_name", interim_name)
                effective_name = previous.get("effective_tsv_name", effective_name)
                base_name = previous.get("base_tsv_name", base_name)
            base_path = os.path.join(stage, base_name)
            if not os.path.isfile(base_path):
                # Keep the baseline inside the project so pending reviews
                # remain valid after copying the project to another machine.
                source = (previous or {}).get("base_tsv") or source_tsv
                if source and os.path.isfile(source):
                    shutil.copyfile(source, base_path)
                else:
                    _write_rows(base_path, {})

            present = _label_ids(new_mask)
            changed = old_mask != new_mask
            touched = set()
            if np.any(changed):
                touched |= _label_ids(old_mask[changed])
                touched |= _label_ids(new_mask[changed])
            interim_path = os.path.join(stage, interim_name)
            interim = _read_rows(interim_path)
            for roi_id in touched:
                interim.pop(roi_id, None)
            _write_rows(interim_path, interim)
            reviewed = set(_read_rows(base_path))
            reviewed |= set(interim)
            pending = set(old_state["pending_ids"]) | (touched & present)
            pending |= present - reviewed
            pending &= present
            if old_state["status"] == "unknown":
                pending = set(present)

            record = {
                "schema_version": 1,
                **_identity(temp_mask, os.path.basename(mask_path)),
                "pending_roi_ids": sorted(pending),
                "present_roi_ids": sorted(present),
                "unknown_change": False,
                "finalized": False,
                "active_tsv_name": active_name,
                "interim_tsv_name": interim_name,
                "effective_tsv_name": effective_name,
                "base_tsv_name": base_name,
                "legacy_global_marker_was_present": (
                    (previous or {}).get("legacy_global_marker_was_present", False)
                    or os.path.isfile(os.path.join(val_dir, "_validation_complete"))
                ),
            }
            _write_rows(os.path.join(stage, active_name), {})
            had_global_marker = os.path.isfile(
                os.path.join(stage, "_validation_complete")
            )
            if had_global_marker:
                os.unlink(os.path.join(stage, "_validation_complete"))
            if channel is not None:
                marker = os.path.join(stage, f"_validation_complete_ch{channel}")
                if os.path.isfile(marker):
                    os.unlink(marker)
            # A channel-neutral TSV must also be harmless to older readers.
            legacy_path = os.path.join(stage, "roi_verification_results.tsv")
            if os.path.isfile(legacy_path) and (
                    active_name == "roi_verification_results.tsv"
                    or had_global_marker):
                _write_rows(legacy_path, {})
            _write_rows(os.path.join(stage, effective_name),
                        _effective_rows(record, stage, present))
            _atomic_json(os.path.join(stage,
                                      os.path.basename(sidecar_path(sample_dir, channel))),
                         record)

            moved_old = False
            installed_stage = False
            try:
                if os.path.isdir(val_dir):
                    os.replace(val_dir, snapshot)
                    moved_old = True
                os.replace(stage, val_dir)
                installed_stage = True
                os.replace(temp_mask, mask_path)
            except Exception:
                if installed_stage and os.path.isdir(val_dir):
                    shutil.rmtree(val_dir)
                if moved_old and os.path.isdir(snapshot):
                    os.replace(snapshot, val_dir)
                raise
        finally:
            if os.path.isdir(stage):
                shutil.rmtree(stage)
        return get_change_state(sample_dir, channel, mask_path)
    finally:
        if os.path.exists(temp_mask):
            os.unlink(temp_mask)


def begin_review(project_folder: str, channel: int | None,
                 mask_path: str) -> dict:
    """Import an older app's decisions or review an externally changed mask."""
    state = get_change_state(project_folder, channel, mask_path)
    if state["status"] != "unknown":
        return state
    adopted = _adopt_external_review(project_folder, channel, mask_path)
    if adopted is not None:
        return adopted
    # Reuse the same safe directory swap as a normal edit: older applications
    # must no longer see a complete marker or a full active TSV.
    current_mask = tifffile.imread(mask_path)
    return save_edited_mask(mask_path, channel, current_mask, current_mask)


def _adopt_external_review(project_folder: str, channel: int | None,
                           mask_path: str) -> dict | None:
    """Preserve valid decisions written by an older application while pending.

    Returns None when the mask itself changed, since the old decisions can no
    longer be matched safely. Conflicting or malformed external decisions are
    left untouched and reported to the user for inspection.
    """
    try:
        record = _read_record(project_folder, channel)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if os.path.basename(mask_path) != record["mask_file"] \
            or _hash_file(mask_path) != record["mask_sha256"]:
        return None
    val_dir = os.path.join(project_folder, VALIDATION_FOLDER)
    if not _external_review_present(val_dir, channel, record):
        return None

    present = {int(v) for v in record["present_roi_ids"]}
    external = {}
    paths_with_data = [path for path in _external_review_paths(
        val_dir, channel, record) if _tsv_has_body(path)]
    legacy_path = os.path.join(val_dir, "roi_verification_results.tsv")
    if legacy_path in paths_with_data and channel is not None:
        if len(paths_with_data) > 1 or _multiple_predicted_channels(project_folder):
            raise RuntimeError(
                "A channel-neutral validation file was edited while several "
                "channels are available. Its channel cannot be identified "
                f"safely; the file has been preserved: {legacy_path}"
            )
    for path in paths_with_data:
        if not _tsv_has_body(path):
            continue
        try:
            rows = _read_external_rows_strict(path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Validation file has unreadable decisions at {path}: {exc}. "
                "The original file has been preserved."
            ) from exc
        unknown_ids = set(rows) - present
        if unknown_ids:
            raise RuntimeError(
                f"Validation file contains cell IDs absent from this mask: "
                f"{sorted(unknown_ids)}. The file has been preserved: {path}"
            )
        for roi_id, row in rows.items():
            if roi_id in external and external[roi_id] != row:
                raise RuntimeError(
                    f"Two validation files disagree about cell {roi_id}. "
                    "Both files have been preserved for review."
                )
            external[roi_id] = row

    completion_marker = (f"_validation_complete_ch{channel}"
                         if channel is not None else "_validation_complete")
    has_marker = (os.path.exists(os.path.join(val_dir, completion_marker))
                  or os.path.exists(os.path.join(val_dir, "_validation_complete")))
    if has_marker and set(external) != present:
        raise RuntimeError(
            "An older application marked validation complete, but its file "
            "does not contain every cell. The file has been preserved; "
            "please inspect it before continuing."
        )
    if not external:
        raise RuntimeError(
            "A validation marker was added without readable decisions. "
            "The files have been preserved for inspection."
        )
    interim_path = os.path.join(val_dir, record["interim_tsv_name"])
    interim = _read_rows(interim_path)
    conflicts = {roi_id for roi_id, row in external.items()
                 if roi_id in interim and interim[roi_id] != row}

    history = os.path.join(project_folder, ".validation_history")
    os.makedirs(history, exist_ok=True)
    snapshot = os.path.join(history, uuid.uuid4().hex)
    stage = tempfile.mkdtemp(prefix=".validation_stage_", dir=project_folder)
    try:
        shutil.copytree(val_dir, stage, dirs_exist_ok=True)
        # A conflicting decision is shown to the user again in the focused
        # validator. Both originals remain in the snapshot; neither wins by
        # accident simply because it was written last.
        for roi_id in conflicts:
            interim.pop(roi_id, None)
        interim.update({roi_id: row for roi_id, row in external.items()
                        if roi_id not in conflicts})
        _write_rows(os.path.join(stage, record["interim_tsv_name"]), interim)
        record["pending_roi_ids"] = sorted(
            ({int(v) for v in record["pending_roi_ids"]} - set(external))
            | conflicts
        )
        record["finalized"] = False
        for path in _external_review_paths(stage, channel, record):
            if os.path.isfile(path):
                _write_rows(path, {})
        for marker in (completion_marker, "_validation_complete"):
            path = os.path.join(stage, marker)
            if os.path.isfile(path):
                os.unlink(path)
        _write_rows(os.path.join(stage, record["effective_tsv_name"]),
                    _effective_rows(record, stage, present))
        _atomic_json(os.path.join(stage,
                                  os.path.basename(sidecar_path(project_folder, channel))),
                     record)
        os.replace(val_dir, snapshot)
        try:
            os.replace(stage, val_dir)
        except Exception:
            os.replace(snapshot, val_dir)
            raise
    finally:
        if os.path.isdir(stage):
            shutil.rmtree(stage)
    if not record["pending_roi_ids"]:
        _finalize(project_folder, channel, mask_path, record)
    return get_change_state(project_folder, channel, mask_path)


def _finalize(project_folder: str, channel: int | None,
              mask_path: str, record: dict) -> None:
    val_dir = os.path.join(project_folder, VALIDATION_FOLDER)
    state = get_change_state(project_folder, channel, mask_path,
                             verify_hash=True)
    if state["status"] != "pending":
        raise RuntimeError(
            (state["reason"] or "Validation changed during this session.")
            + " Reopen the review window."
        )
    present = {int(v) for v in record["present_roi_ids"]}
    complete = _effective_rows(record, val_dir, present)
    missing = present - set(complete)
    if missing:
        record["pending_roi_ids"] = sorted(missing)
        _atomic_json(sidecar_path(project_folder, channel), record)
        raise RuntimeError("Some cells still lack decisions; validation remains open")
    active_name = record["active_tsv_name"]
    legacy_name = "roi_verification_results.tsv"
    global_tsv_is_ambiguous = (
        active_name == legacy_name
        and channel is not None
        and _multiple_predicted_channels(project_folder)
    )
    # A channel-neutral TSV cannot safely describe an edited channel when
    # other predicted channels are present. Keep it empty for old readers and
    # publish the verified decisions in the channel-specific TSV instead.
    _write_rows(os.path.join(val_dir, active_name),
                {} if global_tsv_is_ambiguous else complete)
    if channel is not None:
        # Newer readers prefer a channel-specific TSV and marker.
        _write_rows(os.path.join(val_dir,
                                 f"roi_verification_results_ch{channel}.tsv"), complete)
    if active_name == legacy_name and not global_tsv_is_ambiguous:
        _write_rows(os.path.join(val_dir, legacy_name), complete)
    marker = (f"_validation_complete_ch{channel}" if channel is not None
              else "_validation_complete")
    Path(val_dir, marker).write_text("done")
    if (record.get("legacy_global_marker_was_present")
            and active_name == legacy_name
            and not global_tsv_is_ambiguous):
        # The original channel-neutral validation becomes current again only
        # when this is the sole tracked channel for the sample.
        others = [name for name in os.listdir(val_dir)
                  if name.startswith("validation_changes_")
                  and name.endswith(".json")
                  and name != os.path.basename(sidecar_path(project_folder, channel))]
        if not others:
            Path(val_dir, "_validation_complete").write_text("done")
    record["finalized"] = True
    _atomic_json(sidecar_path(project_folder, channel), record)


def finalize_if_ready(project_folder: str, channel: int | None,
                      mask_path: str) -> dict:
    """Finish a review interrupted after its last decision was saved."""
    state = get_change_state(project_folder, channel, mask_path)
    if state["status"] == "pending" and not state["pending_ids"]:
        record = _read_record(project_folder, channel)
        _finalize(project_folder, channel, mask_path, record)
        return get_change_state(project_folder, channel, mask_path)
    return state


def get_review_decision(project_folder: str, channel: int | None,
                        roi_id: int) -> dict | None:
    """Return a decision saved during the current focused review, if any."""
    try:
        record = _read_record(project_folder, channel)
        path = os.path.join(project_folder, VALIDATION_FOLDER,
                            record["interim_tsv_name"])
        return _read_rows(path).get(int(roi_id))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save_review_decision(project_folder: str, channel: int | None,
                         mask_path: str, roi_id: int,
                         decision: str, note: str) -> dict:
    """Save one focused decision without exposing partial rows to old apps."""
    state = get_change_state(project_folder, channel, mask_path,
                             verify_hash=False)
    if state["status"] != "pending":
        raise RuntimeError(
            (state["reason"] or "Validation changed during this session.")
            + " Reopen the review window."
        )
    record = _read_record(project_folder, channel)
    val_dir = os.path.join(project_folder, VALIDATION_FOLDER)
    interim_path = os.path.join(val_dir, record["interim_tsv_name"])
    interim = _read_rows(interim_path)
    roi_id = int(roi_id)
    present = {int(v) for v in record["present_roi_ids"]}
    if roi_id not in present:
        raise ValueError("The selected cell no longer exists in the mask")
    interim[roi_id] = {"ROI_ID": roi_id, "Decision": decision, "Note": note}
    _write_rows(interim_path, interim)
    record["pending_roi_ids"] = [int(v) for v in record["pending_roi_ids"]
                                 if int(v) != roi_id]
    _write_rows(os.path.join(val_dir, record["effective_tsv_name"]),
                _effective_rows(record, val_dir, present))
    _atomic_json(sidecar_path(project_folder, channel), record)
    if not record["pending_roi_ids"]:
        _finalize(project_folder, channel, mask_path, record)
    return get_change_state(project_folder, channel, mask_path,
                            verify_hash=False)
