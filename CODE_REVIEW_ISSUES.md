# Code review issue log

Reviewed: 28 October 2026. This log records **open findings outside the current changes**. The findings identify workflows in which the code may produce an incorrect result, lose work, or fail; they do not establish that any existing project has been affected. Severity describes the possible impact, not the frequency. Existing project files must remain readable when any fix is made.

## Status of the five discussed items

- **Items 2 and 5 — completed:** Missing analysis channels now produce explicit skipped rows with blank measurements. Other sample failures produce error rows, counts, and a traceback report. See `cellmatchr/analysis.py`, `cellmatchr/launcher.py`, and `tests/test_analysis_reporting.py`.
- **Items 1, 3, and 4 — addressed separately:** New segmentation runs use distinct folders and analysis asks which run to use. Existing analysis results require a simple Yes/No replacement choice and receive a recoverable backup. Mask edits queue only affected cells for focused review. The old `Segmentation/` layout and existing project JSON, validation TSVs, and completion markers remain readable; these items are not duplicated in the open list below.
- **Earlier mask-save finding — addressed with item 4:** The ROI editor now writes a temporary TIFF and replaces the prediction mask only after a safe validation snapshot is prepared. The old in-place mask-write risk is removed from the open list below.
- **Leica LIF routing and direct-read name collisions — addressed:** Confocal and stitched slide-scanner scenes are routed separately, including when input files are one folder below the selected root. Direct-read segmentation assigns stable distinct names to colliding samples. The selected converter output also supplies its matching ROI mask folder to analysis. The remaining converter-specific naming risk is recorded as R02.

## Open findings

### R01 — Run Pipeline may treat partial segmentation as complete

**Severity:** High — a later analysis may run on an incomplete set of samples or channels.

**Evidence and impact:** `cellmatchr/launcher.py:4573-4582` treats the existence of any prediction TIFF in any segmentation variant as segmentation completion for its status indicator. Run Pipeline now asks whether to use existing masks or segment the current input, but choosing an existing run can still analyze partial masks. `cellmatchr/segmentation.py:1932-1935` skips a requested channel when it is absent, while `cellmatchr/segmentation.py:2032-2035` still marks the sample complete. A sample exception is caught and processing continues at `cellmatchr/segmentation.py:2025-2029`. The progress writer records whether the full run finished at `cellmatchr/segmentation.py:2042-2047`, but the launcher does not use that flag for its status indicator.

**Recommended fix:** For newly produced runs, use the progress record and verify that every planned sample has the requested channel masks before Run Pipeline skips segmentation. Make partial-run status visible in the UI and allow resume. For older runs without a progress record, retain the ability to select and analyze them; avoid interpreting a single mask as proof of full completion. Test missing-channel, exception, interrupted, and legacy-no-progress cases.

**Compatibility risk:** A strict new completeness rule could incorrectly block older project folders, especially those generated before progress files existed. Use the richer check only when enough run metadata is available and provide a clear legacy fallback.

### R02 — Converter outputs can collide for same-named slide scans

**Severity:** High when multiple same-named slide scans are exported into one converter output folder — one slide's ROI export or mask could overwrite another's.

**Evidence and impact:** The slide converter constructs flat ROI export and mask names from the source basename, slice, and ROI numbers in `cellmatchr/vsi_converter.py`. Two same-named slide scans selected into one converter output folder can produce identical paths. Direct-read segmentation now disambiguates duplicate sample names before writing; the converter still needs its own source-aware output names.

**Recommended fix:** Detect a second slide with the same proposed output prefix before its first export. Give it a stable source-derived prefix, display the chosen name, and record a source-to-export mapping. Preserve existing ROI-to-sample matching in analysis.

**Compatibility risk:** Renaming already exported samples would break references to masks, validation, and analysis. Preserve old names; apply disambiguation only to new conflicting outputs.

### R03 — Slide-scanner ROI masks may be misaligned with exported crops

**Severity:** High when slide-scanner ROI masks are used — counts near ROI borders may change.

**Evidence and impact:** The converter calculates full-resolution image crop bounds from the polygon's vertex bounding box in `cellmatchr/vsi_converter.py:1217-1241`. Analysis instead crops an overview-space mask to its nonzero pixel bounds and resizes it to the target image at `cellmatchr/analysis.py:254-275`. Polygon vertex bounds and rasterized nonzero bounds can differ by a pixel or more, so the two transformations need not align exactly.

**Recommended fix:** Save the exact converter crop coordinates and overview-to-full-resolution transform alongside future exports, then use those coordinates when applying the ROI mask. Compare representative old and new ROI counts, especially cells touching boundaries.

**Compatibility risk:** Historical exports may not contain coordinates. Keep the current fallback for them and label its alignment as approximate; do not reinterpret old masks silently.

### R04 — Some axis orders can project the wrong dimension

**Severity:** High for uncommon microscope axis layouts — the resulting 2D channel may have the wrong shape or content.

**Evidence and impact:** `_split_array_by_channel` knows the named `Z` axis (`cellmatchr/segmentation.py:1117-1136`), but after selecting a channel it repeatedly projects axis 0 (`cellmatchr/segmentation.py:1149-1155`). For a `YXZC` array, that projects `Y`, not `Z`. A synthetic `YXZC` example reproduced the wrong output shape during the review.

**Recommended fix:** After dropping the channel axis, track the remaining axis names; project the actual `Z` axis and then arrange the result as `YX`. Add synthetic tests for `CZYX`, `YXZC`, `ZYXC`, single-channel, and ordinary 2D inputs.

**Compatibility risk:** The fix changes future outputs for affected axis orders. Do not rewrite existing split TIFFs or masks; identify the source axes in run metadata so historical results can be interpreted.

### R05 — Cells crossing tile cores can receive separate labels

**Severity:** Medium–high scientific correctness risk; frequency depends on cell size and tile placement.

**Evidence and impact:** `cellmatchr/segmentation.py:949-982` cuts away each tile's halo to keep a core; `cellmatchr/segmentation.py:996-1011` offsets label IDs and writes each core independently. There is no border reconciliation between adjacent cores. One cell crossing a seam may be split, double-counted, or have a fragment removed by size filtering.

**Recommended fix:** Compare touching objects across neighboring tile boundaries using overlap or image continuity and merge only well-supported matches. Validate with synthetic seam-spanning cells and representative images before changing production counts.

**Compatibility risk:** Merge rules alter scientific measurements. Keep old mask files untouched and version any future stitching behavior in run metadata so results remain comparable.

### R06 — Per-channel filter overrides may not affect mask generation as intended

**Severity:** Medium–high — channel-specific settings can give unexpected object counts.

**Evidence and impact:** `generate_kwargs["min_size"]` is captured once from the global minimum at `cellmatchr/segmentation.py:1881-1883`. Per-channel `MIN_OBJECT_SIZE` changes later at `cellmatchr/segmentation.py:1943-1953`, but the generation call still receives the earlier `generate_kwargs` at `cellmatchr/segmentation.py:1973-1984`. A lower per-channel minimum cannot recover objects already excluded during generation. Global filter values are restored only after successful channel processing (`cellmatchr/segmentation.py:2021-2024`); an exception can leave overrides active for the next sample.

**Recommended fix:** Build generation arguments inside each channel iteration after applying that channel's settings. Restore all global settings in a `finally` block. Test both a lower per-channel minimum and a failed channel followed by a different sample.

**Compatibility risk:** Future masks may differ from existing masks with the same settings. Retain historical outputs and record effective filter settings per run.

### R07 — Single-channel validation layout can fail

**Severity:** Medium — users of one-channel data may be unable to open validation.

**Evidence and impact:** `cellmatchr/validator.py:445-467` divides a GridSpec into merged and context views. With one channel, the context slice is empty (`gs[1:1, 3]`), which raised an `IndexError` in a synthetic review check.

**Recommended fix:** Give one-channel validation a dedicated layout with one merged/context area, while keeping the current multi-channel layout. Add a one-channel UI construction test.

**Compatibility risk:** Low; the TSV and mask formats need no change.

### R08 — Custom validation decision words may be ignored by analysis

**Severity:** Medium — decision summaries can show zero despite recorded choices.

**Evidence and impact:** The validator allows renaming decision words (`cellmatchr/validator.py:627-641`), but analysis counts only literal `YES`, `NO`, and `MAYBE` (`cellmatchr/analysis.py:425-432`). An older TSV containing a custom word will not appear in those three count sets. `ARTIFACT` still has its own literal handling.

**Recommended fix:** Store a stable category identifier separately from the displayed word in future TSVs. For existing TSVs, continue recognizing the literal legacy values and use saved button configuration to map custom words where the mapping is available; ask for an explicit mapping if ambiguous.

**Compatibility risk:** Never reinterpret an unknown custom word as `YES` or `NO` by guesswork. Existing TSVs must remain readable even when a mapping cannot be recovered.

### R09 — Ambiguous ROI mask names can select the wrong mask

**Severity:** Medium–high when multiple similarly named ROI files are present.

**Evidence and impact:** `cellmatchr/analysis.py:207-248` first tries normalized exact names, then substring matches. If several files match either way, it warns but selects the first sorted file. That can associate a sample with the wrong ROI mask and change ROI-specific counts.

**Recommended fix:** Require an explicit selection when several masks match, or use an exact source-to-ROI mapping saved during export. Include the chosen mask path in analysis metadata.

**Compatibility risk:** A blanket rejection of ambiguity could block previously generated projects. Keep unambiguous legacy matching and provide a clear selection path for ambiguous folders.

### R10 — Legacy channel-neutral validation may be applied to either channel

**Severity:** Medium–high when an old project validates only one channel but analyzes another.

**Evidence and impact:** `cellmatchr/validation_state.py:171-179` falls back from a channel-specific TSV to `roi_verification_results.tsv`, and `cellmatchr/analysis.py:439-450` treats `_validation_complete` as complete for a requested channel. These old names do not encode which channel they describe.

**Recommended fix:** Preserve the fallback for older projects, but when more than one candidate channel is analyzed, ask which channel the legacy validation belongs to once and save that association as new metadata. Display a warning until it is assigned.

**Compatibility risk:** Removing the fallback would break old projects. Automatically assigning the old TSV to both channels could filter the wrong objects.

### R11 — Polygon deletion can shift ROI names and colors

**Severity:** Medium — exported ROI names may no longer describe the intended polygon.

**Evidence and impact:** ROI polygons and the separate name/color arrays are aligned by index. After shape changes, `_sync_roi_ui` truncates those arrays to the new count (`cellmatchr/vsi_converter.py:1140-1154`). Deleting a polygon other than the final one can therefore transfer a later polygon's name/color to the preceding index.

**Recommended fix:** Attach a stable ID and name/color to each polygon or update the arrays using the actual removed index from the shape event. Test deleting the first, middle, and last polygon.

**Compatibility risk:** Existing exports keep their filenames and recorded geometry. Apply the correction to editing sessions going forward; do not rename old files automatically.

### R12 — Closing editors may leave unsaved work

**Severity:** Medium — user-entered paint or polygons can be lost.

**Evidence and impact:** The ROI editor's Finish button calls `viewer.close` directly (`cellmatchr/roi_editor.py:581-588`); its autosave path is used when navigating backward (`cellmatchr/roi_editor.py:713-740`), so unfinished paint on the current image need not be saved on Finish. The converter relies on explicit save/export actions (`cellmatchr/vsi_converter.py:1404-1418`); closing its window may discard in-memory polygons.

**Recommended fix:** For both windows, intercept Finish/close when edits are pending and offer three plain choices: **Save and close**, **Discard edits**, **Cancel**. If no edits are pending, close normally. Test each close path.

**Compatibility risk:** Low for file formats. The main risk is accidentally auto-saving a partial edit, so the choice must remain explicit.

### R13 — Project settings can be overwritten or left out of sync

**Severity:** Medium — a reopened project may use different settings from the last session.

**Evidence and impact:** New Project creates an existing folder with `exist_ok=True` and writes project settings (`cellmatchr/launcher.py:1772-1786`), allowing an old `cellmatchr_project.json` to be overwritten. Save & Close writes only the global configuration (`cellmatchr/launcher.py:829-834`), not the project copy.

**Recommended fix:** If New Project points to an existing project JSON, offer **Open existing project** or **Choose a different folder** before writing. Save the project copy alongside the global configuration when an active valid project is present, using an atomic write.

**Compatibility risk:** Do not migrate or overwrite old project settings automatically. The UI should show which project is open before saving.

### R14 — Parameter Wizard can reset thresholds for an absent channel

**Severity:** Medium — prior channel filters may be silently disabled.

**Evidence and impact:** If the first sample lacks a selected channel, the wizard records a result whose picks are all `None` (`cellmatchr/parameter_wizard.py:204-210`). At finish, it visits every recorded channel and converts those `None` picks into disabled values, replacing the saved filter entry (`cellmatchr/parameter_wizard.py:463-488`).

**Recommended fix:** Distinguish **not present** from **opened and intentionally unset**; leave the existing filter unchanged for absent channels. Show a short completion summary of channels changed versus kept.

**Compatibility risk:** Low for project readability. The behavior of future wizard saves changes, so test existing filter configurations.

### R15 — Declared Python and backend dependencies do not match imports

**Severity:** Medium — a supported installation may fail before the app starts or before choosing a backend.

**Evidence and impact:** `pyproject.toml:10` declares Python 3.9 support, while the code uses unquoted `X | None` annotations that require Python 3.10 or newer. The `seg` extra supplies Micro-SAM (`pyproject.toml:32-35`), but `cellmatchr/segmentation.py:12-13` imports it unconditionally before backend selection, including a Cellpose-only run.

**Recommended fix:** Align `requires-python` with the actual minimum interpreter and import backend-specific packages only when that backend is selected. Add installation smoke tests for the documented environment and each backend.

**Compatibility risk:** Changing metadata may exclude old Python 3.9 installations that never worked with current annotations; existing project data is unaffected. Lazy imports should retain normal Micro-SAM behavior.

### R16 — Large images can exceed memory during overview and IoU calculations

**Severity:** Medium; impact rises with large Leica slides and dense masks.

**Evidence and impact:** The Leica wrapper loads and caches all channels and Z projections for a full scene before producing an overview (`cellmatchr/vsi_converter.py:468-492`). IoU matching converts every overlapping pixel into a pandas row before grouping (`cellmatchr/analysis.py:456-487`). Both approaches can multiply peak memory use on large images.

**Recommended fix:** Read and resize Leica overview data in blocks, and count label-pair intersections with NumPy or chunked aggregation rather than a per-pixel DataFrame. Compare outputs numerically with the current implementation on small and representative large fixtures.

**Compatibility risk:** Performance changes can change tie handling or rounding if reimplemented carelessly. Keep output schemas and verify exact counts and IoU values.

### R17 — Preview image objects are created on a worker thread

**Severity:** Low–medium — intermittent Tk errors or missing previews may occur.

**Evidence and impact:** `cellmatchr/launcher.py:2966` constructs `ImageTk.PhotoImage` inside `_load_channel_previews`, and `cellmatchr/launcher.py:2982` starts that function on a background thread. Tk image creation should occur on the UI thread.

**Recommended fix:** Decode and scale raw image arrays in the worker, then create `PhotoImage` objects in the callback scheduled with `root.after` on the Tk thread. Test repeated preview loading and window closure during loading.

**Compatibility risk:** Low; project files are unaffected.

### R18 — Converter ROI names can become unsafe or confusing paths

**Severity:** Low–medium — some ROI names may fail to save or create unexpected paths.

**Evidence and impact:** The converter inserts ROI names directly into output suffixes (`cellmatchr/vsi_converter.py:1294-1295`, `cellmatchr/vsi_converter.py:1358-1359`). Separators, reserved characters, or duplicate names are not checked at those points. The launcher now checks path separators in newly entered run suffixes, but converter ROI names still need validation.

**Recommended fix:** Validate names at entry with a short inline message, show the final output name before saving, and reject separators and path traversal. Keep the original display name in metadata if a safe filename differs.

**Compatibility risk:** Restrict new names only; existing folders and files must still open with their historical names.

### R20 — Regression coverage is narrow

**Severity:** Medium as a maintenance risk.

**Evidence and impact:** The repository now has targeted analysis, compatibility, validation-change, and LIF-routing checks, but no tracked regression tests for segmentation axes, tile seams, single-channel validation, or converter ROI naming/alignment. Changes in those paths could alter outputs without a quick signal.

**Recommended fix:** Add small synthetic fixtures for each confirmed failure and representative historical project layouts beyond the current synthetic compatibility checks. Run those tests in CI without requiring GPU models or a display where possible.

**Compatibility risk:** None for project files. Avoid tests that write into real project folders.

### R21 — Resume cannot detect a same-size source replacement

**Severity:** Medium when an input image is replaced in place — a continued run could combine masks from different source contents.

**Evidence and impact:** New run progress records compare relative source paths, byte sizes, Leica scene indices, and planned sample names before Continue. A replacement with the same path, size, and scene layout passes that lightweight check even if image pixels changed. Historical progress records have no source fingerprint and remain subject to the explicit legacy warning.

**Recommended fix:** Offer a stronger source check before Continue, such as a saved content digest or a clear confirmation when a file timestamp changes. Hashing every multi-gigabyte LIF on every launch would be slow; record any digest once and compare only when needed. Starting a new run remains the safe choice after source files change.

**Compatibility risk:** A strict new hash requirement would block historical runs with no digest. Keep the current legacy warning path and apply stronger checks only to runs that recorded them.

## Compatibility rule for future fixes

Keep existing project JSON, TIFF, validation TSV, and completion marker names readable. Add new metadata beside old files rather than changing their schema in place. When a historical file lacks information needed to disambiguate a run, channel, or ROI, ask the user instead of guessing. Never recalculate or overwrite an existing scientific result as part of opening a project.
