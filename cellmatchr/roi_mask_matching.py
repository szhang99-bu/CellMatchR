"""Resolve ROI mask names consistently in the launcher and analysis script."""

import os
import re
import unicodedata

if __package__:
    from .file_paths import filesystem_path
else:
    from file_paths import filesystem_path


def normalize_name_for_matching(name):
    name = unicodedata.normalize("NFC", os.path.basename(name)).casefold()
    # Sample names can contain meaningful dots. Strip only known extensions.
    name = re.sub(r"\.(roi|tif|tiff|czi|lif|nd2|oib|oif|png)$", "", name)
    name = re.sub(r"_predict_original_ch\d+(?:_.*)?$", "", name)
    name = re.sub(r"(?:_mask_overview|_vertices_overview_yx|_overview|_split_ch\d+)$", "", name)
    name = re.sub(r"(^|_)fullres(?=_|$)", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def roi_mask_files(folder):
    if not folder or not os.path.isdir(filesystem_path(folder)):
        return []
    with os.scandir(filesystem_path(folder)) as entries:
        return sorted(entry.path for entry in entries if entry.is_file()
                      and entry.name.lower().endswith((".tif", ".tiff", ".roi")))


def roi_match_candidates(sample, mask_files):
    """Return the automatic match or a list of choices needing confirmation."""
    name = normalize_name_for_matching(sample)
    if not name:
        return None, []
    named = [(path, normalize_name_for_matching(path)) for path in mask_files]
    exact = [path for path, key in named if key == name]
    if exact:
        return (exact[0], []) if len(exact) == 1 else (None, exact)

    # Older manual masks may have a prefix or suffix. Offer token-aligned
    # candidates for review, so e.g. Mouse10 cannot be used for Mouse1.
    def contains(outer, inner):
        return bool(inner and re.search(
            rf"(?<![^\W_]){re.escape(inner)}(?![^\W_])", outer))

    return None, [path for path, key in named if contains(key, name) or contains(name, key)]


def roi_choice_context(run, folder):
    return [os.path.normcase(os.path.realpath(path)) for path in (run, folder)]
