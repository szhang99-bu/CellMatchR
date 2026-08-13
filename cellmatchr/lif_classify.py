"""
Classify Leica .lif acquisitions as "confocal" or "slide_scanner".

Leica writes both instrument types into the *same* .lif container (unlike
Zeiss .czi or Olympus .vsi, which are format-distinct), so the file extension
alone can't tell them apart. This module inspects per-image metadata instead.

Signals, both confirmed against real files
(/Users/szhang24/Desktop/LiF_Optimization/Confocal_63X.lif and
.../Slidescanner_10x rep.lif):

  - dims.m > 1   -> a raw, un-stitched tile-scan mosaic tile. Only slide
                    scans are acquired as tiles.
  - settings == {} (empty dict) -> a synthesized/stitched image (e.g. the
                    "... Merged" image LAS X writes after a tile scan).
                    LAS X only records the full scanner-hardware settings
                    dict (ScanMode, Zoom, ObjectiveName, Pinhole,
                    StagePosX/Y, ...) for a direct single-frame acquisition,
                    never for a derived/merged one — so this catches the
                    slide-scanner "Merged" image too, which has dims.m == 1
                    just like a confocal scene.

Any image that fails both checks — or whose metadata can't be read —
defaults to "confocal" so unusual-but-valid confocal files are never newly
excluded by a metadata read hiccup.
"""

from __future__ import annotations


def classify_lif_image(img) -> str:
    """Classify one readlif LifImage as "confocal" or "slide_scanner"."""
    try:
        if int(img.dims.m) > 1:
            return "slide_scanner"
    except Exception:
        pass
    try:
        settings = img.settings
        if isinstance(settings, dict) and len(settings) == 0:
            return "slide_scanner"
    except Exception:
        pass
    return "confocal"


def classify_lif_file(path: str) -> dict:
    """Classify every image/scene inside a .lif file.

    Returns:
        {
          "file_type": "confocal" | "slide_scanner" | "mixed" | "unknown",
          "scenes": [(index, name, "confocal"|"slide_scanner"), ...],
          "error": <str, only present on failure>,
        }
    """
    try:
        from readlif.reader import LifFile
    except ImportError:
        return {"file_type": "unknown", "scenes": [], "error": "readlif not installed"}

    try:
        lif = LifFile(path)
        imgs = list(lif.get_iter_image())
    except Exception as exc:
        return {"file_type": "unknown", "scenes": [], "error": str(exc)}

    scenes = [(i, im.name, classify_lif_image(im)) for i, im in enumerate(imgs)]
    types = {t for _, _, t in scenes}

    if not types:
        file_type = "unknown"
    elif len(types) == 1:
        file_type = types.pop()
    else:
        file_type = "mixed"

    return {"file_type": file_type, "scenes": scenes}
