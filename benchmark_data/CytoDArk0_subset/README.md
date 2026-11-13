# Expert-reviewed Nissl reference subset

This local folder contains 28 paired 512 × 512 Nissl images and expert-reviewed instance masks from [CytoDArk0](https://zenodo.org/records/13694738). Each `.png` is an image; the `.tiff` with the same stem is its cell-label mask. Label 0 is background and each positive integer is a separate cell. `scores.json` records the per-patch and pooled comparison values described in [NISSL_MODEL_EVALUATION.md](../../NISSL_MODEL_EVALUATION.md).

The selected sources are IDs 1, 28, 54, 55, 58, 62 and 66, with four quadrants each. The image and mask files are kept locally and ignored by Git to avoid adding the dataset to the application repository. The original Zenodo record contains the complete dataset and attribution.

To inspect a pair, open the PNG and matching TIFF in Fiji/ImageJ, napari, or QuPath. The TIFF contains integer labels, not a grayscale stain image.
Two local `preview_*_human_labels.png` files show the human-reviewed cell outlines in yellow over the stain image for a quick visual check.
