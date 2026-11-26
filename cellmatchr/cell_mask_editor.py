"""Napari brush editor for a single crop handed off by Individual Object Review."""

import argparse

import numpy as np


def merge_cell_crop(original, roi_id, edited):
    """Replace only the selected object's pixels and protect existing neighbors."""
    if edited.shape != original.shape or edited.dtype != np.bool_:
        raise ValueError("The edited mask does not match the cell crop.")
    painted = edited & ((original == 0) | (original == roi_id))
    if not painted.any():
        raise ValueError("Keep part of the cell mask, or mark the prediction as Artifact.")
    updated = original.copy()
    updated[updated == roi_id] = 0
    updated[painted] = roi_id
    return updated


def _open_editor(source, result, roi_id):
    import napari
    from qtpy.QtWidgets import (
        QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton,
        QSpinBox, QVBoxLayout, QWidget,
    )

    with np.load(source, allow_pickle=False) as data:
        images, names = data["images"], data["names"]
        mask, blocked = data["mask"], data["blocked"]

    viewer = napari.Viewer(title=f"Edit Mask — ROI {roi_id}", show=False)
    image = viewer.add_image(images[0], rgb=True, name="Cell crop")
    labels = viewer.add_labels(mask.astype(np.uint8), name="Cell mask", opacity=0.45)
    labels.selected_label = 1
    labels.brush_size = 6
    labels.mode = "paint"

    panel = QWidget()
    layout = QVBoxLayout(panel)
    tools = QHBoxLayout()
    layout.addLayout(tools)
    modes = QButtonGroup(panel)
    for caption in ("Paint", "Erase"):
        button = QPushButton(caption)
        button.setCheckable(True)
        button.setChecked(caption == "Paint")
        button.clicked.connect(lambda _checked, mode=caption.lower(): setattr(labels, "mode", mode))
        modes.addButton(button)
        tools.addWidget(button)
    size_label = QLabel("Brush size")
    size = QSpinBox()
    size.setRange(1, 100)
    size.setValue(labels.brush_size)
    size.valueChanged.connect(lambda value: setattr(labels, "brush_size", value))
    size_label.setBuddy(size)
    tools.addWidget(size_label)
    tools.addWidget(size)
    if len(images) > 1:
        view = QComboBox()
        view.addItems(names.tolist())
        view.setAccessibleName("Image view")
        view.currentIndexChanged.connect(lambda index: setattr(image, "data", images[index]))
        tools.addWidget(view, 1)
    tools.addStretch()
    show_mask = QCheckBox("Show mask")
    show_mask.setChecked(True)
    show_mask.toggled.connect(lambda shown: setattr(labels, "opacity", 0.45 if shown else 0))
    tools.addWidget(show_mask)

    actions = QHBoxLayout()
    layout.addLayout(actions)
    status = QLabel()
    status.setWordWrap(True)
    actions.addWidget(status, 1)

    def save():
        edited = (labels.data > 0) & ~blocked
        if not edited.any():
            status.setText("Keep part of the mask, or mark the cell as Artifact in the reviewer.")
            return
        try:
            np.save(result, edited, allow_pickle=False)
        except OSError as error:
            status.setText(f"Could not save the edited crop: {error}")
            return
        viewer.close()

    cancel = QPushButton("Cancel")
    cancel.clicked.connect(viewer.close)
    actions.addWidget(cancel)
    save_button = QPushButton("Save Mask")
    save_button.clicked.connect(save)
    actions.addWidget(save_button)

    viewer.window.add_dock_widget(panel, area="bottom", name="Mask editing")
    viewer.window._qt_viewer.dockLayerList.hide()
    viewer.window._qt_viewer.dockLayerControls.hide()
    viewer.window.show()
    viewer.window._qt_window.resize(690, 610)
    viewer.reset_view()
    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("result")
    parser.add_argument("roi_id", type=int)
    args = parser.parse_args()
    _open_editor(args.source, args.result, args.roi_id)
