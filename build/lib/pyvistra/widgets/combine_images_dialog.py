import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from pyvistra.ui.manager import manager

from .output_selector import ImageOutputSelector

# Axis name -> (index into the 5D (T, Z, C, Y, X) shape, human label)
_AXES = [
    ("t", "Time (T)", 0),
    ("z", "Z-stack (Z)", 1),
    ("c", "Channel (C)", 2),
]


def _combine_metadata(meta_a, meta_b, axis):
    """Merge per-axis metadata for a concatenation of *a* then *b*.

    Only fields whose length is tied to the concatenated axis need
    merging -- everything else is inherited from *a* as-is. Renderers
    already tolerate a "channels" list shorter than C (bounds-checked
    per-channel lookup), so a partial merge here is safe.
    """
    out_meta = dict(meta_a or {})

    if axis == "c":
        channels_a = list((meta_a or {}).get("channels") or [])
        channels_b = list((meta_b or {}).get("channels") or [])
        if channels_a or channels_b:
            out_meta["channels"] = [dict(c) for c in channels_a] + [
                dict(c) for c in channels_b
            ]
    elif axis == "t":
        for key in ("timestamps", "timestamp_seconds"):
            vals_a = (meta_a or {}).get(key)
            vals_b = (meta_b or {}).get(key)
            if isinstance(vals_a, (list, tuple)) and isinstance(vals_b, (list, tuple)):
                out_meta[key] = list(vals_a) + list(vals_b)
            else:
                out_meta.pop(key, None)

    return out_meta


class CombineImagesDialog(QDialog):
    """Stack/concatenate this window's image with another open window's,
    along T, Z, or C.

    The two images must agree on every axis except the one being
    concatenated (Y/X always included).
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Combine Images")
        self.setWindowFlags(Qt.Tool)
        self.resize(380, 260)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(self._update_preview)
        form.addRow("Other window:", self.window_combo)

        self.order_combo = QComboBox()
        self.order_combo.addItem("This window first", False)
        self.order_combo.addItem("Other window first", True)
        self.order_combo.currentIndexChanged.connect(self._update_preview)
        form.addRow("Order:", self.order_combo)

        self.axis_combo = QComboBox()
        for key, label, _idx in _AXES:
            self.axis_combo.addItem(label, key)
        self.axis_combo.currentIndexChanged.connect(self._update_preview)
        form.addRow("Concatenate along:", self.axis_combo)

        main_layout.addLayout(form)

        self.preview_label = QLabel("")
        self.preview_label.setStyleSheet("color: #888;")
        main_layout.addWidget(self.preview_label)

        self.output_selector = ImageOutputSelector(
            default_title="Combined",
            formats=[".tif", ".ims"],
        )
        main_layout.addWidget(self.output_selector)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply)
        buttons.addWidget(self.apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        buttons.addStretch()
        main_layout.addLayout(buttons)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")
        main_layout.addWidget(self.status_label)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_window_combo()

    def _refresh_window_combo(self):
        """List every open window except this one as a candidate operand."""
        current = self.window_combo.currentData()
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        for _wid, win in sorted(manager.get_all().items()):
            if win is self.viewer:
                continue
            self.window_combo.addItem(win.windowTitle(), win)
        idx = self.window_combo.findData(current)
        self.window_combo.setCurrentIndex(max(idx, 0))
        self.window_combo.blockSignals(False)
        self._update_preview()

    def _selected_axis(self):
        return self.axis_combo.currentData()

    def _ordered_shapes(self):
        """Return (shape_first, shape_second, other_window) honoring the
        Order combo, or None if no valid other window is selected."""
        other = self.window_combo.currentData()
        if other is None:
            return None
        a_shape = tuple(self.viewer.img_data.shape)
        b_shape = tuple(other.img_data.shape)
        if self.order_combo.currentData():
            return b_shape, a_shape, other
        return a_shape, b_shape, other

    def _validate(self):
        """Return (axis, axis_idx, other_window, result_shape) or raise
        ValueError with a user-facing message."""
        ordered = self._ordered_shapes()
        if ordered is None:
            raise ValueError("pick a second window")
        shape_first, shape_second, other = ordered

        axis = self._selected_axis()
        axis_idx = dict((k, i) for k, _label, i in _AXES)[axis]

        for i in range(5):
            if i == axis_idx:
                continue
            if shape_first[i] != shape_second[i]:
                names = ("T", "Z", "C", "Y", "X")
                raise ValueError(
                    f"{names[i]} must match ({shape_first[i]} vs {shape_second[i]})"
                )

        result_shape = list(shape_first)
        result_shape[axis_idx] = shape_first[axis_idx] + shape_second[axis_idx]
        return axis, axis_idx, other, tuple(result_shape)

    def _update_preview(self):
        try:
            _axis, _axis_idx, _other, result_shape = self._validate()
        except ValueError as exc:
            self.preview_label.setText(str(exc))
            self.preview_label.setStyleSheet("color: #888;")
            return
        self.preview_label.setText(
            f"Result shape (T, Z, C, Y, X): {result_shape}"
        )
        self.preview_label.setStyleSheet("color: #888;")

    def _set_error(self, message):
        self.status_label.setText(f"Error: {message}")
        self.status_label.setStyleSheet("color: #F44;")

    def _apply(self):
        try:
            axis, axis_idx, other, result_shape = self._validate()
        except ValueError as exc:
            self._set_error(str(exc))
            return

        if self.order_combo.currentData():
            first_win, second_win = other, self.viewer
        else:
            first_win, second_win = self.viewer, other

        a = np.asarray(first_win.img_data[:])
        b = np.asarray(second_win.img_data[:])
        out_dtype = np.result_type(a.dtype, b.dtype)
        result = np.concatenate(
            [a.astype(out_dtype, copy=False), b.astype(out_dtype, copy=False)],
            axis=axis_idx,
        )

        out_meta = _combine_metadata(first_win.meta, second_win.meta, axis)
        out_meta["shape"] = result_shape
        base = str(out_meta.get("filename", out_meta.get("name", "Image")))
        out_meta["filename"] = f"{base}_combined"
        out_meta["name"] = out_meta["filename"]

        from pyvistra.io import ImageBuffer

        buffer = ImageBuffer(shape=result_shape, dtype=out_dtype, metadata=out_meta)
        buffer[:] = result

        sent = self.output_selector.send(buffer, out_meta)
        if sent is not None:
            self.status_label.setText("Done")
            self.status_label.setStyleSheet("color: #4A4;")
        else:
            self.status_label.setText("Computed (output cancelled)")
            self.status_label.setStyleSheet("color: #888;")
