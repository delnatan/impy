import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pyvistra import colors as tokens
from pyvistra.ui.manager import manager

from .output_selector import ImageOutputSelector

_OPS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.true_divide,
}

_CONSTANT = "constant"
_WINDOW = "window"


class ImageMathDialog(QDialog):
    """Elementwise image math: constant or window-vs-window, with broadcasting.

    The second operand only has to match the current window's lateral (Y, X)
    dimensions; mismatched T/Z/C broadcast via ordinary numpy rules (e.g. a
    single-channel or single-timepoint window applied across a whole stack).
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Image Math")
        self.setWindowFlags(Qt.Tool)
        self.resize(360, 260)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.op_combo = QComboBox()
        self.op_combo.addItem("Add", "add")
        self.op_combo.addItem("Subtract", "sub")
        self.op_combo.addItem("Multiply", "mul")
        self.op_combo.addItem("Divide", "div")
        form.addRow("Operation:", self.op_combo)

        self.operand_combo = QComboBox()
        self.operand_combo.addItem("Constant", _CONSTANT)
        self.operand_combo.addItem("Another window", _WINDOW)
        self.operand_combo.currentIndexChanged.connect(self._on_operand_changed)
        form.addRow("Operand:", self.operand_combo)

        main_layout.addLayout(form)

        self._operand_pages = QStackedWidget()

        constant_page = QWidget()
        constant_layout = QHBoxLayout(constant_page)
        constant_layout.setContentsMargins(0, 0, 0, 0)
        self.constant_spin = QDoubleSpinBox()
        self.constant_spin.setRange(-1e9, 1e9)
        self.constant_spin.setDecimals(4)
        self.constant_spin.setValue(0.0)
        constant_layout.addWidget(self.constant_spin, 1)
        self._operand_pages.addWidget(constant_page)

        window_page = QWidget()
        window_layout = QHBoxLayout(window_page)
        window_layout.setContentsMargins(0, 0, 0, 0)
        self.window_combo = QComboBox()
        window_layout.addWidget(self.window_combo, 1)
        self._operand_pages.addWidget(window_page)

        main_layout.addWidget(self._operand_pages)

        self.output_selector = ImageOutputSelector(
            default_title="Image Math Result",
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
        self.status_label.setStyleSheet(f"color: {tokens.TEXT_FAINT};")
        main_layout.addWidget(self.status_label)

        self._on_operand_changed(self.operand_combo.currentIndex())

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_window_combo()

    def _on_operand_changed(self, _index):
        is_window = self.operand_combo.currentData() == _WINDOW
        self._operand_pages.setCurrentIndex(1 if is_window else 0)
        if is_window:
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

    def _set_error(self, message):
        self.status_label.setText(f"Error: {message}")
        self.status_label.setStyleSheet(f"color: {tokens.DANGER};")

    def _apply(self):
        a = np.asarray(self.viewer.img_data[:]).astype(np.float32, copy=False)

        if self.operand_combo.currentData() == _CONSTANT:
            b = float(self.constant_spin.value())
            operand_desc = f"constant={b:g}"
        else:
            other = self.window_combo.currentData()
            if other is None:
                self._set_error("pick a second window")
                return
            a_yx = self.viewer.img_data.shape[-2:]
            b_yx = other.img_data.shape[-2:]
            if a_yx != b_yx:
                self._set_error(f"Y/X must match ({a_yx} vs {b_yx})")
                return
            b = np.asarray(other.img_data[:]).astype(np.float32, copy=False)
            operand_desc = f"window '{other.windowTitle()}'"

        op_name = self.op_combo.currentData()
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                result = _OPS[op_name](a, b).astype(np.float32, copy=False)
        except ValueError as exc:
            self._set_error(str(exc))
            return

        from pyvistra.io import ImageBuffer

        out_meta = dict(self.viewer.meta or {})
        out_meta["shape"] = result.shape
        base = str(out_meta.get("filename", out_meta.get("name", "Image")))
        out_meta["filename"] = f"{base}_{op_name}"
        out_meta["name"] = out_meta["filename"]
        out_meta["image_math"] = {"operation": op_name, "operand": operand_desc}

        buffer = ImageBuffer(shape=result.shape, dtype=np.float32, metadata=out_meta)
        buffer[:] = result

        sent = self.output_selector.send(buffer, out_meta)
        if sent is not None:
            self.status_label.setText("Done")
            self.status_label.setStyleSheet(f"color: {tokens.SUCCESS};")
        else:
            self.status_label.setText("Computed (output cancelled)")
            self.status_label.setStyleSheet(f"color: {tokens.TEXT_FAINT};")
