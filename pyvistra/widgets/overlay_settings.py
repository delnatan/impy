from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QColorDialog,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QVBoxLayout,
)


class OverlaySettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Overlay Settings")
        self.resize(430, 520)

        cfg = dict(config or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.chk_scale = QCheckBox()
        self.chk_scale.setChecked(cfg.get("show_scale_bar", True))
        form.addRow("Show Scale Bar", self.chk_scale)

        self.scale_length = QDoubleSpinBox()
        self.scale_length.setRange(0.001, 1e9)
        self.scale_length.setDecimals(4)
        self.scale_length.setValue(float(cfg.get("scale_bar_length", 10.0)))
        form.addRow("Scale Bar Length", self.scale_length)

        self.scale_unit = QComboBox()
        self.scale_unit.addItems(["px", "nm", "um", "mm"])
        self.scale_unit.setCurrentText(str(cfg.get("scale_bar_unit", "um")))
        form.addRow("Scale Bar Unit", self.scale_unit)

        self.scale_anchor = QComboBox()
        self.scale_anchor.addItems(
            ["bottom-left", "bottom-right", "top-left", "top-right"]
        )
        self.scale_anchor.setCurrentText(
            str(cfg.get("scale_anchor", "bottom-left"))
        )
        form.addRow("Scale Anchor", self.scale_anchor)

        self.scale_offset_x = QDoubleSpinBox()
        self.scale_offset_x.setRange(0.0, 1.0)
        self.scale_offset_x.setDecimals(3)
        self.scale_offset_x.setSingleStep(0.01)
        self.scale_offset_x.setValue(float(cfg.get("scale_offset_x", 0.04)))
        form.addRow("Scale Offset X (0-1)", self.scale_offset_x)

        self.scale_offset_y = QDoubleSpinBox()
        self.scale_offset_y.setRange(0.0, 1.0)
        self.scale_offset_y.setDecimals(3)
        self.scale_offset_y.setSingleStep(0.01)
        self.scale_offset_y.setValue(float(cfg.get("scale_offset_y", 0.06)))
        form.addRow("Scale Offset Y (0-1)", self.scale_offset_y)

        self.scale_color_btn = QPushButton()
        self._scale_color = self._tuple_to_qcolor(
            cfg.get("scale_color", (1.0, 1.0, 1.0, 0.95))
        )
        self.scale_color_btn.clicked.connect(self._choose_scale_color)
        self._sync_color_button(self.scale_color_btn, self._scale_color)
        form.addRow("Scale Color", self.scale_color_btn)

        self.chk_time = QCheckBox()
        self.chk_time.setChecked(cfg.get("show_timestamp", True))
        form.addRow("Show Timestamp", self.chk_time)

        self.time_mode = QComboBox()
        self.time_mode.addItem("Elapsed", "elapsed")
        self.time_mode.addItem("Absolute", "absolute")
        self.time_mode.addItem("Absolute + Elapsed", "both")
        mode = str(cfg.get("timestamp_mode", "elapsed"))
        idx = self.time_mode.findData(mode)
        self.time_mode.setCurrentIndex(0 if idx < 0 else idx)
        form.addRow("Timestamp Mode", self.time_mode)

        self.frame_interval = QDoubleSpinBox()
        self.frame_interval.setRange(1e-6, 1e9)
        self.frame_interval.setDecimals(6)
        self.frame_interval.setValue(float(cfg.get("frame_interval_value", 1.0)))
        form.addRow("Frame Interval", self.frame_interval)

        self.frame_unit = QComboBox()
        self.frame_unit.addItems(["us", "ms", "s", "min", "h"])
        self.frame_unit.setCurrentText(str(cfg.get("frame_interval_unit", "s")))
        form.addRow("Frame Interval Unit", self.frame_unit)

        self.time_anchor = QComboBox()
        self.time_anchor.addItems(
            ["top-left", "top-right", "bottom-left", "bottom-right"]
        )
        self.time_anchor.setCurrentText(
            str(cfg.get("timestamp_anchor", "top-left"))
        )
        form.addRow("Timestamp Anchor", self.time_anchor)

        self.time_offset_x = QDoubleSpinBox()
        self.time_offset_x.setRange(0.0, 1.0)
        self.time_offset_x.setDecimals(3)
        self.time_offset_x.setSingleStep(0.01)
        self.time_offset_x.setValue(float(cfg.get("timestamp_offset_x", 0.04)))
        form.addRow("Timestamp Offset X (0-1)", self.time_offset_x)

        self.time_offset_y = QDoubleSpinBox()
        self.time_offset_y.setRange(0.0, 1.0)
        self.time_offset_y.setDecimals(3)
        self.time_offset_y.setSingleStep(0.01)
        self.time_offset_y.setValue(float(cfg.get("timestamp_offset_y", 0.06)))
        form.addRow("Timestamp Offset Y (0-1)", self.time_offset_y)

        self.time_color_btn = QPushButton()
        self._time_color = self._tuple_to_qcolor(
            cfg.get("timestamp_color", (1.0, 1.0, 1.0, 0.95))
        )
        self.time_color_btn.clicked.connect(self._choose_time_color)
        self._sync_color_button(self.time_color_btn, self._time_color)
        form.addRow("Timestamp Color", self.time_color_btn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self):
        return {
            "show_scale_bar": self.chk_scale.isChecked(),
            "scale_bar_length": self.scale_length.value(),
            "scale_bar_unit": self.scale_unit.currentText(),
            "scale_anchor": self.scale_anchor.currentText(),
            "scale_offset_x": self.scale_offset_x.value(),
            "scale_offset_y": self.scale_offset_y.value(),
            "scale_color": self._qcolor_to_tuple(self._scale_color),
            "show_timestamp": self.chk_time.isChecked(),
            "timestamp_mode": self.time_mode.currentData(),
            "frame_interval_value": self.frame_interval.value(),
            "frame_interval_unit": self.frame_unit.currentText(),
            "timestamp_anchor": self.time_anchor.currentText(),
            "timestamp_offset_x": self.time_offset_x.value(),
            "timestamp_offset_y": self.time_offset_y.value(),
            "timestamp_color": self._qcolor_to_tuple(self._time_color),
        }

    def _choose_scale_color(self):
        color = QColorDialog.getColor(
            self._scale_color,
            self,
            "Choose Scale Color",
            options=QColorDialog.ShowAlphaChannel,
        )
        if color.isValid():
            self._scale_color = color
            self._sync_color_button(self.scale_color_btn, color)

    def _choose_time_color(self):
        color = QColorDialog.getColor(
            self._time_color,
            self,
            "Choose Timestamp Color",
            options=QColorDialog.ShowAlphaChannel,
        )
        if color.isValid():
            self._time_color = color
            self._sync_color_button(self.time_color_btn, color)

    def _sync_color_button(self, button, color):
        button.setText(color.name(QColor.HexArgb))
        button.setStyleSheet(
            f"background-color: rgba({color.red()}, {color.green()}, "
            f"{color.blue()}, {color.alpha()});"
        )

    def _tuple_to_qcolor(self, color):
        if (
            isinstance(color, (list, tuple))
            and len(color) == 4
            and all(isinstance(c, (int, float)) for c in color)
        ):
            r, g, b, a = [min(max(float(c), 0.0), 1.0) for c in color]
            return QColor.fromRgbF(r, g, b, a)
        return QColor.fromRgbF(1.0, 1.0, 1.0, 0.95)

    def _qcolor_to_tuple(self, color):
        return (color.redF(), color.greenF(), color.blueF(), color.alphaF())
