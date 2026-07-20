from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QVBoxLayout,
)


class ZMontageSettingsDialog(QDialog):
    """Settings for ZMontageViewer's per-tile Z labels and tile spacing."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Montage Settings")
        self.resize(360, 220)

        cfg = dict(config or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.chk_show_labels = QCheckBox()
        self.chk_show_labels.setChecked(cfg.get("show_labels", True))
        form.addRow("Show Z Labels", self.chk_show_labels)

        self.label_anchor = QComboBox()
        self.label_anchor.addItems(
            ["top-left", "top-right", "bottom-left", "bottom-right"]
        )
        self.label_anchor.setCurrentText(
            str(cfg.get("label_anchor", "top-left"))
        )
        form.addRow("Label Placement", self.label_anchor)

        self.label_font_size = QDoubleSpinBox()
        self.label_font_size.setRange(4.0, 48.0)
        self.label_font_size.setDecimals(1)
        self.label_font_size.setValue(float(cfg.get("label_font_size", 10.0)))
        form.addRow("Label Font Size", self.label_font_size)

        self.label_color_btn = QPushButton()
        self._label_color = self._tuple_to_qcolor(
            cfg.get("label_color", (1.0, 1.0, 1.0, 0.8))
        )
        self.label_color_btn.clicked.connect(self._choose_label_color)
        self._sync_color_button(self.label_color_btn, self._label_color)
        form.addRow("Label Color", self.label_color_btn)

        self.gap_fraction = QDoubleSpinBox()
        self.gap_fraction.setRange(0.0, 0.5)
        self.gap_fraction.setDecimals(2)
        self.gap_fraction.setSingleStep(0.01)
        self.gap_fraction.setValue(float(cfg.get("gap_fraction", 0.05)))
        form.addRow("Tile Gap (fraction of tile size)", self.gap_fraction)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self):
        return {
            "show_labels": self.chk_show_labels.isChecked(),
            "label_anchor": self.label_anchor.currentText(),
            "label_font_size": self.label_font_size.value(),
            "label_color": self._qcolor_to_tuple(self._label_color),
            "gap_fraction": self.gap_fraction.value(),
        }

    def _choose_label_color(self):
        color = QColorDialog.getColor(
            self._label_color,
            self,
            "Choose Label Color",
            options=QColorDialog.ShowAlphaChannel,
        )
        if color.isValid():
            self._label_color = color
            self._sync_color_button(self.label_color_btn, color)

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
        return QColor.fromRgbF(1.0, 1.0, 1.0, 0.8)

    def _qcolor_to_tuple(self, color):
        return (color.redF(), color.greenF(), color.blueF(), color.alphaF())
