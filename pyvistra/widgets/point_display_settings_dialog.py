"""PointDisplaySettingsDialog — modal editor for a point layer's visual style.

Exposes the full style dict understood by
:class:`pyvistra.visuals.points.PointLayerVisual` (marker size, screen-space
clamps, colors, optional circle overlay, and label display) so it can be
tuned from the UI instead of only via ``window.set_point_layer_style(...)``.

Property-based selection/highlighting lives in the separate Property
Inspector dock (``pyvistra.widgets.property_inspector_panel``), not here —
this dialog is pure visual styling, independent of the layer's data.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from .color_button import ColorButton


class PointDisplaySettingsDialog(QDialog):
    def __init__(self, style: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Point Display Settings")

        cfg = dict(style or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.size = QDoubleSpinBox()
        self.size.setRange(0.1, 1000.0)
        self.size.setDecimals(2)
        self.size.setValue(float(cfg.get("size", 9.0)))
        form.addRow("Marker Size", self.size)

        self.min_screen_px = QDoubleSpinBox()
        self.min_screen_px.setRange(0.0, 1000.0)
        self.min_screen_px.setValue(float(cfg.get("min_screen_px", 3.0)))
        form.addRow("Min Screen Size (px)", self.min_screen_px)

        self.max_screen_px = QDoubleSpinBox()
        self.max_screen_px.setRange(0.0, 2000.0)
        self.max_screen_px.setValue(float(cfg.get("max_screen_px", 80.0)))
        form.addRow("Max Screen Size (px)", self.max_screen_px)

        self.edge_width = QDoubleSpinBox()
        self.edge_width.setRange(0.1, 50.0)
        self.edge_width.setDecimals(2)
        self.edge_width.setValue(float(cfg.get("edge_width", 1.2)))
        form.addRow("Edge Width", self.edge_width)

        self.layer_color = ColorButton(
            cfg.get("layer_color", (1.0, 1.0, 0.0, 1.0)), title="Choose Point Color"
        )
        form.addRow("Point Color", self.layer_color)

        self.selected_color = ColorButton(
            cfg.get("selected_color", (1.0, 0.3, 0.3, 1.0)),
            title="Choose Selected-Point Color",
        )
        form.addRow("Selected Color", self.selected_color)

        self.show_circle = QCheckBox()
        self.show_circle.setChecked(bool(cfg.get("show_circle", False)))
        form.addRow("Show Circle", self.show_circle)

        self.circle_radius_px = QDoubleSpinBox()
        self.circle_radius_px.setRange(0.1, 500.0)
        self.circle_radius_px.setValue(float(cfg.get("circle_radius_px", 4.0)))
        form.addRow("Circle Radius (px)", self.circle_radius_px)

        self.z_tolerance = QDoubleSpinBox()
        self.z_tolerance.setRange(0.0, 1000.0)
        self.z_tolerance.setDecimals(2)
        self.z_tolerance.setValue(float(cfg.get("z_tolerance", 0.5)))
        form.addRow("Z Tolerance", self.z_tolerance)

        self.label_visible = QCheckBox()
        self.label_visible.setChecked(bool(cfg.get("label_visible", False)))
        form.addRow("Show Labels", self.label_visible)

        self.label_template = QLineEdit(str(cfg.get("label_template", "{point_id}")))
        form.addRow("Label Template", self.label_template)

        self.label_font_size = QSpinBox()
        self.label_font_size.setRange(4, 72)
        self.label_font_size.setValue(int(cfg.get("label_font_size", 9)))
        form.addRow("Label Font Size", self.label_font_size)

        self.dense_label_threshold = QSpinBox()
        self.dense_label_threshold.setRange(1, 1_000_000)
        self.dense_label_threshold.setValue(int(cfg.get("dense_label_threshold", 2000)))
        form.addRow("Label Count Threshold", self.dense_label_threshold)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_style(self) -> dict:
        return {
            "size": self.size.value(),
            "min_screen_px": self.min_screen_px.value(),
            "max_screen_px": self.max_screen_px.value(),
            "edge_width": self.edge_width.value(),
            "layer_color": self.layer_color.rgba,
            "selected_color": self.selected_color.rgba,
            "show_circle": self.show_circle.isChecked(),
            "circle_radius_px": self.circle_radius_px.value(),
            "z_tolerance": self.z_tolerance.value(),
            "label_visible": self.label_visible.isChecked(),
            "label_template": self.label_template.text(),
            "label_font_size": self.label_font_size.value(),
            "dense_label_threshold": self.dense_label_threshold.value(),
        }
