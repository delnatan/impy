"""TrackDisplaySettingsDialog — modal editor for a track layer's visual style.

Exposes the style dict understood by
:class:`pyvistra.visuals.tracks.TrackLayerVisual` (trail length, line width,
opacity, head markers, and color mode) so it can be tuned from the UI instead
of only via ``window.set_track_layer_style(...)``.

Color-by-property colors one whole trajectory by the mean of a numeric
properties column (not per-vertex/time-varying) — a general viewing concern,
independent of what produced the property (e.g. a tracking pipeline like
magikit).

Property-based selection/highlighting lives in the separate Property
Inspector dock (``pyvistra.widgets.property_inspector_panel``), not here —
this dialog is pure visual styling, independent of the layer's data (aside
from ``available_properties``, still needed for the color-by-property combo).
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
)

from .. import colormaps as cmap
from .color_button import ColorButton
from .property_filter_widget import PropertyRangeInfo

_COLOR_MODES = [
    ("Auto (by Track ID)", "auto"),
    ("Fixed Color", "fixed"),
    ("By Property", "property"),
]


class TrackDisplaySettingsDialog(QDialog):
    def __init__(self, style: dict, available_properties: list[PropertyRangeInfo], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Track Display Settings")

        cfg = dict(style or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        trail_window = cfg.get("trail_window", 30)
        self.no_trail_limit = QCheckBox()
        self.no_trail_limit.setChecked(trail_window is None)
        form.addRow("No Trail Limit", self.no_trail_limit)

        self.trail_window = QSpinBox()
        self.trail_window.setRange(1, 100_000)
        self.trail_window.setValue(int(trail_window) if trail_window is not None else 30)
        self.trail_window.setEnabled(trail_window is not None)
        form.addRow("Trail Window (frames)", self.trail_window)
        self.no_trail_limit.toggled.connect(
            lambda checked: self.trail_window.setEnabled(not checked)
        )

        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.1, 50.0)
        self.line_width.setDecimals(2)
        self.line_width.setValue(float(cfg.get("line_width", 2.0)))
        form.addRow("Line Width", self.line_width)

        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.0, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setDecimals(2)
        self.opacity.setValue(float(cfg.get("opacity", 0.9)))
        form.addRow("Opacity", self.opacity)

        self.show_heads = QCheckBox()
        self.show_heads.setChecked(bool(cfg.get("show_heads", True)))
        form.addRow("Show Track Heads", self.show_heads)

        self.head_size = QDoubleSpinBox()
        self.head_size.setRange(1.0, 200.0)
        self.head_size.setValue(float(cfg.get("head_size", 8.0)))
        form.addRow("Head Marker Size", self.head_size)

        self.min_track_length = QSpinBox()
        self.min_track_length.setRange(1, 100_000)
        self.min_track_length.setValue(int(cfg.get("min_track_length", 1)))
        self.min_track_length.setToolTip(
            "Hide tracks with fewer than this many total frames"
            " (e.g. set to 3 to hide singleton/2-frame 'dead' tracks)."
        )
        form.addRow("Min Track Length (frames)", self.min_track_length)

        self.color_mode = QComboBox()
        for label, value in _COLOR_MODES:
            self.color_mode.addItem(label, value)
        current_mode = str(cfg.get("color_mode", "auto"))
        idx = self.color_mode.findData(current_mode)
        self.color_mode.setCurrentIndex(0 if idx < 0 else idx)
        form.addRow("Color Mode", self.color_mode)

        self.fixed_color = ColorButton(
            cfg.get("fixed_color", (1.0, 1.0, 1.0, 1.0)), title="Choose Track Color"
        )
        form.addRow("Fixed Color", self.fixed_color)

        self.color_property = QComboBox()
        self.color_property.addItems([info.name for info in available_properties])
        color_property = cfg.get("color_property")
        if color_property is not None:
            idx = self.color_property.findText(str(color_property))
            if idx >= 0:
                self.color_property.setCurrentIndex(idx)
        form.addRow("Color Property", self.color_property)

        self.colormap_name = QComboBox()
        self.colormap_name.addItems(cmap.names())
        cmap_name = str(cfg.get("colormap_name", "viridis"))
        idx = self.colormap_name.findText(cmap_name)
        if idx >= 0:
            self.colormap_name.setCurrentIndex(idx)
        form.addRow("Colormap", self.colormap_name)

        self.color_mode.currentIndexChanged.connect(self._sync_color_controls)
        self._sync_color_controls()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _sync_color_controls(self) -> None:
        mode = self.color_mode.currentData()
        self.fixed_color.setEnabled(mode == "fixed")
        self.color_property.setEnabled(mode == "property")
        self.colormap_name.setEnabled(mode == "property")

    def get_style(self) -> dict:
        return {
            "trail_window": None if self.no_trail_limit.isChecked() else self.trail_window.value(),
            "line_width": self.line_width.value(),
            "opacity": self.opacity.value(),
            "show_heads": self.show_heads.isChecked(),
            "head_size": self.head_size.value(),
            "min_track_length": self.min_track_length.value(),
            "color_mode": self.color_mode.currentData(),
            "fixed_color": self.fixed_color.rgba,
            "color_property": self.color_property.currentText() or None,
            "colormap_name": self.colormap_name.currentText(),
        }
