"""
Gel Image Analyzer Widget

Provides molecular weight/size estimation from gel electrophoresis images.
Works with both protein gels (kDa) and DNA gels (bp).
Uses rectangle shapes to define lanes with integrated peak markers.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QPointF
from qtpy.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import find_peaks
from vispy import scene

from ..data.shapes import EVT_EDITED, RECTANGLE, rectangle_bounds
from ..ui.manager import manager
from ..widgets.histogram import TEXT_COLOR, WIDGET_BG


def get_builtin_ladders_path():
    return Path(__file__).resolve().parents[1] / "_resources" / "ladders.json"


def load_ladders(path=None):
    if path is None:
        path = get_builtin_ladders_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("ladders", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading ladders: {e}")
        return []


GEL_LANE_KEY = "gel_lane"
GEL_MARKERS_KEY = "gel_markers"
GEL_MARKER_COLOR_KEY = "gel_marker_color"
GEL_MARKER_CHANGED = "gel_marker_changed"
GEL_MARKER_MOVED = "gel_marker_moved"
GEL_MARKER_RELEASED = "gel_marker_released"


@dataclass(frozen=True)
class ShapeLaneRef:
    layer_name: str
    shape_id: int


class ShapeLane:
    """Small adapter that lets gel analysis operate on rectangle shapes."""

    def __init__(self, window, layer, shape_id):
        self.window = window
        self.layer = layer
        self.shape_id = int(shape_id)

    @property
    def rec(self):
        return self.layer.data.get(self.shape_id)

    @property
    def ref(self):
        return ShapeLaneRef(self.layer.name, self.shape_id)

    @property
    def name(self):
        rec = self.rec
        return rec.label or rec.properties.get("name") or f"Shape {self.shape_id}"

    @property
    def markers(self):
        return self.rec.properties.setdefault(GEL_MARKERS_KEY, [])

    def mark_as_lane(self, color="#00BCD4"):
        props = self.rec.properties
        props[GEL_LANE_KEY] = True
        props[GEL_MARKER_COLOR_KEY] = color
        props.setdefault(GEL_MARKERS_KEY, [])
        self._emit_changed()

    def bounds(self, yx_shape=None):
        return rectangle_bounds(self.rec, yx_shape)

    def get_region(self, data):
        x0, y0, x1, y1 = self.bounds(data.shape[-2:])
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 0), dtype=np.asarray(data).dtype)
        return np.asarray(data)[y0:y1, x0:x1]

    def set_markers(self, marker_data):
        self.rec.properties[GEL_MARKERS_KEY] = [dict(m) for m in marker_data]
        self.rec.properties[GEL_LANE_KEY] = True
        self._emit_changed()

    def clear_markers(self):
        self.rec.properties[GEL_MARKERS_KEY] = []
        self._emit_changed()

    def get_marker_positions(self):
        return sorted(float(m.get("y_local", 0.0)) for m in self.markers)

    def update_marker_labels(self, labels):
        markers = [dict(m) for m in self.markers]
        order = sorted(range(len(markers)), key=lambda i: markers[i].get("y_local", 0.0))
        for label_idx, marker_idx in enumerate(order):
            if label_idx >= len(labels):
                break
            markers[marker_idx]["label"] = labels[label_idx]
        self.rec.properties[GEL_MARKERS_KEY] = markers
        self._emit_changed()

    def _emit_changed(self):
        self.layer.data._emit(EVT_EDITED, self.shape_id)


def _color_button_style(hex_color):
    return (
        f"QPushButton {{ background-color: {hex_color}; border: 1px solid #666; "
        f"border-radius: 3px; min-width: 48px; }}"
        f"QPushButton:hover {{ border: 1px solid #aaa; }}"
    )


class LaneProfileWidget(QWidget):
    """
    Compact pure-Qt lane intensity profile plot.

    Draws the 1D mean-intensity profile of a gel lane with optional
    peak position markers overlaid as dashed vertical lines and dots.
    """

    ML = 54   # left margin
    MR = 10
    MT = 10
    MB = 28

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.profile = None          # 1D float ndarray
        self.peaks = []              # list of int pixel indices
        self.profile_color = "#AAAAAA"
        self.peak_color = "#FF6B35"

    def set_data(self, profile, peaks=None, profile_color=None, peak_color=None):
        self.profile = np.asarray(profile, dtype=float) if profile is not None else None
        self.peaks = list(peaks) if peaks is not None else []
        if profile_color is not None:
            self.profile_color = profile_color
        if peak_color is not None:
            self.peak_color = peak_color
        self.update()

    def set_peaks(self, peaks=None):
        self.peaks = list(peaks) if peaks is not None else []
        self.update()

    def clear(self):
        self.profile = None
        self.peaks = []
        self.update()

    def _plot_rect(self):
        w = max(1, self.width() - self.ML - self.MR)
        h = max(1, self.height() - self.MT - self.MB)
        return self.ML, self.MT, w, h

    @staticmethod
    def _to_px(v, vmin, vmax, origin, span):
        if vmax == vmin:
            return origin + span / 2
        return origin + (v - vmin) / (vmax - vmin) * span

    @staticmethod
    def _to_py(v, vmin, vmax, origin, span):
        if vmax == vmin:
            return origin + span / 2
        return origin + span - (v - vmin) / (vmax - vmin) * span

    @staticmethod
    def _fmt(v):
        a = abs(v)
        if a == 0:
            return "0"
        if a >= 10000 or a < 0.01:
            return f"{v:.1e}"
        if a < 10:
            return f"{v:.2f}"
        return f"{v:.0f}"

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), WIDGET_BG)

        if self.profile is None or len(self.profile) == 0:
            painter.setPen(TEXT_COLOR)
            font = QFont()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "Select a lane to preview")
            return

        px, py, pw, ph = self._plot_rect()
        profile = self.profile
        n = len(profile)

        y_min = float(np.min(profile))
        y_max = float(np.max(profile))
        y_rng = y_max - y_min
        if y_rng > 0:
            y_min -= y_rng * 0.05
            y_max += y_rng * 0.05
        else:
            y_min -= 1.0
            y_max += 1.0

        self._draw_axes(painter, px, py, pw, ph, n, y_min, y_max)
        self._draw_peaks(painter, px, py, pw, ph, n, y_min, y_max, profile)
        self._draw_profile(painter, px, py, pw, ph, n, y_min, y_max, profile)

    def _draw_axes(self, painter, px, py, pw, ph, n, y_min, y_max):
        painter.setPen(QPen(QColor(80, 80, 80)))
        painter.drawRect(px, py, pw, ph)

        grid_pen = QPen(QColor(55, 55, 55))
        grid_pen.setStyle(Qt.DotLine)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            gy = int(py + (i / 4) * ph)
            painter.drawLine(px, gy, px + pw, gy)

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(TEXT_COLOR)

        for y_val in (y_min, (y_min + y_max) / 2, y_max):
            y_px = int(self._to_py(y_val, y_min, y_max, py, ph))
            painter.drawText(2, y_px - 6, self.ML - 4, 12,
                             Qt.AlignRight | Qt.AlignVCenter, self._fmt(y_val))

        for x_val in (0, (n - 1) / 2, n - 1):
            x_px = int(self._to_px(x_val, 0, n - 1, px, pw))
            painter.drawText(x_px - 15, py + ph + 4, 30, 14,
                             Qt.AlignCenter, str(int(x_val)))

        painter.drawText(int(px + pw / 2 - 35), py + ph + 17, 70, 11,
                         Qt.AlignCenter, "Position (px)")

    def _draw_peaks(self, painter, px, py, pw, ph, n, y_min, y_max, profile):
        if not self.peaks:
            return

        pk_color = QColor(self.peak_color)

        dash_pen = QPen(pk_color)
        dash_pen.setStyle(Qt.DashLine)
        dash_pen.setWidth(1)
        painter.setPen(dash_pen)
        for pk in self.peaks:
            if 0 <= pk < n:
                xp = int(self._to_px(pk, 0, n - 1, px, pw))
                painter.drawLine(xp, py, xp, py + ph)

        painter.setPen(QPen(pk_color))
        painter.setBrush(pk_color)
        for pk in self.peaks:
            if 0 <= pk < n:
                xp = self._to_px(pk, 0, n - 1, px, pw)
                yp = self._to_py(float(profile[pk]), y_min, y_max, py, ph)
                painter.drawEllipse(int(xp - 3), int(yp - 3), 6, 6)
        painter.setBrush(Qt.NoBrush)

    def _draw_profile(self, painter, px, py, pw, ph, n, y_min, y_max, profile):
        pen = QPen(QColor(self.profile_color))
        pen.setWidth(1)
        painter.setPen(pen)

        points = [
            QPointF(
                self._to_px(i, 0, n - 1, px, pw),
                self._to_py(float(v), y_min, y_max, py, ph),
            )
            for i, v in enumerate(profile)
        ]
        if len(points) > 1:
            painter.drawPolyline(QPolygonF(points))


class GelAnalyzerWidget(QWidget):
    """
    Widget for analyzing gel electrophoresis images.

    Detects bands in shape-backed lanes, uses one lane as a
    molecular weight ladder, and estimates MW of bands in other lanes.
    Supports both protein (kDa) and DNA (bp) gels.
    """

    def __init__(self, window_manager=None):
        super().__init__()
        self.window_manager = window_manager or manager
        self.ladders = load_ladders()
        self._lanes = []
        self._std_ref = None
        self._calibration = None
        self._marker_visuals = {}
        self._shape_unsubscribers = []
        self._handling_shape_event = False

        # Appearance settings
        self._ladder_color = "#FFBB16"
        self._sample_color = "#00BCD4"
        self._label_font_size = 8
        self._label_offset = 3
        self._label_side = "top"
        self._ladder_label_side = "left"

        self._connected_window = None   # window whose shape selection we're watching

        self.setWindowTitle("Gel Analyzer")
        self.resize(340, 440)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_workflow_tab(), "Workflow")
        self.tabs.addTab(self._build_settings_tab(), "Settings")
        self.tabs.addTab(self._build_profile_tab(), "Profile")
        self.tabs.addTab(self._build_results_tab(), "Results")
        layout.addWidget(self.tabs)

        # Connect after all tabs are built so all widgets exist
        self.lane_combo.currentIndexChanged.connect(self._update_profile_plot)
        self.invert_check.stateChanged.connect(self._update_profile_plot)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_workflow_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        ladder_row = QHBoxLayout()
        self.ladder_combo = QComboBox()
        self.ladder_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for ladder in self.ladders:
            unit = ladder.get("unit", "kDa")
            self.ladder_combo.addItem(f"{ladder['name']} ({unit})", userData=ladder)
        ladder_row.addWidget(self.ladder_combo)
        btn_load_custom = QPushButton("Load…")
        btn_load_custom.clicked.connect(self._load_custom_ladder)
        ladder_row.addWidget(btn_load_custom)
        form.addRow("Ladder:", ladder_row)

        lane_row = QHBoxLayout()
        self.lane_combo = QComboBox()
        self.lane_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lane_row.addWidget(self.lane_combo)
        btn_refresh = QPushButton("↺")
        btn_refresh.setFixedWidth(28)
        btn_refresh.setToolTip("Refresh lane list")
        btn_refresh.clicked.connect(self._refresh_lanes)
        lane_row.addWidget(btn_refresh)
        form.addRow("Std lane:", lane_row)

        layout.addLayout(form)

        detect_group = QGroupBox("Detection")
        detect_layout = QVBoxLayout(detect_group)
        detect_layout.setContentsMargins(6, 4, 6, 6)
        detect_layout.setSpacing(4)

        self.invert_check = QCheckBox("Invert intensity (dark bands)")
        self.invert_check.setChecked(True)
        self.invert_check.setToolTip(
            "Check for Coomassie/silver stains (dark bands on light background)\n"
            "Uncheck for fluorescent stains (bright bands on dark background)"
        )
        detect_layout.addWidget(self.invert_check)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Prom:"))
        self.prominence_spin = QSpinBox()
        self.prominence_spin.setRange(100, 50000)
        self.prominence_spin.setValue(1000)
        self.prominence_spin.setSingleStep(100)
        params_row.addWidget(self.prominence_spin)
        params_row.addSpacing(10)
        params_row.addWidget(QLabel("Min dist:"))
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(1, 100)
        self.distance_spin.setValue(5)
        params_row.addWidget(self.distance_spin)
        detect_layout.addLayout(params_row)

        layout.addWidget(detect_group)

        row1 = QHBoxLayout()
        btn_convert = QPushButton("Convert Lanes")
        btn_convert.setToolTip(
            "Tag rectangle shapes as gel lanes without peak detection.\n"
            "Then Shift+Click to add markers manually."
        )
        btn_convert.clicked.connect(self._convert_lanes)
        row1.addWidget(btn_convert)
        btn_detect = QPushButton("Detect Peaks")
        btn_detect.clicked.connect(self._detect_peaks)
        row1.addWidget(btn_detect)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_calibrate = QPushButton("Calibrate")
        self.btn_calibrate.clicked.connect(self._calibrate)
        self.btn_calibrate.setToolTip("Build calibration curve from ladder lane")
        row2.addWidget(self.btn_calibrate)
        self.btn_apply = QPushButton("Apply Calibration")
        self.btn_apply.clicked.connect(self._apply_calibration)
        self.btn_apply.setEnabled(False)
        self.btn_apply.setToolTip("Estimate MW for sample lanes")
        row2.addWidget(self.btn_apply)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        btn_align = QPushButton("Align Lanes")
        btn_align.setToolTip("Align all rectangle lanes to the standard lane height")
        btn_align.clicked.connect(self._align_lanes)
        row3.addWidget(btn_align)
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._clear_markers)
        row3.addWidget(btn_clear)
        layout.addLayout(row3)

        layout.addStretch()
        return w

    def _build_settings_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)

        self.label_side_combo = QComboBox()
        self.label_side_combo.addItems(["Top", "Left", "Right"])
        self.label_side_combo.setCurrentText("Top")
        form.addRow("Sample labels:", self.label_side_combo)

        self.ladder_label_side_combo = QComboBox()
        self.ladder_label_side_combo.addItems(["Left", "Right", "Top"])
        self.ladder_label_side_combo.setCurrentText("Left")
        form.addRow("Ladder labels:", self.ladder_label_side_combo)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 20)
        self.font_size_spin.setValue(self._label_font_size)
        form.addRow("Font size:", self.font_size_spin)

        self.label_offset_spin = QSpinBox()
        self.label_offset_spin.setRange(0, 50)
        self.label_offset_spin.setValue(3)
        self.label_offset_spin.setSuffix(" px")
        self.label_offset_spin.setToolTip("Gap between band line and label text")
        form.addRow("Label offset:", self.label_offset_spin)

        self.btn_ladder_color = QPushButton()
        self.btn_ladder_color.setStyleSheet(_color_button_style(self._ladder_color))
        self.btn_ladder_color.setFixedHeight(22)
        self.btn_ladder_color.clicked.connect(lambda: self._pick_color("ladder"))
        form.addRow("Ladder color:", self.btn_ladder_color)

        self.btn_sample_color = QPushButton()
        self.btn_sample_color.setStyleSheet(_color_button_style(self._sample_color))
        self.btn_sample_color.setFixedHeight(22)
        self.btn_sample_color.clicked.connect(lambda: self._pick_color("sample"))
        form.addRow("Sample color:", self.btn_sample_color)

        layout.addLayout(form)

        vis_group = QGroupBox("Visibility")
        vis_layout = QVBoxLayout(vis_group)
        vis_layout.setContentsMargins(6, 4, 6, 6)
        vis_layout.setSpacing(2)

        self.show_labels_check = QCheckBox("Show peak labels")
        self.show_labels_check.setChecked(True)
        self.show_labels_check.stateChanged.connect(self._on_show_labels_changed)
        vis_layout.addWidget(self.show_labels_check)

        self.show_borders_check = QCheckBox("Show lane borders")
        self.show_borders_check.setChecked(True)
        self.show_borders_check.stateChanged.connect(self._on_show_borders_changed)
        vis_layout.addWidget(self.show_borders_check)

        layout.addWidget(vis_group)

        tip = QLabel(
            "Shift+Click  add band\n"
            "Ctrl/Cmd+Click  remove band\n"
            "Drag line  adjust position"
        )
        tip.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(tip)

        btn_apply_settings = QPushButton("Apply to Lanes")
        btn_apply_settings.setToolTip(
            "Re-apply color, font size, and label position to all current lanes"
        )
        btn_apply_settings.clicked.connect(self._apply_appearance_settings)
        layout.addWidget(btn_apply_settings)

        layout.addStretch()
        return w

    def _build_profile_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        self.profile_widget = LaneProfileWidget()
        layout.addWidget(self.profile_widget, 1)

        self.profile_status = QLabel("Select a lane to preview")
        self.profile_status.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.profile_status)

        return w

    def _build_results_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 8, 6, 6)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        btn_row = QHBoxLayout()
        btn_copy = QPushButton("Copy to Clipboard")
        btn_copy.clicked.connect(self._copy_results)
        btn_row.addWidget(btn_copy)
        btn_export = QPushButton("Export CSV…")
        btn_export.clicked.connect(self._export_csv)
        btn_row.addWidget(btn_export)
        layout.addLayout(btn_row)

        return w

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_lanes()
        self._connect_to_window(self._active_window())

    def closeEvent(self, event):
        self._clear_markers()
        self._disconnect_from_window()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Window connection (for live profile updates on shape click)
    # ------------------------------------------------------------------

    def _active_window(self):
        window = getattr(self.window_manager, "active_window", None)
        if window is not None:
            self._connect_to_window(window)
            return window
        windows = self.window_manager.get_all()
        if windows:
            window = next(iter(windows.values()))
            self._connect_to_window(window)
            return window
        return None

    def _connect_to_window(self, window):
        if window is self._connected_window:
            return
        self._disconnect_from_window()
        if window is not None:
            if hasattr(window, "shape_selection_changed"):
                window.shape_selection_changed.connect(self._on_shape_selected)
            if hasattr(window, "layer_added"):
                window.layer_added.connect(self._on_layers_changed)
            if hasattr(window, "layer_removed"):
                window.layer_removed.connect(self._on_layers_changed)
            self._connected_window = window
            self._connect_shape_data()
            self._refresh_marker_visuals()

    def _disconnect_from_window(self):
        if self._connected_window is not None:
            try:
                if hasattr(self._connected_window, "shape_selection_changed"):
                    self._connected_window.shape_selection_changed.disconnect(
                        self._on_shape_selected
                    )
            except (TypeError, RuntimeError):
                pass
            for signal_name in ("layer_added", "layer_removed"):
                try:
                    getattr(self._connected_window, signal_name).disconnect(
                        self._on_layers_changed
                    )
                except (AttributeError, TypeError, RuntimeError):
                    pass
            self._disconnect_shape_data()
            self._clear_marker_visuals()
            self._connected_window = None

    def _connect_shape_data(self):
        self._disconnect_shape_data()
        window = self._connected_window
        if window is None:
            return
        for layer in window.layers.by_type("shapes"):
            self._shape_unsubscribers.append(
                layer.data.subscribe(
                    lambda kind, sid, _layer=layer: self._on_shape_data_changed(
                        kind, sid, _layer
                    )
                )
            )

    def _disconnect_shape_data(self):
        for unsub in self._shape_unsubscribers:
            try:
                unsub()
            except Exception:
                pass
        self._shape_unsubscribers = []

    def _on_layers_changed(self, _layer):
        self._connect_shape_data()
        self._refresh_lanes()
        self._refresh_marker_visuals()

    def _on_shape_data_changed(self, event_kind, shape_id, layer=None):
        if self._handling_shape_event:
            return
        if event_kind == GEL_MARKER_MOVED:
            self._refresh_marker_visuals(layer, shape_id)
            self._update_profile_marker_overlay(layer, shape_id)
            return

        self._handling_shape_event = True
        try:
            if event_kind in {"added", "removed", "label", "bulk"}:
                self._refresh_lanes(preserve_selection=True)
            self._refresh_marker_visuals(layer, shape_id)
            if self._calibration is not None:
                self._apply_calibration()
            else:
                self._update_profile_plot()
        finally:
            self._handling_shape_event = False

    def _update_profile_marker_overlay(self, layer, shape_id):
        ref = ShapeLaneRef(layer.name, int(shape_id)) if layer is not None else None
        if ref != self.lane_combo.currentData():
            return
        lane = self._resolve_lane(ref)
        if lane is None:
            return
        peaks = [int(round(m.get("y_local", 0.0))) for m in lane.markers]
        self.profile_widget.set_peaks(peaks or None)
        n = len(peaks)
        if n:
            self.profile_status.setText(
                f"Lane {self._lane_label(lane)}  ·  "
                f"{n} peak{'s' if n != 1 else ''} detected"
            )

    def _on_shape_selected(self, layer, shape_id):
        """Sync lane combo and refresh profile when a rectangle shape is clicked."""
        if layer is None or shape_id is None:
            return
        window = self._active_window()
        if not window:
            return
        if layer.layer_type != "shapes" or shape_id not in layer.data:
            return
        rec = layer.data.get(shape_id)
        if rec.shape_type != RECTANGLE:
            return
        ref = ShapeLaneRef(layer.name, int(shape_id))
        if self._combo_index_for_ref(ref) is None:
            self._refresh_lanes()
        for i in range(self.lane_combo.count()):
            if self.lane_combo.itemData(i) == ref:
                if self.lane_combo.currentIndex() == i:
                    self._update_profile_plot()
                else:
                    self.lane_combo.setCurrentIndex(i)
                self.tabs.setCurrentIndex(2)
                return

        self._update_profile_plot()
        self.tabs.setCurrentIndex(2)

    def _combo_index_for_ref(self, ref):
        for i in range(self.lane_combo.count()):
            if self.lane_combo.itemData(i) == ref:
                return i
        return None

    # ------------------------------------------------------------------
    # Profile plot
    # ------------------------------------------------------------------

    def _update_profile_plot(self):
        """Refresh the Profile tab for the currently selected lane."""
        window = self._active_window()
        if not window:
            self.profile_widget.clear()
            self.profile_status.setText("No active window")
            return

        ref = self.lane_combo.currentData()
        lane = self._resolve_lane(ref)
        if lane is None:
            self.profile_widget.clear()
            self.profile_status.setText("No lane selected")
            return

        gray = self._get_current_image()
        if gray is None:
            self.profile_widget.clear()
            self.profile_status.setText("No image data")
            return

        gray_proc = (gray.max() - gray) if self.invert_check.isChecked() else gray

        region = lane.get_region(gray_proc)
        if region.size == 0:
            self.profile_widget.clear()
            return

        profile = region.mean(axis=1)

        peaks = [int(round(m.get("y_local", 0.0))) for m in lane.markers]

        is_ladder = ref == self._std_ref
        color = self._ladder_color if is_ladder else self._sample_color

        self.profile_widget.set_data(
            profile, peaks or None, profile_color=color, peak_color="#FF6B35"
        )

        status = f"Lane {self._lane_label(lane)}  ·  {len(profile)} px"
        if peaks:
            n = len(peaks)
            status += f"  ·  {n} peak{'s' if n != 1 else ''} detected"
        self.profile_status.setText(status)

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

    def _pick_color(self, which):
        current = self._ladder_color if which == "ladder" else self._sample_color
        color = QColorDialog.getColor(
            initial=QColor(current), parent=self, title=f"Pick {which} color"
        )
        if not color.isValid():
            return
        hex_color = color.name()
        if which == "ladder":
            self._ladder_color = hex_color
            self.btn_ladder_color.setStyleSheet(_color_button_style(hex_color))
        else:
            self._sample_color = hex_color
            self.btn_sample_color.setStyleSheet(_color_button_style(hex_color))

    def _apply_appearance_settings(self):
        self._label_side = self.label_side_combo.currentText().lower()
        self._ladder_label_side = self.ladder_label_side_combo.currentText().lower()
        self._label_font_size = self.font_size_spin.value()
        self._label_offset = self.label_offset_spin.value()

        window = self._active_window()
        if not window:
            return

        self._handling_shape_event = True
        try:
            for lane in self._iter_lanes(window):
                is_ladder = lane.ref == self._std_ref
                color = self._ladder_color if is_ladder else self._sample_color
                label_side = self._ladder_label_side if is_ladder else self._label_side
                props = lane.rec.properties
                props[GEL_MARKER_COLOR_KEY] = color
                props["gel_label_side"] = label_side
                props["gel_label_font_size"] = self._label_font_size
                props["gel_label_offset"] = self._label_offset
                props["gel_show_border"] = self.show_borders_check.isChecked()
                for marker in props.get(GEL_MARKERS_KEY, []):
                    marker["color"] = color
                lane._emit_changed()
        finally:
            self._handling_shape_event = False

        window.canvas.update()
        self._refresh_marker_visuals()
        self._update_profile_plot()

    def _on_show_labels_changed(self, *_args):
        self._refresh_marker_visuals()

    def _on_show_borders_changed(self, *_args):
        window = self._active_window()
        if not window:
            return
        self._handling_shape_event = True
        try:
            for lane in self._iter_lanes(window):
                lane.rec.properties["gel_show_border"] = self.show_borders_check.isChecked()
                lane._emit_changed()
        finally:
            self._handling_shape_event = False
        self._refresh_marker_visuals()

    # ------------------------------------------------------------------
    # Ladder loading
    # ------------------------------------------------------------------

    def _load_custom_ladder(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ladder Preset", "", "JSON Files (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            if "ladders" in data:
                new_ladders = data["ladders"]
            elif "name" in data and ("sizes" in data or "weights_kda" in data):
                new_ladders = [data]
            else:
                print("Invalid ladder format")
                return

            for ladder in new_ladders:
                if "weights_kda" in ladder and "sizes" not in ladder:
                    ladder["sizes"] = ladder["weights_kda"]
                    ladder["unit"] = "kDa"
                self.ladders.append(ladder)
                unit = ladder.get("unit", "kDa")
                self.ladder_combo.addItem(
                    f"{ladder['name']} ({unit})", userData=ladder
                )

            self.ladder_combo.setCurrentIndex(self.ladder_combo.count() - 1)
            print(f"Loaded {len(new_ladders)} ladder(s) from {path}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading ladder file: {e}")

    # ------------------------------------------------------------------
    # Lane management
    # ------------------------------------------------------------------

    def _iter_lanes(self, window, marked_only=False):
        if window is None:
            return
        for layer in window.layers.by_type("shapes"):
            for rec in layer.data.get_time_slice(window.t_idx, window.z_idx):
                if rec.shape_type != RECTANGLE:
                    continue
                if marked_only and not rec.properties.get(GEL_LANE_KEY):
                    continue
                yield ShapeLane(window, layer, rec.shape_id)

    def _resolve_lane(self, ref):
        window = self._active_window()
        if window is None or not isinstance(ref, ShapeLaneRef):
            return None
        if ref.layer_name not in window.layers:
            return None
        layer = window.layers[ref.layer_name]
        if ref.shape_id not in layer.data:
            return None
        rec = layer.data.get(ref.shape_id)
        if rec.shape_type != RECTANGLE:
            return None
        return ShapeLane(window, layer, ref.shape_id)

    def _lane_label(self, lane):
        return f"{lane.layer.name}:{lane.shape_id} ({lane.name})"

    def _refresh_lanes(self, preserve_selection=False):
        current = self.lane_combo.currentData() if preserve_selection else None
        self.lane_combo.clear()
        window = self._active_window()
        if not window:
            return
        restore_idx = None
        for lane in self._iter_lanes(window):
            idx = self.lane_combo.count()
            self.lane_combo.addItem(self._lane_label(lane), userData=lane.ref)
            if current is not None and lane.ref == current:
                restore_idx = idx
        if restore_idx is not None:
            self.lane_combo.setCurrentIndex(restore_idx)

    def _get_current_image(self):
        window = self._active_window()
        if not window:
            return None
        cache = window.renderer.current_slice_cache
        if cache is None:
            return None
        return cache.mean(axis=0) if cache.ndim == 3 else cache

    def _align_lanes(self):
        """Align every rectangle lane to the standard lane's Y bounds."""
        window = self._active_window()
        if not window:
            self.results_text.setText("No active window")
            return

        std_ref = self._std_ref or self.lane_combo.currentData()
        std_lane = self._resolve_lane(std_ref)
        if std_lane is None:
            self.results_text.setText("No standard lane selected")
            return

        _sx0, sy0, _sx1, sy1 = std_lane.bounds()
        aligned = 0
        self._handling_shape_event = True
        try:
            for lane in self._iter_lanes(window):
                if lane.ref == std_lane.ref:
                    continue
                x0, _y0, x1, _y1 = lane.bounds()
                lane.layer.data.update(lane.shape_id, [x0, sy0, x1, sy1])
                aligned += 1
        finally:
            self._handling_shape_event = False

        self._std_ref = std_lane.ref
        self._refresh_lanes(preserve_selection=True)
        self._refresh_marker_visuals()
        window.canvas.update()
        self.results_text.setText(
            f"Aligned {aligned} lane(s) to {self._lane_label(std_lane)}."
        )

    def _clear_marker_visuals(self):
        for visuals in self._marker_visuals.values():
            for visual in visuals:
                visual.parent = None
        self._marker_visuals = {}

    def _clear_marker_visuals_for(self, ref):
        visuals = self._marker_visuals.pop(ref, [])
        for visual in visuals:
            visual.parent = None

    def _marker_text_config(self, lane, x0, x1, y_global):
        props = lane.rec.properties
        side = props.get("gel_label_side", self._label_side)
        off = float(props.get("gel_label_offset", self._label_offset))
        if side == "right":
            return "left", "center", (x1 + off, y_global, 0)
        if side == "top":
            return "center", "bottom", ((x0 + x1) / 2, y_global - off, 0)
        return "right", "center", (x0 - off, y_global, 0)

    def _refresh_marker_visuals(self, layer=None, shape_id=None):
        window = self._active_window()
        if not window:
            return

        if layer is not None and shape_id is not None:
            ref = ShapeLaneRef(layer.name, int(shape_id))
            if shape_id in layer.data:
                lane = ShapeLane(window, layer, shape_id)
                if lane.rec.properties.get(GEL_LANE_KEY):
                    self._sync_marker_visuals_for_lane(window, lane)
                else:
                    self._clear_marker_visuals_for(ref)
            else:
                self._clear_marker_visuals_for(ref)
            window.canvas.update()
            return

        self._clear_marker_visuals()
        for lane in self._iter_lanes(window, marked_only=True):
            self._create_marker_visuals_for_lane(window, lane)
        window.canvas.update()

    def _create_marker_visuals_for_lane(self, window, lane):
        markers = lane.markers
        if not markers:
            return
        show_labels = self.show_labels_check.isChecked()
        x0, y0, x1, _y1 = lane.bounds()
        visuals = []
        for marker in markers:
            try:
                y_global = y0 + float(marker.get("y_local", 0.0))
            except (TypeError, ValueError):
                continue
            color = marker.get("color") or lane.rec.properties.get(
                GEL_MARKER_COLOR_KEY, self._sample_color
            )
            line = scene.visuals.Line(
                pos=np.array(
                    [[x0, y_global, 0], [x1, y_global, 0]], dtype=np.float32
                ),
                color=color,
                width=2,
                parent=window.view.scene,
            )
            anchor_x, anchor_y, pos = self._marker_text_config(lane, x0, x1, y_global)
            text = scene.visuals.Text(
                text=str(marker.get("label", "")),
                color=color,
                font_size=int(
                    lane.rec.properties.get(
                        "gel_label_font_size", self._label_font_size
                    )
                ),
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                parent=window.view.scene,
            )
            text.pos = pos
            text.visible = show_labels and bool(marker.get("label", ""))
            visuals.extend([line, text])
        self._marker_visuals[lane.ref] = visuals

    def _sync_marker_visuals_for_lane(self, window, lane):
        markers = lane.markers
        ref = lane.ref
        if not markers:
            self._clear_marker_visuals_for(ref)
            return

        visuals = self._marker_visuals.get(ref)
        if visuals is None or len(visuals) != len(markers) * 2:
            self._clear_marker_visuals_for(ref)
            self._create_marker_visuals_for_lane(window, lane)
            return

        show_labels = self.show_labels_check.isChecked()
        x0, y0, x1, _y1 = lane.bounds()
        font_size = int(
            lane.rec.properties.get("gel_label_font_size", self._label_font_size)
        )
        for marker_idx, marker in enumerate(markers):
            try:
                y_global = y0 + float(marker.get("y_local", 0.0))
            except (TypeError, ValueError):
                continue
            color = marker.get("color") or lane.rec.properties.get(
                GEL_MARKER_COLOR_KEY, self._sample_color
            )
            line = visuals[marker_idx * 2]
            text = visuals[marker_idx * 2 + 1]
            line.set_data(
                pos=np.array(
                    [[x0, y_global, 0], [x1, y_global, 0]], dtype=np.float32
                ),
                color=color,
            )
            anchor_x, anchor_y, pos = self._marker_text_config(lane, x0, x1, y_global)
            text.text = str(marker.get("label", ""))
            text.color = color
            text.font_size = font_size
            text.anchor_x = anchor_x
            text.anchor_y = anchor_y
            text.pos = pos
            text.visible = show_labels and bool(marker.get("label", ""))

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _convert_lanes(self):
        """Tag rectangle shapes as gel lanes without peak detection.

        Lets the user place markers manually via Shift+Click.
        """
        window = self._active_window()
        if not window:
            return

        self._label_side = self.label_side_combo.currentText().lower()
        self._ladder_label_side = self.ladder_label_side_combo.currentText().lower()
        self._label_font_size = self.font_size_spin.value()
        self._label_offset = self.label_offset_spin.value()
        converted = 0
        selected_ref = self._std_ref or self.lane_combo.currentData()
        self._handling_shape_event = True
        try:
            for lane in self._iter_lanes(window):
                is_ladder = lane.ref == selected_ref
                color = self._ladder_color if is_ladder else self._sample_color
                props = lane.rec.properties
                props["gel_label_side"] = (
                    self._ladder_label_side if is_ladder else self._label_side
                )
                props["gel_label_font_size"] = self._label_font_size
                props["gel_label_offset"] = self._label_offset
                props["gel_show_border"] = self.show_borders_check.isChecked()
                lane.mark_as_lane(color)
                converted += 1
        finally:
            self._handling_shape_event = False

        if converted:
            self._refresh_lanes()
            self._refresh_marker_visuals()
            window.canvas.update()
            self.results_text.setText(
                f"Converted {converted} lane(s).\n\n"
                "Shift+Click on a lane to add a marker.\n"
                "Ctrl/Cmd+Click on a marker to remove it.\n"
                "Then click 'Calibrate' to build the calibration curve."
            )
            self.tabs.setCurrentIndex(0)

    def _detect_peaks(self):
        self._clear_markers()
        self._calibration = None
        self.btn_apply.setEnabled(False)

        window = self._active_window()
        if not window:
            self.results_text.setText("No active window")
            return

        self._std_ref = self.lane_combo.currentData()
        if self._std_ref is None:
            self.results_text.setText("No standard lane selected")
            return

        gray = self._get_current_image()
        if gray is None:
            self.results_text.setText("No image data available")
            return

        gray_proc = (gray.max() - gray) if self.invert_check.isChecked() else gray

        prominence = self.prominence_spin.value()
        min_distance = self.distance_spin.value()

        self._label_side = self.label_side_combo.currentText().lower()
        self._ladder_label_side = self.ladder_label_side_combo.currentText().lower()
        self._label_font_size = self.font_size_spin.value()
        self._label_offset = self.label_offset_spin.value()

        self._lanes = list(self._iter_lanes(window))

        if not self._lanes:
            self.results_text.setText("No rectangle shape lanes found")
            return

        total_peaks = 0
        self._handling_shape_event = True
        try:
            for lane in self._lanes:
                region = lane.get_region(gray_proc)
                if region.size == 0:
                    continue

                profile = region.mean(axis=1)
                peaks, _ = find_peaks(
                    profile, prominence=prominence, distance=min_distance
                )

                is_ladder = lane.ref == self._std_ref
                color = self._ladder_color if is_ladder else self._sample_color
                label_side = self._ladder_label_side if is_ladder else self._label_side

                props = lane.rec.properties
                props["gel_label_side"] = label_side
                props["gel_label_font_size"] = self._label_font_size
                props["gel_label_offset"] = self._label_offset
                props["gel_show_border"] = self.show_borders_check.isChecked()

                marker_data = [
                    {"y_local": float(pk), "label": "", "color": color}
                    for pk in peaks
                ]
                lane.mark_as_lane(color)
                lane.set_markers(marker_data)

                total_peaks += len(peaks)
        finally:
            self._handling_shape_event = False

        self._refresh_marker_visuals()
        window.canvas.update()

        self.results_text.setText(
            "\n".join([
                f"Detected {total_peaks} peaks across {len(self._lanes)} lanes.",
                "",
                "Next steps:",
                "  1. Adjust markers if needed (check Profile tab)",
                "  2. Click 'Calibrate' to build calibration from ladder lane",
                "  3. Click 'Apply Calibration' to estimate MW for sample lanes",
            ])
        )

        # Refresh profile for the currently selected lane
        self._update_profile_plot()

    def _calibrate(self):
        window = self._active_window()
        if not window:
            self.results_text.setText("No active window")
            return

        ladder_data = self.ladder_combo.currentData()
        if ladder_data is None:
            self.results_text.setText("No ladder preset selected")
            return

        ladder_sizes = ladder_data.get("sizes", ladder_data.get("weights_kda", []))
        unit = ladder_data.get("unit", "kDa")

        if self._std_ref is None:
            self._std_ref = self.lane_combo.currentData()
        if self._std_ref is None:
            self.results_text.setText("No standard lane selected")
            return

        std_lane = self._resolve_lane(self._std_ref)
        if std_lane is None:
            self.results_text.setText("Standard lane not found")
            return

        std_peaks = std_lane.get_marker_positions()
        if len(std_peaks) == 0:
            self.results_text.setText("No markers in standard lane")
            return

        n_peaks = min(len(std_peaks), len(ladder_sizes))
        if n_peaks < 2:
            self.results_text.setText(
                f"Need at least 2 markers in standard lane (found {len(std_peaks)})"
            )
            return

        std_positions = np.array(std_peaks[:n_peaks])
        log_sizes = np.log(np.array(ladder_sizes[:n_peaks]))

        self._calibration = {
            "std_positions": std_positions,
            "log_sizes": log_sizes,
            "unit": unit,
            "ladder_sizes": ladder_sizes[:n_peaks],
        }

        labels = [f"{sz} {unit}" for sz in ladder_sizes[:n_peaks]]
        labels.extend([""] * (len(std_peaks) - n_peaks))
        self._handling_shape_event = True
        std_lane.update_marker_labels(labels)
        self._handling_shape_event = False

        self.btn_apply.setEnabled(True)

        self.results_text.setText(
            "\n".join([
                "Calibration created!",
                f"  Standard Lane: {self._lane_label(std_lane)}",
                f"  Using {n_peaks} bands: "
                f"{', '.join(str(s) for s in ladder_sizes[:n_peaks])} {unit}",
                "",
                "Click 'Apply Calibration' to estimate MW for sample lanes.",
            ])
        )
        self._refresh_marker_visuals()
        window.canvas.update()

    def _apply_calibration(self):
        if self._calibration is None:
            self.results_text.setText(
                "No calibration available. Click 'Calibrate' first."
            )
            return

        window = self._active_window()
        if not window:
            self.results_text.setText("No active window")
            return

        std_positions = self._calibration["std_positions"]
        log_sizes = self._calibration["log_sizes"]
        unit = self._calibration["unit"]
        std_lane = self._resolve_lane(self._std_ref)
        std_label = self._lane_label(std_lane) if std_lane is not None else "standard"

        results = [
            f"Standard Lane ({std_label}):",
            f"  Calibration: {len(std_positions)} bands",
            "",
        ]

        self._handling_shape_event = True
        try:
            for lane in self._iter_lanes(window, marked_only=True):
                if lane.ref == self._std_ref:
                    continue

                peaks = lane.get_marker_positions()
                if len(peaks) == 0:
                    results.append(f"Lane {self._lane_label(lane)}: No markers")
                    continue

                log_mw = np.interp(peaks, std_positions, log_sizes)
                mw = np.exp(log_mw)

                lane.update_marker_labels([f"~{w:.1f} {unit}" for w in mw])

                results.append(f"Lane {self._lane_label(lane)}:")
                results.append(
                    f"  Size ({unit}): {', '.join(f'{w:.1f}' for w in mw)}"
                )
        finally:
            self._handling_shape_event = False

        self.results_text.setText("\n".join(results))
        self._refresh_marker_visuals()
        window.canvas.update()

    def _on_markers_adjusted(self):
        if self._calibration is not None:
            self._apply_calibration()
        self._update_profile_plot()

    def _clear_markers(self):
        window = self._active_window()
        if not window:
            return
        self._handling_shape_event = True
        try:
            for lane in self._iter_lanes(window, marked_only=True):
                lane.clear_markers()
        finally:
            self._handling_shape_event = False
        self._lanes = []
        self._std_ref = None
        self._calibration = None
        self.btn_apply.setEnabled(False)
        self._clear_marker_visuals()
        window.canvas.update()
        self.profile_widget.clear()
        self.profile_status.setText("Select a lane to preview")

    # ------------------------------------------------------------------
    # Results export
    # ------------------------------------------------------------------

    def _copy_results(self):
        from qtpy.QtWidgets import QApplication
        QApplication.clipboard().setText(self.results_text.toPlainText())

    def _export_csv(self):
        text = self.results_text.toPlainText()
        if not text:
            return

        ladder_data = self.ladder_combo.currentData()
        unit = ladder_data.get("unit", "kDa") if ladder_data else "kDa"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "gel_analysis.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        lines = text.strip().split("\n")
        with open(path, "w") as f:
            f.write(f"Lane,Shape_Name,Size_{unit}\n")
            current_lane = ""
            current_name = ""
            for line in lines:
                line = line.strip()
                if line.startswith("Lane"):
                    parts = line.rstrip(":").split(" ", 2)
                    if len(parts) >= 2:
                        current_lane = parts[1]
                        if len(parts) > 2 and "(" in parts[2]:
                            current_name = parts[2].strip("():")
                elif line.startswith("Size"):
                    mw_part = line.split(": ", 1)
                    if len(mw_part) == 2:
                        for mw in mw_part[1].split(", "):
                            f.write(f"{current_lane},{current_name},{mw}\n")

        print(f"Results exported to {path}")


# Singleton instance
_gel_analyzer_instance = None


def get_gel_analyzer(window_manager=None):
    global _gel_analyzer_instance
    if _gel_analyzer_instance is None:
        _gel_analyzer_instance = GelAnalyzerWidget(window_manager)
    return _gel_analyzer_instance


def show_gel_analyzer(window_manager=None):
    analyzer = get_gel_analyzer(window_manager)
    analyzer.show()
    analyzer.raise_()
    analyzer.activateWindow()
    return analyzer
