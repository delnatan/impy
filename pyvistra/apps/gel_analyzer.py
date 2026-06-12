"""
Gel Image Analyzer Widget

Provides molecular weight/size estimation from gel electrophoresis images.
Works with both protein gels (kDa) and DNA gels (bp).
Uses LaneROI to define lanes with integrated peak markers.
"""

import json
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

from ..rois import LaneROI, RectangleROI
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

    Detects bands in lanes (LaneROI), uses one lane as a
    molecular weight ladder, and estimates MW of bands in other lanes.
    Supports both protein (kDa) and DNA (bp) gels.
    """

    def __init__(self, annotation_manager):
        super().__init__()
        self.annotation_manager = annotation_manager
        self.ladders = load_ladders()
        self._lane_rois = []
        self._std_idx = None
        self._calibration = None

        # Appearance settings
        self._ladder_color = "#FFBB16"
        self._sample_color = "#00BCD4"
        self._label_font_size = 8
        self._label_offset = 3
        self._label_side = "top"
        self._ladder_label_side = "left"

        self._connected_window = None   # window whose roi_selection_changed we're watching

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
            "Convert rectangle ROIs to Lane ROIs without peak detection.\n"
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
        self._connect_to_window(self.annotation_manager.active_window)

    def closeEvent(self, event):
        self._disconnect_from_window()
        self._clear_markers()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Window connection (for live profile updates on ROI click)
    # ------------------------------------------------------------------

    def _connect_to_window(self, window):
        if window is self._connected_window:
            return
        self._disconnect_from_window()
        if window is not None:
            window.roi_selection_changed.connect(self._on_roi_selected)
            self._connected_window = window

    def _disconnect_from_window(self):
        if self._connected_window is not None:
            try:
                self._connected_window.roi_selection_changed.disconnect(
                    self._on_roi_selected
                )
            except (TypeError, RuntimeError):
                pass
            self._connected_window = None

    def _on_roi_selected(self, roi):
        """Sync lane combo and refresh profile when a lane ROI is clicked."""
        if not isinstance(roi, (LaneROI, RectangleROI)):
            return
        window = self.annotation_manager.active_window
        if not window:
            return
        try:
            roi_idx = window.rois.index(roi)
        except ValueError:
            return
        for i in range(self.lane_combo.count()):
            if self.lane_combo.itemData(i) == roi_idx:
                if self.lane_combo.currentIndex() == i:
                    # Same lane re-clicked — signal won't fire, force refresh
                    self._update_profile_plot()
                else:
                    self.lane_combo.setCurrentIndex(i)  # triggers via signal
                self.tabs.setCurrentIndex(2)  # switch to Profile tab
                return
        # ROI not yet in combo (e.g. just converted); update profile directly
        self._update_profile_plot()
        self.tabs.setCurrentIndex(2)

    # ------------------------------------------------------------------
    # Profile plot
    # ------------------------------------------------------------------

    def _update_profile_plot(self):
        """Refresh the Profile tab for the currently selected lane."""
        window = self.annotation_manager.active_window
        if not window:
            self.profile_widget.clear()
            self.profile_status.setText("No active window")
            return

        roi_idx = self.lane_combo.currentData()
        if roi_idx is None or roi_idx >= len(window.rois):
            self.profile_widget.clear()
            self.profile_status.setText("No lane selected")
            return

        roi = window.rois[roi_idx]
        if not isinstance(roi, (RectangleROI, LaneROI)):
            self.profile_widget.clear()
            return

        gray = self._get_current_image()
        if gray is None:
            self.profile_widget.clear()
            self.profile_status.setText("No image data")
            return

        gray_proc = (gray.max() - gray) if self.invert_check.isChecked() else gray

        region = roi.get_region(gray_proc)
        if region.size == 0:
            self.profile_widget.clear()
            return

        profile = region.mean(axis=1)

        # Collect detected peaks from markers if already run
        peaks = []
        if isinstance(roi, LaneROI) and roi._markers:
            peaks = [int(round(m["y_local"])) for m in roi._markers]

        is_ladder = roi_idx == self._std_idx
        color = self._ladder_color if is_ladder else self._sample_color

        self.profile_widget.set_data(
            profile, peaks or None, profile_color=color, peak_color="#FF6B35"
        )

        status = f"Lane {roi_idx} ({roi.name})  ·  {len(profile)} px"
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

        window = self.annotation_manager.active_window
        if not window:
            return

        show_labels = self.show_labels_check.isChecked()
        show_borders = self.show_borders_check.isChecked()

        for i, roi in enumerate(window.rois):
            if not isinstance(roi, LaneROI):
                continue
            is_ladder = i == self._std_idx
            color = self._ladder_color if is_ladder else self._sample_color
            label_side = self._ladder_label_side if is_ladder else self._label_side

            for m in roi._markers:
                m["color"] = color

            roi.label_side = label_side
            roi.label_font_size = self._label_font_size
            roi.label_offset = self._label_offset
            roi.rebuild_marker_visuals()
            roi.set_marker_labels_visible(show_labels)
            roi.show_border = show_borders

        window.canvas.update()
        self._update_profile_plot()

    def _on_show_labels_changed(self):
        visible = self.show_labels_check.isChecked()
        window = self.annotation_manager.active_window
        if window:
            for roi in window.rois:
                if isinstance(roi, LaneROI):
                    roi.set_marker_labels_visible(visible)
            window.canvas.update()

    def _on_show_borders_changed(self):
        show = self.show_borders_check.isChecked()
        window = self.annotation_manager.active_window
        if window:
            for roi in window.rois:
                if isinstance(roi, LaneROI):
                    roi.show_border = show
            window.canvas.update()

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

    def _refresh_lanes(self):
        self.lane_combo.clear()
        window = self.annotation_manager.active_window
        if not window:
            return
        for i, roi in enumerate(window.rois):
            if isinstance(roi, (RectangleROI, LaneROI)):
                self.lane_combo.addItem(f"{i}: {roi.name}", userData=i)

    def _get_current_image(self):
        window = self.annotation_manager.active_window
        if not window:
            return None
        cache = window.renderer.current_slice_cache
        if cache is None:
            return None
        return cache.mean(axis=0) if cache.ndim == 3 else cache

    def _convert_to_lane_roi(self, window, roi_idx, roi):
        if isinstance(roi, LaneROI):
            return roi
        lane = LaneROI(window.view, name=roi.name)
        lane.update(roi.data["p1"], roi.data["p2"])
        window.rois[roi_idx] = lane
        roi.remove()
        window.roi_removed.emit(roi)
        window.roi_added.emit(lane)
        return lane

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _convert_lanes(self):
        """Convert all rectangle ROIs to LaneROIs without peak detection.

        Lets the user place markers manually via Shift+Click.
        """
        window = self.annotation_manager.active_window
        if not window:
            return

        self._label_side = self.label_side_combo.currentText().lower()
        self._ladder_label_side = self.ladder_label_side_combo.currentText().lower()
        self._label_font_size = self.font_size_spin.value()
        self._label_offset = self.label_offset_spin.value()
        show_labels = self.show_labels_check.isChecked()
        show_borders = self.show_borders_check.isChecked()

        converted = 0
        for i, roi in enumerate(window.rois):
            if not isinstance(roi, (RectangleROI, LaneROI)):
                continue
            lane = self._convert_to_lane_roi(window, i, roi)
            is_ladder = i == (self._std_idx or self.lane_combo.currentData())
            lane.label_side = self._ladder_label_side if is_ladder else self._label_side
            lane.label_font_size = self._label_font_size
            lane.label_offset = self._label_offset
            lane.set_marker_labels_visible(show_labels)
            lane.show_border = show_borders
            lane.set_markers_changed_callback(self._on_markers_adjusted)
            converted += 1

        if converted:
            self._refresh_lanes()
            window.canvas.update()
            self.annotation_manager.refresh_list()
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

        window = self.annotation_manager.active_window
        if not window:
            self.results_text.setText("No active window")
            return

        self._std_idx = self.lane_combo.currentData()
        if self._std_idx is None:
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

        self._lane_rois = []
        roi_indices = []

        for i, roi in enumerate(window.rois):
            if isinstance(roi, (RectangleROI, LaneROI)):
                lane = self._convert_to_lane_roi(window, i, roi)
                self._lane_rois.append(lane)
                roi_indices.append(i)

        if not self._lane_rois:
            self.results_text.setText("No rectangle/lane ROIs found")
            return

        total_peaks = 0
        show_labels = self.show_labels_check.isChecked()
        show_borders = self.show_borders_check.isChecked()

        for lane, roi_idx in zip(self._lane_rois, roi_indices):
            region = lane.get_region(gray_proc)
            if region.size == 0:
                continue

            profile = region.mean(axis=1)
            peaks, _ = find_peaks(
                profile, prominence=prominence, distance=min_distance
            )

            is_ladder = roi_idx == self._std_idx
            color = self._ladder_color if is_ladder else self._sample_color
            label_side = self._ladder_label_side if is_ladder else self._label_side

            lane.label_side = label_side
            lane.label_font_size = self._label_font_size
            lane.label_offset = self._label_offset

            marker_data = [
                {"y_local": float(pk), "label": "", "color": color}
                for pk in peaks
            ]
            lane.set_markers(marker_data)
            lane.locked = True
            lane.set_markers_changed_callback(self._on_markers_adjusted)
            lane.set_marker_labels_visible(show_labels)
            lane.show_border = show_borders

            total_peaks += len(peaks)

        window.canvas.update()
        self.annotation_manager.refresh_list()

        self.results_text.setText(
            "\n".join([
                f"Detected {total_peaks} peaks across {len(self._lane_rois)} lanes.",
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
        window = self.annotation_manager.active_window
        if not window:
            self.results_text.setText("No active window")
            return

        ladder_data = self.ladder_combo.currentData()
        if ladder_data is None:
            self.results_text.setText("No ladder preset selected")
            return

        ladder_sizes = ladder_data.get("sizes", ladder_data.get("weights_kda", []))
        unit = ladder_data.get("unit", "kDa")

        if self._std_idx is None:
            self._std_idx = self.lane_combo.currentData()
        if self._std_idx is None:
            self.results_text.setText("No standard lane selected")
            return

        std_lane = None
        for i, roi in enumerate(window.rois):
            if i == self._std_idx and isinstance(roi, LaneROI):
                std_lane = roi
                break

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
        std_lane.update_marker_labels(labels)

        self.btn_apply.setEnabled(True)

        self.results_text.setText(
            "\n".join([
                "Calibration created!",
                f"  Standard Lane: ROI {self._std_idx}",
                f"  Using {n_peaks} bands: "
                f"{', '.join(str(s) for s in ladder_sizes[:n_peaks])} {unit}",
                "",
                "Click 'Apply Calibration' to estimate MW for sample lanes.",
            ])
        )
        window.canvas.update()

    def _apply_calibration(self):
        if self._calibration is None:
            self.results_text.setText(
                "No calibration available. Click 'Calibrate' first."
            )
            return

        window = self.annotation_manager.active_window
        if not window:
            self.results_text.setText("No active window")
            return

        std_positions = self._calibration["std_positions"]
        log_sizes = self._calibration["log_sizes"]
        unit = self._calibration["unit"]

        results = [
            f"Standard Lane (ROI {self._std_idx}):",
            f"  Calibration: {len(std_positions)} bands",
            "",
        ]

        for i, roi in enumerate(window.rois):
            if not isinstance(roi, LaneROI) or i == self._std_idx:
                continue

            peaks = roi.get_marker_positions()
            if len(peaks) == 0:
                results.append(f"Lane {i} ({roi.name}): No markers")
                continue

            log_mw = np.interp(peaks, std_positions, log_sizes)
            mw = np.exp(log_mw)

            roi.update_marker_labels([f"~{w:.1f} {unit}" for w in mw])

            results.append(f"Lane {i} ({roi.name}):")
            results.append(f"  Size ({unit}): {', '.join(f'{w:.1f}' for w in mw)}")

        self.results_text.setText("\n".join(results))
        window.canvas.update()

    def _on_markers_adjusted(self):
        if self._calibration is not None:
            self._apply_calibration()
        self._update_profile_plot()

    def _clear_markers(self):
        window = self.annotation_manager.active_window
        if not window:
            return
        for roi in window.rois:
            if isinstance(roi, LaneROI):
                roi.clear_markers()
                roi.locked = False
                roi.show_border = True
        self._lane_rois = []
        self._std_idx = None
        self._calibration = None
        self.btn_apply.setEnabled(False)
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
            f.write(f"Lane,ROI_Name,Size_{unit}\n")
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


def get_gel_analyzer(annotation_manager):
    global _gel_analyzer_instance
    if _gel_analyzer_instance is None:
        _gel_analyzer_instance = GelAnalyzerWidget(annotation_manager)
    return _gel_analyzer_instance


def show_gel_analyzer(annotation_manager):
    analyzer = get_gel_analyzer(annotation_manager)
    analyzer.show()
    analyzer.raise_()
    analyzer.activateWindow()
