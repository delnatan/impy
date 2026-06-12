"""
Line Profile Dialog - Displays and compares intensity profiles along LineROIs.

Supports multi-window overlay plotting and CSV/TSV export.
"""

import csv
from collections import OrderedDict

import numpy as np
from qtpy.QtCore import Qt, QPointF
from qtpy.QtGui import QColor, QPainter, QPen, QFont, QPolygonF
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .histogram import WIDGET_BG, TEXT_COLOR
from ..ui.manager import manager
from ..rois import LineROI

# Singleton instance
_line_profile_dialog = None


FALLBACK_COLORS = [
    "#66CCFF",
    "#FF9966",
    "#99FF99",
    "#FFCC66",
    "#CC99FF",
    "#FF6699",
    "#66FFCC",
    "#CCCCCC",
]


def get_line_profile_dialog():
    """Get or create the singleton LineProfileDialog instance."""
    global _line_profile_dialog
    if _line_profile_dialog is None:
        _line_profile_dialog = LineProfileDialog()
    return _line_profile_dialog


def line_profile_dialog_exists():
    """Check if the line profile dialog has been created."""
    return _line_profile_dialog is not None


class LineProfileWidget(QWidget):
    """Custom widget for drawing overlaid intensity profiles with QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setMinimumWidth(360)

        # List of dicts: label, color, distances, values, visible
        self.series = []

        # Plot margins
        self.margin_left = 56
        self.margin_right = 15
        self.margin_top = 15
        self.margin_bottom = 30

    def set_series(self, series):
        """Update series data and trigger repaint."""
        self.series = list(series) if series else []
        self.update()

    def clear(self):
        """Clear all profile data."""
        self.series = []
        self.update()

    def _get_plot_rect(self):
        x = self.margin_left
        y = self.margin_top
        w = self.width() - self.margin_left - self.margin_right
        h = self.height() - self.margin_top - self.margin_bottom
        return x, y, max(1, w), max(1, h)

    @staticmethod
    def _x_to_px(x_val, x_min, x_max, plot_x, plot_w):
        if x_max == x_min:
            return plot_x
        ratio = (x_val - x_min) / (x_max - x_min)
        return plot_x + ratio * plot_w

    @staticmethod
    def _y_to_px(y_val, y_min, y_max, plot_y, plot_h):
        if y_max == y_min:
            return plot_y + plot_h / 2
        ratio = (y_val - y_min) / (y_max - y_min)
        return plot_y + plot_h - ratio * plot_h

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), WIDGET_BG)

        visible_series = [s for s in self.series if s.get("visible", True)]
        if not visible_series:
            painter.setPen(TEXT_COLOR)
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Select a LineROI and add one or more windows to compare",
            )
            return

        plot_x, plot_y, plot_w, plot_h = self._get_plot_rect()

        x_min = min(float(np.nanmin(s["distances"])) for s in visible_series)
        x_max = max(float(np.nanmax(s["distances"])) for s in visible_series)

        y_arrays = []
        for s in visible_series:
            vals = np.asarray(s["values"], dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size > 0:
                y_arrays.append(finite)

        if y_arrays:
            all_visible = np.concatenate(y_arrays)
            y_min = float(np.min(all_visible))
            y_max = float(np.max(all_visible))
        else:
            y_min, y_max = 0.0, 1.0

        y_range = y_max - y_min
        if y_range > 0:
            y_min -= y_range * 0.05
            y_max += y_range * 0.05
        else:
            y_min -= 0.5
            y_max += 0.5

        self._draw_axes(
            painter, plot_x, plot_y, plot_w, plot_h, x_min, x_max, y_min, y_max
        )

        for idx, s in enumerate(visible_series):
            self._draw_series(
                painter,
                s,
                idx,
                plot_x,
                plot_y,
                plot_w,
                plot_h,
                x_min,
                x_max,
                y_min,
                y_max,
            )

    def _draw_axes(self, painter, plot_x, plot_y, plot_w, plot_h, x_min, x_max, y_min, y_max):
        border_pen = QPen(QColor(80, 80, 80))
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.drawRect(plot_x, plot_y, plot_w, plot_h)

        grid_pen = QPen(QColor(60, 60, 60))
        grid_pen.setStyle(Qt.DotLine)
        painter.setPen(grid_pen)

        for i in range(1, 4):
            y = plot_y + (i / 4) * plot_h
            painter.drawLine(int(plot_x), int(y), int(plot_x + plot_w), int(y))

        for i in range(1, 4):
            x = plot_x + (i / 4) * plot_w
            painter.drawLine(int(x), int(plot_y), int(x), int(plot_y + plot_h))

        painter.setPen(TEXT_COLOR)
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        y_labels = [y_min, (y_min + y_max) / 2, y_max]
        for y_val in y_labels:
            y_px = self._y_to_px(y_val, y_min, y_max, plot_y, plot_h)
            label = self._format_value(y_val)
            painter.drawText(
                2,
                int(y_px - 6),
                self.margin_left - 6,
                12,
                Qt.AlignRight | Qt.AlignVCenter,
                label,
            )

        x_labels = [x_min, (x_min + x_max) / 2, x_max]
        for x_val in x_labels:
            x_px = self._x_to_px(x_val, x_min, x_max, plot_x, plot_w)
            label = f"{x_val:.0f}"
            painter.drawText(
                int(x_px - 25),
                int(plot_y + plot_h + 5),
                50,
                20,
                Qt.AlignCenter,
                label,
            )

        painter.drawText(
            int(plot_x + plot_w / 2 - 45),
            int(plot_y + plot_h + 18),
            90,
            15,
            Qt.AlignCenter,
            "Distance (px)",
        )

    def _draw_series(
        self,
        painter,
        series,
        series_idx,
        plot_x,
        plot_y,
        plot_w,
        plot_h,
        x_min,
        x_max,
        y_min,
        y_max,
    ):
        qcolor = _to_qcolor(series.get("color", "#FFFFFF"), QColor(255, 255, 255))

        pen = QPen(qcolor)
        pen.setWidth(2)
        if series_idx > 0:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)

        distances = np.asarray(series["distances"], dtype=float)
        values = np.asarray(series["values"], dtype=float)

        points = []
        for d, v in zip(distances, values):
            if not np.isfinite(v):
                if len(points) > 1:
                    painter.drawPolyline(QPolygonF(points))
                points = []
                continue
            x_px = self._x_to_px(d, x_min, x_max, plot_x, plot_w)
            y_px = self._y_to_px(v, y_min, y_max, plot_y, plot_h)
            points.append(QPointF(x_px, y_px))

        if len(points) > 1:
            painter.drawPolyline(QPolygonF(points))

    @staticmethod
    def _format_value(value):
        if abs(value) < 0.01 or abs(value) >= 10000:
            return f"{value:.1e}"
        if abs(value) < 1:
            return f"{value:.2f}"
        if abs(value) < 100:
            return f"{value:.1f}"
        return f"{value:.0f}"


class LineProfileDialog(QDialog):
    """
    Floating dialog that displays and compares intensity profiles along LineROIs.

    Use the selected LineROI as source geometry, then add target windows to
    overlay profiles in one plot.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Line Profile")
        self.setWindowFlags(Qt.Tool)
        self.resize(760, 460)

        self.active_window = None
        self.source_window = None
        self.current_roi = None
        self.current_line_data = None  # {p1: (x,y), p2: (x,y)}

        # When the profile source is a shape-layer record, we subscribe to
        # the layer's data so handle/body edits refresh the profile live.
        self._source_shape_layer = None
        self._source_shape_id = None
        self._source_shape_unsub = None

        # wid -> {window, channel, visible, label, color}
        self.series_config = OrderedDict()
        self._computed_series = []

        self._connected_windows = set()
        self._is_shutting_down = False

        self._setup_ui()

        manager.window_registered.connect(self._on_window_registered)
        for window in manager.get_all().values():
            self._connect_window(window)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        controls = QHBoxLayout()

        btn_add_active = QPushButton("Add Active Window")
        btn_add_active.clicked.connect(self._add_active_window_series)
        controls.addWidget(btn_add_active)

        btn_add_window = QPushButton("Add Window...")
        btn_add_window.clicked.connect(self._add_window_series_dialog)
        controls.addWidget(btn_add_window)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._remove_selected_series)
        controls.addWidget(btn_remove)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_series)
        controls.addWidget(btn_clear)

        controls.addStretch()

        btn_export = QPushButton("Export...")
        btn_export.clicked.connect(self._export_profiles)
        controls.addWidget(btn_export)

        layout.addLayout(controls)

        layout.addWidget(QLabel("Compared Series:"))
        self.series_list = QListWidget()
        self.series_list.itemChanged.connect(self._on_series_item_changed)
        self.series_list.itemSelectionChanged.connect(self._on_series_selection_changed)
        layout.addWidget(self.series_list, 0)

        channel_row = QHBoxLayout()
        self.all_channels_cb = QCheckBox("All Channels")
        self.all_channels_cb.setChecked(False)
        self.all_channels_cb.toggled.connect(self._on_all_channels_toggled)
        channel_row.addWidget(self.all_channels_cb)
        channel_row.addWidget(QLabel("Channel:"))
        self.series_channel_spin = QSpinBox()
        self.series_channel_spin.setRange(1, 1)
        self.series_channel_spin.setEnabled(False)
        self.series_channel_spin.valueChanged.connect(self._on_selected_channel_changed)
        channel_row.addWidget(self.series_channel_spin)
        channel_row.addStretch()
        layout.addLayout(channel_row)

        self.profile_widget = LineProfileWidget()
        layout.addWidget(self.profile_widget, 1)

        self.status_label = QLabel("Select a LineROI to start")
        self.status_label.setStyleSheet("color: #AAA; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _on_window_registered(self, window):
        if self._is_shutting_down:
            return
        self._connect_window(window)

    def _connect_window(self, window):
        if window in self._connected_windows:
            return

        window.window_activated.connect(self._on_window_activated)
        window.window_closing.connect(self._on_window_closing)
        window.roi_selection_changed.connect(self._on_roi_selection_changed)
        if hasattr(window, "roi_modified"):
            window.roi_modified.connect(self._on_roi_modified)
        if hasattr(window, "view_changed"):
            window.view_changed.connect(self._on_view_changed)

        self._connected_windows.add(window)

    def _disconnect_window(self, window):
        if window not in self._connected_windows:
            return

        try:
            window.window_activated.disconnect(self._on_window_activated)
            window.window_closing.disconnect(self._on_window_closing)
            window.roi_selection_changed.disconnect(self._on_roi_selection_changed)
            if hasattr(window, "roi_modified"):
                window.roi_modified.disconnect(self._on_roi_modified)
            if hasattr(window, "view_changed"):
                window.view_changed.disconnect(self._on_view_changed)
        except (TypeError, RuntimeError):
            pass

        self._connected_windows.discard(window)

    def _on_window_activated(self, window):
        if self._is_shutting_down:
            return
        self.active_window = window

        for roi in window.rois:
            if roi.selected and isinstance(roi, LineROI):
                self._update_profile(roi)
                return

        if self.current_line_data is not None:
            self._refresh_profiles()

    def _on_window_closing(self, window):
        self._disconnect_window(window)

        wid = getattr(window, "window_id", None)
        if wid in self.series_config:
            self.series_config.pop(wid, None)
            self._refresh_series_list()
            self._refresh_profiles()

        if window == self.active_window:
            self.active_window = None
        if window == self.source_window:
            self.source_window = None
            self.current_roi = None
            self._unsubscribe_from_shape_source()

    def _on_roi_selection_changed(self, roi):
        if self._is_shutting_down:
            return

        if roi is None or not isinstance(roi, LineROI):
            if self.current_line_data is None:
                self.profile_widget.clear()
                self.status_label.setText("Select a LineROI to start")
            return

        self._update_profile(roi)

    def set_shape_source(self, window, layer, shape_id):
        """Use a shape-layer LINE/POLYLINE record as the profile source.

        Equivalent to selecting a legacy ``LineROI`` in the source window,
        but driven by the unified shape layer system. The dialog refreshes
        immediately and adds the source window's series if not present, and
        subscribes to the layer so the profile updates live as the shape is
        edited (handle drag, body move, vertex insert/remove).
        """
        if shape_id not in layer.data:
            return
        if not self._extract_shape_line_data(layer, shape_id):
            return

        self.source_window = window
        self.active_window = window
        self.current_roi = None

        self._subscribe_to_shape_source(layer, shape_id)
        self._ensure_source_series()
        self._refresh_profiles()

    def _extract_shape_line_data(self, layer, shape_id):
        """Populate ``self.current_line_data`` from a shape record.

        Returns True if the shape is a LINE/POLYLINE and data was set.
        """
        from ..data.shapes import LINE as _LINE, POLYLINE as _POLYLINE
        if shape_id not in layer.data:
            return False
        rec = layer.data.get(shape_id)
        if rec.shape_type == _LINE:
            p = rec.params
            self.current_line_data = {
                "p1": (float(p[0]), float(p[1])),
                "p2": (float(p[2]), float(p[3])),
                "path": None,
            }
            return True
        if rec.shape_type == _POLYLINE and rec.vertices is not None:
            from .kymograph_dialog import _sample_coords, _raw_pixel_length

            length = max(2.0, _raw_pixel_length(rec))
            path = _sample_coords(rec, max(2, int(np.ceil(length))))
            self.current_line_data = {
                "p1": (float(path[0, 0]), float(path[0, 1])),
                "p2": (float(path[-1, 0]), float(path[-1, 1])),
                "path": np.asarray(path, dtype=float),
            }
            return True
        return False

    def _subscribe_to_shape_source(self, layer, shape_id):
        """Subscribe to ``layer.data`` so edits to ``shape_id`` refresh the
        profile. Replaces any previous subscription.
        """
        from ..data.shapes import EVT_EDITED, EVT_REMOVED, EVT_BULK

        self._unsubscribe_from_shape_source()
        self._source_shape_layer = layer
        self._source_shape_id = shape_id

        def _on_shape_event(event_kind, sid):
            if self._is_shutting_down:
                return
            if sid != self._source_shape_id and event_kind != EVT_BULK:
                return
            if event_kind == EVT_REMOVED:
                self._unsubscribe_from_shape_source()
                self.current_line_data = None
                self.profile_widget.clear()
                self.status_label.setText("Select a LineROI to start")
                return
            if event_kind in (EVT_EDITED, EVT_BULK):
                if self._extract_shape_line_data(
                    self._source_shape_layer, self._source_shape_id
                ):
                    self._refresh_profiles()

        self._source_shape_unsub = layer.data.subscribe(_on_shape_event)

    def _unsubscribe_from_shape_source(self):
        if self._source_shape_unsub is not None:
            try:
                self._source_shape_unsub()
            except Exception:
                pass
        self._source_shape_unsub = None
        self._source_shape_layer = None
        self._source_shape_id = None

    def _on_roi_modified(self, roi):
        if self._is_shutting_down:
            return

        if roi == self.current_roi and isinstance(roi, LineROI):
            self._update_profile(roi)

    def _on_view_changed(self, window):
        if self._is_shutting_down or self.current_line_data is None:
            return

        wid = getattr(window, "window_id", None)
        source_wid = getattr(self.source_window, "window_id", None)
        if wid in self.series_config or wid == source_wid:
            self._refresh_profiles()

    def _update_profile(self, roi):
        self.current_roi = roi
        if self.active_window is not None:
            self.source_window = self.active_window

        # A legacy LineROI took over as the source; drop any shape-layer
        # subscription so its edits don't fight ours.
        self._unsubscribe_from_shape_source()

        p1 = roi.data.get("p1", (0, 0))
        p2 = roi.data.get("p2", (0, 0))
        self.current_line_data = {
            "p1": (float(p1[0]), float(p1[1])),
            "p2": (float(p2[0]), float(p2[1])),
            "path": None,
        }

        self._ensure_source_series()
        self._refresh_profiles()

    def _ensure_source_series(self):
        if self.source_window is None:
            return
        self._add_series_for_window(self.source_window)

    def _add_active_window_series(self):
        if self.active_window is None:
            self.status_label.setText("No active window")
            return
        self._add_series_for_window(self.active_window)
        self._refresh_profiles()

    def _add_window_series_dialog(self):
        windows = []
        labels = []
        for win in manager.get_all().values():
            if hasattr(win, "roi_added"):
                windows.append(win)
                labels.append(f"[{win.window_id}] {win.windowTitle()}")

        if not windows:
            self.status_label.setText("No image windows available")
            return

        item, ok = QInputDialog.getItem(
            self,
            "Add Window",
            "Select window to compare:",
            labels,
            0,
            False,
        )
        if not ok or not item:
            return

        idx = labels.index(item)
        self._add_series_for_window(windows[idx])
        self._refresh_profiles()

    def _add_series_for_window(self, window, channel_idx=None):
        wid = window.window_id

        if wid in self.series_config:
            return

        if channel_idx is None:
            channel_idx = int(getattr(window, "c_idx", 0))

        num_channels = int(getattr(window, "C", 1))
        channel_idx = int(np.clip(channel_idx, 0, max(0, num_channels - 1)))

        color = self._get_window_channel_color(window, channel_idx, len(self.series_config))

        self.series_config[wid] = {
            "window": window,
            "channel": channel_idx,
            "visible": True,
            "label": f"[{window.window_id}] {window.windowTitle()}",
            "color": color,
        }
        self._refresh_series_list(select_wid=wid)

    def _remove_selected_series(self):
        item = self.series_list.currentItem()
        if item is None:
            return

        wid = item.data(Qt.UserRole)
        self.series_config.pop(wid, None)

        self._refresh_series_list()
        self._refresh_profiles()

    def _clear_series(self):
        self.series_config.clear()
        self._computed_series = []
        self.current_roi = None
        self.current_line_data = None
        self.source_window = None
        self._unsubscribe_from_shape_source()

        self._refresh_series_list()
        self.profile_widget.clear()
        self.status_label.setText("Select a LineROI to start")

    def _refresh_series_list(self, select_wid=None):
        selected_wid = None
        current_item = self.series_list.currentItem()
        if current_item is not None:
            selected_wid = current_item.data(Qt.UserRole)

        self.series_list.blockSignals(True)
        self.series_list.clear()

        all_ch = self.all_channels_cb.isChecked()
        for wid, cfg in self.series_config.items():
            label = cfg.get("label", f"[{wid}] Window")
            if all_ch:
                n_ch = self._num_channels(cfg.get("window")) if cfg.get("window") else 1
                ch_text = f"All ({n_ch}ch)"
            else:
                ch_text = f"Ch{int(cfg.get('channel', 0)) + 1}"
            item = QListWidgetItem(f"{label} | {ch_text}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if cfg.get("visible", True) else Qt.Unchecked)
            item.setData(Qt.UserRole, wid)
            self.series_list.addItem(item)

        target_wid = select_wid if select_wid is not None else selected_wid
        if target_wid is not None:
            for i in range(self.series_list.count()):
                item = self.series_list.item(i)
                if item.data(Qt.UserRole) == target_wid:
                    self.series_list.setCurrentItem(item)
                    break

        self.series_list.blockSignals(False)
        self._on_series_selection_changed()

    def _on_series_item_changed(self, item):
        wid = item.data(Qt.UserRole)
        if wid not in self.series_config:
            return

        self.series_config[wid]["visible"] = item.checkState() == Qt.Checked
        self._refresh_profiles()

    def _on_series_selection_changed(self):
        item = self.series_list.currentItem()
        if item is None:
            self.series_channel_spin.blockSignals(True)
            self.series_channel_spin.setRange(1, 1)
            self.series_channel_spin.setValue(1)
            self.series_channel_spin.blockSignals(False)
            self.series_channel_spin.setEnabled(False)
            return

        wid = item.data(Qt.UserRole)
        if wid not in self.series_config:
            return

        cfg = self.series_config[wid]
        window = cfg["window"]
        n_channels = int(getattr(window, "C", 1))
        ch = int(cfg.get("channel", 0)) + 1

        self.series_channel_spin.blockSignals(True)
        self.series_channel_spin.setRange(1, max(1, n_channels))
        self.series_channel_spin.setValue(int(np.clip(ch, 1, max(1, n_channels))))
        self.series_channel_spin.blockSignals(False)
        self.series_channel_spin.setEnabled(
            n_channels > 1 and not self.all_channels_cb.isChecked()
        )

    def _on_all_channels_toggled(self, checked):
        self.series_channel_spin.setEnabled(not checked and self.series_list.currentItem() is not None)
        self._refresh_series_list()
        self._refresh_profiles()

    def _on_selected_channel_changed(self, value):
        item = self.series_list.currentItem()
        if item is None:
            return

        wid = item.data(Qt.UserRole)
        if wid not in self.series_config:
            return

        cfg = self.series_config[wid]
        window = cfg["window"]
        n_channels = int(getattr(window, "C", 1))
        new_channel = int(np.clip(value - 1, 0, max(0, n_channels - 1)))

        cfg["channel"] = new_channel
        cfg["color"] = self._get_window_channel_color(window, new_channel, 0)

        self._refresh_series_list(select_wid=wid)
        self._refresh_profiles()

    def _refresh_profiles(self):
        if self.current_line_data is None:
            self.profile_widget.clear()
            self._computed_series = []
            self.status_label.setText("Select a LineROI to start")
            return

        p1 = self.current_line_data["p1"]
        p2 = self.current_line_data["p2"]
        path = self.current_line_data.get("path")

        if path is not None and len(path) >= 2:
            seg = np.diff(path, axis=0)
            seg_len = np.hypot(seg[:, 0], seg[:, 1])
            cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            length_px = float(cum[-1])
            num_points = max(2, len(path))
            distances_px = cum
        else:
            length_px = float(
                np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            )
            num_points = max(2, int(np.ceil(length_px)))
            distances_px = np.linspace(0.0, length_px, num_points)

        length_um = self._line_length_um_from_source(p1, p2)
        if path is not None and length_um is not None:
            # Re-derive length_um from arc length: avg of sx, sy applied to
            # cumulative px distance.
            scale = getattr(self.source_window, "meta", {}).get("scale")
            if scale is not None and len(scale) >= 3:
                sx, sy = float(scale[2]), float(scale[1])
                length_um = length_px * 0.5 * (sx + sy)
        distances_um = None
        if length_um is not None and length_px > 0:
            distances_um = distances_px * (length_um / length_px)

        all_channels = self.all_channels_cb.isChecked()
        plot_series = []
        computed_series = []
        stale_wids = []
        color_idx = 0

        for wid, cfg in self.series_config.items():
            window = cfg.get("window")
            if window is None:
                stale_wids.append(wid)
                continue

            visible = bool(cfg.get("visible", True))
            label = f"[{window.window_id}] {window.windowTitle()}"
            cfg["label"] = label

            if all_channels:
                n_ch = self._num_channels(window)
                channels = range(n_ch)
            else:
                channels = [int(cfg.get("channel", 0))]

            for ch in channels:
                profile, ch_used = self._sample_profile(
                    window, p1, p2, num_points, ch, path=path
                )
                if profile is None:
                    continue

                color = self._get_window_channel_color(window, ch_used, color_idx)
                color_idx += 1
                ch_label = f"{label} Ch{ch_used + 1}"

                plot_series.append(
                    {
                        "label": ch_label,
                        "color": color,
                        "distances": distances_px,
                        "values": profile,
                        "visible": visible,
                    }
                )

                computed_series.append(
                    {
                        "window_id": window.window_id,
                        "window_title": window.windowTitle(),
                        "channel": ch_used,
                        "color": color,
                        "visible": visible,
                        "distances_px": distances_px,
                        "distances_um": distances_um,
                        "values": profile,
                        "p1": p1,
                        "p2": p2,
                        "t_idx": getattr(window, "t_idx", None),
                        "z_idx": getattr(window, "z_idx", None),
                    }
                )

            if not all_channels and channels:
                cfg["color"] = self._get_window_channel_color(
                    window, channels[0], 0
                )
                cfg["channel"] = channels[0]

        for wid in stale_wids:
            self.series_config.pop(wid, None)

        if stale_wids:
            self._refresh_series_list()

        self._computed_series = computed_series
        self.profile_widget.set_series(plot_series)

        n_visible = sum(1 for s in plot_series if s.get("visible", True))
        if distances_um is not None:
            self.status_label.setText(
                f"Line: {length_px:.1f} px ({length_um:.2f} um) | "
                f"({p1[0]:.0f}, {p1[1]:.0f}) -> ({p2[0]:.0f}, {p2[1]:.0f}) | "
                f"Series: {n_visible}/{len(plot_series)}"
            )
        else:
            self.status_label.setText(
                f"Line: {length_px:.1f} px | "
                f"({p1[0]:.0f}, {p1[1]:.0f}) -> ({p2[0]:.0f}, {p2[1]:.0f}) | "
                f"Series: {n_visible}/{len(plot_series)}"
            )

    def _line_length_um_from_source(self, p1, p2):
        if self.source_window is None:
            return None

        scale = getattr(self.source_window, "meta", {}).get("scale")
        if scale is None or len(scale) < 3:
            return None

        sy = float(scale[1])
        sx = float(scale[2])
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        return float(np.sqrt((dx * sx) ** 2 + (dy * sy) ** 2))

    def _sample_profile(self, window, p1, p2, num_points, channel_idx, *, path=None):
        cache = window.renderer.current_slice_cache
        if cache is None:
            return None, channel_idx

        if cache.ndim == 2:
            image_2d = cache
            channel_used = 0
        elif cache.ndim == 3:
            channel_used = int(np.clip(channel_idx, 0, max(0, cache.shape[0] - 1)))
            image_2d = cache[channel_used]
        else:
            return None, channel_idx

        from scipy.ndimage import map_coordinates

        if path is not None and len(path) >= 2:
            xs = np.asarray(path[:, 0], dtype=float)
            ys = np.asarray(path[:, 1], dtype=float)
        else:
            x1, y1 = p1
            x2, y2 = p2
            xs = np.linspace(x1, x2, num_points)
            ys = np.linspace(y1, y2, num_points)
        coords = np.array([ys, xs], dtype=float)

        profile = map_coordinates(
            np.asarray(image_2d, dtype=float),
            coords,
            order=1,
            mode="nearest",
        )
        return profile, channel_used

    @staticmethod
    def _num_channels(window):
        cache = window.renderer.current_slice_cache
        if cache is not None and cache.ndim == 3:
            return cache.shape[0]
        return int(getattr(window, "C", 1))

    def _get_window_channel_color(self, window, channel_idx, fallback_idx):
        colors = getattr(window.renderer, "channel_colors", [])
        if channel_idx < len(colors):
            return _to_qcolor(colors[channel_idx], QColor(FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)])).name()
        return FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]

    def _export_profiles(self):
        visible = [s for s in self._computed_series if s.get("visible", True)]
        if not visible:
            self.status_label.setText("No profile data to export")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Line Profiles",
            "line_profiles.csv",
            "CSV Files (*.csv);;TSV Files (*.tsv)",
        )
        if not path:
            return

        if path.lower().endswith(".tsv") or "TSV" in selected_filter:
            delimiter = "\t"
        else:
            delimiter = ","

        p1 = visible[0].get("p1", (0, 0))
        p2 = visible[0].get("p2", (0, 0))
        distances_px = np.asarray(visible[0]["distances_px"], dtype=float)
        distances_um = visible[0].get("distances_um")
        has_um = distances_um is not None
        if has_um:
            distances_um = np.asarray(distances_um, dtype=float)

        scale = getattr(self.source_window, "meta", {}).get("scale") if self.source_window else None

        try:
            with open(path, "w", newline="") as f:
                f.write(f"# Line Profile\n")
                f.write(f"# p1_xy: ({p1[0]:.2f}, {p1[1]:.2f})\n")
                f.write(f"# p2_xy: ({p2[0]:.2f}, {p2[1]:.2f})\n")
                if scale is not None and len(scale) >= 3:
                    f.write(f"# pixel_size_yx: ({float(scale[1]):.6g}, {float(scale[2]):.6g}) um\n")
                f.write(f"# line_length_px: {float(distances_px[-1]):.2f}\n")
                if has_um:
                    f.write(f"# line_length_um: {float(distances_um[-1]):.4f}\n")
                f.write(f"# all_channels: {self.all_channels_cb.isChecked()}\n")
                f.write(f"#\n")

                col_labels = []
                for s in visible:
                    col_labels.append(
                        f"[{s['window_id']}] {s['window_title']} Ch{s['channel'] + 1}"
                    )

                header = ["distance_px"]
                if has_um:
                    header.append("distance_um")
                header.extend(col_labels)

                writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(header)

                for i in range(len(distances_px)):
                    row = [f"{float(distances_px[i]):.4f}"]
                    if has_um:
                        row.append(f"{float(distances_um[i]):.6f}")
                    for s in visible:
                        vals = np.asarray(s["values"], dtype=float)
                        row.append(f"{float(vals[i]):.6g}")
                    writer.writerow(row)

        except Exception as e:
            self.status_label.setText(f"Export failed: {e}")
            return

        self.status_label.setText(f"Exported {len(visible)} series to {path}")

    def showEvent(self, event):
        super().showEvent(event)

        for window in manager.get_all().values():
            self._connect_window(window)
            for roi in window.rois:
                if roi.selected and isinstance(roi, LineROI):
                    self.active_window = window
                    self._update_profile(roi)
                    return

    def closeEvent(self, event):
        if self._is_shutting_down:
            super().closeEvent(event)
        else:
            event.ignore()
            self.hide()

    def cleanup(self):
        self._is_shutting_down = True

        try:
            manager.window_registered.disconnect(self._on_window_registered)
        except (TypeError, RuntimeError):
            pass

        for window in list(self._connected_windows):
            self._disconnect_window(window)

        self._unsubscribe_from_shape_source()
        self.active_window = None
        self.source_window = None
        self.current_roi = None
        self.current_line_data = None


def _to_qcolor(value, default):
    if isinstance(value, QColor):
        return value

    # Handle rgb float triples/lists from renderer
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        try:
            r, g, b = value[:3]
            if max(r, g, b) <= 1.0:
                return QColor.fromRgbF(float(r), float(g), float(b))
            return QColor(int(r), int(g), int(b))
        except Exception:
            return default

    qcolor = QColor(value)
    if not qcolor.isValid():
        return default
    return qcolor
