"""
Line Profile Dialog - Displays and compares intensity profiles along line/polyline shapes.

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
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import colors as tokens
from ..data import calibration
from ..data.calibration import window_is_frequency_space
from .histogram import WIDGET_BG, TEXT_COLOR
from .series_colors import _to_qcolor
from .window_series_mixin import WindowSeriesMixin
from ..ui.comparison import paired_window
from ..ui.manager import manager

# Singleton instance
_line_profile_dialog = None


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
        self.x_axis_label = "Distance (px)"

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
        self.x_axis_label = "Distance (px)"
        self.update()

    def set_x_axis_label(self, label):
        self.x_axis_label = str(label)
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
                "Select a line/polyline shape and add one or more windows to compare",
            )
            return

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
            y_min_data = float(np.min(all_visible))
            y_max_data = float(np.max(all_visible))
        else:
            y_min_data, y_max_data = 0.0, 1.0

        y_min, y_max = self._padded_y_range(y_min_data, y_max_data)
        y_ticks = self._nice_ticks(y_min, y_max)

        plot_x, plot_y, plot_w, plot_h = self._get_plot_rect()

        self._draw_axes(
            painter,
            plot_x,
            plot_y,
            plot_w,
            plot_h,
            x_min,
            x_max,
            y_min,
            y_max,
            y_ticks,
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

    def _draw_axes(
        self,
        painter,
        plot_x,
        plot_y,
        plot_w,
        plot_h,
        x_min,
        x_max,
        y_min,
        y_max,
        y_ticks,
    ):
        border_pen = QPen(QColor(80, 80, 80))
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.drawRect(plot_x, plot_y, plot_w, plot_h)

        grid_pen = QPen(QColor(60, 60, 60))
        grid_pen.setStyle(Qt.DotLine)
        painter.setPen(grid_pen)

        for y_val in y_ticks:
            y = self._y_to_px(y_val, y_min, y_max, plot_y, plot_h)
            if y <= plot_y or y >= plot_y + plot_h:
                continue
            painter.drawLine(int(plot_x), int(y), int(plot_x + plot_w), int(y))

        for i in range(1, 4):
            x = plot_x + (i / 4) * plot_w
            painter.drawLine(int(x), int(plot_y), int(x), int(plot_y + plot_h))

        if y_min < 0 < y_max:
            zero_pen = QPen(QColor(100, 100, 100))
            zero_pen.setWidth(1)
            painter.setPen(zero_pen)
            y = self._y_to_px(0.0, y_min, y_max, plot_y, plot_h)
            painter.drawLine(int(plot_x), int(y), int(plot_x + plot_w), int(y))

        painter.setPen(TEXT_COLOR)
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        for y_val in y_ticks:
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
            label = self._format_distance(x_val, self.x_axis_label)
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
            self.x_axis_label,
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
    def _padded_y_range(y_min, y_max):
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return 0.0, 1.0

        if y_max < y_min:
            y_min, y_max = y_max, y_min

        y_range = y_max - y_min
        if y_range > 0:
            pad = y_range * 0.05
            view_min = y_min - pad
            view_max = y_max + pad
        else:
            pad = max(abs(y_min) * 0.05, 0.5)
            view_min = y_min - pad
            view_max = y_max + pad

        if y_min >= 0 and view_min < 0:
            view_min = 0.0

        if view_max <= view_min:
            view_max = view_min + 1.0
        return view_min, view_max

    @staticmethod
    def _nice_ticks(v_min, v_max, max_ticks=5):
        if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
            return [0.0, 1.0]

        raw_step = (v_max - v_min) / max(1, max_ticks - 1)
        magnitude = 10 ** np.floor(np.log10(raw_step))
        residual = raw_step / magnitude
        if residual <= 1:
            step = magnitude
        elif residual <= 2:
            step = 2 * magnitude
        elif residual <= 5:
            step = 5 * magnitude
        else:
            step = 10 * magnitude

        first = np.ceil(v_min / step) * step
        last = np.floor(v_max / step) * step
        ticks = []
        tick = first
        while tick <= last + step * 0.5:
            if abs(tick) < step * 1e-10:
                tick = 0.0
            ticks.append(float(tick))
            tick += step

        if not ticks:
            ticks = [v_min, v_max]
        return ticks

    @staticmethod
    def _format_distance(value, axis_label):
        if axis_label.endswith("(px)"):
            return f"{value:.0f}"
        return LineProfileWidget._format_value(value)

    @staticmethod
    def _format_value(value):
        if abs(value) < 1e-12:
            return "0"
        if abs(value) < 0.01 or abs(value) >= 10000:
            return f"{value:.1e}"
        if abs(value) < 1:
            return f"{value:.2f}"
        if abs(value) < 100:
            return f"{value:.1f}"
        return f"{value:.0f}"


class LineProfileDialog(WindowSeriesMixin, QDialog):
    """
    Floating dialog that displays and compares intensity profiles along line/polyline shapes.

    Use the selected line/polyline shape as source geometry, then add target windows to
    overlay profiles in one plot. Window/series bookkeeping is shared with
    RadialProfileDialog via :class:`WindowSeriesMixin`; see its hook
    overrides below for line-profile-specific behavior (overlays,
    refresh-on-activation).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Line Profile")
        self.setWindowFlags(Qt.Tool)
        self.resize(760, 460)

        self.active_window = None
        self.source_window = None
        self.current_line_data = None  # {p1: (x,y), p2: (x,y)}

        # When the profile source is a shape-layer record, we subscribe to
        # the layer's data so handle/body edits refresh the profile live.
        self._source_shape_layer = None
        self._source_shape_id = None
        self._source_shape_unsub = None

        # wid -> {window, channel, visible, label, color}
        self.series_config = OrderedDict()
        self._computed_series = []

        # Key used with ImageWindow.show_line_overlay/hide_line_overlay to
        # draw a read-only indicator of the profile path on compared
        # windows (the source window already shows its own editable
        # shape, so it never gets one of these).
        self._overlay_key = "line_profile"

        self._is_shutting_down = False

        self._setup_ui()
        self._init_window_series()

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

        self.normalize_cb = QCheckBox("Normalize (per series)")
        self.normalize_cb.setChecked(False)
        self.normalize_cb.setToolTip(
            "Rescale each plotted profile to its own [min, max] -- compares "
            "shape across series with very different absolute intensity "
            "scales (e.g. different exposure/gain), independent of the "
            "actual values. Export always uses absolute values."
        )
        self.normalize_cb.toggled.connect(self._on_normalize_toggled)
        channel_row.addWidget(self.normalize_cb)

        channel_row.addStretch()
        layout.addLayout(channel_row)

        self.profile_widget = LineProfileWidget()
        layout.addWidget(self.profile_widget, 1)

        self.status_label = QLabel("Select a line/polyline shape to start")
        self.status_label.setStyleSheet(f"color: {tokens.TEXT_SECONDARY}; font-size: 10px;")
        layout.addWidget(self.status_label)

    # -- WindowSeriesMixin hooks --------------------------------------
    def _connect_extra_series_signals(self, window):
        window.roi_selection_changed.connect(self._on_roi_selection_changed)
        if hasattr(window, "view_changed"):
            window.view_changed.connect(self._on_view_changed)

    def _disconnect_extra_series_signals(self, window):
        window.roi_selection_changed.disconnect(self._on_roi_selection_changed)
        if hasattr(window, "view_changed"):
            window.view_changed.disconnect(self._on_view_changed)

    def _on_active_window_changed(self):
        if self.current_line_data is not None:
            self._refresh_profiles()

    def _has_shape_source(self) -> bool:
        return self.current_line_data is not None

    def _eligible_series_windows(self):
        return [w for w in manager.get_all().values() if hasattr(w, "roi_added")]

    def _on_series_removed(self, cfg):
        self._hide_profile_overlay(cfg.get("window"))

    def _on_series_cleared_pre(self):
        self._hide_all_overlays()

    def _reset_shape_source_data(self):
        self.current_line_data = None

    def _empty_status_text(self) -> str:
        return "Select a line/polyline shape to start"

    def _on_cleanup_extra(self):
        self._hide_all_overlays()

    # -- Shape source ---------------------------------------------------
    def _on_roi_selection_changed(self, roi):
        if self._is_shutting_down:
            return

        if self.current_line_data is None:
            self.profile_widget.clear()
            self.status_label.setText("Select a line/polyline shape to start")

    def set_shape_source(self, window, layer, shape_id):
        """Use a shape-layer LINE/POLYLINE record as the profile source.

        The dialog refreshes immediately and adds the source window's series
        if not present, and subscribes to the layer so the profile updates
        live as the shape is edited (handle drag, body move, vertex
        insert/remove).
        """
        if shape_id not in layer.data:
            return
        if not self._extract_shape_line_data(layer, shape_id):
            return

        self.source_window = window
        self.active_window = window

        self._subscribe_to_shape_source(layer, shape_id)
        self._ensure_source_series()
        # If the source window is one half of an explicit comparison pair
        # (see ui/comparison.py), the whole point of the pairing is to
        # compare this shape against the other side too -- add it instead
        # of making the user click "Add Window..." for it every time.
        partner = paired_window(window)
        if partner is not None:
            self._add_series_for_window(partner)
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
                self.status_label.setText("Select a line/polyline shape to start")
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

    def _ensure_source_series(self):
        if self.source_window is None:
            return
        self._add_series_for_window(self.source_window)

    def _on_normalize_toggled(self, checked):
        self._refresh_profiles()

    def _refresh_profiles(self):
        if self.current_line_data is None:
            self.profile_widget.clear()
            self._computed_series = []
            self.status_label.setText("Select a line/polyline shape to start")
            return

        p1 = self.current_line_data["p1"]
        p2 = self.current_line_data["p2"]
        path = self.current_line_data.get("path")

        if path is not None and len(path) >= 2:
            path_px_source = np.asarray(path, dtype=float)
        else:
            num_points = max(
                2, int(np.ceil(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
            )
            path_px_source = np.column_stack(
                [
                    np.linspace(p1[0], p2[0], num_points),
                    np.linspace(p1[1], p2[1], num_points),
                ]
            )

        seg = np.diff(path_px_source, axis=0)
        distances_px = np.concatenate(
            [[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
        )
        length_px = float(distances_px[-1])

        # A source window computed from an FFT has "frequency" pixel space
        # (see fft_dialog.py / ImageWindow.pixel_space) -- its "scale" is
        # cycles/unit per pixel index, not a real-space distance, so the
        # same px->physical conversion means something different on the
        # axis label, and it isn't the same physical quantity as a
        # real-space distance from another window (handled below).
        is_frequency_space = window_is_frequency_space(self.source_window)
        src_scale_yx = calibration.window_scale_yx(self.source_window)
        has_phys = src_scale_yx is not None

        if has_phys:
            # Physical-space path: scales each axis independently, so
            # anisotropic pixel sizes (sx != sy) and off-axis segments are
            # handled correctly rather than approximated by an averaged
            # sx/sy factor. This is also the coordinate space shared across
            # windows with different pixel sizes (see the per-window
            # conversion below) -- the x-axis is now genuinely physical,
            # not borrowed from one specific window's pixel grid.
            path_phys = calibration.points_px_to_phys(path_px_source, src_scale_yx)
            seg_phys = np.diff(path_phys, axis=0)
            distances_plot = np.concatenate(
                [[0.0], np.cumsum(np.hypot(seg_phys[:, 0], seg_phys[:, 1]))]
            )
            length_um = float(distances_plot[-1])
            x_axis_label = (
                "Spatial freq. (1/um)" if is_frequency_space else "Distance (um)"
            )
        else:
            path_phys = None
            distances_plot = distances_px
            length_um = None
            x_axis_label = (
                "Spatial freq. (cycles/px)" if is_frequency_space else "Distance (px)"
            )
        distances_um = distances_plot if has_phys else None

        all_channels = self.all_channels_cb.isChecked()
        normalize = self.normalize_cb.isChecked()
        plot_series = []
        computed_series = []
        stale_wids = []
        skipped_labels = []
        color_idx = 0

        for wid, cfg in self.series_config.items():
            window = cfg.get("window")
            if window is None:
                stale_wids.append(wid)
                continue

            visible = bool(cfg.get("visible", True))
            label = f"[{window.window_id}] {window.windowTitle()}"
            cfg["label"] = label

            if path_phys is not None:
                win_is_freq = window_is_frequency_space(window)
                if win_is_freq != is_frequency_space:
                    # Real-space distance and spatial frequency aren't the
                    # same physical quantity -- comparing them on one axis
                    # would be meaningless, so skip rather than mislabel.
                    skipped_labels.append(label)
                    continue
                win_scale_yx = calibration.window_scale_yx(window) or (1.0, 1.0)
                # Convert the shared physical path into *this* window's own
                # pixel grid -- sampling with the source window's raw pixel
                # coordinates would land on the wrong physical location
                # whenever pixel sizes differ between windows.
                path_px_window = calibration.points_phys_to_px(path_phys, win_scale_yx)
            else:
                path_px_window = path_px_source

            if all_channels:
                n_ch = self._num_channels(window)
                channels = range(n_ch)
            else:
                channels = [int(cfg.get("channel", 0))]

            if window is self.source_window or not visible:
                self._hide_profile_overlay(window)
            else:
                overlay_channel = channels[0] if channels else int(cfg.get("channel", 0))
                overlay_color = self._get_window_channel_color(window, overlay_channel, 0)
                self._sync_profile_overlay(window, path_px_window, overlay_color)

            for ch in channels:
                profile, ch_used = self._sample_profile(window, path_px_window, ch)
                if profile is None:
                    continue

                color = self._get_window_channel_color(window, ch_used, color_idx)
                color_idx += 1
                ch_label = f"{label} Ch{ch_used + 1}"

                plot_values = profile
                if normalize:
                    plot_values = self._normalize_profile(profile)

                plot_series.append(
                    {
                        "label": ch_label,
                        "color": color,
                        "distances": distances_plot,
                        "values": plot_values,
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
        self.profile_widget.set_x_axis_label(x_axis_label)
        self.profile_widget.set_series(plot_series)

        n_visible = sum(1 for s in plot_series if s.get("visible", True))
        if has_phys:
            unit_text = (
                f"{length_um:.4g} 1/um" if is_frequency_space else f"{length_um:.2f} um"
            )
            status = (
                f"Line: {length_px:.1f} px ({unit_text}) | "
                f"({p1[0]:.0f}, {p1[1]:.0f}) -> ({p2[0]:.0f}, {p2[1]:.0f}) | "
                f"Series: {n_visible}/{len(plot_series)}"
            )
        else:
            status = (
                f"Line: {length_px:.1f} px | "
                f"({p1[0]:.0f}, {p1[1]:.0f}) -> ({p2[0]:.0f}, {p2[1]:.0f}) | "
                f"Series: {n_visible}/{len(plot_series)}"
            )
        if skipped_labels:
            status += f" | {len(skipped_labels)} skipped (mismatched pixel space)"
        if normalize:
            status += " | normalized (per series)"
        self.status_label.setText(status)

    def _sample_profile(self, window, path_px, channel_idx):
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

        profile = calibration.sample_along_path(image_2d, path_px)
        return profile, channel_used

    @staticmethod
    def _normalize_profile(values):
        """Min-max rescale *values* to its own [0, 1] -- compares profile
        *shape* across series with very different absolute intensity
        scales, independent of the actual values. A flat (zero-range)
        profile has no shape to normalize and is returned as all-zero
        rather than dividing by zero."""
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return values
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        rng = vmax - vmin
        if rng <= 0:
            return np.where(np.isfinite(values), 0.0, values)
        return (values - vmin) / rng

    def _sync_profile_overlay(self, window, path_px, color):
        """Draw/update the profile-path indicator on a compared window.

        Skipped for the source window, which already renders the real,
        editable shape the profile was taken from.
        """
        if window is None or window is self.source_window:
            return
        if not hasattr(window, "show_line_overlay"):
            return

        qcolor = _to_qcolor(color, QColor(255, 255, 0))
        rgba = (qcolor.redF(), qcolor.greenF(), qcolor.blueF(), 0.9)
        window.show_line_overlay(self._overlay_key, path_px, color=rgba)

    def _hide_profile_overlay(self, window):
        if window is not None and hasattr(window, "hide_line_overlay"):
            window.hide_line_overlay(self._overlay_key)

    def _hide_all_overlays(self):
        for cfg in self.series_config.values():
            self._hide_profile_overlay(cfg.get("window"))

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

        scale_yx = calibration.window_scale_yx(self.source_window)
        is_frequency_space = window_is_frequency_space(self.source_window)
        phys_unit = "1/um" if is_frequency_space else "um"
        phys_col = "spatial_freq_1_per_um" if is_frequency_space else "distance_um"
        phys_length_key = "spatial_freq_1_per_um" if is_frequency_space else "length_um"

        try:
            with open(path, "w", newline="") as f:
                f.write(f"# Line Profile\n")
                f.write(f"# p1_xy: ({p1[0]:.2f}, {p1[1]:.2f})\n")
                f.write(f"# p2_xy: ({p2[0]:.2f}, {p2[1]:.2f})\n")
                if scale_yx is not None:
                    f.write(f"# pixel_size_yx: ({scale_yx[0]:.6g}, {scale_yx[1]:.6g}) {phys_unit}\n")
                f.write(f"# line_length_px: {float(distances_px[-1]):.2f}\n")
                if has_um:
                    f.write(f"# line_{phys_length_key}: {float(distances_um[-1]):.4f}\n")
                f.write(f"# all_channels: {self.all_channels_cb.isChecked()}\n")
                f.write(f"#\n")

                col_labels = []
                for s in visible:
                    col_labels.append(
                        f"[{s['window_id']}] {s['window_title']} Ch{s['channel'] + 1}"
                    )

                header = ["distance_px"]
                if has_um:
                    header.append(phys_col)
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

        # Restore any previously-configured overlays (hidden on the last
        # hideEvent).
        self._refresh_profiles()

    def hideEvent(self, event):
        # Don't leave stale profile-path indicators on other windows while
        # this dialog isn't visible; showEvent restores them.
        self._hide_all_overlays()
        super().hideEvent(event)

