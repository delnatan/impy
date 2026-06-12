"""Convergence plot widgets — live iteration-history plots for iterative
algorithms (phase retrieval, PSF distillation, etc.).

Two pieces:

* :class:`ConvergencePlotWidget` — embeddable Qt-painter plot. Default
  log-y, multi-series, follows the line_profile.py visual style.
* :class:`ConvergencePlotDialog` — singleton standalone window that holds
  one or more named runs so the user can compare convergence across
  parameter changes, save the histories to CSV, or remove individual runs.

The widget API is small on purpose:

    plot.set_series("MSE", values, color="#66CCFF")
    plot.set_series("Support error", values, color="#FF9966")
    plot.clear()
    plot.set_y_log(True)

so callers can drop it under a progress bar and just push fresh histories
every progress tick.
"""

from __future__ import annotations

import csv
from collections import OrderedDict

import numpy as np
from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .histogram import TEXT_COLOR, WIDGET_BG

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


def _to_qcolor(spec, default):
    if isinstance(spec, QColor):
        return spec
    if isinstance(spec, str):
        c = QColor(spec)
        if c.isValid():
            return c
    return default


class ConvergencePlotWidget(QWidget):
    """Embeddable iteration-history plot (log-y by default).

    Series are stored in insertion order. Calling :meth:`set_series` with
    an existing label updates that series in place (cheap repaint, no
    flicker), which is the canonical pattern during live iteration.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMinimumWidth(280)

        self._series = OrderedDict()  # label -> {color, values, visible}
        self._y_log = True
        self._x_label = "Iteration"
        self._y_label = "MSE"
        self._title = ""

        self.margin_left = 56
        self.margin_right = 12
        self.margin_top = 18
        self.margin_bottom = 28

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_series(self, label, values, color=None, visible=True):
        """Insert or replace a series. ``values`` is any 1-D iterable."""
        vals = np.asarray(list(values), dtype=float)
        entry = self._series.get(label)
        if entry is None:
            entry = {
                "color": color or FALLBACK_COLORS[len(self._series) % len(FALLBACK_COLORS)],
                "values": vals,
                "visible": visible,
            }
            self._series[label] = entry
        else:
            entry["values"] = vals
            if color is not None:
                entry["color"] = color
            entry["visible"] = visible
        self.update()

    def remove_series(self, label):
        if label in self._series:
            del self._series[label]
            self.update()

    def set_series_visible(self, label, visible):
        if label in self._series:
            self._series[label]["visible"] = bool(visible)
            self.update()

    def clear(self):
        self._series.clear()
        self.update()

    def labels(self):
        return list(self._series.keys())

    def series_data(self, label):
        e = self._series.get(label)
        return None if e is None else np.asarray(e["values"])

    def set_y_log(self, enabled):
        self._y_log = bool(enabled)
        self.update()

    def set_labels(self, x_label=None, y_label=None, title=None):
        if x_label is not None:
            self._x_label = x_label
        if y_label is not None:
            self._y_label = y_label
        if title is not None:
            self._title = title
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def _plot_rect(self):
        x = self.margin_left
        y = self.margin_top
        w = self.width() - self.margin_left - self.margin_right
        h = self.height() - self.margin_top - self.margin_bottom
        return x, y, max(1, w), max(1, h)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), WIDGET_BG)

        visible = [(lbl, s) for lbl, s in self._series.items() if s["visible"] and s["values"].size > 0]
        if not visible:
            painter.setPen(TEXT_COLOR)
            painter.drawText(self.rect(), Qt.AlignCenter, "(no convergence data yet)")
            return

        plot_x, plot_y, plot_w, plot_h = self._plot_rect()

        x_max = max(s["values"].size for _, s in visible)
        x_min = 1
        if x_max < 2:
            x_max = 2

        all_finite = []
        for _, s in visible:
            v = s["values"]
            if self._y_log:
                v = v[np.isfinite(v) & (v > 0)]
            else:
                v = v[np.isfinite(v)]
            if v.size:
                all_finite.append(v)
        if all_finite:
            cat = np.concatenate(all_finite)
            y_min = float(np.min(cat))
            y_max = float(np.max(cat))
        else:
            y_min, y_max = 1e-6, 1.0

        if self._y_log:
            y_min = max(y_min, 1e-300)
            if y_max <= y_min:
                y_max = y_min * 10.0
            log_y_min = np.log10(y_min)
            log_y_max = np.log10(y_max)
            pad = max(0.05 * (log_y_max - log_y_min), 0.1)
            log_y_min -= pad
            log_y_max += pad
        else:
            if y_max == y_min:
                y_min -= 0.5
                y_max += 0.5
            else:
                pad = 0.05 * (y_max - y_min)
                y_min -= pad
                y_max += pad
            log_y_min = log_y_max = None  # silence linter

        self._draw_axes(
            painter, plot_x, plot_y, plot_w, plot_h,
            x_min, x_max, y_min, y_max, log_y_min, log_y_max,
        )
        self._draw_legend(painter, plot_x, plot_y, plot_w, visible)
        for idx, (label, s) in enumerate(visible):
            self._draw_series(
                painter, s, idx, plot_x, plot_y, plot_w, plot_h,
                x_min, x_max, y_min, y_max, log_y_min, log_y_max,
            )

    def _x_to_px(self, x_val, x_min, x_max, plot_x, plot_w):
        if x_max == x_min:
            return plot_x
        return plot_x + (x_val - x_min) / (x_max - x_min) * plot_w

    def _y_to_px(self, y_val, y_min, y_max, plot_y, plot_h, log_y_min, log_y_max):
        if self._y_log:
            if y_val <= 0 or not np.isfinite(y_val):
                return None
            r = (np.log10(y_val) - log_y_min) / (log_y_max - log_y_min)
        else:
            if not np.isfinite(y_val):
                return None
            if y_max == y_min:
                return plot_y + plot_h / 2
            r = (y_val - y_min) / (y_max - y_min)
        return plot_y + plot_h - r * plot_h

    def _draw_axes(
        self, painter, plot_x, plot_y, plot_w, plot_h,
        x_min, x_max, y_min, y_max, log_y_min, log_y_max,
    ):
        border = QPen(QColor(80, 80, 80))
        border.setWidth(1)
        painter.setPen(border)
        painter.drawRect(plot_x, plot_y, plot_w, plot_h)

        grid = QPen(QColor(60, 60, 60))
        grid.setStyle(Qt.DotLine)
        painter.setPen(grid)
        for i in range(1, 4):
            yy = plot_y + i / 4 * plot_h
            painter.drawLine(int(plot_x), int(yy), int(plot_x + plot_w), int(yy))
            xx = plot_x + i / 4 * plot_w
            painter.drawLine(int(xx), int(plot_y), int(xx), int(plot_y + plot_h))

        painter.setPen(TEXT_COLOR)
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        # Y tick labels
        if self._y_log:
            n_ticks = 4
            for i in range(n_ticks + 1):
                v_log = log_y_min + (i / n_ticks) * (log_y_max - log_y_min)
                v = 10 ** v_log
                yp = self._y_to_px(v, y_min, y_max, plot_y, plot_h, log_y_min, log_y_max)
                if yp is None:
                    continue
                painter.drawText(
                    2, int(yp - 6), self.margin_left - 6, 12,
                    Qt.AlignRight | Qt.AlignVCenter, _fmt(v),
                )
        else:
            for v in (y_min, (y_min + y_max) / 2, y_max):
                yp = self._y_to_px(v, y_min, y_max, plot_y, plot_h, log_y_min, log_y_max)
                if yp is None:
                    continue
                painter.drawText(
                    2, int(yp - 6), self.margin_left - 6, 12,
                    Qt.AlignRight | Qt.AlignVCenter, _fmt(v),
                )

        # X tick labels
        for v in (x_min, (x_min + x_max) // 2, x_max):
            xp = self._x_to_px(v, x_min, x_max, plot_x, plot_w)
            painter.drawText(
                int(xp - 25), int(plot_y + plot_h + 4), 50, 14,
                Qt.AlignCenter, f"{int(v)}",
            )

        painter.drawText(
            int(plot_x + plot_w / 2 - 60), int(plot_y + plot_h + 14),
            120, 14, Qt.AlignCenter, self._x_label,
        )
        # Y-axis label, rotated.
        painter.save()
        painter.translate(12, plot_y + plot_h / 2)
        painter.rotate(-90)
        painter.drawText(-60, -2, 120, 14, Qt.AlignCenter, self._y_label)
        painter.restore()

        if self._title:
            painter.drawText(
                int(plot_x), int(plot_y - 14), int(plot_w), 14,
                Qt.AlignLeft | Qt.AlignVCenter, self._title,
            )

    def _draw_legend(self, painter, plot_x, plot_y, plot_w, visible):
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        x = plot_x + plot_w - 8
        y = plot_y + 4
        for label, s in visible:
            qc = _to_qcolor(s["color"], QColor(255, 255, 255))
            text = label
            tw = painter.fontMetrics().horizontalAdvance(text)
            box_w = tw + 22
            painter.fillRect(int(x - box_w), int(y), 10, 10, qc)
            painter.setPen(TEXT_COLOR)
            painter.drawText(
                int(x - box_w + 14), int(y - 2), tw + 4, 14,
                Qt.AlignLeft | Qt.AlignVCenter, text,
            )
            y += 14

    def _draw_series(
        self, painter, series, idx, plot_x, plot_y, plot_w, plot_h,
        x_min, x_max, y_min, y_max, log_y_min, log_y_max,
    ):
        qc = _to_qcolor(series["color"], QColor(255, 255, 255))
        pen = QPen(qc)
        pen.setWidth(2)
        if idx > 0:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)

        values = series["values"]
        points = []
        for i, v in enumerate(values, start=1):
            yp = self._y_to_px(v, y_min, y_max, plot_y, plot_h, log_y_min, log_y_max)
            if yp is None:
                if len(points) > 1:
                    painter.drawPolyline(QPolygonF(points))
                points = []
                continue
            xp = self._x_to_px(i, x_min, x_max, plot_x, plot_w)
            points.append(QPointF(xp, yp))
        if len(points) > 1:
            painter.drawPolyline(QPolygonF(points))


def _fmt(v):
    av = abs(v)
    if av == 0:
        return "0"
    if av < 1e-2 or av >= 1e4:
        return f"{v:.1e}"
    if av < 1:
        return f"{v:.3f}"
    if av < 100:
        return f"{v:.2f}"
    return f"{v:.0f}"


# ---------------------------------------------------------------------------
# Standalone comparison dialog (singleton)
# ---------------------------------------------------------------------------


_comparison_dialog = None


def get_convergence_comparison_dialog():
    """Return (creating if needed) the singleton comparison dialog."""
    global _comparison_dialog
    if _comparison_dialog is None:
        _comparison_dialog = ConvergencePlotDialog()
    return _comparison_dialog


class ConvergencePlotDialog(QDialog):
    """Standalone comparison window for convergence histories.

    Runs are added via :meth:`add_run`. The dialog deduplicates labels by
    appending a counter (``"GS retrieval (2)"``). Series can be toggled,
    removed, or exported to CSV (one column per series, padded with empty
    cells when histories differ in length).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Convergence comparison")
        self.setWindowFlags(Qt.Tool)
        self.resize(720, 460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        self.log_check = QCheckBox("Log Y")
        self.log_check.setChecked(True)
        self.log_check.toggled.connect(self._on_log_toggled)
        controls.addWidget(self.log_check)
        controls.addStretch()
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self._remove_selected)
        controls.addWidget(self.remove_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_all)
        controls.addWidget(self.clear_btn)
        self.export_btn = QPushButton("Save CSV…")
        self.export_btn.clicked.connect(self._export_csv)
        controls.addWidget(self.export_btn)
        layout.addLayout(controls)

        self.plot = ConvergencePlotWidget()
        layout.addWidget(self.plot, 1)

        self.list = QListWidget()
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.setMaximumHeight(120)
        layout.addWidget(QLabel("Series:"))
        layout.addWidget(self.list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_run(self, label, values, color=None):
        """Add a run with a unique label. Returns the actual label used."""
        unique = self._unique_label(label)
        self.plot.set_series(unique, values, color=color, visible=True)
        item = QListWidgetItem(unique)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.list.addItem(item)
        self.show()
        self.raise_()
        return unique

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unique_label(self, base):
        existing = set(self.plot.labels())
        if base not in existing:
            return base
        i = 2
        while f"{base} ({i})" in existing:
            i += 1
        return f"{base} ({i})"

    def _on_log_toggled(self, checked):
        self.plot.set_y_log(checked)

    def _on_item_changed(self, item):
        self.plot.set_series_visible(item.text(), item.checkState() == Qt.Checked)

    def _remove_selected(self):
        for item in list(self.list.selectedItems()):
            self.plot.remove_series(item.text())
            self.list.takeItem(self.list.row(item))

    def _clear_all(self):
        self.plot.clear()
        self.list.clear()

    def _export_csv(self):
        labels = self.plot.labels()
        if not labels:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save convergence histories", "convergence.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        cols = [(lbl, self.plot.series_data(lbl)) for lbl in labels]
        n = max(int(a.size) for _, a in cols)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["iteration"] + labels)
            for i in range(n):
                row = [i + 1]
                for _, arr in cols:
                    row.append(f"{arr[i]:.10g}" if i < arr.size else "")
                w.writerow(row)
