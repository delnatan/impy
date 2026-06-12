"""PointIntensityDialog — readout + Z-profile + T-profile for one point.

Three sections, all per channel:

1. **Header**: point id, (x, y), and the current ``(t, z, c)`` intensity.
2. **Z-profile**: intensity over Z at fixed ``(t, x, y)``, sampled with
   bilinear interpolation in YX.
3. **T-profile**: intensity over T at fixed ``(z, x, y)``.

The plots use pure ``QPainter`` (no matplotlib — per CLAUDE.md). The
dialog subscribes to the parent window's ``view_changed`` signal so the
header readout follows the current slice; profile axes only redraw when
the point itself moves or the dialog is reopened.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QColor, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..data.point_commands import PEVT_REMOVED


def _sample_xy(frame_yx: np.ndarray, x: float, y: float) -> float:
    """Bilinear sample at (x, y) on a 2D array. NaN outside bounds."""
    H, W = frame_yx.shape
    if x < 0 or y < 0 or x > W - 1 or y > H - 1:
        return float("nan")
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1 = min(W - 1, x0 + 1); y1 = min(H - 1, y0 + 1)
    fx = x - x0; fy = y - y0
    a = frame_yx[y0, x0]; b = frame_yx[y0, x1]
    c = frame_yx[y1, x0]; d = frame_yx[y1, x1]
    return float(
        a * (1 - fx) * (1 - fy)
        + b * fx * (1 - fy)
        + c * (1 - fx) * fy
        + d * fx * fy
    )


class _MiniPlot(QWidget):
    """One profile widget; multiple series stacked, simple linear axes."""

    def __init__(self, title: str, x_label: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._x_label = x_label
        self._series: list[dict] = []
        self.setMinimumHeight(140)

    def set_series(self, series: list[dict]) -> None:
        """series item: {label, color: QColor, x: np.ndarray, y: np.ndarray}."""
        self._series = series
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        p.fillRect(rect, QColor(30, 30, 30))
        margin_l, margin_r, margin_t, margin_b = 40, 12, 24, 24
        plot_w = max(1, rect.width() - margin_l - margin_r)
        plot_h = max(1, rect.height() - margin_t - margin_b)

        # Title
        p.setPen(QColor(200, 200, 200))
        p.drawText(margin_l, 16, self._title)

        if not self._series:
            p.setPen(QColor(120, 120, 120))
            p.drawText(margin_l, margin_t + plot_h // 2, "(no data)")
            return

        all_x = np.concatenate([np.asarray(s["x"], dtype=float) for s in self._series])
        all_y = np.concatenate([np.asarray(s["y"], dtype=float) for s in self._series])
        finite = np.isfinite(all_y)
        if not finite.any():
            return
        x_min, x_max = float(all_x.min()), float(all_x.max())
        y_min = float(np.nanmin(all_y[finite]))
        y_max = float(np.nanmax(all_y[finite]))
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_max = y_min + 1.0

        # Axes
        p.setPen(QPen(QColor(80, 80, 80), 1))
        p.drawLine(margin_l, margin_t, margin_l, margin_t + plot_h)
        p.drawLine(margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h)

        # Tick labels
        p.setPen(QColor(160, 160, 160))
        p.drawText(margin_l, margin_t + plot_h + 18, f"{x_min:.0f}")
        p.drawText(
            margin_l + plot_w - 30, margin_t + plot_h + 18, f"{x_max:.0f}"
        )
        p.drawText(2, margin_t + 8, f"{y_max:.1f}")
        p.drawText(2, margin_t + plot_h, f"{y_min:.1f}")
        p.drawText(
            margin_l + plot_w // 2 - 20,
            margin_t + plot_h + 18,
            self._x_label,
        )

        # Series
        for s in self._series:
            x = np.asarray(s["x"], dtype=float)
            y = np.asarray(s["y"], dtype=float)
            poly = QPolygonF()
            for xi, yi in zip(x, y):
                if not np.isfinite(yi):
                    continue
                px = margin_l + (xi - x_min) / (x_max - x_min) * plot_w
                py = margin_t + (1 - (yi - y_min) / (y_max - y_min)) * plot_h
                poly.append(QPointF(px, py))
            color = s.get("color", QColor(200, 200, 200))
            p.setPen(QPen(color, 1.6))
            p.drawPolyline(poly)


def _channel_colors(window, n: int) -> list[QColor]:
    """Best-effort channel colors from the window's renderer."""
    fallback = [
        QColor(220, 220, 220), QColor(120, 200, 120), QColor(200, 120, 120),
        QColor(120, 160, 220), QColor(220, 200, 120),
    ]
    colors = getattr(getattr(window, "renderer", None), "channel_colors", None) or []
    out: list[QColor] = []
    for c in range(n):
        if c < len(colors) and colors[c] is not None:
            r, g, b = colors[c][:3]
            out.append(QColor(int(r * 255), int(g * 255), int(b * 255)))
        else:
            out.append(fallback[c % len(fallback)])
    return out


class PointIntensityDialog(QDialog):
    def __init__(self, window, layer, point_id: int, parent=None):
        super().__init__(parent if parent is not None else window)
        self.setWindowTitle("Point Intensity")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(500, 480)

        self._window = window
        self._layer = layer
        self._point_id = int(point_id)
        self._unsubs: list = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        self._header = QLabel("")
        self._header.setTextFormat(Qt.RichText)
        outer.addWidget(self._header)

        self._all_channels = QCheckBox("All channels")
        self._all_channels.setChecked(True)
        self._all_channels.toggled.connect(self._refresh_profiles)
        outer.addWidget(self._all_channels)

        self._z_plot = _MiniPlot("Z-profile", "z")
        self._t_plot = _MiniPlot("T-profile", "t")
        outer.addWidget(self._z_plot, 1)
        outer.addWidget(self._t_plot, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        refresh = QPushButton("Refresh"); refresh.clicked.connect(self._refresh_profiles)
        btn_row.addWidget(refresh)
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        btn_row.addWidget(close)
        outer.addLayout(btn_row)

        self._unsubs.append(layer.data.subscribe(self._on_point_event))
        if hasattr(window, "view_changed"):
            try:
                window.view_changed.connect(self._refresh_header)
                self._unsubs.append(("view_changed", window.view_changed))
            except Exception:
                pass

        self._refresh_profiles()

    # ------------------------------------------------------------------
    def _row(self):
        tbl = getattr(self._layer.data, "table", None)
        return None if tbl is None else tbl.get_point(self._point_id)

    def _refresh_header(self, *_args):
        row = self._row()
        if row is None:
            self._header.setText("(point not found)")
            return
        x, y = float(row["x"]), float(row["y"])
        try:
            frame = np.asarray(
                self._window.img_data[
                    self._window.t_idx, self._window.z_idx, :, :, :
                ]
            )
        except Exception:
            self._header.setText(
                f"<b>Point {self._point_id}</b> at ({x:.2f}, {y:.2f})"
            )
            return
        C, _Y, _X = frame.shape
        parts = []
        for c in range(C):
            v = _sample_xy(frame[c], x, y)
            parts.append(f"ch{c}={v:.1f}")
        intensities = ", ".join(parts)
        self._header.setText(
            f"<b>Point {self._point_id}</b> at "
            f"(x={x:.2f}, y={y:.2f}) · "
            f"t={self._window.t_idx}, z={self._window.z_idx} · "
            f"{intensities}"
        )

    def _refresh_profiles(self, *_args):
        self._refresh_header()
        row = self._row()
        if row is None:
            self._z_plot.set_series([])
            self._t_plot.set_series([])
            return
        x, y = float(row["x"]), float(row["y"])

        T, Z, C, _Y, _X = self._window.img_data.shape
        channels = list(range(C)) if self._all_channels.isChecked() else [self._window.c_idx]
        colors = _channel_colors(self._window, C)

        # Z-profile at fixed t = current t_idx.
        z_series = []
        if Z > 1:
            z_vals = np.arange(Z)
            for c in channels:
                ys = np.empty(Z, dtype=float)
                try:
                    stack = np.asarray(
                        self._window.img_data[self._window.t_idx, :, c, :, :]
                    )
                except Exception:
                    ys[:] = np.nan
                else:
                    for zi in range(Z):
                        ys[zi] = _sample_xy(stack[zi], x, y)
                z_series.append({
                    "label": f"ch{c}",
                    "color": colors[c],
                    "x": z_vals,
                    "y": ys,
                })
        self._z_plot.set_series(z_series)

        # T-profile at fixed z = current z_idx.
        t_series = []
        if T > 1:
            t_vals = np.arange(T)
            for c in channels:
                ys = np.empty(T, dtype=float)
                for ti in range(T):
                    try:
                        frame = np.asarray(
                            self._window.img_data[ti, self._window.z_idx, c, :, :]
                        )
                        ys[ti] = _sample_xy(frame, x, y)
                    except Exception:
                        ys[ti] = float("nan")
                t_series.append({
                    "label": f"ch{c}",
                    "color": colors[c],
                    "x": t_vals,
                    "y": ys,
                })
        self._t_plot.set_series(t_series)

    def _on_point_event(self, kind, pid):
        if kind == PEVT_REMOVED and int(pid) == self._point_id:
            self.close()
            return
        if pid != self._point_id and pid != -1:
            return
        self._refresh_profiles()

    def closeEvent(self, event):
        for u in self._unsubs:
            try:
                if callable(u):
                    u()
                else:
                    kind, sig = u
                    sig.disconnect(self._refresh_header)
            except Exception:
                pass
        self._unsubs.clear()
        super().closeEvent(event)
