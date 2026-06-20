"""RegionStatisticsDialog — per-channel summary stats inside a shape.

Modeless dialog showing mean / std / sum / min / max / area for the pixels
inside the selected shape, computed on the current ``(t, z)`` frame. A
checkbox toggles whether all channels are shown or just the currently
displayed one.

Supported shape types: :data:`RECTANGLE`, :data:`CIRCLE`, :data:`POLYLINE`
(closed). Auto-refreshes via :class:`ShapeData.subscribe` and the parent
window's ``view_changed`` signal.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..data.shapes import (
    CIRCLE,
    EVT_REMOVED,
    POLYLINE,
    RECTANGLE,
    get_outline,
    polyline_is_closed,
    rectangle_bounds,
)


def _rect_mask(rec, Y: int, X: int) -> np.ndarray | None:
    x0, y0, x1, y1 = rectangle_bounds(rec, (Y, X))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((Y, X), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _circle_mask(rec, Y: int, X: int) -> np.ndarray | None:
    p = rec.params
    cx, cy = float(p[0]), float(p[1])
    radius = float(np.hypot(p[2] - cx, p[3] - cy))
    if radius <= 0:
        return None
    yy, xx = np.ogrid[:Y, :X]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def _polygon_mask(rec, Y: int, X: int) -> np.ndarray | None:
    if not polyline_is_closed(rec):
        return None
    outline = get_outline(rec)
    if len(outline) < 4:
        return None
    poly = outline[:-1]  # drop closing duplicate
    yy, xx = np.mgrid[:Y, :X]
    px = xx.ravel().astype(np.float32)
    py = yy.ravel().astype(np.float32)
    n = len(poly)
    inside = np.zeros(px.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        cond = ((yi > py) != (yj > py))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        crosses = cond & (px < x_cross)
        inside ^= crosses
        j = i
    return inside.reshape(Y, X)


def _mask_for(rec, Y: int, X: int) -> np.ndarray | None:
    if rec.shape_type == RECTANGLE:
        return _rect_mask(rec, Y, X)
    if rec.shape_type == CIRCLE:
        return _circle_mask(rec, Y, X)
    if rec.shape_type == POLYLINE:
        return _polygon_mask(rec, Y, X)
    return None


_STAT_KEYS = ["mean", "std", "sum", "min", "max", "n_pixels"]


class RegionStatisticsDialog(QDialog):
    def __init__(self, window, layer, shape_id: int, parent=None):
        super().__init__(parent if parent is not None else window)
        self.setWindowTitle(f"Region Statistics — {layer.data.get(shape_id).label or shape_id}")
        # Tool flag keeps the dialog floating above the parent without
        # stealing focus or interfering with canvas mouse routing — same
        # pattern KymographDialog uses.
        self.setWindowFlags(Qt.Tool)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self._window = window
        self._layer = layer
        self._shape_id = int(shape_id)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        self._all_channels = QCheckBox("All channels")
        self._all_channels.setChecked(True)
        self._all_channels.toggled.connect(self._refresh)
        outer.addWidget(self._all_channels)

        self._table = QTableWidget(0, len(_STAT_KEYS) + 1)
        headers = ["channel"] + _STAT_KEYS
        self._table.setHorizontalHeaderLabels(headers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        outer.addWidget(self._table)

        self._status = QLabel("")
        outer.addWidget(self._status)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh)
        btn_row.addWidget(refresh)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        btn_row.addWidget(close)
        outer.addLayout(btn_row)

        self._unsubs: list = []
        self._unsubs.append(layer.data.subscribe(self._on_shape_event))
        if hasattr(window, "view_changed"):
            try:
                window.view_changed.connect(self._refresh)
                self._unsubs.append(("view_changed", window.view_changed))
            except Exception:
                pass

        self._refresh()

    def _on_shape_event(self, kind, sid):
        if kind == EVT_REMOVED and int(sid) == self._shape_id:
            self.close()
            return
        if sid != self._shape_id and sid != -1:
            return
        self._refresh()

    def _refresh(self, *_args):
        if self._shape_id not in self._layer.data:
            return
        rec = self._layer.data.get(self._shape_id)
        try:
            frame = np.asarray(
                self._window.img_data[
                    self._window.t_idx, self._window.z_idx, :, :, :
                ]
            )
        except Exception as e:
            self._status.setText(f"Could not read frame: {e}")
            return
        C, Y, X = frame.shape
        mask = _mask_for(rec, Y, X)
        if mask is None or not mask.any():
            self._status.setText("Region is empty or shape is not maskable.")
            self._table.setRowCount(0)
            return

        channels = list(range(C)) if self._all_channels.isChecked() else [self._window.c_idx]
        channels = [c for c in channels if 0 <= c < C]

        self._table.setRowCount(len(channels))
        for r, c in enumerate(channels):
            data = frame[c][mask]
            stats = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "sum": float(np.sum(data, dtype=np.float64)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "n_pixels": int(data.size),
            }
            self._table.setItem(r, 0, QTableWidgetItem(self._channel_label(c)))
            for j, key in enumerate(_STAT_KEYS, start=1):
                val = stats[key]
                txt = f"{val:.3f}" if key != "n_pixels" else f"{int(val)}"
                self._table.setItem(r, j, QTableWidgetItem(txt))
        self._status.setText(
            f"Frame t={self._window.t_idx}, z={self._window.z_idx} · "
            f"{int(mask.sum())} px in mask"
        )

    def _channel_label(self, c: int) -> str:
        meta_channels = (self._window.meta or {}).get("channels") or []
        if c < len(meta_channels) and isinstance(meta_channels[c], dict):
            name = meta_channels[c].get("name")
            if name:
                return f"{c} — {name}"
        return f"{c}"

    def closeEvent(self, event):
        for u in self._unsubs:
            try:
                if callable(u):
                    u()
                else:
                    kind, sig = u
                    sig.disconnect(self._refresh)
            except Exception:
                pass
        self._unsubs.clear()
        super().closeEvent(event)
