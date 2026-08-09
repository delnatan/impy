"""RegionStatisticsDialog — per-channel summary stats inside a shape.

Modeless dialog showing mean / std / sum / min / max / area for the pixels
inside the selected shape, computed on the current ``(t, z)`` frame. A
checkbox toggles whether all channels are shown or just the currently
displayed one.

Supported shape types: :data:`RECTANGLE`, :data:`CIRCLE`, :data:`POLYLINE`
(closed). Auto-refreshes via :class:`ShapeData.subscribe` and the target
window's ``view_changed`` signal.

The shape is drawn on a "source" window, but a "Sample from" picker lets it
be evaluated against any other open window with physical calibration
(``meta["scale"]``) — the shape's geometry is converted into the target's
own pixel grid via :func:`~pyvistra.data.shapes.rescale_shape` before
masking, the same "convert the geometry, not an already-rasterized result"
approach line/radial profile use for paths.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..data import calibration
from ..data.shapes import EVT_REMOVED, mask_for, rescale_shape
from .window_picker import list_other_image_windows

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
        self._target_window = window
        self._target_view_signal = None
        self._target_closing_signal = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Sample from:"))
        self._target_combo = QComboBox()
        self._populate_target_combo()
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)
        target_row.addWidget(self._target_combo, 1)
        outer.addLayout(target_row)

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

        # The source window owning the shape closing makes the shape itself
        # unreachable regardless of which window is currently the sample
        # target -- close the dialog. Connected once, for the dialog's
        # whole lifetime (unlike the target's signals, which change with
        # the combo box).
        self._window.window_closing.connect(self._on_source_window_closing)
        self._connect_target_signals()

        self._refresh()

    # ------------------------------------------------------------------
    # Target-window picker
    # ------------------------------------------------------------------
    def _populate_target_combo(self):
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        self._target_combo.addItem(
            f"[{self._window.window_id}] {self._window.windowTitle()} (source)", self._window
        )
        for w in list_other_image_windows(exclude=self._window):
            self._target_combo.addItem(f"[{w.window_id}] {w.windowTitle()}", w)
        idx = self._target_combo.findData(self._target_window)
        self._target_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._target_combo.blockSignals(False)

    def _on_target_changed(self, index):
        window = self._target_combo.itemData(index)
        if window is None or window is self._target_window:
            return
        self._disconnect_target_signals()
        self._target_window = window
        self._connect_target_signals()
        self._refresh()

    def _connect_target_signals(self):
        w = self._target_window
        if hasattr(w, "view_changed"):
            try:
                w.view_changed.connect(self._refresh)
                self._target_view_signal = w
            except Exception:
                pass
        # The source window's own closing is handled separately (and
        # always) by _on_source_window_closing; only a *different* target
        # needs its own closing handler.
        if w is not self._window:
            try:
                w.window_closing.connect(self._on_target_window_closing)
                self._target_closing_signal = w
            except Exception:
                pass

    def _disconnect_target_signals(self):
        if self._target_view_signal is not None:
            try:
                self._target_view_signal.view_changed.disconnect(self._refresh)
            except Exception:
                pass
            self._target_view_signal = None
        if self._target_closing_signal is not None:
            try:
                self._target_closing_signal.window_closing.disconnect(self._on_target_window_closing)
            except Exception:
                pass
            self._target_closing_signal = None

    def _on_source_window_closing(self, window):
        self.close()

    def _on_target_window_closing(self, window):
        # Fall back to the source window rather than leaving a dangling
        # reference to a closed target.
        self._disconnect_target_signals()
        self._target_window = self._window
        self._connect_target_signals()
        self._populate_target_combo()
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
        target = self._target_window
        cross_window = target is not self._window

        if cross_window:
            src_scale = calibration.window_scale_yx(self._window)
            dst_scale = calibration.window_scale_yx(target)
            if src_scale is None or dst_scale is None:
                self._status.setText(
                    "Cannot sample: source and target windows must both have "
                    "physical calibration (scale) to compare."
                )
                self._table.setRowCount(0)
                return
            if calibration.window_is_frequency_space(self._window) != calibration.window_is_frequency_space(target):
                self._status.setText(
                    "Cannot sample: source and target are not the same kind "
                    "of pixel space (real vs. frequency)."
                )
                self._table.setRowCount(0)
                return
            rec = rescale_shape(rec, src_scale, dst_scale)

        try:
            frame = np.asarray(
                target.img_data[target.t_idx, target.z_idx, :, :, :]
            )
        except Exception as e:
            self._status.setText(f"Could not read frame: {e}")
            return
        C, Y, X = frame.shape
        mask = mask_for(rec, Y, X)
        if mask is None or not mask.any():
            self._status.setText("Region is empty or shape is not maskable.")
            self._table.setRowCount(0)
            return

        channels = list(range(C)) if self._all_channels.isChecked() else [target.c_idx]
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
            self._table.setItem(r, 0, QTableWidgetItem(self._channel_label(target, c)))
            for j, key in enumerate(_STAT_KEYS, start=1):
                val = stats[key]
                txt = f"{val:.3f}" if key != "n_pixels" else f"{int(val)}"
                self._table.setItem(r, j, QTableWidgetItem(txt))
        target_note = f" on [{target.window_id}]" if cross_window else ""
        self._status.setText(
            f"Frame t={target.t_idx}, z={target.z_idx}{target_note} · "
            f"{int(mask.sum())} px in mask"
        )

    @staticmethod
    def _channel_label(window, c: int) -> str:
        meta_channels = (window.meta or {}).get("channels") or []
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

        self._disconnect_target_signals()
        try:
            self._window.window_closing.disconnect(self._on_source_window_closing)
        except Exception:
            pass

        super().closeEvent(event)
