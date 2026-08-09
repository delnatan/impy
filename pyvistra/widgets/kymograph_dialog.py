"""KymographDialog — sample a LINE/POLYLINE shape across time.

For each frame ``t`` in ``[t_start, t_end]``, the worker samples the
source 2D plane (at the chosen Z) along the shape's path at
``num_points`` evenly spaced arc-length positions, and writes one row
into an ``ImageBuffer``. The result is a 5D ``(1, 1, C, T_window,
num_points)`` array — a standard kymograph (time on Y, position on X).

Threading and output routing reuse :class:`BufferProcessingRunner` and
:class:`ImageOutputSelector`, so destinations (new window / existing
window / file) come for free.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .. import colors as tokens
from ..data.shapes import (
    ALL_FRAMES,
    LINE,
    POLYLINE,
    polyline_is_closed,
)
from .output_selector import ImageOutputSelector
from .processing_helper import BufferProcessingRunner


def _sample_coords(rec, num_points: int) -> np.ndarray:
    """Return ``(num_points, 2)`` ``(x, y)`` coordinates along the shape."""
    if rec.shape_type == LINE:
        p = rec.params
        xs = np.linspace(float(p[0]), float(p[2]), int(num_points))
        ys = np.linspace(float(p[1]), float(p[3]), int(num_points))
        return np.column_stack([xs, ys])
    if rec.shape_type == POLYLINE and rec.vertices is not None:
        verts = rec.vertices.astype(np.float64, copy=False)
        if polyline_is_closed(rec):
            verts = np.vstack([verts, verts[0:1]])
        seg = np.diff(verts, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        if seg_len.sum() == 0:
            xs = np.full(num_points, verts[0, 0])
            ys = np.full(num_points, verts[0, 1])
            return np.column_stack([xs, ys])
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        s = np.linspace(0.0, cum[-1], int(num_points))
        xs = np.interp(s, cum, verts[:, 0])
        ys = np.interp(s, cum, verts[:, 1])
        return np.column_stack([xs, ys])
    raise ValueError(f"Unsupported shape type for kymograph: {rec.shape_type!r}")


def _path_length(rec) -> float:
    coords = _sample_coords(rec, max(2, int(np.ceil(_raw_pixel_length(rec)))))
    seg = np.diff(coords, axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def _raw_pixel_length(rec) -> float:
    if rec.shape_type == LINE:
        p = rec.params
        return float(np.hypot(p[2] - p[0], p[3] - p[1]))
    if rec.shape_type == POLYLINE and rec.vertices is not None:
        verts = rec.vertices
        if polyline_is_closed(rec):
            verts = np.vstack([verts, verts[0:1]])
        seg = np.diff(verts, axis=0)
        return float(np.hypot(seg[:, 0], seg[:, 1]).sum())
    return 0.0


class KymographWorker(QObject):
    """Background worker — samples ``frame`` along the path for each ``t``."""

    progress = Signal(int, int)
    finished = Signal()
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, source, buffer, params):
        super().__init__()
        self._source = source
        self._buffer = buffer
        self._coords = np.asarray(params["coords"], dtype=np.float64)
        self._z = int(params["z"])
        self._t_start = int(params["t_start"])
        self._t_end = int(params["t_end"])
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from scipy.ndimage import map_coordinates

            _T, _Z, C, _Y, _X = self._source.shape
            t_values = range(self._t_start, self._t_end + 1)
            total = max(1, len(list(range(self._t_start, self._t_end + 1))) * C)
            done = 0
            # map_coordinates takes [rows, cols] = [y, x].
            ys = self._coords[:, 1]
            xs = self._coords[:, 0]
            for row_idx, t in enumerate(t_values):
                if self._cancel:
                    self.cancelled.emit()
                    return
                for c in range(C):
                    if self._cancel:
                        self.cancelled.emit()
                    frame = np.asarray(self._source[t, self._z, c, :, :])
                    sampled = map_coordinates(
                        frame, [ys, xs], order=1, mode="nearest"
                    )
                    self._buffer[0, 0, c, row_idx, :] = np.asarray(
                        sampled, dtype=self._buffer.dtype
                    )
                    done += 1
                    self.progress.emit(done, total)
            self.finished.emit()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class KymographDialog(QDialog):
    """Modeless kymograph dialog bound to a single ``(layer, shape_id)``."""

    def __init__(self, viewer, layer, shape_id: int, parent=None):
        super().__init__(parent if parent is not None else viewer)
        self.setWindowTitle("Kymograph")
        self.setWindowFlags(Qt.Tool)
        self.resize(440, 320)
        self._viewer = viewer
        self._layer = layer
        self._shape_id = int(shape_id)
        self._runner = None

        rec = layer.data.get(shape_id)
        path_len = _raw_pixel_length(rec)
        default_npoints = max(2, int(np.ceil(path_len)))

        T, Z, _C, _Y, _X = viewer.img_data.shape
        max_t = max(0, T - 1)
        max_z = max(0, Z - 1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        shape_lbl = QLabel(
            f"{rec.shape_type.capitalize()} #{shape_id} · "
            f"length ≈ {path_len:.1f} px"
        )
        outer.addWidget(shape_lbl)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._t_start = QSpinBox(); self._t_start.setRange(0, max_t); self._t_start.setValue(0)
        self._t_end = QSpinBox(); self._t_end.setRange(0, max_t); self._t_end.setValue(max_t)
        self._t_start.setKeyboardTracking(False)
        self._t_end.setKeyboardTracking(False)
        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("From:")); t_row.addWidget(self._t_start)
        t_row.addSpacing(10)
        t_row.addWidget(QLabel("To:")); t_row.addWidget(self._t_end)
        form.addRow(f"Time range (T = {T}):", t_row)

        self._z_spin = QSpinBox(); self._z_spin.setRange(0, max_z)
        default_z = (
            int(rec.z) if rec.z != ALL_FRAMES else int(viewer.z_idx)
        )
        self._z_spin.setValue(max(0, min(max_z, default_z)))
        self._z_spin.setEnabled(Z > 1)
        form.addRow(f"Z (Z = {Z}):", self._z_spin)

        self._npoints = QSpinBox()
        self._npoints.setRange(2, 100_000)
        self._npoints.setValue(default_npoints)
        self._npoints.setKeyboardTracking(False)
        form.addRow("Samples along path:", self._npoints)

        outer.addLayout(form)

        self._output_selector = ImageOutputSelector(
            default_title=f"Kymograph — shape {shape_id}",
            formats=[".tif", ".ims"],
        )
        outer.addWidget(self._output_selector)
        self._runner = BufferProcessingRunner(viewer, self._output_selector)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        outer.addWidget(self._progress)

        self._status = QLabel("Ready")
        self._status.setStyleSheet(f"color: {tokens.TEXT_FAINT};")
        outer.addWidget(self._status)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._start_btn = QPushButton("Start"); self._start_btn.clicked.connect(self._start)
        btn_row.addWidget(self._start_btn)
        self._cancel_btn = QPushButton("Cancel"); self._cancel_btn.setEnabled(False); self._cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self._cancel_btn)
        close = QPushButton("Close"); close.clicked.connect(self.close)
        btn_row.addWidget(close)
        outer.addLayout(btn_row)

    def _resample_coords(self):
        rec = self._layer.data.get(self._shape_id)
        n = int(self._npoints.value())
        return _sample_coords(rec, n)

    def _start(self):
        if self._runner.is_running():
            return
        if self._shape_id not in self._layer.data:
            self._status.setText("Shape no longer exists.")
            self._status.setStyleSheet(f"color: {tokens.DANGER};")
            return
        t0, t1 = sorted(
            (int(self._t_start.value()), int(self._t_end.value()))
        )
        n = int(self._npoints.value())
        coords = self._resample_coords()

        T, _Z, C, _Y, _X = self._viewer.img_data.shape
        T_window = t1 - t0 + 1

        meta = dict(self._viewer.meta or {})
        base_name = meta.get("filename") or self._viewer.windowTitle() or "image"
        meta["filename"] = f"{base_name} [kymograph]"

        source, buffer = self._runner.prepare_output(
            output_shape=(1, 1, C, T_window, n),
            output_dtype=np.dtype(self._viewer.img_data.dtype),
            output_meta=meta,
        )

        params = {
            "coords": coords,
            "z": int(self._z_spin.value()),
            "t_start": t0,
            "t_end": t1,
        }

        total = T_window * C
        self._progress.setRange(0, max(1, total))
        self._progress.setValue(0)
        self._status.setText(f"Running kymograph... 0/{total}")
        self._status.setStyleSheet(f"color: {tokens.TEXT_FAINT};")
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)

        worker = KymographWorker(source, buffer, params)
        self._runner.start_worker(
            worker=worker,
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_cancelled=self._on_cancelled,
            on_error=self._on_error,
            on_thread_finished=self._cleanup_thread,
        )

    def _cancel(self):
        if self._runner.worker is not None:
            self._runner.cancel()
            self._status.setText("Cancelling...")
            self._status.setStyleSheet(f"color: {tokens.WARNING};")
            self._cancel_btn.setEnabled(False)

    def _on_progress(self, done, total):
        self._progress.setValue(done)
        self._status.setText(f"Running... {done}/{total}")

    def _on_finished(self):
        if self._runner.output_type == "file":
            result = self._runner.finalize_output()
            if result:
                self._status.setText(f"Completed: saved to {result}")
                self._status.setStyleSheet(f"color: {tokens.SUCCESS};")
            else:
                self._status.setText("Completed (save cancelled)")
                self._status.setStyleSheet(f"color: {tokens.WARNING};")
        else:
            self._status.setText("Completed")
            self._status.setStyleSheet(f"color: {tokens.SUCCESS};")
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    def _on_cancelled(self):
        self._status.setText("Cancelled (partial result kept)")
        self._status.setStyleSheet(f"color: {tokens.WARNING};")
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    def _on_error(self, msg):
        self._status.setText(f"Error: {msg}")
        self._status.setStyleSheet(f"color: {tokens.DANGER};")
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)

    def _cleanup_thread(self):
        self._runner.cleanup()

    def closeEvent(self, event):
        if self._runner is not None and self._runner.worker is not None:
            self._cancel()
        super().closeEvent(event)
