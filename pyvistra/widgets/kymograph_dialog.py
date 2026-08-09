"""KymographDialog — sample a LINE/POLYLINE shape across time.

For each frame ``t`` in ``[t_start, t_end]``, the worker samples the
target 2D plane (at the chosen Z) along the shape's path at
``num_points`` evenly spaced arc-length positions, and writes one row
into an ``ImageBuffer``. The result is a 5D ``(1, 1, C, T_window,
num_points)`` array — a standard kymograph (time on Y, position on X).

Threading and output routing reuse :class:`BufferProcessingRunner` and
:class:`ImageOutputSelector`, so destinations (new window / existing
window / file) come for free.

The shape is drawn on a "source" window, but a "Sample from" picker lets
it be evaluated against any other open window with physical calibration
(``meta["scale"]``) — the path is converted into the target's own pixel
grid via ``data/calibration.py`` before sampling, the same approach
line/radial profile and region-statistics use.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
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
from ..data import calibration
from ..data.shapes import (
    ALL_FRAMES,
    LINE,
    POLYLINE,
    polyline_is_closed,
)
from ..ui.manager import manager
from .output_selector import ImageOutputSelector
from .processing_helper import BufferProcessingRunner
from .window_picker import list_other_image_windows


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
        self.resize(440, 360)
        self._viewer = viewer
        self._layer = layer
        self._shape_id = int(shape_id)
        self._runner = None
        self._target_window = viewer
        self._target_closing_signal = None

        rec = layer.data.get(shape_id)
        path_len = _raw_pixel_length(rec)
        default_npoints = max(2, int(np.ceil(path_len)))
        self._default_z_anchor = int(rec.z) if rec.z != ALL_FRAMES else int(viewer.z_idx)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        shape_lbl = QLabel(
            f"{rec.shape_type.capitalize()} #{shape_id} · "
            f"length ≈ {path_len:.1f} px"
        )
        outer.addWidget(shape_lbl)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Sample from:"))
        self._target_combo = QComboBox()
        self._populate_target_combo()
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)
        target_row.addWidget(self._target_combo, 1)
        outer.addLayout(target_row)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._t_start = QSpinBox()
        self._t_end = QSpinBox()
        self._t_start.setKeyboardTracking(False)
        self._t_end.setKeyboardTracking(False)
        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("From:")); t_row.addWidget(self._t_start)
        t_row.addSpacing(10)
        t_row.addWidget(QLabel("To:")); t_row.addWidget(self._t_end)
        form.addRow("Time range:", t_row)

        self._z_spin = QSpinBox()
        form.addRow("Z:", self._z_spin)

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

        self._update_target_ranges(initial=True)
        self._connect_target_signals()

    # ------------------------------------------------------------------
    # Target-window picker
    # ------------------------------------------------------------------
    def _populate_target_combo(self):
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        self._target_combo.addItem(
            f"[{self._viewer.window_id}] {self._viewer.windowTitle()} (source)", self._viewer
        )
        for w in list_other_image_windows(exclude=self._viewer):
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
        self._update_target_ranges()

    def _connect_target_signals(self):
        w = self._target_window
        # Only a *different* target needs a closing guard here -- if the
        # source window itself closes, _start()'s existing shape-id check
        # already makes the dialog inert (pre-existing behavior, unchanged).
        if w is not self._viewer:
            try:
                w.window_closing.connect(self._on_target_window_closing)
                self._target_closing_signal = w
            except Exception:
                pass

    def _disconnect_target_signals(self):
        if self._target_closing_signal is not None:
            try:
                self._target_closing_signal.window_closing.disconnect(self._on_target_window_closing)
            except Exception:
                pass
            self._target_closing_signal = None

    def _on_target_window_closing(self, window):
        # Fall back to the source window rather than leaving a dangling
        # reference to a closed target.
        self._disconnect_target_signals()
        self._target_window = self._viewer
        self._connect_target_signals()
        self._populate_target_combo()
        self._update_target_ranges()

    def _update_target_ranges(self, initial: bool = False):
        """Refresh the T/Z spin-box ranges for the current target window.

        These were previously derived once, from the source window, at
        construction time -- now the target can change after construction,
        so they must be re-derived on every target change, not just once.
        """
        T, Z, _C, _Y, _X = self._target_window.img_data.shape
        max_t = max(0, T - 1)
        max_z = max(0, Z - 1)

        self._t_start.setRange(0, max_t)
        self._t_end.setRange(0, max_t)
        if initial:
            self._t_start.setValue(0)
            self._t_end.setValue(max_t)
        else:
            self._t_start.setValue(min(self._t_start.value(), max_t))
            self._t_end.setValue(min(self._t_end.value(), max_t))

        self._z_spin.setRange(0, max_z)
        self._z_spin.setEnabled(Z > 1)
        if initial:
            self._z_spin.setValue(max(0, min(max_z, self._default_z_anchor)))
        else:
            self._z_spin.setValue(min(self._z_spin.value(), max_z))

    def _resample_coords(self):
        """Path in the current target window's own pixel grid.

        Sampled first in the source window's pixel space (unchanged), then
        converted through a shared physical coordinate into the target's
        grid when target != source — mirrors line/radial profile's path
        conversion. Returns ``None`` if either window lacks calibration;
        callers should already have refused via ``_cross_window_error``
        before reaching this, but this never raises either way.
        """
        rec = self._layer.data.get(self._shape_id)
        n = int(self._npoints.value())
        coords = _sample_coords(rec, n)
        if self._target_window is self._viewer:
            return coords
        src_scale = calibration.window_scale_yx(self._viewer)
        dst_scale = calibration.window_scale_yx(self._target_window)
        if src_scale is None or dst_scale is None:
            return None
        phys = calibration.points_px_to_phys(coords, src_scale)
        return calibration.points_phys_to_px(phys, dst_scale)

    def _cross_window_error(self) -> str | None:
        """``None`` if sampling ``self._target_window`` is currently valid;
        otherwise a user-facing reason it isn't."""
        if self._target_window is self._viewer:
            return None
        src_scale = calibration.window_scale_yx(self._viewer)
        dst_scale = calibration.window_scale_yx(self._target_window)
        if src_scale is None or dst_scale is None:
            return (
                "Cannot sample: source and target windows must both have "
                "physical calibration (scale) to compare."
            )
        if calibration.window_is_frequency_space(self._viewer) != calibration.window_is_frequency_space(
            self._target_window
        ):
            return (
                "Cannot sample: source and target are not the same kind "
                "of pixel space (real vs. frequency)."
            )
        return None

    def _start(self):
        if self._runner.is_running():
            return
        if self._shape_id not in self._layer.data:
            self._status.setText("Shape no longer exists.")
            self._status.setStyleSheet(f"color: {tokens.DANGER};")
            return

        # Defensive re-check: normally the target is kept in sync by
        # _on_target_window_closing (a direct, same-thread signal
        # connection fires synchronously on window_closing), but don't
        # trust a stale reference right before reading its data.
        if self._target_window is not self._viewer and self._target_window not in manager.get_all().values():
            self._disconnect_target_signals()
            self._target_window = self._viewer
            self._connect_target_signals()
            self._populate_target_combo()
            self._update_target_ranges()

        err = self._cross_window_error()
        if err is not None:
            self._status.setText(err)
            self._status.setStyleSheet(f"color: {tokens.DANGER};")
            return

        t0, t1 = sorted(
            (int(self._t_start.value()), int(self._t_end.value()))
        )
        n = int(self._npoints.value())
        coords = self._resample_coords()
        if coords is None:
            self._status.setText("Could not resolve sample coordinates.")
            self._status.setStyleSheet(f"color: {tokens.DANGER};")
            return

        target = self._target_window
        T, _Z, C, _Y, _X = target.img_data.shape
        T_window = t1 - t0 + 1

        meta = dict(target.meta or {})
        base_name = meta.get("filename") or target.windowTitle() or "image"
        meta["filename"] = f"{base_name} [kymograph]"

        source, buffer = self._runner.prepare_output(
            output_shape=(1, 1, C, T_window, n),
            output_dtype=np.dtype(target.img_data.dtype),
            output_meta=meta,
            source_window=target,
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
        self._disconnect_target_signals()
        super().closeEvent(event)
