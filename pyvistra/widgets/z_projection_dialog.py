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

from pyvistra.io import build_z_projection_metadata

from .output_selector import ImageOutputSelector
from .processing_helper import BufferProcessingRunner

# method name -> (reducer, output dtype is float)
_METHODS = {
    "max": (np.max, False),
    "min": (np.min, False),
    "mean": (np.mean, True),
    "sum": (np.sum, True),
    "std": (np.std, True),
    "median": (np.median, True),
}


class ZProjectionWorker(QObject):
    """Background worker for Z projections."""

    progress = Signal(int, int)  # done, total
    finished = Signal()
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, source, buffer, params):
        super().__init__()
        self._source = source
        self._buffer = buffer
        self._params = params
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            T, _Z, C, Y, X = self._source.shape
            z_start = int(self._params["z_start"])
            z_end = int(self._params["z_end"])
            method = str(self._params["method"]).lower()

            total = T * C
            done = 0

            for t in range(T):
                for c in range(C):
                    if self._cancel_requested:
                        self.cancelled.emit()
                        return

                    stack = np.asarray(
                        self._source[t, z_start : z_end + 1, c, :, :]
                    )

                    if method not in _METHODS:
                        raise ValueError(
                            f"Unsupported Z projection method: {method}"
                        )
                    reducer, _is_float = _METHODS[method]

                    if stack.ndim == 2:
                        stack = stack[np.newaxis, ...]
                    projected = reducer(stack, axis=0)

                    if projected.shape != (Y, X):
                        projected = np.asarray(projected).reshape(Y, X)

                    self._buffer[t, 0, c, :, :] = np.asarray(
                        projected, dtype=self._buffer.dtype
                    )

                    done += 1
                    self.progress.emit(done, total)

            self.finished.emit()

        except Exception as exc:
            self.error.emit(str(exc))


class ZProjectionDialog(QDialog):
    """Dialog for max-intensity projection along Z."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Z Projection")
        self.setWindowFlags(Qt.Tool)
        self.resize(420, 260)

        self._runner = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.method_combo = QComboBox()
        self.method_combo.addItem("Maximum Intensity", "max")
        self.method_combo.addItem("Minimum Intensity", "min")
        self.method_combo.addItem("Mean", "mean")
        self.method_combo.addItem("Sum", "sum")
        self.method_combo.addItem("Standard Deviation", "std")
        self.method_combo.addItem("Median", "median")
        form.addRow("Method:", self.method_combo)

        _T, Z, _C, _Y, _X = self.viewer.img_data.shape
        max_z = max(0, Z - 1)

        self.z_start_spin = QSpinBox()
        self.z_start_spin.setRange(0, max_z)
        self.z_start_spin.setValue(0)
        self.z_start_spin.setKeyboardTracking(False)
        self.z_start_spin.editingFinished.connect(self._on_z_range_changed)

        self.z_end_spin = QSpinBox()
        self.z_end_spin.setRange(0, max_z)
        self.z_end_spin.setValue(max_z)
        self.z_end_spin.setKeyboardTracking(False)
        self.z_end_spin.editingFinished.connect(self._on_z_range_changed)

        z_row = QHBoxLayout()
        z_row.setContentsMargins(0, 0, 0, 0)
        z_row.addWidget(QLabel("Start:"))
        z_row.addWidget(self.z_start_spin)
        z_row.addSpacing(10)
        z_row.addWidget(QLabel("End:"))
        z_row.addWidget(self.z_end_spin)
        form.addRow("Z range:", z_row)

        main_layout.addLayout(form)

        self.output_selector = ImageOutputSelector(
            default_title="Z Projection",
            formats=[".tif", ".ims"],
        )
        main_layout.addWidget(self.output_selector)
        self._runner = BufferProcessingRunner(self.viewer, self.output_selector)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")
        main_layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.addStretch()

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start)
        buttons.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        buttons.addWidget(self.cancel_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        buttons.addStretch()

        main_layout.addLayout(buttons)

    def _on_z_range_changed(self):
        start = self.z_start_spin.value()
        end = self.z_end_spin.value()
        if start > end:
            self.z_end_spin.setValue(start)

    def _collect_params(self):
        return {
            "method": self.method_combo.currentData(),
            "z_start": int(self.z_start_spin.value()),
            "z_end": int(self.z_end_spin.value()),
        }

    def _start(self):
        if self._runner.is_running():
            return

        T, Z, C, Y, X = self.viewer.img_data.shape
        z_start = int(self.z_start_spin.value())
        z_end = int(self.z_end_spin.value())
        if z_start < 0 or z_end >= Z:
            self.status_label.setText(
                f"Invalid Z range: {z_start}-{z_end} (valid: 0-{Z - 1})"
            )
            self.status_label.setStyleSheet("color: #F44;")
            return

        method = str(self.method_combo.currentData())
        output_meta = build_z_projection_metadata(
            metadata=self.viewer.meta.copy(),
            source_shape=(T, Z, C, Y, X),
            z_range=(z_start, z_end),
            method=method,
        )

        _reducer, is_float = _METHODS[method]
        output_dtype = (
            np.dtype(np.float32) if is_float else np.dtype(self.viewer.img_data.dtype)
        )

        source, buffer = self._runner.prepare_output(
            output_shape=(T, 1, C, Y, X),
            output_dtype=output_dtype,
            output_meta=output_meta,
        )

        total = T * C
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(0)

        if self._runner.output_type == "file":
            self.status_label.setText(
                f"Running Z projection to file... 0/{total}"
            )
        else:
            self.status_label.setText(f"Running Z projection... 0/{total}")

        self.status_label.setStyleSheet("color: #888;")
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        worker = ZProjectionWorker(source, buffer, self._collect_params())
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
            self.status_label.setText("Cancelling after current projection...")
            self.status_label.setStyleSheet("color: #C84;")
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, done, total):
        self.progress_bar.setValue(done)
        self.status_label.setText(f"Running... {done}/{total}")

    def _on_finished(self):
        if self._runner.output_type == "file":
            result = self._runner.finalize_output()
            if result:
                self.status_label.setText(f"Completed: saved to {result}")
                self.status_label.setStyleSheet("color: #4A4;")
            else:
                self.status_label.setText("Completed (save cancelled)")
                self.status_label.setStyleSheet("color: #C84;")
        else:
            self.status_label.setText("Completed")
            self.status_label.setStyleSheet("color: #4A4;")

        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_cancelled(self):
        self.status_label.setText("Cancelled (partial result kept in buffer)")
        self.status_label.setStyleSheet("color: #C84;")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_error(self, message):
        self.status_label.setText(f"Error: {message}")
        self.status_label.setStyleSheet("color: #F44;")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _cleanup_thread(self):
        self._runner.cleanup()

    def closeEvent(self, event):
        if self._runner.worker is not None:
            self._cancel()
            event.ignore()
            return
        super().closeEvent(event)
