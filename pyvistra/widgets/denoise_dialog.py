import numpy as np
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .output_selector import ImageOutputSelector
from .processing_helper import BufferProcessingRunner


class DenoiseWorker(QObject):
    """Background worker for timelapse denoising."""

    progress = Signal(int, int)  # done, total
    plane_done = Signal(int, int, int)  # output_t, z, c
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
            from mlp_denoiser.inference import denoise_image

            _, Z, C, _, _ = self._source.shape
            t_start = int(self._params["t_start"])
            t_end = int(self._params["t_end"])
            t_indices = range(t_start, t_end + 1)
            total = len(t_indices) * Z * C
            done = 0

            for out_t, src_t in enumerate(t_indices):
                for z in range(Z):
                    for c in range(C):
                        if self._cancel_requested:
                            self.cancelled.emit()
                            return

                        plane = np.asarray(
                            self._source[src_t, z, c, :, :], dtype=np.float32
                        )

                        seed = int(self._params["seed"]) + done
                        denoised = denoise_image(
                            plane,
                            sigma=self._params["sigma"],
                            mapping_dim=self._params["mapping_dim"],
                            n_steps=self._params["n_steps"],
                            batch_size=self._params["batch_size"],
                            learning_rate=self._params["learning_rate"],
                            loss_type=self._params["loss_type"],
                            normalize_method=self._params["normalize_method"],
                            background=self._params["background"],
                            show_progress=False,
                            seed=seed,
                        )

                        self._buffer[out_t, z, c, :, :] = np.asarray(
                            denoised, dtype=np.float32
                        )

                        done += 1
                        self.progress.emit(done, total)
                        self.plane_done.emit(out_t, z, c)

            self.finished.emit()

        except Exception as exc:
            self.error.emit(str(exc))


class Denoise2DTimelapseDialog(QDialog):
    """Dialog for denoising a 2D timelapse into a streaming ImageBuffer."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Denoise 2D Timelapse")
        self.setWindowFlags(Qt.Tool)
        self.resize(440, 390)

        self._runner = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 100.0)
        self.sigma_spin.setValue(10.0)
        self.sigma_spin.setDecimals(2)
        form.addRow("Sigma:", self.sigma_spin)

        self.mapping_dim_spin = QSpinBox()
        self.mapping_dim_spin.setRange(16, 4096)
        self.mapping_dim_spin.setValue(256)
        form.addRow("Mapping dim:", self.mapping_dim_spin)

        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setRange(50, 200000)
        self.n_steps_spin.setValue(2000)
        form.addRow("Training steps:", self.n_steps_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(64, 131072)
        self.batch_size_spin.setSingleStep(64)
        self.batch_size_spin.setValue(4096)
        form.addRow("Batch size:", self.batch_size_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-7, 1.0)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setSingleStep(1e-5)
        self.learning_rate_spin.setValue(1e-4)
        form.addRow("Learning rate:", self.learning_rate_spin)

        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["poisson", "mse"])
        form.addRow("Loss:", self.loss_combo)

        self.normalize_combo = QComboBox()
        self.normalize_combo.addItem("auto", None)
        self.normalize_combo.addItem("minmax", "minmax")
        self.normalize_combo.addItem("mean", "mean")
        self.normalize_combo.addItem("percentile", "percentile")
        self.normalize_combo.addItem("none", "none")
        form.addRow("Normalize:", self.normalize_combo)

        self.background_spin = QDoubleSpinBox()
        self.background_spin.setRange(-1e6, 1e6)
        self.background_spin.setDecimals(3)
        self.background_spin.setValue(0.0)
        form.addRow("Background:", self.background_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)
        form.addRow("Seed:", self.seed_spin)

        # Time range
        T = self.viewer.img_data.shape[0]
        self.t_start_spin = QSpinBox()
        self.t_end_spin = QSpinBox()
        max_t = max(0, T - 1)
        self.t_start_spin.setRange(0, max_t)
        self.t_end_spin.setRange(0, max_t)
        self.t_start_spin.setValue(0)
        self.t_end_spin.setValue(max_t)
        # Validate on commit to avoid mid-typing jumps (e.g. entering "29").
        self.t_start_spin.setKeyboardTracking(False)
        self.t_end_spin.setKeyboardTracking(False)
        self.t_start_spin.editingFinished.connect(self._on_time_range_changed)
        self.t_end_spin.editingFinished.connect(self._on_time_range_changed)

        time_row = QHBoxLayout()
        time_row.setContentsMargins(0, 0, 0, 0)
        time_row.addWidget(QLabel("Start:"))
        time_row.addWidget(self.t_start_spin)
        time_row.addSpacing(10)
        time_row.addWidget(QLabel("End:"))
        time_row.addWidget(self.t_end_spin)
        form.addRow("Time range:", time_row)

        main_layout.addLayout(form)

        self.output_selector = ImageOutputSelector(
            default_title="Denoised Timelapse",
            formats=[("TIFF", ".tif"), ("Imaris", ".ims")],
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

    def _collect_params(self):
        return {
            "sigma": float(self.sigma_spin.value()),
            "mapping_dim": int(self.mapping_dim_spin.value()),
            "n_steps": int(self.n_steps_spin.value()),
            "batch_size": int(self.batch_size_spin.value()),
            "learning_rate": float(self.learning_rate_spin.value()),
            "loss_type": self.loss_combo.currentText(),
            "normalize_method": self.normalize_combo.currentData(),
            "background": float(self.background_spin.value()),
            "seed": int(self.seed_spin.value()),
            "t_start": int(self.t_start_spin.value()),
            "t_end": int(self.t_end_spin.value()),
        }

    def _on_time_range_changed(self):
        """Keep time range valid (start <= end)."""
        start = self.t_start_spin.value()
        end = self.t_end_spin.value()
        if start > end:
            # If invalid, snap end up to start.
            self.t_end_spin.setValue(start)

    def _start(self):
        if self._runner.is_running():
            return

        T, Z, C, Y, X = self.viewer.img_data.shape
        t_start = int(self.t_start_spin.value())
        t_end = int(self.t_end_spin.value())
        t_count = t_end - t_start + 1
        total = t_count * Z * C
        if total <= 0:
            self.status_label.setText("No data to process")
            self.status_label.setStyleSheet("color: #F44;")
            return

        output_meta = self.viewer.meta.copy()
        output_meta["source_time_range"] = [t_start, t_end]
        output_meta["filename"] = (
            f"{self.viewer.meta.get('filename', 'Image')}_denoised_t{t_start}-{t_end}"
        )
        source, buffer = self._runner.prepare_output(
            output_shape=(t_count, Z, C, Y, X),
            output_dtype=np.float32,
            output_meta=output_meta,
        )

        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        if self._runner.output_type == "file":
            self.status_label.setText(
                f"Running t={t_start}-{t_end} to file... 0/{total}"
            )
        else:
            self.status_label.setText(f"Running t={t_start}-{t_end}... 0/{total}")
        self.status_label.setStyleSheet("color: #888;")
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        worker = DenoiseWorker(
            source,
            buffer,
            self._collect_params(),
        )
        self._runner.start_worker(
            worker=worker,
            on_progress=self._on_progress,
            on_plane_done=self._on_plane_done,
            on_finished=self._on_finished,
            on_cancelled=self._on_cancelled,
            on_error=self._on_error,
            on_thread_finished=self._cleanup_thread,
        )

    def _cancel(self):
        if self._runner.worker is not None:
            self._runner.cancel()
            self.status_label.setText("Cancelling after current plane...")
            self.status_label.setStyleSheet("color: #C84;")
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, done, total):
        self.progress_bar.setValue(done)
        self.status_label.setText(f"Running... {done}/{total}")

    def _on_plane_done(self, t, z, _c):
        self._runner.refresh_output_view(t, z)

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
