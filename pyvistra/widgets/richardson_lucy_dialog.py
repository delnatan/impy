"""Deconvolution dialog — Richardson-Lucy engine
(deconlib.deconvolution.richardson_lucy_*).

Two-tab layout, mirroring the NLCG dialog (`deconvolution_dialog.py`):
    * Source, PSF && Model — ROI / frame / channel / PSF window picker,
      forward-model zoom (shared `SourcePSFPanelMixin`).
    * Solver — RL iterations/eval-interval/verbose, output region, and
      optional tiling for large fields. RL has no regularizer in deconlib
      (early stopping alone controls noise amplification), so this tab
      is simpler than NLCG's.

The dialog reads its widgets into a :class:`~.decon_rl.RLDialogState`,
builds a :class:`~.decon_rl.PreparedInputs`, and hands the result to
:class:`~.richardson_lucy_worker.RLDeconvolutionWorker`.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .decon_source_panel import SourcePSFPanelMixin
from .output_selector import ImageOutputSelector
from .processing_helper import BufferProcessingRunner
from . import decon_rl as drl
from .richardson_lucy_worker import RLDeconvolutionWorker


class RichardsonLucyDialog(SourcePSFPanelMixin, QDialog):
    """Single-channel 2D/3D deconvolution dialog (Richardson-Lucy)."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Richardson-Lucy Deconvolution")
        self.setWindowFlags(Qt.Tool)
        self.resize(600, 640)
        self.setMinimumSize(540, 460)

        self._runner = None
        self._running_status_base: Optional[str] = None
        self._run_started_at: Optional[float] = None
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_running_status)

        self._init_source_psf_panel()

        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 8)
        main.setSpacing(6)

        tabs = QTabWidget()
        tabs.addTab(self._build_source_psf_tab(), "Source, PSF && Model")
        tabs.addTab(self._build_solver_tab(), "Solver")
        main.addWidget(tabs, 1)

        output_group = QWidget()
        output_v = QVBoxLayout(output_group)
        output_v.setContentsMargins(0, 0, 0, 0)
        output_v.setSpacing(6)

        self.output_selector = ImageOutputSelector(
            default_title="Deconvolved (RL)",
            formats=[".tif", ".ims"],
        )
        output_v.addWidget(self.output_selector)

        self._runner = BufferProcessingRunner(self.viewer, self.output_selector)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        output_v.addWidget(self.progress_bar)

        self.status_label = QPlainTextEdit("Ready")
        self.status_label.setReadOnly(True)
        self.status_label.setFixedHeight(56)
        self.status_label.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.status_label.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.status_label.setStyleSheet("color: #888;")
        output_v.addWidget(self.status_label)

        main.addWidget(output_group)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start)
        btn_row.addWidget(self.start_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.cancel_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        btn_row.addStretch()
        main.addLayout(btn_row)

        self._finish_source_psf_panel_init()

        self._on_tiling_toggled(self.tiled_check.isChecked())
        self._refresh_compute_target_hint()
        self._refresh_recipe_preview()

    # ------------------------------------------------------------------ #
    # Solver tab

    def _build_solver_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        rl_grp = QGroupBox("Richardson-Lucy solver")
        grid = QGridLayout(rl_grp)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.num_iter_spin = QSpinBox()
        self.num_iter_spin.setRange(1, 5000)
        self.num_iter_spin.setValue(100)
        self.num_iter_spin.setFixedWidth(64)
        self.num_iter_spin.setToolTip(
            "Number of RL iterations. RL has no regularizer in deconlib — "
            "early stopping is the only control over noise amplification."
        )
        self.eval_interval_spin = QSpinBox()
        self.eval_interval_spin.setRange(1, 500)
        self.eval_interval_spin.setValue(5)
        self.eval_interval_spin.setFixedWidth(64)
        self.eval_interval_spin.setToolTip(
            "Interval (iterations) for live-preview writes and, if Verbose "
            "is on, the mean I-divergence readout."
        )
        self.verbose_check = QCheckBox("Verbose")
        self.verbose_check.setToolTip(
            "Print per-iteration diagnostics (mean I-divergence) to stdout, "
            "and show the running I-divergence in the status box below."
        )

        grid.addWidget(QLabel("Iterations:"), 0, 0, Qt.AlignRight)
        grid.addWidget(self.num_iter_spin, 0, 1)
        grid.addWidget(QLabel("Eval interval:"), 0, 2, Qt.AlignRight)
        grid.addWidget(self.eval_interval_spin, 0, 3)
        grid.addWidget(self.verbose_check, 1, 0, 1, 4)
        vbox.addWidget(rl_grp)

        out_grp = QGroupBox("Output")
        out_v = QVBoxLayout(out_grp)
        out_v.setContentsMargins(8, 6, 8, 6)
        out_v.setSpacing(6)

        region_row = QHBoxLayout()
        region_row.setSpacing(8)
        self.region_full_radio = QRadioButton("Full object domain")
        self.region_valid_radio = QRadioButton("Visible (cropped)")
        self.region_valid_radio.setChecked(True)
        region_row.addWidget(self.region_full_radio)
        region_row.addWidget(self.region_valid_radio)
        region_row.addStretch()
        out_v.addLayout(region_row)
        self._out_region_group = QButtonGroup(self)
        self._out_region_group.addButton(self.region_full_radio)
        self._out_region_group.addButton(self.region_valid_radio)
        self._out_region_group.buttonToggled.connect(
            lambda *_: self._refresh_recipe_preview()
        )
        self.region_full_radio.setToolTip(
            "Keep the full padded reconstruction domain, including the "
            "PSF-support margin deconlib adds automatically."
        )
        self.region_valid_radio.setToolTip(
            "Crop the result back down to the visible (zoom-scaled "
            "detector) field."
        )

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        out_v.addWidget(line)

        self.tiled_check = QCheckBox("Process in tiles (large field)")
        self.tiled_check.setToolTip(
            "Split the field into overlap-save tiles sharing one forward "
            "model, stitching owned cores back together. Tiled output is "
            "always cropped to the visible field."
        )
        self.tiled_check.toggled.connect(self._on_tiling_toggled)
        out_v.addWidget(self.tiled_check)

        tile_row = QWidget()
        tile_form = QFormLayout(tile_row)
        tile_form.setLabelAlignment(Qt.AlignRight)
        tile_form.setContentsMargins(0, 0, 0, 0)
        tile_form.setVerticalSpacing(4)
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(32, 8192)
        self.tile_size_spin.setValue(256)
        self.tile_size_spin.setFixedWidth(80)
        self.tile_size_spin.setToolTip(
            "Nominal core size per tile (data pixels, guard excluded)."
        )
        tile_form.addRow("Tile size:", self.tile_size_spin)
        self.guard_px_spin = QSpinBox()
        self.guard_px_spin.setRange(0, 256)
        self.guard_px_spin.setValue(0)
        self.guard_px_spin.setFixedWidth(80)
        self.guard_px_spin.setToolTip(
            "Guard pixels (data space) on each side of a tile's core. "
            "0 = deconlib's default (half the PSF's lateral width)."
        )
        tile_form.addRow("Guard px:", self.guard_px_spin)
        self.min_z_slices_spin = QSpinBox()
        self.min_z_slices_spin.setRange(1, 8192)
        self.min_z_slices_spin.setValue(48)
        self.min_z_slices_spin.setFixedWidth(80)
        self.min_z_slices_spin.setToolTip(
            "Keep the full Z extent in one tile when Nz is at or below this."
        )
        tile_form.addRow("Min Z slices:", self.min_z_slices_spin)
        out_v.addWidget(tile_row)
        self._tile_row_widget = tile_row

        vbox.addWidget(out_grp)

        preview_grp = QGroupBox("Recipe preview")
        preview_layout = QVBoxLayout(preview_grp)
        preview_layout.setContentsMargins(8, 6, 8, 6)
        preview_layout.setSpacing(4)
        self.recipe_preview = QPlainTextEdit()
        self.recipe_preview.setReadOnly(True)
        self.recipe_preview.setFixedHeight(70)
        self.recipe_preview.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.recipe_preview.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.recipe_preview.setStyleSheet(
            "font-family: Menlo, Consolas, monospace; font-size: 10px;"
        )
        preview_layout.addWidget(self.recipe_preview)
        vbox.addWidget(preview_grp)

        diag_grp = QGroupBox("Advanced diagnostics")
        diag_grp.setCheckable(True)
        diag_grp.setChecked(False)
        diag_form = QFormLayout(diag_grp)
        diag_form.setLabelAlignment(Qt.AlignRight)
        diag_form.setVerticalSpacing(3)
        diag_form.setContentsMargins(8, 6, 8, 6)
        self.detector_domain_label = QLabel("—")
        self.visible_domain_label = QLabel("—")
        self.psf_kernel_label = QLabel("—")
        self.object_domain_label = QLabel("—")
        diag_form.addRow("Detector field:", self.detector_domain_label)
        diag_form.addRow("Visible shape:", self.visible_domain_label)
        diag_form.addRow("PSF kernel:", self.psf_kernel_label)
        diag_form.addRow("Object/padded shape:", self.object_domain_label)
        self._diagnostics_group = diag_grp
        diag_grp.toggled.connect(self._on_diagnostics_toggled)
        vbox.addWidget(diag_grp)

        vbox.addStretch()
        for spin in (
            self.num_iter_spin, self.eval_interval_spin,
            self.tile_size_spin, self.guard_px_spin, self.min_z_slices_spin,
        ):
            spin.valueChanged.connect(lambda *_: self._refresh_recipe_preview())
        self._on_diagnostics_toggled(False)
        return tab

    def _on_diagnostics_toggled(self, checked: bool) -> None:
        self._toggle_group_children(self._diagnostics_group, checked)

    def _on_tiling_toggled(self, checked: bool) -> None:
        self._tile_row_widget.setVisible(bool(checked))
        if checked:
            self.region_valid_radio.setChecked(True)
        self.region_full_radio.setEnabled(not checked)
        self.region_full_radio.setToolTip(
            "Tiled runs always output the visible/cropped field."
            if checked else
            "Keep the full padded reconstruction domain, including the "
            "PSF-support margin deconlib adds automatically."
        )
        self._refresh_recipe_preview()

    # ------------------------------------------------------------------ #
    # Mixin hooks

    def _on_source_psf_changed(self) -> None:
        if hasattr(self, "compute_target_label"):
            self._refresh_compute_target_hint()
        self._refresh_recipe_preview()

    # ------------------------------------------------------------------ #
    # Recipe preview

    def _current_state_for_shape(self) -> drl.RLDialogState:
        return drl.RLDialogState(
            zoom_xy=self.sr_xy_spin.value(),
            zoom_z=self.sr_z_spin.value(),
            crop_to_visible=self.region_valid_radio.isChecked(),
            tiled=self.tiled_check.isChecked(),
            tile_size=self.tile_size_spin.value(),
            guard_px=self.guard_px_spin.value(),
            min_z_slices=self.min_z_slices_spin.value(),
        )

    def _refresh_recipe_preview(self) -> None:
        if not hasattr(self, "recipe_preview"):
            return
        data_shape = self._current_data_shape()
        if data_shape is None:
            self.recipe_preview.setPlainText(
                "Select a detector region to preview the recipe."
            )
            return
        state = self._current_state_for_shape()
        ndim = len(data_shape)
        zoom = drl.per_axis_zoom(state, ndim)
        visible = drl.visible_shape(data_shape, zoom)
        kernel = self._current_psf_kernel_shape()
        padded = (
            drl.padded_shape(visible, kernel)
            if kernel is not None and len(kernel) == ndim
            else visible
        )
        output_spatial_shape = drl.output_shape(
            state, data_shape, kernel or tuple(1 for _ in range(ndim))
        )

        lines = [
            f"Richardson-Lucy, {self.num_iter_spin.value()} iter",
            f"data {self._format_shape(data_shape)} -> "
            f"visible {self._format_shape(visible)} -> "
            f"output {self._format_shape(output_spatial_shape)}",
        ]
        sr_text = " × ".join(f"{float(v):g}" for v in zoom)
        lines.append(f"zoom {sr_text}; object/padded shape {self._format_shape(padded)}")
        if self.tiled_check.isChecked():
            n_tiles, tile_shape = drl.estimate_tile_plan(state, data_shape)
            lines.append(
                f"tiled: ~{n_tiles} tile(s) of shape {self._format_shape(tile_shape)}"
            )
        self.recipe_preview.setPlainText("\n".join(lines))

        self.detector_domain_label.setText(self._format_shape(data_shape))
        self.visible_domain_label.setText(self._format_shape(visible))
        self.psf_kernel_label.setText(self._format_shape(kernel))
        self.object_domain_label.setText(self._format_shape(padded))

    # ------------------------------------------------------------------ #
    # State / inputs

    def _read_state(self) -> drl.RLDialogState:
        return drl.RLDialogState(
            zoom_xy=self.sr_xy_spin.value(),
            zoom_z=self.sr_z_spin.value(),
            num_iter=self.num_iter_spin.value(),
            background=self.background_spin.value(),
            eval_interval=self.eval_interval_spin.value(),
            verbose=self.verbose_check.isChecked(),
            crop_to_visible=self.region_valid_radio.isChecked(),
            tiled=self.tiled_check.isChecked(),
            tile_size=self.tile_size_spin.value(),
            guard_px=self.guard_px_spin.value(),
            min_z_slices=self.min_z_slices_spin.value(),
        )

    def _prepare(self, t: Optional[int] = None) -> Optional[object]:
        cropped = self._crop_observation_and_psf(t)
        if cropped is None:
            return None

        state = self._read_state()
        try:
            prepared = drl.prepare_inputs(
                state=state,
                y_obs=cropped.y_obs,
                psf_array=cropped.psf_data,
                psf_pixel_size_um=cropped.psf_pixel_size,
            )
        except Exception as exc:
            self._set_status(f"{type(exc).__name__}: {exc}", error=True)
            return None

        return prepared, state, cropped.t, cropped.c

    # ------------------------------------------------------------------ #
    # Run / cancel

    def _start(self):
        if self._runner.is_running():
            return
        frame_ts = list(self._frame_range())
        first = self._prepare(frame_ts[0])
        if first is None:
            return
        prepared, state, t, c = first

        stack_channels = self.stack_channels_check.isChecked()
        output_channels = self.viewer.C if stack_channels else 1
        output_channel = c if stack_channels else 0
        output_shape = drl.output_5d_shape(
            state, prepared.y.shape, prepared.psf.shape,
            n_channels=output_channels, n_frames=len(frame_ts),
        )
        drl.log.info(
            "output buffer shape=%s channel=%d (tiled=%s crop_to_visible=%s "
            "frames=%s)",
            output_shape, output_channel, state.tiled, state.crop_to_visible,
            frame_ts,
        )

        # Output spacing = fine-grid spacing (data spacing ÷ zoom), corrected
        # for the rounding `visible_shape` applies when data_shape * zoom
        # isn't an exact integer.
        vs = drl.effective_voxel_spacing(
            prepared.voxel_spacing, prepared.zoom, prepared.y.shape
        )
        if len(vs) == 2:
            out_scale = (1.0, float(vs[0]), float(vs[1]))
        else:
            out_scale = tuple(float(s) for s in vs)
        meta_channels = list((self.viewer.meta or {}).get("channels") or [])
        if stack_channels:
            channels = [
                meta_channels[i]
                if i < len(meta_channels)
                else {"name": f"Channel {i}"}
                for i in range(output_channels)
            ]
        else:
            channels = [
                meta_channels[c]
                if c < len(meta_channels)
                else {"name": f"Channel {c}"}
            ]

        output_meta = {
            "filename": "Deconvolved (RL)",
            "scale": out_scale,
            "is_rgb": False,
            "channels": channels,
            "source_channel": c,
            "source_frame": frame_ts[0] if len(frame_ts) == 1 else list(frame_ts),
            "algorithm": "richardson_lucy",
        }

        def prepare_for_t(frame_t):
            result = self._prepare(frame_t)
            if result is None:
                return None
            frame_prepared, frame_state, _t, _c = result
            return frame_prepared, frame_state

        def make_worker(frame_data, output_frame):
            frame_prepared, frame_state = frame_data
            return RLDeconvolutionWorker(
                prepared=frame_prepared,
                state=frame_state,
                buffer=self._runner.output_buffer,
                output_channel=output_channel,
                output_frame=output_frame,
            )

        if state.tiled:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, max(1, state.num_iter))

        self.progress_bar.setValue(0)
        self._begin_run_status(self._describe_run(state))
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self._runner.run_frames(
            frame_ts=frame_ts,
            output_shape=output_shape,
            output_dtype=np.dtype(np.float32),
            output_meta=output_meta,
            reuse_existing_buffer=stack_channels,
            prepare_for_t=prepare_for_t,
            make_worker=make_worker,
            on_progress=self._on_progress,
            on_status=self._on_status,
            on_all_finished=self._on_finished,
            on_cancelled=self._on_cancelled,
            on_error=self._on_error,
        )

    def _cancel(self):
        if self._runner.worker is not None:
            self._runner.cancel()
            self._stop_run_status()
            self._set_status("Cancelling…", warn=True)
            self.cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------ #
    # Signal handlers

    def _on_progress(self, done, total):
        if total > 0 and self.progress_bar.maximum() != total:
            self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)

    def _on_status(self, msg: str):
        self._begin_run_status(msg)

    def _on_finished(self, result=None):
        self._stop_run_status()
        if self._runner.output_type == "file":
            result = self._runner.finalize_output()
            msg = (
                f"Done — saved to {result}" if result else "Done (save cancelled)"
            )
            self._set_status(msg, ok=bool(result), warn=not bool(result))
        else:
            self._set_status("Done", ok=True)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_cancelled(self):
        self._stop_run_status()
        self._set_status("Cancelled", warn=True)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_error(self, message):
        self._stop_run_status()
        self._set_status(f"Error: {message}", error=True)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _set_status(self, msg: str, *, ok=False, warn=False, error=False):
        self.status_label.setPlainText(msg)
        color = "#4A4" if ok else "#C84" if warn else "#F44" if error else "#888"
        self.status_label.setStyleSheet(f"color: {color};")

    def _begin_run_status(self, msg: str) -> None:
        self._running_status_base = msg
        if self._run_started_at is None:
            self._run_started_at = time.monotonic()
        self._refresh_running_status()
        self._status_timer.start()

    def _stop_run_status(self) -> None:
        self._status_timer.stop()
        self._running_status_base = None
        self._run_started_at = None

    def _refresh_running_status(self) -> None:
        if self._running_status_base is None:
            return
        elapsed = 0.0 if self._run_started_at is None else (
            time.monotonic() - self._run_started_at
        )
        minutes, seconds = divmod(int(elapsed), 60)
        self.status_label.setPlainText(
            f"{self._running_status_base}\nElapsed: {minutes:02d}:{seconds:02d}"
        )
        self.status_label.setStyleSheet("color: #888;")

    def _describe_run(self, state: drl.RLDialogState) -> str:
        if state.tiled:
            return f"Running tiled Richardson-Lucy for up to {state.num_iter} iterations per tile."
        return f"Running Richardson-Lucy for up to {state.num_iter} iterations."

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_psf_combo()
        self._refresh_shape_combo()

    def closeEvent(self, event):
        if self._runner and self._runner.worker is not None:
            self._cancel()
            event.ignore()
            return
        super().closeEvent(event)
