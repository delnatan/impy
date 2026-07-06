"""Deconvolution dialog — NLCG engine (deconlib.deconvolution.nlcg_*).

Two-tab layout, mirroring the Richardson-Lucy/MaxEnt dialogs:
    * Source, PSF && Model — ROI / frame / channel / PSF window picker,
      forward-model zoom (shared `SourcePSFPanelMixin`).
    * Solver — optional regularizer, NLCG solver knobs, output region,
      and optional tiling for large fields.

The dialog reads its widgets into a :class:`~.decon_nlcg.NLCGDialogState`,
builds a :class:`~.decon_nlcg.PreparedInputs`, and hands the result to
:class:`~.deconvolution_worker.NLCGDeconvolutionWorker`.
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

from qtpy.QtGui import QValidator

from pyvistra.data.shapes import rectangle_bounds

from .decon_source_panel import SourcePSFPanelMixin
from .output_selector import ImageOutputSelector
from .processing_helper import BufferProcessingRunner
from . import decon_nlcg as dnl
from .deconvolution_worker import NLCGDeconvolutionWorker


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """A `QDoubleSpinBox` that displays/accepts scientific notation.

    Meant for small-magnitude tolerances/weights (e.g. ``1e-4``) where a
    fixed-decimals spinbox is either unreadable (too many zeros) or loses
    precision (too few decimals silently rounds the value to 0). Internal
    storage keeps full double precision (`decimals()` stays high); only
    display/typing uses scientific notation. Arrow keys step by one order
    of magnitude (×10 / ÷10) rather than a fixed linear increment, which is
    the natural step for this kind of value.
    """

    def __init__(self, parent=None, display_precision: int = 1):
        super().__init__(parent)
        self._display_precision = display_precision
        self.setDecimals(15)

    def textFromValue(self, value: float) -> str:
        return f"{value:.{self._display_precision}e}"

    def valueFromText(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return 0.0

    def validate(self, text: str, pos: int):
        text = text.strip()
        if text in ("", "-", "+"):
            return (QValidator.Intermediate, text, pos)
        try:
            float(text)
            return (QValidator.Acceptable, text, pos)
        except ValueError:
            pass
        import re
        if re.fullmatch(r"[+-]?\d*\.?\d*[eE][+-]?\d*", text):
            return (QValidator.Intermediate, text, pos)
        return (QValidator.Invalid, text, pos)

    def stepBy(self, steps: int) -> None:
        value = self.value()
        if value <= 0.0:
            floor = self.minimum() if self.minimum() > 0.0 else 1e-6
            self.setValue(floor if steps > 0 else 0.0)
            return
        self.setValue(value * (10.0 ** steps))


class DeconvolutionDialog(SourcePSFPanelMixin, QDialog):
    """Single-channel 2D/3D deconvolution dialog (NLCG)."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Deconvolution")
        self.setWindowFlags(Qt.Tool)
        self.resize(560, 540)
        self.setMinimumSize(520, 420)

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
        # Stretch factor 1: the tabs absorb all extra height on a vertical
        # resize (their scroll areas just need to scroll less), while the
        # output/progress/status group below -- added with the default
        # stretch of 0 -- stays at its natural size and so stays pinned to
        # the bottom instead of drifting or stretching along with it.
        main.addWidget(tabs, 1)

        output_group = QWidget()
        output_v = QVBoxLayout(output_group)
        output_v.setContentsMargins(0, 0, 0, 0)
        output_v.setSpacing(6)

        self.output_selector = ImageOutputSelector(
            default_title="Deconvolved",
            formats=[".tif", ".ims"],
        )
        output_v.addWidget(self.output_selector)

        self._runner = BufferProcessingRunner(self.viewer, self.output_selector)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        output_v.addWidget(self.progress_bar)

        # Use a read-only QPlainTextEdit instead of a QLabel so long error
        # messages (tracebacks, multi-line diagnostics) wrap inside the
        # widget instead of stretching the dialog horizontally.
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

        # apply initial visibility once everything is wired
        self._on_regularizer_mode_changed()
        self._on_tiling_toggled(self.tiled_check.isChecked())
        self._refresh_compute_target_hint()
        self._refresh_recipe_preview()

    # ------------------------------------------------------------------ #
    # Tab builders

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

        # Regularization (optional; off by default).
        reg_grp = QGroupBox("Regularization (optional)")
        reg_grp.setToolTip(
            "Early stopping alone is usually enough — leave this off unless "
            "you see residual axial streaking or noise amplification."
        )
        reg_layout = QHBoxLayout(reg_grp)
        reg_layout.setContentsMargins(8, 6, 8, 6)
        reg_layout.setSpacing(8)
        self.reg_none_radio = QRadioButton("Off")
        self.reg_none_radio.setChecked(True)
        self.reg_gradient_radio = QRadioButton("Gradient")
        self.reg_hessian_radio = QRadioButton("Hessian")
        reg_layout.addWidget(self.reg_none_radio)
        reg_layout.addWidget(self.reg_gradient_radio)
        reg_layout.addWidget(self.reg_hessian_radio)
        reg_layout.addSpacing(10)
        reg_layout.addWidget(QLabel("weight:"))
        self.reg_weight_spin = ScientificDoubleSpinBox()
        self.reg_weight_spin.setRange(0.0, 1e6)
        self.reg_weight_spin.setValue(0.0)
        self.reg_weight_spin.setFixedWidth(96)
        self.reg_weight_spin.setEnabled(False)
        self.reg_weight_spin.setToolTip(
            "Regularization weight (beta). Scene/SNR/scale dependent — start "
            "small (e.g. 1e-5) and increase until residual streaking/noise "
            "amplification is controlled without over-smoothing real structure."
        )
        reg_layout.addWidget(self.reg_weight_spin)
        reg_layout.addStretch()
        self._reg_group = QButtonGroup(self)
        self._reg_group.addButton(self.reg_none_radio)
        self._reg_group.addButton(self.reg_gradient_radio)
        self._reg_group.addButton(self.reg_hessian_radio)
        self._reg_group.buttonToggled.connect(self._on_regularizer_mode_changed)
        vbox.addWidget(reg_grp)

        # NLCG solver knobs — primary fields, with an "Advanced" collapsible
        # group for knobs that rarely need tuning.
        self._nlcg_box = self._build_nlcg_knobs()
        vbox.addWidget(self._nlcg_box)
        self._nlcg_advanced_box = self._build_nlcg_advanced_knobs()
        vbox.addWidget(self._nlcg_advanced_box)

        # Output region + tiling — one group: tiling forces the cropped
        # region, so they're presented together instead of as two boxes.
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
            lambda *_: self._on_source_psf_changed()
        )
        self.region_full_radio.setToolTip(
            "Keep the full padded reconstruction domain, including the "
            "PSF-support margin deconlib adds automatically."
        )
        self.region_valid_radio.setToolTip(
            "Crop the result back down to the visible (zoom-scaled "
            "detector) field."
        )
        self._output_group = out_grp

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
        for spin in (self.tile_size_spin, self.guard_px_spin, self.min_z_slices_spin):
            spin.valueChanged.connect(lambda *_: self._on_source_psf_changed())

        vbox.addWidget(out_grp)

        preview_grp = QGroupBox("Recipe preview")
        preview_layout = QVBoxLayout(preview_grp)
        preview_layout.setContentsMargins(8, 6, 8, 6)
        preview_layout.setSpacing(4)
        self.recipe_preview = QPlainTextEdit()
        self.recipe_preview.setReadOnly(True)
        self.recipe_preview.setFixedHeight(78)
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
        self._connect_recipe_preview_signals()
        self._on_diagnostics_toggled(False)
        return tab

    def _build_nlcg_knobs(self) -> QGroupBox:
        grp = QGroupBox("NLCG solver")
        grid = QGridLayout(grp)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        def _ispin(lo, hi, val, w=64):
            s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setFixedWidth(w); return s

        def _dspin(lo, hi, val, dec=3, step=None, w=72):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec)
            s.setValue(val); s.setFixedWidth(w)
            if step is not None: s.setSingleStep(step)
            return s

        self.num_iter_spin = _ispin(1, 5000, 150)
        self.slack_spin = _dspin(0.0, 10.0, 1.25, dec=3, step=0.05)
        self.slack_spin.setToolTip(
            "Discrepancy-principle target multiplier (unregularized only). "
            "0 disables it, falling through to the Eq. 17 test."
        )
        self.verbose_check = QCheckBox("Verbose")
        self.verbose_check.setToolTip(
            "Print per-iteration diagnostics (mean I-divergence, step length, "
            "CG mixing) to stdout, and show the running I-divergence in the "
            "status box below — useful for tuning."
        )

        grid.addWidget(QLabel("Iterations:"), 0, 0, Qt.AlignRight)
        grid.addWidget(self.num_iter_spin, 0, 1)
        grid.addWidget(QLabel("Slack:"), 0, 2, Qt.AlignRight)
        grid.addWidget(self.slack_spin, 0, 3)
        grid.addWidget(self.verbose_check, 1, 0, 1, 4)
        return grp

    def _build_nlcg_advanced_knobs(self) -> QGroupBox:
        grp = QGroupBox("Advanced solver knobs")
        grp.setCheckable(True)
        grp.setChecked(False)
        grid = QGridLayout(grp)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        def _ispin(lo, hi, val, w=64):
            s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setFixedWidth(w); return s

        def _dspin(lo, hi, val, dec=3, step=None, w=72):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec)
            s.setValue(val); s.setFixedWidth(w)
            if step is not None: s.setSingleStep(step)
            return s

        self.eval_interval_spin = _ispin(1, 500, 1)
        self.tol_spin = ScientificDoubleSpinBox()
        self.tol_spin.setRange(0.0, 1.0)
        self.tol_spin.setValue(1e-4)
        self.tol_spin.setFixedWidth(80)
        self.tol_spin.setToolTip(
            "Relative iterate-to-iterate change (Eq. 17). Stops early once it "
            "drops below this — the fallback safety net for when the "
            "discrepancy-principle target (Slack) is unreachable, or the "
            "primary stopping test when a regularizer is enabled. "
            "0 disables it."
        )
        self.min_iter_spin = _ispin(0, 500, 10)
        self.restart_interval_spin = _ispin(0, 500, 0)
        self.restart_interval_spin.setToolTip(
            "Force a steepest-descent restart every N iterations. 0 disables it."
        )
        self.newton_iters_spin = _ispin(1, 20, 3)

        def _row(r, la, wa, lb, wb):
            grid.addWidget(QLabel(la), r, 0, Qt.AlignRight)
            grid.addWidget(wa,         r, 1)
            grid.addWidget(QLabel(lb), r, 2, Qt.AlignRight)
            grid.addWidget(wb,         r, 3)

        _row(0, "Eval interval:", self.eval_interval_spin, "Convergence tol:", self.tol_spin)
        _row(1, "Min iter:", self.min_iter_spin, "Restart every:", self.restart_interval_spin)
        grid.addWidget(QLabel("Newton iters:"), 2, 0, Qt.AlignRight)
        grid.addWidget(self.newton_iters_spin, 2, 1)
        grp.toggled.connect(lambda checked: self._toggle_group_children(grp, checked))
        self._toggle_group_children(grp, False)
        return grp

    # ------------------------------------------------------------------ #
    # Dynamic visibility

    def _connect_recipe_preview_signals(self) -> None:
        widgets = [
            self.reg_none_radio, self.reg_gradient_radio, self.reg_hessian_radio,
        ]
        for widget in widgets:
            widget.toggled.connect(lambda *_: self._on_source_psf_changed())
        for spin in (
            self.num_iter_spin, self.slack_spin,
            self.reg_weight_spin, self.eval_interval_spin, self.tol_spin,
            self.min_iter_spin, self.restart_interval_spin, self.newton_iters_spin,
        ):
            spin.valueChanged.connect(lambda *_: self._on_source_psf_changed())

    # ------------------------------------------------------------------ #
    # Mixin hooks

    def _on_source_psf_changed(self) -> None:
        if hasattr(self, "compute_target_label"):
            self._refresh_compute_target_hint()
        if hasattr(self, "recipe_preview"):
            self._refresh_recipe_preview()

    def _on_regularizer_mode_changed(self, *_):
        self.reg_weight_spin.setEnabled(not self.reg_none_radio.isChecked())
        self._on_source_psf_changed()

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
        self._on_source_psf_changed()

    def _on_diagnostics_toggled(self, checked: bool) -> None:
        self._toggle_group_children(self._diagnostics_group, checked)

    def _current_state_for_shape(self) -> Optional[dnl.NLCGDialogState]:
        # A lightweight state good enough for shape math (zoom + region
        # knobs only); the full state is built in _read_state().
        return dnl.NLCGDialogState(
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
        zoom = dnl.per_axis_zoom(state, ndim)
        visible = dnl.visible_shape(data_shape, zoom)
        kernel = self._current_psf_kernel_shape()
        padded = (
            dnl.padded_shape(visible, kernel)
            if kernel is not None and len(kernel) == ndim
            else visible
        )
        output_spatial_shape = dnl.output_shape(
            state, data_shape, kernel or tuple(1 for _ in range(ndim))
        )

        reg_kind = (
            "gradient" if self.reg_gradient_radio.isChecked()
            else "hessian" if self.reg_hessian_radio.isChecked()
            else "off"
        )
        reg_text = (
            f"regularizer {reg_kind} (weight={self.reg_weight_spin.value():g})"
            if reg_kind != "off" else "no regularizer (early stopping only)"
        )

        lines = [
            f"NLCG, {self.num_iter_spin.value()} iter max, {reg_text}",
            f"data {self._format_shape(data_shape)} -> "
            f"visible {self._format_shape(visible)} -> "
            f"output {self._format_shape(output_spatial_shape)}",
        ]
        sr_text = " × ".join(f"{float(v):g}" for v in zoom)
        lines.append(f"zoom {sr_text}; object/padded shape {self._format_shape(padded)}")
        if self.tiled_check.isChecked():
            n_tiles, tile_shape = dnl.estimate_tile_plan(state, data_shape)
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

    def _read_state(self) -> dnl.NLCGDialogState:
        reg_kind = (
            "gradient" if self.reg_gradient_radio.isChecked()
            else "hessian" if self.reg_hessian_radio.isChecked()
            else "none"
        )
        return dnl.NLCGDialogState(
            zoom_xy=self.sr_xy_spin.value(),
            zoom_z=self.sr_z_spin.value(),
            regularizer_kind=reg_kind,
            reg_weight=self.reg_weight_spin.value(),
            num_iter=self.num_iter_spin.value(),
            background=self.background_spin.value(),
            eval_interval=self.eval_interval_spin.value(),
            slack=self.slack_spin.value(),
            tol=self.tol_spin.value(),
            min_iter=self.min_iter_spin.value(),
            restart_interval=self.restart_interval_spin.value(),
            newton_iters=self.newton_iters_spin.value(),
            verbose=self.verbose_check.isChecked(),
            crop_to_visible=self.region_valid_radio.isChecked(),
            tiled=self.tiled_check.isChecked(),
            tile_size=self.tile_size_spin.value(),
            guard_px=self.guard_px_spin.value(),
            min_z_slices=self.min_z_slices_spin.value(),
        )

    def _prepare(self, t: Optional[int] = None) -> Optional[object]:
        """Read inputs, slice the source, prepare the deconlib payload.

        `t` selects the frame to crop; defaults to `t_start_spin` (the
        first/only frame) when omitted. Returns ``(prepared, state, t, c)``
        on success, or `None` on a user error (with the status label
        updated).
        """
        _T, Z, _C, Y, X = self.viewer.img_data.shape
        c = self.channel_spin.value()
        if t is None:
            t = self.t_start_spin.value()

        if self.full_radio.isChecked():
            z_slice = slice(0, Z)
            y_slice = slice(0, Y)
            x_slice = slice(0, X)
        else:
            idx = self.shape_combo.currentIndex()
            if idx < 0 or idx >= len(self._rect_shapes):
                self._set_status("No rectangle shape selected.", error=True)
                return None
            _, rec = self._rect_shapes[idx]
            xl, yl, xr, yr = rectangle_bounds(
                rec, self.viewer.img_data.shape[-2:]
            )
            if yl == yr or xl == xr:
                self._set_status("Shape has zero area.", error=True)
                return None
            y_slice = slice(yl, yr)
            x_slice = slice(xl, xr)
            if Z > 1 and self.around_z_radio.isChecked():
                half = self.z_half_spin.value()
                z_center = self.z_center_spin.value()
                z_slice = slice(max(0, z_center - half), min(Z, z_center + half + 1))
            else:
                z_slice = slice(0, Z)

        psf_win = self.psf_combo.currentData()
        if psf_win is None:
            self._set_status("No PSF window selected.", error=True)
            return None
        c_psf = self.psf_channel_spin.value()
        _Tp, Zp, _Cp, _Yp, _Xp = psf_win.img_data.shape
        psf_data = np.asarray(
            psf_win.img_data[0, :, c_psf, :, :]
        ).astype(np.float32)
        if Zp == 1:
            psf_data = psf_data[0]
        if not psf_win.meta.get("psf_dc_corner", False):
            psf_data = np.fft.fftshift(psf_data)

        y_obs = np.ascontiguousarray(
            self.viewer.img_data[t, z_slice, c, y_slice, x_slice]
        ).astype(np.float32)
        dnl.log.info(
            "dialog: t=%d c=%d z_slice=%s y_slice=%s x_slice=%s "
            "y_obs=%s psf_window='%s' psf_channel=%d psf_window_shape=%s "
            "psf_dc_corner=%s psf_spacing=%s viewer_scale=%s",
            t, c, z_slice, y_slice, x_slice,
            tuple(y_obs.shape),
            psf_win.windowTitle(), c_psf, tuple(psf_data.shape),
            psf_win.meta.get("psf_dc_corner", False),
            psf_win.meta.get("spacing", psf_win.meta.get("scale")),
            self.viewer.meta.get("scale"),
        )
        if y_obs.shape[0] == 1:
            y_obs = y_obs[0]

        if psf_data.ndim != y_obs.ndim:
            self._set_status(
                f"PSF is {psf_data.ndim}D but observation is {y_obs.ndim}D.",
                error=True,
            )
            return None

        psf_spacing = psf_win.meta.get(
            "spacing", psf_win.meta.get("scale", (1.0, 1.0, 1.0))
        )
        if psf_data.ndim == 2:
            psf_pixel_size = tuple(float(s) for s in psf_spacing[-2:])
        else:
            psf_pixel_size = tuple(float(s) for s in psf_spacing[-3:])

        state = self._read_state()

        try:
            prepared = dnl.prepare_inputs(
                state=state,
                y_obs=y_obs,
                psf_array=psf_data,
                psf_pixel_size_um=psf_pixel_size,
            )
        except Exception as exc:
            self._set_status(f"{type(exc).__name__}: {exc}", error=True)
            return None

        return prepared, state, t, c

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
        output_shape = dnl.output_5d_shape(
            state, prepared.y.shape, prepared.psf.shape,
            n_channels=output_channels, n_frames=len(frame_ts),
        )
        dnl.log.info(
            "output buffer shape=%s channel=%d (tiled=%s crop_to_visible=%s "
            "frames=%s)",
            output_shape, output_channel, state.tiled, state.crop_to_visible,
            frame_ts,
        )

        # Output spacing = fine-grid spacing (data spacing ÷ zoom).
        vs = prepared.voxel_spacing
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
            "filename": "Deconvolved",
            "scale": out_scale,
            "is_rgb": False,
            "channels": channels,
            "source_channel": c,
            "source_frame": frame_ts[0] if len(frame_ts) == 1 else list(frame_ts),
            "algorithm": "nlcg",
        }

        def prepare_for_t(frame_t):
            result = self._prepare(frame_t)
            if result is None:
                return None
            frame_prepared, frame_state, _t, _c = result
            regularizer = dnl.build_regularizer(
                frame_state, frame_prepared.y.ndim, frame_prepared.voxel_spacing
            )
            return frame_prepared, frame_state, regularizer

        def make_worker(frame_data, output_frame):
            frame_prepared, frame_state, regularizer = frame_data
            return NLCGDeconvolutionWorker(
                prepared=frame_prepared,
                state=frame_state,
                regularizer=regularizer,
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

    def _describe_run(self, state: dnl.NLCGDialogState) -> str:
        if state.tiled:
            return f"Running tiled NLCG for up to {state.num_iter} iterations per tile."
        return f"Running NLCG for up to {state.num_iter} iterations."

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
