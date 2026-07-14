"""Deconvolution dialog — jetnewton engine (deconlib.deconvolution.jetnewton_mlx).

Non-dimensional log-penalty deconvolution solved by an *exact*-Hessian
active-set projected Newton method (Arigovindan/Elnatan-style log-curvature
regularizer, but no Gauss-Newton surrogate) -- the successor to, and full
replacement for, ER-Decon. See `deconlib.deconvolution.jetnewton_mlx`'s
module docstring and `deconlib/scripts/widefield_jetnewton_realdata_demo.py`
for the algorithm and its "UI design notes".

The parameter surface is deliberately narrow (see that docstring):
    * `beta` (regularization strength) is the *only* free knob -- a
      log-scale slider, ~1e-4 to 1e-1.
    * `s0` (intensity scale), `ell`/`kappa` (PSF length scale), and `eta`
      (penalty saturation threshold) are always auto-calibrated from
      required acquisition inputs (noise sigma, PSF, pixel spacing) --
      never typed in. The "Calibration (auto)" group shows the derived
      values read-only, for transparency, not as editable fields.
    * Optimizer knobs (CG steps, Newton/tol thresholds, active-set
      parameters) are advanced/hidden with sane fixed defaults.
    * No wavelet/combined regularizer, no missing-cone/OTF term -- jetnewton
      is curvature-only by design (see module docstring for why both were
      tried and dropped).

Two-tab layout, mirroring the ER-Decon dialog it replaces:
    * Source, PSF && Model — ROI / frame / channel / PSF window picker,
      forward-model zoom (shared `SourcePSFPanelMixin`).
    * Solver — data term, required noise-sigma input, the beta slider, the
      read-only auto-calibration panel, advanced optimizer knobs, output
      region, and optional tiling for large fields.

The dialog reads its widgets into a
:class:`~.decon_jetnewton.JetNewtonDialogState`, builds a
:class:`~.decon_jetnewton.PreparedInputs` plus a
:class:`~.decon_jetnewton.Calibration`, and hands the result to
:class:`~.jetnewton_worker.JetNewtonWorker`.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
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
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .decon_source_panel import SourcePSFPanelMixin
from .deconvolution_dialog import ScientificDoubleSpinBox
from .output_selector import ImageOutputSelector
from .processing_helper import BufferProcessingRunner
from . import decon_jetnewton as jn
from .jetnewton_worker import JetNewtonWorker


class _LogSlider(QWidget):
    """Log-scale slider + exact-value spinbox, kept in sync both ways.

    `beta` is the one genuine UI knob for jetnewton (see module docstring)
    -- a plain linear slider would waste almost all its travel above 1e-2
    and give no useful resolution below it, so the slider position maps to
    `log10(value)` instead.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        minimum: float = 1e-4,
        maximum: float = 1e-1,
        value: float = 1e-2,
        steps: int = 300,
        parent=None,
    ):
        super().__init__(parent)
        self._log_min = math.log10(minimum)
        self._log_max = math.log10(maximum)
        self._steps = steps
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, steps)
        layout.addWidget(self._slider, 1)

        self._spin = ScientificDoubleSpinBox()
        self._spin.setRange(minimum, maximum)
        self._spin.setFixedWidth(96)
        layout.addWidget(self._spin)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin.valueChanged.connect(self._on_spin_changed)
        self.setValue(value)

    def _pos_from_value(self, value: float) -> int:
        value = min(max(value, 10**self._log_min), 10**self._log_max)
        frac = (math.log10(value) - self._log_min) / (self._log_max - self._log_min)
        return round(frac * self._steps)

    def _value_from_pos(self, pos: int) -> float:
        frac = pos / self._steps
        return 10 ** (self._log_min + frac * (self._log_max - self._log_min))

    def _on_slider_changed(self, pos: int) -> None:
        if self._updating:
            return
        self._updating = True
        value = self._value_from_pos(pos)
        self._spin.setValue(value)
        self._updating = False
        self.valueChanged.emit(value)

    def _on_spin_changed(self, value: float) -> None:
        if self._updating:
            return
        self._updating = True
        self._slider.setValue(self._pos_from_value(value))
        self._updating = False
        self.valueChanged.emit(value)

    def value(self) -> float:
        return self._spin.value()

    def setValue(self, value: float) -> None:
        self._spin.setValue(value)
        self._slider.setValue(self._pos_from_value(value))


class JetNewtonDialog(SourcePSFPanelMixin, QDialog):
    """Single-channel 2D/3D deconvolution dialog (jetnewton)."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("jetnewton (Active-Set Projected Newton)")
        self.setWindowFlags(Qt.Tool)
        self.resize(600, 720)
        self.setMinimumSize(540, 480)

        self._runner = None
        self._running_status_base: Optional[str] = None
        self._run_started_at: Optional[float] = None
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_running_status)

        self._calib_timer = QTimer(self)
        self._calib_timer.setSingleShot(True)
        self._calib_timer.setInterval(600)
        self._calib_timer.timeout.connect(self._refresh_calibration)

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
            default_title="Deconvolved (jetnewton)",
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
        self._refresh_calibration()

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

        # Data term.
        data_grp = QGroupBox("Data term")
        data_layout = QHBoxLayout(data_grp)
        data_layout.setContentsMargins(8, 6, 8, 6)
        data_layout.setSpacing(8)
        self.data_poisson_radio = QRadioButton("Poisson (shot-noise)")
        self.data_poisson_radio.setChecked(True)
        self.data_poisson_radio.setToolTip(
            "Shot-noise I-divergence, the statistically correct term for "
            "photon-limited data. Model the pedestal via Background rather "
            "than pre-subtracting it."
        )
        self.data_gaussian_radio = QRadioButton("Gaussian (least-squares)")
        self.data_gaussian_radio.setToolTip(
            "Variance-normalized least-squares (uses Noise sigma below as "
            "the Gaussian sigma) — read-noise-limited data."
        )
        data_layout.addWidget(self.data_poisson_radio)
        data_layout.addWidget(self.data_gaussian_radio)
        data_layout.addStretch()
        self._data_term_group = QButtonGroup(self)
        self._data_term_group.addButton(self.data_poisson_radio)
        self._data_term_group.addButton(self.data_gaussian_radio)
        vbox.addWidget(data_grp)

        # Required acquisition input — noise sigma. Background lives on the
        # Source/PSF tab (shared `background_spin`).
        acq_grp = QGroupBox("Acquisition input")
        acq_form = QFormLayout(acq_grp)
        acq_form.setLabelAlignment(Qt.AlignRight)
        acq_form.setVerticalSpacing(4)
        acq_form.setContentsMargins(8, 6, 8, 6)
        self.noise_sigma_spin = ScientificDoubleSpinBox()
        self.noise_sigma_spin.setRange(1e-6, 1e6)
        self.noise_sigma_spin.setValue(15.0)
        self.noise_sigma_spin.setFixedWidth(96)
        self.noise_sigma_spin.setToolTip(
            "Measured data-space noise sigma (e.g. from camera calibration) "
            "— NOT derived from Background, which is just the camera's "
            "baseline clamp, a DC offset unrelated to noise. Required, "
            "explicit input; sets the intensity scale s0 below."
        )
        acq_form.addRow("Noise sigma (data):", self.noise_sigma_spin)
        vbox.addWidget(acq_grp)

        # The one genuine knob.
        reg_grp = QGroupBox("Regularization")
        reg_v = QVBoxLayout(reg_grp)
        reg_v.setContentsMargins(8, 6, 8, 6)
        reg_v.setSpacing(4)
        reg_hint = QLabel(
            "Beta is the only free parameter — everything else below is "
            "either a required input or auto-calibrated."
        )
        reg_hint.setStyleSheet("color: #888; font-size: 10px;")
        reg_hint.setWordWrap(True)
        reg_v.addWidget(reg_hint)
        beta_row = QHBoxLayout()
        beta_row.setSpacing(8)
        beta_row.addWidget(QLabel("Beta:"))
        self.beta_slider = _LogSlider(
            minimum=1e-6, maximum=1e3, value=1e-2, steps=900
        )
        self.beta_slider.setToolTip(
            "Overall regularization weight (log scale). Lower idiv (better "
            "fit) trades off against axial/structural collapse as beta "
            "shrinks — sweep and watch both together; real data typically "
            "plateaus in idiv well above the idealized Poisson floor, so "
            "don't chase it past that plateau."
        )
        beta_row.addWidget(self.beta_slider, 1)
        reg_v.addLayout(beta_row)
        vbox.addWidget(reg_grp)

        # Auto-calibrated, read-only.
        calib_grp = QGroupBox("Calibration (auto)")
        calib_grp.setToolTip(
            "Derived from the inputs above — PSF optics, pixel spacing, "
            "Noise sigma — never guessed or hand-tuned. Recomputes shortly "
            "after any relevant change; use Recalculate to force it now."
        )
        calib_form = QFormLayout(calib_grp)
        calib_form.setLabelAlignment(Qt.AlignRight)
        calib_form.setVerticalSpacing(4)
        calib_form.setContentsMargins(8, 6, 8, 6)
        self.calib_s0_label = QLabel("—")
        self.calib_eta_label = QLabel("—")
        self.calib_ell_label = QLabel("—")
        self.calib_kappa_label = QLabel("—")
        for lbl in (
            self.calib_s0_label, self.calib_eta_label,
            self.calib_ell_label, self.calib_kappa_label,
        ):
            lbl.setStyleSheet("font-size: 10px;")
            lbl.setWordWrap(True)
        calib_form.addRow("s0 (intensity scale):", self.calib_s0_label)
        calib_form.addRow("eta (penalty threshold):", self.calib_eta_label)
        calib_form.addRow("ell (PSF length scale):", self.calib_ell_label)
        calib_form.addRow("kappa (ell / spacing):", self.calib_kappa_label)
        recalc_btn = QPushButton("Recalculate now")
        recalc_btn.clicked.connect(self._refresh_calibration)
        calib_form.addRow("", recalc_btn)
        vbox.addWidget(calib_grp)

        # Solver knobs — primary.
        solver_grp = QGroupBox("Solver")
        solver_grid = QGridLayout(solver_grp)
        solver_grid.setContentsMargins(8, 6, 8, 6)
        solver_grid.setHorizontalSpacing(10)
        solver_grid.setVerticalSpacing(4)
        solver_grid.setColumnStretch(1, 1)
        solver_grid.setColumnStretch(3, 1)

        def _ispin(lo, hi, val, w=64):
            s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setFixedWidth(w); return s

        self.num_iter_spin = _ispin(1, 5000, 60)
        self.num_iter_spin.setToolTip(
            "Maximum outer Newton iterations — headroom only; Newton tol "
            "(Advanced) is what actually stops the solve."
        )
        self.eval_interval_spin = _ispin(1, 500, 5)
        self.eval_interval_spin.setToolTip(
            "Interval (iterations) for live-preview writes and objective logging."
        )
        self.verbose_check = QCheckBox("Verbose")
        self.verbose_check.setToolTip(
            "Print per-iteration diagnostics (I-divergence, Newton decrement, "
            "active-set size, CG steps) to stdout."
        )
        solver_grid.addWidget(QLabel("Max iterations:"), 0, 0, Qt.AlignRight)
        solver_grid.addWidget(self.num_iter_spin, 0, 1)
        solver_grid.addWidget(QLabel("Eval interval:"), 0, 2, Qt.AlignRight)
        solver_grid.addWidget(self.eval_interval_spin, 0, 3)
        solver_grid.addWidget(self.verbose_check, 1, 0, 1, 4)
        vbox.addWidget(solver_grp)

        self._jetnewton_advanced_box = self._build_advanced_knobs()
        vbox.addWidget(self._jetnewton_advanced_box)

        # Output region + tiling.
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
            "model and one calibration (s0/eta), stitching owned cores back "
            "together. Tiled output is always cropped to the visible field."
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

    def _build_advanced_knobs(self) -> QGroupBox:
        grp = QGroupBox("Advanced optimizer knobs")
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

        self.newton_tol_spin = ScientificDoubleSpinBox()
        self.newton_tol_spin.setRange(0.0, 1.0)
        self.newton_tol_spin.setValue(1e-4)
        self.newton_tol_spin.setFixedWidth(80)
        self.newton_tol_spin.setToolTip(
            "Primary convergence test. Stops once the Newton decrement "
            "(predicted decrease in F this step) drops below this fraction "
            "of its first-iteration value. 0 disables it."
        )
        self.tol_spin = ScientificDoubleSpinBox()
        self.tol_spin.setRange(0.0, 1.0)
        self.tol_spin.setValue(0.0)
        self.tol_spin.setFixedWidth(80)
        self.tol_spin.setToolTip(
            "Secondary convergence test, off by default (0). When positive, "
            "also stops once the projected-gradient inf-norm (relative to "
            "its first-iteration value) drops below this."
        )
        self.min_iter_spin = _ispin(0, 500, 3)
        self.min_iter_spin.setToolTip(
            "Minimum outer iterations before either convergence test fires."
        )
        self.cg_max_steps_spin = _ispin(1, 1000, 150)
        self.cg_max_steps_spin.setToolTip(
            "Max inner (masked, unpreconditioned) CG steps per Newton solve."
        )
        self.ls_max_backtracks_spin = _ispin(1, 200, 30)
        self.ls_max_backtracks_spin.setToolTip(
            "Max Armijo halvings before declaring the projected line search stuck."
        )
        self.ls_sigma_spin = ScientificDoubleSpinBox()
        self.ls_sigma_spin.setRange(1e-12, 1.0)
        self.ls_sigma_spin.setValue(1e-4)
        self.ls_sigma_spin.setFixedWidth(80)
        self.ls_sigma_spin.setToolTip("Armijo sufficient-decrease constant.")

        self.eps_bar_spin = ScientificDoubleSpinBox()
        self.eps_bar_spin.setRange(1e-8, 1.0)
        self.eps_bar_spin.setValue(1e-2)
        self.eps_bar_spin.setFixedWidth(80)
        self.eps_bar_spin.setToolTip(
            "Active-set identification cap (Bertsekas rule) — shrinks as the "
            "iterate converges, capped here."
        )
        self.freeze_tau_spin = ScientificDoubleSpinBox()
        self.freeze_tau_spin.setRange(1e-8, 1.0)
        self.freeze_tau_spin.setValue(1e-3)
        self.freeze_tau_spin.setFixedWidth(80)
        self.freeze_tau_spin.setToolTip(
            "Voxel value below this, with a sufficiently positive gradient "
            "(Freeze delta), gets permanently pinned at zero for the rest "
            "of the run."
        )
        self.freeze_delta_spin = ScientificDoubleSpinBox()
        self.freeze_delta_spin.setRange(1e-12, 1.0)
        self.freeze_delta_spin.setValue(1e-6)
        self.freeze_delta_spin.setFixedWidth(80)
        self.freeze_delta_spin.setToolTip(
            "Gradient threshold for permanently freezing a near-zero voxel "
            "(see Freeze tau)."
        )

        def _row(r, la, wa, lb, wb):
            grid.addWidget(QLabel(la), r, 0, Qt.AlignRight)
            grid.addWidget(wa,         r, 1)
            grid.addWidget(QLabel(lb), r, 2, Qt.AlignRight)
            grid.addWidget(wb,         r, 3)

        _row(0, "Newton tol:", self.newton_tol_spin, "Convergence tol:", self.tol_spin)
        _row(1, "Min iter:", self.min_iter_spin, "CG max steps:", self.cg_max_steps_spin)
        _row(2, "Line search max:", self.ls_max_backtracks_spin, "Armijo sigma:", self.ls_sigma_spin)
        _row(3, "Active-set eps:", self.eps_bar_spin, "Freeze tau:", self.freeze_tau_spin)
        grid.addWidget(QLabel("Freeze delta:"), 4, 0, Qt.AlignRight)
        grid.addWidget(self.freeze_delta_spin, 4, 1)
        grp.toggled.connect(lambda checked: self._toggle_group_children(grp, checked))
        self._toggle_group_children(grp, False)
        return grp

    # ------------------------------------------------------------------ #
    # Dynamic visibility

    def _connect_recipe_preview_signals(self) -> None:
        widgets = [self.data_poisson_radio, self.data_gaussian_radio]
        for widget in widgets:
            widget.toggled.connect(lambda *_: self._on_source_psf_changed())
        self.beta_slider.valueChanged.connect(lambda *_: self._on_source_psf_changed())
        self.noise_sigma_spin.valueChanged.connect(
            lambda *_: self._on_source_psf_changed()
        )
        for spin in (
            self.num_iter_spin, self.eval_interval_spin,
            self.newton_tol_spin, self.tol_spin, self.min_iter_spin,
            self.cg_max_steps_spin, self.ls_max_backtracks_spin, self.ls_sigma_spin,
            self.eps_bar_spin, self.freeze_tau_spin, self.freeze_delta_spin,
        ):
            spin.valueChanged.connect(lambda *_: self._on_source_psf_changed())

    # ------------------------------------------------------------------ #
    # Mixin hooks

    def _on_source_psf_changed(self) -> None:
        if hasattr(self, "compute_target_label"):
            self._refresh_compute_target_hint()
        if hasattr(self, "recipe_preview"):
            self._refresh_recipe_preview()
        if hasattr(self, "_calib_timer"):
            self._calib_timer.start()

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

    def _current_state_for_shape(self) -> jn.JetNewtonDialogState:
        # A lightweight state good enough for shape math (zoom + region
        # knobs only); the full state is built in _read_state().
        return jn.JetNewtonDialogState(
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
        zoom = jn.per_axis_zoom(state, ndim)
        visible = jn.visible_shape(data_shape, zoom)
        kernel = self._current_psf_kernel_shape()
        padded = (
            jn.padded_shape(visible, kernel)
            if kernel is not None and len(kernel) == ndim
            else visible
        )
        output_spatial_shape = jn.output_shape(
            state, data_shape, kernel or tuple(1 for _ in range(ndim))
        )

        data_term = "poisson" if self.data_poisson_radio.isChecked() else "gaussian"
        lines = [
            f"jetnewton ({data_term}), {self.num_iter_spin.value()} iter max, "
            f"beta={self.beta_slider.value():.3g}",
            f"data {self._format_shape(data_shape)} -> "
            f"visible {self._format_shape(visible)} -> "
            f"output {self._format_shape(output_spatial_shape)}",
        ]
        sr_text = " × ".join(f"{float(v):g}" for v in zoom)
        lines.append(f"zoom {sr_text}; object/padded shape {self._format_shape(padded)}")
        if self.tiled_check.isChecked():
            n_tiles, tile_shape = jn.estimate_tile_plan(state, data_shape)
            lines.append(
                f"tiled: ~{n_tiles} tile(s) of shape {self._format_shape(tile_shape)}"
            )
        self.recipe_preview.setPlainText("\n".join(lines))

        self.detector_domain_label.setText(self._format_shape(data_shape))
        self.visible_domain_label.setText(self._format_shape(visible))
        self.psf_kernel_label.setText(self._format_shape(kernel))
        self.object_domain_label.setText(self._format_shape(padded))

    # ------------------------------------------------------------------ #
    # Calibration preview (auto s0/eta/ell/kappa)

    def _refresh_calibration(self) -> None:
        if not hasattr(self, "calib_s0_label"):
            return
        if self.psf_combo.currentData() is None or self._current_data_shape() is None:
            self._set_calibration_labels(None)
            return
        result = self._prepare(self.t_start_spin.value())
        if result is None:
            self._set_calibration_labels(None)
            return
        _prepared, _state, _hessian, calib, _t, _c = result
        self._set_calibration_labels(calib)

    def _set_calibration_labels(self, calib: Optional[jn.Calibration]) -> None:
        if calib is None:
            for lbl in (
                self.calib_s0_label, self.calib_eta_label,
                self.calib_ell_label, self.calib_kappa_label,
            ):
                lbl.setText("—")
            return
        self.calib_s0_label.setText(f"{calib.s0:.4g}")
        nf = calib.noise_floor
        self.calib_eta_label.setText(
            f"{calib.eta:.4g}  (noise-floor median; p1={nf['p1']:.3g}, "
            f"p99={nf['p99']:.3g})"
        )
        self.calib_ell_label.setText(
            " × ".join(f"{v:.4g}" for v in calib.ell) + " μm"
        )
        self.calib_kappa_label.setText(
            " × ".join(f"{v:.4g}" for v in calib.kappa)
        )

    # ------------------------------------------------------------------ #
    # State / inputs

    def _read_state(self) -> jn.JetNewtonDialogState:
        data_term = "poisson" if self.data_poisson_radio.isChecked() else "gaussian"
        return jn.JetNewtonDialogState(
            zoom_xy=self.sr_xy_spin.value(),
            zoom_z=self.sr_z_spin.value(),
            data_term=data_term,
            background=self.background_spin.value(),
            noise_sigma=self.noise_sigma_spin.value(),
            beta=self.beta_slider.value(),
            num_iter=self.num_iter_spin.value(),
            eval_interval=self.eval_interval_spin.value(),
            verbose=self.verbose_check.isChecked(),
            cg_max_steps=self.cg_max_steps_spin.value(),
            newton_tol=self.newton_tol_spin.value(),
            tol=self.tol_spin.value(),
            min_iter=self.min_iter_spin.value(),
            eps_bar=self.eps_bar_spin.value(),
            freeze_tau=self.freeze_tau_spin.value(),
            freeze_delta=self.freeze_delta_spin.value(),
            ls_max_backtracks=self.ls_max_backtracks_spin.value(),
            ls_sigma=self.ls_sigma_spin.value(),
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
            prepared = jn.prepare_inputs(
                state=state,
                y_obs=cropped.y_obs,
                psf_array=cropped.psf_data,
                psf_pixel_size_um=cropped.psf_pixel_size,
            )
            hessian, calib = jn.calibrate(state, prepared)
        except Exception as exc:
            self._set_status(f"{type(exc).__name__}: {exc}", error=True)
            return None

        return prepared, state, hessian, calib, cropped.t, cropped.c

    # ------------------------------------------------------------------ #
    # Run / cancel

    def _start(self):
        if self._runner.is_running():
            return
        frame_ts = list(self._frame_range())
        first = self._prepare(frame_ts[0])
        if first is None:
            return
        prepared, state, _hessian, calib, t, c = first

        stack_channels = self.stack_channels_check.isChecked()
        output_channels = self.viewer.C if stack_channels else 1
        output_channel = c if stack_channels else 0
        output_shape = jn.output_5d_shape(
            state, prepared.y.shape, prepared.psf.shape,
            n_channels=output_channels, n_frames=len(frame_ts),
        )
        jn.log.info(
            "output buffer shape=%s channel=%d (tiled=%s crop_to_visible=%s "
            "frames=%s) s0=%g eta=%g",
            output_shape, output_channel, state.tiled, state.crop_to_visible,
            frame_ts, calib.s0, calib.eta,
        )

        vs = jn.effective_voxel_spacing(
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
            "filename": "Deconvolved (jetnewton)",
            "scale": out_scale,
            "is_rgb": False,
            "channels": channels,
            "source_channel": c,
            "source_frame": frame_ts[0] if len(frame_ts) == 1 else list(frame_ts),
            "algorithm": "jetnewton",
        }

        def prepare_for_t(frame_t):
            result = self._prepare(frame_t)
            if result is None:
                return None
            frame_prepared, frame_state, frame_hessian, frame_calib, _t, _c = result
            return frame_prepared, frame_state, frame_hessian, frame_calib

        def make_worker(frame_data, output_frame):
            frame_prepared, frame_state, frame_hessian, frame_calib = frame_data
            return JetNewtonWorker(
                prepared=frame_prepared,
                state=frame_state,
                hessian=frame_hessian,
                s0=frame_calib.s0,
                eta=frame_calib.eta,
                buffer=self._runner.output_buffer,
                output_channel=output_channel,
                output_frame=output_frame,
            )

        if state.tiled:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, max(1, state.num_iter))

        self.progress_bar.setValue(0)
        self._begin_run_status(self._describe_run(state, calib))
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

    def _describe_run(self, state: jn.JetNewtonDialogState, calib: jn.Calibration) -> str:
        base = (
            f"Running tiled jetnewton for up to {state.num_iter} iterations per tile."
            if state.tiled else
            f"Running jetnewton for up to {state.num_iter} iterations."
        )
        return f"{base} (beta={state.beta:.3g}, eta={calib.eta:.3g}, s0={calib.s0:.3g})"

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
