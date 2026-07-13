"""Deconvolution dialog — ER-Decon engine (deconlib.deconvolution.erdecon_*).

Edge-preserving Hessian-log deconvolution by Gauss-Newton-CG (Arigovindan,
Fung, Elnatan et al. 2013, simplified — see
`deconlib.deconvolution.erdecon_mlx` module docstring). Two-tab layout,
mirroring the NLCG/Richardson-Lucy/MaxEnt dialogs:
    * Source, PSF && Model — ROI / frame / channel / PSF window picker,
      forward-model zoom (shared `SourcePSFPanelMixin`).
    * Solver — data term (Gaussian/Poisson), the (always-on) edge-preserving
      log(eps + curvature) regularizer's lambda/eps/floor and curvature
      operator (spacing-weighted Hessian or a-trous wavelet), Gauss-Newton-CG
      solver knobs, output region, and optional tiling for large fields.

The dialog reads its widgets into a :class:`~.decon_erdecon.ERDeconDialogState`,
builds a :class:`~.decon_erdecon.PreparedInputs`, and hands the result to
:class:`~.erdecon_worker.ERDeconWorker`.
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
from .deconvolution_dialog import ScientificDoubleSpinBox
from .output_selector import ImageOutputSelector
from .processing_helper import BufferProcessingRunner
from . import decon_erdecon as der
from .erdecon_worker import ERDeconWorker


class ERDeconDialog(SourcePSFPanelMixin, QDialog):
    """Single-channel 2D/3D deconvolution dialog (ER-Decon)."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("ER-Decon (Edge-Preserving)")
        self.setWindowFlags(Qt.Tool)
        self.resize(600, 680)
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
            default_title="Deconvolved (ER-Decon)",
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

        # Data term — Gaussian LS (default, cheaper) or Poisson I-divergence.
        data_grp = QGroupBox("Data term")
        data_layout = QHBoxLayout(data_grp)
        data_layout.setContentsMargins(8, 6, 8, 6)
        data_layout.setSpacing(8)
        self.data_gaussian_radio = QRadioButton("Gaussian (least-squares)")
        self.data_gaussian_radio.setChecked(True)
        self.data_gaussian_radio.setToolTip(
            "||K g - f||^2 — read-noise-limited or well-exposed data; cheaper."
        )
        self.data_poisson_radio = QRadioButton("Poisson (shot-noise)")
        self.data_poisson_radio.setToolTip(
            "Shot-noise I-divergence, the statistically correct term for "
            "photon-limited data. Model the pedestal via Background rather "
            "than pre-subtracting it."
        )
        data_layout.addWidget(self.data_gaussian_radio)
        data_layout.addWidget(self.data_poisson_radio)
        data_layout.addStretch()
        self._data_term_group = QButtonGroup(self)
        self._data_term_group.addButton(self.data_gaussian_radio)
        self._data_term_group.addButton(self.data_poisson_radio)
        vbox.addWidget(data_grp)

        # Edge-preserving Hessian-log regularizer — always on (intrinsic to
        # the algorithm, unlike NLCG's optional gradient/Hessian term).
        reg_grp = QGroupBox("Edge-preserving regularizer")
        reg_grp.setToolTip(
            "log(eps + |Hg|^2) curvature penalty — smooths noise, preserves "
            "edges. Not optional; this is what distinguishes ER-Decon from "
            "plain NLCG/RL."
        )
        reg_grid = QGridLayout(reg_grp)
        reg_grid.setContentsMargins(8, 6, 8, 6)
        reg_grid.setHorizontalSpacing(10)
        reg_grid.setVerticalSpacing(4)
        reg_grid.setColumnStretch(1, 1)
        reg_grid.setColumnStretch(3, 1)

        self.reg_weight_spin = ScientificDoubleSpinBox()
        self.reg_weight_spin.setRange(0.0, 1e6)
        self.reg_weight_spin.setValue(0.05)
        self.reg_weight_spin.setFixedWidth(96)
        self.reg_weight_spin.setToolTip(
            "Smoothness weight lambda. Scene/SNR/scale dependent — start at "
            "the default and increase to control noise amplification."
        )
        self.eps_reg_spin = ScientificDoubleSpinBox()
        self.eps_reg_spin.setRange(1e-12, 1e6)
        self.eps_reg_spin.setValue(1e-2)
        self.eps_reg_spin.setFixedWidth(96)
        self.eps_reg_spin.setToolTip(
            "Curvature threshold eps, in units of |Hg|^2 (an absolute "
            "curvature scale on the normalized [0, 1] data, not a fraction "
            "of lambda). Curvature below it is smoothed as noise, above it "
            "preserved as an edge — broad, flat optimum; tune to the "
            "reconstruction's curvature, not to lambda."
        )
        self.floor_frac_spin = ScientificDoubleSpinBox()
        self.floor_frac_spin.setRange(0.0, 1.0)
        self.floor_frac_spin.setValue(0.0)
        self.floor_frac_spin.setFixedWidth(96)
        self.floor_frac_spin.setToolTip(
            "Quadratic-in-curvature IRLS floor (0 = off). Without it, once a "
            "voxel's curvature crosses Eps the penalty's weight keeps falling "
            "to 0 as curvature grows further, so nothing pulls flux back -- "
            "the optimizer can over-sharpen a real-but-modest bump into an "
            "isolated near-delta 'hot pixel' spike. Start small (0.01-0.05) "
            "and raise (up to ~0.5) if spikes persist; it barely affects "
            "genuine multi-pixel edges but costs some I-divergence fit."
        )
        reg_grid.addWidget(QLabel("Lambda:"), 0, 0, Qt.AlignRight)
        reg_grid.addWidget(self.reg_weight_spin, 0, 1)
        reg_grid.addWidget(QLabel("Eps:"), 0, 2, Qt.AlignRight)
        reg_grid.addWidget(self.eps_reg_spin, 0, 3)
        reg_grid.addWidget(QLabel("Floor frac:"), 1, 0, Qt.AlignRight)
        reg_grid.addWidget(self.floor_frac_spin, 1, 1)

        # Curvature operator: default Hessian2D/3D, or a single-scale a-trous
        # wavelet operator (dominates the Hessian on smoothness/accuracy at
        # levels=1, but doesn't correct axial/lateral anisotropy the way the
        # Hessian path does -- see decon_erdecon.build_regularizer).
        self.reg_hessian_radio = QRadioButton("Hessian (spacing-corrected)")
        self.reg_hessian_radio.setChecked(True)
        self.reg_wavelet_radio = QRadioButton("A-trous wavelet")
        self.reg_wavelet_radio.setToolTip(
            "Curvature measured as 'image minus one smoothed copy of itself' "
            "at each scale, noise-calibrated per scale, instead of a fixed "
            "second-difference stencil -- tends to dominate the Hessian on "
            "smoothness and accuracy. Does not correct for axial/lateral "
            "physical anisotropy (unlike the Hessian path); capped at 2 "
            "levels, the range validated against ringing."
        )
        self.reg_combined_radio = QRadioButton("Combined (Hessian + wavelet)")
        self.reg_combined_radio.setToolTip(
            "Both operators stacked and thresholded independently -- a "
            "spurious curvature spike has to fool the Hessian and the "
            "wavelet channels simultaneously to escape regularization. "
            "Fixes a real-dataset failure mode where the wavelet operator "
            "alone produced an isolated near-delta 'hot pixel' spike. Each "
            "operator's channels are noise-calibrated independently "
            "(slightly more setup cost than either operator alone)."
        )
        self._reg_type_group = QButtonGroup(self)
        self._reg_type_group.addButton(self.reg_hessian_radio)
        self._reg_type_group.addButton(self.reg_wavelet_radio)
        self._reg_type_group.addButton(self.reg_combined_radio)
        reg_type_row = QHBoxLayout()
        reg_type_row.setSpacing(8)
        reg_type_row.addWidget(self.reg_hessian_radio)
        reg_type_row.addWidget(self.reg_wavelet_radio)
        reg_type_row.addWidget(self.reg_combined_radio)
        reg_type_row.addStretch()
        reg_grid.addLayout(reg_type_row, 2, 0, 1, 4)

        self.wavelet_levels_spin = QSpinBox()
        self.wavelet_levels_spin.setRange(1, 2)
        self.wavelet_levels_spin.setValue(1)
        self.wavelet_levels_spin.setFixedWidth(64)
        self.wavelet_levels_spin.setToolTip(
            "Number of wavelet scales (levels=1 is the validated/recommended "
            "default; levels=2 can start ringing on strong edges)."
        )
        self._wavelet_levels_label = QLabel("Wavelet levels:")
        reg_grid.addWidget(self._wavelet_levels_label, 3, 0, Qt.AlignRight)
        reg_grid.addWidget(self.wavelet_levels_spin, 3, 1)
        self._reg_type_group.buttonToggled.connect(self._on_regularizer_type_toggled)
        self._on_regularizer_type_toggled()
        vbox.addWidget(reg_grp)

        # Gauss-Newton-CG solver knobs — primary fields, with an "Advanced"
        # collapsible group for knobs that rarely need tuning.
        self._erdecon_box = self._build_erdecon_knobs()
        vbox.addWidget(self._erdecon_box)
        self._erdecon_advanced_box = self._build_erdecon_advanced_knobs()
        vbox.addWidget(self._erdecon_advanced_box)

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

    def _build_erdecon_knobs(self) -> QGroupBox:
        grp = QGroupBox("Gauss-Newton-CG solver")
        grid = QGridLayout(grp)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        def _ispin(lo, hi, val, w=64):
            s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setFixedWidth(w); return s

        self.num_iter_spin = _ispin(1, 5000, 50)
        self.num_iter_spin.setToolTip("Maximum outer Newton iterations.")
        self.eval_interval_spin = _ispin(1, 500, 5)
        self.eval_interval_spin.setToolTip(
            "Interval (iterations) for live-preview writes and objective logging."
        )
        self.verbose_check = QCheckBox("Verbose")
        self.verbose_check.setToolTip(
            "Print per-iteration diagnostics (I-divergence, phi, step length, "
            "Newton decrement) to stdout."
        )

        grid.addWidget(QLabel("Iterations:"), 0, 0, Qt.AlignRight)
        grid.addWidget(self.num_iter_spin, 0, 1)
        grid.addWidget(QLabel("Eval interval:"), 0, 2, Qt.AlignRight)
        grid.addWidget(self.eval_interval_spin, 0, 3)
        grid.addWidget(self.verbose_check, 1, 0, 1, 4)
        return grp

    def _build_erdecon_advanced_knobs(self) -> QGroupBox:
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

        self.newton_tol_spin = ScientificDoubleSpinBox()
        self.newton_tol_spin.setRange(0.0, 1.0)
        self.newton_tol_spin.setValue(1e-3)
        self.newton_tol_spin.setFixedWidth(80)
        self.newton_tol_spin.setToolTip(
            "Primary convergence test. Stops once the Newton decrement "
            "(predicted decrease in phi) drops below this fraction of its "
            "first-iteration value. 0 disables it."
        )
        self.tol_spin = ScientificDoubleSpinBox()
        self.tol_spin.setRange(0.0, 1.0)
        self.tol_spin.setValue(0.0)
        self.tol_spin.setFixedWidth(80)
        self.tol_spin.setToolTip(
            "Secondary convergence test, off by default (0). When positive, "
            "also stops once the relative change in the (regularizer-free) "
            "data misfit falls below this."
        )
        self.min_iter_spin = _ispin(0, 500, 5)
        self.min_iter_spin.setToolTip(
            "Minimum outer iterations before either convergence test fires."
        )
        self.cg_max_steps_spin = _ispin(1, 500, 25)
        self.cg_max_steps_spin.setToolTip("Max inner CG steps per Newton solve.")
        self.cg_tol_spin = _dspin(0.0, 1.0, 0.1, dec=3, step=0.01)
        self.cg_tol_spin.setToolTip(
            "Inner CG relative-residual tolerance. A loose default gives an "
            "inexact/truncated Newton step, cheaper and usually as good."
        )
        self.ls_max_backtracks_spin = _ispin(1, 200, 30)
        self.ls_max_backtracks_spin.setToolTip(
            "Max Armijo halvings before declaring the line search stuck."
        )
        self.ls_c1_spin = ScientificDoubleSpinBox()
        self.ls_c1_spin.setRange(1e-12, 1.0)
        self.ls_c1_spin.setValue(1e-4)
        self.ls_c1_spin.setFixedWidth(80)
        self.ls_c1_spin.setToolTip("Armijo sufficient-decrease constant.")
        self.normalize_check = QCheckBox("Normalize data amplitude")
        self.normalize_check.setChecked(True)
        self.normalize_check.setToolTip(
            "Divide the data by its max before solving so Lambda/Eps refer "
            "to a fixed [0, 1] amplitude; the result is scaled back to "
            "original units. Uncheck only if Lambda/Eps are already tuned "
            "to this data's raw amplitude."
        )

        def _row(r, la, wa, lb, wb):
            grid.addWidget(QLabel(la), r, 0, Qt.AlignRight)
            grid.addWidget(wa,         r, 1)
            grid.addWidget(QLabel(lb), r, 2, Qt.AlignRight)
            grid.addWidget(wb,         r, 3)

        _row(0, "Newton tol:", self.newton_tol_spin, "Convergence tol:", self.tol_spin)
        _row(1, "Min iter:", self.min_iter_spin, "CG max steps:", self.cg_max_steps_spin)
        _row(2, "CG tol:", self.cg_tol_spin, "Line search max:", self.ls_max_backtracks_spin)
        grid.addWidget(QLabel("Armijo c1:"), 3, 0, Qt.AlignRight)
        grid.addWidget(self.ls_c1_spin, 3, 1)
        grid.addWidget(self.normalize_check, 3, 2, 1, 2)
        grp.toggled.connect(lambda checked: self._toggle_group_children(grp, checked))
        self._toggle_group_children(grp, False)
        return grp

    # ------------------------------------------------------------------ #
    # Dynamic visibility

    def _connect_recipe_preview_signals(self) -> None:
        widgets = [
            self.data_gaussian_radio, self.data_poisson_radio,
            self.reg_hessian_radio, self.reg_wavelet_radio, self.reg_combined_radio,
        ]
        for widget in widgets:
            widget.toggled.connect(lambda *_: self._on_source_psf_changed())
        self.wavelet_levels_spin.valueChanged.connect(
            lambda *_: self._on_source_psf_changed()
        )
        for spin in (
            self.reg_weight_spin, self.eps_reg_spin, self.floor_frac_spin,
            self.num_iter_spin, self.eval_interval_spin,
            self.newton_tol_spin, self.tol_spin, self.min_iter_spin,
            self.cg_max_steps_spin, self.cg_tol_spin,
            self.ls_max_backtracks_spin, self.ls_c1_spin,
        ):
            spin.valueChanged.connect(lambda *_: self._on_source_psf_changed())
        self.normalize_check.toggled.connect(lambda *_: self._on_source_psf_changed())

    # ------------------------------------------------------------------ #
    # Mixin hooks

    def _on_source_psf_changed(self) -> None:
        if hasattr(self, "compute_target_label"):
            self._refresh_compute_target_hint()
        if hasattr(self, "recipe_preview"):
            self._refresh_recipe_preview()

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

    def _on_regularizer_type_toggled(self, *_args) -> None:
        show_levels = self.reg_wavelet_radio.isChecked() or self.reg_combined_radio.isChecked()
        self._wavelet_levels_label.setVisible(show_levels)
        self.wavelet_levels_spin.setVisible(show_levels)
        self._on_source_psf_changed()

    def _regularizer_type(self) -> str:
        if self.reg_wavelet_radio.isChecked():
            return "wavelet"
        if self.reg_combined_radio.isChecked():
            return "combined"
        return "hessian"

    def _on_diagnostics_toggled(self, checked: bool) -> None:
        self._toggle_group_children(self._diagnostics_group, checked)

    def _current_state_for_shape(self) -> der.ERDeconDialogState:
        # A lightweight state good enough for shape math (zoom + region
        # knobs only); the full state is built in _read_state().
        return der.ERDeconDialogState(
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
        zoom = der.per_axis_zoom(state, ndim)
        visible = der.visible_shape(data_shape, zoom)
        kernel = self._current_psf_kernel_shape()
        padded = (
            der.padded_shape(visible, kernel)
            if kernel is not None and len(kernel) == ndim
            else visible
        )
        output_spatial_shape = der.output_shape(
            state, data_shape, kernel or tuple(1 for _ in range(ndim))
        )

        data_term = "poisson" if self.data_poisson_radio.isChecked() else "gaussian"
        floor_frac = self.floor_frac_spin.value()
        floor_text = f" floor={floor_frac:g}" if floor_frac > 0 else ""
        if self.reg_wavelet_radio.isChecked():
            reg_text = f"wavelet(levels={self.wavelet_levels_spin.value()})"
        elif self.reg_combined_radio.isChecked():
            reg_text = f"combined(hessian+wavelet levels={self.wavelet_levels_spin.value()})"
        else:
            reg_text = "hessian"
        lines = [
            f"ER-Decon ({data_term}), {self.num_iter_spin.value()} iter max, "
            f"reg={reg_text} "
            f"lambda={self.reg_weight_spin.value():g} eps={self.eps_reg_spin.value():g}"
            f"{floor_text}",
            f"data {self._format_shape(data_shape)} -> "
            f"visible {self._format_shape(visible)} -> "
            f"output {self._format_shape(output_spatial_shape)}",
        ]
        sr_text = " × ".join(f"{float(v):g}" for v in zoom)
        lines.append(f"zoom {sr_text}; object/padded shape {self._format_shape(padded)}")
        if self.tiled_check.isChecked():
            n_tiles, tile_shape = der.estimate_tile_plan(state, data_shape)
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

    def _read_state(self) -> der.ERDeconDialogState:
        data_term = "poisson" if self.data_poisson_radio.isChecked() else "gaussian"
        return der.ERDeconDialogState(
            zoom_xy=self.sr_xy_spin.value(),
            zoom_z=self.sr_z_spin.value(),
            reg_weight=self.reg_weight_spin.value(),
            eps_reg=self.eps_reg_spin.value(),
            floor_frac=self.floor_frac_spin.value(),
            regularizer_type=self._regularizer_type(),
            wavelet_levels=self.wavelet_levels_spin.value(),
            data_term=data_term,
            num_iter=self.num_iter_spin.value(),
            background=self.background_spin.value(),
            normalize=self.normalize_check.isChecked(),
            eval_interval=self.eval_interval_spin.value(),
            newton_tol=self.newton_tol_spin.value(),
            tol=self.tol_spin.value(),
            min_iter=self.min_iter_spin.value(),
            cg_max_steps=self.cg_max_steps_spin.value(),
            cg_tol=self.cg_tol_spin.value(),
            ls_max_backtracks=self.ls_max_backtracks_spin.value(),
            ls_c1=self.ls_c1_spin.value(),
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
            prepared = der.prepare_inputs(
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
        output_shape = der.output_5d_shape(
            state, prepared.y.shape, prepared.psf.shape,
            n_channels=output_channels, n_frames=len(frame_ts),
        )
        der.log.info(
            "output buffer shape=%s channel=%d (tiled=%s crop_to_visible=%s "
            "frames=%s)",
            output_shape, output_channel, state.tiled, state.crop_to_visible,
            frame_ts,
        )

        # Output spacing = fine-grid spacing (data spacing ÷ zoom), corrected
        # for the rounding `visible_shape` applies when data_shape * zoom
        # isn't an exact integer.
        vs = der.effective_voxel_spacing(
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
            "filename": "Deconvolved (ER-Decon)",
            "scale": out_scale,
            "is_rgb": False,
            "channels": channels,
            "source_channel": c,
            "source_frame": frame_ts[0] if len(frame_ts) == 1 else list(frame_ts),
            "algorithm": "erdecon",
        }

        def prepare_for_t(frame_t):
            result = self._prepare(frame_t)
            if result is None:
                return None
            frame_prepared, frame_state, _t, _c = result
            regularizer, combine_channels = der.build_regularizer(
                frame_state, frame_prepared
            )
            return frame_prepared, frame_state, regularizer, combine_channels

        def make_worker(frame_data, output_frame):
            frame_prepared, frame_state, regularizer, combine_channels = frame_data
            return ERDeconWorker(
                prepared=frame_prepared,
                state=frame_state,
                hessian=regularizer,
                combine_channels=combine_channels,
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

    def _describe_run(self, state: der.ERDeconDialogState) -> str:
        if state.tiled:
            return f"Running tiled ER-Decon for up to {state.num_iter} iterations per tile."
        return f"Running ER-Decon for up to {state.num_iter} iterations."

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
