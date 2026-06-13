"""Deconvolution dialog — recipe-driven via deconlib.

Two-tab layout:
    * Source & PSF — ROI / frame / channel / PSF window picker.
    * Solver — algorithm radio (MEM | RL), forward-model knobs,
      ICF group (MEM only), and per-algorithm solver settings.

The dialog reads its widgets into a :class:`DecondialogState`, builds a
:class:`PreparedInputs`, and hands the result to either
:class:`MemDeconvolutionWorker` or :class:`RLDeconvolutionWorker`.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyvistra.ui.manager import manager
from pyvistra.data.shapes import RECTANGLE

from .output_selector import ImageOutputSelector
from .processing_helper import BufferProcessingRunner
from .decon_recipe import (
    DecondialogState,
    build_icf_sweep,
    build_mem_config,
    build_posterior,
    build_rl_config,
    build_wavelet_config,
    log,
    output_5d_shape,
    prepare_inputs,
)
from .deconvolution_worker import (
    MemDeconvolutionWorker,
    RLDeconvolutionWorker,
)


class DeconvolutionDialog(QDialog):
    """Single-channel 2D/3D deconvolution dialog (MEM + Richardson-Lucy)."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Deconvolution")
        self.setWindowFlags(Qt.Tool)
        self.resize(560, 600)
        self.setMinimumSize(520, 460)

        self._runner = None
        self._rect_shapes: list = []
        self._last_run_state: Optional[DecondialogState] = None
        self._last_icf_sigma_um: Optional[float] = None
        self._running_status_base: Optional[str] = None
        self._run_started_at: Optional[float] = None
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_running_status)

        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 8)
        main.setSpacing(6)

        tabs = QTabWidget()
        tabs.addTab(self._build_source_psf_tab(), "Source & PSF")
        tabs.addTab(self._build_solver_tab(), "Solver")
        main.addWidget(tabs)

        self.output_selector = ImageOutputSelector(
            default_title="Deconvolved",
            formats=[".tif", ".ims"],
        )
        main.addWidget(self.output_selector)

        self._runner = BufferProcessingRunner(self.viewer, self.output_selector)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main.addWidget(self.progress_bar)

        # Use a read-only QPlainTextEdit instead of a QLabel so long error
        # messages (tracebacks, multi-line diagnostics) wrap inside the
        # widget instead of stretching the dialog horizontally.
        self.status_label = QPlainTextEdit("Ready")
        self.status_label.setReadOnly(True)
        self.status_label.setFixedHeight(56)
        self.status_label.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.status_label.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.status_label.setStyleSheet("color: #888;")
        main.addWidget(self.status_label)

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

        # apply initial visibility once everything is wired
        self._on_algorithm_changed()
        self._on_icf_mode_changed()
        self._refresh_compute_target_hint()
        self._refresh_recipe_preview()

    # ------------------------------------------------------------------ #
    # Tab builders

    def _build_source_psf_tab(self) -> QWidget:
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        T, _Z, C, _Y, _X = self.viewer.img_data.shape

        cf_grp = QGroupBox("Detector / data")
        cf_form = QFormLayout(cf_grp)
        cf_form.setLabelAlignment(Qt.AlignRight)
        cf_form.setVerticalSpacing(4)
        cf_form.setContentsMargins(8, 6, 8, 6)

        ch_t = QWidget()
        ch_t_layout = QHBoxLayout(ch_t)
        ch_t_layout.setContentsMargins(0, 0, 0, 0)
        ch_t_layout.setSpacing(6)
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, max(0, C - 1))
        self.channel_spin.setValue(
            min(max(0, getattr(self.viewer, "c_idx", 0)), max(0, C - 1))
        )
        self.channel_spin.setFixedWidth(52)
        self.channel_spin.valueChanged.connect(
            lambda _v: self._refresh_recipe_preview()
        )
        ch_t_layout.addWidget(self.channel_spin)
        ch_t_layout.addWidget(QLabel("Frame T:"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, max(0, T - 1))
        self.frame_spin.setValue(getattr(self.viewer, "t_idx", 0))
        self.frame_spin.setFixedWidth(52)
        self.frame_spin.valueChanged.connect(
            lambda _v: self._refresh_recipe_preview()
        )
        ch_t_layout.addWidget(self.frame_spin)
        ch_t_layout.addStretch()
        cf_form.addRow("Channel:", ch_t)

        self.stack_channels_check = QCheckBox("Stack into source channel index")
        self.stack_channels_check.setToolTip(
            "Create/reuse a multi-channel output buffer and write the selected "
            "source channel into the same channel index. Re-run for other "
            "channels into the same output window to fill the stack."
        )
        self.stack_channels_check.setChecked(C > 1)
        cf_form.addRow("", self.stack_channels_check)

        # Region picker
        region_w = QWidget()
        region_v = QVBoxLayout(region_w)
        region_v.setContentsMargins(0, 0, 0, 0)
        region_v.setSpacing(3)

        radio_row = QHBoxLayout()
        radio_row.setSpacing(8)
        self.full_radio = QRadioButton("Full field")
        self.full_radio.setChecked(True)
        self.shape_radio = QRadioButton("ROI:")
        radio_row.addWidget(self.full_radio)
        radio_row.addWidget(self.shape_radio)
        self.shape_combo = QComboBox()
        self.shape_combo.setEnabled(False)
        self.shape_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        radio_row.addWidget(self.shape_combo, 1)
        region_v.addLayout(radio_row)

        self._region_group = QButtonGroup(self)
        self._region_group.addButton(self.full_radio)
        self._region_group.addButton(self.shape_radio)
        self._region_group.buttonToggled.connect(self._on_region_toggled)
        self.shape_combo.currentIndexChanged.connect(self._on_shape_selected)

        self.bounds_label = QLabel("")
        self.bounds_label.setStyleSheet("color: #888; font-size: 10px;")
        self.bounds_label.setVisible(False)
        region_v.addWidget(self.bounds_label)

        z_row = QWidget()
        z_layout = QHBoxLayout(z_row)
        z_layout.setContentsMargins(0, 0, 0, 0)
        z_layout.setSpacing(6)
        self.all_z_radio = QRadioButton("All Z")
        self.all_z_radio.setChecked(True)
        self.around_z_radio = QRadioButton("Around Z ±")
        self.z_half_spin = QSpinBox()
        self.z_half_spin.setRange(1, 256)
        self.z_half_spin.setValue(4)
        self.z_half_spin.setFixedWidth(48)
        self.z_half_spin.setEnabled(False)
        z_layout.addWidget(self.all_z_radio)
        z_layout.addWidget(self.around_z_radio)
        z_layout.addWidget(self.z_half_spin)
        z_layout.addStretch()
        z_row.setVisible(False)
        self._z_row_widget = z_row
        region_v.addWidget(z_row)

        self._z_group = QButtonGroup(self)
        self._z_group.addButton(self.all_z_radio)
        self._z_group.addButton(self.around_z_radio)
        self.around_z_radio.toggled.connect(self.z_half_spin.setEnabled)
        self.around_z_radio.toggled.connect(
            lambda *_: self._refresh_workflow_summary()
        )
        self.all_z_radio.toggled.connect(
            lambda *_: self._refresh_workflow_summary()
        )
        self.z_half_spin.valueChanged.connect(
            lambda *_: self._refresh_workflow_summary()
        )

        cf_form.addRow("Region:", region_w)
        vbox.addWidget(cf_grp)

        # PSF
        psf_grp = QGroupBox("PSF source")
        psf_form = QFormLayout(psf_grp)
        psf_form.setLabelAlignment(Qt.AlignRight)
        psf_form.setVerticalSpacing(4)
        psf_form.setContentsMargins(8, 6, 8, 6)

        psf_win_row = QWidget()
        psf_win_layout = QHBoxLayout(psf_win_row)
        psf_win_layout.setContentsMargins(0, 0, 0, 0)
        psf_win_layout.setSpacing(6)
        self.psf_combo = QComboBox()
        self.psf_combo.currentIndexChanged.connect(self._on_psf_selected)
        psf_win_layout.addWidget(self.psf_combo, 1)
        psf_win_layout.addWidget(QLabel("Ch:"))
        self.psf_channel_spin = QSpinBox()
        self.psf_channel_spin.setRange(0, 0)
        self.psf_channel_spin.setFixedWidth(48)
        psf_win_layout.addWidget(self.psf_channel_spin)
        psf_form.addRow("Use:", psf_win_row)

        self.scale_label = QLabel("—")
        self.scale_label.setStyleSheet("font-size: 10px;")
        self.scale_label.setToolTip(
            "Compares PSF sampling with the current data sampling and "
            "super-resolution factors."
        )
        psf_form.addRow("Sampling:", self.scale_label)

        compute_row = QWidget()
        compute_layout = QHBoxLayout(compute_row)
        compute_layout.setContentsMargins(0, 0, 0, 0)
        compute_layout.setSpacing(6)
        self.compute_psf_btn = QPushButton("Compute PSF…")
        self.compute_psf_btn.setToolTip(
            "Open the Compute PSF dialog with shape, spacing, and optics "
            "pre-filled for the current ROI, super-res factor, and detector "
            "padding."
        )
        self.compute_psf_btn.clicked.connect(self._spawn_psf_dialog)
        compute_layout.addWidget(self.compute_psf_btn)
        self.compute_target_label = QLabel("")
        self.compute_target_label.setStyleSheet("color: #888; font-size: 10px;")
        compute_layout.addWidget(self.compute_target_label, 1)
        psf_form.addRow("", compute_row)

        vbox.addWidget(psf_grp)
        vbox.addStretch()

        self._refresh_shape_combo()
        self._refresh_psf_combo()
        return tab

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

        _T, Z_dim, _C, _Y, _X = self.viewer.img_data.shape
        is_3d = Z_dim > 1

        # Algorithm
        algo_grp = QGroupBox("Algorithm")
        algo_layout = QHBoxLayout(algo_grp)
        algo_layout.setContentsMargins(8, 6, 8, 6)
        self.mem_radio = QRadioButton("MaxEnt (memsolve)")
        self.mem_radio.setChecked(True)
        self.rl_radio = QRadioButton("Richardson–Lucy")
        algo_layout.addWidget(self.mem_radio)
        algo_layout.addWidget(self.rl_radio)
        algo_layout.addStretch()
        self._algo_group = QButtonGroup(self)
        self._algo_group.addButton(self.mem_radio)
        self._algo_group.addButton(self.rl_radio)
        self.mem_radio.toggled.connect(self._on_algorithm_changed)
        vbox.addWidget(algo_grp)

        # Recipe
        fm_grp = QGroupBox("Recipe")
        fm_form = QFormLayout(fm_grp)
        fm_form.setLabelAlignment(Qt.AlignRight)
        fm_form.setVerticalSpacing(4)
        fm_form.setContentsMargins(8, 6, 8, 6)

        sr_row = QWidget()
        sr_layout = QHBoxLayout(sr_row)
        sr_layout.setContentsMargins(0, 0, 0, 0)
        sr_layout.setSpacing(6)
        sr_layout.addWidget(QLabel("XY ×:"))
        self.sr_xy_spin = QSpinBox()
        self.sr_xy_spin.setRange(1, 8)
        self.sr_xy_spin.setSingleStep(1)
        self.sr_xy_spin.setValue(1)
        self.sr_xy_spin.setFixedWidth(72)
        self.sr_xy_spin.valueChanged.connect(self._on_sr_changed)
        self.sr_xy_spin.setToolTip(
            "Fine-grid reconstruction factor in X/Y. 1 keeps native "
            "camera sampling; whole-number values above 1 use detector-area integration."
        )
        sr_layout.addWidget(self.sr_xy_spin)
        sr_layout.addSpacing(10)
        sr_layout.addWidget(QLabel("Z ×:"))
        self.sr_z_spin = QSpinBox()
        self.sr_z_spin.setRange(1, 8)
        self.sr_z_spin.setSingleStep(1)
        self.sr_z_spin.setValue(1)
        self.sr_z_spin.setFixedWidth(72)
        self.sr_z_spin.valueChanged.connect(self._on_sr_changed)
        self.sr_z_spin.setToolTip(
            "Axial fine-grid factor for 3D data. Disabled for single-plane data."
        )
        if not is_3d:
            # ndim=2 recipe path drops sr_z entirely; disable so the user
            # doesn't set a value that silently has no effect.
            self.sr_z_spin.setEnabled(False)
            self.sr_z_spin.setToolTip(
                "Image has a single Z plane — axial super-res has no effect."
            )
        sr_layout.addWidget(self.sr_z_spin)
        sr_layout.addStretch()
        fm_form.addRow("Super-res:", sr_row)

        self.margin_check = QCheckBox("Model signal just outside detector")
        self.margin_check.setToolTip(
            "Adds an unknown border around the measured detector field so "
            "edge photons can be explained without wrap-around artifacts."
        )
        self.margin_check.setChecked(True)
        self.margin_check.toggled.connect(self._on_margin_toggled)
        fm_form.addRow("Finite detector:", self.margin_check)

        pad_row = QWidget()
        pad_layout = QHBoxLayout(pad_row)
        pad_layout.setContentsMargins(0, 0, 0, 0)
        pad_layout.setSpacing(6)
        # Pad defaults: pick non-zero values so the first call to
        # "Compute PSF…" already passes a padded shape to the PSF dialog.
        pad_layout.addWidget(QLabel("XY:"))
        self.pad_xy_spin = QSpinBox()
        self.pad_xy_spin.setRange(0, 256)
        self.pad_xy_spin.setValue(16)
        self.pad_xy_spin.setFixedWidth(56)
        self.pad_xy_spin.setToolTip("Symmetric lateral margin in camera pixels.")
        self.pad_xy_spin.valueChanged.connect(self._on_margin_value_changed)
        pad_layout.addWidget(self.pad_xy_spin)
        pad_layout.addSpacing(10)
        pad_layout.addWidget(QLabel("Z:"))
        self.pad_z_spin = QSpinBox()
        self.pad_z_spin.setRange(0, 256)
        self.pad_z_spin.setValue(4 if is_3d else 0)
        self.pad_z_spin.setFixedWidth(56)
        self.pad_z_spin.setToolTip("Symmetric axial margin in camera pixels.")
        self.pad_z_spin.valueChanged.connect(self._on_margin_value_changed)
        if not is_3d:
            # ndim=2 recipe path drops pad_z; lock to 0 to match the recipe.
            self.pad_z_spin.setEnabled(False)
            self.pad_z_spin.setToolTip(
                "Image has a single Z plane — Z padding has no effect."
            )
        pad_layout.addWidget(self.pad_z_spin)
        self.margin_auto_btn = QPushButton("Auto")
        self.margin_auto_btn.setToolTip(
            "Use half of the selected PSF kernel size as the finite-detector margin."
        )
        self.margin_auto_btn.clicked.connect(self._auto_margin_from_psf)
        pad_layout.addWidget(self.margin_auto_btn)
        pad_layout.addStretch()
        self._margin_row_widget = pad_row
        fm_form.addRow("Margin:", pad_row)

        # Return region: applies to both MEM and RL. "valid" crops out
        # the detector-pad border from the hidden grid; useful when you
        # padded only to keep convolution edges clean.
        region_row = QWidget()
        region_layout = QHBoxLayout(region_row)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.setSpacing(8)
        self.region_full_radio = QRadioButton("Full canvas")
        self.region_full_radio.setChecked(True)
        self.region_valid_radio = QRadioButton("Detector field")
        region_layout.addWidget(self.region_full_radio)
        region_layout.addWidget(self.region_valid_radio)
        region_layout.addStretch()
        self._region_group_btn = QButtonGroup(self)
        self._region_group_btn.addButton(self.region_full_radio)
        self._region_group_btn.addButton(self.region_valid_radio)
        self._region_group_btn.buttonToggled.connect(
            lambda *_: self._refresh_workflow_summary()
        )
        self.region_full_radio.setToolTip(
            "Keep the full reconstructed domain, including any finite-detector margin."
        )
        self.region_valid_radio.setToolTip(
            "Crop the result to the measured detector field after convergence."
        )
        fm_form.addRow("Output:", region_row)

        vbox.addWidget(fm_grp)

        # ICF / wavelet group (MEM only)
        icf_grp = QGroupBox("ICF / wavelet (MEM only)")
        icf_v = QVBoxLayout(icf_grp)
        icf_v.setContentsMargins(8, 6, 8, 6)
        icf_v.setSpacing(4)

        icf_mode_row = QHBoxLayout()
        icf_mode_row.setSpacing(8)
        self.icf_off_radio = QRadioButton("Off")
        self.icf_off_radio.setChecked(True)
        self.icf_fixed_radio = QRadioButton("Gaussian")
        self.icf_sweep_radio = QRadioButton("Sweep")
        self.wavelet_radio = QRadioButton("Wavelet")
        icf_mode_row.addWidget(self.icf_off_radio)
        icf_mode_row.addWidget(self.icf_fixed_radio)
        icf_mode_row.addWidget(self.icf_sweep_radio)
        icf_mode_row.addWidget(self.wavelet_radio)
        icf_mode_row.addStretch()
        icf_v.addLayout(icf_mode_row)

        self._icf_group = QButtonGroup(self)
        self._icf_group.addButton(self.icf_off_radio)
        self._icf_group.addButton(self.icf_fixed_radio)
        self._icf_group.addButton(self.icf_sweep_radio)
        self._icf_group.addButton(self.wavelet_radio)
        self._icf_group.buttonToggled.connect(self._on_icf_mode_changed)

        self._icf_stack = QStackedWidget()
        # off page
        self._icf_stack.addWidget(QWidget())
        # fixed page
        fixed_w = QWidget()
        fixed_layout = QHBoxLayout(fixed_w)
        fixed_layout.setContentsMargins(0, 0, 0, 0)
        fixed_layout.addWidget(QLabel("σ (μm):"))
        self.icf_sigma_spin = QDoubleSpinBox()
        self.icf_sigma_spin.setRange(0.001, 10.0)
        self.icf_sigma_spin.setDecimals(3)
        self.icf_sigma_spin.setSingleStep(0.05)
        self.icf_sigma_spin.setValue(0.15)
        self.icf_sigma_spin.setFixedWidth(80)
        fixed_layout.addWidget(self.icf_sigma_spin)
        self.icf_use_last_btn = QPushButton("Use last sweep")
        self.icf_use_last_btn.setEnabled(False)
        self.icf_use_last_btn.clicked.connect(self._apply_last_sweep_sigma)
        fixed_layout.addWidget(self.icf_use_last_btn)
        fixed_layout.addStretch()
        self._icf_stack.addWidget(fixed_w)
        # sweep page
        sweep_w = QWidget()
        sweep_layout = QHBoxLayout(sweep_w)
        sweep_layout.setContentsMargins(0, 0, 0, 0)
        sweep_layout.addWidget(QLabel("σ list (μm, comma-sep):"))
        self.icf_sweep_edit = QLineEdit("0.05, 0.1, 0.2, 0.4")
        sweep_layout.addWidget(self.icf_sweep_edit, 1)
        self.icf_refine_check = QCheckBox("Refine")
        self.icf_refine_check.setChecked(True)
        sweep_layout.addWidget(self.icf_refine_check)
        self._icf_stack.addWidget(sweep_w)
        # wavelet page
        wavelet_w = QWidget()
        wavelet_layout = QHBoxLayout(wavelet_w)
        wavelet_layout.setContentsMargins(0, 0, 0, 0)
        wavelet_layout.setSpacing(6)
        wavelet_layout.addWidget(QLabel("Levels:"))
        self.wavelet_levels_spin = QSpinBox()
        self.wavelet_levels_spin.setRange(1, 8)
        self.wavelet_levels_spin.setValue(3)
        self.wavelet_levels_spin.setFixedWidth(56)
        self.wavelet_levels_spin.setToolTip(
            "Number of a-trous wavelet detail levels in signed coefficient space."
        )
        wavelet_layout.addWidget(self.wavelet_levels_spin)
        wavelet_layout.addWidget(QLabel("Kernel:"))
        self.wavelet_kernel_combo = QComboBox()
        self.wavelet_kernel_combo.addItem("B3 spline", "b3spline")
        self.wavelet_kernel_combo.addItem("Triangle", "triangle")
        self.wavelet_kernel_combo.setToolTip("Wavelet smoothing kernel.")
        wavelet_layout.addWidget(self.wavelet_kernel_combo)
        wavelet_layout.addWidget(QLabel("Prior ×:"))
        self.wavelet_prior_scale_spin = QDoubleSpinBox()
        self.wavelet_prior_scale_spin.setRange(0.01, 100.0)
        self.wavelet_prior_scale_spin.setDecimals(2)
        self.wavelet_prior_scale_spin.setSingleStep(0.5)
        self.wavelet_prior_scale_spin.setValue(5.0)
        self.wavelet_prior_scale_spin.setFixedWidth(72)
        self.wavelet_prior_scale_spin.setToolTip(
            "Multiplier for coefficient prior scales; larger values damp less."
        )
        wavelet_layout.addWidget(self.wavelet_prior_scale_spin)
        self.wavelet_allow_poisson_check = QCheckBox("Poisson")
        self.wavelet_allow_poisson_check.setToolTip(
            "Experimental: signed wavelet synthesis does not guarantee "
            "nonnegative visible intensities. Gaussian likelihood is safer."
        )
        wavelet_layout.addWidget(self.wavelet_allow_poisson_check)
        wavelet_layout.addStretch()
        self._icf_stack.addWidget(wavelet_w)

        icf_v.addWidget(self._icf_stack)
        self.icf_last_sweep_label = QLabel("Last sweep: —")
        self.icf_last_sweep_label.setStyleSheet("color: #888; font-size: 10px;")
        icf_v.addWidget(self.icf_last_sweep_label)
        vbox.addWidget(icf_grp)
        self._icf_group_widget = icf_grp

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
        diag_grp.setToolTip(
            "Shows derived array sizes, including the computed FFT canvas. "
            "There is no manual FFT padding control."
        )
        diag_form = QFormLayout(diag_grp)
        diag_form.setLabelAlignment(Qt.AlignRight)
        diag_form.setVerticalSpacing(3)
        diag_form.setContentsMargins(8, 6, 8, 6)
        self.detector_domain_label = QLabel("—")
        self.hidden_canvas_label = QLabel("—")
        self.psf_kernel_label = QLabel("—")
        self.fft_canvas_label = QLabel("—")
        diag_form.addRow("Detector domain:", self.detector_domain_label)
        diag_form.addRow("Hidden canvas:", self.hidden_canvas_label)
        diag_form.addRow("PSF kernel:", self.psf_kernel_label)
        diag_form.addRow("FFT canvas:", self.fft_canvas_label)
        self._diagnostics_group = diag_grp
        self._diagnostics_widgets = [
            self.detector_domain_label,
            self.hidden_canvas_label,
            self.psf_kernel_label,
            self.fft_canvas_label,
        ]
        diag_grp.toggled.connect(self._on_diagnostics_toggled)
        vbox.addWidget(diag_grp)

        # Algorithm-specific knob stacks
        self._mem_box = self._build_mem_knobs()
        self._rl_box = self._build_rl_knobs()
        vbox.addWidget(self._mem_box)
        vbox.addWidget(self._rl_box)
        vbox.addStretch()
        self._connect_recipe_preview_signals()
        self._on_margin_toggled(self.margin_check.isChecked())
        self._on_diagnostics_toggled(False)
        return tab

    def _build_mem_knobs(self) -> QGroupBox:
        grp = QGroupBox("MaxEnt solver")
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

        # Likelihood row spans 4 columns
        lk_row = QWidget()
        lk_layout = QHBoxLayout(lk_row)
        lk_layout.setContentsMargins(0, 0, 0, 0)
        lk_layout.setSpacing(8)
        self.poisson_radio = QRadioButton("Poisson")
        self.poisson_radio.setChecked(True)
        self.gaussian_radio = QRadioButton("Gaussian")
        lk_layout.addWidget(self.poisson_radio)
        lk_layout.addWidget(self.gaussian_radio)
        lk_layout.addWidget(QLabel("σ:"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.01, 1e6)
        self.sigma_spin.setDecimals(1)
        self.sigma_spin.setValue(50.0)
        self.sigma_spin.setSingleStep(10.0)
        self.sigma_spin.setFixedWidth(72)
        self.sigma_spin.setEnabled(False)
        lk_layout.addWidget(self.sigma_spin)
        lk_layout.addStretch()
        grid.addWidget(QLabel("Likelihood:"), 0, 0, Qt.AlignRight)
        grid.addWidget(lk_row, 0, 1, 1, 3)
        self._lk_group = QButtonGroup(self)
        self._lk_group.addButton(self.poisson_radio)
        self._lk_group.addButton(self.gaussian_radio)
        self.gaussian_radio.toggled.connect(self.sigma_spin.setEnabled)

        self.max_iter_spin    = _ispin(1, 500, 60)
        self.tol_omega_spin   = _dspin(1e-4, 1.0, 0.05, dec=4, step=0.01)
        self.rate_spin        = _dspin(0.01, 2.0, 0.3, dec=2, step=0.05)
        self.omega_mode_combo = QComboBox()
        self.omega_mode_combo.addItem("auto", "auto")
        self.omega_mode_combo.addItem("classic", "classic")
        self.cg_epsilon_spin  = _dspin(1e-6, 0.5, 0.01, dec=4, step=0.005)
        self.cg_max_steps_spin = _ispin(10, 500, 30)
        self.n_probe_g_spin   = _ispin(1, 16, 1, w=52)
        self.map_space_combo  = QComboBox()
        self.map_space_combo.addItem("hidden", "hidden")
        self.map_space_combo.addItem("data", "data")
        self.posterior_n_spin = _ispin(0, 1024, 0)

        def _row(r, la, wa, lb, wb):
            grid.addWidget(QLabel(la), r, 0, Qt.AlignRight)
            grid.addWidget(wa,         r, 1)
            grid.addWidget(QLabel(lb), r, 2, Qt.AlignRight)
            grid.addWidget(wb,         r, 3)

        _row(1, "max_iter:",   self.max_iter_spin,    "tol_ω:",     self.tol_omega_spin)
        _row(2, "rate:",       self.rate_spin,         "ω mode:",    self.omega_mode_combo)
        _row(3, "CG ε:",       self.cg_epsilon_spin,   "CG steps:",  self.cg_max_steps_spin)
        _row(4, "G probes:",   self.n_probe_g_spin,    "map_space:", self.map_space_combo)
        grid.addWidget(QLabel("Posterior samples:"), 5, 0, Qt.AlignRight)
        grid.addWidget(self.posterior_n_spin, 5, 1)
        post_note = QLabel("(0 = skip sampling)")
        post_note.setStyleSheet("color: #888; font-size: 10px;")
        grid.addWidget(post_note, 5, 2, 1, 2)
        return grp

    def _build_rl_knobs(self) -> QGroupBox:
        grp = QGroupBox("Richardson–Lucy solver")
        form = QFormLayout(grp)
        form.setLabelAlignment(Qt.AlignRight)
        form.setContentsMargins(8, 6, 8, 6)
        form.setVerticalSpacing(4)

        self.rl_iter_spin = QSpinBox()
        self.rl_iter_spin.setRange(1, 5000)
        self.rl_iter_spin.setValue(50)
        self.rl_iter_spin.setFixedWidth(72)
        form.addRow("Iterations:", self.rl_iter_spin)

        self.rl_bg_spin = QDoubleSpinBox()
        self.rl_bg_spin.setRange(0.0, 1e6)
        self.rl_bg_spin.setDecimals(2)
        self.rl_bg_spin.setValue(0.0)
        self.rl_bg_spin.setFixedWidth(80)
        form.addRow("Background:", self.rl_bg_spin)

        self.rl_eval_spin = QSpinBox()
        self.rl_eval_spin.setRange(1, 500)
        self.rl_eval_spin.setValue(10)
        self.rl_eval_spin.setFixedWidth(72)
        form.addRow("Eval interval:", self.rl_eval_spin)
        return grp

    # ------------------------------------------------------------------ #
    # Dynamic visibility

    def _connect_recipe_preview_signals(self) -> None:
        widgets = [
            self.mem_radio, self.rl_radio,
            self.icf_off_radio, self.icf_fixed_radio,
            self.icf_sweep_radio, self.wavelet_radio,
            self.icf_refine_check, self.wavelet_allow_poisson_check,
        ]
        for widget in widgets:
            widget.toggled.connect(lambda *_: self._refresh_workflow_summary())
        for spin in (
            self.icf_sigma_spin, self.wavelet_levels_spin,
            self.wavelet_prior_scale_spin, self.rl_iter_spin,
            self.rl_bg_spin, self.rl_eval_spin, self.max_iter_spin,
            self.posterior_n_spin,
        ):
            spin.valueChanged.connect(lambda *_: self._refresh_workflow_summary())
        self.icf_sweep_edit.textChanged.connect(
            lambda *_: self._refresh_workflow_summary()
        )
        self.wavelet_kernel_combo.currentIndexChanged.connect(
            lambda *_: self._refresh_workflow_summary()
        )

    def _refresh_workflow_summary(self) -> None:
        if hasattr(self, "compute_target_label"):
            self._refresh_compute_target_hint()
        if hasattr(self, "recipe_preview"):
            self._refresh_recipe_preview()

    def _on_margin_toggled(self, checked: bool) -> None:
        self._margin_row_widget.setVisible(bool(checked))
        self.pad_xy_spin.setEnabled(bool(checked))
        self.margin_auto_btn.setEnabled(bool(checked))
        is_3d = self.viewer.img_data.shape[1] > 1
        self.pad_z_spin.setEnabled(bool(checked) and is_3d)
        self._refresh_workflow_summary()

    def _on_margin_value_changed(self, _value) -> None:
        self._refresh_workflow_summary()
        win = self.psf_combo.currentData()
        if win is not None:
            self._check_scale_match(win)

    def _auto_margin_from_psf(self) -> None:
        kernel = self._current_psf_kernel_shape()
        if kernel is None:
            self._set_status("Select a PSF window before using Auto margin.", warn=True)
            return
        if len(kernel) == 2:
            py = px = max(0, int(kernel[-1]) // 2)
            pz = 0
        else:
            pz = max(0, int(kernel[0]) // 2)
            py = px = max(0, int(kernel[-1]) // 2)
        self.margin_check.setChecked(True)
        self.pad_xy_spin.setValue(max(py, px))
        self.pad_z_spin.setValue(pz if self.viewer.img_data.shape[1] > 1 else 0)
        self._refresh_workflow_summary()

    def _on_diagnostics_toggled(self, checked: bool) -> None:
        for child in self._diagnostics_group.findChildren(QWidget):
            if child is not self._diagnostics_group:
                child.setVisible(bool(checked))
        self._diagnostics_group.setMaximumHeight(16777215 if checked else 24)

    def _effective_pad(self) -> tuple[int, int]:
        if not self.margin_check.isChecked():
            return 0, 0
        return int(self.pad_xy_spin.value()), int(self.pad_z_spin.value())

    def _current_data_shape(self) -> Optional[tuple[int, ...]]:
        _T, Z, _C, Y, X = self.viewer.img_data.shape
        if self.full_radio.isChecked():
            n_z, n_y, n_x = Z, Y, X
        else:
            idx = self.shape_combo.currentIndex()
            if idx < 0 or idx >= len(self._rect_shapes):
                return None
            _, rec = self._rect_shapes[idx]
            p = rec.params
            n_y = int(abs(p[3] - p[1]))
            n_x = int(abs(p[2] - p[0]))
            if Z > 1 and self.around_z_radio.isChecked():
                half = self.z_half_spin.value()
                n_z = min(Z, rec.z + half + 1) - max(0, rec.z - half)
            else:
                n_z = Z
        if n_z <= 1:
            return (n_y, n_x)
        return (n_z, n_y, n_x)

    def _current_shape_plan(self) -> Optional[dict[str, tuple[int, ...]]]:
        data_shape = self._current_data_shape()
        if data_shape is None:
            return None
        ndim = len(data_shape)
        f_xy = max(float(self.sr_xy_spin.value()), 1.0)
        f_z = max(float(self.sr_z_spin.value()), 1.0)
        pad_xy, pad_z = self._effective_pad()
        factors = ((f_z, f_xy, f_xy) if ndim == 3 else (f_xy, f_xy))
        pads = ((pad_z, pad_xy, pad_xy) if ndim == 3 else (pad_xy, pad_xy))
        detector_domain = tuple(d + 2 * p for d, p in zip(data_shape, pads))
        hidden = tuple(
            int(round(d * f)) for d, f in zip(detector_domain, factors)
        )
        valid = tuple(int(round(d * f)) for d, f in zip(data_shape, factors))
        return {
            "data": tuple(data_shape),
            "detector_domain": detector_domain,
            "hidden": hidden,
            "valid": valid,
            "pads": tuple(pads),
            "factors": tuple(factors),
        }

    def _current_psf_kernel_shape(self) -> Optional[tuple[int, ...]]:
        win = self.psf_combo.currentData()
        if win is None:
            return None
        _T, Zp, _C, Yp, Xp = win.img_data.shape
        if self.viewer.img_data.shape[1] <= 1 or Zp <= 1:
            return (Yp, Xp)
        return (Zp, Yp, Xp)

    def _format_shape(self, shape: Optional[tuple[int, ...]]) -> str:
        if shape is None:
            return "—"
        return " × ".join(str(int(v)) for v in shape)

    def _next_smooth_number(self, n: int) -> int:
        candidate = max(1, int(n))
        while True:
            m = candidate
            for p in (2, 3, 5):
                while m % p == 0:
                    m //= p
            if m == 1:
                return candidate
            candidate += 1

    def _fft_canvas_shape(
        self,
        signal_shape: tuple[int, ...],
        kernel_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        return tuple(
            self._next_smooth_number(max(int(n) + int(m) - 1, int(m)))
            for n, m in zip(signal_shape, kernel_shape)
        )

    def _refresh_recipe_preview(self) -> None:
        if not hasattr(self, "recipe_preview"):
            return
        plan = self._current_shape_plan()
        if plan is None:
            text = "Select a detector region to preview the recipe."
            self.recipe_preview.setPlainText(text)
            return
        pad_any = any(int(p) > 0 for p in plan["pads"])
        sr_any = any(abs(float(f) - 1.0) > 1e-6 for f in plan["factors"])
        recipe_kind = "super_res_idc" if sr_any else "fft_conv"
        output_shape = (
            plan["valid"] if self.region_valid_radio.isChecked() else plan["hidden"]
        )

        if self.mem_radio.isChecked():
            if self.wavelet_radio.isChecked():
                solver = (
                    f"MEM wavelet, {self.wavelet_levels_spin.value()} levels, "
                    f"{self.wavelet_kernel_combo.currentData()}"
                )
            elif self.icf_sweep_radio.isChecked():
                solver = f"MEM + Gaussian ICF sweep ({self.icf_sweep_edit.text()})"
            elif self.icf_fixed_radio.isChecked():
                solver = f"MEM + Gaussian ICF σ={self.icf_sigma_spin.value():g} µm"
            else:
                solver = "MEM"
        else:
            solver = f"Richardson-Lucy, {self.rl_iter_spin.value()} iter"

        pad_text = self._format_shape(plan["pads"]) if pad_any else "none"
        sr_text = " × ".join(f"{float(v):g}" for v in plan["factors"])
        lines = [
            f"{solver}",
            f"{recipe_kind}: data {self._format_shape(plan['data'])} -> "
            f"hidden {self._format_shape(plan['hidden'])} -> "
            f"output {self._format_shape(output_shape)}",
            f"super-res {sr_text}; finite margin {pad_text}",
        ]
        self.recipe_preview.setPlainText("\n".join(lines))

        kernel = self._current_psf_kernel_shape()
        fft_shape = (
            self._fft_canvas_shape(plan["hidden"], kernel)
            if kernel is not None and len(kernel) == len(plan["hidden"])
            else None
        )
        self.detector_domain_label.setText(self._format_shape(plan["detector_domain"]))
        self.hidden_canvas_label.setText(self._format_shape(plan["hidden"]))
        self.psf_kernel_label.setText(self._format_shape(kernel))
        self.fft_canvas_label.setText(self._format_shape(fft_shape))

    def _on_algorithm_changed(self, *_):
        is_mem = self.mem_radio.isChecked()
        self._icf_group_widget.setVisible(is_mem)
        self._mem_box.setVisible(is_mem)
        self._rl_box.setVisible(not is_mem)
        self._refresh_recipe_preview()

    def _on_icf_mode_changed(self, *_):
        if self.icf_off_radio.isChecked():
            self._icf_stack.setCurrentIndex(0)
        elif self.icf_fixed_radio.isChecked():
            self._icf_stack.setCurrentIndex(1)
        elif self.icf_sweep_radio.isChecked():
            self._icf_stack.setCurrentIndex(2)
        else:
            self._icf_stack.setCurrentIndex(3)
            if not self.wavelet_allow_poisson_check.isChecked():
                self.gaussian_radio.setChecked(True)
            self.posterior_n_spin.setValue(0)
        self.posterior_n_spin.setEnabled(not self.wavelet_radio.isChecked())
        self._refresh_recipe_preview()

    def _apply_last_sweep_sigma(self):
        if self._last_icf_sigma_um is None:
            return
        self.icf_fixed_radio.setChecked(True)
        self.icf_sigma_spin.setValue(self._last_icf_sigma_um)
        self._set_status(
            f"Using last sweep σ = {self._last_icf_sigma_um:.4g} µm for fixed ICF.",
            ok=True,
        )

    def _on_sr_changed(self, _value):
        self._refresh_workflow_summary()
        # PSF expected fine-grid spacing depends on the sr factors, so the
        # ✓/✗ scale indicator needs to refresh whenever they change.
        win = self.psf_combo.currentData()
        if win is not None:
            self._check_scale_match(win)

    # ------------------------------------------------------------------ #
    # PSF compute spawn

    def _expected_psf_shape_spacing(self):
        """Return (shape, spacing) the PSF needs given the current ROI + recipe.

        Both are returned in `(Nz, Ny, Nx)` / `(dz, dy, dx)` order when 3D,
        falling back to 2D `(Ny, Nx)` / `(dy, dx)` when the ROI has a
        single Z plane. Returns `(None, None)` if no ROI is resolvable.
        """
        data_shape = self._current_data_shape()
        if data_shape is None:
            return None, None
        if len(data_shape) == 2:
            n_y, n_x = data_shape
            n_z = 1
        else:
            n_z, n_y, n_x = data_shape

        f_xy = max(float(self.sr_xy_spin.value()), 1.0)
        f_z = max(float(self.sr_z_spin.value()), 1.0)
        pad_xy, pad_z = self._effective_pad()
        n_y_h = int(round((n_y + 2 * pad_xy) * f_xy))
        n_x_h = int(round((n_x + 2 * pad_xy) * f_xy))
        n_z_h = int(round((n_z + 2 * pad_z) * f_z))

        scale = self.viewer.meta.get("scale", (1.0, 1.0, 1.0))
        dz, dy, dx = float(scale[0]), float(scale[1]), float(scale[2])
        dy /= f_xy
        dx /= f_xy
        dz /= f_z

        if n_z <= 1:
            return (n_y_h, n_x_h), (dy, dx)
        return (n_z_h, n_y_h, n_x_h), (dz, dy, dx)

    def _refresh_compute_target_hint(self):
        shape, spacing = self._expected_psf_shape_spacing()
        if shape is None:
            self.compute_target_label.setText("")
            return
        if len(shape) == 2:
            self.compute_target_label.setText(
                f"{shape[0]}×{shape[1]} px, "
                f"{spacing[0]*1000:.0f}×{spacing[1]*1000:.0f} nm"
            )
        else:
            self.compute_target_label.setText(
                f"{shape[0]}×{shape[1]}×{shape[2]} px, "
                f"{spacing[0]*1000:.0f}×{spacing[1]*1000:.0f}×{spacing[2]*1000:.0f} nm"
            )

    def _spawn_psf_dialog(self):
        from .psf_dialog import PSFComputeDialog

        shape, spacing = self._expected_psf_shape_spacing()
        if shape is None:
            self._set_status(
                "Select a rectangle ROI before computing a PSF.", warn=True
            )
            return

        # Seed optics from the currently selected PSF window if any;
        # otherwise leave the PSF dialog's defaults in place.
        seed = {}
        psf_win = self.psf_combo.currentData()
        if psf_win is not None:
            params = psf_win.meta.get("parameters", {})
            modality = psf_win.meta.get("modality", "widefield")
            seed["modality"] = modality
            if modality == "spinning_disk":
                wl = params.get("wavelength_em")
            else:
                wl = params.get("wavelength")
            if wl is not None:
                seed["wavelength_um"] = wl
            for k in ("na", "ni", "ns"):
                if k in params:
                    seed[k] = params[k]

        dlg = PSFComputeDialog(parent=self)
        dlg.preset(shape=shape, spacing=spacing, **seed)
        dlg.psf_computed.connect(lambda _buf: self._refresh_psf_combo())
        dlg.show()

    # ------------------------------------------------------------------ #
    # PSF / shape helpers

    def _refresh_psf_combo(self):
        self.psf_combo.blockSignals(True)
        self.psf_combo.clear()
        for win in manager.get_all().values():
            if getattr(win, "is_psf", False) and win is not self.viewer:
                label = win.windowTitle().split("]", 1)[-1].strip()
                self.psf_combo.addItem(label, win)
        self.psf_combo.blockSignals(False)
        self._on_psf_selected(self.psf_combo.currentIndex())

    def _on_psf_selected(self, _idx):
        win = self.psf_combo.currentData()
        if win is None:
            self.psf_channel_spin.setRange(0, 0)
            self.scale_label.setText("No PSF window")
            self.scale_label.setStyleSheet("color: #F44; font-size: 10px;")
            self._refresh_recipe_preview()
            return
        _T, _Zp, C, _Yp, _Xp = win.img_data.shape
        self.psf_channel_spin.setRange(0, max(0, C - 1))
        self._check_scale_match(win)
        self._refresh_recipe_preview()

    def _check_scale_match(self, psf_win):
        """Compare PSF spacing against the expected fine-grid spacing.

        The PSF is computed at fine-grid sampling — i.e. ``src_scale``
        divided by the per-axis super-res factor — and for a 2D image
        only the lateral axes participate (the recipe drops the Z axis
        for ``ndim==2``). Both effects are accounted for here so the
        ✓/✗ indicator reflects the same axes the recipe actually uses.
        """
        src_scale = self.viewer.meta.get("scale", (1.0, 1.0, 1.0))
        psf_scale = psf_win.meta.get("scale", psf_win.meta.get("spacing"))
        if psf_scale is None:
            self.scale_label.setText("PSF has no scale metadata")
            self.scale_label.setStyleSheet("color: #C84; font-size: 10px;")
            return

        Z_dim = self.viewer.img_data.shape[1]
        f_xy = max(float(self.sr_xy_spin.value()), 1.0)
        f_z = max(float(self.sr_z_spin.value()), 1.0)
        # Fine-grid spacing the PSF should be sampled at.
        expected = [
            float(src_scale[0]) / f_z,
            float(src_scale[1]) / f_xy,
            float(src_scale[2]) / f_xy,
        ]
        # 2D image: ignore Z, only check (dy, dx).
        if Z_dim <= 1:
            expected_cmp = expected[1:]
            psf_cmp = list(psf_scale)[-2:]
        else:
            expected_cmp = expected
            psf_cmp = list(psf_scale)[:3]

        # PSFComputeDialog stores spacing rounded to 1 nm (its spinners use
        # 3-decimal μm precision). Compare at that same precision so the ✓/✗
        # indicator agrees with the displayed nm values — a sub-nm float-
        # rounding gap (e.g. 0.054166 μm preset → 0.054 μm stored) is not a
        # real mismatch.
        nm_e_vals = [round(s * 1000) for s in expected_cmp]
        nm_p_vals = [round(s * 1000) for s in psf_cmp]
        match = (
            len(nm_e_vals) == len(nm_p_vals)
            and nm_e_vals == nm_p_vals
        )
        nm_e = "×".join(str(v) for v in nm_e_vals)
        if match:
            self.scale_label.setText(f"✓  {nm_e} nm")
            self.scale_label.setStyleSheet("color: #4A4; font-size: 10px;")
        else:
            nm_p = "×".join(str(v) for v in nm_p_vals)
            self.scale_label.setText(
                f"✗  expected {nm_e} ≠ PSF {nm_p} nm"
            )
            self.scale_label.setStyleSheet("color: #F44; font-size: 10px;")

    def _refresh_shape_combo(self):
        self.shape_combo.blockSignals(True)
        self.shape_combo.clear()
        self._rect_shapes = []
        for layer in self.viewer.layers.by_type("shapes"):
            for rec in (layer.data.get(sid) for sid in layer.data.shape_ids):
                if rec.shape_type == RECTANGLE:
                    self._rect_shapes.append((layer.name, rec))
                    p = rec.params
                    x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
                    xl, xr = int(min(x1, x2)), int(max(x1, x2))
                    yt, yb = int(min(y1, y2)), int(max(y1, y2))
                    self.shape_combo.addItem(
                        f"{layer.name} #{rec.shape_id} "
                        f"({xr-xl}×{yb-yt})"
                    )
        self.shape_combo.blockSignals(False)
        has_shapes = bool(self._rect_shapes)
        self.shape_radio.setEnabled(has_shapes)
        if not has_shapes and self.shape_radio.isChecked():
            self.full_radio.setChecked(True)
        self._on_shape_selected(self.shape_combo.currentIndex())

    def _on_region_toggled(self, _btn, _checked):
        use_shape = self.shape_radio.isChecked()
        self.shape_combo.setEnabled(use_shape)
        self.bounds_label.setVisible(use_shape)
        self._on_shape_selected(self.shape_combo.currentIndex())
        self._refresh_workflow_summary()

    def _on_shape_selected(self, idx):
        if not self.shape_radio.isChecked() or idx < 0 or idx >= len(self._rect_shapes):
            self.bounds_label.setText("")
            self._z_row_widget.setVisible(False)
            return
        _, rec = self._rect_shapes[idx]
        p = rec.params
        x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
        xl, xr = int(min(x1, x2)), int(max(x1, x2))
        yt, yb = int(min(y1, y2)), int(max(y1, y2))
        _T, Z, _C, _Y, _X = self.viewer.img_data.shape
        self.bounds_label.setText(
            f"y [{yt}:{yb}]  x [{xl}:{xr}]  →  {yb-yt} × {xr-xl} px"
        )
        self._z_row_widget.setVisible(Z > 1)
        self.frame_spin.setValue(rec.t)
        self._refresh_workflow_summary()

    # ------------------------------------------------------------------ #
    # State / inputs

    def _read_state(self) -> DecondialogState:
        algo = "memsolve_mem" if self.mem_radio.isChecked() else "richardson_lucy"
        if self.icf_off_radio.isChecked():
            icf_mode = "off"
        elif self.icf_fixed_radio.isChecked():
            icf_mode = "fixed"
        elif self.icf_sweep_radio.isChecked():
            icf_mode = "sweep"
        else:
            icf_mode = "wavelet"

        sweep_sigmas: tuple = ()
        if icf_mode == "sweep":
            raw = self.icf_sweep_edit.text().strip()
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            try:
                sweep_sigmas = tuple(float(p) for p in parts)
            except ValueError:
                sweep_sigmas = ()

        pad_xy, pad_z = self._effective_pad()
        return DecondialogState(
            algorithm=algo,
            super_res_xy=self.sr_xy_spin.value(),
            super_res_z=self.sr_z_spin.value(),
            pad_xy=pad_xy,
            pad_z=pad_z,
            icf_mode=icf_mode,
            icf_sigma_um=self.icf_sigma_spin.value(),
            icf_sweep_sigmas_um=sweep_sigmas,
            icf_refine=self.icf_refine_check.isChecked(),
            wavelet_levels=self.wavelet_levels_spin.value(),
            wavelet_kernel=self.wavelet_kernel_combo.currentData(),
            wavelet_prior_scale=self.wavelet_prior_scale_spin.value(),
            wavelet_allow_poisson=self.wavelet_allow_poisson_check.isChecked(),
            likelihood="poisson" if self.poisson_radio.isChecked() else "gaussian",
            sigma_gaussian=self.sigma_spin.value(),
            max_iter=self.max_iter_spin.value(),
            tol_omega=self.tol_omega_spin.value(),
            rate=self.rate_spin.value(),
            omega_mode=self.omega_mode_combo.currentData(),
            cg_epsilon=self.cg_epsilon_spin.value(),
            cg_max_steps=self.cg_max_steps_spin.value(),
            n_probe_g=self.n_probe_g_spin.value(),
            map_space=self.map_space_combo.currentData(),
            posterior_n_samples=self.posterior_n_spin.value(),
            rl_num_iter=self.rl_iter_spin.value(),
            rl_background=self.rl_bg_spin.value(),
            rl_eval_interval=self.rl_eval_spin.value(),
            return_region=("valid" if self.region_valid_radio.isChecked()
                           else "full"),
        )

    def _build_optics_from_psf_window(self, psf_win):
        """Build a `deconlib.Optics` from a PSF window's stored parameters."""
        from deconlib import Optics

        params = psf_win.meta.get("parameters", {})
        modality = psf_win.meta.get("modality", "widefield")
        if modality == "spinning_disk":
            wavelength = params.get("wavelength_em", 0.525)
        else:
            wavelength = params.get("wavelength", 0.525)
        na = params.get("na", 1.4)
        ni = params.get("ni", 1.515)
        ns = params.get("ns", ni)
        return Optics(wavelength=float(wavelength), na=float(na),
                      ni=float(ni), ns=float(ns))

    def _prepare(self) -> Optional[object]:
        """Read inputs, slice the source, prepare the deconlib payload.

        Returns a :class:`PreparedInputs` on success, or `None` on a user
        error (with the status label updated).
        """
        _T, Z, _C, Y, X = self.viewer.img_data.shape
        c = self.channel_spin.value()
        t = self.frame_spin.value()

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
            p = rec.params
            x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
            yl, yr = int(min(y1, y2)), int(max(y1, y2))
            xl, xr = int(min(x1, x2)), int(max(x1, x2))
            if yl == yr or xl == xr:
                self._set_status("Shape has zero area.", error=True)
                return None
            y_slice = slice(yl, yr)
            x_slice = slice(xl, xr)
            if Z > 1 and self.around_z_radio.isChecked():
                half = self.z_half_spin.value()
                z_slice = slice(max(0, rec.z - half), min(Z, rec.z + half + 1))
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
        if not psf_win.meta.get("psf_dc_corner", True):
            psf_data = np.fft.fftshift(psf_data)

        y_obs = np.ascontiguousarray(
            self.viewer.img_data[t, z_slice, c, y_slice, x_slice]
        ).astype(np.float32)
        log.info(
            "dialog: t=%d c=%d z_slice=%s y_slice=%s x_slice=%s "
            "y_obs=%s psf_window='%s' psf_channel=%d psf_window_shape=%s "
            "psf_dc_corner=%s psf_spacing=%s viewer_scale=%s",
            t, c, z_slice, y_slice, x_slice,
            tuple(y_obs.shape),
            psf_win.windowTitle(), c_psf, tuple(psf_data.shape),
            psf_win.meta.get("psf_dc_corner", True),
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

        optics = self._build_optics_from_psf_window(psf_win)

        state = self._read_state()
        if state.icf_mode == "sweep" and not state.icf_sweep_sigmas_um:
            self._set_status(
                "ICF sweep requires a comma-separated σ list.", error=True
            )
            return None
        if state.icf_mode == "wavelet":
            if state.posterior_n_samples > 0:
                self._set_status(
                    "Wavelet MEM does not support posterior sampling yet.",
                    error=True,
                )
                return None
            if state.likelihood == "poisson" and not state.wavelet_allow_poisson:
                self._set_status(
                    "Wavelet MEM uses Gaussian likelihood by default. Enable "
                    "the Poisson checkbox for experimental Poisson runs.",
                    error=True,
                )
                return None

        try:
            prepared = prepare_inputs(
                state=state,
                y_obs=y_obs,
                psf_array=psf_data,
                psf_pixel_size_um=psf_pixel_size,
                optics=optics,
                require_psf_match=False,
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
        result = self._prepare()
        if result is None:
            return
        prepared, state, t, c = result
        self._last_run_state = state

        stack_channels = self.stack_channels_check.isChecked()
        output_channels = self.viewer.C if stack_channels else 1
        output_channel = c if stack_channels else 0
        output_shape = output_5d_shape(
            state, prepared.geometry.data_shape, n_channels=output_channels
        )
        log.info(
            "output buffer shape=%s channel=%d (algorithm=%s, return_region=%s)",
            output_shape, output_channel, state.algorithm, state.return_region,
        )

        # Output spacing = fine-grid spacing (data spacing ÷ super-res factor).
        # voxel_spacing in the geometry is already (dz, dy, dx) for 3D or
        # (dy, dx) for 2D — pad to a 3-tuple for the (T,Z,C,Y,X) viewer.
        vs = prepared.geometry.voxel_spacing
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
            "source_frame": t,
            "algorithm": state.algorithm,
        }

        source, buffer = self._runner.prepare_output(
            output_shape=output_shape,
            output_dtype=np.dtype(np.float32),
            output_meta=output_meta,
            reuse_existing_buffer=stack_channels,
        )

        if state.algorithm == "memsolve_mem":
            self.progress_bar.setRange(0, 0)
            worker = MemDeconvolutionWorker(
                prepared=prepared,
                map_config=build_mem_config(state),
                icf_sweep=build_icf_sweep(state),
                posterior=build_posterior(state),
                buffer=buffer,
                wavelet_config=build_wavelet_config(state),
                output_channel=output_channel,
                likelihood=state.likelihood,
            )
        else:
            self.progress_bar.setRange(0, max(1, state.rl_num_iter))
            worker = RLDeconvolutionWorker(
                prepared=prepared,
                rl_config=build_rl_config(state),
                buffer=buffer,
                output_channel=output_channel,
            )

        worker.status.connect(self._on_status)
        self.progress_bar.setValue(0)
        self._begin_run_status(self._describe_run(state))
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

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
        completion_note = self._capture_mem_sweep_result(result)
        if self._runner.output_type == "file":
            result = self._runner.finalize_output()
            msg = (
                f"Done — saved to {result}" if result else "Done (save cancelled)"
            )
            if completion_note:
                msg = f"{msg}\n{completion_note}"
            self._set_status(
                msg,
                ok=bool(result), warn=not bool(result),
            )
        else:
            msg = "Done"
            if completion_note:
                msg = f"{msg}\n{completion_note}"
            self._set_status(msg, ok=True)
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

    def _cleanup_thread(self):
        self._runner.cleanup()

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

    def _describe_run(self, state: DecondialogState) -> str:
        if state.algorithm != "memsolve_mem":
            return (
                f"Running Richardson–Lucy for {state.rl_num_iter} iterations "
                f"(eval every {state.rl_eval_interval})."
            )

        if state.icf_mode == "wavelet":
            return (
                f"Running wavelet MEM with {state.wavelet_levels} a-trous "
                f"level(s) ({state.wavelet_kernel})."
            )

        if state.icf_mode == "sweep":
            sigmas = ", ".join(f"{s:g}" for s in state.icf_sweep_sigmas_um)
            refine_text = (
                "then refine the best σ"
                if state.icf_refine else
                "without refinement"
            )
            return (
                f"Running MEM workflow: baseline plus ICF sweep over "
                f"{len(state.icf_sweep_sigmas_um)} σ values ({sigmas}), "
                f"{refine_text}. Best σ will be available for fixed ICF reuse."
            )

        if state.icf_mode == "fixed":
            return (
                f"Running MEM workflow with fixed ICF σ = "
                f"{state.icf_sigma_um:.4g} µm."
            )

        return "Running MEM workflow without ICF."

    def _capture_mem_sweep_result(self, result) -> Optional[str]:
        state = self._last_run_state
        if state is None or state.algorithm != "memsolve_mem":
            return None

        sigma_um = self._extract_icf_sigma_um(result)
        scan_rows = len(getattr(result, "scan", ()) or ())
        refined = bool(getattr(result, "refined", False))

        if sigma_um is not None:
            self._last_icf_sigma_um = sigma_um
            self.icf_use_last_btn.setEnabled(True)
            refine_suffix = " after refinement" if refined else ""
            self.icf_last_sweep_label.setText(
                f"Last sweep: best σ = {sigma_um:.4g} µm "
                f"from {scan_rows} candidate(s){refine_suffix}"
            )

            if state.icf_mode == "sweep" and not state.icf_refine:
                self.icf_fixed_radio.setChecked(True)
                self.icf_sigma_spin.setValue(sigma_um)
                return (
                    f"Sweep done. Best σ = {sigma_um:.4g} µm copied to fixed ICF "
                    f"for the next run."
                )
        elif state.icf_mode == "sweep":
            self.icf_last_sweep_label.setText(
                f"Last sweep: completed ({scan_rows} candidate(s)); "
                f"best σ unavailable from result metadata"
            )
        return None

    def _extract_icf_sigma_um(self, result) -> Optional[float]:
        chosen = getattr(result, "chosen_recipe", None)
        icf = getattr(chosen, "icf", None)
        if isinstance(icf, dict):
            sigmas = icf.get("sigmas_um")
            if isinstance(sigmas, (tuple, list)) and sigmas:
                try:
                    return float(sigmas[0])
                except (TypeError, ValueError):
                    return None
        return None

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
