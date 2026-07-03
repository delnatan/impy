"""PSF distillation dialog — NLCG engine (deconlib.deconvolution.nlcg_*).

PSF distillation is deconvolution with the roles of PSF and data swapped
(see ``psf_distillation_nlcg.py`` and
``deconlib/scripts/psf_distillation_nlcg_demo.py``): a bead-stack image is
the observation, an ideal point-source comb detected from it is the fixed
kernel, and the PSF is the unknown NLCG recovers. This dialog mirrors
:class:`~.deconvolution_dialog.DeconvolutionDialog`'s two-tab layout and
solver UI, with the source/detection controls adapted for beads instead of
a sample + known PSF:

    * Data & Detection — bead-stack ROI/frame/channel picker, an initial
      PSF (matched-filter template + NLCG starting point), point-source
      detection knobs, and the fixed bead-comb "object model".
    * Solver — optional regularizer, NLCG solver knobs, and output options.
      There is no forward-model zoom or tiling here: the swapped operator
      has no super-resolution factor, and the reconstruction domain (PSF
      support) is always small enough to solve in one shot.

The dialog reads its widgets into a :class:`~.psf_distillation_nlcg.PSFDistillDialogState`,
builds a :class:`~.psf_distillation_nlcg.PreparedDistillInputs`, and hands the result to
:class:`~.psf_distillation_worker.PSFDistillationWorker`.
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

from pyvistra.ui.manager import manager
from pyvistra.data.points import PointTable
from pyvistra.data.shapes import RECTANGLE, rectangle_bounds

from .deconvolution_dialog import ScientificDoubleSpinBox
from .output_selector import ImageOutputSelector
from .processing_helper import BufferProcessingRunner
from . import psf_distillation_nlcg as dpd
from .psf_distillation_worker import PSFDistillationWorker

# Vispy renders pixel index i spanning data coordinates [i, i+1) -- its
# visual center is i + 0.5. Bead positions from detection are plain array
# indices, so this offset is added when placing points (and subtracted
# back symmetrically when reading them for distillation) so the cross
# marker lands on the pixel's center, not its corner.
_PIXEL_CENTER = 0.5


class _CroppedSource:
    """Bead-stack crop + initial PSF, shared by "Find Beads" and "Distill".

    Both steps must operate on byte-identical data, so this bundles the
    single cropping/PSF-lookup implementation (:meth:`PSFDistillationDialog._crop_source`)
    both call, instead of duplicating it.
    """

    __slots__ = (
        "y_obs", "psf_data", "pixel_size_um", "t", "c",
        "z_off", "y_off", "x_off", "psf_window",
    )

    def __init__(self, *, y_obs, psf_data, pixel_size_um, t, c,
                 z_off, y_off, x_off, psf_window):
        self.y_obs = y_obs
        self.psf_data = psf_data
        self.pixel_size_um = pixel_size_um
        self.t = t
        self.c = c
        self.z_off = z_off      # None when y_obs is 2D (single Z plane)
        self.y_off = y_off
        self.x_off = x_off
        self.psf_window = psf_window


class PSFDistillationDialog(QDialog):
    """Distill a PSF from a sparse bead stack via the swapped-operator NLCG solve."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("PSF Distillation (NLCG)")
        self.setWindowFlags(Qt.Tool)
        self.resize(560, 700)
        self.setMinimumSize(520, 440)

        self._runner = None
        self._rect_shapes: list = []
        self._last_prepared = None
        self._bead_layer_name = "Detected Beads"
        self._bead_signature = None
        self._bead_light_signature = None
        self._running_status_base: Optional[str] = None
        self._run_started_at: Optional[float] = None
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_running_status)

        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 8)
        main.setSpacing(6)

        tabs = QTabWidget()
        tabs.addTab(self._build_data_detection_tab(), "Data && Detection")
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
            default_title="Distilled PSF",
            formats=[".tif", ".psf.h5"],
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

        self._on_regularizer_mode_changed()
        self._refresh_compute_target_hint()
        self._refresh_recipe_preview()

    # ------------------------------------------------------------------ #
    # Tab builders

    def _build_data_detection_tab(self) -> QWidget:
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

        T, _Z, C, _Y, _X = self.viewer.img_data.shape

        cf_grp = QGroupBox("Bead stack")
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
        ch_t_layout.addSpacing(10)
        ch_t_layout.addWidget(QLabel("Background:"))
        self.background_spin = QDoubleSpinBox()
        self.background_spin.setRange(0.0, 1e6)
        self.background_spin.setDecimals(2)
        self.background_spin.setSingleStep(1.0)
        self.background_spin.setValue(0.0)
        self.background_spin.setFixedWidth(72)
        self.background_spin.setToolTip(
            "Camera baseline for the selected channel — beads are sparse, "
            "so the whole-field median is usually a robust estimate."
        )
        ch_t_layout.addWidget(self.background_spin)
        ch_t_layout.addStretch()
        cf_form.addRow("Channel:", ch_t)

        # Region picker (full field / ROI, plus an optional Z window).
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

        # Initial PSF -- doubles as the matched-filter template and the
        # NLCG starting point. Must be sampled at the bead stack's own
        # pixel spacing (this problem has no forward-model zoom).
        psf_grp = QGroupBox("Initial PSF")
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
            "Compares the initial PSF's sampling with the bead stack's own "
            "pixel spacing (must match — there is no zoom factor here)."
        )
        psf_form.addRow("Sampling:", self.scale_label)

        compute_row = QWidget()
        compute_layout = QHBoxLayout(compute_row)
        compute_layout.setContentsMargins(0, 0, 0, 0)
        compute_layout.setSpacing(6)
        self.compute_psf_btn = QPushButton("Compute PSF…")
        self.compute_psf_btn.setToolTip(
            "Open the Compute PSF dialog with shape and spacing pre-filled "
            "for the bead stack's own sampling."
        )
        self.compute_psf_btn.clicked.connect(self._spawn_psf_dialog)
        compute_layout.addWidget(self.compute_psf_btn)
        self.compute_target_label = QLabel("")
        self.compute_target_label.setStyleSheet("color: #888; font-size: 10px;")
        compute_layout.addWidget(self.compute_target_label, 1)
        psf_form.addRow("", compute_row)

        vbox.addWidget(psf_grp)

        # Point-source detection -- matched filter against the initial PSF.
        det_grp = QGroupBox("Point-source detection")
        det_form = QFormLayout(det_grp)
        det_form.setLabelAlignment(Qt.AlignRight)
        det_form.setVerticalSpacing(4)
        det_form.setContentsMargins(8, 6, 8, 6)

        det_row = QWidget()
        det_layout = QHBoxLayout(det_row)
        det_layout.setContentsMargins(0, 0, 0, 0)
        det_layout.setSpacing(6)
        det_layout.addWidget(QLabel("Min sep (px):"))
        self.min_sep_spin = QSpinBox()
        self.min_sep_spin.setRange(1, 1000)
        self.min_sep_spin.setValue(15)
        self.min_sep_spin.setFixedWidth(64)
        self.min_sep_spin.setToolTip(
            "Minimum lateral pixel distance between detected beads."
        )
        det_layout.addWidget(self.min_sep_spin)
        det_layout.addSpacing(10)
        det_layout.addWidget(QLabel("Min intensity:"))
        self.min_int_spin = QDoubleSpinBox()
        self.min_int_spin.setRange(0.0, 1e9)
        self.min_int_spin.setValue(100.0)
        self.min_int_spin.setDecimals(1)
        self.min_int_spin.setFixedWidth(88)
        self.min_int_spin.setToolTip(
            "Minimum background-subtracted intensity at a detected peak."
        )
        det_layout.addWidget(self.min_int_spin)
        det_layout.addStretch()
        det_form.addRow("", det_row)

        find_row = QWidget()
        find_layout = QHBoxLayout(find_row)
        find_layout.setContentsMargins(0, 0, 0, 0)
        find_layout.setSpacing(6)
        self.find_beads_btn = QPushButton("Find Beads")
        self._find_beads_tooltip = (
            "Run point-source detection and mark hits as a points layer on "
            "the bead-stack window -- review, add, move, or delete points "
            "there (point tool: Alt-click to delete, drag to move) before "
            "clicking Start. Distillation always uses the points layer's "
            "current contents, not a fresh blind detection."
        )
        self.find_beads_btn.setToolTip(self._find_beads_tooltip)
        self.find_beads_btn.setEnabled(False)
        self.find_beads_btn.clicked.connect(self._find_beads)
        find_layout.addWidget(self.find_beads_btn)
        self.bead_count_label = QLabel("No beads found yet.")
        self.bead_count_label.setStyleSheet("color: #888; font-size: 10px;")
        find_layout.addWidget(self.bead_count_label, 1)
        det_form.addRow("", find_row)
        vbox.addWidget(det_grp)

        # Object model -- the fixed bead-comb kernel. Folding the bead's
        # physical extent + pixel integration in here lets NLCG recover
        # the pure optical PSF (measurement blur divided out).
        obj_grp = QGroupBox("Object model (bead comb)")
        obj_form = QFormLayout(obj_grp)
        obj_form.setLabelAlignment(Qt.AlignRight)
        obj_form.setVerticalSpacing(4)
        obj_form.setContentsMargins(8, 6, 8, 6)

        obj_row = QWidget()
        obj_layout = QHBoxLayout(obj_row)
        obj_layout.setContentsMargins(0, 0, 0, 0)
        obj_layout.setSpacing(6)
        obj_layout.addWidget(QLabel("Bead diameter (µm):"))
        self.bead_diameter_spin = QDoubleSpinBox()
        self.bead_diameter_spin.setRange(0.0, 10.0)
        self.bead_diameter_spin.setDecimals(3)
        self.bead_diameter_spin.setSingleStep(0.01)
        self.bead_diameter_spin.setValue(0.0)
        self.bead_diameter_spin.setFixedWidth(80)
        self.bead_diameter_spin.setToolTip(
            "Fluorescent bead diameter. 0 treats beads as point sources; "
            "> 0 folds the bead's finite extent into the fixed kernel so "
            "the recovered PSF is the pure optical PSF."
        )
        obj_layout.addWidget(self.bead_diameter_spin)
        obj_layout.addSpacing(10)
        self.pixel_integration_check = QCheckBox("Pixel integration")
        self.pixel_integration_check.setChecked(True)
        self.pixel_integration_check.setToolTip(
            "Fold the detector's lateral pixel footprint (sinc) into the "
            "fixed kernel as well."
        )
        obj_layout.addWidget(self.pixel_integration_check)
        obj_layout.addStretch()
        obj_form.addRow("", obj_row)
        vbox.addWidget(obj_grp)

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

        # Regularization (optional; off by default).
        reg_grp = QGroupBox("Regularization (optional)")
        reg_grp.setToolTip(
            "Early stopping alone is usually enough — leave this off unless "
            "you see noise amplification in the recovered PSF."
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
        self.reg_weight_spin.setRange(0.0, 1e9)
        self.reg_weight_spin.setValue(0.0)
        self.reg_weight_spin.setFixedWidth(96)
        self.reg_weight_spin.setEnabled(False)
        self.reg_weight_spin.setToolTip(
            "Regularization weight (beta). Unlike the deconvolution dialog, "
            "the reconstruction here is the PSF itself (unit-flux, values "
            "~0.01-0.05) rather than photon counts in the thousands, so this "
            "weight needs to be much larger to compete with the data term — "
            "start around 1e3-1e4 and increase toward 1e5-1e6 (scales with "
            "total bead brightness); 1e-5-1 has no visible effect."
        )
        reg_layout.addWidget(self.reg_weight_spin)
        reg_layout.addStretch()
        self._reg_group = QButtonGroup(self)
        self._reg_group.addButton(self.reg_none_radio)
        self._reg_group.addButton(self.reg_gradient_radio)
        self._reg_group.addButton(self.reg_hessian_radio)
        self._reg_group.buttonToggled.connect(self._on_regularizer_mode_changed)
        vbox.addWidget(reg_grp)

        self._nlcg_box = self._build_nlcg_knobs()
        vbox.addWidget(self._nlcg_box)
        self._nlcg_advanced_box = self._build_nlcg_advanced_knobs()
        vbox.addWidget(self._nlcg_advanced_box)

        out_grp = QGroupBox("Output")
        out_v = QVBoxLayout(out_grp)
        out_v.setContentsMargins(8, 6, 8, 6)
        out_v.setSpacing(6)
        self.center_output_check = QCheckBox("Center output (ifftshift)")
        self.center_output_check.setChecked(True)
        self.center_output_check.setToolTip(
            "Shift the PSF peak to the array centre for display/inspection. "
            "Uncheck to keep the FFT-native DC-at-corner convention (as "
            "the deconvolution dialog's PSF combo expects)."
        )
        self.center_output_check.toggled.connect(
            lambda *_: self._refresh_workflow_summary()
        )
        out_v.addWidget(self.center_output_check)
        vbox.addWidget(out_grp)

        preview_grp = QGroupBox("Recipe preview")
        preview_layout = QVBoxLayout(preview_grp)
        preview_layout.setContentsMargins(8, 6, 8, 6)
        preview_layout.setSpacing(4)
        self.recipe_preview = QPlainTextEdit()
        self.recipe_preview.setReadOnly(True)
        self.recipe_preview.setFixedHeight(60)
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
        self.psf_kernel_label = QLabel("—")
        self.object_domain_label = QLabel("—")
        diag_form.addRow("Detector field:", self.detector_domain_label)
        diag_form.addRow("PSF/kernel shape:", self.psf_kernel_label)
        diag_form.addRow("Padded/object shape:", self.object_domain_label)
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
        self.slack_spin = _dspin(0.0, 10.0, 0.0, dec=3, step=0.05)
        self.slack_spin.setToolTip(
            "Discrepancy-principle target multiplier (unregularized only). "
            "With sparse beads, background-only voxels dominate and the "
            "whole-volume I-divergence is trivially well-fit from iteration "
            "0 regardless of PSF quality — leave at 0 to rely on the "
            "Eq. 17 relative-iterate-change test instead."
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

        self.eval_interval_spin = _ispin(1, 500, 5)
        self.tol_spin = ScientificDoubleSpinBox()
        self.tol_spin.setRange(0.0, 1.0)
        self.tol_spin.setValue(1e-4)
        self.tol_spin.setFixedWidth(80)
        self.tol_spin.setToolTip(
            "Relative iterate-to-iterate change (Eq. 17). Stops early once "
            "it drops below this. 0 disables it."
        )
        self.min_iter_spin = _ispin(0, 500, 20)
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

    def _toggle_group_children(self, group: QGroupBox, checked: bool) -> None:
        for child in group.findChildren(QWidget):
            if child is not group:
                child.setVisible(bool(checked))
        group.setMaximumHeight(16777215 if checked else 24)

    def _connect_recipe_preview_signals(self) -> None:
        for widget in (self.reg_none_radio, self.reg_gradient_radio, self.reg_hessian_radio):
            widget.toggled.connect(lambda *_: self._refresh_workflow_summary())
        for spin in (
            self.num_iter_spin, self.background_spin, self.slack_spin,
            self.reg_weight_spin, self.eval_interval_spin, self.tol_spin,
            self.min_iter_spin, self.restart_interval_spin, self.newton_iters_spin,
            self.min_sep_spin, self.min_int_spin, self.bead_diameter_spin,
        ):
            spin.valueChanged.connect(lambda *_: self._refresh_workflow_summary())
        self.pixel_integration_check.toggled.connect(
            lambda *_: self._refresh_workflow_summary()
        )

    def _refresh_workflow_summary(self) -> None:
        if hasattr(self, "compute_target_label"):
            self._refresh_compute_target_hint()
        if hasattr(self, "recipe_preview"):
            self._refresh_recipe_preview()
        self._refresh_bead_status_label()

    def _current_light_signature(self):
        """Cheap, widget-only proxy for :meth:`_current_crop_signature`.

        Used only to flag the bead-count label as stale on any relevant
        widget change, without re-slicing image data reactively on every
        keystroke. The authoritative (data-based) check still happens in
        :meth:`_prepare` via :meth:`_current_crop_signature`.
        """
        if self.full_radio.isChecked():
            region_key = ("full",)
        else:
            region_key = (
                "roi", self.shape_combo.currentIndex(),
                self.around_z_radio.isChecked(), self.z_half_spin.value(),
            )
        return (
            self.frame_spin.value(), self.channel_spin.value(), region_key,
            id(self.psf_combo.currentData()),
        )

    def _refresh_bead_status_label(self) -> None:
        if not hasattr(self, "bead_count_label") or self._bead_signature is None:
            return
        if self._current_light_signature() != self._bead_light_signature:
            self.bead_count_label.setText(
                "Channel/frame/region or PSF changed -- click 'Find Beads' again."
            )
            self.bead_count_label.setStyleSheet("color: #C84; font-size: 10px;")

    def _on_regularizer_mode_changed(self, *_):
        self.reg_weight_spin.setEnabled(not self.reg_none_radio.isChecked())
        self._refresh_workflow_summary()

    def _on_diagnostics_toggled(self, checked: bool) -> None:
        self._toggle_group_children(self._diagnostics_group, checked)

    def _current_data_shape(self) -> Optional[tuple]:
        _T, Z, _C, Y, X = self.viewer.img_data.shape
        if self.full_radio.isChecked():
            n_z, n_y, n_x = Z, Y, X
        else:
            idx = self.shape_combo.currentIndex()
            if idx < 0 or idx >= len(self._rect_shapes):
                return None
            _, rec = self._rect_shapes[idx]
            x0, y0, x1, y1 = rectangle_bounds(
                rec, self.viewer.img_data.shape[-2:]
            )
            n_y = y1 - y0
            n_x = x1 - x0
            if Z > 1 and self.around_z_radio.isChecked():
                half = self.z_half_spin.value()
                n_z = min(Z, rec.z + half + 1) - max(0, rec.z - half)
            else:
                n_z = Z
        if n_z <= 1:
            return (n_y, n_x)
        return (n_z, n_y, n_x)

    def _current_psf_kernel_shape(self) -> Optional[tuple]:
        win = self.psf_combo.currentData()
        if win is None:
            return None
        _T, Zp, _C, Yp, Xp = win.img_data.shape
        if self.viewer.img_data.shape[1] <= 1 or Zp <= 1:
            return (Yp, Xp)
        return (Zp, Yp, Xp)

    def _format_shape(self, shape) -> str:
        if shape is None:
            return "—"
        return " × ".join(str(int(v)) for v in shape)

    def _refresh_recipe_preview(self) -> None:
        if not hasattr(self, "recipe_preview"):
            return
        data_shape = self._current_data_shape()
        if data_shape is None:
            self.recipe_preview.setPlainText(
                "Select a bead-stack region to preview the recipe."
            )
            return
        kernel = self._current_psf_kernel_shape()
        padded = (
            dpd.padded_shape(data_shape, kernel)
            if kernel is not None and len(kernel) == len(data_shape)
            else None
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
            f"data {self._format_shape(data_shape)}, "
            f"PSF/kernel {self._format_shape(kernel)}, "
            f"padded {self._format_shape(padded)}",
        ]
        vsep = "on" if self.bead_diameter_spin.value() > 0 else "off"
        lines.append(
            f"detection min_sep={self.min_sep_spin.value()}px "
            f"min_int={self.min_int_spin.value():g}; "
            f"bead-extent kernel {vsep}, "
            f"pixel integration "
            f"{'on' if self.pixel_integration_check.isChecked() else 'off'}"
        )
        self.recipe_preview.setPlainText("\n".join(lines))

        self.detector_domain_label.setText(self._format_shape(data_shape))
        self.psf_kernel_label.setText(self._format_shape(kernel))
        self.object_domain_label.setText(self._format_shape(padded))

    # ------------------------------------------------------------------ #
    # PSF compute spawn

    def _expected_psf_shape_spacing(self):
        """Return (shape, spacing) for the PSF compute preset.

        No forward-model zoom in this problem — the initial PSF must be
        sampled at the bead stack's own pixel spacing.
        """
        data_shape = self._current_data_shape()
        if data_shape is None:
            return None, None
        scale = self.viewer.meta.get("scale", (1.0, 1.0, 1.0))
        if len(data_shape) == 2:
            shape = dpd.compact_psf_shape_for_data(data_shape, (1.0, 1.0))
            return shape, (float(scale[1]), float(scale[2]))
        shape = dpd.compact_psf_shape_for_data(data_shape, (1.0, 1.0, 1.0))
        return shape, (float(scale[0]), float(scale[1]), float(scale[2]))

    def _refresh_compute_target_hint(self):
        shape, spacing = self._expected_psf_shape_spacing()
        if shape is None:
            self.compute_target_label.setText("")
            return
        if len(shape) == 2:
            self.compute_target_label.setText(
                f"PSF support preset {shape[0]}×{shape[1]} px, "
                f"{spacing[0]*1000:.0f}×{spacing[1]*1000:.0f} nm"
            )
        else:
            self.compute_target_label.setText(
                f"PSF support preset {shape[0]}×{shape[1]}×{shape[2]} px, "
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

        seed = {}
        psf_win = self.psf_combo.currentData()
        if psf_win is not None:
            params = psf_win.meta.get("parameters", {})
            modality = psf_win.meta.get("modality", "widefield")
            seed["modality"] = modality
            wl = params.get("wavelength_em") if modality == "spinning_disk" else params.get("wavelength")
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
        """List every open window as a candidate initial PSF.

        Not restricted to windows flagged ``is_psf`` (e.g. computed via the
        Compute PSF dialog) -- a PSF loaded from disk into a plain window is
        just as valid an initial estimate/matched-filter template. Windows
        that *are* flagged are annotated so they're still easy to spot.
        """
        self.psf_combo.blockSignals(True)
        self.psf_combo.clear()
        for _wid, win in sorted(manager.get_all().items()):
            if win is self.viewer:
                continue
            label = win.windowTitle().split("]", 1)[-1].strip()
            if getattr(win, "is_psf", False):
                label = f"{label}  (PSF)"
            self.psf_combo.addItem(label, win)
        self.psf_combo.blockSignals(False)
        self._on_psf_selected(self.psf_combo.currentIndex())

    def _on_psf_selected(self, _idx):
        win = self.psf_combo.currentData()
        self._update_find_beads_enabled()
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

    def _update_find_beads_enabled(self) -> None:
        """Grey out "Find Beads" until an initial PSF is actually selected.

        The PSF picker sits above the detection section, which makes the
        dependency easy to miss -- disabling the button (with a tooltip
        explaining why) surfaces it up front instead of only on click via
        a status-bar error.
        """
        if not hasattr(self, "find_beads_btn"):
            return
        has_psf = self.psf_combo.currentData() is not None
        self.find_beads_btn.setEnabled(has_psf)
        self.find_beads_btn.setToolTip(
            self._find_beads_tooltip if has_psf else
            "Select or compute an initial PSF above first -- it's used as "
            "the matched-filter template for detection."
        )

    def _check_scale_match(self, psf_win):
        """Compare the initial PSF's spacing against the bead stack's own spacing.

        Unlike deconvolution, there is no zoom factor here — the initial
        PSF must be sampled at the same pixel spacing as the bead data.
        """
        src_scale = self.viewer.meta.get("scale", (1.0, 1.0, 1.0))
        psf_scale = psf_win.meta.get("scale", psf_win.meta.get("spacing"))
        if psf_scale is None:
            self.scale_label.setText("PSF has no scale metadata")
            self.scale_label.setStyleSheet("color: #C84; font-size: 10px;")
            return

        Z_dim = self.viewer.img_data.shape[1]
        if Z_dim <= 1:
            expected_cmp = list(src_scale)[-2:]
            psf_cmp = list(psf_scale)[-2:]
        else:
            expected_cmp = list(src_scale)[:3]
            psf_cmp = list(psf_scale)[:3]

        nm_e_vals = [round(float(s) * 1000) for s in expected_cmp]
        nm_p_vals = [round(float(s) * 1000) for s in psf_cmp]
        match = len(nm_e_vals) == len(nm_p_vals) and nm_e_vals == nm_p_vals
        nm_e = "×".join(str(v) for v in nm_e_vals)
        if match:
            self.scale_label.setText(f"✓  {nm_e} nm")
            self.scale_label.setStyleSheet("color: #4A4; font-size: 10px;")
        else:
            nm_p = "×".join(str(v) for v in nm_p_vals)
            self.scale_label.setText(f"✗  expected {nm_e} ≠ PSF {nm_p} nm")
            self.scale_label.setStyleSheet("color: #F44; font-size: 10px;")

    def _refresh_shape_combo(self):
        self.shape_combo.blockSignals(True)
        self.shape_combo.clear()
        self._rect_shapes = []
        for layer in self.viewer.layers.by_type("shapes"):
            for rec in (layer.data.get(sid) for sid in layer.data.shape_ids):
                if rec.shape_type == RECTANGLE:
                    self._rect_shapes.append((layer.name, rec))
                    xl, yt, xr, yb = rectangle_bounds(
                        rec, self.viewer.img_data.shape[-2:]
                    )
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
        xl, yt, xr, yb = rectangle_bounds(rec, self.viewer.img_data.shape[-2:])
        _T, Z, _C, _Y, _X = self.viewer.img_data.shape
        self.bounds_label.setText(
            f"y [{yt}:{yb}]  x [{xl}:{xr}]  →  {yb-yt} × {xr-xl} px"
        )
        self._z_row_widget.setVisible(Z > 1)
        self.frame_spin.setValue(rec.t)
        self._refresh_workflow_summary()

    # ------------------------------------------------------------------ #
    # State / inputs

    def _read_state(self) -> dpd.PSFDistillDialogState:
        reg_kind = (
            "gradient" if self.reg_gradient_radio.isChecked()
            else "hessian" if self.reg_hessian_radio.isChecked()
            else "none"
        )
        return dpd.PSFDistillDialogState(
            min_separation=self.min_sep_spin.value(),
            min_intensity=self.min_int_spin.value(),
            bead_diameter=self.bead_diameter_spin.value(),
            pixel_integration=self.pixel_integration_check.isChecked(),
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
            center_output=self.center_output_check.isChecked(),
        )

    def _crop_source(self) -> Optional[_CroppedSource]:
        """Crop the bead-stack observation and fetch the initial PSF.

        Shared by :meth:`_find_beads` and :meth:`_prepare` so both operate
        on byte-identical data -- detection and distillation must see the
        exact same crop or the reviewed point positions won't line up.
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
                z_slice = slice(max(0, rec.z - half), min(Z, rec.z + half + 1))
            else:
                z_slice = slice(0, Z)

        psf_win = self.psf_combo.currentData()
        if psf_win is None:
            self._set_status("No initial PSF window selected.", error=True)
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
        dpd.log.info(
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
        z_off = None if y_obs.shape[0] == 1 else z_slice.start
        if y_obs.shape[0] == 1:
            y_obs = y_obs[0]

        if psf_data.ndim != y_obs.ndim:
            self._set_status(
                f"Initial PSF is {psf_data.ndim}D but observation is "
                f"{y_obs.ndim}D.",
                error=True,
            )
            return None

        psf_spacing = psf_win.meta.get(
            "spacing", psf_win.meta.get("scale", (1.0, 1.0, 1.0))
        )
        if psf_data.ndim == 2:
            pixel_size_um = tuple(float(s) for s in psf_spacing[-2:])
        else:
            pixel_size_um = tuple(float(s) for s in psf_spacing[-3:])

        return _CroppedSource(
            y_obs=y_obs, psf_data=psf_data, pixel_size_um=pixel_size_um,
            t=t, c=c, z_off=z_off, y_off=y_slice.start, x_off=x_slice.start,
            psf_window=psf_win,
        )

    def _current_crop_signature(self, cropped: _CroppedSource):
        """Identify the exact crop + PSF a bead-find run was based on.

        Compared against the *current* crop at Distill time -- if the
        user changes channel/frame/region/initial-PSF after finding beads,
        the reviewed points no longer correspond to the data about to be
        fit, so distillation must be blocked until beads are found again.
        Detection thresholds (min sep/intensity) are deliberately excluded:
        they only affect a fresh "Find Beads" run, not the points already
        placed.
        """
        return (
            cropped.t, cropped.c, cropped.z_off, cropped.y_off, cropped.x_off,
            tuple(cropped.y_obs.shape), id(cropped.psf_window),
        )

    def _find_beads(self):
        """Detect beads and mark them as a reviewable points layer.

        Distillation (Start) never re-detects -- it reads back whatever is
        currently in this points layer, so the user can inspect, delete
        false positives, nudge positions, or add missed beads before
        committing to a run.
        """
        cropped = self._crop_source()
        if cropped is None:
            return
        state = self._read_state()

        try:
            result = dpd.detect_beads(
                state=state, y_obs=cropped.y_obs, init_psf_array=cropped.psf_data,
            )
        except Exception as exc:
            self._set_status(f"{type(exc).__name__}: {exc}", error=True)
            return

        positions = result.positions
        amplitudes = result.amplitudes
        ndim = positions.shape[1]
        # +0.5 so the cross marker lands on the pixel's visual center
        # (pixel index i spans data coordinates [i, i+1)) rather than its
        # corner. Undone symmetrically in _prepare() when reading points
        # back for distillation.
        if ndim == 2:
            y_full = positions[:, 0] + cropped.y_off + _PIXEL_CENTER
            x_full = positions[:, 1] + cropped.x_off + _PIXEL_CENTER
            z_full = None
        else:
            y_full = positions[:, 1] + cropped.y_off + _PIXEL_CENTER
            x_full = positions[:, 2] + cropped.x_off + _PIXEL_CENTER
            z_full = positions[:, 0] + (cropped.z_off or 0)

        table = PointTable.from_arrays(
            x=x_full, y=y_full, z=z_full,
            t=np.full(len(x_full), cropped.t, dtype=np.int32),
            properties={"amplitude": amplitudes},
        )
        try:
            if self._bead_layer_name not in self.viewer.layers:
                self.viewer.add_point_layer(self._bead_layer_name, points=table)
            else:
                self.viewer.set_point_layer_points(self._bead_layer_name, table)
            self.viewer.layers.set_active("points", self._bead_layer_name)
        except Exception as exc:
            dpd.log.exception("failed to publish bead points layer")
            self._set_status(
                f"Beads detected but the points layer failed: "
                f"{type(exc).__name__}: {exc}",
                error=True,
            )
            return

        self._bead_signature = self._current_crop_signature(cropped)
        self._bead_light_signature = self._current_light_signature()
        self.bead_count_label.setText(
            f"{len(x_full)} beads in '{self._bead_layer_name}' -- review, "
            "then click Start."
        )
        self.bead_count_label.setStyleSheet("color: #4A4; font-size: 10px;")
        self._set_status(
            f"Found {len(x_full)} beads -- review the points layer, then "
            "click Start.",
            ok=True,
        )

    def _prepare(self):
        """Crop the source and build the deconlib payload from reviewed beads.

        Returns ``(prepared, state, t, c)`` on success, or `None` on a
        user error (with the status label updated). Requires a prior
        "Find Beads" run against the exact same channel/frame/region/PSF
        selection -- see :meth:`_current_crop_signature`.
        """
        cropped = self._crop_source()
        if cropped is None:
            return None

        if self._bead_layer_name not in self.viewer.layers:
            self._set_status("Click 'Find Beads' first.", error=True)
            return None
        if self._current_crop_signature(cropped) != self._bead_signature:
            self._set_status(
                "Channel/frame/region or initial PSF changed since beads "
                "were found -- click 'Find Beads' again.",
                error=True,
            )
            return None

        table = self.viewer.layers[self._bead_layer_name].data.table
        if table.n_rows == 0:
            self._set_status(
                "No beads in the review layer -- find or add some first.",
                error=True,
            )
            return None

        if cropped.y_obs.ndim == 2:
            positions = np.column_stack([
                table.y - cropped.y_off - _PIXEL_CENTER,
                table.x - cropped.x_off - _PIXEL_CENTER,
            ])
        else:
            if table.z is None:
                self._set_status(
                    "Bead points are missing Z -- click 'Find Beads' again.",
                    error=True,
                )
                return None
            positions = np.column_stack([
                table.z - (cropped.z_off or 0),
                table.y - cropped.y_off - _PIXEL_CENTER,
                table.x - cropped.x_off - _PIXEL_CENTER,
            ])

        state = self._read_state()

        try:
            prepared = dpd.prepare_inputs(
                state=state,
                y_obs=cropped.y_obs,
                init_psf_array=cropped.psf_data,
                pixel_size_um=cropped.pixel_size_um,
                positions=positions,
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
        result = self._prepare()
        if result is None:
            return
        prepared, state, t, c = result
        self._last_prepared = prepared

        output_shape = dpd.output_5d_shape(prepared.init_psf.shape)

        vs = prepared.voxel_spacing
        if len(vs) == 2:
            out_scale = (1.0, float(vs[0]), float(vs[1]))
        else:
            out_scale = tuple(float(s) for s in vs)

        psf_win = self.psf_combo.currentData()
        parameters = dict(psf_win.meta.get("parameters", {})) if psf_win is not None else {}

        output_meta = {
            "filename": "Distilled PSF",
            "scale": out_scale,
            "spacing": out_scale,
            "is_rgb": False,
            "psf_dc_corner": not state.center_output,
            "psf_source": "distilled_nlcg",
            "parameters": parameters,
            "algorithm": "nlcg_psf_distillation",
            "source_channel": c,
            "source_frame": t,
            "n_beads": int(len(prepared.positions)),
        }

        source, buffer = self._runner.prepare_output(
            output_shape=output_shape,
            output_dtype=np.dtype(np.float32),
            output_meta=output_meta,
        )
        if self._runner.output_viewer is not None:
            self._runner.output_viewer.is_psf = True

        regularizer = dpd.build_regularizer(
            state, prepared.y.ndim, prepared.voxel_spacing
        )

        self.progress_bar.setRange(0, max(1, state.num_iter))

        worker = PSFDistillationWorker(
            prepared=prepared,
            state=state,
            regularizer=regularizer,
            buffer=buffer,
            output_channel=0,
        )

        worker.status.connect(self._on_status)
        self.progress_bar.setValue(0)
        self._begin_run_status(
            f"Running NLCG for up to {state.num_iter} iterations "
            f"({len(prepared.positions)} beads)."
        )
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
