import os
import traceback

import numpy as np
from qtpy.QtCore import QObject, Qt, QThread, QTimer, Signal
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .convergence_plot import ConvergencePlotWidget, get_convergence_comparison_dialog
from .output_selector import ImageOutputSelector
from .region_selector import RegionSelector

# Source modes for the PSF dialog. The order here matches the combo and
# QStackedWidget index, so adding a mode means appending in both places.
SRC_THEORETICAL = 0
SRC_DISTILLED = 1
SRC_PUPIL = 2

# Curated Zernike modes (OSA/ANSI index, label) exposed as sliders. Piston
# and the two tilts are skipped — they translate the PSF, not its shape.
# Plain ints are used as dict keys (ZernikeAberration accepts int or
# ZernikeMode) so this module never needs to import deconlib at load time.
_ZERNIKE_MODES = [
    (3, "Astig (oblique)"),
    (4, "Defocus"),
    (5, "Astig (vertical)"),
    (6, "Trefoil (Y)"),
    (7, "Coma (Y)"),
    (8, "Coma (X)"),
    (9, "Trefoil (X)"),
    (10, "Quadrafoil (Y)"),
    (11, "Astig-2 (oblique)"),
    (12, "Spherical"),
    (13, "Astig-2 (vertical)"),
    (14, "Quadrafoil (X)"),
]
_ZERNIKE_SLIDER_SCALE = 100.0  # slider is int radians*100, range [-3, 3] rad


class PSFComputeDialog(QDialog):
    """
    Dialog for computing Point Spread Functions (PSF) for widefield
    and spinning disk microscopy modalities using deconlib.
    """

    psf_computed = Signal(object)  # emits ImageBuffer

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compute PSF")
        self.setWindowFlags(Qt.Tool)
        self.resize(460, 700)
        self.setMinimumSize(420, 520)

        self._buffer = None  # Holds computed PSF as ImageBuffer
        self._metadata = None  # PSF parameters for saving

        self._preview_viewer = None  # Live-preview OrthoViewer, when open
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._refresh_preview)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Scroll area for form content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Source + Modality on a single compact row. The modality combo is
        # shown only when the source is "Theoretical" — toggled in
        # _on_source_changed. Connect AFTER items are added and AFTER the
        # spinning-disk group is constructed below — the handler touches
        # self.sd_group, which doesn't exist yet at this point.
        src_row = QHBoxLayout()
        src_row.setSpacing(6)
        src_row.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(
            ["Theoretical", "Distilled from beads", "From pupil file"]
        )
        src_row.addWidget(self.source_combo, 1)
        self.modality_label = QLabel("Modality:")
        src_row.addWidget(self.modality_label)
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["Widefield", "Spinning Disk"])
        self.modality_combo.currentIndexChanged.connect(self._on_modality_changed)
        src_row.addWidget(self.modality_combo, 1)
        layout.addLayout(src_row)

        # Source-specific pages. We swap them in/out via show/hide rather
        # than QStackedWidget so the dialog only reserves the height of the
        # active source's controls — theoretical mode has no controls at all
        # and shouldn't reserve space for the (much taller) distill page.

        # --- Page 0: Theoretical ---
        # Modality control lives on the top row above; the page is empty.
        # Kept as a placeholder so index → page lookup stays aligned with
        # SRC_THEORETICAL/SRC_DISTILLED/SRC_PUPIL.
        theory_page = QWidget()

        # --- Page 1: Distilled ---
        distill_page = QWidget()
        distill_layout = QVBoxLayout(distill_page)
        distill_layout.setContentsMargins(0, 0, 0, 0)
        distill_layout.setSpacing(6)

        bead_row = QHBoxLayout()
        bead_row.addWidget(QLabel("Bead stack:"))
        self.bead_window_combo = QComboBox()
        self.bead_window_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        bead_row.addWidget(self.bead_window_combo, 1)
        self.bead_refresh_btn = QPushButton("Refresh")
        self.bead_refresh_btn.clicked.connect(self._refresh_bead_windows)
        bead_row.addWidget(self.bead_refresh_btn)
        distill_layout.addLayout(bead_row)

        # --- Detection (bead-finding by matched filter) ---
        det_group = QGroupBox("Detection")
        det_layout = QVBoxLayout(det_group)
        det_layout.setContentsMargins(8, 6, 8, 6)
        det_layout.setSpacing(4)

        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Background:"))
        self.bead_bg_spin = QDoubleSpinBox()
        self.bead_bg_spin.setRange(0.0, 1e7)
        self.bead_bg_spin.setValue(100.0)
        self.bead_bg_spin.setDecimals(1)
        bg_row.addWidget(self.bead_bg_spin)
        bg_row.addWidget(QLabel("Min sep (px):"))
        self.bead_minsep_spin = QSpinBox()
        self.bead_minsep_spin.setRange(1, 1000)
        self.bead_minsep_spin.setValue(20)
        bg_row.addWidget(self.bead_minsep_spin)
        det_layout.addLayout(bg_row)

        int_row = QHBoxLayout()
        int_row.addWidget(QLabel("Min intensity:"))
        self.bead_minint_spin = QDoubleSpinBox()
        self.bead_minint_spin.setRange(0.0, 1e9)
        self.bead_minint_spin.setValue(500.0)
        self.bead_minint_spin.setDecimals(1)
        int_row.addWidget(self.bead_minint_spin)
        int_row.addStretch()
        det_layout.addLayout(int_row)

        distill_layout.addWidget(det_group)

        # --- Distillation iteration (Richardson-Lucy + amplitude solve) ---
        iter_group = QGroupBox("Distillation")
        iter_layout = QVBoxLayout(iter_group)
        iter_layout.setContentsMargins(8, 6, 8, 6)
        iter_layout.setSpacing(4)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Max outer iters:"))
        self.bead_iters_spin = QSpinBox()
        self.bead_iters_spin.setRange(1, 500)
        self.bead_iters_spin.setValue(40)
        iter_row.addWidget(self.bead_iters_spin)
        iter_row.addWidget(QLabel("RL steps / outer:"))
        self.bead_rl_spin = QSpinBox()
        self.bead_rl_spin.setRange(1, 100)
        self.bead_rl_spin.setValue(5)
        iter_row.addWidget(self.bead_rl_spin)
        iter_layout.addLayout(iter_row)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("PSF tolerance:"))
        self.bead_tol_spin = QDoubleSpinBox()
        self.bead_tol_spin.setRange(1e-6, 1.0)
        self.bead_tol_spin.setDecimals(6)
        self.bead_tol_spin.setSingleStep(1e-4)
        self.bead_tol_spin.setValue(1e-3)
        tol_row.addWidget(self.bead_tol_spin)
        tol_row.addStretch()
        iter_layout.addLayout(tol_row)

        self.bead_coverslip_check = QCheckBox(
            "Coverslip data (beads in a single focal plane)"
        )
        self.bead_coverslip_check.setToolTip(
            "Sets min_pad=(0, None, None) — halves the axial FFT size when "
            "all beads lie at the same Z."
        )
        iter_layout.addWidget(self.bead_coverslip_check)

        distill_layout.addWidget(iter_group)

        distill_note = QLabel(
            "Initial PSF is built from the Optical Parameters and Output Shape below. "
            "Support is unrestricted (all-ones); the OTF mask provides the band limit."
        )
        distill_note.setStyleSheet("color: #888; font-size: 10px;")
        distill_note.setWordWrap(True)
        distill_layout.addWidget(distill_note)

        # Optional region restriction (rect ROI + Z range).
        self.distill_region = RegionSelector(title="Restrict to region (optional)")
        distill_layout.addWidget(self.distill_region)
        self.bead_window_combo.currentIndexChanged.connect(
            self._on_bead_window_changed
        )

        # --- Page 2: From Pupil ---
        pupil_page = QWidget()
        pupil_layout = QVBoxLayout(pupil_page)
        pupil_layout.setContentsMargins(0, 0, 0, 0)
        pupil_layout.setSpacing(6)

        file_row = QHBoxLayout()
        self.pupil_path_label = QLabel("(no pupil loaded)")
        self.pupil_path_label.setStyleSheet("color: #888;")
        file_row.addWidget(self.pupil_path_label, 1)
        self.pupil_load_btn = QPushButton("Load .pupil.h5…")
        self.pupil_load_btn.clicked.connect(self._load_pupil_file)
        file_row.addWidget(self.pupil_load_btn)
        pupil_layout.addLayout(file_row)

        self.pupil_info_label = QLabel("")
        self.pupil_info_label.setStyleSheet("color: #888; font-size: 10px;")
        self.pupil_info_label.setWordWrap(True)
        pupil_layout.addWidget(self.pupil_info_label)

        pupil_note = QLabel(
            "Optical Parameters and lateral spacing/shape are taken from the loaded pupil. "
            "Set Nz and dz below to control the axial grid."
        )
        pupil_note.setStyleSheet("color: #888; font-size: 10px;")
        pupil_note.setWordWrap(True)
        pupil_layout.addWidget(pupil_note)

        self._loaded_pupil = None  # deconlib.Pupil instance when loaded

        self._source_pages = [theory_page, distill_page, pupil_page]
        for i, page in enumerate(self._source_pages):
            layout.addWidget(page)
            page.setVisible(i == SRC_THEORETICAL)

        # Optical Parameters — 2-column grid keeps the dialog compact on
        # laptop screens by packing two field/spinbox pairs per row.
        optical_group = QGroupBox("Optical Parameters")
        optical_grid = QGridLayout(optical_group)
        optical_grid.setContentsMargins(8, 6, 8, 6)
        optical_grid.setHorizontalSpacing(8)
        optical_grid.setVerticalSpacing(4)
        optical_grid.setColumnStretch(1, 1)
        optical_grid.setColumnStretch(3, 1)

        _SPIN_WIDTH = 88

        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(300, 1000)
        self.wavelength_spin.setValue(525.0)
        self.wavelength_spin.setDecimals(1)
        self.wavelength_spin.setMaximumWidth(_SPIN_WIDTH)

        self.na_spin = QDoubleSpinBox()
        self.na_spin.setRange(0.1, 2.0)
        self.na_spin.setValue(1.40)
        self.na_spin.setDecimals(2)
        self.na_spin.setSingleStep(0.01)
        self.na_spin.setMaximumWidth(_SPIN_WIDTH)

        self.ni_spin = QDoubleSpinBox()
        self.ni_spin.setRange(1.0, 2.0)
        self.ni_spin.setDecimals(3)
        self.ni_spin.setSingleStep(0.001)
        self.ni_spin.setValue(1.515)
        self.ni_spin.setMaximumWidth(_SPIN_WIDTH)

        self.ns_spin = QDoubleSpinBox()
        self.ns_spin.setRange(1.0, 2.0)
        self.ns_spin.setDecimals(3)
        self.ns_spin.setSingleStep(0.001)
        self.ns_spin.setValue(1.33)
        self.ns_spin.setMaximumWidth(_SPIN_WIDTH)

        def _grid_row(grid, r, la, wa, lb, wb):
            grid.addWidget(QLabel(la), r, 0, Qt.AlignRight)
            grid.addWidget(wa,         r, 1)
            grid.addWidget(QLabel(lb), r, 2, Qt.AlignRight)
            grid.addWidget(wb,         r, 3)

        _grid_row(optical_grid, 0, "λ (nm):", self.wavelength_spin,
                  "NA:", self.na_spin)
        _grid_row(optical_grid, 1, "n_immersion:", self.ni_spin,
                  "n_sample:", self.ns_spin)
        layout.addWidget(optical_group)

        # Spinning Disk Parameters — 2-column grid, conditionally visible
        self.sd_group = QGroupBox("Spinning Disk")
        sd_grid = QGridLayout(self.sd_group)
        sd_grid.setContentsMargins(8, 6, 8, 6)
        sd_grid.setHorizontalSpacing(8)
        sd_grid.setVerticalSpacing(4)
        sd_grid.setColumnStretch(1, 1)
        sd_grid.setColumnStretch(3, 1)

        self.wavelength_exc_spin = QDoubleSpinBox()
        self.wavelength_exc_spin.setRange(300, 1000)
        self.wavelength_exc_spin.setValue(488.0)
        self.wavelength_exc_spin.setDecimals(1)
        self.wavelength_exc_spin.setMaximumWidth(_SPIN_WIDTH)

        self.wavelength_em_spin = QDoubleSpinBox()
        self.wavelength_em_spin.setRange(300, 1000)
        self.wavelength_em_spin.setValue(525.0)
        self.wavelength_em_spin.setDecimals(1)
        self.wavelength_em_spin.setMaximumWidth(_SPIN_WIDTH)

        self.pinhole_spin = QDoubleSpinBox()
        self.pinhole_spin.setRange(1.0, 200.0)
        self.pinhole_spin.setValue(50.0)
        self.pinhole_spin.setDecimals(1)
        self.pinhole_spin.setMaximumWidth(_SPIN_WIDTH)

        self.magnification_spin = QDoubleSpinBox()
        self.magnification_spin.setRange(1.0, 200.0)
        self.magnification_spin.setValue(100.0)
        self.magnification_spin.setDecimals(1)
        self.magnification_spin.setMaximumWidth(_SPIN_WIDTH)

        self.disk_magnification_spin = QDoubleSpinBox()
        self.disk_magnification_spin.setRange(0.1, 10.0)
        self.disk_magnification_spin.setValue(1.0)
        self.disk_magnification_spin.setDecimals(2)
        self.disk_magnification_spin.setMaximumWidth(_SPIN_WIDTH)

        _grid_row(sd_grid, 0, "λ exc (nm):", self.wavelength_exc_spin,
                  "λ em (nm):", self.wavelength_em_spin)
        _grid_row(sd_grid, 1, "Pinhole (μm):", self.pinhole_spin,
                  "Magnification:", self.magnification_spin)
        sd_grid.addWidget(QLabel("Disk mag.:"), 2, 0, Qt.AlignRight)
        sd_grid.addWidget(self.disk_magnification_spin, 2, 1)

        layout.addWidget(self.sd_group)
        self.sd_group.setVisible(False)  # Hidden by default (Widefield)

        # Now that sd_group exists, connect the source-change handler.
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

        # Output Shape — single compact row, spinboxes capped to a sane width.
        self.shape_group = QGroupBox("Output Shape")
        shape_layout = QHBoxLayout(self.shape_group)
        shape_layout.setContentsMargins(8, 6, 8, 6)
        shape_layout.setSpacing(6)
        self.nz_spin = QSpinBox()
        self.nz_spin.setRange(1, 512)
        self.nz_spin.setValue(64)
        self.nz_spin.setMaximumWidth(_SPIN_WIDTH)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(1, 2048)
        self.ny_spin.setValue(128)
        self.ny_spin.setMaximumWidth(_SPIN_WIDTH)
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(1, 2048)
        self.nx_spin.setValue(128)
        self.nx_spin.setMaximumWidth(_SPIN_WIDTH)
        for lab, sp in (("Nz:", self.nz_spin),
                        ("Ny:", self.ny_spin),
                        ("Nx:", self.nx_spin)):
            shape_layout.addWidget(QLabel(lab))
            shape_layout.addWidget(sp)
        shape_layout.addStretch()
        layout.addWidget(self.shape_group)

        # In distillation mode the output shape is derived from the
        # (region-cropped) bead stack — never user-set. The region selector
        # broadcasts region_changed; mirror that into the spinners.
        self.distill_region.region_changed.connect(self._sync_distill_output_shape)

        # Voxel Spacing — single compact row, matching Output Shape layout.
        spacing_group = QGroupBox("Voxel Spacing (μm)")
        spacing_layout = QHBoxLayout(spacing_group)
        spacing_layout.setContentsMargins(8, 6, 8, 6)
        spacing_layout.setSpacing(6)
        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.001, 10.0)
        self.dz_spin.setDecimals(3)
        self.dz_spin.setSingleStep(0.01)
        self.dz_spin.setValue(0.200)
        self.dz_spin.setMaximumWidth(_SPIN_WIDTH)
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.001, 10.0)
        self.dy_spin.setDecimals(3)
        self.dy_spin.setSingleStep(0.001)
        self.dy_spin.setValue(0.065)
        self.dy_spin.setMaximumWidth(_SPIN_WIDTH)
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.001, 10.0)
        self.dx_spin.setDecimals(3)
        self.dx_spin.setSingleStep(0.001)
        self.dx_spin.setValue(0.065)
        self.dx_spin.setMaximumWidth(_SPIN_WIDTH)
        for lab, sp in (("dz:", self.dz_spin),
                        ("dy:", self.dy_spin),
                        ("dx:", self.dx_spin)):
            spacing_layout.addWidget(QLabel(lab))
            spacing_layout.addWidget(sp)
        spacing_layout.addStretch()
        layout.addWidget(spacing_group)

        # Options + Aberrations — single compact row each.
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout(options_group)
        options_layout.setContentsMargins(8, 6, 8, 6)
        options_layout.setSpacing(8)
        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.setChecked(True)
        self.vectorial_check = QCheckBox("Vectorial")
        self.vectorial_check.setChecked(False)
        self.center_check = QCheckBox("Center output (ifftshift)")
        self.center_check.setChecked(True)
        options_layout.addWidget(self.normalize_check)
        options_layout.addWidget(self.vectorial_check)
        options_layout.addWidget(self.center_check)
        options_layout.addStretch()
        layout.addWidget(options_group)

        aberr_group = QGroupBox("Aberrations")
        aberr_layout = QHBoxLayout(aberr_group)
        aberr_layout.setContentsMargins(8, 6, 8, 6)
        aberr_layout.setSpacing(6)
        self.aberr_check = QCheckBox("Index mismatch")
        self.aberr_check.setChecked(False)
        aberr_layout.addWidget(self.aberr_check)
        aberr_layout.addSpacing(8)
        aberr_layout.addWidget(QLabel("Emitter depth (μm):"))
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(-100.0, 100.0)
        self.depth_spin.setValue(-1.0)
        self.depth_spin.setDecimals(2)
        self.depth_spin.setSingleStep(0.1)
        self.depth_spin.setMaximumWidth(_SPIN_WIDTH)
        aberr_layout.addWidget(self.depth_spin)
        depth_note = QLabel("(negative = into sample)")
        depth_note.setStyleSheet("color: #888; font-size: 10px;")
        aberr_layout.addWidget(depth_note)
        aberr_layout.addStretch()
        layout.addWidget(aberr_group)

        # Zernike Aberrations — interactive coefficients + live preview.
        # Applies only to the Theoretical source (SRC_THEORETICAL), where
        # deconlib's compute_widefield_psf/compute_spinning_disk_psf accept
        # a `zernike={osa_index: coeff}` kwarg directly.
        zernike_group = QGroupBox("Zernike Aberrations")
        zernike_layout = QVBoxLayout(zernike_group)
        zernike_layout.setContentsMargins(8, 6, 8, 6)
        zernike_layout.setSpacing(4)

        zernike_header = QHBoxLayout()
        self.zernike_check = QCheckBox("Enable")
        zernike_header.addWidget(self.zernike_check)
        self.zernike_reset_btn = QPushButton("Reset")
        self.zernike_reset_btn.clicked.connect(self._reset_zernike_coeffs)
        zernike_header.addWidget(self.zernike_reset_btn)
        zernike_header.addStretch()
        self.zernike_preview_btn = QPushButton("Live Preview")
        self.zernike_preview_btn.clicked.connect(self._toggle_preview)
        zernike_header.addWidget(self.zernike_preview_btn)
        zernike_layout.addLayout(zernike_header)

        # Sliders live in their own widget (not just a sub-layout) so they
        # can be shown/hidden as a block, keyed off "Enable" — collapsed by
        # default since this is the single tallest section in the dialog
        # (12 rows) and irrelevant until aberrations are turned on.
        self._zernike_sliders_widget = QWidget()
        zernike_grid = QGridLayout(self._zernike_sliders_widget)
        zernike_grid.setContentsMargins(0, 0, 0, 0)
        zernike_grid.setHorizontalSpacing(6)
        zernike_grid.setVerticalSpacing(2)
        zernike_grid.setColumnStretch(1, 1)
        self._zernike_sliders = {}
        self._zernike_spins = {}
        for row, (idx, label) in enumerate(_ZERNIKE_MODES):
            zernike_grid.addWidget(QLabel(f"{label}:"), row, 0, Qt.AlignRight)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(-300, 300)
            slider.setValue(0)
            zernike_grid.addWidget(slider, row, 1)

            spin = QDoubleSpinBox()
            spin.setRange(-3.0, 3.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(0.0)
            spin.setMaximumWidth(_SPIN_WIDTH)
            zernike_grid.addWidget(spin, row, 2)

            slider.valueChanged.connect(
                lambda v, s=spin: self._sync_zernike_spin_from_slider(s, v)
            )
            spin.valueChanged.connect(
                lambda v, sl=slider: self._sync_zernike_slider_from_spin(sl, v)
            )
            self._zernike_sliders[idx] = slider
            self._zernike_spins[idx] = spin
        zernike_layout.addWidget(self._zernike_sliders_widget)
        layout.addWidget(zernike_group)
        self._zernike_sliders_widget.setVisible(self.zernike_check.isChecked())
        self.zernike_check.toggled.connect(self._zernike_sliders_widget.setVisible)

        # Any control that affects the theoretical PSF schedules a debounced
        # live-preview recompute (no-op when no preview window is open).
        for spin in (
            self.wavelength_spin, self.na_spin, self.ni_spin, self.ns_spin,
            self.nz_spin, self.ny_spin, self.nx_spin,
            self.dz_spin, self.dy_spin, self.dx_spin,
            self.wavelength_exc_spin, self.wavelength_em_spin,
            self.pinhole_spin, self.magnification_spin,
            self.disk_magnification_spin, self.depth_spin,
        ):
            spin.valueChanged.connect(self._schedule_preview_refresh)
        self.modality_combo.currentIndexChanged.connect(self._schedule_preview_refresh)
        for check in (
            self.normalize_check, self.vectorial_check, self.center_check,
            self.aberr_check, self.zernike_check,
        ):
            check.toggled.connect(self._schedule_preview_refresh)

        # Output Selector
        self.output_selector = ImageOutputSelector(
            default_title="Computed PSF",
            formats=[".tif", ".psf.h5"],
        )
        layout.addWidget(self.output_selector)

        # Finalize scroll area
        layout.addStretch()
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, stretch=1)

        # Compute Button (pinned at bottom, outside scroll area)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._compute_psf)
        btn_layout.addWidget(self.compute_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # Progress bar — shown only during async distillation.
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Embedded convergence plot — visible only during/after distillation.
        self.convergence_plot = ConvergencePlotWidget()
        self.convergence_plot.set_labels(
            x_label="Outer iteration",
            y_label="error",
            title="χ² + ‖Δh‖ + ‖Δa‖",
        )
        self.convergence_plot.setMinimumHeight(140)
        self.convergence_plot.setVisible(False)
        main_layout.addWidget(self.convergence_plot)

        conv_btn_row = QHBoxLayout()
        self.send_compare_btn = QPushButton("Send to comparison…")
        self.send_compare_btn.setToolTip(
            "Push χ²/‖Δh‖/‖Δa‖ histories to the shared comparison window."
        )
        self.send_compare_btn.clicked.connect(self._send_to_comparison)
        self.send_compare_btn.setEnabled(False)
        self.send_compare_btn.setVisible(False)
        conv_btn_row.addWidget(self.send_compare_btn)
        conv_btn_row.addStretch()
        main_layout.addLayout(conv_btn_row)

        # Distillation convergence history (filled by _on_distill_progress).
        self._chi2_hist = []
        self._dpsf_hist = []
        self._damp_hist = []

        # Status Label (pinned at bottom, outside scroll area)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")
        main_layout.addWidget(self.status_label)

        # Threading state for the distillation worker (lazy).
        self._distill_thread = None
        self._distill_worker = None

    def preset(self, *, shape=None, spacing=None, wavelength_um=None,
               na=None, ni=None, ns=None, modality=None):
        """Pre-fill output shape, spacing and optics from external context.

        Used by the deconvolution dialog to spawn a PSF compute dialog
        already sampled on the recipe's fine/object grid. The PSF shape
        is kernel support, not the deconvolution object canvas; any
        keyword left as None is untouched.

        `shape` is `(Nz, Ny, Nx)` (3D) or `(Ny, Nx)` (2D).
        `spacing` is `(dz, dy, dx)` (3D) or `(dy, dx)` (2D), in μm.
        """
        if shape is not None:
            if len(shape) == 2:
                Nz, Ny, Nx = 1, int(shape[0]), int(shape[1])
            else:
                Nz, Ny, Nx = int(shape[0]), int(shape[1]), int(shape[2])
            self.nz_spin.setValue(Nz)
            self.ny_spin.setValue(Ny)
            self.nx_spin.setValue(Nx)
        if spacing is not None:
            if len(spacing) == 2:
                dz = self.dz_spin.value()
                dy, dx = float(spacing[0]), float(spacing[1])
            else:
                dz, dy, dx = (float(spacing[0]), float(spacing[1]),
                              float(spacing[2]))
                self.dz_spin.setValue(dz)
            self.dy_spin.setValue(dy)
            self.dx_spin.setValue(dx)
        if wavelength_um is not None:
            self.wavelength_spin.setValue(float(wavelength_um) * 1000.0)
            self.wavelength_em_spin.setValue(float(wavelength_um) * 1000.0)
        if na is not None:
            self.na_spin.setValue(float(na))
        if ni is not None:
            self.ni_spin.setValue(float(ni))
        if ns is not None:
            self.ns_spin.setValue(float(ns))
        if modality is not None:
            idx = 0 if modality == "widefield" else 1
            self.modality_combo.setCurrentIndex(idx)

    def _on_modality_changed(self, index):
        """Show/hide spinning disk parameters based on selection."""
        is_spinning_disk = index == 1
        self.sd_group.setVisible(is_spinning_disk)

    def _sync_zernike_spin_from_slider(self, spin, value):
        spin.blockSignals(True)
        spin.setValue(value / _ZERNIKE_SLIDER_SCALE)
        spin.blockSignals(False)
        self._schedule_preview_refresh()

    def _sync_zernike_slider_from_spin(self, slider, value):
        slider.blockSignals(True)
        slider.setValue(int(round(value * _ZERNIKE_SLIDER_SCALE)))
        slider.blockSignals(False)
        self._schedule_preview_refresh()

    def _reset_zernike_coeffs(self):
        for idx, spin in self._zernike_spins.items():
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
            slider = self._zernike_sliders[idx]
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self._schedule_preview_refresh()

    def _zernike_coeffs(self):
        """Current {osa_index: coefficient_radians} for every slider row."""
        return {idx: spin.value() for idx, spin in self._zernike_spins.items()}

    def _schedule_preview_refresh(self, *_args):
        if self._preview_viewer is not None:
            self._preview_timer.start()

    def _toggle_preview(self):
        """Open the live-preview OrthoViewer, or just raise the existing one."""
        if self._preview_viewer is not None:
            self._preview_viewer.raise_()
            self._preview_viewer.activateWindow()
            return
        self._open_preview()

    def _open_preview(self):
        if self.source_combo.currentIndex() != SRC_THEORETICAL:
            self.status_label.setText(
                "Live preview only supports the Theoretical source"
            )
            self.status_label.setStyleSheet("color: #F44;")
            return

        from ..ui.workspace import present_window
        from ..viewers.ortho import OrthoViewer

        try:
            psf, _ = self._compute_theoretical_psf()
        except Exception as exc:
            self.status_label.setText(f"Preview error: {exc}")
            self.status_label.setStyleSheet("color: #F44;")
            return

        psf_5d = psf[np.newaxis, :, np.newaxis, :, :]
        meta = {
            "scale": (
                self.dz_spin.value(), self.dy_spin.value(), self.dx_spin.value(),
            )
        }
        viewer = OrthoViewer(psf_5d, meta=meta, title="PSF Live Preview")
        viewer.destroyed.connect(lambda *_a, v=viewer: self._clear_preview_if(v))
        self._preview_viewer = viewer
        present_window(viewer, floating=True)

    def _clear_preview_if(self, viewer):
        if self._preview_viewer is viewer:
            self._preview_viewer = None

    def _refresh_preview(self):
        viewer = self._preview_viewer
        if viewer is None or self.source_combo.currentIndex() != SRC_THEORETICAL:
            return
        try:
            psf, _ = self._compute_theoretical_psf()
        except Exception as exc:
            self.status_label.setText(f"Preview error: {exc}")
            self.status_label.setStyleSheet("color: #F44;")
            return
        psf_5d = psf[np.newaxis, :, np.newaxis, :, :]
        if psf_5d.shape == viewer.data.shape:
            viewer.data[:] = psf_5d
            viewer.update_views()
        else:
            viewer.close()
            self._open_preview()

    def _on_source_changed(self, index):
        """Switch the source-specific parameter page."""
        for i, page in enumerate(self._source_pages):
            page.setVisible(i == index)
        is_theoretical = index == SRC_THEORETICAL
        # Modality / spinning-disk knobs only apply to the theoretical path.
        self.modality_label.setVisible(is_theoretical)
        self.modality_combo.setVisible(is_theoretical)
        if not is_theoretical:
            self.sd_group.setVisible(False)
        else:
            self._on_modality_changed(self.modality_combo.currentIndex())
        if index == SRC_DISTILLED:
            self._refresh_bead_windows()
            self._sync_distill_output_shape()
        else:
            # Other modes: lift the source-derived caps on the spinners.
            self._reset_output_shape_bounds()
        # Shrink the dialog to fit the now-active page. Qt won't auto-shrink
        # when a child widget hides, even though the layout reclaims space.
        self.adjustSize()

    def _refresh_bead_windows(self):
        """Populate the bead-stack combo with eligible open windows."""
        from ..ui.manager import manager

        self.bead_window_combo.clear()
        for wid, win in manager.get_all().items():
            if not hasattr(win, "img_data"):
                continue
            shape = getattr(win.img_data, "shape", None)
            if shape is None or len(shape) != 5:
                continue
            T, Z, C, Y, X = shape
            if T != 1 or C != 1 or Z < 2:
                continue
            label = f"[{wid}] {win.windowTitle()}  ({Z}×{Y}×{X})"
            self.bead_window_combo.addItem(label, userData=wid)
        self._on_bead_window_changed(self.bead_window_combo.currentIndex())

    def _on_bead_window_changed(self, _index):
        """Sync the region selector + pull spacing from the selected window."""
        from ..ui.manager import manager

        wid = self.bead_window_combo.currentData()
        source = manager.get(wid) if wid is not None else None
        self.distill_region.set_source(source)
        if source is not None:
            self._adopt_window_spacing(source)
        # set_source() emits region_changed which already fires the sync;
        # call explicitly when source is None so the spinners reset cleanly.
        if source is None:
            self._sync_distill_output_shape()

    _DEFAULT_OUTPUT_SHAPE_MAX = (512, 2048, 2048)  # (Nz, Ny, Nx) hard maxima

    def _reset_output_shape_bounds(self):
        """Restore the dialog-wide Output Shape ranges (for non-distill modes)."""
        self.shape_group.setTitle("Output Shape")
        for spin, hard_max in zip(
            (self.nz_spin, self.ny_spin, self.nx_spin),
            self._DEFAULT_OUTPUT_SHAPE_MAX,
        ):
            spin.blockSignals(True)
            spin.setMaximum(hard_max)
            spin.blockSignals(False)

    def _sync_distill_output_shape(self):
        """Cap Output Shape spinners at the (region-cropped) bead-stack shape.

        The PSF grid can be smaller than the bead image (compact support
        in any axis is fine — Z is typically the smallest), but it cannot
        exceed the source: there is no information to support a larger
        grid, and ``distill_psf`` requires ``psf_shape <= image_shape``.

        Behavior:
            - Each spinner's max is clamped to the effective source dim.
            - If the spinner's current value now exceeds that max, it is
              clamped down. Smaller user-chosen values are preserved.
        """
        if self.source_combo.currentIndex() != SRC_DISTILLED:
            return
        eff = self._effective_source_shape()
        if eff is None:
            self._reset_output_shape_bounds()
            return
        nz_max, ny_max, nx_max = eff

        self.shape_group.setTitle(
            f"Output Shape  (≤ source: {nz_max}×{ny_max}×{nx_max})"
        )
        for spin, hard_cap in zip(
            (self.nz_spin, self.ny_spin, self.nx_spin),
            (nz_max, ny_max, nx_max),
        ):
            spin.blockSignals(True)
            spin.setMaximum(int(hard_cap))
            if spin.value() > hard_cap:
                spin.setValue(int(hard_cap))
            spin.blockSignals(False)

    def _effective_source_shape(self):
        """Return ``(Nz, Ny, Nx)`` of the bead stack after the region crop, or None."""
        from ..ui.manager import manager

        wid = self.bead_window_combo.currentData()
        source = manager.get(wid) if wid is not None else None
        if source is None:
            return None
        shape = getattr(source.img_data, "shape", None)
        if shape is None or len(shape) != 5:
            return None
        _, Z, _, Y, X = shape

        zr = self.distill_region.z_range()
        bb = self.distill_region.bbox()
        nz = (zr[1] - zr[0]) if zr is not None else Z
        if bb is not None:
            y0, x0, y1, x1 = bb
            ny, nx = y1 - y0, x1 - x0
        else:
            ny, nx = Y, X
        return int(nz), int(ny), int(nx)

    def _adopt_window_spacing(self, window):
        """Mirror the window's voxel spacing (microns) into dz/dy/dx."""
        scale = getattr(window, "meta", {}).get("scale")
        if not scale or len(scale) < 3:
            return
        sz, sy, sx = (float(scale[0]), float(scale[1]), float(scale[2]))
        if sz > 0:
            self.dz_spin.setValue(sz)
        if sy > 0:
            self.dy_spin.setValue(sy)
        if sx > 0:
            self.dx_spin.setValue(sx)

    def _load_pupil_file(self):
        """Open file picker for .pupil.h5 and auto-fill from the loaded Pupil."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pupil", "", "Pupil files (*.pupil.h5);;All files (*)"
        )
        if not path:
            return
        try:
            from deconlib import load_pupil
            pupil = load_pupil(path)
        except Exception as exc:
            self.status_label.setText(f"Error loading pupil: {exc}")
            self.status_label.setStyleSheet("color: #F44;")
            return

        self._loaded_pupil = pupil
        self.pupil_path_label.setText(os.path.basename(path))
        self.pupil_path_label.setStyleSheet("")
        ny, nx = pupil.shape
        dy, dx = pupil.spacing
        self.pupil_info_label.setText(
            f"shape={ny}×{nx}  spacing=({dy:.4f}, {dx:.4f}) μm  "
            f"λ={pupil.optics.wavelength * 1000:.1f} nm  NA={pupil.optics.na}  "
            f"ni={pupil.optics.ni}  ns={pupil.optics.ns}  source={pupil.source}"
        )

        # Mirror pupil's lateral sampling + optics into the shared fields.
        self.wavelength_spin.setValue(pupil.optics.wavelength * 1000.0)
        self.na_spin.setValue(pupil.optics.na)
        self.ni_spin.setValue(pupil.optics.ni)
        self.ns_spin.setValue(pupil.optics.ns)
        self.ny_spin.setValue(ny)
        self.nx_spin.setValue(nx)
        self.dy_spin.setValue(dy)
        self.dx_spin.setValue(dx)

    def _validate_parameters(self):
        """Validate parameters before computation."""
        errors = []
        if self.na_spin.value() >= self.ni_spin.value():
            errors.append("NA must be < n_immersion")
        if self.nz_spin.value() < 1:
            errors.append("Nz must be >= 1")
        if self.ny_spin.value() < 1:
            errors.append("Ny must be >= 1")
        if self.nx_spin.value() < 1:
            errors.append("Nx must be >= 1")
        return errors

    def _compute_psf(self):
        """Compute the PSF based on current parameters and source mode.

        Theoretical and pupil paths are fast and run synchronously here.
        Distillation is iterative and offloaded to a QThread worker; the
        publish step happens in :meth:`_on_distill_finished`.
        """
        if self._distill_thread is not None:
            return  # already running

        errors = self._validate_parameters()
        if errors:
            self.status_label.setText(f"Error: {errors[0]}")
            self.status_label.setStyleSheet("color: #F44;")
            return

        source = self.source_combo.currentIndex()

        if source == SRC_DISTILLED:
            try:
                self._start_distill_async()
            except Exception as exc:
                self.status_label.setText(f"Error: {exc}")
                self.status_label.setStyleSheet("color: #F44;")
                self.compute_btn.setEnabled(True)
            return

        # Synchronous paths.
        self.status_label.setText("Computing...")
        self.status_label.setStyleSheet("color: #888;")
        self.compute_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            if source == SRC_THEORETICAL:
                psf, source_tag = self._compute_theoretical_psf()
            elif source == SRC_PUPIL:
                psf, source_tag = self._compute_pupil_psf()
            else:
                raise ValueError(f"Unknown source index: {source}")
            self._publish_psf(psf, source_tag)
        except ImportError:
            self.status_label.setText("Error: deconlib not installed")
            self.status_label.setStyleSheet("color: #F44;")
        except Exception as exc:
            self.status_label.setText(f"Error: {exc}")
            self.status_label.setStyleSheet("color: #F44;")
        finally:
            self.compute_btn.setEnabled(True)

    def _compute_theoretical_psf(self):
        """Run deconlib's analytical widefield/spinning-disk PSF computation."""
        from deconlib import fft_coords
        from deconlib.psf import compute_spinning_disk_psf, compute_widefield_psf
        from numpy.fft import ifftshift

        Nz = self.nz_spin.value()
        Ny = self.ny_spin.value()
        Nx = self.nx_spin.value()
        dz = self.dz_spin.value()
        dy = self.dy_spin.value()
        dx = self.dx_spin.value()

        na = self.na_spin.value()
        ni = self.ni_spin.value()
        ns = self.ns_spin.value()
        normalize = self.normalize_check.isChecked()
        vectorial = self.vectorial_check.isChecked()
        center_output = self.center_check.isChecked()

        z = fft_coords(Nz, dz)

        aberrations = None
        if self.aberr_check.isChecked():
            from deconlib.psf.aberrations import IndexMismatch
            aberrations = [IndexMismatch(self.depth_spin.value())]

        zernike = self._zernike_coeffs() if self.zernike_check.isChecked() else None

        modality = "widefield" if self.modality_combo.currentIndex() == 0 else "spinning_disk"

        if modality == "widefield":
            wavelength_um = self.wavelength_spin.value() / 1000.0
            psf = compute_widefield_psf(
                wavelength=wavelength_um,
                na=na, ni=ni, ns=ns,
                shape=(Ny, Nx), spacing=(dy, dx), z=z,
                normalize=normalize,
                aberrations=aberrations,
                zernike=zernike,
                vectorial=vectorial,
            )
        else:
            wavelength_exc_um = self.wavelength_exc_spin.value() / 1000.0
            wavelength_em_um = self.wavelength_em_spin.value() / 1000.0
            psf = compute_spinning_disk_psf(
                wavelength_exc=wavelength_exc_um,
                wavelength_em=wavelength_em_um,
                na=na, ni=ni, ns=ns,
                shape=(Ny, Nx), spacing=(dy, dx), z=z,
                pinhole_um=self.pinhole_spin.value(),
                magnification=self.magnification_spin.value(),
                disk_magnification=self.disk_magnification_spin.value(),
                normalize=normalize,
                aberrations=aberrations,
                zernike=zernike,
                vectorial=vectorial,
            )

        if center_output:
            psf = ifftshift(psf)
        return psf.astype(np.float32, copy=False), "theoretical"

    def _start_distill_async(self):
        """Validate, crop, and kick off the distillation worker thread.

        Returns immediately; the result is published from
        :meth:`_on_distill_finished` once the worker emits ``finished``.
        """
        from ..ui.manager import manager

        wid = self.bead_window_combo.currentData()
        if wid is None:
            raise ValueError("No bead stack selected — pick a Z-stack window")
        window = manager.get(wid)
        if window is None:
            raise ValueError("Selected window is no longer open")

        # Bead image as 3D (Z, Y, X); enforced by the picker filter. Crop
        # via the region selector before handing off to the worker.
        image_5d = np.asarray(window.img_data[:])
        image = image_5d[0, :, 0, :, :].astype(np.float32, copy=False)
        image = np.ascontiguousarray(self.distill_region.crop_zyx(image))

        # PSF grid is user-set. Each axis must be ≤ the (cropped) image
        # axis: distill_psf needs image_shape >= psf_shape.
        psf_shape = (
            int(self.nz_spin.value()),
            int(self.ny_spin.value()),
            int(self.nx_spin.value()),
        )
        for axis, (p, i) in enumerate(zip(psf_shape, image.shape)):
            if p > i:
                name = ("Nz", "Ny", "Nx")[axis]
                raise ValueError(
                    f"PSF {name}={p} exceeds (cropped) bead-stack dimension {i}. "
                    f"Reduce {name} or widen the region."
                )

        params = {
            "psf_shape": psf_shape,
            "dz": float(self.dz_spin.value()),
            "dy": float(self.dy_spin.value()),
            "dx": float(self.dx_spin.value()),
            "wavelength_um": self._distill_wavelength_um(),
            "na": float(self.na_spin.value()),
            "ni": float(self.ni_spin.value()),
            "ns": float(self.ns_spin.value()),
            "background": float(self.bead_bg_spin.value()),
            "min_separation": int(self.bead_minsep_spin.value()),
            "min_intensity": float(self.bead_minint_spin.value()),
            "max_outer": int(self.bead_iters_spin.value()),
            "rl_steps_per_outer": int(self.bead_rl_spin.value()),
            "psf_tol": float(self.bead_tol_spin.value()),
            "min_pad": (
                (0, None, None)
                if self.bead_coverslip_check.isChecked()
                else None
            ),
            "center": self.center_check.isChecked(),
        }

        # Spin up the worker thread.
        self.compute_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, params["max_outer"])
        self.progress_bar.setValue(0)
        self._chi2_hist = []
        self._dpsf_hist = []
        self._damp_hist = []
        self.convergence_plot.clear()
        self.convergence_plot.setVisible(True)
        self.send_compare_btn.setVisible(True)
        self.send_compare_btn.setEnabled(False)
        self.status_label.setStyleSheet("color: #888;")
        self.status_label.setText("Distilling — detecting beads...")

        self._distill_worker = _PSFDistillWorker(image, params)
        self._distill_thread = QThread(self)
        self._distill_worker.moveToThread(self._distill_thread)
        self._distill_thread.started.connect(self._distill_worker.run)
        self._distill_worker.progress.connect(self._on_distill_progress)
        self._distill_worker.finished.connect(self._on_distill_finished)
        self._distill_worker.error.connect(self._on_distill_error)
        # Always stop the thread when the worker emits a terminal signal.
        self._distill_worker.finished.connect(self._distill_thread.quit)
        self._distill_worker.error.connect(self._distill_thread.quit)
        self._distill_thread.finished.connect(self._cleanup_distill_thread)
        self._distill_thread.start()

    def _distill_wavelength_um(self):
        """Pick the wavelength to seed the initial PSF, mirroring the example.

        Widefield modality → emission wavelength (the shared wavelength_spin).
        Spinning disk → emission wavelength. Stored as µm.
        """
        if self.modality_combo.currentIndex() == 0:
            return self.wavelength_spin.value() / 1000.0
        return self.wavelength_em_spin.value() / 1000.0

    def _on_distill_progress(self, outer, max_outer, chi2, dpsf, damp):
        self.progress_bar.setMaximum(max_outer)
        self.progress_bar.setValue(outer)
        self.status_label.setText(
            f"Iter {outer}/{max_outer}  χ²={chi2:.3f}  "
            f"‖Δh‖={dpsf:.2e}  ‖Δa‖={damp:.2e}"
        )
        self._chi2_hist.append(float(chi2))
        self._dpsf_hist.append(float(dpsf))
        self._damp_hist.append(float(damp))
        self.convergence_plot.set_series("χ²", self._chi2_hist, color="#66CCFF")
        self.convergence_plot.set_series("‖Δh‖", self._dpsf_hist, color="#FF9966")
        self.convergence_plot.set_series("‖Δa‖", self._damp_hist, color="#99FF99")

    def _on_distill_finished(self):
        worker = self._distill_worker
        psf = worker.psf if worker is not None else None
        result = worker.result if worker is not None else None
        n_beads = len(result.positions) if result is not None else 0
        if psf is None:
            self._on_distill_error("Worker returned no PSF")
            return
        self._last_distillation_result = result
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.send_compare_btn.setEnabled(True)
        try:
            self._publish_psf(psf, "distilled")
            # Augment status with bead count if publish didn't error out.
            cur = self.status_label.text()
            if not cur.lower().startswith("error"):
                self.status_label.setText(f"{cur}  ({n_beads} beads)")
        except Exception as exc:
            self.status_label.setText(f"Error publishing PSF: {exc}")
            self.status_label.setStyleSheet("color: #F44;")

    def _on_distill_error(self, message):
        self.status_label.setText(f"Error: {message}")
        self.status_label.setStyleSheet("color: #F44;")

    def _send_to_comparison(self):
        if not self._chi2_hist:
            return
        dlg = get_convergence_comparison_dialog()
        n = len(self._chi2_hist)
        psf_shape = (
            int(self.nz_spin.value()),
            int(self.ny_spin.value()),
            int(self.nx_spin.value()),
        )
        base = f"distill {psf_shape[0]}×{psf_shape[1]}×{psf_shape[2]} ({n} iters)"
        dlg.add_run(f"{base} χ²", self._chi2_hist)
        dlg.add_run(f"{base} ‖Δh‖", self._dpsf_hist)
        dlg.add_run(f"{base} ‖Δa‖", self._damp_hist)

    def _cleanup_distill_thread(self):
        if self._distill_worker is not None:
            self._distill_worker.deleteLater()
        if self._distill_thread is not None:
            self._distill_thread.deleteLater()
        self._distill_worker = None
        self._distill_thread = None
        self.compute_btn.setEnabled(True)
        # Leave the progress bar at its final state so the user can read it.

    def closeEvent(self, event):
        # Don't tear down the dialog while a worker thread is alive.
        if self._distill_thread is not None:
            self.status_label.setText(
                "Distillation in progress — wait or kill the app to abort."
            )
            self.status_label.setStyleSheet("color: #C84;")
            event.ignore()
            return
        if self._preview_viewer is not None:
            self._preview_viewer.close()
        super().closeEvent(event)

    def _compute_pupil_psf(self):
        """Sample a 3D PSF from a loaded Pupil at the requested z grid."""
        from deconlib import fft_coords, pupil_to_psf
        from numpy.fft import ifftshift

        if self._loaded_pupil is None:
            raise ValueError("Load a .pupil.h5 file first")

        pupil = self._loaded_pupil
        dz = self.dz_spin.value()
        Nz = self.nz_spin.value()
        z = fft_coords(Nz, dz)
        geom = pupil.geometry
        psf = pupil_to_psf(
            pupil.pupil, geom, z, normalize=self.normalize_check.isChecked()
        )
        if self.center_check.isChecked():
            psf = ifftshift(psf)
        return psf.astype(np.float32, copy=False), "pupil_sampled"

    def _publish_psf(self, psf, source_tag):
        """Wrap a finished 3D PSF as an ImageBuffer and send to the selector."""
        from pyvistra.io import ImageBuffer

        modality = "widefield" if self.modality_combo.currentIndex() == 0 else "spinning_disk"
        self._metadata = self._build_metadata(modality)
        self._metadata["psf_source"] = source_tag

        psf_5d = psf[np.newaxis, :, np.newaxis, :, :]
        self._buffer = ImageBuffer(
            shape=psf_5d.shape,
            dtype=np.float32,
            metadata=self._metadata,
        )
        self._buffer[:] = psf_5d.astype(np.float32)

        output_meta = {
            "filename": "Computed PSF",
            "shape": self._buffer.shape,
            "scale": tuple(self._metadata["spacing"]),
            "is_rgb": False,
            # True when PSF peak is at array corner (FFT-native); False when
            # center_check shifted it to the array centre for display.
            "psf_dc_corner": not self.center_check.isChecked(),
        }
        output_meta.update(self._metadata)

        result = self.output_selector.send(self._buffer, output_meta)
        if result:
            if isinstance(result, str):
                self.status_label.setText(f"Saved to {result}")
            else:
                result.is_psf = True
                self.status_label.setText("PSF sent successfully")
            self.status_label.setStyleSheet("color: #4F4;")
        else:
            self.status_label.setText("PSF computed (output cancelled)")
            self.status_label.setStyleSheet("color: #888;")

        self.psf_computed.emit(self._buffer)

    def _build_metadata(self, modality):
        """Build metadata dict for PSF."""
        from datetime import datetime

        Nz = self.nz_spin.value()
        Ny = self.ny_spin.value()
        Nx = self.nx_spin.value()
        dz = self.dz_spin.value()
        dy = self.dy_spin.value()
        dx = self.dx_spin.value()

        meta = {
            "psf_format_version": "1.0",
            "modality": modality,
            "parameters": {
                "na": self.na_spin.value(),
                "ni": self.ni_spin.value(),
                "ns": self.ns_spin.value(),
                "normalize": self.normalize_check.isChecked(),
                "vectorial": self.vectorial_check.isChecked(),
                "centered": self.center_check.isChecked(),
            },
            "shape": [1, Nz, 1, Ny, Nx],
            "spacing": [dz, dy, dx],
            "aberrations": {
                "index_mismatch": self.aberr_check.isChecked(),
                "emitter_depth_um": self.depth_spin.value() if self.aberr_check.isChecked() else 0.0,
                "zernike_enabled": self.zernike_check.isChecked(),
                "zernike_coefficients": (
                    self._zernike_coeffs() if self.zernike_check.isChecked() else {}
                ),
            },
            "computed_at": datetime.now().isoformat(),
        }

        if modality == "widefield":
            meta["parameters"]["wavelength"] = self.wavelength_spin.value() / 1000.0
        else:
            meta["parameters"]["wavelength_exc"] = self.wavelength_exc_spin.value() / 1000.0
            meta["parameters"]["wavelength_em"] = self.wavelength_em_spin.value() / 1000.0
            meta["parameters"]["pinhole_um"] = self.pinhole_spin.value()
            meta["parameters"]["magnification"] = self.magnification_spin.value()
            meta["parameters"]["disk_magnification"] = self.disk_magnification_spin.value()

        return meta


# ---------------------------------------------------------------------------
# Worker: runs deconlib.distill_psf off the GUI thread.
# ---------------------------------------------------------------------------


class _PSFDistillWorker(QObject):
    """Background worker for PSF distillation.

    Builds a theoretical widefield init PSF + an OTF-masked support seed,
    then runs ``deconlib.distill_psf``. Progress is forwarded via
    ``distill_psf``'s callback hook. The final PSF is stashed on
    ``self.psf`` so the dialog can publish it through the output selector.
    """

    progress = Signal(int, int, float, float, float)  # outer, max, chi2, dpsf, damp
    finished = Signal()
    error = Signal(str)

    def __init__(self, image, params):
        super().__init__()
        self._image = image   # cropped (Z, Y, X) float32
        self._params = params
        self.psf = None       # final 3D PSF, written before finished
        self.result = None    # raw PsfDistillationResult

    def run(self):
        try:
            from numpy.fft import ifftshift
            from deconlib import (
                distill_psf,
                fft_coords,
                make_otf_mask,
                project_psf,
            )
            from deconlib.psf import compute_widefield_psf

            p = self._params
            psf_shape = p["psf_shape"]
            dz, dy, dx = p["dz"], p["dy"], p["dx"]
            wavelength = p["wavelength_um"]
            na, ni, ns = p["na"], p["ni"], p["ns"]

            # 1. Background-subtract once on the GUI-thread copy.
            background = p["background"]
            image_bg = self._image - background

            # 2. Theoretical widefield seed at the requested PSF grid.
            z = fft_coords(n=psf_shape[0], spacing=dz)
            init_psf = compute_widefield_psf(
                wavelength=wavelength,
                na=na, ni=ni, ns=ns,
                shape=(psf_shape[1], psf_shape[2]),
                spacing=(dy, dx),
                z=z,
                normalize=True,
            ).astype(np.float32)

            # 3. All-ones support; rely on the OTF mask for the band limit.
            support = np.ones(psf_shape, dtype=bool)
            otf_mask = make_otf_mask(
                psf_shape,
                pixel_pitch=(dz, dy, dx),
                numerical_aperture=na,
                wavelength=wavelength,
                refractive_index=ni,
            )
            init_psf = project_psf(
                init_psf, support, otf_mask=otf_mask
            ).astype(np.float32)

            max_outer = p["max_outer"]

            def progress_cb(outer, chi2, dpsf, damp):
                self.progress.emit(
                    int(outer), int(max_outer),
                    float(chi2), float(dpsf), float(damp),
                )

            # 4. Distill — Richardson-Lucy + amplitude solve.
            self.result = distill_psf(
                image_bg,
                background=0.0,
                support_mask=support,
                init_psf=init_psf,
                min_separation=p["min_separation"],
                min_intensity=p["min_intensity"],
                noise_floor=background,
                rl_steps_per_outer=p["rl_steps_per_outer"],
                max_outer=max_outer,
                psf_tol=p["psf_tol"],
                otf_mask=otf_mask,
                min_pad=p["min_pad"],
                store_history=False,
                verbose=False,
                callback=progress_cb,
            )

            psf = self.result.psf
            if p["center"]:
                psf = ifftshift(psf)
            self.psf = psf.astype(np.float32, copy=False)
            self.finished.emit()

        except Exception as exc:
            traceback.print_exc()
            self.error.emit(str(exc))
