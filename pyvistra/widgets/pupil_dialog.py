"""Dialog for producing a complex pupil function.

Two sources:

* **Theoretical** — a flat-amplitude pupil masked by the NA support
  (``deconlib.make_pupil``). Useful as a baseline and as a starting point
  for phase-retrieval.
* **Phase Retrieval** — iterative GS/ER/HIO from a measured bead-stack PSF
  intensity (``deconlib.retrieve_phase``), run on a background thread
  with live MSE/support-error reporting.

In retrieval mode the lateral pupil grid is forced to match the (region-
cropped) PSF stack — any other choice is either lossy (smaller) or
inventing samples (larger / zero-padded). The corresponding spinners are
disabled and tracked from the selected window + region.

The dialog publishes the pupil as a 2-channel image ``(amplitude, phase)``
through :class:`ImageOutputSelector`. Phase outside the NA support is
masked to NaN so it doesn't crush the contrast. For file destinations the
full :class:`deconlib.Pupil` is stashed in ``metadata["pupil"]`` so the
``.pupil.h5`` saver round-trips amplitude + phase + optics + retrieval
diagnostics. The embedded :class:`ConvergencePlotWidget` shows MSE +
support-error history and can push the current run into the shared
comparison window for cross-run review.
"""

import traceback

import numpy as np
from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .convergence_plot import ConvergencePlotWidget, get_convergence_comparison_dialog
from .output_selector import ImageOutputSelector
from .region_selector import RegionSelector

PSRC_THEORETICAL = 0
PSRC_RETRIEVAL = 1


class PupilComputeDialog(QDialog):
    """Compute a complex pupil function (theoretical or phase-retrieved)."""

    pupil_computed = Signal(object)  # emits deconlib.Pupil

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compute Pupil")
        self.setWindowFlags(Qt.Tool)
        self.resize(460, 640)

        self._buffer = None
        self._pupil = None  # deconlib.Pupil after a successful compute
        self._retrieve_thread = None
        self._retrieve_worker = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # --- Source Selector ---
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Theoretical", "Phase retrieval"])
        src_row.addWidget(self.source_combo)
        src_row.addStretch()
        layout.addLayout(src_row)

        self.source_stack = QStackedWidget()

        # Theoretical page — no params beyond the shared groups.
        theo_page = QWidget()
        theo_layout = QVBoxLayout(theo_page)
        theo_layout.setContentsMargins(0, 0, 0, 0)
        theo_note = QLabel(
            "A unit-amplitude pupil masked by the NA support. "
            "Uses the Optical Parameters and lateral grid below."
        )
        theo_note.setStyleSheet("color: #888; font-size: 10px;")
        theo_note.setWordWrap(True)
        theo_layout.addWidget(theo_note)

        # Phase Retrieval page.
        ret_page = QWidget()
        ret_layout = QVBoxLayout(ret_page)
        ret_layout.setContentsMargins(0, 0, 0, 0)
        ret_layout.setSpacing(6)

        bead_row = QHBoxLayout()
        bead_row.addWidget(QLabel("PSF stack:"))
        self.bead_window_combo = QComboBox()
        self.bead_window_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        bead_row.addWidget(self.bead_window_combo, 1)
        self.bead_refresh_btn = QPushButton("Refresh")
        self.bead_refresh_btn.clicked.connect(self._refresh_bead_windows)
        bead_row.addWidget(self.bead_refresh_btn)
        ret_layout.addLayout(bead_row)

        dz_row = QHBoxLayout()
        dz_row.addWidget(QLabel("dz (um):"))
        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.001, 10.0)
        self.dz_spin.setDecimals(3)
        self.dz_spin.setSingleStep(0.01)
        self.dz_spin.setValue(0.200)
        dz_row.addWidget(self.dz_spin)
        dz_row.addStretch()
        ret_layout.addLayout(dz_row)

        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["GS", "ER", "HIO"])
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        algo_row.addWidget(self.method_combo)
        algo_row.addWidget(QLabel("Max iters:"))
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 5000)
        self.max_iter_spin.setValue(200)
        algo_row.addWidget(self.max_iter_spin)
        ret_layout.addLayout(algo_row)

        self.vectorial_check = QCheckBox("Vectorial retrieval")
        self.vectorial_check.setChecked(True)
        self.vectorial_check.setToolTip(
            "Use retrieve_phase_vectorial (recommended at high NA). "
            "Uncheck for the scalar retrieve_phase model."
        )
        ret_layout.addWidget(self.vectorial_check)

        self.psf_centered_check = QCheckBox("Input PSF is centered (peak at array center)")
        self.psf_centered_check.setChecked(True)
        self.psf_centered_check.setToolTip(
            "Phase retrieval needs DC-at-corner. With this checked the "
            "PSF stack is ifftshift'ed before retrieval — the right "
            "thing for any bead-stack/distilled PSF you'd normally view "
            "in pyvistra. Uncheck only if the stack is already "
            "DC-at-corner."
        )
        ret_layout.addWidget(self.psf_centered_check)

        beta_row = QHBoxLayout()
        self.beta_label = QLabel("HIO β:")
        beta_row.addWidget(self.beta_label)
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.01, 1.5)
        self.beta_spin.setValue(0.9)
        self.beta_spin.setDecimals(2)
        self.beta_spin.setSingleStep(0.05)
        beta_row.addWidget(self.beta_spin)
        beta_row.addStretch()
        ret_layout.addLayout(beta_row)
        self.beta_label.setVisible(False)
        self.beta_spin.setVisible(False)

        self.unit_amp_check = QCheckBox("Enforce unit amplitude (phase-only)")
        self.unit_amp_check.setChecked(False)
        self.unit_amp_check.setToolTip(
            "If unchecked, amplitude apodization is recovered jointly with "
            "phase. The vectorial recipe recommends leaving this off."
        )
        ret_layout.addWidget(self.unit_amp_check)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Tolerance:"))
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(1e-12, 1.0)
        self.tol_spin.setDecimals(12)
        self.tol_spin.setSingleStep(1e-9)
        self.tol_spin.setValue(1e-10)
        self.tol_spin.setToolTip(
            "Stop when MSE drops below this. Set very small to run all "
            "max-iters; set higher to short-circuit obviously-converged runs."
        )
        tol_row.addWidget(self.tol_spin)
        tol_row.addStretch()
        ret_layout.addLayout(tol_row)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("NA edge sigma:"))
        self.boundary_sigma_spin = QDoubleSpinBox()
        self.boundary_sigma_spin.setRange(0.0, 10.0)
        self.boundary_sigma_spin.setDecimals(2)
        self.boundary_sigma_spin.setSingleStep(0.1)
        self.boundary_sigma_spin.setValue(1.5)
        self.boundary_sigma_spin.setToolTip(
            "Gaussian sigma (in pupil pixels) for softening the NA support "
            "edge. 0 = hard edge. ~1.5 is the recommended setting for "
            "phase retrieval — softens the discontinuity that drives "
            "speckle at the NA boundary."
        )
        smooth_row.addWidget(self.boundary_sigma_spin)
        smooth_row.addStretch()
        ret_layout.addLayout(smooth_row)

        # Real-space regularization (make_pupil_real_filter).
        reg_group = QGroupBox("Real-space regularization")
        reg_group.setCheckable(True)
        reg_group.setChecked(True)
        reg_group.setToolTip(
            "Apply a real-space prior on the pupil's IFFT each iteration. "
            "Suppresses per-pixel speckle from FFT wrap-around and from "
            "the pupil being underdetermined."
        )
        reg_layout = QVBoxLayout(reg_group)
        reg_layout.setContentsMargins(8, 6, 8, 6)
        reg_layout.setSpacing(4)

        reg_kind_row = QHBoxLayout()
        reg_kind_row.addWidget(QLabel("Kind:"))
        self.reg_kind_combo = QComboBox()
        self.reg_kind_combo.addItems(["biharmonic", "tukey"])
        self.reg_kind_combo.setToolTip(
            "biharmonic = soft 1/(1+(r/R)^4) profile (Laplacian smoothness). "
            "tukey = hard disc of radius R with a cosine taper of width alpha."
        )
        self.reg_kind_combo.currentIndexChanged.connect(self._on_reg_kind_changed)
        reg_kind_row.addWidget(self.reg_kind_combo)
        reg_kind_row.addWidget(QLabel("Radius (μm):"))
        self.reg_radius_spin = QDoubleSpinBox()
        self.reg_radius_spin.setRange(0.05, 100.0)
        self.reg_radius_spin.setDecimals(3)
        self.reg_radius_spin.setSingleStep(0.1)
        self.reg_radius_spin.setValue(3.0)
        self.reg_radius_spin.setToolTip(
            "Real-space cutoff/transition radius (μm). Should be a few "
            "times larger than the expected PSF extent."
        )
        reg_kind_row.addWidget(self.reg_radius_spin)
        reg_layout.addLayout(reg_kind_row)

        self.reg_alpha_label = QLabel("α (tukey taper):")
        self.reg_alpha_spin = QDoubleSpinBox()
        self.reg_alpha_spin.setRange(0.0, 1.0)
        self.reg_alpha_spin.setDecimals(2)
        self.reg_alpha_spin.setSingleStep(0.05)
        self.reg_alpha_spin.setValue(0.25)
        self.reg_alpha_spin.setToolTip(
            "Cosine-taper fraction for tukey. 0 = rectangular disc, "
            "1 = Hann. Ignored for biharmonic."
        )
        reg_alpha_row = QHBoxLayout()
        reg_alpha_row.addWidget(self.reg_alpha_label)
        reg_alpha_row.addWidget(self.reg_alpha_spin)
        reg_alpha_row.addStretch()
        reg_layout.addLayout(reg_alpha_row)
        self.reg_group = reg_group
        ret_layout.addWidget(reg_group)
        self._on_reg_kind_changed(self.reg_kind_combo.currentIndex())

        ret_note = QLabel(
            "Lateral pupil grid (Ny, Nx) is locked to the (cropped) PSF "
            "stack. Cropping a region below resizes the pupil accordingly."
        )
        ret_note.setStyleSheet("color: #888; font-size: 10px;")
        ret_note.setWordWrap(True)
        ret_layout.addWidget(ret_note)

        # Optional region restriction (rect ROI + Z planes).
        self.retrieval_region = RegionSelector(title="Restrict to region (optional)")
        ret_layout.addWidget(self.retrieval_region)
        self.bead_window_combo.currentIndexChanged.connect(
            self._on_bead_window_changed
        )
        self.retrieval_region.region_changed.connect(self._sync_retrieval_grid)

        self.source_stack.addWidget(theo_page)
        self.source_stack.addWidget(ret_page)
        layout.addWidget(self.source_stack)

        # --- Optical Parameters ---
        opt_group = QGroupBox("Optical Parameters")
        opt_layout = QVBoxLayout(opt_group)

        wl_row = QHBoxLayout()
        wl_row.addWidget(QLabel("Wavelength (nm):"))
        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(300, 1000)
        self.wavelength_spin.setValue(525.0)
        self.wavelength_spin.setDecimals(1)
        wl_row.addWidget(self.wavelength_spin)
        opt_layout.addLayout(wl_row)

        na_row = QHBoxLayout()
        na_row.addWidget(QLabel("NA:"))
        self.na_spin = QDoubleSpinBox()
        self.na_spin.setRange(0.1, 2.0)
        self.na_spin.setValue(1.40)
        self.na_spin.setDecimals(2)
        self.na_spin.setSingleStep(0.01)
        na_row.addWidget(self.na_spin)
        opt_layout.addLayout(na_row)

        ni_row = QHBoxLayout()
        ni_row.addWidget(QLabel("n_immersion:"))
        self.ni_spin = QDoubleSpinBox()
        self.ni_spin.setRange(1.0, 2.0)
        self.ni_spin.setDecimals(3)
        self.ni_spin.setSingleStep(0.001)
        self.ni_spin.setValue(1.515)
        ni_row.addWidget(self.ni_spin)
        opt_layout.addLayout(ni_row)

        ns_row = QHBoxLayout()
        ns_row.addWidget(QLabel("n_sample:"))
        self.ns_spin = QDoubleSpinBox()
        self.ns_spin.setRange(1.0, 2.0)
        self.ns_spin.setDecimals(3)
        self.ns_spin.setSingleStep(0.001)
        self.ns_spin.setValue(1.33)
        ns_row.addWidget(self.ns_spin)
        opt_layout.addLayout(ns_row)

        layout.addWidget(opt_group)

        # --- Pupil Grid (lateral shape + spacing) ---
        self.grid_group = QGroupBox("Pupil Grid")
        grid_layout = QVBoxLayout(self.grid_group)

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Ny:"))
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(8, 2048)
        self.ny_spin.setValue(128)
        shape_row.addWidget(self.ny_spin)
        shape_row.addWidget(QLabel("Nx:"))
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(8, 2048)
        self.nx_spin.setValue(128)
        shape_row.addWidget(self.nx_spin)
        shape_row.addStretch()
        grid_layout.addLayout(shape_row)

        spacing_row = QHBoxLayout()
        spacing_row.addWidget(QLabel("dy (um):"))
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.001, 10.0)
        self.dy_spin.setDecimals(3)
        self.dy_spin.setSingleStep(0.001)
        self.dy_spin.setValue(0.065)
        spacing_row.addWidget(self.dy_spin)
        spacing_row.addWidget(QLabel("dx (um):"))
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.001, 10.0)
        self.dx_spin.setDecimals(3)
        self.dx_spin.setSingleStep(0.001)
        self.dx_spin.setValue(0.065)
        spacing_row.addWidget(self.dx_spin)
        spacing_row.addStretch()
        grid_layout.addLayout(spacing_row)

        layout.addWidget(self.grid_group)

        # --- Options ---
        opts_group = QGroupBox("Options")
        opts_layout = QHBoxLayout(opts_group)
        self.center_check = QCheckBox("Center pupil (ifftshift)")
        self.center_check.setChecked(True)
        self.center_check.setToolTip(
            "Apply ifftshift to the pupil so DC sits at the array center "
            "for display. Storage convention (corner-origin) is preserved "
            "in the saved .pupil.h5."
        )
        opts_layout.addWidget(self.center_check)
        opts_layout.addStretch()
        layout.addWidget(opts_group)

        # --- Convergence plot (retrieval only) ---
        self.convergence_group = QGroupBox("Convergence")
        conv_layout = QVBoxLayout(self.convergence_group)
        conv_layout.setContentsMargins(8, 6, 8, 6)
        conv_layout.setSpacing(4)
        self.convergence_plot = ConvergencePlotWidget()
        self.convergence_plot.set_labels(
            x_label="Iteration", y_label="error", title="MSE + support error"
        )
        conv_layout.addWidget(self.convergence_plot)
        conv_btn_row = QHBoxLayout()
        self.send_compare_btn = QPushButton("Send to comparison…")
        self.send_compare_btn.setToolTip(
            "Push the current MSE + support-error histories to the shared "
            "comparison window so they can be reviewed alongside other runs."
        )
        self.send_compare_btn.clicked.connect(self._send_to_comparison)
        self.send_compare_btn.setEnabled(False)
        conv_btn_row.addWidget(self.send_compare_btn)
        conv_btn_row.addStretch()
        conv_layout.addLayout(conv_btn_row)
        layout.addWidget(self.convergence_group)

        # --- Output Selector ---
        self.output_selector = ImageOutputSelector(
            default_title="Computed Pupil",
            formats=[".pupil.h5"],
        )
        layout.addWidget(self.output_selector)

        layout.addStretch()
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, stretch=1)

        # --- Compute button + progress + status ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._compute)
        btn_layout.addWidget(self.compute_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")
        main_layout.addWidget(self.status_label)

        # Connect source-change handler after both pages exist.
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self._on_source_changed(self.source_combo.currentIndex())

    # ------------------------------------------------------------------
    # UI handlers
    # ------------------------------------------------------------------

    def _on_source_changed(self, index):
        self.source_stack.setCurrentIndex(index)
        retrieval = index == PSRC_RETRIEVAL
        # In retrieval mode the lateral grid is driven by the PSF stack.
        self.ny_spin.setEnabled(not retrieval)
        self.nx_spin.setEnabled(not retrieval)
        self.convergence_group.setVisible(retrieval)
        if retrieval:
            self.grid_group.setTitle("Pupil Grid  (Ny, Nx locked to PSF stack)")
            self._refresh_bead_windows()
        else:
            self.grid_group.setTitle("Pupil Grid")

    def _on_method_changed(self, _index):
        is_hio = self.method_combo.currentText() == "HIO"
        self.beta_label.setVisible(is_hio)
        self.beta_spin.setVisible(is_hio)

    def _on_reg_kind_changed(self, _index):
        is_tukey = self.reg_kind_combo.currentText() == "tukey"
        self.reg_alpha_label.setVisible(is_tukey)
        self.reg_alpha_spin.setVisible(is_tukey)

    def _refresh_bead_windows(self):
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
        self.retrieval_region.set_source(source)
        if source is not None:
            self._adopt_window_spacing(source)
        self._sync_retrieval_grid()

    def _sync_retrieval_grid(self):
        """Lock Ny/Nx to the (cropped) PSF stack lateral shape."""
        if self.source_combo.currentIndex() != PSRC_RETRIEVAL:
            return
        eff = self._effective_source_shape()
        if eff is None:
            return
        _, ny, nx = eff
        for spin, val in ((self.ny_spin, ny), (self.nx_spin, nx)):
            spin.blockSignals(True)
            # Range may be tighter than our locked value; widen if needed.
            if val > spin.maximum():
                spin.setMaximum(int(val))
            if val < spin.minimum():
                spin.setMinimum(int(val))
            spin.setValue(int(val))
            spin.blockSignals(False)

    def _effective_source_shape(self):
        """Return ``(Nz, Ny, Nx)`` of the PSF stack after the region crop."""
        from ..ui.manager import manager

        wid = self.bead_window_combo.currentData()
        source = manager.get(wid) if wid is not None else None
        if source is None:
            return None
        shape = getattr(source.img_data, "shape", None)
        if shape is None or len(shape) != 5:
            return None
        _, Z, _, Y, X = shape
        zr = self.retrieval_region.z_range()
        bb = self.retrieval_region.bbox()
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

    # ------------------------------------------------------------------
    # Compute paths
    # ------------------------------------------------------------------

    def _compute(self):
        if self._retrieve_thread is not None:
            return  # already running
        if self.na_spin.value() >= self.ni_spin.value():
            self._set_status("NA must be < n_immersion", error=True)
            return

        source = self.source_combo.currentIndex()
        if source == PSRC_RETRIEVAL:
            try:
                self._start_retrieval_async()
            except Exception as exc:
                self._set_status(f"Error: {exc}", error=True)
                self.compute_btn.setEnabled(True)
            return

        self._set_status("Computing...")
        self.compute_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            pupil = self._compute_theoretical()
            self._publish_pupil(pupil)
        except ImportError:
            self._set_status("Error: deconlib not installed", error=True)
        except Exception as exc:
            self._set_status(f"Error: {exc}", error=True)
        finally:
            self.compute_btn.setEnabled(True)

    def _build_optics(self):
        from deconlib import Optics

        return Optics(
            wavelength=self.wavelength_spin.value() / 1000.0,
            na=self.na_spin.value(),
            ni=self.ni_spin.value(),
            ns=self.ns_spin.value(),
        )

    def _compute_theoretical(self):
        from deconlib import Pupil, make_geometry, make_pupil

        optics = self._build_optics()
        shape = (self.ny_spin.value(), self.nx_spin.value())
        spacing = (self.dy_spin.value(), self.dx_spin.value())
        geom = make_geometry(shape, spacing, optics)
        field = make_pupil(geom).astype(np.complex64)
        return Pupil(
            pupil=field,
            optics=optics,
            shape=shape,
            spacing=spacing,
            source="theoretical",
        )

    def _start_retrieval_async(self):
        """Validate inputs, snapshot the PSF stack, and kick off the worker."""
        from ..ui.manager import manager

        wid = self.bead_window_combo.currentData()
        if wid is None:
            raise ValueError("No PSF stack selected")
        window = manager.get(wid)
        if window is None:
            raise ValueError("Selected window is no longer open")

        image_5d = np.asarray(window.img_data[:])
        psf_stack = image_5d[0, :, 0, :, :].astype(np.float32, copy=False)
        psf_stack = np.ascontiguousarray(
            self.retrieval_region.crop_zyx(psf_stack)
        )

        # FFT convention: retrieve_phase expects DC-at-corner. The
        # displayed PSF is centered (peak at the array center), so move
        # the origin from the centre back to index 0 on every axis.
        # Clamp negatives — measured intensity is non-negative, and the
        # retrieval takes √intensity, which blows up on negatives.
        if self.psf_centered_check.isChecked():
            psf_stack = np.fft.ifftshift(psf_stack)
        psf_stack = np.ascontiguousarray(
            np.maximum(psf_stack, 0.0).astype(np.float32, copy=False)
        )
        nz, ny, nx = psf_stack.shape

        # The lateral grid is locked to the stack — this is a safety net.
        if (ny, nx) != (self.ny_spin.value(), self.nx_spin.value()):
            raise ValueError(
                f"Internal: pupil grid ({self.ny_spin.value()}, "
                f"{self.nx_spin.value()}) drifted from PSF stack ({ny}, {nx})."
            )

        optics = self._build_optics()
        shape = (ny, nx)
        spacing = (self.dy_spin.value(), self.dx_spin.value())

        params = {
            "shape": shape,
            "spacing": spacing,
            "dz": float(self.dz_spin.value()),
            "max_iter": int(self.max_iter_spin.value()),
            "method": self.method_combo.currentText(),
            "enforce_unit_amplitude": self.unit_amp_check.isChecked(),
            "beta": float(self.beta_spin.value()),
            "tol": float(self.tol_spin.value()),
            "vectorial": self.vectorial_check.isChecked(),
            "boundary_smoothing_sigma": float(self.boundary_sigma_spin.value()),
            "regularization": (
                {
                    "kind": self.reg_kind_combo.currentText(),
                    "radius": float(self.reg_radius_spin.value()),
                    "alpha": float(self.reg_alpha_spin.value()),
                }
                if self.reg_group.isChecked()
                else None
            ),
        }

        self.compute_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, params["max_iter"])
        self.progress_bar.setValue(0)
        self.convergence_plot.clear()
        self.send_compare_btn.setEnabled(False)
        self._set_status(
            f"Retrieving phase ({params['method']})…", error=False
        )

        self._retrieve_worker = _PupilRetrievalWorker(psf_stack, optics, params)
        self._retrieve_thread = QThread(self)
        self._retrieve_worker.moveToThread(self._retrieve_thread)
        self._retrieve_thread.started.connect(self._retrieve_worker.run)
        self._retrieve_worker.progress.connect(self._on_retrieval_progress)
        self._retrieve_worker.finished.connect(self._on_retrieval_finished)
        self._retrieve_worker.error.connect(self._on_retrieval_error)
        self._retrieve_worker.finished.connect(self._retrieve_thread.quit)
        self._retrieve_worker.error.connect(self._retrieve_thread.quit)
        self._retrieve_thread.finished.connect(self._cleanup_retrieve_thread)
        self._retrieve_thread.start()

    def _on_retrieval_progress(self, iteration, max_iter, mse_hist, supp_hist):
        self.progress_bar.setMaximum(max_iter)
        self.progress_bar.setValue(iteration)
        if mse_hist:
            self._set_status(
                f"Iter {iteration}/{max_iter}  "
                f"mse={mse_hist[-1]:.3e}  supp={supp_hist[-1]:.3e}"
            )
        self.convergence_plot.set_series("MSE", mse_hist, color="#66CCFF")
        self.convergence_plot.set_series(
            "Support error", supp_hist, color="#FF9966"
        )

    def _on_retrieval_finished(self):
        worker = self._retrieve_worker
        pupil = worker.pupil if worker is not None else None
        if pupil is None:
            self._on_retrieval_error("Worker returned no pupil")
            return
        self.progress_bar.setValue(self.progress_bar.maximum())
        diag = pupil.retrieval_diagnostics or {}
        self.convergence_plot.set_series(
            "MSE", diag.get("mse_history", []), color="#66CCFF"
        )
        self.convergence_plot.set_series(
            "Support error",
            diag.get("support_error_history", []),
            color="#FF9966",
        )
        self.send_compare_btn.setEnabled(True)
        try:
            self._publish_pupil(pupil)
            cur = self.status_label.text()
            if not cur.lower().startswith("error"):
                conv = "converged" if diag.get("converged") else "stopped"
                self._set_status(
                    f"{cur}  ({diag.get('iterations', '?')} iters, {conv})",
                    ok=True,
                )
        except Exception as exc:
            self._set_status(f"Error publishing pupil: {exc}", error=True)

    def _on_retrieval_error(self, message):
        self._set_status(f"Error: {message}", error=True)

    def _cleanup_retrieve_thread(self):
        if self._retrieve_worker is not None:
            self._retrieve_worker.deleteLater()
        if self._retrieve_thread is not None:
            self._retrieve_thread.deleteLater()
        self._retrieve_worker = None
        self._retrieve_thread = None
        self.compute_btn.setEnabled(True)

    def closeEvent(self, event):
        if self._retrieve_thread is not None:
            self._set_status(
                "Phase retrieval in progress — wait or kill the app to abort.",
                error=False,
            )
            event.ignore()
            return
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Output routing
    # ------------------------------------------------------------------

    def _send_to_comparison(self):
        diag = (self._pupil.retrieval_diagnostics or {}) if self._pupil else {}
        mse = diag.get("mse_history")
        if not mse:
            return
        dlg = get_convergence_comparison_dialog()
        method = self.method_combo.currentText()
        max_iter = self.max_iter_spin.value()
        base = f"{method} max={max_iter}"
        dlg.add_run(f"{base} MSE", mse)
        supp = diag.get("support_error_history")
        if supp:
            dlg.add_run(f"{base} supp-err", supp)

    def _publish_pupil(self, pupil):
        """Package the pupil as (amp, phase) channels and route via the selector.

        Phase is masked to NaN outside the NA support so it doesn't crush
        the contrast. The selector's "Center pupil (ifftshift)" option
        controls whether DC sits at the array centre for display.
        """
        from pyvistra.io import ImageBuffer

        self._pupil = pupil
        field = pupil.pupil
        if self.center_check.isChecked():
            field = np.fft.ifftshift(field)
            try:
                mask = np.fft.ifftshift(pupil.geometry.mask)
            except Exception:
                mask = None
        else:
            try:
                mask = pupil.geometry.mask
            except Exception:
                mask = None

        amp = np.abs(field).astype(np.float32)
        phase = np.angle(field).astype(np.float32)
        if mask is not None:
            phase = np.where(mask, phase, np.nan).astype(np.float32)

        # (T, Z, C, Y, X) with C=2: amplitude, phase
        amp_5d = amp[np.newaxis, np.newaxis, np.newaxis, :, :]
        phase_5d = phase[np.newaxis, np.newaxis, np.newaxis, :, :]
        pupil_5d = np.concatenate([amp_5d, phase_5d], axis=2)

        metadata = {
            "shape": pupil_5d.shape,
            "scale": (1.0, pupil.spacing[0], pupil.spacing[1]),
            "is_rgb": False,
            "channels": [
                {"name": "amplitude"},
                {"name": "phase"},
            ],
            "pupil": pupil,
            "pupil_source": pupil.source,
            "pupil_centered": self.center_check.isChecked(),
            "parameters": {
                "wavelength": pupil.optics.wavelength,
                "na": pupil.optics.na,
                "ni": pupil.optics.ni,
                "ns": pupil.optics.ns,
            },
            "spacing": list(pupil.spacing),
        }

        self._buffer = ImageBuffer(
            shape=pupil_5d.shape, dtype=np.float32, metadata=metadata
        )
        self._buffer[:] = pupil_5d

        output_meta = {"filename": "Computed Pupil"}
        output_meta.update(metadata)

        result = self.output_selector.send(self._buffer, output_meta)
        if result:
            if isinstance(result, str):
                self._set_status(f"Saved to {result}", ok=True)
            else:
                self._set_status("Pupil sent", ok=True)
        else:
            self._set_status("Pupil computed (output cancelled)")
        self.pupil_computed.emit(pupil)

    def _set_status(self, text, *, ok=False, error=False):
        self.status_label.setText(text)
        if error:
            self.status_label.setStyleSheet("color: #F44;")
        elif ok:
            self.status_label.setStyleSheet("color: #4F4;")
        else:
            self.status_label.setStyleSheet("color: #888;")


# ---------------------------------------------------------------------------
# Worker: runs deconlib.retrieve_phase off the GUI thread.
# ---------------------------------------------------------------------------


class _PupilRetrievalWorker(QObject):
    """Background worker for iterative pupil phase retrieval.

    Builds the geometry + z-grid from the supplied stack shape and optics,
    then calls ``deconlib.retrieve_phase`` with a callback that forwards
    each iteration's MSE + support error back to the GUI thread. The
    final :class:`deconlib.Pupil` is stashed on ``self.pupil`` so the
    dialog can publish it from ``_on_retrieval_finished``.
    """

    progress = Signal(int, int, list, list)  # iter, max_iter, mse_hist, supp_hist
    finished = Signal()
    error = Signal(str)

    def __init__(self, psf_stack, optics, params):
        super().__init__()
        self._psf = psf_stack
        self._optics = optics
        self._params = params
        self.pupil = None
        self._mse_hist = []
        self._supp_hist = []

    def run(self):
        try:
            from deconlib import (
                Pupil,
                fft_coords,
                make_geometry,
                make_pupil_real_filter,
                retrieve_phase,
                retrieve_phase_vectorial,
            )

            p = self._params
            geom = make_geometry(
                p["shape"], p["spacing"], self._optics,
                boundary_smoothing_sigma=p["boundary_smoothing_sigma"],
            )
            nz = self._psf.shape[0]
            z_planes = fft_coords(nz, p["dz"])
            max_iter = p["max_iter"]

            pupil_filter = None
            reg = p.get("regularization")
            if reg is not None:
                pupil_filter = make_pupil_real_filter(
                    geom,
                    radius=reg["radius"],
                    kind=reg["kind"],
                    alpha=reg["alpha"],
                )

            def callback(iteration, mse, support_error):
                self._mse_hist.append(float(mse))
                self._supp_hist.append(float(support_error))
                # Hand a copy to the GUI thread — the lists keep growing
                # here and Qt's auto-connection queues the snapshot.
                self.progress.emit(
                    int(iteration), int(max_iter),
                    list(self._mse_hist), list(self._supp_hist),
                )

            kwargs = dict(
                measured_psf=self._psf,
                z_planes=z_planes,
                geom=geom,
                max_iter=max_iter,
                method=p["method"],
                tol=p["tol"],
                enforce_unit_amplitude=p["enforce_unit_amplitude"],
                pupil_real_filter=pupil_filter,
                callback=callback,
            )
            if p["method"] == "HIO":
                kwargs["beta"] = p["beta"]

            if p["vectorial"]:
                result = retrieve_phase_vectorial(optics=self._optics, **kwargs)
            else:
                result = retrieve_phase(**kwargs)
            self.pupil = Pupil.from_retrieval(
                result,
                optics=self._optics,
                shape=p["shape"],
                spacing=p["spacing"],
                boundary_smoothing_sigma=p["boundary_smoothing_sigma"],
            )
            self.finished.emit()
        except Exception as exc:
            traceback.print_exc()
            self.error.emit(str(exc))
