import numpy as np
from qtpy.QtCore import Qt, Signal
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
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .output_selector import ImageOutputSelector


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
        self.resize(400, 580)

        self._buffer = None  # Holds computed PSF as ImageBuffer
        self._metadata = None  # PSF parameters for saving

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Scroll area for form content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Modality Selector
        modality_layout = QHBoxLayout()
        modality_layout.addWidget(QLabel("Modality:"))
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["Widefield", "Spinning Disk"])
        self.modality_combo.currentIndexChanged.connect(self._on_modality_changed)
        modality_layout.addWidget(self.modality_combo)
        modality_layout.addStretch()
        layout.addLayout(modality_layout)

        # Optical Parameters Group
        optical_group = QGroupBox("Optical Parameters")
        optical_layout = QVBoxLayout(optical_group)

        # Wavelength
        wl_row = QHBoxLayout()
        wl_row.addWidget(QLabel("Wavelength (nm):"))
        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(300, 1000)
        self.wavelength_spin.setValue(525.0)
        self.wavelength_spin.setDecimals(1)
        wl_row.addWidget(self.wavelength_spin)
        optical_layout.addLayout(wl_row)

        # NA
        na_row = QHBoxLayout()
        na_row.addWidget(QLabel("NA:"))
        self.na_spin = QDoubleSpinBox()
        self.na_spin.setRange(0.1, 2.0)
        self.na_spin.setValue(1.40)
        self.na_spin.setDecimals(2)
        self.na_spin.setSingleStep(0.01)
        na_row.addWidget(self.na_spin)
        optical_layout.addLayout(na_row)

        # n_immersion
        ni_row = QHBoxLayout()
        ni_row.addWidget(QLabel("n_immersion:"))
        self.ni_spin = QDoubleSpinBox()
        self.ni_spin.setRange(1.0, 2.0)
        self.ni_spin.setValue(1.515)
        self.ni_spin.setDecimals(3)
        self.ni_spin.setSingleStep(0.001)
        ni_row.addWidget(self.ni_spin)
        optical_layout.addLayout(ni_row)

        # n_sample
        ns_row = QHBoxLayout()
        ns_row.addWidget(QLabel("n_sample:"))
        self.ns_spin = QDoubleSpinBox()
        self.ns_spin.setRange(1.0, 2.0)
        self.ns_spin.setValue(1.33)
        self.ns_spin.setDecimals(3)
        self.ns_spin.setSingleStep(0.001)
        ns_row.addWidget(self.ns_spin)
        optical_layout.addLayout(ns_row)

        layout.addWidget(optical_group)

        # Spinning Disk Parameters Group (conditionally visible)
        self.sd_group = QGroupBox("Spinning Disk")
        sd_layout = QVBoxLayout(self.sd_group)

        # Excitation Wavelength
        exc_row = QHBoxLayout()
        exc_row.addWidget(QLabel("Exc. Wavelength (nm):"))
        self.wavelength_exc_spin = QDoubleSpinBox()
        self.wavelength_exc_spin.setRange(300, 1000)
        self.wavelength_exc_spin.setValue(488.0)
        self.wavelength_exc_spin.setDecimals(1)
        exc_row.addWidget(self.wavelength_exc_spin)
        sd_layout.addLayout(exc_row)

        # Emission Wavelength
        em_row = QHBoxLayout()
        em_row.addWidget(QLabel("Em. Wavelength (nm):"))
        self.wavelength_em_spin = QDoubleSpinBox()
        self.wavelength_em_spin.setRange(300, 1000)
        self.wavelength_em_spin.setValue(525.0)
        self.wavelength_em_spin.setDecimals(1)
        em_row.addWidget(self.wavelength_em_spin)
        sd_layout.addLayout(em_row)

        # Pinhole
        pinhole_row = QHBoxLayout()
        pinhole_row.addWidget(QLabel("Pinhole (um):"))
        self.pinhole_spin = QDoubleSpinBox()
        self.pinhole_spin.setRange(1.0, 200.0)
        self.pinhole_spin.setValue(50.0)
        self.pinhole_spin.setDecimals(1)
        pinhole_row.addWidget(self.pinhole_spin)
        sd_layout.addLayout(pinhole_row)

        # Magnification
        mag_row = QHBoxLayout()
        mag_row.addWidget(QLabel("Magnification:"))
        self.magnification_spin = QDoubleSpinBox()
        self.magnification_spin.setRange(1.0, 200.0)
        self.magnification_spin.setValue(100.0)
        self.magnification_spin.setDecimals(1)
        mag_row.addWidget(self.magnification_spin)
        sd_layout.addLayout(mag_row)

        # Disk Magnification
        disk_mag_row = QHBoxLayout()
        disk_mag_row.addWidget(QLabel("Disk Magnification:"))
        self.disk_magnification_spin = QDoubleSpinBox()
        self.disk_magnification_spin.setRange(0.1, 10.0)
        self.disk_magnification_spin.setValue(1.0)
        self.disk_magnification_spin.setDecimals(2)
        disk_mag_row.addWidget(self.disk_magnification_spin)
        sd_layout.addLayout(disk_mag_row)

        layout.addWidget(self.sd_group)
        self.sd_group.setVisible(False)  # Hidden by default (Widefield)

        # Output Shape Group
        shape_group = QGroupBox("Output Shape")
        shape_layout = QHBoxLayout(shape_group)

        shape_layout.addWidget(QLabel("Nz:"))
        self.nz_spin = QSpinBox()
        self.nz_spin.setRange(1, 512)
        self.nz_spin.setValue(64)
        shape_layout.addWidget(self.nz_spin)

        shape_layout.addWidget(QLabel("Ny:"))
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(1, 2048)
        self.ny_spin.setValue(128)
        shape_layout.addWidget(self.ny_spin)

        shape_layout.addWidget(QLabel("Nx:"))
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(1, 2048)
        self.nx_spin.setValue(128)
        shape_layout.addWidget(self.nx_spin)

        layout.addWidget(shape_group)

        # Voxel Spacing Group
        spacing_group = QGroupBox("Voxel Spacing (um)")
        spacing_layout = QHBoxLayout(spacing_group)

        spacing_layout.addWidget(QLabel("dz:"))
        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.001, 10.0)
        self.dz_spin.setValue(0.200)
        self.dz_spin.setDecimals(3)
        self.dz_spin.setSingleStep(0.01)
        spacing_layout.addWidget(self.dz_spin)

        spacing_layout.addWidget(QLabel("dy:"))
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.001, 10.0)
        self.dy_spin.setValue(0.065)
        self.dy_spin.setDecimals(3)
        self.dy_spin.setSingleStep(0.001)
        spacing_layout.addWidget(self.dy_spin)

        spacing_layout.addWidget(QLabel("dx:"))
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.001, 10.0)
        self.dx_spin.setValue(0.065)
        self.dx_spin.setDecimals(3)
        self.dx_spin.setSingleStep(0.001)
        spacing_layout.addWidget(self.dx_spin)

        layout.addWidget(spacing_group)

        # Options Group
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout(options_group)

        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.setChecked(True)
        options_layout.addWidget(self.normalize_check)

        self.vectorial_check = QCheckBox("Vectorial")
        self.vectorial_check.setChecked(False)
        options_layout.addWidget(self.vectorial_check)

        self.center_check = QCheckBox("Center output (ifftshift)")
        self.center_check.setChecked(True)
        options_layout.addWidget(self.center_check)

        layout.addWidget(options_group)

        # Aberrations Group
        aberr_group = QGroupBox("Aberrations")
        aberr_layout = QVBoxLayout(aberr_group)

        self.aberr_check = QCheckBox("Index Mismatch")
        self.aberr_check.setChecked(False)
        aberr_layout.addWidget(self.aberr_check)

        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Emitter depth (um):"))
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(-100.0, 100.0)
        self.depth_spin.setValue(-1.0)
        self.depth_spin.setDecimals(2)
        self.depth_spin.setSingleStep(0.1)
        depth_row.addWidget(self.depth_spin)
        aberr_layout.addLayout(depth_row)

        depth_note = QLabel("(negative = into sample)")
        depth_note.setStyleSheet("color: #888; font-size: 10px;")
        aberr_layout.addWidget(depth_note)

        layout.addWidget(aberr_group)

        # Output Selector
        self.output_selector = ImageOutputSelector(
            default_title="Computed PSF",
            formats=[("TIFF", ".tif"), ("PSF Zarr", ".psf.zarr")],
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

        # Status Label (pinned at bottom, outside scroll area)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")
        main_layout.addWidget(self.status_label)

    def _on_modality_changed(self, index):
        """Show/hide spinning disk parameters based on selection."""
        is_spinning_disk = index == 1
        self.sd_group.setVisible(is_spinning_disk)

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
        """Compute the PSF based on current parameters."""
        # Validate parameters
        errors = self._validate_parameters()
        if errors:
            self.status_label.setText(f"Error: {errors[0]}")
            self.status_label.setStyleSheet("color: #F44;")
            return

        self.status_label.setText("Computing...")
        self.status_label.setStyleSheet("color: #888;")
        self.compute_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            from deconlib import fft_coords
            from deconlib.psf import compute_widefield_psf, compute_spinning_disk_psf
            from numpy.fft import ifftshift

            # Get parameters
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

            # Build z-coordinates (FFT convention: DC at z[0])
            z = fft_coords(Nz, dz)

            # Build aberrations if enabled
            aberrations = None
            if self.aberr_check.isChecked():
                from deconlib.psf.aberrations import IndexMismatch
                aberrations = [IndexMismatch(self.depth_spin.value())]

            # Compute PSF based on modality
            modality = "widefield" if self.modality_combo.currentIndex() == 0 else "spinning_disk"

            if modality == "widefield":
                wavelength_um = self.wavelength_spin.value() / 1000.0  # nm to um
                psf = compute_widefield_psf(
                    wavelength=wavelength_um,
                    na=na,
                    ni=ni,
                    ns=ns,
                    shape=(Ny, Nx),
                    spacing=(dy, dx),
                    z=z,
                    normalize=normalize,
                    aberrations=aberrations,
                    vectorial=vectorial,
                )
            else:
                # Spinning disk
                wavelength_exc_um = self.wavelength_exc_spin.value() / 1000.0
                wavelength_em_um = self.wavelength_em_spin.value() / 1000.0
                pinhole_um = self.pinhole_spin.value()
                magnification = self.magnification_spin.value()
                disk_magnification = self.disk_magnification_spin.value()

                psf = compute_spinning_disk_psf(
                    wavelength_exc=wavelength_exc_um,
                    wavelength_em=wavelength_em_um,
                    na=na,
                    ni=ni,
                    ns=ns,
                    shape=(Ny, Nx),
                    spacing=(dy, dx),
                    z=z,
                    pinhole_um=pinhole_um,
                    magnification=magnification,
                    disk_magnification=disk_magnification,
                    normalize=normalize,
                    aberrations=aberrations,
                    vectorial=vectorial,
                )

            # Center if requested (shift DC to center of volume)
            if center_output:
                psf = ifftshift(psf)

            # Store in ImageBuffer (5D: T, Z, C, Y, X)
            psf_5d = psf[np.newaxis, :, np.newaxis, :, :]

            from pyvistra.io import ImageBuffer
            self._buffer = ImageBuffer(
                shape=psf_5d.shape,
                dtype=np.float32,
                metadata=self._build_metadata(modality),
            )
            self._buffer[:] = psf_5d.astype(np.float32)

            self._metadata = self._build_metadata(modality)

            # Build output metadata
            output_meta = {
                "filename": "Computed PSF",
                "shape": self._buffer.shape,
                "scale": tuple(self._metadata["spacing"]),
                "is_rgb": False,
            }
            output_meta.update(self._metadata)

            # Send to selected output destination
            result = self.output_selector.send(self._buffer, output_meta)
            if result:
                if isinstance(result, str):
                    self.status_label.setText(f"Saved to {result}")
                else:
                    self.status_label.setText("PSF sent successfully")
                self.status_label.setStyleSheet("color: #4F4;")
            else:
                self.status_label.setText("PSF computed (output cancelled)")
                self.status_label.setStyleSheet("color: #888;")

            # Emit signal
            self.psf_computed.emit(self._buffer)

        except ImportError as e:
            self.status_label.setText("Error: deconlib not installed")
            self.status_label.setStyleSheet("color: #F44;")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #F44;")
        finally:
            self.compute_btn.setEnabled(True)

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
