import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .output_selector import ImageOutputSelector

_MODE_2D = "2d"
_MODE_3D = "3d"

_KIND_REAL = "real"
_KIND_COMPLEX = "complex"

_DISPLAY_LOG_MAG = "log_mag"
_DISPLAY_MAG = "mag"
_DISPLAY_PHASE = "phase"


def _transform(data, kind, axes):
    """FFT ``data`` over ``axes``, shifted so DC sits at the array center.

    Real-input transforms (``rfftn``) only keep non-negative frequencies on
    the *last* transformed axis (Hermitian symmetry) -- that axis has no
    negative-frequency half to shift, so it's excluded from the fftshift.
    """
    if kind == _KIND_REAL:
        spectrum = np.fft.rfftn(data, axes=axes)
        shift_axes = axes[:-1]
        if shift_axes:
            spectrum = np.fft.fftshift(spectrum, axes=shift_axes)
    else:
        spectrum = np.fft.fftn(data, axes=axes)
        spectrum = np.fft.fftshift(spectrum, axes=axes)
    return spectrum


def _coerce_scale(scale):
    """Return ``scale`` as a ``(z, y, x)`` float triple, defaulting to 1.0."""
    if isinstance(scale, (list, tuple, np.ndarray)) and len(scale) >= 3:
        return float(scale[0]), float(scale[1]), float(scale[2])
    return 1.0, 1.0, 1.0


def _reduce(spectrum, display):
    if display == _DISPLAY_LOG_MAG:
        return np.log1p(np.abs(spectrum)).astype(np.float32, copy=False)
    elif display == _DISPLAY_MAG:
        return np.abs(spectrum).astype(np.float32, copy=False)
    else:
        return np.angle(spectrum).astype(np.float32, copy=False)


class FFTDialog(QDialog):
    """2D (per Z-slice) or 3D (Z-stack) FFT for inspecting frequency content.

    Real-input mode uses ``rfft``/``rfftn`` (compact, Hermitian-symmetric
    output); full-complex mode uses ``fft``/``fftn``. Either way the result
    is fftshifted so DC lands at the array center before display, since
    vispy can only render a real-valued image.
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("FFT")
        self.setWindowFlags(Qt.Tool)
        self.resize(360, 240)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("2D (per Z-slice)", _MODE_2D)
        self.mode_combo.addItem("3D (Z stack)", _MODE_3D)
        form.addRow("Mode:", self.mode_combo)

        self.kind_combo = QComboBox()
        self.kind_combo.addItem("Real input (rfft)", _KIND_REAL)
        self.kind_combo.addItem("Full complex (fft)", _KIND_COMPLEX)
        form.addRow("Transform:", self.kind_combo)

        self.display_combo = QComboBox()
        self.display_combo.addItem("Log magnitude", _DISPLAY_LOG_MAG)
        self.display_combo.addItem("Magnitude", _DISPLAY_MAG)
        self.display_combo.addItem("Phase", _DISPLAY_PHASE)
        form.addRow("Display:", self.display_combo)

        main_layout.addLayout(form)

        self.output_selector = ImageOutputSelector(
            default_title="FFT",
            formats=[".tif", ".ims"],
        )
        main_layout.addWidget(self.output_selector)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._compute)
        buttons.addWidget(self.compute_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        buttons.addStretch()
        main_layout.addLayout(buttons)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")
        main_layout.addWidget(self.status_label)

    def _set_error(self, message):
        self.status_label.setText(f"Error: {message}")
        self.status_label.setStyleSheet("color: #F44;")

    def _compute(self):
        T, Z, C, Y, X = self.viewer.img_data.shape
        mode = self.mode_combo.currentData()
        kind = self.kind_combo.currentData()
        display = self.display_combo.currentData()

        if mode == _MODE_3D and Z < 2:
            self._set_error("3D mode needs Z >= 2")
            return

        x_out = X // 2 + 1 if kind == _KIND_REAL else X
        out_shape = (T, Z, C, Y, x_out)

        self.status_label.setText("Computing...")
        self.status_label.setStyleSheet("color: #888;")
        self.compute_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            from pyvistra.io import ImageBuffer

            source = self.viewer.img_data
            out_meta = dict(self.viewer.meta or {})
            out_meta["shape"] = out_shape
            base = str(out_meta.get("filename", out_meta.get("name", "Image")))
            out_meta["filename"] = f"{base}_fft_{mode}_{kind}"
            out_meta["name"] = out_meta["filename"]

            # Pixels in the output are spatial-frequency samples, not
            # distance -- flag the space and replace "scale" with the DFT's
            # frequency spacing (1 / (N * pixel_size)) per transformed axis
            # so unit-aware consumers (line/radial profile) label and
            # convert correctly instead of treating it as real-space um.
            sz, sy, sx = _coerce_scale(out_meta.get("scale"))
            fy, fx = 1.0 / (Y * sy), 1.0 / (X * sx)
            if mode == _MODE_3D:
                fz = 1.0 / (Z * sz)
                dc_index = (Z // 2, Y // 2, 0 if kind == _KIND_REAL else X // 2)
            else:
                fz = sz  # Z axis isn't transformed in 2D mode -- keep real space
                dc_index = (None, Y // 2, 0 if kind == _KIND_REAL else X // 2)
            out_meta["scale"] = (fz, fy, fx)
            out_meta["space"] = "frequency"
            out_meta["fft"] = {
                "mode": mode,
                "transform": kind,
                "display": display,
                "origin": "center (fftshifted)",
                "source_scale": (sz, sy, sx),
                "dc_index": dc_index,
            }

            buffer = ImageBuffer(shape=out_shape, dtype=np.float32, metadata=out_meta)

            if mode == _MODE_2D:
                for t in range(T):
                    for c in range(C):
                        for z in range(Z):
                            plane = np.asarray(source[t, z, c, :, :]).astype(
                                np.float32, copy=False
                            )
                            spectrum = _transform(plane, kind, axes=(-2, -1))
                            buffer[t, z, c, :, :] = _reduce(spectrum, display)
            else:
                for t in range(T):
                    for c in range(C):
                        vol = np.asarray(source[t, :, c, :, :]).astype(
                            np.float32, copy=False
                        )
                        spectrum = _transform(vol, kind, axes=(-3, -2, -1))
                        buffer[t, :, c, :, :] = _reduce(spectrum, display)

            sent = self.output_selector.send(buffer, out_meta)
            if sent is not None:
                self.status_label.setText("Done")
                self.status_label.setStyleSheet("color: #4A4;")
            else:
                self.status_label.setText("Computed (output cancelled)")
                self.status_label.setStyleSheet("color: #888;")
        except Exception as exc:
            self._set_error(str(exc))
        finally:
            self.compute_btn.setEnabled(True)
