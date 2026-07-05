"""Shared "Source, PSF && Model" tab for the NLCG, Richardson-Lucy, and
MaxEnt (memsolve) deconvolution dialogs.

`SourcePSFPanelMixin` is a mixin (no `__init__` of its own) providing the
detector/data region picker, PSF combo + scale-match + Compute-PSF spawn,
and the forward-model super-res zoom spinboxes -- the part of every
deconvolution engine's UI that is identical regardless of solver, so none
of the three engine dialogs (NLCG, Richardson-Lucy, MaxEnt/memsolve) needs
its own copy of the same ~500 lines of region/PSF plumbing.

A host dialog using this mixin must provide, before calling
`_init_source_psf_panel()`:
    * `self.viewer` -- the source `ImageWindow`.
    * `self._set_status(msg, *, ok=False, warn=False, error=False)`.
    * `self._on_source_psf_changed()` -- called whenever the region, PSF
      selection, or zoom factors change; the host uses this to refresh its
      own recipe preview / diagnostics labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pyvistra.ui.manager import manager
from pyvistra.data.shapes import ALL_FRAMES, RECTANGLE, rectangle_bounds

from .decon_common import compact_psf_shape_for_data


@dataclass(frozen=True)
class CroppedSource:
    """Result of cropping the viewer/PSF window down to solver inputs."""

    y_obs: np.ndarray
    psf_data: np.ndarray
    psf_pixel_size: Tuple[float, ...]
    t: int
    c: int


class SourcePSFPanelMixin:
    """Mixin providing the Source/PSF/Model tab. See module docstring."""

    # ------------------------------------------------------------------ #
    # Setup

    def _init_source_psf_panel(self) -> None:
        """Create timers/subscriptions. Call once, early in `__init__`."""
        self._rect_shapes: list = []
        self._shape_layer_unsubs: dict = {}
        self._shape_refresh_timer = QTimer(self)
        self._shape_refresh_timer.setSingleShot(True)
        self._shape_refresh_timer.setInterval(150)
        self._shape_refresh_timer.timeout.connect(self._refresh_shape_combo)
        self.viewer.layer_added.connect(self._schedule_shape_refresh)
        self.viewer.layer_removed.connect(self._schedule_shape_refresh)

        self._psf_refresh_timer = QTimer(self)
        self._psf_refresh_timer.setSingleShot(True)
        self._psf_refresh_timer.setInterval(150)
        self._psf_refresh_timer.timeout.connect(self._refresh_psf_combo)
        manager.window_registered.connect(self._schedule_psf_refresh)
        manager.window_unregistered.connect(self._schedule_psf_refresh)

    def _finish_source_psf_panel_init(self) -> None:
        """Populate combos. Call once, after all tabs/widgets exist."""
        self._refresh_shape_combo()
        self._refresh_psf_combo()

    # ------------------------------------------------------------------ #
    # Tab builder

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
            lambda _v: self._on_source_psf_changed()
        )
        ch_t_layout.addWidget(self.channel_spin)
        ch_t_layout.addWidget(QLabel("Frames T:"))
        t_default = min(max(0, getattr(self.viewer, "t_idx", 0)), max(0, T - 1))
        self.t_start_spin = QSpinBox()
        self.t_start_spin.setRange(0, max(0, T - 1))
        self.t_start_spin.setValue(t_default)
        self.t_start_spin.setFixedWidth(52)
        self.t_end_spin = QSpinBox()
        self.t_end_spin.setRange(0, max(0, T - 1))
        self.t_end_spin.setValue(t_default)
        self.t_end_spin.setFixedWidth(52)
        self.t_end_spin.setToolTip(
            "Deconvolution runs once per frame in [start, end], sequentially, "
            "writing each result into its own frame of the output."
        )
        self.t_start_spin.valueChanged.connect(self._on_t_start_changed)
        self.t_end_spin.valueChanged.connect(self._on_t_end_changed)
        ch_t_layout.addWidget(self.t_start_spin)
        ch_t_layout.addWidget(QLabel("to"))
        ch_t_layout.addWidget(self.t_end_spin)
        ch_t_layout.addSpacing(10)
        ch_t_layout.addWidget(QLabel("Background:"))
        self.background_spin = QDoubleSpinBox()
        self.background_spin.setRange(0.0, 1e6)
        self.background_spin.setDecimals(2)
        self.background_spin.setSingleStep(1.0)
        self.background_spin.setValue(0.0)
        self.background_spin.setFixedWidth(72)
        self.background_spin.setToolTip(
            "Constant detector background (camera offset + ambient counts) "
            "for the selected channel. This is channel-dependent — check it "
            "when switching channels."
        )
        self.background_spin.valueChanged.connect(
            lambda _v: self._on_source_psf_changed()
        )
        ch_t_layout.addWidget(self.background_spin)
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
        z_layout.addWidget(QLabel("center:"))
        self.z_center_spin = QSpinBox()
        self.z_center_spin.setRange(0, 0)
        self.z_center_spin.setFixedWidth(48)
        self.z_center_spin.setEnabled(False)
        self.z_center_spin.setToolTip(
            "Z-plane the crop window is centered on. Independent of any "
            "shape's own t/z anchor -- defaults to the viewer's current Z "
            "when a shape is selected, but is always explicit and editable "
            "here."
        )
        z_layout.addWidget(self.z_center_spin)
        z_layout.addStretch()
        z_row.setVisible(False)
        self._z_row_widget = z_row
        region_v.addWidget(z_row)

        self._z_group = QButtonGroup(self)
        self._z_group.addButton(self.all_z_radio)
        self._z_group.addButton(self.around_z_radio)
        self.around_z_radio.toggled.connect(self.z_half_spin.setEnabled)
        self.around_z_radio.toggled.connect(self.z_center_spin.setEnabled)
        self.around_z_radio.toggled.connect(
            lambda *_: self._on_source_psf_changed()
        )
        self.all_z_radio.toggled.connect(
            lambda *_: self._on_source_psf_changed()
        )
        self.z_half_spin.valueChanged.connect(
            lambda *_: self._on_source_psf_changed()
        )
        self.z_center_spin.valueChanged.connect(
            lambda *_: self._on_source_psf_changed()
        )

        cf_form.addRow("Region:", region_w)
        vbox.addWidget(cf_grp)

        # Forward model — zoom (super-resolution) factors. Lives here (not
        # in the Solver tab) because PSF sampling-match below depends on it.
        _T, Z_dim, _C, _Y, _X = self.viewer.img_data.shape
        is_3d = Z_dim > 1

        fm_grp = QGroupBox("Forward model")
        fm_form = QFormLayout(fm_grp)
        fm_form.setLabelAlignment(Qt.AlignRight)
        fm_form.setVerticalSpacing(4)
        fm_form.setContentsMargins(8, 6, 8, 6)

        sr_row = QWidget()
        sr_layout = QHBoxLayout(sr_row)
        sr_layout.setContentsMargins(0, 0, 0, 0)
        sr_layout.setSpacing(6)
        sr_layout.addWidget(QLabel("XY ×:"))
        self.sr_xy_spin = QDoubleSpinBox()
        self.sr_xy_spin.setRange(1.0, 8.0)
        self.sr_xy_spin.setDecimals(3)
        self.sr_xy_spin.setSingleStep(0.25)
        self.sr_xy_spin.setValue(1.0)
        self.sr_xy_spin.setFixedWidth(72)
        self.sr_xy_spin.valueChanged.connect(self._on_sr_changed)
        self.sr_xy_spin.setToolTip(
            "Fine-grid reconstruction factor in X/Y. 1 keeps native camera "
            "sampling; fractional values use detector-area integration."
        )
        sr_layout.addWidget(self.sr_xy_spin)
        sr_layout.addSpacing(10)
        sr_layout.addWidget(QLabel("Z ×:"))
        self.sr_z_spin = QDoubleSpinBox()
        self.sr_z_spin.setRange(1.0, 8.0)
        self.sr_z_spin.setDecimals(3)
        self.sr_z_spin.setSingleStep(0.25)
        self.sr_z_spin.setValue(1.0)
        self.sr_z_spin.setFixedWidth(72)
        self.sr_z_spin.valueChanged.connect(self._on_sr_changed)
        self.sr_z_spin.setToolTip(
            "Axial fine-grid ratio for 3D data. Disabled for single-plane data."
        )
        if not is_3d:
            self.sr_z_spin.setEnabled(False)
            self.sr_z_spin.setToolTip(
                "Image has a single Z plane — axial super-res has no effect."
            )
        sr_layout.addWidget(self.sr_z_spin)
        sr_layout.addStretch()
        fm_form.addRow("Super-res:", sr_row)
        vbox.addWidget(fm_grp)

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
            "pre-filled for the current fine-grid sampling. PSF support is "
            "editable and does not need to match the object canvas."
        )
        self.compute_psf_btn.clicked.connect(self._spawn_psf_dialog)
        compute_layout.addWidget(self.compute_psf_btn)
        self.compute_target_label = QLabel("")
        self.compute_target_label.setStyleSheet("color: #888; font-size: 10px;")
        compute_layout.addWidget(self.compute_target_label, 1)
        psf_form.addRow("", compute_row)

        vbox.addWidget(psf_grp)
        vbox.addStretch()

        return tab

    # ------------------------------------------------------------------ #
    # Collapsible-group helper (shared by any "Advanced knobs" group)

    def _toggle_group_children(self, group: QGroupBox, checked: bool) -> None:
        for child in group.findChildren(QWidget):
            if child is not group:
                child.setVisible(bool(checked))
        group.setMaximumHeight(16777215 if checked else 24)

    # ------------------------------------------------------------------ #
    # Shape helpers

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
                z_center = self.z_center_spin.value()
                n_z = min(Z, z_center + half + 1) - max(0, z_center - half)
            else:
                n_z = Z
        if n_z <= 1:
            return (n_y, n_x)
        return (n_z, n_y, n_x)

    def _on_t_start_changed(self, value: int) -> None:
        if value > self.t_end_spin.value():
            self.t_end_spin.setValue(value)
        self._on_source_psf_changed()

    def _on_t_end_changed(self, value: int) -> None:
        if value < self.t_start_spin.value():
            self.t_start_spin.setValue(value)
        self._on_source_psf_changed()

    def _frame_range(self) -> range:
        return range(self.t_start_spin.value(), self.t_end_spin.value() + 1)

    def _current_psf_kernel_shape(self) -> Optional[tuple]:
        win = self.psf_combo.currentData()
        if win is None:
            return None
        _T, Zp, _C, Yp, Xp = win.img_data.shape
        if self.viewer.img_data.shape[1] <= 1 or Zp <= 1:
            return (Yp, Xp)
        return (Zp, Yp, Xp)

    def _format_shape(self, shape: Optional[tuple]) -> str:
        if shape is None:
            return "—"
        return " × ".join(str(int(v)) for v in shape)

    def _on_sr_changed(self, _value):
        self._on_source_psf_changed()
        win = self.psf_combo.currentData()
        if win is not None:
            self._check_scale_match(win)

    # ------------------------------------------------------------------ #
    # PSF compute spawn

    def _expected_psf_shape_spacing(self):
        """Return (shape, spacing) for the PSF compute preset.

        Both are returned in `(Nz, Ny, Nx)` / `(dz, dy, dx)` order when 3D,
        falling back to 2D `(Ny, Nx)` / `(dy, dx)` when the ROI has a
        single Z plane.
        """
        data_shape = self._current_data_shape()
        if data_shape is None:
            return None, None
        if len(data_shape) == 2:
            n_y, n_x = data_shape
        else:
            n_z, n_y, n_x = data_shape

        f_xy = max(float(self.sr_xy_spin.value()), 1.0)
        f_z = max(float(self.sr_z_spin.value()), 1.0)
        scale = self.viewer.meta.get("scale", (1.0, 1.0, 1.0))
        dz, dy, dx = float(scale[0]), float(scale[1]), float(scale[2])
        dy /= f_xy
        dx /= f_xy
        dz /= f_z

        if len(data_shape) == 2:
            shape = compact_psf_shape_for_data((n_y, n_x), (f_xy, f_xy))
            return shape, (dy, dx)
        shape = compact_psf_shape_for_data((n_z, n_y, n_x), (f_z, f_xy, f_xy))
        return shape, (dz, dy, dx)

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
    # PSF / shape combo refresh

    def _refresh_psf_combo(self):
        """List every open window as a candidate PSF.

        Not restricted to windows flagged ``is_psf`` (e.g. computed via the
        Compute PSF dialog) -- a PSF loaded from disk into a plain window is
        just as valid. Windows that *are* flagged are annotated so they're
        still easy to spot.
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
        if win is None:
            self.psf_channel_spin.setRange(0, 0)
            self.scale_label.setText("No PSF window")
            self.scale_label.setStyleSheet("color: #F44; font-size: 10px;")
            self._on_source_psf_changed()
            return
        _T, _Zp, C, _Yp, _Xp = win.img_data.shape
        self.psf_channel_spin.setRange(0, max(0, C - 1))
        self._check_scale_match(win)
        self._on_source_psf_changed()

    def _check_scale_match(self, psf_win):
        """Compare PSF spacing against the expected fine-grid spacing.

        The PSF is computed at fine-grid sampling — i.e. ``src_scale``
        divided by the per-axis zoom factor — and for a 2D image only the
        lateral axes participate. Both effects are accounted for here so
        the ✓/✗ indicator reflects the same axes the forward model uses.
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

    def _schedule_shape_refresh(self, *_args) -> None:
        self._shape_refresh_timer.start()

    def _schedule_psf_refresh(self, *_args) -> None:
        self._psf_refresh_timer.start()

    def _sync_shape_subscriptions(self) -> None:
        """Keep exactly one data-subscription per live shape layer.

        Shapes drawn into an *existing* layer don't emit ``layer_added``
        -- only ``ShapeData.subscribe`` sees those -- so this re-syncs on
        every combo refresh (including the ones triggered by
        ``layer_added``/``layer_removed`` themselves), ensuring a
        freshly-added layer gets its own subscription before shapes are
        drawn into it.
        """
        current = {l.name: l for l in self.viewer.layers.by_type("shapes")}
        for name in list(self._shape_layer_unsubs):
            if name not in current:
                self._shape_layer_unsubs.pop(name)()
        for name, layer in current.items():
            if name not in self._shape_layer_unsubs:
                self._shape_layer_unsubs[name] = layer.data.subscribe(
                    self._schedule_shape_refresh
                )

    def _refresh_shape_combo(self):
        self._sync_shape_subscriptions()
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
        self._on_source_psf_changed()

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
        t_sync = (
            rec.t if rec.t != ALL_FRAMES
            else min(max(0, getattr(self.viewer, "t_idx", 0)), max(0, _T - 1))
        )
        self.t_start_spin.setValue(t_sync)
        self.t_end_spin.setValue(t_sync)
        self.z_center_spin.setRange(0, max(0, Z - 1))
        self.z_center_spin.setValue(
            min(max(0, getattr(self.viewer, "z_idx", 0)), max(0, Z - 1))
        )
        self._on_source_psf_changed()

    # ------------------------------------------------------------------ #
    # Observation / PSF cropping (shared `_prepare()` step)

    def _crop_observation_and_psf(self, t: Optional[int] = None) -> Optional[CroppedSource]:
        """Read region + PSF widgets and return cropped numeric inputs.

        `t` selects the frame to crop; defaults to `t_start_spin` (the
        first/only frame) when omitted. Returns ``None`` on a user error,
        after calling ``self._set_status(...)`` to explain it.
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
        if not psf_win.meta.get("psf_dc_corner", True):
            psf_data = np.fft.fftshift(psf_data)

        y_obs = np.ascontiguousarray(
            self.viewer.img_data[t, z_slice, c, y_slice, x_slice]
        ).astype(np.float32)
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

        return CroppedSource(
            y_obs=y_obs,
            psf_data=psf_data,
            psf_pixel_size=psf_pixel_size,
            t=t,
            c=c,
        )
