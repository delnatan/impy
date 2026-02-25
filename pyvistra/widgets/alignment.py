from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


class AlignmentDialog(QDialog):
    """
    Dialog for aligning two images via overlay.
    User selects a reference and query image, then adjusts rotation/translation
    to align them. The query image is overlaid on the reference with adjustable opacity.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Align Images")
        self.setWindowFlags(Qt.Tool)
        self.resize(380, 320)

        # Import manager here to avoid circular imports
        from pyvistra.ui.manager import manager

        self.manager = manager

        self._overlay_layers = []  # Track overlay layers for cleanup
        self._reference_window = None
        self._query_window = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Image Selection
        select_group = QGroupBox("Image Selection")
        select_layout = QVBoxLayout(select_group)

        # Reference image
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference:"))
        self.ref_combo = QComboBox()
        self.ref_combo.currentIndexChanged.connect(self._on_reference_changed)
        ref_row.addWidget(self.ref_combo, 1)
        select_layout.addLayout(ref_row)

        # Query image
        query_row = QHBoxLayout()
        query_row.addWidget(QLabel("Query:"))
        self.query_combo = QComboBox()
        self.query_combo.currentIndexChanged.connect(self._on_query_changed)
        query_row.addWidget(self.query_combo, 1)
        select_layout.addLayout(query_row)

        layout.addWidget(select_group)

        # Transform Controls
        transform_group = QGroupBox("Query Transform")
        transform_layout = QVBoxLayout(transform_group)

        # Rotation
        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rotation:"))
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-180.0, 180.0)
        self.rotation_spin.setSingleStep(0.5)
        self.rotation_spin.setDecimals(2)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.valueChanged.connect(self._on_transform_changed)
        rot_row.addWidget(self.rotation_spin)

        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-1800, 1800)
        self.rotation_slider.valueChanged.connect(
            self._on_rotation_slider_changed
        )
        rot_row.addWidget(self.rotation_slider)
        transform_layout.addLayout(rot_row)

        # X translation
        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("X Offset:"))
        self.translate_x_spin = QDoubleSpinBox()
        self.translate_x_spin.setRange(-10000.0, 10000.0)
        self.translate_x_spin.setSingleStep(1.0)
        self.translate_x_spin.setDecimals(1)
        self.translate_x_spin.setSuffix(" px")
        self.translate_x_spin.valueChanged.connect(self._on_transform_changed)
        x_row.addWidget(self.translate_x_spin)
        transform_layout.addLayout(x_row)

        # Y translation
        y_row = QHBoxLayout()
        y_row.addWidget(QLabel("Y Offset:"))
        self.translate_y_spin = QDoubleSpinBox()
        self.translate_y_spin.setRange(-10000.0, 10000.0)
        self.translate_y_spin.setSingleStep(1.0)
        self.translate_y_spin.setDecimals(1)
        self.translate_y_spin.setSuffix(" px")
        self.translate_y_spin.valueChanged.connect(self._on_transform_changed)
        y_row.addWidget(self.translate_y_spin)
        transform_layout.addLayout(y_row)

        layout.addWidget(transform_group)

        # Opacity Control
        opacity_group = QGroupBox("Overlay Opacity")
        opacity_layout = QHBoxLayout(opacity_group)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)

        self.opacity_label = QLabel("50%")
        self.opacity_label.setFixedWidth(40)
        opacity_layout.addWidget(self.opacity_label)

        layout.addWidget(opacity_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_transform)
        btn_layout.addWidget(reset_btn)

        apply_btn = QPushButton("Apply to Query")
        apply_btn.setToolTip(
            "Apply the current transform to the query window's view"
        )
        apply_btn.clicked.connect(self._apply_to_query)
        btn_layout.addWidget(apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._close_dialog)
        btn_layout.addWidget(close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Populate dropdowns
        self._refresh_window_list()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_window_list()

    def closeEvent(self, event):
        self._remove_overlay()
        super().closeEvent(event)

    def _close_dialog(self):
        self._remove_overlay()
        self.close()

    def _refresh_window_list(self):
        """Refresh the dropdown lists with current windows."""
        self.ref_combo.blockSignals(True)
        self.query_combo.blockSignals(True)

        self.ref_combo.clear()
        self.query_combo.clear()

        self.ref_combo.addItem("-- Select Reference --", None)
        self.query_combo.addItem("-- Select Query --", None)

        for wid, window in self.manager.get_all().items():
            title = window.windowTitle()
            display_name = f"[{wid}] {title[:40]}"
            self.ref_combo.addItem(display_name, wid)
            self.query_combo.addItem(display_name, wid)

        self.ref_combo.blockSignals(False)
        self.query_combo.blockSignals(False)

    def _on_reference_changed(self, index):
        wid = self.ref_combo.currentData()
        if wid is not None:
            self._reference_window = self.manager.get(wid)
        else:
            self._reference_window = None
        self._update_overlay()

    def _on_query_changed(self, index):
        wid = self.query_combo.currentData()
        if wid is not None:
            self._query_window = self.manager.get(wid)
        else:
            self._query_window = None
        self._update_overlay()

    def _on_rotation_slider_changed(self, value):
        rot_deg = value / 10.0
        self.rotation_spin.blockSignals(True)
        self.rotation_spin.setValue(rot_deg)
        self.rotation_spin.blockSignals(False)
        self._update_overlay_transform()

    def _on_transform_changed(self):
        # Sync slider with spinbox
        self.rotation_slider.blockSignals(True)
        self.rotation_slider.setValue(int(self.rotation_spin.value() * 10))
        self.rotation_slider.blockSignals(False)
        self._update_overlay_transform()

    def _on_opacity_changed(self, value):
        self.opacity_label.setText(f"{value}%")
        self._update_overlay_opacity()

    def _remove_overlay(self):
        """Remove any existing overlay layers."""
        for layer in self._overlay_layers:
            try:
                layer.parent = None
            except Exception:
                pass
        self._overlay_layers = []
        if self._reference_window:
            self._reference_window.canvas.update()

    def _update_overlay(self):
        """Create or update the overlay of query on reference."""
        self._remove_overlay()

        if not self._reference_window or not self._query_window:
            return

        if self._reference_window is self._query_window:
            return  # Same window, no overlay needed

        # Get query image data (current slice)
        query_cache = self._query_window.renderer.current_slice_cache
        if query_cache is None:
            return

        # Create overlay layers in reference window
        from vispy import scene
        from vispy.visuals.transforms.linear import (
            MatrixTransform,
            STTransform,
        )

        opacity = self.opacity_slider.value() / 100.0

        # Add each channel from query as an overlay
        for c in range(query_cache.shape[0]):
            plane = query_cache[c]

            # Get colormap from query
            cmap_name = self._query_window.renderer.get_colormap_name(c)
            from pyvistra.visuals import get_colormap

            cmap, _ = get_colormap(cmap_name)

            # Create image visual
            overlay = scene.visuals.Image(
                plane,
                cmap=cmap,
                clim=self._query_window.renderer.get_clim(c),
                parent=self._reference_window.view.scene,
                method="auto",
                interpolation="nearest",
            )

            # Inherit gamma from query window's contrast settings
            overlay.gamma = self._query_window.renderer.get_gamma(c)

            # Set blending for overlay
            overlay.set_gl_state(
                preset="translucent",
                blend=True,
                blend_func=("src_alpha", "one_minus_src_alpha"),
                depth_test=False,
            )
            overlay.opacity = opacity
            overlay.order = 100 + c  # Render on top

            # Apply transform
            overlay.transform = self._build_overlay_transform()

            self._overlay_layers.append(overlay)

        self._reference_window.canvas.update()

    def _build_overlay_transform(self):
        """Build transform for overlay layers."""

        from vispy.visuals.transforms.linear import (
            MatrixTransform,
            STTransform,
        )

        if not self._query_window:
            return STTransform()

        # Get query image dimensions
        _, _, _, Y, X = self._query_window.img_data.shape
        sy, sx = self._query_window.renderer.scale

        # Image center in scaled coordinates
        cx = X * sx / 2
        cy = Y * sy / 2

        rot_deg = self.rotation_spin.value()
        tx = self.translate_x_spin.value()
        ty = self.translate_y_spin.value()

        if rot_deg == 0.0 and tx == 0.0 and ty == 0.0:
            return STTransform(scale=(sx, sy))

        # Build transform for rotation around image center:
        # 1. Scale, 2. Translate center to origin, 3. Rotate, 4. Translate back + offset
        transform = MatrixTransform()
        transform.scale((sx, sy, 1))
        transform.translate((-cx, -cy, 0))
        transform.rotate(rot_deg, (0, 0, 1))
        transform.translate((cx + tx, cy + ty, 0))

        return transform

    def _update_overlay_transform(self):
        """Update transform on existing overlay layers."""
        if not self._overlay_layers:
            return

        transform = self._build_overlay_transform()
        for layer in self._overlay_layers:
            layer.transform = transform

        if self._reference_window:
            self._reference_window.canvas.update()

    def _update_overlay_opacity(self):
        """Update opacity on existing overlay layers."""
        opacity = self.opacity_slider.value() / 100.0
        for layer in self._overlay_layers:
            layer.opacity = opacity

        if self._reference_window:
            self._reference_window.canvas.update()

    def _reset_transform(self):
        """Reset transform controls to zero."""
        self.rotation_spin.blockSignals(True)
        self.rotation_slider.blockSignals(True)
        self.translate_x_spin.blockSignals(True)
        self.translate_y_spin.blockSignals(True)

        self.rotation_spin.setValue(0.0)
        self.rotation_slider.setValue(0)
        self.translate_x_spin.setValue(0.0)
        self.translate_y_spin.setValue(0.0)

        self.rotation_spin.blockSignals(False)
        self.rotation_slider.blockSignals(False)
        self.translate_x_spin.blockSignals(False)
        self.translate_y_spin.blockSignals(False)

        self._update_overlay_transform()

    def _apply_to_query(self):
        """Apply the current transform settings to the query window's renderer."""
        if not self._query_window:
            return

        self._query_window.renderer.set_transform(
            rotation_deg=self.rotation_spin.value(),
            translate_x=self.translate_x_spin.value(),
            translate_y=self.translate_y_spin.value(),
        )
        self._query_window.canvas.update()

        # Remove overlay since transform is now applied
        self._remove_overlay()
