from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


class TransformDialog(QDialog):
    """
    Dialog for adjusting image rotation and translation.
    Transforms the view without affecting ROIs.
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Transform Image")
        self.setWindowFlags(Qt.Tool)
        self.resize(320, 180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Window ID
        if hasattr(viewer, "window_id"):
            wid_label = QLabel(f"<b>Window: {viewer.window_id}</b>")
            wid_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(wid_label)

        # Rotation
        rot_group = QGroupBox("Rotation")
        rot_layout = QHBoxLayout(rot_group)

        rot_layout.addWidget(QLabel("Angle:"))
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-180.0, 180.0)
        self.rotation_spin.setSingleStep(0.5)
        self.rotation_spin.setDecimals(2)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.setValue(viewer.renderer.rotation_deg)
        self.rotation_spin.valueChanged.connect(self._on_rotation_changed)
        rot_layout.addWidget(self.rotation_spin)

        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(
            -1800, 1800
        )  # -180.0 to 180.0 in 0.1 increments
        self.rotation_slider.setValue(int(viewer.renderer.rotation_deg * 10))
        self.rotation_slider.valueChanged.connect(
            self._on_rotation_slider_changed
        )
        rot_layout.addWidget(self.rotation_slider)

        layout.addWidget(rot_group)

        # Translation
        trans_group = QGroupBox("Translation")
        trans_layout = QVBoxLayout(trans_group)

        # X translation
        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("X:"))
        self.translate_x_spin = QDoubleSpinBox()
        self.translate_x_spin.setRange(-10000.0, 10000.0)
        self.translate_x_spin.setSingleStep(1.0)
        self.translate_x_spin.setDecimals(1)
        self.translate_x_spin.setSuffix(" px")
        self.translate_x_spin.setValue(viewer.renderer.translate_x)
        self.translate_x_spin.valueChanged.connect(
            self._on_translate_x_changed
        )
        x_row.addWidget(self.translate_x_spin)
        trans_layout.addLayout(x_row)

        # Y translation
        y_row = QHBoxLayout()
        y_row.addWidget(QLabel("Y:"))
        self.translate_y_spin = QDoubleSpinBox()
        self.translate_y_spin.setRange(-10000.0, 10000.0)
        self.translate_y_spin.setSingleStep(1.0)
        self.translate_y_spin.setDecimals(1)
        self.translate_y_spin.setSuffix(" px")
        self.translate_y_spin.setValue(viewer.renderer.translate_y)
        self.translate_y_spin.valueChanged.connect(
            self._on_translate_y_changed
        )
        y_row.addWidget(self.translate_y_spin)
        trans_layout.addLayout(y_row)

        layout.addWidget(trans_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_transform)
        btn_layout.addWidget(reset_btn)

        self.apply_btn = QPushButton("Apply Transform")
        self.apply_btn.clicked.connect(self._apply_transform)
        btn_layout.addWidget(self.apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _on_rotation_changed(self, value):
        self.rotation_slider.blockSignals(True)
        self.rotation_slider.setValue(int(value * 10))
        self.rotation_slider.blockSignals(False)
        self.viewer.renderer.rotation_deg = value
        self.viewer.canvas.update()

    def _on_rotation_slider_changed(self, value):
        rot_deg = value / 10.0
        self.rotation_spin.blockSignals(True)
        self.rotation_spin.setValue(rot_deg)
        self.rotation_spin.blockSignals(False)
        self.viewer.renderer.rotation_deg = rot_deg
        self.viewer.canvas.update()

    def _on_translate_x_changed(self, value):
        self.viewer.renderer.translate_x = value
        self.viewer.canvas.update()

    def _on_translate_y_changed(self, value):
        self.viewer.renderer.translate_y = value
        self.viewer.canvas.update()

    def _reset_transform(self):
        self.viewer.renderer.reset_transform()
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

        self.viewer.canvas.update()

    def _apply_transform(self):
        """Bake current rotation/translation into image data."""
        from pyvistra.io import apply_transform

        rotation = self.rotation_spin.value()
        tx = self.translate_x_spin.value()
        ty = self.translate_y_spin.value()

        # Skip if no transform applied
        if rotation == 0 and tx == 0 and ty == 0:
            return

        # Disable button during processing
        self.apply_btn.setEnabled(False)

        try:
            # Create transformed buffer
            buffer = apply_transform(
                self.viewer.img_data,
                rotation,
                (tx, ty),
                metadata=self.viewer.meta.copy(),
            )

            # Switch viewer to use buffer
            self.viewer.img_data = buffer
            self.viewer.renderer.data = buffer
            self.viewer.meta = buffer.metadata

            # Reset visual transform (data is now transformed)
            self.viewer.renderer.reset_transform()

            # Reset UI controls
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

            # Refresh display
            self.viewer.update_view()

        finally:
            self.apply_btn.setEnabled(True)

    def refresh_ui(self):
        """Refresh UI to match current renderer state."""
        self.rotation_spin.blockSignals(True)
        self.rotation_slider.blockSignals(True)
        self.translate_x_spin.blockSignals(True)
        self.translate_y_spin.blockSignals(True)

        self.rotation_spin.setValue(self.viewer.renderer.rotation_deg)
        self.rotation_slider.setValue(
            int(self.viewer.renderer.rotation_deg * 10)
        )
        self.translate_x_spin.setValue(self.viewer.renderer.translate_x)
        self.translate_y_spin.setValue(self.viewer.renderer.translate_y)

        self.rotation_spin.blockSignals(False)
        self.rotation_slider.blockSignals(False)
        self.translate_x_spin.blockSignals(False)
        self.translate_y_spin.blockSignals(False)
