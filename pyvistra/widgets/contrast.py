import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from pyvistra.visuals import COLORMAPS
from .histogram import HistogramWidget, configure_spinbox_for_range


class ContrastDialog(QDialog):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Brightness / Contrast")
        self.resize(450, 280)
        self.setWindowFlags(Qt.Tool)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # 0. Window ID Label
        if hasattr(viewer, "window_id"):
            wid_label = QLabel(f"<b>Window ID: {viewer.window_id}</b>")
            wid_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(wid_label)

        # 1. Channel Selector
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Channel:"))
        self.combo = QComboBox()
        self.combo.addItems([f"Channel {i + 1}" for i in range(viewer.C)])
        self.combo.currentIndexChanged.connect(self.refresh_ui)
        row1.addWidget(self.combo)
        layout.addLayout(row1)

        # 1b. Colormap Selector
        row_cmap = QHBoxLayout()
        row_cmap.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(list(COLORMAPS.keys()))
        self.cmap_combo.currentTextChanged.connect(self.on_colormap_changed)
        row_cmap.addWidget(self.cmap_combo)
        layout.addLayout(row_cmap)

        # 2. Interactive Histogram with Min/Max Spinboxes
        hist_layout = QHBoxLayout()
        hist_layout.setSpacing(6)

        # Min spinbox
        min_label = QLabel("Min:")
        min_label.setStyleSheet("color: #AAA; font-size: 10px;")
        hist_layout.addWidget(min_label)

        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(1)
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setSingleStep(10)
        self.min_spin.setFixedWidth(75)
        self.min_spin.setToolTip("Minimum intensity")
        self.min_spin.valueChanged.connect(self.on_min_spin_changed)
        hist_layout.addWidget(self.min_spin)

        self.hist_widget = HistogramWidget()
        self.hist_widget.climChanged.connect(self.on_histogram_clim_changed)
        hist_layout.addWidget(self.hist_widget, 1)

        # Max spinbox
        max_label = QLabel("Max:")
        max_label.setStyleSheet("color: #AAA; font-size: 10px;")
        hist_layout.addWidget(max_label)

        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(1)
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setSingleStep(10)
        self.max_spin.setFixedWidth(75)
        self.max_spin.setToolTip("Maximum intensity")
        self.max_spin.valueChanged.connect(self.on_max_spin_changed)
        hist_layout.addWidget(self.max_spin)

        layout.addLayout(hist_layout)

        # 3. Gamma Control
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(1, 400)  # 0.01 to 4.00
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.on_gamma_slider_changed)
        gamma_layout.addWidget(self.gamma_slider)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 4.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.valueChanged.connect(self.on_gamma_spin_changed)
        gamma_layout.addWidget(self.gamma_spin)

        layout.addLayout(gamma_layout)

        # 4. Auto/Manual Contrast Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_loosen = QPushButton("-")
        self.btn_loosen.setToolTip("Loosen Contrast (Expand Range)")
        self.btn_loosen.setFixedWidth(30)
        self.btn_loosen.clicked.connect(lambda: self.adjust_contrast(-1))
        btn_layout.addWidget(self.btn_loosen)

        self.btn_auto = QPushButton("Auto Contrast")
        self.btn_auto.setCursor(Qt.PointingHandCursor)
        self.btn_auto.clicked.connect(self.reset_auto_contrast)
        btn_layout.addWidget(self.btn_auto)

        self.btn_tighten = QPushButton("+")
        self.btn_tighten.setToolTip("Tighten Contrast (Shrink Range)")
        self.btn_tighten.setFixedWidth(30)
        self.btn_tighten.clicked.connect(lambda: self.adjust_contrast(1))
        btn_layout.addWidget(self.btn_tighten)

        self.chk_all_channels = QCheckBox("All Channels")
        self.chk_all_channels.setChecked(True)
        btn_layout.addWidget(self.chk_all_channels)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # State for auto-contrast
        self.pct_low = 0.2
        self.pct_high = 99.98

        # Initial Load
        self.refresh_ui()

    def refresh_ui(self):
        c_idx = self.combo.currentIndex()
        cache = self.viewer.renderer.current_slice_cache
        if cache is None:
            return

        if c_idx < cache.shape[0]:
            plane = cache[c_idx]

            # Update Colormap Dropdown
            cmap_name = self.viewer.renderer.get_colormap_name(c_idx)
            self.cmap_combo.blockSignals(True)
            idx = self.cmap_combo.findText(cmap_name)
            if idx >= 0:
                self.cmap_combo.setCurrentIndex(idx)
            self.cmap_combo.blockSignals(False)

            # Update Histogram Data
            color = self.viewer.renderer.channel_colors[c_idx % 6]
            self.hist_widget.set_data(plane, color)

            # Configure spinboxes for the data range (adaptive precision)
            data_min = self.hist_widget.data_min
            data_max = self.hist_widget.data_max
            configure_spinbox_for_range(self.min_spin, data_min, data_max)
            configure_spinbox_for_range(self.max_spin, data_min, data_max)

            # Get current clim from renderer
            curr_min, curr_max = self.viewer.renderer.get_clim(c_idx)

            # Update histogram and spinboxes without triggering signal loop
            self.block_clim_signals(True)
            self.hist_widget.set_clim(curr_min, curr_max)
            self.min_spin.setValue(curr_min)
            self.max_spin.setValue(curr_max)
            self.block_clim_signals(False)

            # Update Gamma
            gamma = self.viewer.renderer.get_gamma(c_idx)
            self.block_gamma_signals(True)
            self.gamma_spin.setValue(gamma)
            self.gamma_slider.setValue(int(gamma * 100))
            self.block_gamma_signals(False)

    def block_clim_signals(self, block):
        """Block or unblock signals from clim-related widgets."""
        self.hist_widget.blockSignals(block)
        self.min_spin.blockSignals(block)
        self.max_spin.blockSignals(block)

    def block_gamma_signals(self, block):
        self.gamma_slider.blockSignals(block)
        self.gamma_spin.blockSignals(block)

    def on_gamma_slider_changed(self, val):
        gamma = val / 100.0
        self.gamma_spin.blockSignals(True)
        self.gamma_spin.setValue(gamma)
        self.gamma_spin.blockSignals(False)
        self.update_gamma(gamma)

    def on_gamma_spin_changed(self, val):
        self.gamma_slider.blockSignals(True)
        self.gamma_slider.setValue(int(val * 100))
        self.gamma_slider.blockSignals(False)
        self.update_gamma(val)

    def update_gamma(self, gamma):
        c_idx = self.combo.currentIndex()
        self.viewer.renderer.set_gamma(c_idx, gamma)
        self.viewer.canvas.update()

    def on_histogram_clim_changed(self, vmin, vmax):
        """Handle histogram clim change (from dragging handles)."""
        # Update spinboxes
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        # Update renderer
        c_idx = self.combo.currentIndex()
        self.viewer.renderer.set_clim(c_idx, vmin, vmax)
        self.viewer.canvas.update()

    def on_min_spin_changed(self, value):
        """Handle min spinbox change."""
        max_val = self.max_spin.value()
        if value < max_val:
            # Update histogram
            self.hist_widget.blockSignals(True)
            self.hist_widget.set_clim(value, max_val)
            self.hist_widget.blockSignals(False)
            # Update renderer
            c_idx = self.combo.currentIndex()
            self.viewer.renderer.set_clim(c_idx, value, max_val)
            self.viewer.canvas.update()

    def on_max_spin_changed(self, value):
        """Handle max spinbox change."""
        min_val = self.min_spin.value()
        if value > min_val:
            # Update histogram
            self.hist_widget.blockSignals(True)
            self.hist_widget.set_clim(min_val, value)
            self.hist_widget.blockSignals(False)
            # Update renderer
            c_idx = self.combo.currentIndex()
            self.viewer.renderer.set_clim(c_idx, min_val, value)
            self.viewer.canvas.update()

    def on_colormap_changed(self, cmap_name):
        c_idx = self.combo.currentIndex()
        self.viewer.renderer.set_colormap(c_idx, cmap_name)
        self.viewer.canvas.update()

        # Update histogram color to match new colormap
        color = self.viewer.renderer.channel_colors[c_idx % 6]
        cache = self.viewer.renderer.current_slice_cache
        if cache is not None and c_idx < cache.shape[0]:
            self.hist_widget.set_data(cache[c_idx], color)

    def reset_auto_contrast(self):
        """Reset to default robust percentiles."""
        self.pct_low = 0.5
        self.pct_high = 99.98
        self.apply_auto_contrast()

    def adjust_contrast(self, direction):
        """
        Adjust percentiles to tighten (+1) or loosen (-1) contrast.
        Step size: 0.01%
        """
        step = 0.01

        if direction > 0:  # Tighten
            self.pct_low += step
            self.pct_high -= step
        else:  # Loosen
            self.pct_low -= step
            self.pct_high += step

        # Clamp
        self.pct_low = max(0.0, min(self.pct_low, 49.0))
        self.pct_high = max(51.0, min(self.pct_high, 100.0))

        self.apply_auto_contrast()

    def apply_auto_contrast(self):
        c_idx = self.combo.currentIndex()
        cache = self.viewer.renderer.current_slice_cache

        if self.chk_all_channels.isChecked():
            # Apply to all channels
            for ch_idx in range(self.combo.count()):
                plane = cache[ch_idx]
                # Ignore zeros (background)
                valid_data = plane[plane > 0]
                if valid_data.size == 0:
                    valid_data = plane  # Fallback if all zeros

                mn, mx = map(
                    float,
                    np.nanpercentile(
                        valid_data, (self.pct_low, self.pct_high)
                    ),
                )

                # Update Renderer
                self.viewer.renderer.set_clim(ch_idx, mn, mx)

            # Update widgets for currently selected channel
            curr_min, curr_max = self.viewer.renderer.get_clim(c_idx)
            self.block_clim_signals(True)
            self.hist_widget.set_clim(curr_min, curr_max)
            self.min_spin.setValue(curr_min)
            self.max_spin.setValue(curr_max)
            self.block_clim_signals(False)
            self.viewer.canvas.update()
            return

        if cache is not None:
            plane = cache[c_idx]
            # Ignore zeros (background)
            valid_data = plane[plane > 0]
            if valid_data.size == 0:
                valid_data = plane  # Fallback if all zeros

            mn, mx = map(
                float,
                np.nanpercentile(valid_data, (self.pct_low, self.pct_high)),
            )

            # Update Renderer
            self.viewer.renderer.set_clim(c_idx, mn, mx)
            self.viewer.canvas.update()

            # Update widgets
            self.block_clim_signals(True)
            self.hist_widget.set_clim(mn, mx)
            self.min_spin.setValue(mn)
            self.max_spin.setValue(mx)
            self.block_clim_signals(False)
