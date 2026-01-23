import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyvistra.visuals import COLORMAPS
from .histogram import CompactHistogramWidget, configure_spinbox_for_range


class ChannelRow(QWidget):
    """
    A single row representing one channel with:
    - Visibility checkbox
    - Color swatch (clickable for colormap selection)
    - Channel name label
    - Min/Max spinboxes for intensity range
    - Compact histogram
    - Gamma adjustment
    """

    visibilityChanged = Signal(int, bool)  # channel_idx, visible
    climChanged = Signal(int, float, float)  # channel_idx, vmin, vmax
    colormapChanged = Signal(int, str)  # channel_idx, colormap_name
    gammaChanged = Signal(int, float)  # channel_idx, gamma

    def __init__(self, channel_idx, channel_name, color, parent=None):
        super().__init__(parent)
        self.channel_idx = channel_idx

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Visibility checkbox
        self.chk_visible = QCheckBox()
        self.chk_visible.setChecked(True)
        self.chk_visible.setToolTip("Toggle channel visibility")
        self.chk_visible.toggled.connect(self._on_visibility_changed)
        layout.addWidget(self.chk_visible)

        # Color swatch (button that opens colormap menu)
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(20, 20)
        self.color_btn.setCursor(Qt.PointingHandCursor)
        self.color_btn.setToolTip("Change colormap")
        self._update_color_swatch(color)
        self.color_btn.clicked.connect(self._show_colormap_menu)
        layout.addWidget(self.color_btn)

        # Channel name label
        self.name_label = QLabel(channel_name)
        self.name_label.setFixedWidth(40)
        self.name_label.setStyleSheet("color: #EEE; font-size: 11px;")
        self.name_label.setToolTip(channel_name)  # Show full name on hover
        layout.addWidget(self.name_label)

        # Min spinbox for contrast
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(1)
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setSingleStep(10)
        self.min_spin.setFixedWidth(65)
        self.min_spin.setToolTip("Minimum intensity")
        self.min_spin.valueChanged.connect(self._on_min_changed)
        layout.addWidget(self.min_spin)

        # Compact histogram
        self.histogram = CompactHistogramWidget()
        self.histogram.climChanged.connect(self._on_histogram_clim_changed)
        layout.addWidget(self.histogram, 1)

        # Max spinbox for contrast
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(1)
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setSingleStep(10)
        self.max_spin.setFixedWidth(65)
        self.max_spin.setToolTip("Maximum intensity")
        self.max_spin.valueChanged.connect(self._on_max_changed)
        layout.addWidget(self.max_spin)

        # Gamma spinbox
        gamma_label = QLabel("γ")
        gamma_label.setStyleSheet("color: #AAA; font-size: 10px;")
        gamma_label.setFixedWidth(10)
        layout.addWidget(gamma_label)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 4.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setFixedWidth(50)
        self.gamma_spin.setToolTip("Gamma correction")
        self.gamma_spin.valueChanged.connect(self._on_gamma_changed)
        layout.addWidget(self.gamma_spin)

        self.current_colormap = "White"

    def _update_color_swatch(self, color):
        """Update the color swatch button background."""
        self.color_btn.setStyleSheet(
            f"background-color: {color}; border: 1px solid #555; border-radius: 3px;"
        )

    def _on_visibility_changed(self, checked):
        self.visibilityChanged.emit(self.channel_idx, checked)

    def _on_min_changed(self, value):
        """Handle min spinbox change."""
        max_val = self.max_spin.value()
        if value < max_val:
            self.climChanged.emit(self.channel_idx, value, max_val)
            # Update histogram display
            self.histogram.blockSignals(True)
            self.histogram.set_clim(value, max_val)
            self.histogram.blockSignals(False)

    def _on_max_changed(self, value):
        """Handle max spinbox change."""
        min_val = self.min_spin.value()
        if value > min_val:
            self.climChanged.emit(self.channel_idx, min_val, value)
            # Update histogram display
            self.histogram.blockSignals(True)
            self.histogram.set_clim(min_val, value)
            self.histogram.blockSignals(False)

    def _on_histogram_clim_changed(self, vmin, vmax):
        """Handle histogram clim change (from dragging handles)."""
        # Update spinboxes
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        # Emit signal to parent
        self.climChanged.emit(self.channel_idx, vmin, vmax)

    def _on_gamma_changed(self, value):
        self.gammaChanged.emit(self.channel_idx, value)

    def _show_colormap_menu(self):
        """Show a popup menu for colormap selection."""
        from qtpy.QtWidgets import QMenu

        menu = QMenu(self)

        for cmap_name in COLORMAPS.keys():
            action = menu.addAction(cmap_name)
            action.triggered.connect(
                lambda checked, name=cmap_name: self._on_colormap_selected(
                    name
                )
            )

        menu.exec_(
            self.color_btn.mapToGlobal(self.color_btn.rect().bottomLeft())
        )

    def _on_colormap_selected(self, cmap_name):
        self.current_colormap = cmap_name
        self.colormapChanged.emit(self.channel_idx, cmap_name)

    def set_data(self, data_slice, color):
        """Update histogram data and color."""
        self._update_color_swatch(color)
        self.histogram.set_data(data_slice, color)

        # Configure spinboxes for the data range (adaptive precision)
        data_min = self.histogram.data_min
        data_max = self.histogram.data_max
        configure_spinbox_for_range(self.min_spin, data_min, data_max)
        configure_spinbox_for_range(self.max_spin, data_min, data_max)

    def set_clim(self, vmin, vmax):
        """Update contrast limits display (histogram and spinboxes)."""
        self.histogram.blockSignals(True)
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.histogram.set_clim(vmin, vmax)
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.histogram.blockSignals(False)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)

    def set_visible_state(self, visible):
        """Update checkbox state without emitting signal."""
        self.chk_visible.blockSignals(True)
        self.chk_visible.setChecked(visible)
        self.chk_visible.blockSignals(False)

    def set_gamma(self, gamma):
        """Update gamma spinbox without emitting signal."""
        self.gamma_spin.blockSignals(True)
        self.gamma_spin.setValue(gamma)
        self.gamma_spin.blockSignals(False)


class ChannelPanel(QDialog):
    """
    Floating dialog that displays all channels stacked vertically,
    each with visibility toggle, colormap selector, histogram, and intensity spinboxes.
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Channels")
        self.setWindowFlags(Qt.Tool)
        self.resize(480, min(200 + viewer.C * 60, 500))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Window ID Label
        if hasattr(viewer, "window_id"):
            wid_label = QLabel(f"<b>Window: {viewer.window_id}</b>")
            wid_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(wid_label)

        # Scroll area for channel rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        self.rows_layout = QVBoxLayout(scroll_content)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Channel rows
        self.channel_rows = []
        self._setup_channel_rows()

        # Auto-contrast button row
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_auto = QPushButton("Auto Contrast All")
        btn_auto.clicked.connect(self._auto_contrast_all)
        btn_layout.addWidget(btn_auto)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Initial data load
        self.refresh_ui()

    def _setup_channel_rows(self):
        """Create a row widget for each channel."""
        n_channels = self.viewer.C
        meta_channels = self.viewer.meta.get("channels", [])

        for c in range(n_channels):
            # Get channel name from metadata or use default
            if c < len(meta_channels) and "name" in meta_channels[c]:
                ch_name = meta_channels[c]["name"]
            else:
                ch_name = f"Ch {c + 1}"

            # Get color from renderer
            color = self.viewer.renderer.channel_colors[
                c % len(self.viewer.renderer.channel_colors)
            ]

            row = ChannelRow(c, ch_name, color)
            row.visibilityChanged.connect(self._on_visibility_changed)
            row.climChanged.connect(self._on_clim_changed)
            row.colormapChanged.connect(self._on_colormap_changed)
            row.gammaChanged.connect(self._on_gamma_changed)

            self.channel_rows.append(row)
            self.rows_layout.addWidget(row)

        self.rows_layout.addStretch()

    def _on_visibility_changed(self, channel_idx, visible):
        """Handle visibility toggle for a channel."""
        self.viewer.renderer.set_channel_visible(channel_idx, visible)
        self.viewer.canvas.update()

    def _on_clim_changed(self, channel_idx, vmin, vmax):
        """Handle contrast change for a channel."""
        self.viewer.renderer.set_clim(channel_idx, vmin, vmax)
        self.viewer.canvas.update()

    def _on_colormap_changed(self, channel_idx, cmap_name):
        """Handle colormap change for a channel."""
        self.viewer.renderer.set_colormap(channel_idx, cmap_name)
        self.viewer.canvas.update()

        # Update color swatch
        color = self.viewer.renderer.channel_colors[
            channel_idx % len(self.viewer.renderer.channel_colors)
        ]
        self.channel_rows[channel_idx]._update_color_swatch(color)

        # Refresh histogram with new color
        self.refresh_ui()

    def _on_gamma_changed(self, channel_idx, gamma):
        """Handle gamma change for a channel."""
        self.viewer.renderer.set_gamma(channel_idx, gamma)
        self.viewer.canvas.update()

    def _auto_contrast_all(self):
        """Apply auto contrast to all channels."""
        cache = self.viewer.renderer.current_slice_cache
        if cache is None:
            return

        for c in range(len(self.channel_rows)):
            if c < cache.shape[0]:
                plane = cache[c]
                valid_data = plane[plane > 0]
                if valid_data.size == 0:
                    valid_data = plane

                mn, mx = map(float, np.nanpercentile(valid_data, (0.5, 99.98)))
                self.viewer.renderer.set_clim(c, mn, mx)
                self.channel_rows[c].set_clim(mn, mx)

        self.viewer.canvas.update()

    def refresh_ui(self):
        """Refresh all channel rows with current data."""
        cache = self.viewer.renderer.current_slice_cache
        if cache is None:
            return

        for c, row in enumerate(self.channel_rows):
            if c < cache.shape[0]:
                plane = cache[c]
                color = self.viewer.renderer.channel_colors[
                    c % len(self.viewer.renderer.channel_colors)
                ]
                row.set_data(plane, color)

                # Update clim
                vmin, vmax = self.viewer.renderer.get_clim(c)
                row.set_clim(vmin, vmax)

                # Update visibility state
                visible = self.viewer.renderer.get_channel_visible(c)
                row.set_visible_state(visible)

                # Update gamma
                gamma = self.viewer.renderer.get_gamma(c)
                row.set_gamma(gamma)
