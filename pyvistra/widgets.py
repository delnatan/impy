import numpy as np
from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont, QPainter, QPen
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pyvistra.visuals import COLORMAPS

# Theme Constants
WIDGET_BG = QColor(32, 32, 32)
TEXT_COLOR = QColor(224, 224, 224)
HANDLE_COLOR = QColor(255, 255, 255)
HANDLE_WIDTH = 6


def compute_spinbox_params(data_min, data_max):
    """
    Compute adaptive spinbox parameters based on data range.
    Returns (decimals, step_size) appropriate for the data's dynamic range.

    For normalized images (0-1 range with small values), this provides
    sufficient precision. For 16-bit images (0-65535), it avoids
    excessive decimals.
    """
    import math

    span = data_max - data_min
    if span <= 0:
        span = 1.0

    # Calculate order of magnitude of the span
    # For span=1, order=0; span=0.001, order=-3; span=65535, order=4.8
    try:
        order = math.floor(math.log10(abs(span)))
    except ValueError:
        order = 0

    # Decimals: provide ~3-4 significant figures within the span
    # For span=1 (order=0): decimals=4 (can represent 0.0001)
    # For span=0.001 (order=-3): decimals=7 (can represent 0.0000001)
    # For span=65535 (order=4): decimals=0
    decimals = max(0, min(4 - order, 10))  # Cap at 10 decimals

    # Step size: ~1% of span, rounded to a clean number
    step = span * 0.01
    if step > 0:
        try:
            step_order = math.floor(math.log10(abs(step)))
            # Round to 1 significant figure
            step = round(step, -step_order) if step_order >= 0 else round(
                step, abs(step_order)
            )
        except (ValueError, OverflowError):
            step = span * 0.01

    return decimals, step


def format_value_adaptive(value, data_min, data_max):
    """
    Format a value with adaptive precision based on data range.
    Returns a string representation appropriate for the dynamic range.
    """
    import math

    span = data_max - data_min
    if span <= 0:
        span = 1.0

    try:
        order = math.floor(math.log10(abs(span)))
    except ValueError:
        order = 0

    # Use similar logic to spinbox decimals
    decimals = max(0, min(4 - order, 8))

    # For very small or very large values, use scientific notation
    abs_val = abs(value) if value != 0 else abs(span)
    if abs_val > 0:
        try:
            val_order = math.floor(math.log10(abs_val))
            if val_order < -4 or val_order > 6:
                return f"{value:.2e}"
        except ValueError:
            pass

    return f"{value:.{decimals}f}"


def configure_spinbox_for_range(spinbox, data_min, data_max):
    """
    Configure a QDoubleSpinBox with appropriate decimals, step, and range
    based on the data's dynamic range.
    """
    decimals, step = compute_spinbox_params(data_min, data_max)

    # Add some margin to the range
    span = data_max - data_min
    margin = span * 0.1 if span > 0 else 1.0

    spinbox.blockSignals(True)
    spinbox.setDecimals(decimals)
    spinbox.setSingleStep(step)
    spinbox.setRange(data_min - margin, data_max + margin)
    spinbox.blockSignals(False)


class HistogramWidget(QWidget):
    """
    Interactive Histogram Widget.
    Displays a log-histogram and allows dragging two handles (min/max)
    to adjust contrast limits.
    """

    climChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMouseTracking(True)

        # Data
        self.hist_data = None
        self.data_min = 0.0
        self.data_max = 1.0
        self.color = TEXT_COLOR

        # State
        self.clim_min = 0.0
        self.clim_max = 1.0

        # Interaction
        self._dragging = None  # 'min', 'max', 'center', or None
        self._last_mouse_x = 0

    def set_data(self, data_slice, color_name):
        # 1. Compute Histogram
        # We use a fixed number of bins for display
        self.data_min = float(np.nanmin(data_slice))
        self.data_max = float(np.nanmax(data_slice))

        if self.data_max <= self.data_min:
            self.data_max = self.data_min + 1e-5

        y, x = np.histogram(
            data_slice, bins=100, range=(self.data_min, self.data_max)
        )
        self.hist_data = np.log1p(y)

        self.color = QColor(color_name)
        self.update()

    def set_clim(self, vmin, vmax):
        self.clim_min = vmin
        self.clim_max = vmax
        self.update()

    def _val_to_x(self, val):
        w = self.width()
        span = self.data_max - self.data_min
        if span <= 0:
            return 0
        ratio = (val - self.data_min) / span
        x = int(ratio * w)
        # Clamp to 32-bit signed integer range to prevent Qt OverflowError
        return max(-2147483648, min(x, 2147483647))

    def _x_to_val(self, x):
        w = self.width()
        span = self.data_max - self.data_min
        ratio = x / w
        val = self.data_min + (ratio * span)
        return max(self.data_min, min(val, self.data_max))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), WIDGET_BG)

        w = self.width()
        h = self.height()

        # 1. Draw Histogram
        if self.hist_data is not None:
            max_log = np.max(self.hist_data)
            if max_log == 0:
                max_log = 1

            fill_color = QColor(self.color)
            fill_color.setAlpha(100)
            painter.setBrush(QBrush(fill_color))
            painter.setPen(Qt.NoPen)

            n_bins = len(self.hist_data)
            bin_w = w / n_bins

            for i, val in enumerate(self.hist_data):
                bar_h = (val / max_log) * (h - 20)
                x = i * bin_w
                y = h - bar_h
                painter.drawRect(QRectF(x, y, bin_w, bar_h))

        # 2. Draw Overlay (Darken outside selection)
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)

        dark_overlay = QColor(0, 0, 0, 150)
        painter.fillRect(0, 0, x_min, h, dark_overlay)
        painter.fillRect(x_max, 0, w - x_max, h, dark_overlay)

        # 3. Draw Handles
        pen = QPen(HANDLE_COLOR)
        pen.setWidth(2)
        painter.setPen(pen)

        # Min Handle
        painter.drawLine(x_min, 0, x_min, h)
        # Max Handle
        painter.drawLine(x_max, 0, x_max, h)

        # 4. Text Labels
        painter.setPen(TEXT_COLOR)
        font = QFont()
        painter.setFont(font)

        # Draw min/max values at handles (adaptive formatting)
        min_str = format_value_adaptive(self.clim_min, self.data_min, self.data_max)
        max_str = format_value_adaptive(self.clim_max, self.data_min, self.data_max)

        # Adjust text position to stay on screen
        fm = painter.fontMetrics()
        tw_min = fm.width(min_str)
        tw_max = fm.width(max_str)

        draw_x_min = max(2, min(x_min - tw_min - 2, w - tw_min - 2))
        draw_x_max = min(w - tw_max - 2, max(x_max + 2, 2))

        # If handles are close, push text apart
        if abs(x_max - x_min) < (tw_min + tw_max + 10):
            draw_x_min = x_min - tw_min - 5
            draw_x_max = x_max + 5

        painter.drawText(int(draw_x_min), h - 5, min_str)
        painter.drawText(int(draw_x_max), h - 5, max_str)

    def mousePressEvent(self, event):
        x = event.x()
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)

        # Hit test
        dist_min = abs(x - x_min)
        dist_max = abs(x - x_max)

        if dist_min < 10:
            self._dragging = "min"
        elif dist_max < 10:
            self._dragging = "max"
        elif x_min < x < x_max:
            self._dragging = "center"
        else:
            self._dragging = None

        self._last_mouse_x = x

    def mouseMoveEvent(self, event):
        x = event.x()

        # Cursor updates
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)
        dist_min = abs(x - x_min)
        dist_max = abs(x - x_max)

        if dist_min < 10 or dist_max < 10:
            self.setCursor(Qt.SizeHorCursor)
        elif x_min < x < x_max:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self._dragging is None:
            return

        if self._dragging == "center":
            self.setCursor(Qt.ClosedHandCursor)
            dx_pixels = x - self._last_mouse_x

            # Convert pixel delta to value delta
            # We need to be careful because _x_to_val is absolute
            # Calculate span per pixel
            w = self.width()
            data_span = self.data_max - self.data_min
            if w > 0:
                val_per_pixel = data_span / w
                d_val = dx_pixels * val_per_pixel

                new_min = self.clim_min + d_val
                new_max = self.clim_max + d_val

                # Clamp to data bounds
                if new_min >= self.data_min and new_max <= self.data_max:
                    self.clim_min = new_min
                    self.clim_max = new_max
                    self.climChanged.emit(self.clim_min, self.clim_max)
                    self.update()

        else:
            val = self._x_to_val(x)
            if self._dragging == "min":
                self.clim_min = min(val, self.clim_max - 1e-5)
            elif self._dragging == "max":
                self.clim_max = max(val, self.clim_min + 1e-5)

            self.climChanged.emit(self.clim_min, self.clim_max)
            self.update()

        self._last_mouse_x = x

    def mouseReleaseEvent(self, event):
        self._dragging = None
        self.setCursor(Qt.ArrowCursor)


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


from qtpy.QtWidgets import (
    QFrame,
    QHeaderView,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
)


class CompactHistogramWidget(QWidget):
    """
    A compact version of HistogramWidget for use in stacked channel panels.
    Smaller height, no text labels, focused on visual feedback.
    """

    climChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setMaximumHeight(50)
        self.setMouseTracking(True)

        # Data
        self.hist_data = None
        self.data_min = 0.0
        self.data_max = 1.0
        self.color = TEXT_COLOR

        # State
        self.clim_min = 0.0
        self.clim_max = 1.0

        # Interaction
        self._dragging = None  # 'min', 'max', 'center', or None
        self._last_mouse_x = 0

    def set_data(self, data_slice, color_name):
        self.data_min = float(np.nanmin(data_slice))
        self.data_max = float(np.nanmax(data_slice))

        if self.data_max <= self.data_min:
            self.data_max = self.data_min + 1e-5

        y, x = np.histogram(
            data_slice, bins=100, range=(self.data_min, self.data_max)
        )
        self.hist_data = np.log1p(y)

        self.color = QColor(color_name)
        self.update()

    def set_clim(self, vmin, vmax):
        self.clim_min = vmin
        self.clim_max = vmax
        self.update()

    def _val_to_x(self, val):
        w = self.width()
        span = self.data_max - self.data_min
        if span <= 0:
            return 0
        ratio = (val - self.data_min) / span
        x = int(ratio * w)
        return max(-2147483648, min(x, 2147483647))

    def _x_to_val(self, x):
        w = self.width()
        span = self.data_max - self.data_min
        ratio = x / w
        val = self.data_min + (ratio * span)
        return max(self.data_min, min(val, self.data_max))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), WIDGET_BG)

        w = self.width()
        h = self.height()

        # Draw Histogram
        if self.hist_data is not None:
            max_log = np.max(self.hist_data)
            if max_log == 0:
                max_log = 1

            fill_color = QColor(self.color)
            fill_color.setAlpha(100)
            painter.setBrush(QBrush(fill_color))
            painter.setPen(Qt.NoPen)

            n_bins = len(self.hist_data)
            bin_w = w / n_bins

            for i, val in enumerate(self.hist_data):
                bar_h = (val / max_log) * (h - 4)
                x = i * bin_w
                y = h - bar_h
                painter.drawRect(QRectF(x, y, bin_w, bar_h))

        # Draw Overlay (Darken outside selection)
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)

        dark_overlay = QColor(0, 0, 0, 150)
        painter.fillRect(0, 0, x_min, h, dark_overlay)
        painter.fillRect(x_max, 0, w - x_max, h, dark_overlay)

        # Draw Handles (thinner for compact view)
        pen = QPen(HANDLE_COLOR)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(x_min, 0, x_min, h)
        painter.drawLine(x_max, 0, x_max, h)

    def mousePressEvent(self, event):
        x = event.x()
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)

        dist_min = abs(x - x_min)
        dist_max = abs(x - x_max)

        if dist_min < 10:
            self._dragging = "min"
        elif dist_max < 10:
            self._dragging = "max"
        elif x_min < x < x_max:
            self._dragging = "center"
        else:
            self._dragging = None

        self._last_mouse_x = x

    def mouseMoveEvent(self, event):
        x = event.x()

        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)
        dist_min = abs(x - x_min)
        dist_max = abs(x - x_max)

        if dist_min < 10 or dist_max < 10:
            self.setCursor(Qt.SizeHorCursor)
        elif x_min < x < x_max:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self._dragging is None:
            return

        if self._dragging == "center":
            self.setCursor(Qt.ClosedHandCursor)
            dx_pixels = x - self._last_mouse_x

            w = self.width()
            data_span = self.data_max - self.data_min
            if w > 0:
                val_per_pixel = data_span / w
                d_val = dx_pixels * val_per_pixel

                new_min = self.clim_min + d_val
                new_max = self.clim_max + d_val

                if new_min >= self.data_min and new_max <= self.data_max:
                    self.clim_min = new_min
                    self.clim_max = new_max
                    self.climChanged.emit(self.clim_min, self.clim_max)
                    self.update()
        else:
            val = self._x_to_val(x)
            if self._dragging == "min":
                self.clim_min = min(val, self.clim_max - 1e-5)
            elif self._dragging == "max":
                self.clim_max = max(val, self.clim_min + 1e-5)

            self.climChanged.emit(self.clim_min, self.clim_max)
            self.update()

        self._last_mouse_x = x

    def mouseReleaseEvent(self, event):
        self._dragging = None
        self.setCursor(Qt.ArrowCursor)


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


class MetadataDialog(QDialog):
    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Metadata")
        self.resize(400, 500)

        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.table)

        self.populate_table(metadata)

    def populate_table(self, metadata):
        self.table.setRowCount(0)
        for key, value in metadata.items():
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Key
            k_item = QTableWidgetItem(str(key))
            k_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 0, k_item)

            # Value
            v_str = str(value)
            # If value is a long list/array, truncate it?
            if (
                isinstance(value, (list, tuple, np.ndarray))
                and len(value) > 10
            ):
                v_str = f"{type(value).__name__} shape={np.shape(value)}"

            v_item = QTableWidgetItem(v_str)
            v_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 1, v_item)


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
        from .io import apply_transform

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
        from .manager import manager

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
        from vispy.visuals.transforms.chain import ChainTransform
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


class ImageOutputSelector(QWidget):
    """Reusable widget for selecting image output destination."""

    output_sent = Signal(object)  # Emits ImageWindow or filepath after send

    # Special item data values
    _NEW_WINDOW = "__new__"
    _SAVE_FILE = "__file__"

    def __init__(self, parent=None, default_title="Result", formats=None):
        """
        Args:
            parent: Parent widget
            default_title: Default title for new windows
            formats: List of supported formats for saving, e.g.
                     [("TIFF", ".tif"), ("PSF Zarr", ".psf.zarr")]
                     If None, defaults to [("TIFF", ".tif")]
        """
        super().__init__(parent)
        self._default_title = default_title
        self._formats = formats or [("TIFF", ".tif")]

        from qtpy.QtWidgets import QFileDialog, QLineEdit, QStackedWidget

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Row 1: Output dropdown
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Output:"))
        self._combo = QComboBox()
        self._combo.currentIndexChanged.connect(self._on_selection_changed)
        row1.addWidget(self._combo, 1)
        layout.addLayout(row1)

        # Row 2: Secondary input area (stacked widget)
        self._secondary = QStackedWidget()

        # Page 0: Empty (for existing window)
        self._secondary.addWidget(QWidget())

        # Page 1: Title field (for new window)
        title_page = QWidget()
        title_layout = QHBoxLayout(title_page)
        title_layout.setContentsMargins(0, 4, 0, 0)
        title_layout.addWidget(QLabel("Title:"))
        self._title_edit = QLineEdit(default_title)
        title_layout.addWidget(self._title_edit)
        self._secondary.addWidget(title_page)

        # Page 2: Format + path (for file)
        file_page = QWidget()
        file_layout = QHBoxLayout(file_page)
        file_layout.setContentsMargins(0, 4, 0, 0)
        file_layout.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        for name, ext in self._formats:
            self._format_combo.addItem(name, ext)
        file_layout.addWidget(self._format_combo)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path (optional, will prompt)")
        file_layout.addWidget(self._path_edit, 1)
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self._browse_btn)
        self._secondary.addWidget(file_page)

        layout.addWidget(self._secondary)

        # Initial population
        self.refresh_windows()

    def refresh_windows(self):
        """Update list of available windows from WindowManager."""
        self._combo.blockSignals(True)
        self._combo.clear()

        from .manager import manager

        windows = manager.get_all()

        # Add existing windows
        for wid, win in sorted(windows.items()):
            title = win.windowTitle()
            display = f"[{wid}] {title[:30]}"
            self._combo.addItem(display, wid)

        # Separator if windows exist
        if windows:
            self._combo.insertSeparator(self._combo.count())

        # Special items
        self._combo.addItem("New Window", self._NEW_WINDOW)
        self._combo.addItem("Save to File", self._SAVE_FILE)

        # Default to "New Window"
        idx = self._combo.findData(self._NEW_WINDOW)
        self._combo.setCurrentIndex(idx)

        self._combo.blockSignals(False)
        self._on_selection_changed(self._combo.currentIndex())

    def _on_selection_changed(self, index):
        """Show/hide secondary inputs based on selection."""
        data = self._combo.currentData()
        if data == self._NEW_WINDOW:
            self._secondary.setCurrentIndex(1)  # Title field
        elif data == self._SAVE_FILE:
            self._secondary.setCurrentIndex(2)  # Format + path
        else:
            self._secondary.setCurrentIndex(0)  # Empty

    def get_selection_type(self):
        """Returns 'existing', 'new', or 'file'."""
        data = self._combo.currentData()
        if data == self._NEW_WINDOW:
            return "new"
        elif data == self._SAVE_FILE:
            return "file"
        else:
            return "existing"

    def send(self, data, metadata=None):
        """
        Send data to the selected destination.

        Args:
            data: 5D array-like (ImageBuffer, Numpy5DProxy, ndarray)
            metadata: dict with metadata (replaces window.meta entirely)

        Returns:
            ImageWindow: if sent to new/existing window
            str: filepath if saved to file
            None: if cancelled or error
        """
        selection = self._combo.currentData()
        metadata = metadata or {}

        if selection == self._NEW_WINDOW:
            return self._send_to_new_window(data, metadata)
        elif selection == self._SAVE_FILE:
            return self._send_to_file(data, metadata)
        else:
            return self._send_to_existing(selection, data, metadata)

    def _send_to_existing(self, wid, data, metadata):
        from .manager import manager

        window = manager.get(wid)
        if window is None:
            return None
        window.set_data(data, metadata)
        # Update title to reflect new content
        filename = metadata.get("filename", "Result")
        window.setWindowTitle(f"[{wid}] {filename}")
        window.canvas.update()
        self.output_sent.emit(window)
        return window

    def _send_to_new_window(self, data, metadata):
        from .ui import ImageWindow

        title = self._title_edit.text() or self._default_title
        metadata["filename"] = title
        viewer = ImageWindow(data, title=title, meta=metadata)
        viewer.show()
        self.output_sent.emit(viewer)
        self.refresh_windows()  # Update dropdown with new window
        return viewer

    def _send_to_file(self, data, metadata):
        from qtpy.QtWidgets import QFileDialog

        from .io import save_psf, save_tiff

        filepath = self._path_edit.text()
        ext = self._format_combo.currentData()

        if not filepath:
            # Open file dialog
            filter_str = f"*{ext}"
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Output", f"output{ext}", filter_str
            )
            if not filepath:
                return None

        # Ensure correct extension
        if not filepath.endswith(ext):
            filepath += ext

        # Save based on format
        if ext == ".tif":
            scale = metadata.get("scale", (1.0, 1.0, 1.0))
            save_tiff(filepath, data, scale=scale)
        elif ext == ".psf.zarr":
            save_psf(filepath, data, metadata)

        self.output_sent.emit(filepath)
        return filepath

    def _browse_file(self):
        from qtpy.QtWidgets import QFileDialog

        ext = self._format_combo.currentData()
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Output", f"output{ext}", f"*{ext}"
        )
        if filepath:
            self._path_edit.setText(filepath)


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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
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

        # Compute Button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._compute_psf)
        btn_layout.addWidget(self.compute_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

    def _on_modality_changed(self, index):
        """Show/hide spinning disk parameters based on selection."""
        is_spinning_disk = index == 1
        self.sd_group.setVisible(is_spinning_disk)
        # Adjust dialog size
        self.adjustSize()

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

            from .io import ImageBuffer
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
