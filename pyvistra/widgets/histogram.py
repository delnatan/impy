import numpy as np
from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPainter, QPen, QFont
from qtpy.QtWidgets import QWidget

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
