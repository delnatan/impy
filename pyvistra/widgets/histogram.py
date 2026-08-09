import numpy as np
from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QPainter, QPen, QFont
from qtpy.QtWidgets import QWidget

from .. import colors as tokens

# Theme Constants
WIDGET_BG = QColor(tokens.BG_ELEVATED)
TEXT_COLOR = QColor(tokens.TEXT_PRIMARY)
HANDLE_COLOR = QColor(tokens.TEXT_PRIMARY)
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


def get_safe_contrast_bounds(dtype, data_min, data_max):
    """
    Return safe min/max bounds for contrast spinboxes.

    - Integer data: use full dtype range (safe hard limits).
    - Float data: use a wider range around observed data to allow exploration.
    """
    np_dtype = np.dtype(dtype)

    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        return float(info.min), float(info.max)

    if np.issubdtype(np_dtype, np.bool_):
        return 0.0, 1.0

    span = float(data_max - data_min)
    if span <= 0:
        span = max(abs(float(data_max)), abs(float(data_min)), 1.0)

    margin = span * 10.0
    return float(data_min - margin), float(data_max + margin)


class BaseHistogramWidget(QWidget):
    """
    Base class for histogram widgets with shared logic for:
    - Data management (histogram computation, data range tracking)
    - Display range (auto-expands when clim values go outside data range)
    - Mouse interaction (handle dragging, center dragging)
    - Coordinate conversion between values and pixels
    """

    climChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        # Data
        self.hist_data = None
        self.data_min = 0.0
        self.data_max = 1.0
        self.color = TEXT_COLOR

        # Display range (can expand beyond data range to keep handles visible)
        self.display_min = 0.0
        self.display_max = 1.0

        # State
        self.clim_min = 0.0
        self.clim_max = 1.0

        # Interaction
        self._dragging = None  # 'min', 'max', 'center', or None
        self._last_mouse_x = 0

    def set_data(self, data_slice, color_name):
        """Set histogram data and compute bins."""
        data = np.asarray(data_slice)
        finite_data = data[np.isfinite(data)]

        if finite_data.size == 0:
            # Graceful fallback for empty/all-NaN/all-inf slices.
            self.data_min = 0.0
            self.data_max = 1.0
            self.hist_data = np.zeros(100, dtype=float)
        else:
            # Compute in float64: at low-precision dtypes (e.g. float32),
            # a tiny span relative to the values' magnitude (near-constant
            # data with dtype noise) rounds bin edges to non-increasing
            # values, and np.histogram raises "Too many bins for data
            # range." float64 pushes that precision floor down enough
            # that the clamp below is the only guard actually needed.
            finite_data = finite_data.astype(np.float64, copy=False)
            self.data_min = float(np.min(finite_data))
            self.data_max = float(np.max(finite_data))

            span = self.data_max - self.data_min
            min_span = max(abs(self.data_min), abs(self.data_max), 1.0) * 1e-9
            if span < min_span:
                self.data_max = self.data_min + min_span

            y, _ = np.histogram(
                finite_data, bins=100, range=(self.data_min, self.data_max)
            )
            self.hist_data = np.log1p(y)

        self.color = QColor(color_name)
        self._update_display_range()
        self.update()

    def set_clim(self, vmin, vmax):
        """Set contrast limits and auto-expand display range if needed."""
        self.clim_min = vmin
        self.clim_max = vmax
        self._update_display_range()
        self.update()

    def _update_display_range(self):
        """Expand display range to keep handles visible with margin."""
        data_span = self.data_max - self.data_min
        margin = data_span * 0.05

        # Start with data range
        new_min = self.data_min
        new_max = self.data_max

        # Expand if clim values are outside data range
        if self.clim_min < self.data_min:
            new_min = self.clim_min - margin
        if self.clim_max > self.data_max:
            new_max = self.clim_max + margin

        self.display_min = new_min
        self.display_max = new_max

    def _val_to_x(self, val):
        """Convert a data value to pixel x-coordinate using display range."""
        w = self.width()
        span = self.display_max - self.display_min
        if span <= 0:
            return 0
        ratio = (val - self.display_min) / span
        x = int(ratio * w)
        # Clamp to 32-bit signed integer range to prevent Qt OverflowError
        return max(-2147483648, min(x, 2147483647))

    def _x_to_val(self, x):
        """Convert pixel x-coordinate to data value within display range."""
        w = self.width()
        span = self.display_max - self.display_min
        ratio = x / w
        val = self.display_min + (ratio * span)
        # Allow values within display range (not clamped to data range)
        return max(self.display_min, min(val, self.display_max))

    def _draw_histogram_bars(self, painter, h, bottom_margin=0):
        """Draw histogram bars. Subclasses can override bottom_margin."""
        if self.hist_data is None:
            return

        w = self.width()
        max_log = np.max(self.hist_data)
        if max_log == 0:
            max_log = 1

        fill_color = QColor(self.color)
        fill_color.setAlpha(100)
        painter.setBrush(QBrush(fill_color))
        painter.setPen(Qt.NoPen)

        n_bins = len(self.hist_data)
        bin_w = w / n_bins

        # Calculate pixel positions for data range within display range
        data_x_start = self._val_to_x(self.data_min)
        data_x_end = self._val_to_x(self.data_max)
        data_pixel_span = data_x_end - data_x_start

        if data_pixel_span <= 0:
            data_pixel_span = w

        for i, val in enumerate(self.hist_data):
            bar_h = (val / max_log) * (h - bottom_margin)
            # Map bin position to pixel position within data range
            bin_ratio = i / n_bins
            x = data_x_start + (bin_ratio * data_pixel_span)
            actual_bin_w = data_pixel_span / n_bins
            y = h - bar_h
            painter.drawRect(QRectF(x, y, actual_bin_w, bar_h))

    def _draw_overlay(self, painter, h):
        """Draw dark overlay outside the selected range."""
        w = self.width()
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)

        dark_overlay = QColor(0, 0, 0, 150)
        painter.fillRect(0, 0, x_min, h, dark_overlay)
        painter.fillRect(x_max, 0, w - x_max, h, dark_overlay)

    def _draw_handles(self, painter, h):
        """Draw min/max handle lines."""
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)

        pen = QPen(HANDLE_COLOR)
        pen.setWidth(2)
        painter.setPen(pen)

        painter.drawLine(x_min, 0, x_min, h)
        painter.drawLine(x_max, 0, x_max, h)

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

            # Convert pixel delta to value delta using display range
            w = self.width()
            display_span = self.display_max - self.display_min
            if w > 0:
                val_per_pixel = display_span / w
                d_val = dx_pixels * val_per_pixel

                new_min = self.clim_min + d_val
                new_max = self.clim_max + d_val

                # Allow movement within display range
                if new_min >= self.display_min and new_max <= self.display_max:
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

            self._update_display_range()
            self.climChanged.emit(self.clim_min, self.clim_max)
            self.update()

        self._last_mouse_x = x

    def mouseReleaseEvent(self, event):
        self._dragging = None
        self.setCursor(Qt.ArrowCursor)


class HistogramWidget(BaseHistogramWidget):
    """
    Interactive Histogram Widget.
    Displays a log-histogram and allows dragging two handles (min/max)
    to adjust contrast limits. Includes text labels showing current values.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), WIDGET_BG)

        h = self.height()

        # Draw histogram bars (with 20px bottom margin for text)
        self._draw_histogram_bars(painter, h, bottom_margin=20)

        # Draw overlay and handles
        self._draw_overlay(painter, h)
        self._draw_handles(painter, h)

        # Draw text labels
        self._draw_labels(painter, h)

    def _draw_labels(self, painter, h):
        """Draw min/max value labels at handles."""
        w = self.width()
        x_min = self._val_to_x(self.clim_min)
        x_max = self._val_to_x(self.clim_max)

        painter.setPen(TEXT_COLOR)
        font = QFont()
        painter.setFont(font)

        # Format values adaptively
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


class CompactHistogramWidget(BaseHistogramWidget):
    """
    A compact version of HistogramWidget for use in stacked channel panels.
    Smaller height, no text labels, focused on visual feedback.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setMaximumHeight(50)
        # No minimum width otherwise: as a stretch=1 item in ChannelRow's
        # QHBoxLayout, it has nothing stopping the layout from shrinking
        # it to zero once the row narrows enough to be dominated by its
        # fixed-width siblings (checkbox, swatch, spinboxes, ...). This
        # was invisible while ChannelPanel was a floating QDialog with a
        # generous default width; docked into a QMainWindow's side area,
        # Qt's default dock width is much narrower and the histogram
        # collapses to nothing without this floor.
        self.setMinimumWidth(110)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), WIDGET_BG)

        h = self.height()

        # Draw histogram bars (with minimal margin)
        self._draw_histogram_bars(painter, h, bottom_margin=4)

        # Draw overlay and handles
        self._draw_overlay(painter, h)
        self._draw_handles(painter, h)
