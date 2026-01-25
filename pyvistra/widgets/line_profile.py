"""
Line Profile Dialog - Displays intensity profiles along LineROIs.

Supports multiple channels with toggle visibility and live updates when ROIs are moved.
"""

import numpy as np
from qtpy.QtCore import Qt, Signal, QPointF
from qtpy.QtGui import QColor, QPainter, QPen, QFont, QPolygonF
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .histogram import WIDGET_BG, TEXT_COLOR
from ..manager import manager
from ..rois import LineROI

# Singleton instance
_line_profile_dialog = None


def get_line_profile_dialog():
    """Get or create the singleton LineProfileDialog instance."""
    global _line_profile_dialog
    if _line_profile_dialog is None:
        _line_profile_dialog = LineProfileDialog()
    return _line_profile_dialog


def line_profile_dialog_exists():
    """Check if the line profile dialog has been created."""
    return _line_profile_dialog is not None


class LineProfileWidget(QWidget):
    """
    Custom widget for drawing intensity profiles with QPainter.

    Uses the same dark theme as HistogramWidget for visual consistency.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setMinimumWidth(300)

        # Profile data
        self.profiles = None  # (C, num_points) array
        self.distances = None  # (num_points,) array
        self.channel_colors = []
        self.channel_visible = {}

        # Plot margins
        self.margin_left = 50
        self.margin_right = 15
        self.margin_top = 15
        self.margin_bottom = 30

    def set_data(self, distances, profiles, colors):
        """
        Update profile data and trigger repaint.

        Args:
            distances: 1D array of distance values along the line
            profiles: 2D array of shape (C, num_points) with intensity values
            colors: List of color strings for each channel
        """
        self.distances = distances
        self.profiles = profiles
        self.channel_colors = colors

        # Initialize visibility for new channels
        if profiles is not None:
            for c in range(profiles.shape[0]):
                if c not in self.channel_visible:
                    self.channel_visible[c] = True

        self.update()

    def set_channel_visible(self, channel_idx, visible):
        """Set visibility for a specific channel."""
        self.channel_visible[channel_idx] = visible
        self.update()

    def clear(self):
        """Clear all profile data."""
        self.profiles = None
        self.distances = None
        self.update()

    def _get_plot_rect(self):
        """Get the plotting area rectangle."""
        x = self.margin_left
        y = self.margin_top
        w = self.width() - self.margin_left - self.margin_right
        h = self.height() - self.margin_top - self.margin_bottom
        return x, y, max(1, w), max(1, h)

    def _x_to_px(self, x_val, x_min, x_max, plot_x, plot_w):
        """Convert data x-coordinate to pixel x-coordinate."""
        if x_max == x_min:
            return plot_x
        ratio = (x_val - x_min) / (x_max - x_min)
        return plot_x + ratio * plot_w

    def _y_to_px(self, y_val, y_min, y_max, plot_y, plot_h):
        """Convert data y-coordinate to pixel y-coordinate (inverted)."""
        if y_max == y_min:
            return plot_y + plot_h / 2
        ratio = (y_val - y_min) / (y_max - y_min)
        return plot_y + plot_h - ratio * plot_h

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), WIDGET_BG)

        if self.profiles is None or self.distances is None:
            # Draw placeholder text
            painter.setPen(TEXT_COLOR)
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Select a LineROI to show profile"
            )
            return

        plot_x, plot_y, plot_w, plot_h = self._get_plot_rect()

        # Calculate data ranges
        x_min = float(np.min(self.distances))
        x_max = float(np.max(self.distances))

        # Get intensity range from visible channels only
        visible_profiles = []
        for c in range(self.profiles.shape[0]):
            if self.channel_visible.get(c, True):
                visible_profiles.append(self.profiles[c])

        if visible_profiles:
            all_visible = np.concatenate(visible_profiles)
            y_min = float(np.min(all_visible))
            y_max = float(np.max(all_visible))
        else:
            y_min, y_max = 0, 1

        # Add small margin to y-range
        y_range = y_max - y_min
        if y_range > 0:
            y_min -= y_range * 0.05
            y_max += y_range * 0.05
        else:
            y_min -= 0.5
            y_max += 0.5

        # Draw grid and axes
        self._draw_axes(painter, plot_x, plot_y, plot_w, plot_h,
                       x_min, x_max, y_min, y_max)

        # Draw profile curves for visible channels
        for c in range(self.profiles.shape[0]):
            if self.channel_visible.get(c, True):
                self._draw_profile(painter, c, plot_x, plot_y, plot_w, plot_h,
                                  x_min, x_max, y_min, y_max)

    def _draw_axes(self, painter, plot_x, plot_y, plot_w, plot_h,
                   x_min, x_max, y_min, y_max):
        """Draw axes, grid lines, and labels."""
        # Draw border
        border_pen = QPen(QColor(80, 80, 80))
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.drawRect(plot_x, plot_y, plot_w, plot_h)

        # Draw grid lines (light gray, dashed)
        grid_pen = QPen(QColor(60, 60, 60))
        grid_pen.setStyle(Qt.DotLine)
        painter.setPen(grid_pen)

        # Horizontal grid lines (4 lines)
        for i in range(1, 4):
            y = plot_y + (i / 4) * plot_h
            painter.drawLine(int(plot_x), int(y), int(plot_x + plot_w), int(y))

        # Vertical grid lines (4 lines)
        for i in range(1, 4):
            x = plot_x + (i / 4) * plot_w
            painter.drawLine(int(x), int(plot_y), int(x), int(plot_y + plot_h))

        # Draw axis labels
        painter.setPen(TEXT_COLOR)
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        # Y-axis labels
        y_labels = [y_min, (y_min + y_max) / 2, y_max]
        for y_val in y_labels:
            y_px = self._y_to_px(y_val, y_min, y_max, plot_y, plot_h)
            label = self._format_value(y_val)
            painter.drawText(
                2, int(y_px - 6), self.margin_left - 5, 12,
                Qt.AlignRight | Qt.AlignVCenter, label
            )

        # X-axis labels
        x_labels = [x_min, (x_min + x_max) / 2, x_max]
        for x_val in x_labels:
            x_px = self._x_to_px(x_val, x_min, x_max, plot_x, plot_w)
            label = f"{x_val:.0f}"
            painter.drawText(
                int(x_px - 20), int(plot_y + plot_h + 5), 40, 20,
                Qt.AlignCenter, label
            )

        # Axis titles
        painter.drawText(
            int(plot_x + plot_w / 2 - 40),
            int(plot_y + plot_h + 18),
            80, 15,
            Qt.AlignCenter, "Distance (px)"
        )

    def _draw_profile(self, painter, channel_idx, plot_x, plot_y, plot_w, plot_h,
                     x_min, x_max, y_min, y_max):
        """Draw a single channel's profile as a polyline."""
        if channel_idx >= len(self.channel_colors):
            color = QColor(255, 255, 255)
        else:
            color = QColor(self.channel_colors[channel_idx])

        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)

        # Build list of points for the polyline
        points = []
        for i, d in enumerate(self.distances):
            x_px = self._x_to_px(d, x_min, x_max, plot_x, plot_w)
            y_px = self._y_to_px(self.profiles[channel_idx, i], y_min, y_max,
                                 plot_y, plot_h)
            points.append(QPointF(x_px, y_px))

        if len(points) > 1:
            polygon = QPolygonF(points)
            painter.drawPolyline(polygon)

    def _format_value(self, value):
        """Format a value for display."""
        if abs(value) < 0.01 or abs(value) >= 10000:
            return f"{value:.1e}"
        elif abs(value) < 1:
            return f"{value:.2f}"
        elif abs(value) < 100:
            return f"{value:.1f}"
        else:
            return f"{value:.0f}"


class ChannelToggleRow(QWidget):
    """
    Horizontal row of channel checkboxes with color swatches.
    """

    visibilityChanged = Signal(int, bool)

    def __init__(self, num_channels, colors, parent=None):
        super().__init__(parent)
        self.checkboxes = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)

        for c in range(num_channels):
            # Container for checkbox + swatch
            cb_container = QWidget()
            cb_layout = QHBoxLayout(cb_container)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            cb_layout.setSpacing(3)

            # Checkbox
            cb = QCheckBox()
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, idx=c: self.visibilityChanged.emit(idx, checked))
            cb_layout.addWidget(cb)
            self.checkboxes.append(cb)

            # Color swatch
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            color = colors[c] if c < len(colors) else "#FFFFFF"
            swatch.setStyleSheet(
                f"background-color: {color}; border: 1px solid #555; border-radius: 2px;"
            )
            cb_layout.addWidget(swatch)

            # Channel label
            label = QLabel(f"Ch{c+1}")
            label.setStyleSheet("color: #CCC; font-size: 10px;")
            cb_layout.addWidget(label)

            layout.addWidget(cb_container)

        layout.addStretch()

    def update_colors(self, colors):
        """Update color swatches."""
        for i, cb_container in enumerate(self.findChildren(QWidget)):
            if i < len(colors):
                swatches = cb_container.findChildren(QLabel)
                for swatch in swatches:
                    if swatch.text() == "":  # Color swatch has no text
                        swatch.setStyleSheet(
                            f"background-color: {colors[i]}; border: 1px solid #555; border-radius: 2px;"
                        )


class LineProfileDialog(QDialog):
    """
    Floating dialog that displays intensity profiles along LineROIs.

    Follows the singleton pattern like ContrastDialog and ROIManager.
    Tracks the active window and updates when ROIs are modified.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Line Profile")
        self.setWindowFlags(Qt.Tool)
        self.resize(450, 320)

        self.active_window = None
        self.current_roi = None
        self._connected_windows = set()
        self._is_shutting_down = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Channel toggle row (initially hidden, shown when data loaded)
        self.channel_toggle = None
        self.toggle_container = QWidget()
        self.toggle_layout = QVBoxLayout(self.toggle_container)
        self.toggle_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_container)

        # Profile plot widget
        self.profile_widget = LineProfileWidget()
        layout.addWidget(self.profile_widget, 1)

        # Status bar
        self.status_label = QLabel("No LineROI selected")
        self.status_label.setStyleSheet("color: #AAA; font-size: 10px;")
        layout.addWidget(self.status_label)

        # Connect to WindowManager for window tracking
        manager.window_registered.connect(self._on_window_registered)

        # Connect to existing windows
        for window in manager.get_all().values():
            self._connect_window(window)

    def _on_window_registered(self, window):
        """Handle new window registration."""
        if self._is_shutting_down:
            return
        self._connect_window(window)

    def _connect_window(self, window):
        """Connect to a window's signals."""
        if window in self._connected_windows:
            return

        window.window_activated.connect(self._on_window_activated)
        window.window_closing.connect(self._on_window_closing)
        window.roi_selection_changed.connect(self._on_roi_selection_changed)

        # Connect to roi_modified signal if available
        if hasattr(window, 'roi_modified'):
            window.roi_modified.connect(self._on_roi_modified)

        self._connected_windows.add(window)

    def _disconnect_window(self, window):
        """Disconnect from a window's signals."""
        if window not in self._connected_windows:
            return

        try:
            window.window_activated.disconnect(self._on_window_activated)
            window.window_closing.disconnect(self._on_window_closing)
            window.roi_selection_changed.disconnect(self._on_roi_selection_changed)
            if hasattr(window, 'roi_modified'):
                window.roi_modified.disconnect(self._on_roi_modified)
        except (TypeError, RuntimeError):
            pass

        self._connected_windows.discard(window)

    def _on_window_activated(self, window):
        """Handle window activation."""
        if self._is_shutting_down:
            return
        self.active_window = window
        # Check if current selection is a LineROI
        for roi in window.rois:
            if roi.selected and isinstance(roi, LineROI):
                self._update_profile(roi)
                return
        # No LineROI selected
        self.current_roi = None
        self.profile_widget.clear()
        self.status_label.setText("No LineROI selected")

    def _on_window_closing(self, window):
        """Handle window closing."""
        self._disconnect_window(window)
        if window == self.active_window:
            self.active_window = None
            self.current_roi = None
            self.profile_widget.clear()
            self.status_label.setText("No LineROI selected")

    def _on_roi_selection_changed(self, roi):
        """Handle ROI selection change."""
        if self._is_shutting_down:
            return

        if roi is None or not isinstance(roi, LineROI):
            self.current_roi = None
            self.profile_widget.clear()
            self.status_label.setText("No LineROI selected")
            return

        self._update_profile(roi)

    def _on_roi_modified(self, roi):
        """Handle ROI modification (dragging)."""
        if self._is_shutting_down:
            return

        # Only update if this is our current ROI
        if roi == self.current_roi and isinstance(roi, LineROI):
            self._update_profile(roi)

    def _update_profile(self, roi):
        """Extract and display profile for a LineROI."""
        self.current_roi = roi

        if self.active_window is None:
            return

        # Get the current slice cache from renderer (C, Y, X)
        cache = self.active_window.renderer.current_slice_cache
        if cache is None:
            self.status_label.setText("No image data")
            return

        # Extract profile
        try:
            profiles = roi.get_profile(cache)  # Returns (C, num_points)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            return

        # Calculate distance array
        p1 = roi.data.get("p1", (0, 0))
        p2 = roi.data.get("p2", (0, 0))
        length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        distances = np.linspace(0, length, profiles.shape[-1])

        # Get channel colors
        colors = self.active_window.renderer.channel_colors[:profiles.shape[0]]

        # Update channel toggles if number of channels changed
        if self.channel_toggle is None or len(colors) != len(self.channel_toggle.checkboxes):
            self._setup_channel_toggles(profiles.shape[0], colors)

        # Update profile widget
        self.profile_widget.set_data(distances, profiles, colors)

        # Update status bar
        self.status_label.setText(
            f"Length: {length:.1f} px | ({p1[0]:.0f}, {p1[1]:.0f}) \u2192 ({p2[0]:.0f}, {p2[1]:.0f})"
        )

    def _setup_channel_toggles(self, num_channels, colors):
        """Create or update channel toggle row."""
        # Remove old toggle if exists
        if self.channel_toggle is not None:
            self.channel_toggle.setParent(None)
            self.channel_toggle.deleteLater()

        # Don't show toggles for single channel
        if num_channels <= 1:
            self.channel_toggle = None
            return

        self.channel_toggle = ChannelToggleRow(num_channels, colors)
        self.channel_toggle.visibilityChanged.connect(self._on_channel_visibility_changed)
        self.toggle_layout.addWidget(self.channel_toggle)

    def _on_channel_visibility_changed(self, channel_idx, visible):
        """Handle channel visibility toggle."""
        self.profile_widget.set_channel_visible(channel_idx, visible)

    def showEvent(self, event):
        """Handle dialog show event."""
        super().showEvent(event)
        # Try to find active window and selected LineROI
        for window in manager.get_all().values():
            self._connect_window(window)
            for roi in window.rois:
                if roi.selected and isinstance(roi, LineROI):
                    self.active_window = window
                    self._update_profile(roi)
                    return

    def closeEvent(self, event):
        """Override close to hide instead of destroy."""
        if self._is_shutting_down:
            super().closeEvent(event)
        else:
            event.ignore()
            self.hide()

    def cleanup(self):
        """Prepare for application shutdown."""
        self._is_shutting_down = True

        try:
            manager.window_registered.disconnect(self._on_window_registered)
        except (TypeError, RuntimeError):
            pass

        for window in list(self._connected_windows):
            self._disconnect_window(window)

        self.active_window = None
        self.current_roi = None
