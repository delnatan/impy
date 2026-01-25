"""
Tiled Image Viewer for displaying multiple images in a gallery layout.

Inspired by single-particle cryo-EM software (RELION, cryoSPARC).
Supports paginated viewing of many images with GPU-accelerated rendering.
"""

import os
from pathlib import Path

import numpy as np
from qtpy.QtCore import QPoint, QRect, QSize, Qt
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)
from superqt import QRangeSlider
from vispy import scene

from .io import load_image
from .visuals import (
    COLORMAPS,
    DEFAULT_CHANNEL_COLORMAPS,
    CompositeImageVisual,
    get_colormap,
)
from .widgets import CompactHistogramWidget, ContrastDialog
from .widgets.axes_dialog import AxesDialog


class TiledVisualProxy:
    """
    Proxy that broadcasts visual settings to all tile renderers.
    Used by TiledChannelPanel to control all tiles simultaneously.

    Unlike individual tile contrast (which remains per-tile), this proxy
    handles global settings: colormap, gamma, and channel visibility.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self._max_channels = 1

        # Global settings (applied to all tiles)
        self._channel_colormaps = {}  # channel_idx -> colormap_name
        self._channel_gammas = {}  # channel_idx -> gamma value
        self._channel_visibility = {}  # channel_idx -> bool
        self._channel_colors = list(
            DEFAULT_CHANNEL_COLORMAPS
        )  # Display colors

    def update_max_channels(self, max_c):
        """Update the maximum number of channels across all tiles."""
        self._max_channels = max_c
        # Initialize defaults for new channels
        for c in range(max_c):
            if c not in self._channel_colormaps:
                self._channel_colormaps[c] = DEFAULT_CHANNEL_COLORMAPS[
                    c % len(DEFAULT_CHANNEL_COLORMAPS)
                ]
            if c not in self._channel_gammas:
                self._channel_gammas[c] = 1.0
            if c not in self._channel_visibility:
                self._channel_visibility[c] = True

    @property
    def channel_colors(self):
        """Return display colors for histogram rendering."""
        return self._channel_colors

    def _get_tile_renderers(self):
        """Get all renderers from loaded tiles."""
        return [
            t.renderer
            for t in self.viewer.tile_widgets
            if t.renderer is not None
        ]

    def set_colormap(self, channel_idx, cmap_name):
        """Set colormap for a channel across all tiles."""
        self._channel_colormaps[channel_idx] = cmap_name

        # Update display color
        _, display_color = get_colormap(cmap_name)
        if display_color and channel_idx < len(self._channel_colors):
            self._channel_colors[channel_idx] = display_color
        elif not display_color:
            # For matplotlib colormaps, sample a representative color
            import matplotlib.cm as mpl_cm

            spec = COLORMAPS.get(cmap_name)
            if isinstance(spec, str) and spec.startswith("mpl:"):
                mpl_name = spec[4:]
                mpl_cmap = mpl_cm.get_cmap(mpl_name)
                rgb = mpl_cmap(0.75)[:3]
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                )
                if channel_idx < len(self._channel_colors):
                    self._channel_colors[channel_idx] = hex_color

        # Apply to all tiles
        for renderer in self._get_tile_renderers():
            if channel_idx < len(renderer.layers):
                renderer.set_colormap(channel_idx, cmap_name)

    def get_colormap_name(self, channel_idx):
        """Get colormap name for a channel."""
        return self._channel_colormaps.get(channel_idx, "White")

    def set_gamma(self, channel_idx, gamma):
        """Set gamma for a channel across all tiles."""
        self._channel_gammas[channel_idx] = gamma
        for renderer in self._get_tile_renderers():
            if channel_idx < len(renderer.layers):
                renderer.set_gamma(channel_idx, gamma)

    def get_gamma(self, channel_idx):
        """Get gamma for a channel."""
        return self._channel_gammas.get(channel_idx, 1.0)

    def set_channel_visible(self, channel_idx, visible):
        """Set visibility for a channel across all tiles."""
        self._channel_visibility[channel_idx] = visible
        for renderer in self._get_tile_renderers():
            if channel_idx < len(renderer.layers):
                renderer.set_channel_visible(channel_idx, visible)

    def get_channel_visible(self, channel_idx):
        """Get visibility state for a channel."""
        return self._channel_visibility.get(channel_idx, True)

    def set_clim(self, channel_idx, vmin, vmax):
        """Set contrast limits for a channel across all tiles."""
        for renderer in self._get_tile_renderers():
            if channel_idx < len(renderer.layers):
                renderer.set_clim(channel_idx, vmin, vmax)

    def get_aggregate_data(self, channel_idx):
        """
        Get aggregated intensity data for a channel across all tiles.
        Returns concatenated data suitable for histogram computation.
        """
        all_data = []
        for tile in self.viewer.tile_widgets:
            if (
                tile.renderer is None
                or tile.renderer.current_slice_cache is None
            ):
                continue
            cache = tile.renderer.current_slice_cache
            if channel_idx < cache.shape[0]:
                plane = cache[channel_idx]
                # Sample the data to keep histogram computation fast
                # Take every Nth pixel if image is large
                total_pixels = plane.size
                if total_pixels > 100000:
                    # Sample ~10000 pixels
                    step = max(1, int(np.sqrt(total_pixels / 10000)))
                    sampled = plane[::step, ::step].ravel()
                else:
                    sampled = plane.ravel()
                all_data.append(sampled)

        if all_data:
            return np.concatenate(all_data)
        return None

    def apply_settings_to_tile(self, tile):
        """Apply current global settings to a newly loaded tile."""
        if tile.renderer is None:
            return
        for c in range(len(tile.renderer.layers)):
            if c in self._channel_colormaps:
                tile.renderer.set_colormap(c, self._channel_colormaps[c])
            if c in self._channel_gammas:
                tile.renderer.set_gamma(c, self._channel_gammas[c])
            if c in self._channel_visibility:
                tile.renderer.set_channel_visible(
                    c, self._channel_visibility[c]
                )


class TiledChannelRow(QWidget):
    """
    A single row representing one channel in the tiled viewer panel.
    Similar to ChannelRow but designed for aggregate display across multiple tiles.
    """

    visibilityChanged = None  # Will be set as Signal
    colormapChanged = None
    gammaChanged = None

    def __init__(self, channel_idx, channel_name, color, parent=None):
        super().__init__(parent)
        self.channel_idx = channel_idx

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Visibility checkbox
        self.chk_visible = QCheckBox()
        self.chk_visible.setChecked(True)
        self.chk_visible.setToolTip("Toggle channel visibility (all tiles)")
        self.chk_visible.toggled.connect(self._on_visibility_changed)
        layout.addWidget(self.chk_visible)

        # Color swatch (button for colormap menu)
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(20, 20)
        self.color_btn.setCursor(Qt.PointingHandCursor)
        self.color_btn.setToolTip("Change colormap (all tiles)")
        self._update_color_swatch(color)
        self.color_btn.clicked.connect(self._show_colormap_menu)
        layout.addWidget(self.color_btn)

        # Channel name label
        self.name_label = QLabel(channel_name)
        self.name_label.setFixedWidth(40)
        self.name_label.setStyleSheet("color: #EEE; font-size: 11px;")
        layout.addWidget(self.name_label)

        # Min spinbox for contrast
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(1)
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setSingleStep(10)
        self.min_spin.setFixedWidth(65)
        self.min_spin.setToolTip("Minimum intensity (all tiles)")
        self.min_spin.valueChanged.connect(self._on_min_changed)
        layout.addWidget(self.min_spin)

        # Compact histogram (shows aggregate distribution, interactive)
        self.histogram = CompactHistogramWidget()
        self.histogram.climChanged.connect(self._on_histogram_clim_changed)
        layout.addWidget(self.histogram, 1)

        # Max spinbox for contrast
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(1)
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setSingleStep(10)
        self.max_spin.setFixedWidth(65)
        self.max_spin.setToolTip("Maximum intensity (all tiles)")
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
        self.gamma_spin.setToolTip("Gamma correction (all tiles)")
        self.gamma_spin.valueChanged.connect(self._on_gamma_changed)
        layout.addWidget(self.gamma_spin)

        self.current_colormap = "White"

        # Callbacks (set by parent)
        self._visibility_callback = None
        self._colormap_callback = None
        self._gamma_callback = None
        self._clim_callback = None

    def set_callbacks(
        self, visibility_cb, colormap_cb, gamma_cb, clim_cb=None
    ):
        """Set callback functions for changes."""
        self._visibility_callback = visibility_cb
        self._colormap_callback = colormap_cb
        self._gamma_callback = gamma_cb
        self._clim_callback = clim_cb

    def _update_color_swatch(self, color):
        """Update the color swatch button background."""
        self.color_btn.setStyleSheet(
            f"background-color: {color}; border: 1px solid #555; border-radius: 3px;"
        )

    def _on_visibility_changed(self, checked):
        if self._visibility_callback:
            self._visibility_callback(self.channel_idx, checked)

    def _on_gamma_changed(self, value):
        if self._gamma_callback:
            self._gamma_callback(self.channel_idx, value)

    def _on_min_changed(self, value):
        """Handle min spinbox change."""
        max_val = self.max_spin.value()
        if value < max_val and self._clim_callback:
            self._clim_callback(self.channel_idx, value, max_val)
            # Update histogram display
            self.histogram.blockSignals(True)
            self.histogram.set_clim(value, max_val)
            self.histogram.blockSignals(False)

    def _on_max_changed(self, value):
        """Handle max spinbox change."""
        min_val = self.min_spin.value()
        if value > min_val and self._clim_callback:
            self._clim_callback(self.channel_idx, min_val, value)
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
        # Notify parent
        if self._clim_callback:
            self._clim_callback(self.channel_idx, vmin, vmax)

    def _show_colormap_menu(self):
        """Show a popup menu for colormap selection."""
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
        if self._colormap_callback:
            self._colormap_callback(self.channel_idx, cmap_name)

    def set_data(self, data_slice, color):
        """Update histogram data and color."""
        self._update_color_swatch(color)
        if data_slice is not None:
            self.histogram.set_data(data_slice, color)

    def set_clim(self, vmin, vmax):
        """Update histogram and spinbox display range."""
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
        """Update checkbox state without triggering callback."""
        self.chk_visible.blockSignals(True)
        self.chk_visible.setChecked(visible)
        self.chk_visible.blockSignals(False)

    def set_gamma(self, gamma):
        """Update gamma spinbox without triggering callback."""
        self.gamma_spin.blockSignals(True)
        self.gamma_spin.setValue(gamma)
        self.gamma_spin.blockSignals(False)


class TiledChannelPanel(QDialog):
    """
    Floating dialog for global channel control in TiledViewer.
    Controls colormap, gamma, and visibility across all tiles.
    Shows aggregate histogram from all visible tiles.
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.proxy = viewer.visual_proxy

        self.setWindowTitle("Channels (All Tiles)")
        self.setWindowFlags(Qt.Tool)
        self.resize(480, min(180 + viewer.max_C * 55, 450))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Info label
        info_label = QLabel(
            f"<b>Global Channel Settings</b> ({viewer.max_C} channels)"
        )
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        # Note about global controls
        note_label = QLabel(
            "<i>Adjust min/max to set global contrast. "
            "Use Auto All for per-tile optimization.</i>"
        )
        note_label.setStyleSheet("color: #888; font-size: 10px;")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)

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

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_auto = QPushButton("Auto Contrast All")
        btn_auto.setToolTip(
            "Apply percentile-based auto-contrast to each tile"
        )
        btn_auto.clicked.connect(self._auto_contrast_all)
        btn_layout.addWidget(btn_auto)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Initial data load
        self.refresh_ui()

    def _setup_channel_rows(self):
        """Create a row widget for each channel."""
        for c in range(self.viewer.max_C):
            ch_name = f"Ch {c + 1}"
            color = self.proxy.channel_colors[
                c % len(self.proxy.channel_colors)
            ]

            row = TiledChannelRow(c, ch_name, color)
            row.set_callbacks(
                self._on_visibility_changed,
                self._on_colormap_changed,
                self._on_gamma_changed,
                self._on_clim_changed,
            )

            self.channel_rows.append(row)
            self.rows_layout.addWidget(row)

        self.rows_layout.addStretch()

    def _on_visibility_changed(self, channel_idx, visible):
        """Handle visibility toggle for a channel."""
        self.proxy.set_channel_visible(channel_idx, visible)
        self._update_all_canvases()

    def _on_colormap_changed(self, channel_idx, cmap_name):
        """Handle colormap change for a channel."""
        self.proxy.set_colormap(channel_idx, cmap_name)
        self._update_all_canvases()

        # Update color swatch
        color = self.proxy.channel_colors[
            channel_idx % len(self.proxy.channel_colors)
        ]
        self.channel_rows[channel_idx]._update_color_swatch(color)

        # Refresh histogram with new color
        self.refresh_ui()

    def _on_gamma_changed(self, channel_idx, gamma):
        """Handle gamma change for a channel."""
        self.proxy.set_gamma(channel_idx, gamma)
        self._update_all_canvases()

    def _on_clim_changed(self, channel_idx, vmin, vmax):
        """Handle contrast limits change for a channel."""
        self.proxy.set_clim(channel_idx, vmin, vmax)
        self._update_all_canvases()

    def _update_all_canvases(self):
        """Update all tile canvases."""
        for tile in self.viewer.tile_widgets:
            if tile.canvas:
                tile.canvas.update()

    def _auto_contrast_all(self):
        """Apply auto-contrast to all tiles (per-tile percentile-based)."""
        self.viewer._auto_contrast_all()
        self.refresh_ui()

    def refresh_ui(self):
        """Refresh all channel rows with current data."""
        for c, row in enumerate(self.channel_rows):
            # Get aggregate data for this channel
            agg_data = self.proxy.get_aggregate_data(c)
            color = self.proxy.channel_colors[
                c % len(self.proxy.channel_colors)
            ]

            if agg_data is not None and agg_data.size > 0:
                row.set_data(agg_data, color)

                # Set clim based on data range (for histogram display only)
                valid = agg_data[agg_data > 0]
                if valid.size > 0:
                    mn, mx = np.nanpercentile(valid, (0.5, 99.5))
                    row.set_clim(mn, mx)
            else:
                row.set_data(None, color)

            # Update visibility state
            visible = self.proxy.get_channel_visible(c)
            row.set_visible_state(visible)

            # Update gamma
            gamma = self.proxy.get_gamma(c)
            row.set_gamma(gamma)


class FlowLayout(QLayout):
    """
    Layout that arranges widgets in a flowing left-to-right, top-to-bottom manner.
    Similar to CSS flexbox with wrap.
    """

    def __init__(self, parent=None, spacing=8):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margin = self.contentsMargins()
        size += QSize(
            margin.left() + margin.right(), margin.top() + margin.bottom()
        )
        return size

    def _do_layout(self, rect, test_only=False):
        """Arrange items in flow layout within given rect."""
        left, top, right, bottom = self.getContentsMargins()
        effective_rect = rect.adjusted(left, top, -right, -bottom)

        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0

        for item in self._items:
            widget = item.widget()
            if widget is None:
                continue

            space_x = self._spacing
            space_y = self._spacing

            item_width = widget.sizeHint().width()
            item_height = widget.sizeHint().height()

            next_x = x + item_width + space_x

            # Wrap to next row if exceeded width
            if (
                next_x - space_x > effective_rect.right() + 1
                and line_height > 0
            ):
                x = effective_rect.x()
                y = y + line_height + space_y
                next_x = x + item_width + space_x
                line_height = 0

            if not test_only:
                widget.setGeometry(
                    QRect(QPoint(x, y), QSize(item_width, item_height))
                )

            x = next_x
            line_height = max(line_height, item_height)

        return y + line_height - rect.y() + bottom


class TileWidget(QFrame):
    """
    Single tile displaying one image with Vispy canvas.
    Supports independent pan/zoom within the tile.
    """

    def __init__(self, tile_size=200, parent=None):
        super().__init__(parent)
        self._tile_size = tile_size
        self._parent_viewer = parent

        # Data
        self.data = None
        self.meta = None
        self.file_path = None

        # Current view state (for info label)
        self._current_t = 0
        self._current_z = 0
        self._current_mode = "composite"
        self._current_channel = 0
        self._is_projection = False
        self._proj_range = (0, 0)
        self._show_info = True  # Whether to show info label

        # Camera sync callback
        self._camera_change_callback = None
        self._ignore_camera_events = False

        # Setup UI
        self._setup_ui()

        # Context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _setup_ui(self):
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setStyleSheet(
            """
            TileWidget {
                background-color: #1a1a1a;
                border: 1px solid #333;
            }
            TileWidget:hover {
                border: 1px solid #555;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Vispy canvas
        self.canvas = scene.SceneCanvas(
            keys=None, bgcolor="#1a1a1a", show=False
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"
        self.view.camera.aspect = 1

        # Connect to canvas events for camera sync (wheel=zoom, release=end of pan)
        self.canvas.events.mouse_wheel.connect(self._on_camera_change)
        self.canvas.events.mouse_release.connect(self._on_camera_change)

        # Canvas widget
        self.canvas.native.setMinimumSize(50, 50)
        layout.addWidget(self.canvas.native, 1)

        # Info label (shows filename + view state)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(
            "color: #aaa; font-size: 10px; padding: 2px; background: transparent;"
        )
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setFixedHeight(32)  # Two lines
        layout.addWidget(self.info_label, 0)

        # Renderer (created when data is loaded)
        self.renderer = None

        self._update_size()

    def _update_size(self):
        """Update widget size based on tile_size and info visibility."""
        label_height = 32 if self._show_info else 0
        spacing = 8 if self._show_info else 4
        total_height = self._tile_size + label_height + spacing
        self.setFixedSize(self._tile_size, total_height)

    def sizeHint(self):
        label_height = 32 if self._show_info else 0
        spacing = 8 if self._show_info else 4
        total_height = self._tile_size + label_height + spacing
        return QSize(self._tile_size, total_height)

    def set_show_info(self, show: bool):
        """Show or hide the info label."""
        self._show_info = show
        self.info_label.setVisible(show)
        self._update_size()

    def set_tile_size(self, size):
        """Update tile size (does not affect internal zoom)."""
        self._tile_size = size
        self._update_size()
        if self.renderer is not None:
            self._fit_view()

    def get_shape(self):
        """Return image shape (T, Z, C, Y, X) or None if not loaded."""
        if self.data is not None:
            return self.data.shape
        return None

    def load(self, path, dims=None):
        """Load image from path.

        Args:
            path: Image file path
            dims: Optional dimension string for axis reordering (e.g., 'zyx', 'tzcyx')
        """
        self.file_path = path
        self._dims = dims  # Store for potential reload
        try:
            self.data, self.meta = load_image(path, dims=dims)

            # Create renderer
            self.renderer = CompositeImageVisual(self.view, self.data)

            # Display first frame/slice (middle Z)
            z_mid = self.data.shape[1] // 2
            self._current_z = z_mid
            self.renderer.update_slice(0, z_mid)
            self._fit_view()
            self._update_info()

        except Exception as e:
            print(f"Error loading {path}: {e}")
            self.info_label.setText("Load error")
            self.info_label.setStyleSheet("color: #ff6666; font-size: 10px;")

    def unload(self):
        """Release memory and close file handles."""
        if self.renderer is not None:
            # Clear visuals
            for layer in self.renderer.layers:
                layer.parent = None
            self.renderer = None

        # Close data if it has a close method (ImageBuffer, Imaris5DProxy)
        if self.data is not None and hasattr(self.data, "close"):
            try:
                self.data.close()
            except Exception:
                pass

        self.data = None
        self.meta = None

    def update_view(
        self,
        t_idx=0,
        z_idx=None,
        mode="composite",
        channel_idx=0,
        projection=False,
        proj_range=None,
    ):
        """
        Update the displayed slice with global settings.
        Handles bounds checking for this image's dimensions.
        """
        if self.renderer is None or self.data is None:
            return

        T, Z, C, H, W = self.data.shape

        # Bounds check
        t_idx = min(t_idx, T - 1)

        if projection and proj_range is not None:
            # Clamp projection range to this image's Z extent
            z_min = min(proj_range[0], Z - 1)
            z_max = min(proj_range[1], Z - 1)
            z_slice = slice(z_min, z_max + 1)
            self._is_projection = True
            self._proj_range = (z_min, z_max)
        else:
            z_idx = min(z_idx if z_idx is not None else 0, Z - 1)
            z_slice = z_idx
            self._is_projection = False
            self._current_z = z_idx

        self._current_t = t_idx
        self._current_mode = mode
        self._current_channel = min(channel_idx, C - 1)

        # Update renderer mode
        self.renderer.set_mode(mode)
        if mode == "single":
            self.renderer.set_active_channel(self._current_channel)

        # Update slice
        self.renderer.update_slice(t_idx, z_slice)
        self.canvas.update()
        self._update_info()

    def _fit_view(self):
        """Reset camera to fit image."""
        if self.data is not None:
            _, _, _, h, w = self.data.shape
            self.view.camera.set_range(x=(0, w), y=(0, h), margin=0.02)
            self.view.camera.flip = (False, True, False)
            self.canvas.update()

    def _on_camera_change(self, event):
        """Handle camera change event (pan/zoom changed)."""
        if self._ignore_camera_events:
            return
        if self._camera_change_callback is not None:
            self._camera_change_callback(self)

    def set_camera_callback(self, callback):
        """Set callback for camera change events."""
        self._camera_change_callback = callback

    def get_camera_rect(self):
        """Get current camera view rect."""
        return self.view.camera.rect

    def set_camera_rect(self, rect):
        """Set camera view rect (for synchronization)."""
        self._ignore_camera_events = True
        try:
            self.view.camera.rect = rect
            self.canvas.update()
        finally:
            self._ignore_camera_events = False

    def _update_info(self):
        """Update info label with filename and view state."""
        if not self.file_path or self.data is None:
            return

        name = Path(self.file_path).stem
        # Truncate filename if too long
        max_len = self._tile_size // 7
        if len(name) > max_len:
            name = name[: max_len - 2] + ".."

        T, Z, C, H, W = self.data.shape

        # Build state string
        state_parts = []

        # Z info
        if Z > 1:
            if self._is_projection:
                state_parts.append(
                    f"z:{self._proj_range[0]}-{self._proj_range[1]} (max)"
                )
            else:
                state_parts.append(f"z:{self._current_z}/{Z - 1}")

        # Channel info (only in single mode or if multiple channels)
        if C > 1:
            if self._current_mode == "single":
                state_parts.append(f"ch:{self._current_channel}/{C - 1}")
            else:
                state_parts.append(f"{C}ch")

        # Time info
        if T > 1:
            state_parts.append(f"t:{self._current_t}/{T - 1}")

        state_str = " | ".join(state_parts) if state_parts else ""

        # Format: filename on first line, state on second
        if state_str:
            self.info_label.setText(f"{name}\n{state_str}")
        else:
            self.info_label.setText(name)

        self.info_label.setToolTip(self.file_path)

    def _show_context_menu(self, pos):
        """Show right-click context menu."""
        menu = QMenu(self)

        # Auto contrast
        auto_action = menu.addAction("Auto Contrast")
        auto_action.triggered.connect(self._auto_contrast)

        # Contrast dialog
        contrast_action = menu.addAction("Adjust Contrast...")
        contrast_action.triggered.connect(self._show_contrast_dialog)

        menu.addSeparator()

        # Reset view
        reset_action = menu.addAction("Reset View (A)")
        reset_action.triggered.connect(self._fit_view)

        menu.addSeparator()

        # Open in viewer
        open_action = menu.addAction("Open in Viewer")
        open_action.triggered.connect(self._open_in_viewer)

        # Show metadata
        meta_action = menu.addAction("Show Info")
        meta_action.triggered.connect(self._show_metadata)

        menu.exec_(self.mapToGlobal(pos))

    def _auto_contrast(self):
        """Apply auto-contrast to all channels."""
        if self.renderer is None or self.renderer.current_slice_cache is None:
            return

        cache = self.renderer.current_slice_cache
        for c in range(cache.shape[0]):
            plane = cache[c]
            valid = plane[plane > 0]
            if valid.size > 0:
                mn, mx = np.nanpercentile(valid, (0.5, 99.5))
                self.renderer.set_clim(c, mn, max(mx, mn + 1))

        self.canvas.update()

    def _show_contrast_dialog(self):
        """Show contrast adjustment dialog."""
        if self.renderer is None:
            return

        # Create a temporary ImageWindow-like wrapper for ContrastDialog
        class TileWrapper:
            def __init__(self, tile):
                self.renderer = tile.renderer
                self.canvas = tile.canvas
                self.C = tile.data.shape[2] if tile.data is not None else 1
                self.meta = tile.meta or {}

        wrapper = TileWrapper(self)
        dlg = ContrastDialog(wrapper, parent=self)
        dlg.exec_()

    def _open_in_viewer(self):
        """Open this image in a full ImageWindow."""
        if self.file_path:
            from .ui import ImageWindow

            viewer = ImageWindow(self.file_path)
            viewer.show()

    def _show_metadata(self):
        """Show metadata dialog."""
        if self.meta:
            from .widgets import MetadataDialog

            dlg = MetadataDialog(self.meta, parent=self)
            dlg.exec_()


class TiledViewer(QMainWindow):
    """
    Gallery view for multiple images with flow layout and pagination.
    """

    # Maximum tiles per page options
    TILES_PER_PAGE_OPTIONS = [25, 50, 100]

    def __init__(self, image_paths, tiles_per_page=50, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.image_paths = image_paths
        self.tiles_per_page = tiles_per_page
        self.current_page = 0
        self.tile_size = 200  # Default tile size in pixels

        # Global view state
        self.t_idx = 0
        self.z_idx = 0
        self.mode = "composite"
        self.channel_idx = 0
        self.z_projection = False
        self.z_proj_range = (0, 0)

        # Sync pan/zoom state
        self.sync_pan_zoom = False
        self._syncing_camera = False  # Flag to prevent recursive updates

        # Detected max dimensions across all images (populated on first load)
        self.max_T = 1
        self.max_Z = 1
        self.max_C = 1

        # Display settings
        self.show_info = True  # Whether to show info labels on tiles

        # Axis ordering (None = use default from load_image)
        self._current_dims = None

        self.tile_widgets = []

        # Visual proxy for global channel settings
        self.visual_proxy = TiledVisualProxy(self)

        # Channel panel (created lazily)
        self.channel_panel = None

        self._setup_ui()
        self._setup_menu()
        self._load_current_page()

    def _setup_ui(self):
        self.setWindowTitle(f"Tiled Viewer - {len(self.image_paths)} images")
        self.resize(1000, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # === Toolbar Row 1: Page navigation, tiles/page, tile size ===
        toolbar1 = QWidget()
        toolbar1_layout = QHBoxLayout(toolbar1)
        toolbar1_layout.setContentsMargins(5, 5, 5, 5)

        # Page navigation
        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedWidth(30)
        self.prev_btn.clicked.connect(self._prev_page)
        toolbar1_layout.addWidget(self.prev_btn)

        self.page_label = QLabel("Page 1 / 1")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setFixedWidth(100)
        toolbar1_layout.addWidget(self.page_label)

        self.next_btn = QPushButton(">")
        self.next_btn.setFixedWidth(30)
        self.next_btn.clicked.connect(self._next_page)
        toolbar1_layout.addWidget(self.next_btn)

        toolbar1_layout.addSpacing(20)

        # Tiles per page
        toolbar1_layout.addWidget(QLabel("Tiles/page:"))
        self.per_page_combo = QComboBox()
        for n in self.TILES_PER_PAGE_OPTIONS:
            self.per_page_combo.addItem(str(n), n)
        idx = (
            self.TILES_PER_PAGE_OPTIONS.index(self.tiles_per_page)
            if self.tiles_per_page in self.TILES_PER_PAGE_OPTIONS
            else 1
        )
        self.per_page_combo.setCurrentIndex(idx)
        self.per_page_combo.currentIndexChanged.connect(
            self._on_per_page_changed
        )
        toolbar1_layout.addWidget(self.per_page_combo)

        toolbar1_layout.addSpacing(20)

        # Tile size slider
        toolbar1_layout.addWidget(QLabel("Tile size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(100, 400)
        self.size_slider.setValue(self.tile_size)
        self.size_slider.setFixedWidth(150)
        self.size_slider.valueChanged.connect(self._on_tile_size_changed)
        toolbar1_layout.addWidget(self.size_slider)

        self.size_label = QLabel(f"{self.tile_size}px")
        self.size_label.setFixedWidth(50)
        toolbar1_layout.addWidget(self.size_label)

        toolbar1_layout.addSpacing(20)

        # Show info checkbox
        self.show_info_check = QCheckBox("Show Info (I)")
        self.show_info_check.setChecked(True)
        self.show_info_check.toggled.connect(self._on_show_info_toggled)
        toolbar1_layout.addWidget(self.show_info_check)

        toolbar1_layout.addStretch()

        main_layout.addWidget(toolbar1)

        # === Toolbar Row 2: Mode, Channel, Z controls ===
        toolbar2 = QWidget()
        toolbar2_layout = QHBoxLayout(toolbar2)
        toolbar2_layout.setContentsMargins(5, 2, 5, 5)

        # Mode selector
        toolbar2_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Composite", "Single Channel"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        toolbar2_layout.addWidget(self.mode_combo)

        toolbar2_layout.addSpacing(10)

        # Channel selector (hidden initially)
        self.channel_widget = QWidget()
        channel_layout = QHBoxLayout(self.channel_widget)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.addWidget(QLabel("Channel:"))
        self.channel_slider = QSlider(Qt.Horizontal)
        self.channel_slider.setRange(0, 0)
        self.channel_slider.setFixedWidth(80)
        self.channel_slider.valueChanged.connect(self._on_channel_changed)
        channel_layout.addWidget(self.channel_slider)
        self.channel_label = QLabel("0")
        self.channel_label.setFixedWidth(20)
        channel_layout.addWidget(self.channel_label)
        self.channel_widget.setVisible(False)
        toolbar2_layout.addWidget(self.channel_widget)

        toolbar2_layout.addSpacing(20)

        # Z controls
        self.z_widget = QWidget()
        z_layout = QHBoxLayout(self.z_widget)
        z_layout.setContentsMargins(0, 0, 0, 0)

        z_layout.addWidget(QLabel("Z:"))
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, 0)
        self.z_slider.setFixedWidth(100)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        z_layout.addWidget(self.z_slider)
        self.z_label = QLabel("0")
        self.z_label.setFixedWidth(25)
        z_layout.addWidget(self.z_label)

        # Max projection checkbox
        self.proj_check = QCheckBox("Max Proj")
        self.proj_check.toggled.connect(self._on_projection_toggled)
        z_layout.addWidget(self.proj_check)

        # Projection range slider (hidden initially)
        self.proj_range_widget = QWidget()
        proj_range_layout = QHBoxLayout(self.proj_range_widget)
        proj_range_layout.setContentsMargins(0, 0, 0, 0)
        self.proj_range_slider = QRangeSlider(Qt.Horizontal)
        self.proj_range_slider.setRange(0, 0)
        self.proj_range_slider.setValue((0, 0))
        self.proj_range_slider.setFixedWidth(100)
        self.proj_range_slider.valueChanged.connect(
            self._on_proj_range_changed
        )
        proj_range_layout.addWidget(self.proj_range_slider)
        self.proj_range_label = QLabel("0-0")
        self.proj_range_label.setFixedWidth(40)
        proj_range_layout.addWidget(self.proj_range_label)
        self.proj_range_widget.setVisible(False)
        z_layout.addWidget(self.proj_range_widget)

        toolbar2_layout.addWidget(self.z_widget)

        toolbar2_layout.addSpacing(20)

        # Auto contrast all button
        self.auto_all_btn = QPushButton("Auto All (C)")
        self.auto_all_btn.clicked.connect(self._auto_contrast_all)
        toolbar2_layout.addWidget(self.auto_all_btn)

        # Reset all views button
        self.reset_all_btn = QPushButton("Reset Views (A)")
        self.reset_all_btn.clicked.connect(self._reset_all_views)
        toolbar2_layout.addWidget(self.reset_all_btn)

        toolbar2_layout.addSpacing(10)

        # Sync pan/zoom checkbox
        self.sync_panzoom_check = QCheckBox("Sync Pan/Zoom (S)")
        self.sync_panzoom_check.setToolTip(
            "Synchronize pan and zoom across all tiles"
        )
        self.sync_panzoom_check.toggled.connect(self._on_sync_panzoom_toggled)
        toolbar2_layout.addWidget(self.sync_panzoom_check)

        toolbar2_layout.addStretch()

        main_layout.addWidget(toolbar2)

        # Scroll area for tiles
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container with flow layout
        self.flow_container = QWidget()
        self.flow_layout = FlowLayout(self.flow_container, spacing=0)
        self.scroll_area.setWidget(self.flow_container)

        main_layout.addWidget(self.scroll_area, 1)

        # Status bar
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        main_layout.addWidget(self.status_label)

        self._update_page_controls()

    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # Adjust menu
        adjust_menu = menubar.addMenu("Adjust")

        channels_action = QAction("Channels...", self)
        channels_action.setShortcut("Shift+H")
        channels_action.triggered.connect(self.show_channel_panel)
        adjust_menu.addAction(channels_action)

        adjust_menu.addSeparator()

        auto_all_action = QAction("Auto Contrast All", self)
        auto_all_action.setShortcut("C")
        auto_all_action.triggered.connect(self._auto_contrast_all)
        adjust_menu.addAction(auto_all_action)

        reset_all_action = QAction("Reset All Views", self)
        reset_all_action.setShortcut("A")
        reset_all_action.triggered.connect(self._reset_all_views)
        adjust_menu.addAction(reset_all_action)

        adjust_menu.addSeparator()

        axes_action = QAction("Reorder Axes...", self)
        axes_action.triggered.connect(self.show_axes_dialog)
        adjust_menu.addAction(axes_action)

    def show_channel_panel(self):
        """Show the global channel control panel."""
        if self.channel_panel is None:
            self.channel_panel = TiledChannelPanel(self, parent=self)
        self.channel_panel.show()
        self.channel_panel.raise_()
        self.channel_panel.refresh_ui()

    def show_axes_dialog(self):
        """Show dialog to reorder axes for ambiguous TIFF dimensions."""
        from qtpy.QtWidgets import QMessageBox

        # Find a tile with raw_shape metadata
        raw_shape = None
        for tile in self.tile_widgets:
            if tile.meta and "raw_shape" in tile.meta:
                raw_shape = tile.meta["raw_shape"]
                break

        if raw_shape is None:
            QMessageBox.information(
                self,
                "Reorder Axes",
                "Axes reordering is only available for TIFF/PNG/JPEG images.",
            )
            return

        dlg = AxesDialog(raw_shape, parent=self)
        if dlg.exec_():
            dims = dlg.get_dims_string()
            self.reorder_axes(dims)

    def reorder_axes(self, dims):
        """
        Reorder axes for all tiles using a new dimension string.

        Args:
            dims: Dimension string (e.g., 'tyx', 'zyx', 'zcyx', 'tzyx')
        """
        # Store the dims for future page loads
        self._current_dims = dims

        # Reload all current tiles with the new dimension ordering
        for tile in self.tile_widgets:
            if tile.file_path:
                tile.unload()
                tile.load(tile.file_path, dims=dims)
                self.visual_proxy.apply_settings_to_tile(tile)

        # Update dimension controls based on reloaded images
        self._update_dimension_controls()

        # Apply current global settings (mode, z-slice, etc.)
        self._apply_global_settings()

        # Refresh channel panel if open
        if self.channel_panel is not None and self.channel_panel.isVisible():
            self.channel_panel.refresh_ui()

    def _total_pages(self):
        return max(
            1,
            (len(self.image_paths) + self.tiles_per_page - 1)
            // self.tiles_per_page,
        )

    def _update_page_controls(self):
        """Update page navigation UI."""
        total = self._total_pages()
        self.page_label.setText(f"Page {self.current_page + 1} / {total}")
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < total - 1)

    def _update_status(self):
        """Update status bar."""
        start = self.current_page * self.tiles_per_page + 1
        end = min(start + len(self.tile_widgets) - 1, len(self.image_paths))
        self.status_label.setText(
            f"Showing {start}-{end} of {len(self.image_paths)} images"
        )

    def _update_dimension_controls(self):
        """Update dimension sliders based on loaded tiles."""
        # Find max dimensions across all loaded tiles
        max_T, max_Z, max_C = 1, 1, 1
        for tile in self.tile_widgets:
            shape = tile.get_shape()
            if shape:
                T, Z, C, H, W = shape
                max_T = max(max_T, T)
                max_Z = max(max_Z, Z)
                max_C = max(max_C, C)

        self.max_T = max_T
        self.max_Z = max_Z
        self.max_C = max_C

        # Update visual proxy with max channels
        self.visual_proxy.update_max_channels(max_C)

        # Update Z slider
        if max_Z > 1:
            self.z_slider.setRange(0, max_Z - 1)
            self.z_slider.setValue(min(self.z_idx, max_Z - 1))
            self.proj_range_slider.setRange(0, max_Z - 1)
            self.proj_range_slider.setValue((0, max_Z - 1))
            self.z_proj_range = (0, max_Z - 1)
            self.z_widget.setVisible(True)
        else:
            self.z_widget.setVisible(False)

        # Update channel slider
        if max_C > 1:
            self.channel_slider.setRange(0, max_C - 1)
            self.channel_slider.setValue(min(self.channel_idx, max_C - 1))
            # Show channel widget if in single mode
            self.channel_widget.setVisible(self.mode == "single")
        else:
            self.channel_widget.setVisible(False)

        self._update_labels()

    def _update_labels(self):
        """Update dimension labels."""
        self.z_label.setText(str(self.z_idx))
        self.channel_label.setText(str(self.channel_idx))
        self.proj_range_label.setText(
            f"{self.z_proj_range[0]}-{self.z_proj_range[1]}"
        )

    def _apply_global_settings(self):
        """Apply current global settings to all tiles."""
        for tile in self.tile_widgets:
            tile.update_view(
                t_idx=self.t_idx,
                z_idx=self.z_idx,
                mode=self.mode,
                channel_idx=self.channel_idx,
                projection=self.z_projection,
                proj_range=self.z_proj_range if self.z_projection else None,
            )

    def _load_current_page(self):
        """Load tiles for current page."""
        # Clear existing tiles
        self._clear_tiles()

        # Calculate slice
        start = self.current_page * self.tiles_per_page
        end = min(start + self.tiles_per_page, len(self.image_paths))

        # Create and load tiles
        for path in self.image_paths[start:end]:
            tile = TileWidget(self.tile_size, parent=self)
            tile.set_show_info(self.show_info)
            tile.set_camera_callback(self._on_tile_camera_changed)
            tile.load(path, dims=self._current_dims)
            # Apply global visual settings (colormap, gamma, visibility)
            self.visual_proxy.apply_settings_to_tile(tile)
            self.flow_layout.addWidget(tile)
            self.tile_widgets.append(tile)

        # Update dimension controls based on loaded images
        self._update_dimension_controls()

        # Apply current global settings (mode, z-slice, etc.)
        self._apply_global_settings()

        # Refresh channel panel if open
        if self.channel_panel is not None and self.channel_panel.isVisible():
            self.channel_panel.refresh_ui()

        # Update UI
        self._update_page_controls()
        self._update_status()

        # Force layout update
        self.flow_container.adjustSize()

    def _clear_tiles(self):
        """Remove and unload all current tiles."""
        for tile in self.tile_widgets:
            tile.unload()
            self.flow_layout.removeWidget(tile)
            tile.deleteLater()
        self.tile_widgets.clear()

    def _prev_page(self):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self._load_current_page()

    def _next_page(self):
        """Go to next page."""
        if self.current_page < self._total_pages() - 1:
            self.current_page += 1
            self._load_current_page()

    def _on_per_page_changed(self, index):
        """Handle tiles per page change."""
        self.tiles_per_page = self.per_page_combo.currentData()
        # Reset to first page and reload
        self.current_page = 0
        self._load_current_page()

    def _on_tile_size_changed(self, value):
        """Handle tile size slider change."""
        self.tile_size = value
        self.size_label.setText(f"{value}px")

        for tile in self.tile_widgets:
            tile.set_tile_size(value)

        # Trigger reflow
        self.flow_container.adjustSize()

    def _on_mode_changed(self, index):
        """Handle mode change."""
        self.mode = "composite" if index == 0 else "single"
        # Show/hide channel selector
        self.channel_widget.setVisible(
            self.mode == "single" and self.max_C > 1
        )
        self._apply_global_settings()

    def _on_channel_changed(self, value):
        """Handle channel slider change."""
        self.channel_idx = value
        self.channel_label.setText(str(value))
        self._apply_global_settings()

    def _on_z_changed(self, value):
        """Handle Z slider change."""
        self.z_idx = value
        self.z_label.setText(str(value))
        if not self.z_projection:
            self._apply_global_settings()

    def _on_projection_toggled(self, checked):
        """Handle projection checkbox toggle."""
        self.z_projection = checked
        self.z_slider.setVisible(not checked)
        self.proj_range_widget.setVisible(checked)
        self._apply_global_settings()

    def _on_proj_range_changed(self, value):
        """Handle projection range slider change."""
        self.z_proj_range = value
        self.proj_range_label.setText(f"{value[0]}-{value[1]}")
        if self.z_projection:
            self._apply_global_settings()

    def _auto_contrast_all(self):
        """Apply auto-contrast to all visible tiles."""
        for tile in self.tile_widgets:
            tile._auto_contrast()

    def _reset_all_views(self):
        """Reset view (fit image) for all tiles."""
        for tile in self.tile_widgets:
            tile._fit_view()

    def _on_show_info_toggled(self, checked):
        """Handle show info checkbox toggle."""
        self.show_info = checked
        for tile in self.tile_widgets:
            tile.set_show_info(checked)
        # Trigger reflow
        self.flow_container.adjustSize()

    def _on_sync_panzoom_toggled(self, checked):
        """Handle sync pan/zoom checkbox toggle."""
        self.sync_pan_zoom = checked

    def _on_tile_camera_changed(self, source_tile):
        """Handle camera change from a tile and sync to others if enabled."""
        if not self.sync_pan_zoom or self._syncing_camera:
            return

        self._syncing_camera = True
        try:
            rect = source_tile.get_camera_rect()
            for tile in self.tile_widgets:
                if tile is not source_tile:
                    tile.set_camera_rect(rect)
        finally:
            self._syncing_camera = False

    def _toggle_show_info(self):
        """Toggle info label visibility."""
        self.show_info_check.setChecked(not self.show_info_check.isChecked())

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()

        if key == Qt.Key_Left or key == Qt.Key_PageUp:
            self._prev_page()
        elif key == Qt.Key_Right or key == Qt.Key_PageDown:
            self._next_page()
        elif key == Qt.Key_Home:
            self.current_page = 0
            self._load_current_page()
        elif key == Qt.Key_End:
            self.current_page = self._total_pages() - 1
            self._load_current_page()
        elif key == Qt.Key_A:
            # Reset all views (fit to tile)
            self._reset_all_views()
        elif key == Qt.Key_C:
            # Auto contrast all
            self._auto_contrast_all()
        elif key == Qt.Key_I:
            # Toggle info labels
            self._toggle_show_info()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            # Increase tile size
            new_size = min(400, self.tile_size + 25)
            self.size_slider.setValue(new_size)
        elif key == Qt.Key_Minus:
            # Decrease tile size
            new_size = max(100, self.tile_size - 25)
            self.size_slider.setValue(new_size)
        elif key == Qt.Key_Up:
            # Next Z slice
            if self.max_Z > 1 and not self.z_projection:
                new_z = min(self.z_idx + 1, self.max_Z - 1)
                self.z_slider.setValue(new_z)
        elif key == Qt.Key_Down:
            # Previous Z slice
            if self.max_Z > 1 and not self.z_projection:
                new_z = max(self.z_idx - 1, 0)
                self.z_slider.setValue(new_z)
        elif key == Qt.Key_BracketLeft:
            # Previous channel
            if self.max_C > 1 and self.mode == "single":
                new_c = max(self.channel_idx - 1, 0)
                self.channel_slider.setValue(new_c)
        elif key == Qt.Key_BracketRight:
            # Next channel
            if self.max_C > 1 and self.mode == "single":
                new_c = min(self.channel_idx + 1, self.max_C - 1)
                self.channel_slider.setValue(new_c)
        elif key == Qt.Key_H:
            # Open channel panel (Shift+H is handled by menu shortcut)
            self.show_channel_panel()
        elif key == Qt.Key_S:
            # Toggle sync pan/zoom
            self.sync_panzoom_check.setChecked(
                not self.sync_panzoom_check.isChecked()
            )
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Clean up on close."""
        self._clear_tiles()
        super().closeEvent(event)


def open_tiled_viewer(image_paths, tiles_per_page=50):
    """
    Convenience function to open a tiled viewer.

    Args:
        image_paths: List of image file paths
        tiles_per_page: Maximum tiles per page (default 50)

    Returns:
        TiledViewer instance
    """
    # Ensure QApplication exists
    app = QApplication.instance()
    if app is None:
        import sys

        app = QApplication(sys.argv)

    # Apply theme
    from .theme import DARK_THEME

    app.setStyleSheet(DARK_THEME)

    viewer = TiledViewer(image_paths, tiles_per_page=tiles_per_page)
    viewer.show()

    return viewer
