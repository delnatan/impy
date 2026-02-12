import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAction,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from vispy import scene
from vispy.visuals.transforms import STTransform

from .visuals import COLORMAPS, DEFAULT_CHANNEL_COLORMAPS, get_colormap


class VolumeRendererProxy:
    """
    Proxy that provides the renderer interface expected by ContrastDialog/ChannelPanel.
    Manages multiple Volume visuals for multi-channel composite rendering.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self.volumes = []  # One Volume visual per channel
        self.channel_clims = {}
        self.channel_gammas = {}
        self.channel_colormaps = {}
        self.channel_visibility = {}
        self.channel_colors = []  # Display colors for UI

        # Mode state
        self.mode = "composite"
        self.active_channel_idx = 0

        # Cache for current slice (for histogram in ContrastDialog)
        self.current_slice_cache = None

    def _setup_volumes(self, data, view, scale):
        """Create Volume visuals for each channel."""
        T, Z, C, Y, X = data.shape
        sz, sy, sx = scale

        for c in range(C):
            # Determine colormap
            cmap_name = DEFAULT_CHANNEL_COLORMAPS[c % len(DEFAULT_CHANNEL_COLORMAPS)]
            self.channel_colormaps[c] = cmap_name
            cmap, display_color = get_colormap(cmap_name)

            # Store display color
            if display_color:
                self.channel_colors.append(display_color)
            else:
                self.channel_colors.append("white")

            # Get volume data for this channel
            vol_data = self._get_channel_volume(data, c, 0)

            # Create Volume visual
            volume = scene.visuals.Volume(
                vol_data,
                parent=view.scene,
                method="mip",
                cmap=cmap,
                clim="auto",
            )

            # Apply scale transform for non-isotropic voxels
            volume.transform = STTransform(scale=(sx, sy, sz))

            # Set additive blending for composite mode
            volume.set_gl_state(
                preset="additive",
                blend=True,
                blend_func=("src_alpha", "one"),
                depth_test=False,
            )

            # Store initial clim
            vmin = float(np.nanmin(vol_data))
            vmax = float(np.nanmax(vol_data))
            if vmin == vmax:
                vmax = vmin + 1
            self.channel_clims[c] = (vmin, vmax)
            volume.clim = (vmin, vmax)

            # Initialize gamma and visibility
            self.channel_gammas[c] = 1.0
            self.channel_visibility[c] = True

            self.volumes.append(volume)

        # Build current_slice_cache (middle Z slice for histogram)
        self._update_slice_cache(data, 0)

    def _get_channel_volume(self, data, channel, time):
        """Extract single-channel 3D volume from 5D data."""
        vol = data[time, :, channel, :, :]
        return np.ascontiguousarray(vol, dtype=np.float32)

    def _update_slice_cache(self, data, time):
        """Update the slice cache for histogram display (middle Z slice)."""
        T, Z, C, Y, X = data.shape
        mid_z = Z // 2
        # Cache shape is (C, Y, X)
        self.current_slice_cache = np.array(
            [data[time, mid_z, c, :, :] for c in range(C)], dtype=np.float32
        )

    def set_clim(self, channel_idx, vmin, vmax):
        """Set contrast limits for a channel."""
        if channel_idx < len(self.volumes):
            self.channel_clims[channel_idx] = (vmin, vmax)
            self.volumes[channel_idx].clim = (vmin, vmax)

    def get_clim(self, channel_idx):
        """Get contrast limits for a channel."""
        return self.channel_clims.get(channel_idx, (0, 1))

    def set_gamma(self, channel_idx, gamma):
        """Set gamma for a channel."""
        self.channel_gammas[channel_idx] = gamma
        if channel_idx < len(self.volumes):
            self.volumes[channel_idx].gamma = gamma

    def get_gamma(self, channel_idx):
        """Get gamma for a channel."""
        return self.channel_gammas.get(channel_idx, 1.0)

    def set_colormap(self, channel_idx, cmap_name):
        """Set colormap for a channel."""
        if cmap_name not in COLORMAPS:
            return

        if channel_idx < len(self.volumes):
            self.channel_colormaps[channel_idx] = cmap_name
            cmap, display_color = get_colormap(cmap_name)
            self.volumes[channel_idx].cmap = cmap

            # Update display color
            if display_color and channel_idx < len(self.channel_colors):
                self.channel_colors[channel_idx] = display_color

    def get_colormap_name(self, channel_idx):
        """Get colormap name for a channel."""
        return self.channel_colormaps.get(channel_idx, "White")

    def set_channel_visible(self, channel_idx, visible):
        """Set visibility for a channel."""
        if channel_idx < len(self.volumes):
            self.channel_visibility[channel_idx] = visible
            if self.mode == "composite":
                self.volumes[channel_idx].visible = visible
            elif self.mode == "single" and channel_idx == self.active_channel_idx:
                self.volumes[channel_idx].visible = visible

    def get_channel_visible(self, channel_idx):
        """Get visibility for a channel."""
        return self.channel_visibility.get(channel_idx, True)

    def set_mode(self, mode):
        """Set display mode ('composite' or 'single')."""
        self.mode = mode
        self._update_visibility()

    def set_active_channel(self, idx):
        """Set active channel for single-channel mode."""
        self.active_channel_idx = idx
        if self.mode == "single":
            self._update_visibility()

    def _update_visibility(self):
        """Update volume visibility based on mode."""
        for c, volume in enumerate(self.volumes):
            if self.mode == "composite":
                volume.visible = self.channel_visibility.get(c, True)
            else:  # single
                volume.visible = c == self.active_channel_idx

    def set_render_method(self, method):
        """Set rendering method for all volumes."""
        for volume in self.volumes:
            volume.method = method

    def update_volumes(self, data, time):
        """Update all volume data for a new time point."""
        for c, volume in enumerate(self.volumes):
            vol_data = self._get_channel_volume(data, c, time)
            volume.set_data(vol_data)
        self._update_slice_cache(data, time)


class VolumeViewer(QMainWindow):
    """3D volume rendering viewer using vispy's Volume visual.

    Supports multi-channel rendering with Composite and Single Channel modes,
    colormap selection, and contrast adjustment via the standard ContrastDialog.
    """

    RENDER_METHODS = [
        "mip",
        "attenuated_mip",
        "minip",
        "translucent",
        "additive",
        "average",
    ]

    def __init__(
        self,
        data,
        meta=None,
        title="3D Volume",
        channel=0,
        time=0,
        channel_colormaps=None,
    ):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(800, 700)

        self.data = data  # (T, Z, C, Y, X)
        self.meta = meta or {}
        self._channel_colormaps = channel_colormaps
        self.T, self.Z, self.C, self.Y, self.X = self.data.shape

        # Current state
        self.current_time = min(time, self.T - 1)
        self.c_idx = min(channel, self.C - 1)  # For compatibility with ContrastDialog

        # Scale (z, y, x)
        self.scale = self.meta.get("scale", (1.0, 1.0, 1.0))
        sz, sy, sx = self.scale

        # Dialogs
        self.contrast_dialog = None
        self.channel_panel = None

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Vispy Canvas with 3D scene (keys=None to prevent escape from hiding visuals)
        self.canvas = scene.SceneCanvas(keys=None, bgcolor="black")
        self.view = self.canvas.central_widget.add_view()

        # Use TurntableCamera for intuitive 3D interaction
        self.view.camera = scene.TurntableCamera(
            elevation=30, azimuth=45, fov=45,
            distance=max(self.X * sx, self.Y * sy, self.Z * sz) * 2
        )

        self.main_layout.addWidget(self.canvas.native, 1)

        # Create renderer proxy (manages Volume visuals)
        self.renderer = VolumeRendererProxy(self)
        self.renderer._setup_volumes(self.data, self.view, self.scale)
        if self._channel_colormaps:
            for channel_idx, cmap_name in self._channel_colormaps.items():
                self.renderer.set_colormap(channel_idx, cmap_name)

        # Controls
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.addWidget(self.controls_widget)

        self._setup_controls()
        self._setup_menu()

        # Initial camera positioning
        self._reset_camera()

    @property
    def num_channels(self):
        """Alias for C for compatibility."""
        return self.C

    def _reset_camera(self):
        """Reset camera to show full volume."""
        sz, sy, sx = self.scale
        center = (self.X * sx / 2, self.Y * sy / 2, self.Z * sz / 2)
        self.view.camera.center = center
        self.view.camera.distance = max(self.X * sx, self.Y * sy, self.Z * sz) * 1.5

    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # Adjust Menu
        adjust_menu = menubar.addMenu("Adjust")

        bc_action = QAction("Brightness/Contrast", self)
        bc_action.setShortcut("Shift+C")
        bc_action.triggered.connect(self.show_contrast_dialog)
        adjust_menu.addAction(bc_action)

        channels_action = QAction("Channels...", self)
        channels_action.setShortcut("Shift+H")
        channels_action.triggered.connect(self.show_channel_panel)
        adjust_menu.addAction(channels_action)

        # View Menu
        view_menu = menubar.addMenu("View")

        reset_action = QAction("Reset View", self)
        reset_action.setShortcut("A")
        reset_action.triggered.connect(self._reset_camera)
        view_menu.addAction(reset_action)

    def _setup_controls(self):
        # Row 1: Mode selector (Composite/Single)
        if self.C > 1:
            row_mode = QHBoxLayout()
            row_mode.addWidget(QLabel("Mode:"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Composite", "Single Channel"])
            self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
            row_mode.addWidget(self.mode_combo)
            row_mode.addStretch()
            self.controls_layout.addLayout(row_mode)

            # Channel slider (for single channel mode, initially hidden)
            self.channel_row_widget = QWidget()
            c_layout = QHBoxLayout(self.channel_row_widget)
            c_layout.setContentsMargins(0, 0, 0, 0)
            c_layout.addWidget(QLabel("Channel:"))
            self.channel_slider = QSlider(Qt.Horizontal)
            self.channel_slider.setRange(0, self.C - 1)
            self.channel_slider.setValue(self.c_idx)
            self.channel_slider.valueChanged.connect(self._on_channel_change)
            c_layout.addWidget(self.channel_slider)
            self.channel_label = QLabel(str(self.c_idx))
            self.channel_label.setFixedWidth(30)
            c_layout.addWidget(self.channel_label)
            self.controls_layout.addWidget(self.channel_row_widget)
            self.channel_row_widget.setVisible(False)

        # Row 2: Rendering method
        row_method = QHBoxLayout()
        row_method.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(self.RENDER_METHODS)
        self.method_combo.currentTextChanged.connect(self._on_method_change)
        row_method.addWidget(self.method_combo)
        row_method.addStretch()
        self.controls_layout.addLayout(row_method)

        # Row 3: Time slider (if time series)
        if self.T > 1:
            row_time = QHBoxLayout()
            row_time.addWidget(QLabel("Time:"))
            self.time_slider = QSlider(Qt.Horizontal)
            self.time_slider.setRange(0, self.T - 1)
            self.time_slider.setValue(self.current_time)
            self.time_slider.valueChanged.connect(self._on_time_change)
            row_time.addWidget(self.time_slider)
            self.time_label = QLabel(str(self.current_time))
            self.time_label.setFixedWidth(30)
            row_time.addWidget(self.time_label)
            self.controls_layout.addLayout(row_time)

    def _on_mode_change(self, index):
        """Handle mode change between Composite and Single Channel."""
        mode = "composite" if index == 0 else "single"
        self.renderer.set_mode(mode)

        # Toggle channel slider visibility
        if self.C > 1:
            self.channel_row_widget.setVisible(mode == "single")

        self.canvas.update()

    def _on_channel_change(self, value):
        """Handle channel change in single-channel mode."""
        self.c_idx = value
        if hasattr(self, "channel_label"):
            self.channel_label.setText(str(value))
        self.renderer.set_active_channel(value)
        self.canvas.update()

        # Sync ContrastDialog if open
        if self.contrast_dialog and self.contrast_dialog.isVisible():
            self.contrast_dialog.combo.setCurrentIndex(value)
            self.contrast_dialog.refresh_ui()

    def _on_method_change(self, method):
        """Handle rendering method change."""
        self.renderer.set_render_method(method)
        self.canvas.update()

    def _on_time_change(self, value):
        """Handle time change."""
        self.current_time = value
        if hasattr(self, "time_label"):
            self.time_label.setText(str(value))
        self.renderer.update_volumes(self.data, value)
        self.canvas.update()

        # Refresh dialogs if open
        if self.contrast_dialog and self.contrast_dialog.isVisible():
            self.contrast_dialog.refresh_ui()
        if self.channel_panel and self.channel_panel.isVisible():
            self.channel_panel.refresh_ui()

    def show_contrast_dialog(self):
        """Show the Brightness/Contrast dialog."""
        from .widgets import ContrastDialog

        if self.contrast_dialog is None:
            self.contrast_dialog = ContrastDialog(self, parent=self)
        self.contrast_dialog.show()
        self.contrast_dialog.raise_()
        self.contrast_dialog.refresh_ui()

    def show_channel_panel(self):
        """Show the Channels panel."""
        from .widgets import ChannelPanel

        if self.channel_panel is None:
            self.channel_panel = ChannelPanel(self, parent=self)
        self.channel_panel.show()
        self.channel_panel.raise_()
        self.channel_panel.refresh_ui()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_A:
            self._reset_camera()
            self.canvas.update()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Clean up resources on close."""
        # Release data proxy reference
        if hasattr(self.data, "release"):
            try:
                self.data.release()
            except Exception:
                pass
        elif hasattr(self.data, "close"):
            try:
                self.data.close()
            except Exception:
                pass

        self.data = None
        super().closeEvent(event)
