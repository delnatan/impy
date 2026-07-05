import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from vispy import scene
from vispy.visuals.transforms import STTransform

from .. import colormaps as _colormaps
from ..data.channel_state import ChannelDisplayList
from ..visuals.image import (
    DEFAULT_CHANNEL_COLORMAPS,
    default_channel_colormap,
    get_colormap,
)

_NEUTRAL_SWATCH = "#888888"


class VolumeRendererProxy:
    """
    Proxy that provides the renderer interface expected by ChannelPanel.
    Manages multiple Volume visuals for multi-channel composite rendering.

    Per-channel display state lives in :attr:`display`
    (:class:`ChannelDisplayList`); see ``visuals/image.py`` for the same
    pattern used by ``CompositeImageVisual``.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self.volumes = []  # One Volume visual per channel
        self.display = None  # ChannelDisplayList — set in _setup_volumes

        # Mode state
        self.mode = "composite"
        self.active_channel_idx = 0

        # Cache for current slice (for histogram in ChannelPanel)
        self.current_slice_cache = None

    def _setup_volumes(self, data, view, scale, channels_meta=None):
        """Create Volume visuals for each channel."""
        T, Z, C, Y, X = data.shape
        sz, sy, sx = scale
        channels_meta = channels_meta or []

        self.display = ChannelDisplayList(C)

        for c in range(C):
            emission_wavelength = (
                channels_meta[c].get("emission_wavelength")
                if c < len(channels_meta)
                else None
            )
            cmap_name = default_channel_colormap(
                c, C, emission_wavelength=emission_wavelength
            )
            self.display.set_colormap_name(c, cmap_name)
            cmap, _ = get_colormap(cmap_name)

            vol_data = self._get_channel_volume(data, c, 0)

            volume = scene.visuals.Volume(
                vol_data,
                parent=view.scene,
                method="mip",
                cmap=cmap,
                clim="auto",
                texture_format="auto",
            )
            volume.transform = STTransform(scale=(sx, sy, sz))
            volume.set_gl_state(
                preset="additive",
                blend=True,
                blend_func=("src_alpha", "one"),
                depth_test=False,
            )

            vmin = float(np.nanmin(vol_data))
            vmax = float(np.nanmax(vol_data))
            if vmin == vmax:
                vmax = vmin + 1
            self.display.set_clim(c, vmin, vmax)
            volume.clim = (vmin, vmax)

            self.volumes.append(volume)

        # Mirror display-state changes into the underlying volumes.
        self.display.subscribe(self._on_display_changed)

        # Build current_slice_cache (middle Z slice for histogram)
        self._update_slice_cache(data, 0)

    def _on_display_changed(self, idx, field):
        if idx >= len(self.volumes):
            return
        state = self.display[idx]
        volume = self.volumes[idx]
        if field == "clim":
            volume.clim = state.clim
        elif field == "gamma":
            volume.gamma = state.gamma
        elif field == "colormap_name":
            cmap, _ = get_colormap(state.colormap_name)
            volume.cmap = cmap
        elif field == "visible":
            self._update_visibility()

    @property
    def channel_colors(self):
        if self.display is None:
            return []
        return [
            self.display[c].display_color() or _NEUTRAL_SWATCH
            for c in range(len(self.display))
        ]

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

    # Channel state — thin delegations to self.display.

    def set_clim(self, channel_idx, vmin, vmax):
        if self.display is not None:
            self.display.set_clim(channel_idx, vmin, vmax)

    def get_clim(self, channel_idx):
        if self.display is not None and channel_idx < len(self.display):
            return self.display[channel_idx].clim
        return (0.0, 1.0)

    def set_gamma(self, channel_idx, gamma):
        if self.display is not None:
            self.display.set_gamma(channel_idx, gamma)

    def get_gamma(self, channel_idx):
        if self.display is not None and channel_idx < len(self.display):
            return self.display[channel_idx].gamma
        return 1.0

    def set_colormap(self, channel_idx, cmap_name):
        if cmap_name not in _colormaps.names():
            return
        if self.display is not None:
            self.display.set_colormap_name(channel_idx, cmap_name)

    def get_colormap_name(self, channel_idx):
        if self.display is not None and channel_idx < len(self.display):
            return self.display[channel_idx].colormap_name
        return "White"

    def set_channel_visible(self, channel_idx, visible):
        if self.display is not None:
            self.display.set_visible(channel_idx, visible)

    def get_channel_visible(self, channel_idx):
        if self.display is not None and channel_idx < len(self.display):
            return self.display[channel_idx].visible
        return True

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
                volume.visible = (
                    self.display[c].visible if self.display is not None else True
                )
            else:  # single
                volume.visible = c == self.active_channel_idx

    def set_render_method(self, method):
        """Set rendering method for all volumes."""
        for volume in self.volumes:
            volume.method = method

    def set_attenuation(self, value):
        """Set attenuation for attenuated_mip method."""
        for volume in self.volumes:
            volume.attenuation = value

    def set_iso_threshold(self, value):
        """Set threshold for iso method."""
        for volume in self.volumes:
            volume.threshold = value

    def update_volumes(self, data, time):
        """Update all volume data for a new time point."""
        for c, volume in enumerate(self.volumes):
            vol_data = self._get_channel_volume(data, c, time)
            volume.set_data(vol_data)
        self._update_slice_cache(data, time)


class VolumeViewer(QMainWindow):
    """3D volume rendering viewer using vispy's Volume visual.

    Supports multi-channel rendering with Composite and Single Channel modes,
    colormap selection, and contrast adjustment via the standard ChannelPanel.
    """

    # Emitted whenever the displayed volume changes (time step, set_data,
    # …). ChannelPanel listens so per-channel histograms re-pull from
    # renderer.current_slice_cache.
    view_changed = Signal(object)

    # Mirrored onto the Workspace's persistent menu bar while docked
    # (see ui/workspace.py); format documented at window.MENU_SPEC.
    MENU_SPEC = [
        ("Adjust", [
            {"label": "Channels && Contrast...", "shortcut": "Shift+C", "method": "show_channel_panel"},
        ]),
        ("View", [
            {"label": "Reset View", "shortcut": "A", "method": "_reset_camera"},
        ]),
    ]

    RENDER_METHODS = [
        "mip",
        "attenuated_mip",
        "minip",
        "translucent",
        "additive",
        "average",
        "iso",
    ]

    # Methods that expose a tunable parameter in the UI
    METHOD_PARAMS = {
        "attenuated_mip": ("Attenuation", 0.1, 10.0, 1.0, 0.1),
        "iso": ("Threshold", 0.0, 1.0, 0.5, 0.01),
    }

    def __init__(
        self,
        data,
        meta=None,
        title="3D Volume",
        channel=0,
        time=0,
        channel_display=None,
    ):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(800, 700)

        self.data = data  # (T, Z, C, Y, X)
        self.meta = meta or {}
        # Per-channel display state (clim/gamma/colormap/visible) copied
        # from the parent window, applied once volumes exist.
        self._channel_display = channel_display
        self.T, self.Z, self.C, self.Y, self.X = self.data.shape

        # Current state
        self.current_time = min(time, self.T - 1)
        self.c_idx = min(channel, self.C - 1)  # For compatibility with ChannelPanel

        # Scale (z, y, x)
        self.scale = self.meta.get("scale", (1.0, 1.0, 1.0))
        sz, sy, sx = self.scale

        # Dialogs
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
        self.renderer._setup_volumes(
            self.data, self.view, self.scale, channels_meta=self.meta.get("channels")
        )
        if self._channel_display is not None:
            for c in range(min(len(self._channel_display), self.C)):
                state = self._channel_display[c]
                self.renderer.set_clim(c, *state.clim)
                self.renderer.set_gamma(c, state.gamma)
                self.renderer.set_colormap(c, state.colormap_name)
                self.renderer.set_channel_visible(c, state.visible)

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

    @property
    def img_data(self):
        return self.data

    def _reset_camera(self):
        """Reset camera to show full volume."""
        sz, sy, sx = self.scale
        center = (self.X * sx / 2, self.Y * sy / 2, self.Z * sz / 2)
        self.view.camera.center = center
        self.view.camera.distance = max(self.X * sx, self.Y * sy, self.Z * sz) * 1.5

    def _setup_menu(self):
        """Setup menu bar."""
        # Deferred import: ui.window imports the viewers package, so a
        # module-level import back into ui.window would cycle.
        from ..ui.window import build_menus

        self._menu_actions = build_menus(
            self.menuBar(), self.MENU_SPEC, self
        )

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

        # Row 2b: Method-specific parameter (shown only for methods that need it)
        self.param_row_widget = QWidget()
        param_layout = QHBoxLayout(self.param_row_widget)
        param_layout.setContentsMargins(0, 0, 0, 0)
        self.param_label = QLabel("Attenuation:")
        param_layout.addWidget(self.param_label)
        self.param_spinbox = QDoubleSpinBox()
        self.param_spinbox.setDecimals(2)
        self.param_spinbox.setSingleStep(0.1)
        self.param_spinbox.setFixedWidth(80)
        self.param_spinbox.valueChanged.connect(self._on_method_param_change)
        param_layout.addWidget(self.param_spinbox)
        param_layout.addStretch()
        self.controls_layout.addWidget(self.param_row_widget)
        self.param_row_widget.setVisible(False)

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

    def _on_method_change(self, method):
        """Handle rendering method change."""
        self.renderer.set_render_method(method)

        # Show/hide the parameter row and configure it for this method
        if method in self.METHOD_PARAMS:
            label, lo, hi, default, step = self.METHOD_PARAMS[method]
            self.param_label.setText(f"{label}:")
            self.param_spinbox.blockSignals(True)
            self.param_spinbox.setRange(lo, hi)
            self.param_spinbox.setSingleStep(step)
            self.param_spinbox.setValue(default)
            self.param_spinbox.blockSignals(False)
            self.param_row_widget.setVisible(True)
            # Apply default immediately
            self._on_method_param_change(default)
        else:
            self.param_row_widget.setVisible(False)

        self.canvas.update()

    def _on_method_param_change(self, value):
        """Handle method-specific parameter change."""
        method = self.method_combo.currentText()
        if method == "attenuated_mip":
            self.renderer.set_attenuation(value)
        elif method == "iso":
            self.renderer.set_iso_threshold(value)
        self.canvas.update()

    def _on_time_change(self, value):
        """Handle time change."""
        self.current_time = value
        if hasattr(self, "time_label"):
            self.time_label.setText(str(value))
        self.renderer.update_volumes(self.data, value)
        self.canvas.update()
        self.view_changed.emit(self)

    def show_channel_panel(self):
        """Show the Channels panel."""
        from ..widgets import show_channel_dock

        show_channel_dock(self)

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
