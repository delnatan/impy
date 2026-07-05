import numpy as np
from vispy import scene
from vispy.visuals.transforms.linear import MatrixTransform, STTransform

from .. import colormaps as _colormaps
from ..data.channel_state import ChannelDisplayList, ChannelDisplayState

# Default channel colormaps (original microscope colors)
DEFAULT_CHANNEL_COLORMAPS = [
    "Orange",
    "Green",
    "Cyan",
    "Magenta",
    "Yellow",
    "White",
]

# Standard RGB colormaps for RGB images
RGB_COLORMAPS = ["Red", "Pure Green", "Blue"]

# Fallback swatch color for multi-color colormaps (viridis, etc.)
_NEUTRAL_SWATCH = "#888888"

# Dtypes with a GL internal format under texture_format="auto"
# (float64 is downcast to float32 by vispy itself).
_GL_TEXTURE_DTYPES = (
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.float32),
    np.dtype(np.float64),
)


def get_colormap(name):
    """Get a vispy Colormap by name. Returns (Colormap, display_color_or_None)."""
    return _colormaps.get(name)


def default_clim_for_dtype(dtype):
    """Return a reasonable default ``(vmin, vmax)`` for *dtype*."""
    if dtype == np.uint16:
        return (0.0, 65535.0)
    if dtype.kind == "f":
        return (0.0, 1.0)
    return (0.0, 255.0)


def default_channel_colormap(
    channel_idx, n_channels, is_rgb=False, emission_wavelength=None
):
    """Pick a default colormap name for *channel_idx*.

    If *emission_wavelength* (nm) is available, it takes priority over the
    fixed-palette cycle — e.g. a 525nm channel defaults to Green instead of
    whatever the cycle position happens to land on.
    """
    if n_channels == 1:
        return "White"
    if is_rgb and channel_idx < 3:
        return RGB_COLORMAPS[channel_idx]
    wl_colormap = _colormaps.colormap_for_wavelength(emission_wavelength)
    if wl_colormap is not None:
        return wl_colormap
    return DEFAULT_CHANNEL_COLORMAPS[
        channel_idx % len(DEFAULT_CHANNEL_COLORMAPS)
    ]


class CompositeImageVisual:
    """
    Manages multiple Vispy Image visuals to create a composite
    multi-channel rendering using additive blending.

    Per-channel display state (clim, gamma, colormap name, visibility)
    lives in :attr:`display`, a :class:`ChannelDisplayList`. UI widgets
    should call ``display.subscribe(callback)`` to react to changes
    rather than polling. The renderer keeps vispy state in sync by
    subscribing itself.
    """

    def __init__(
        self, view, image_data, scale=(1.0, 1.0), is_rgb=False, channels_meta=None
    ):
        self.data = image_data
        self.view = view
        self.scale = scale  # (sy, sx)
        self.layers = []
        self.is_rgb = is_rgb  # True for RGB color images
        self.num_channels = self.data.shape[2]
        # Mode state
        self.mode = "composite"
        self.active_channel_idx = 0

        self.current_slice_cache = None

        # Transform state (rotation/translation for image alignment)
        self._rotation_deg = 0.0
        self._translate_x = 0.0
        self._translate_y = 0.0

        # Per-channel display state (clim/gamma/colormap/visible).
        default_clim = default_clim_for_dtype(self.data.dtype)
        channels_meta = channels_meta or []
        self.display = ChannelDisplayList(self.num_channels)
        for c in range(self.num_channels):
            self.display.set_clim(c, *default_clim)
            emission_wavelength = (
                channels_meta[c].get("emission_wavelength")
                if c < len(channels_meta)
                else None
            )
            self.display.set_colormap_name(
                c,
                default_channel_colormap(
                    c,
                    self.num_channels,
                    self.is_rgb,
                    emission_wavelength=emission_wavelength,
                ),
            )

        self._setup_layers()

        # Mirror display-state changes into the underlying vispy layers.
        self.display.subscribe(self._on_display_changed)

    def _setup_layers(self):
        for c in range(self.num_channels):
            state = self.display[c]
            cmap, _ = get_colormap(state.colormap_name)

            image_visual = scene.visuals.Image(
                cmap=cmap,
                parent=self.view.scene,
                method="auto",
                interpolation="nearest",
                texture_format="auto",
            )
            image_visual.clim = state.clim
            image_visual.gamma = state.gamma
            image_visual.transform = self._build_transform()
            image_visual.set_gl_state(
                preset="translucent",
                blend=True,
                blend_func=("one", "one"),
                depth_test=False,
            )
            image_visual.order = -c
            self.layers.append(image_visual)

    def _on_display_changed(self, idx, field):
        """Apply a ChannelDisplayList mutation to the underlying vispy layer."""
        if idx >= len(self.layers):
            return
        state = self.display[idx]
        layer = self.layers[idx]
        if field == "clim":
            layer.clim = state.clim
        elif field == "gamma":
            layer.gamma = state.gamma
        elif field == "colormap_name":
            cmap, _ = get_colormap(state.colormap_name)
            layer.cmap = cmap
        elif field == "visible":
            self._update_visibility()

    @property
    def channel_colors(self):
        """Derived list of per-channel swatch colors.

        Computed from each channel's colormap. Multi-color colormaps
        (viridis, plasma, etc.) get a neutral grey swatch.
        """
        return [
            state.display_color() or _NEUTRAL_SWATCH
            for state in (self.display[c] for c in range(len(self.display)))
        ]

    def set_mode(self, mode):
        self.mode = mode
        self._update_visibility()

    def set_active_channel(self, idx):
        self.active_channel_idx = idx
        self._update_visibility()

    def _update_visibility(self):
        # GL blend state is fixed at _setup_layers; only visibility varies.
        for i, layer in enumerate(self.layers):
            if self.mode == "composite":
                layer.visible = self.display[i].visible
            else:
                # In single channel mode, only show active channel
                layer.visible = i == self.active_channel_idx

    def update_slice(self, t_idx, z_idx):
        """Slice ``self.data`` at (t, z) and display it.

        Slicing happens on the caller's thread; callers with an async
        loader should slice off-thread and hand the plane to
        :meth:`set_slice` instead.
        """
        try:
            volume_slice = self.data[t_idx, z_idx, :, :, :]
        except Exception as e:
            print(f"Error slicing data: {e}")
            return

        # Handle Z-Stack Projection
        # If z_idx is a slice, volume_slice will be (Z, C, Y, X) or (Z, Y, X)
        # We need to project it to (C, Y, X) or (Y, X)
        if volume_slice.ndim == 4:  # (Z, C, Y, X)
            volume_slice = np.max(volume_slice, axis=0)
        elif volume_slice.ndim == 3 and isinstance(
            z_idx, slice
        ):  # (Z, Y, X) -> (Y, X)
            volume_slice = np.max(volume_slice, axis=0)

        self.set_slice(volume_slice)

    def set_slice(self, volume_slice):
        """Display an already-loaded (C, Y, X) plane."""
        if volume_slice.ndim == 2:
            volume_slice = volume_slice[np.newaxis, :, :]

        # GPU-scaled textures (texture_format="auto") only have GL internal
        # formats for these dtypes; anything else (signed/wide ints) casts.
        if volume_slice.dtype not in _GL_TEXTURE_DTYPES:
            volume_slice = volume_slice.astype(np.float32)

        self.current_slice_cache = volume_slice

        for c, layer in enumerate(self.layers):
            if c < volume_slice.shape[0]:
                layer.set_data(volume_slice[c])

        self._update_visibility()

    def auto_contrast(self, pct_low=0.1, pct_high=99.9):
        """One-shot percentile-based auto-contrast for every channel.

        Operates on the currently cached slice. Intended to be called
        once after the first slice loads (or on user request), not on
        every slice update.
        """
        from ..contrast import compute_percentile_clim

        cache = self.current_slice_cache
        if cache is None:
            return
        for c in range(cache.shape[0]):
            if c < len(self.layers):
                mn, mx = compute_percentile_clim(cache[c], pct_low, pct_high)
                self.set_clim(c, mn, mx)

    # Channel state — thin delegations to self.display. The display
    # list's subscribe() machinery propagates changes to the vispy
    # layers via _on_display_changed.

    def set_clim(self, channel_idx, vmin, vmax):
        self.display.set_clim(channel_idx, vmin, vmax)

    def get_clim(self, channel_idx):
        if channel_idx < len(self.display):
            return self.display[channel_idx].clim
        return (0.0, 255.0)

    def set_gamma(self, channel_idx, gamma):
        self.display.set_gamma(channel_idx, gamma)

    def get_gamma(self, channel_idx):
        if channel_idx < len(self.display):
            return self.display[channel_idx].gamma
        return 1.0

    def set_colormap(self, channel_idx, cmap_name):
        self.display.set_colormap_name(channel_idx, cmap_name)

    def get_colormap_name(self, channel_idx):
        if channel_idx < len(self.display):
            return self.display[channel_idx].colormap_name
        return "White"

    def set_channel_visible(self, channel_idx, visible):
        self.display.set_visible(channel_idx, visible)

    def get_channel_visible(self, channel_idx):
        if channel_idx < len(self.display):
            return self.display[channel_idx].visible
        return True

    def reset_camera(self, shape):
        _, _, _, Y, X = shape
        sy, sx = self.scale
        self.view.camera.rect = (0, 0, X * sx, Y * sy)
        self.view.camera.flip = (False, True, False)

    def _build_transform(self):
        """Build the combined transform: scale * rotation_around_center * translation."""
        sy, sx = self.scale
        _, _, _, Y, X = self.data.shape

        # Image center in scaled coordinates
        cx = X * sx / 2
        cy = Y * sy / 2

        # If no rotation/translation, just use simple scale
        if (
            self._rotation_deg == 0.0
            and self._translate_x == 0.0
            and self._translate_y == 0.0
        ):
            return STTransform(scale=(sx, sy))

        # Build transform for rotation around image center:
        # 1. Scale the image
        # 2. Translate so scaled center is at origin
        # 3. Rotate
        # 4. Translate back + user offset
        #
        # Matrix form: T_back @ R @ T_to_origin @ S
        #
        # VisPy's methods do: self.matrix = self.matrix @ new_transform
        # So we build left-to-right to get the correct composition
        transform = MatrixTransform()
        transform.scale((sx, sy, 1))
        transform.translate((-cx, -cy, 0))
        transform.rotate(self._rotation_deg, (0, 0, 1))
        transform.translate(
            (cx + self._translate_x, cy + self._translate_y, 0)
        )

        return transform

    def _apply_transform_to_layers(self):
        """Apply the current transform to all image layers."""
        transform = self._build_transform()
        for layer in self.layers:
            layer.transform = transform

    @property
    def rotation_deg(self):
        return self._rotation_deg

    @rotation_deg.setter
    def rotation_deg(self, value):
        self._rotation_deg = float(value)
        self._apply_transform_to_layers()

    @property
    def translate_x(self):
        return self._translate_x

    @translate_x.setter
    def translate_x(self, value):
        self._translate_x = float(value)
        self._apply_transform_to_layers()

    @property
    def translate_y(self):
        return self._translate_y

    @translate_y.setter
    def translate_y(self, value):
        self._translate_y = float(value)
        self._apply_transform_to_layers()

    def set_transform(
        self, rotation_deg=None, translate_x=None, translate_y=None
    ):
        """Set multiple transform parameters at once (avoids multiple rebuilds)."""
        if rotation_deg is not None:
            self._rotation_deg = float(rotation_deg)
        if translate_x is not None:
            self._translate_x = float(translate_x)
        if translate_y is not None:
            self._translate_y = float(translate_y)
        self._apply_transform_to_layers()

    def reset_transform(self):
        """Reset rotation and translation to identity."""
        self._rotation_deg = 0.0
        self._translate_x = 0.0
        self._translate_y = 0.0
        self._apply_transform_to_layers()


class MultiViewChannelProxy:
    """
    Proxies calls to multiple CompositeImageVisual instances to keep them in
    sync. Used by ChannelPanel to control several views/tiles simultaneously
    (e.g. OrthoViewer's 3 panels, or ZMontageViewer's per-Z-slice tiles).
    """

    def __init__(self, visuals):
        self.visuals = visuals
        # We treat the first visual as the "primary" for reading state.
        self.primary = visuals[0]

    @property
    def layers(self):
        # Return layers of primary so dialog can iterate them
        return self.primary.layers

    @property
    def current_slice_cache(self):
        return self.primary.current_slice_cache

    @property
    def channel_colors(self):
        return self.primary.channel_colors

    @property
    def display(self):
        """Shared :class:`ChannelDisplayList` (the primary view's).

        Panels should subscribe here. The other views are kept in sync by
        this proxy's broadcast setters.
        """
        return self.primary.display

    def set_clim(self, channel_idx, vmin, vmax):
        for v in self.visuals:
            v.set_clim(channel_idx, vmin, vmax)

    def get_clim(self, channel_idx):
        return self.primary.get_clim(channel_idx)

    def set_gamma(self, channel_idx, gamma):
        for v in self.visuals:
            v.set_gamma(channel_idx, gamma)

    def get_gamma(self, channel_idx):
        return self.primary.get_gamma(channel_idx)

    def set_mode(self, mode):
        for v in self.visuals:
            v.set_mode(mode)

    def set_active_channel(self, idx):
        for v in self.visuals:
            v.set_active_channel(idx)

    def set_colormap(self, channel_idx, cmap_name):
        for v in self.visuals:
            v.set_colormap(channel_idx, cmap_name)

    def get_colormap_name(self, channel_idx):
        return self.primary.get_colormap_name(channel_idx)

    def set_channel_visible(self, channel_idx, visible):
        for v in self.visuals:
            v.set_channel_visible(channel_idx, visible)

    def get_channel_visible(self, channel_idx):
        return self.primary.get_channel_visible(channel_idx)
