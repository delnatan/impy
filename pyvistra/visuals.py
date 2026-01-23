import matplotlib.cm as mpl_cm
import numpy as np
from vispy import scene
from vispy.color import Colormap
from vispy.visuals.transforms.chain import ChainTransform
from vispy.visuals.transforms.linear import MatrixTransform, STTransform


def mpl_to_vispy_colormap(name, n_colors=256):
    """Convert a matplotlib colormap to a vispy Colormap."""
    mpl_cmap = mpl_cm.get_cmap(name)
    colors = mpl_cmap(np.linspace(0, 1, n_colors))
    return Colormap(colors)


# Available colormaps organized by category
COLORMAPS = {
    # Single-color colormaps (black to color) for additive blending
    "Orange": ["black", "#ffb100"],
    "Green": ["black", "#49FF49"],
    "Cyan": ["black", "#5BD6FF"],
    "Magenta": ["black", "magenta"],
    "Yellow": ["black", "yellow"],
    "White": ["black", "white"],
    # Standard RGB for RGB images
    "Red": ["black", "red"],
    "Pure Green": ["black", "#00FF00"],
    "Blue": ["black", "blue"],
    # Matplotlib colormaps (perceptually uniform)
    "viridis": "mpl:viridis",
    "plasma": "mpl:plasma",
    "magma": "mpl:magma",
    "inferno": "mpl:inferno",
    "cividis": "mpl:cividis",
    # Other useful matplotlib colormaps
    "hot": "mpl:hot",
    "cool": "mpl:cool",
    "coolwarm": "mpl:coolwarm",
    "turbo": "mpl:turbo",
    "gray": "mpl:gray",
    "gray_r": "mpl:gray_r",
}

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


def get_colormap(name):
    """Get a vispy Colormap by name from COLORMAPS dictionary."""
    if name not in COLORMAPS:
        # Fallback to white if unknown
        return Colormap(["black", "white"]), "white"

    spec = COLORMAPS[name]
    if isinstance(spec, str) and spec.startswith("mpl:"):
        # Matplotlib colormap
        mpl_name = spec[4:]  # Remove "mpl:" prefix
        return mpl_to_vispy_colormap(mpl_name), None
    else:
        # Simple two-color colormap
        return Colormap(spec), spec[1]  # Return colormap and end color


class CompositeImageVisual:
    """
    Manages multiple Vispy Image visuals to create a composite
    multi-channel rendering using additive blending.
    """

    def __init__(self, view, image_data, scale=(1.0, 1.0), is_rgb=False):
        self.data = image_data
        self.view = view
        self.scale = scale  # (sy, sx)
        self.layers = []
        self.is_rgb = is_rgb  # True for RGB color images
        self.num_channels = self.data.shape[2]
        # State
        self.mode = "composite"
        self.active_channel_idx = 0

        self.current_slice_cache = None
        self.channel_clims = {}
        self.channel_gammas = {}
        self.channel_colormaps = {}  # Maps channel index to colormap name
        self.channel_visibility = {}  # Maps channel index to visibility (True/False)

        # Transform state (rotation/translation for image alignment)
        self._rotation_deg = 0.0
        self._translate_x = 0.0
        self._translate_y = 0.0

        # Legacy color list for histogram display (derived from colormap)
        self.channel_colors = [
            "#ffb100",
            "#49FF49",
            "#5BD6FF",
            "magenta",
            "cyan",
            "yellow",
        ]

        self._setup_layers()

    def _setup_layers(self):
        n_channels = self.data.shape[2]

        # 1. Determine decent default limits based on dtype
        # This prevents the "squashed handles" look on the slider
        dtype = self.data.dtype
        default_clim = (0, 255)

        if dtype == np.uint16:
            default_clim = (0, 65535)
        elif dtype.kind == "f":
            # For float, we assume 0.0-1.0 until we see data
            default_clim = (0.0, 1.0)

        for c in range(n_channels):
            if n_channels == 1:
                cmap_name = "White"
            elif self.is_rgb and c < 3:
                # Use RGB colormaps for RGB images
                cmap_name = RGB_COLORMAPS[c]
            else:
                cmap_name = DEFAULT_CHANNEL_COLORMAPS[
                    c % len(DEFAULT_CHANNEL_COLORMAPS)
                ]

            # Get colormap and associated display color
            cmap, display_color = get_colormap(cmap_name)
            self.channel_colormaps[c] = cmap_name

            # Update legacy color list for histogram display
            if display_color:
                if c < len(self.channel_colors):
                    self.channel_colors[c] = display_color

            image_visual = scene.visuals.Image(
                cmap=cmap,
                parent=self.view.scene,
                method="auto",
                interpolation="nearest",
            )

            # Apply combined transform (scale + rotation + translation)
            image_visual.transform = self._build_transform()

            # Force Additive Blending
            image_visual.set_gl_state(
                preset="translucent",
                blend=True,
                blend_func=("one", "one"),
                depth_test=False,
            )

            image_visual.order = -c
            self.layers.append(image_visual)

            self.channel_clims[c] = default_clim
            self.channel_gammas[c] = 1.0
            self.channel_visibility[c] = True

    def set_mode(self, mode):
        self.mode = mode
        self._update_visibility()

    def set_active_channel(self, idx):
        self.active_channel_idx = idx
        self._update_visibility()

    def _update_visibility(self):
        for i, layer in enumerate(self.layers):
            if self.mode == "composite":
                # In composite mode, respect per-channel visibility toggle
                layer.visible = self.channel_visibility.get(i, True)
                layer.set_gl_state(
                    blend=True,
                    blend_func=("one", "one"),
                    depth_test=False,
                )
            else:
                # In single channel mode, only show active channel
                layer.visible = i == self.active_channel_idx

    def update_slice(self, t_idx, z_idx):
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
            # Ambiguous if C=Z? But if z_idx is slice, dim 0 is Z.
            volume_slice = np.max(volume_slice, axis=0)

        if volume_slice.ndim == 2:
            volume_slice = volume_slice[np.newaxis, :, :]

        self.current_slice_cache = volume_slice

        for c, layer in enumerate(self.layers):
            if c < volume_slice.shape[0]:
                plane = volume_slice[c]
                layer.set_data(plane)

                # Auto-Contrast Logic (Refinement)
                # If we are strictly at default (e.g. 0-65535) and the actual data
                # is tiny (e.g. max 400), we tighten it.
                current_clim = self.channel_clims[c]
                dtype = self.data.dtype

                # Check for "Suspiciously Default" limits vs "Real Data"
                if dtype == np.uint16 and current_clim == (0, 65535):
                    # Use percentiles for robustness against outliers
                    # Ignore zeros (background/padding) for more accurate contrast
                    valid_data = plane[plane > 0]
                    if valid_data.size > 0:
                        mn, mx = np.nanpercentile(valid_data, (0.1, 99.9))
                        # If data uses less than 10% of dynamic range, auto-scale
                        if mx < 6000:
                            self.set_clim(c, mn, max(mx, 1))

                elif dtype.kind == "f" and current_clim == (0.0, 1.0):
                    valid_data = plane[plane > 0]
                    if valid_data.size > 0:
                        mn, mx = np.nanpercentile(valid_data, (0.1, 99.9))
                        self.set_clim(c, mn, mx)

        self._update_visibility()

    def set_clim(self, channel_idx, vmin, vmax):
        if channel_idx < len(self.layers):
            self.layers[channel_idx].clim = (vmin, vmax)
            self.channel_clims[channel_idx] = (vmin, vmax)

    def get_clim(self, channel_idx):
        return self.channel_clims.get(channel_idx, (0, 255))

    def set_gamma(self, channel_idx, gamma):
        if channel_idx < len(self.layers):
            self.layers[channel_idx].gamma = gamma
            self.channel_gammas[channel_idx] = gamma

    def get_gamma(self, channel_idx):
        return self.channel_gammas.get(channel_idx, 1.0)

    def set_colormap(self, channel_idx, cmap_name):
        """Set the colormap for a specific channel by name."""
        if channel_idx >= len(self.layers):
            return

        if cmap_name not in COLORMAPS:
            print(f"Unknown colormap: {cmap_name}")
            return

        cmap, display_color = get_colormap(cmap_name)
        self.layers[channel_idx].cmap = cmap
        self.channel_colormaps[channel_idx] = cmap_name

        # Update legacy color for histogram display
        if display_color:
            if channel_idx < len(self.channel_colors):
                self.channel_colors[channel_idx] = display_color
        else:
            # For matplotlib colormaps, use a representative color
            # Sample the colormap at 75% to get a bright representative color
            if cmap_name in COLORMAPS:
                spec = COLORMAPS[cmap_name]
                if isinstance(spec, str) and spec.startswith("mpl:"):
                    mpl_name = spec[4:]
                    mpl_cmap = mpl_cm.get_cmap(mpl_name)
                    rgb = mpl_cmap(0.75)[:3]
                    hex_color = "#{:02x}{:02x}{:02x}".format(
                        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                    )
                    if channel_idx < len(self.channel_colors):
                        self.channel_colors[channel_idx] = hex_color

    def get_colormap_name(self, channel_idx):
        """Get the current colormap name for a channel."""
        return self.channel_colormaps.get(channel_idx, "White")

    def set_channel_visible(self, channel_idx, visible):
        """Set visibility for a specific channel in composite mode."""
        if channel_idx < len(self.layers):
            self.channel_visibility[channel_idx] = visible
            self._update_visibility()

    def get_channel_visible(self, channel_idx):
        """Get visibility state for a specific channel."""
        return self.channel_visibility.get(channel_idx, True)

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
