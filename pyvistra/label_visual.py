"""
LabelOverlayVisual - Vispy visual for rendering SparseLabels as RGBA overlay.

Supports multiple render modes (filled, outline, both) and smart adjacency-aware
coloring for improved visual distinction between neighboring labels.
"""

import numpy as np
from vispy import scene

from .colors import DEFAULT_LABEL_PALETTE, compute_smart_colors


class LabelOverlayVisual:
    """
    Renders SparseLabels as a semi-transparent RGBA texture overlay.

    This visual creates an Image visual with per-pixel RGBA values
    where each label gets a distinct color. Renders above image
    channels but below ROIs.

    Supports three render modes:
    - "filled": Solid fill with semi-transparency (default)
    - "outline": Only boundary pixels are rendered
    - "both": Filled with white outline overlay

    Example:
        >>> overlay = LabelOverlayVisual(view, shape_yx=(512, 512))
        >>> overlay.set_labels(sparse_labels)
        >>> overlay.render_mode = "outline"
        >>> overlay.auto_color = True
        >>> overlay.refresh()
    """

    def __init__(
        self,
        view: scene.ViewBox,
        shape_yx: tuple[int, int],
        scale: tuple[float, float] = (1.0, 1.0),
        render_mode: str = "filled",
        auto_color: bool = False,
    ):
        """
        Initialize the label overlay visual.

        Args:
            view: Vispy ViewBox to add visual to
            shape_yx: Image dimensions (Y, X)
            scale: Pixel scale (sy, sx) to match image transform
            render_mode: "filled", "outline", or "both"
            auto_color: Use smart adjacency-aware coloring
        """
        self.view = view
        self.shape_yx = shape_yx
        self.scale = scale

        self._labels = None  # SparseLabels instance
        self._z_idx = 0  # Current Z slice for 3D data
        self._opacity = 0.5  # Global overlay opacity
        self._render_mode = render_mode
        self._auto_color = auto_color

        # Per-label state
        self._label_colors: dict[int, tuple[float, ...]] = {}
        self._label_visible: dict[int, bool] = {}

        # RGBA texture buffer
        self._texture = np.zeros(
            (shape_yx[0], shape_yx[1], 4), dtype=np.float32
        )

        # Create Image visual for the overlay
        self._image = scene.visuals.Image(
            self._texture,
            parent=view.scene,
            method="auto",
            interpolation="nearest",
        )

        # Set up alpha blending
        self._image.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )

        # Apply scale transform to match image
        sy, sx = scale
        from vispy.visuals.transforms.linear import STTransform

        self._image.transform = STTransform(scale=(sx, sy))

        # Render order: above image (negative order), below ROIs
        self._image.order = -100

        # Initially hidden
        self._image.visible = False

    @property
    def visible(self) -> bool:
        """Get overlay visibility."""
        return self._image.visible

    @visible.setter
    def visible(self, value: bool):
        """Set overlay visibility."""
        self._image.visible = value

    @property
    def render_mode(self) -> str:
        """Get render mode ("filled", "outline", or "both")."""
        return self._render_mode

    @render_mode.setter
    def render_mode(self, value: str):
        """Set render mode."""
        if value not in ("filled", "outline", "both"):
            raise ValueError(f"Invalid render_mode: {value}")
        self._render_mode = value
        self.refresh()

    @property
    def auto_color(self) -> bool:
        """Get auto-color mode (smart adjacency-aware coloring)."""
        return self._auto_color

    @auto_color.setter
    def auto_color(self, value: bool):
        """Set auto-color mode."""
        self._auto_color = value
        if value:
            self._recompute_smart_colors()
        self.refresh()

    def _recompute_smart_colors(self) -> None:
        """Recompute colors using smart adjacency-aware algorithm."""
        if self._labels is None or self._labels.n_objects == 0:
            return

        z_idx = self._z_idx if self._labels.ndim == 3 else None
        colors = compute_smart_colors(self._labels, z_idx=z_idx)
        self._label_colors.update(colors)

    def set_labels(self, labels) -> None:
        """
        Set the SparseLabels to render.

        Args:
            labels: SparseLabels instance (or None to clear)
        """
        self._labels = labels

        if labels is None:
            self._texture.fill(0)
            self._image.set_data(self._texture)
            self._image.visible = False
            return

        # Assign colors to new labels
        if self._auto_color:
            self._recompute_smart_colors()
        else:
            for label in labels:
                if label not in self._label_colors:
                    color_idx = (label - 1) % len(DEFAULT_LABEL_PALETTE)
                    self._label_colors[label] = DEFAULT_LABEL_PALETTE[color_idx]

        for label in labels:
            if label not in self._label_visible:
                self._label_visible[label] = True

        self._image.visible = True
        self.refresh()

    def update_slice(self, z_idx: int) -> None:
        """
        Update for 3D data by setting current Z slice.

        Args:
            z_idx: Z index to display
        """
        self._z_idx = z_idx
        if self._auto_color:
            self._recompute_smart_colors()
        self.refresh()

    def refresh(self) -> None:
        """Refresh the texture from current labels."""
        self._texture.fill(0)

        if self._labels is None:
            self._image.set_data(self._texture)
            return

        for label in self._labels:
            if not self._label_visible.get(label, True):
                continue

            color = self._label_colors.get(
                label, DEFAULT_LABEL_PALETTE[(label - 1) % len(DEFAULT_LABEL_PALETTE)]
            )

            coords = self._labels.coords(label)

            if self._labels.ndim == 3:
                # 3D: filter to current Z slice
                z_coords, y_coords, x_coords = coords
                z_mask = z_coords == self._z_idx

                if not np.any(z_mask):
                    continue

                y_slice = y_coords[z_mask]
                x_slice = x_coords[z_mask]
            else:
                # 2D: use directly
                y_slice, x_slice = coords

            # Bounds check
            valid = (
                (y_slice >= 0)
                & (y_slice < self.shape_yx[0])
                & (x_slice >= 0)
                & (x_slice < self.shape_yx[1])
            )
            y_valid = y_slice[valid]
            x_valid = x_slice[valid]

            if len(y_valid) == 0:
                continue

            # Render based on mode
            if self._render_mode == "filled":
                self._render_filled(y_valid, x_valid, color)
            elif self._render_mode == "outline":
                self._render_outline(y_valid, x_valid, color)
            else:  # "both"
                self._render_filled(y_valid, x_valid, color)
                # Overlay white outline
                self._render_outline(y_valid, x_valid, (1.0, 1.0, 1.0, 0.8))

        self._image.set_data(self._texture)

    def _render_filled(self, y_coords, x_coords, color: tuple) -> None:
        """Render filled pixels."""
        rgba = (
            color[0],
            color[1],
            color[2],
            color[3] * self._opacity,
        )
        self._texture[y_coords, x_coords] = rgba

    def _render_outline(self, y_coords, x_coords, color: tuple) -> None:
        """Render only boundary pixels using morphological erosion."""
        if len(y_coords) == 0:
            return

        # Create temporary mask for this label
        mask = np.zeros(self.shape_yx, dtype=bool)
        mask[y_coords, x_coords] = True

        # Find boundary using erosion
        try:
            from scipy.ndimage import binary_erosion

            # Erode the mask to find interior
            interior = binary_erosion(mask)
            # Boundary = original - interior
            boundary = mask & ~interior

            # Get boundary coordinates
            by, bx = np.where(boundary)

            if len(by) > 0:
                rgba = (
                    color[0],
                    color[1],
                    color[2],
                    color[3] * self._opacity,
                )
                self._texture[by, bx] = rgba
        except ImportError:
            # Fallback: render all pixels if scipy not available
            self._render_filled(y_coords, x_coords, color)

    def set_label_color(
        self, label: int, rgba: tuple[float, float, float, float]
    ) -> None:
        """
        Set color for a specific label.

        Args:
            label: Label ID
            rgba: Color as (r, g, b, a) with values in [0, 1]
        """
        self._label_colors[label] = rgba
        self.refresh()

    def get_label_color(self, label: int) -> tuple[float, float, float, float]:
        """Get color for a specific label."""
        return self._label_colors.get(
            label, DEFAULT_LABEL_PALETTE[(label - 1) % len(DEFAULT_LABEL_PALETTE)]
        )

    def set_label_visible(self, label: int, visible: bool) -> None:
        """
        Set visibility for a specific label.

        Args:
            label: Label ID
            visible: Whether to show this label
        """
        self._label_visible[label] = visible
        self.refresh()

    def get_label_visible(self, label: int) -> bool:
        """Get visibility for a specific label."""
        return self._label_visible.get(label, True)

    def set_opacity(self, alpha: float) -> None:
        """
        Set global overlay opacity.

        Args:
            alpha: Opacity value in [0, 1]
        """
        self._opacity = max(0.0, min(1.0, alpha))
        self.refresh()

    def get_opacity(self) -> float:
        """Get global overlay opacity."""
        return self._opacity

    def set_transform(self, scale: tuple[float, float]) -> None:
        """
        Update scale transform to match image.

        Args:
            scale: (sy, sx) pixel scale
        """
        self.scale = scale
        sy, sx = scale
        from vispy.visuals.transforms.linear import STTransform

        self._image.transform = STTransform(scale=(sx, sy))

    def remove(self) -> None:
        """Remove the visual from the scene."""
        self._image.parent = None

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.remove()
        except Exception:
            pass
