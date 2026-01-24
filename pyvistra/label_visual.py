"""
LabelOverlayVisual - Vispy visual for rendering SparseLabels as RGBA overlay.
"""

import numpy as np
from vispy import scene


# Default color palette for labels (colorblind-friendly)
DEFAULT_LABEL_COLORS = [
    (0.90, 0.10, 0.29, 0.5),  # Red
    (0.23, 0.70, 0.29, 0.5),  # Green
    (0.00, 0.51, 0.78, 0.5),  # Blue
    (0.96, 0.51, 0.19, 0.5),  # Orange
    (0.57, 0.12, 0.71, 0.5),  # Purple
    (0.27, 0.94, 0.94, 0.5),  # Cyan
    (0.94, 0.20, 0.90, 0.5),  # Magenta
    (0.98, 0.75, 0.00, 0.5),  # Yellow
    (0.00, 0.50, 0.50, 0.5),  # Teal
    (0.86, 0.75, 1.00, 0.5),  # Lavender
]


class LabelOverlayVisual:
    """
    Renders SparseLabels as a semi-transparent RGBA texture overlay.

    This visual creates an Image visual with per-pixel RGBA values
    where each label gets a distinct color. Renders above image
    channels but below ROIs.

    Example:
        >>> overlay = LabelOverlayVisual(view, shape_yx=(512, 512))
        >>> overlay.set_labels(sparse_labels)
        >>> overlay.refresh()
    """

    def __init__(
        self,
        view: scene.ViewBox,
        shape_yx: tuple[int, int],
        scale: tuple[float, float] = (1.0, 1.0),
    ):
        """
        Initialize the label overlay visual.

        Args:
            view: Vispy ViewBox to add visual to
            shape_yx: Image dimensions (Y, X)
            scale: Pixel scale (sy, sx) to match image transform
        """
        self.view = view
        self.shape_yx = shape_yx
        self.scale = scale

        self._labels = None  # SparseLabels instance
        self._z_idx = 0  # Current Z slice for 3D data
        self._opacity = 0.5  # Global overlay opacity

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
        for label in labels:
            if label not in self._label_colors:
                color_idx = (label - 1) % len(DEFAULT_LABEL_COLORS)
                self._label_colors[label] = DEFAULT_LABEL_COLORS[color_idx]
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
                label, DEFAULT_LABEL_COLORS[(label - 1) % len(DEFAULT_LABEL_COLORS)]
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

            # Apply color with global opacity
            rgba = (
                color[0],
                color[1],
                color[2],
                color[3] * self._opacity,
            )
            self._texture[y_valid, x_valid] = rgba

        self._image.set_data(self._texture)

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
            label, DEFAULT_LABEL_COLORS[(label - 1) % len(DEFAULT_LABEL_COLORS)]
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
