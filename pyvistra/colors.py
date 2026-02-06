"""
Smart coloring algorithms for label visualization.

Provides adjacency-aware color assignment using graph coloring to maximize
visual distinction between neighboring labels.
"""

import numpy as np
from typing import Optional


# Perceptually distinct color palette (10 colors)
# Based on ColorBrewer qualitative palettes with high distinguishability
SMART_LABEL_PALETTE = [
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

# Default palette (same as SMART_LABEL_PALETTE for backward compatibility)
DEFAULT_LABEL_PALETTE = SMART_LABEL_PALETTE


def compute_adjacency_graph(labels, z_idx: Optional[int] = None) -> dict[int, set[int]]:
    """
    Build adjacency graph from label coordinates using morphological dilation.

    Two labels are adjacent if any of their pixels are neighbors (8-connected for 2D).

    Args:
        labels: SparseLabels instance
        z_idx: For 3D data, only consider this Z slice. If None, uses all slices.

    Returns:
        Dictionary mapping label -> set of adjacent labels
    """
    if labels is None or labels.n_objects == 0:
        return {}

    # Build dense mask for faster neighbor checking
    shape = labels.shape[-2:]  # (Y, X)
    label_map = np.zeros(shape, dtype=np.int32)

    # Populate label map
    for label in labels:
        coords = labels.coords(label)

        if labels.ndim == 3:
            if z_idx is not None:
                # Filter to current Z slice
                z_coords, y_coords, x_coords = coords
                z_mask = z_coords == z_idx
                if not np.any(z_mask):
                    continue
                y_slice = y_coords[z_mask]
                x_slice = x_coords[z_mask]
            else:
                # Use all Z (project to 2D)
                _, y_slice, x_slice = coords
        else:
            y_slice, x_slice = coords

        # Bounds check
        valid = (
            (y_slice >= 0) & (y_slice < shape[0]) &
            (x_slice >= 0) & (x_slice < shape[1])
        )
        label_map[y_slice[valid], x_slice[valid]] = label

    # Build adjacency graph using dilation
    adjacency: dict[int, set[int]] = {label: set() for label in labels}

    # 8-connected neighborhood offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for label in labels:
        coords = labels.coords(label)

        if labels.ndim == 3:
            if z_idx is not None:
                z_coords, y_coords, x_coords = coords
                z_mask = z_coords == z_idx
                if not np.any(z_mask):
                    continue
                y_slice = y_coords[z_mask]
                x_slice = x_coords[z_mask]
            else:
                _, y_slice, x_slice = coords
        else:
            y_slice, x_slice = coords

        # Check neighbors for each pixel
        for dy, dx in offsets:
            ny = y_slice + dy
            nx = x_slice + dx

            # Bounds check
            valid = (ny >= 0) & (ny < shape[0]) & (nx >= 0) & (nx < shape[1])
            ny_valid = ny[valid]
            nx_valid = nx[valid]

            if len(ny_valid) == 0:
                continue

            # Get neighbor labels
            neighbor_labels = label_map[ny_valid, nx_valid]

            # Add unique neighbors (excluding self and background)
            for nl in np.unique(neighbor_labels):
                if nl != 0 and nl != label:
                    adjacency[label].add(int(nl))

    return adjacency


def welsh_powell_coloring(adjacency: dict[int, set[int]]) -> dict[int, int]:
    """
    Apply Welsh-Powell graph coloring algorithm.

    This greedy algorithm tends to use few colors by processing vertices
    in order of decreasing degree (number of neighbors).

    Args:
        adjacency: Graph as dict mapping node -> set of neighbors

    Returns:
        Dictionary mapping node -> color index (0-indexed)
    """
    if not adjacency:
        return {}

    # Sort vertices by degree (descending)
    vertices = sorted(adjacency.keys(), key=lambda v: len(adjacency[v]), reverse=True)

    # Color assignment
    colors: dict[int, int] = {}

    for vertex in vertices:
        # Find colors used by neighbors
        neighbor_colors = {colors[n] for n in adjacency[vertex] if n in colors}

        # Find smallest available color
        color = 0
        while color in neighbor_colors:
            color += 1

        colors[vertex] = color

    return colors


def compute_smart_colors(
    labels,
    z_idx: Optional[int] = None,
    palette: list[tuple] = None,
) -> dict[int, tuple]:
    """
    Compute smart adjacency-aware colors for labels.

    Uses the Four Color Theorem approach: builds adjacency graph and applies
    graph coloring to ensure adjacent labels have different colors.

    Args:
        labels: SparseLabels instance
        z_idx: For 3D data, compute adjacency on this Z slice only
        palette: Color palette to use (default: SMART_LABEL_PALETTE)

    Returns:
        Dictionary mapping label -> RGBA color tuple
    """
    if labels is None or labels.n_objects == 0:
        return {}

    if palette is None:
        palette = SMART_LABEL_PALETTE

    # Build adjacency graph
    adjacency = compute_adjacency_graph(labels, z_idx)

    # Apply graph coloring
    color_indices = welsh_powell_coloring(adjacency)

    # Map color indices to palette colors
    result = {}
    n_colors = len(palette)

    for label in labels:
        if label in color_indices:
            color_idx = color_indices[label] % n_colors
        else:
            # Fallback for isolated labels
            color_idx = (label - 1) % n_colors

        result[label] = palette[color_idx]

    return result


def get_distinct_color(
    index: int,
    palette: list[tuple] = None,
    alpha: float = 0.5,
) -> tuple:
    """
    Get a color from the palette by index (with wrapping).

    Args:
        index: Color index (will wrap around palette)
        palette: Color palette to use
        alpha: Override alpha value

    Returns:
        RGBA color tuple
    """
    if palette is None:
        palette = SMART_LABEL_PALETTE

    color = palette[index % len(palette)]
    return (color[0], color[1], color[2], alpha)


def blend_colors(color1: tuple, color2: tuple, ratio: float = 0.5) -> tuple:
    """
    Blend two RGBA colors.

    Args:
        color1: First RGBA color
        color2: Second RGBA color
        ratio: Blend ratio (0 = all color1, 1 = all color2)

    Returns:
        Blended RGBA color
    """
    return tuple(c1 * (1 - ratio) + c2 * ratio for c1, c2 in zip(color1, color2))


def rgb_to_hex(color: tuple) -> str:
    """Convert RGB(A) tuple to hex string."""
    r, g, b = color[:3]
    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255), int(g * 255), int(b * 255)
    )


def hex_to_rgb(hex_color: str, alpha: float = 1.0) -> tuple:
    """Convert hex string to RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r, g, b, alpha)
