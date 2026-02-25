"""Colormap registry for pyvistra. No matplotlib dependency.

Two-color additive-blending colormaps (black → color) for channel display,
plus delegation to vispy's built-in perceptually uniform colormaps and a
user-extensible registry for custom colormaps.

Usage::

    import pyvistra.colormaps as cmap

    cmap.get('viridis')         # returns (vispy.color.Colormap, display_color)
    cmap.get('Green')           # two-stop: black → #49FF49
    cmap.names()                # list all available names

    # Register a custom colormap
    cmap.register('MyLUT', ['black', '#FF6600'])
    cmap.register('MyArray', my_256x4_float_array)
"""

from pathlib import Path

import numpy as np
from vispy.color import Colormap
from vispy.color import get_colormap as _vispy_get_colormap

# Two-color (black → X) colormaps for additive channel blending.
# Value: [start_color, end_color] — end_color is used as the histogram swatch color.
_TWO_COLOR = {
    # Microscope channel defaults
    "Orange":     ["black", "#ffb100"],
    "Green":      ["black", "#49FF49"],
    "Cyan":       ["black", "#5BD6FF"],
    "Magenta":    ["black", "magenta"],
    "Yellow":     ["black", "yellow"],
    "White":      ["black", "white"],
    # Standard RGB primaries
    "Red":        ["black", "red"],
    "Pure Green": ["black", "#00FF00"],
    "Blue":       ["black", "blue"],
}

# Vispy built-in catalog (perceptually uniform and other named colormaps).
_VISPY_NAMED = [
    "viridis", "plasma", "magma", "inferno", "cividis",
    "hot", "cool", "coolwarm", "turbo", "gray", "gray_r",
]

# User-registered custom colormaps: name → stops or (N,4) LUT array
_registry: dict = {}


def register(name: str, stops) -> None:
    """Register a custom colormap, overwriting any existing entry with that name.

    Args:
        name:  Colormap name.
        stops: Color stops — a list of vispy-compatible color specs, or a
               (N, 4) float32 RGBA array (LUT). Examples::

                   ['black', '#FF6600']          # two-stop gradient
                   [(0,0,0,1), (0.5,0,1,1)]      # two-stop RGBA tuples
                   numpy.zeros((256, 4), 'f4')   # full LUT
    """
    _registry[name] = stops


def names() -> list:
    """Return all available colormap names in display order."""
    return list(_TWO_COLOR) + list(_VISPY_NAMED) + list(_registry)


def get(name: str) -> tuple:
    """Get a colormap by name.

    Returns:
        ``(vispy.color.Colormap, display_color)``

        *display_color* is a hex string (e.g. ``'#49FF49'``) used for the
        histogram swatch.  It is ``None`` for multi-color / perceptually
        uniform colormaps where no single representative color exists.
    """
    # User registry takes precedence
    if name in _registry:
        stops = _registry[name]
        cmap = Colormap(stops)
        # Two-stop with a plain string end-color → use it directly
        if (
            isinstance(stops, (list, tuple))
            and len(stops) == 2
            and isinstance(stops[1], str)
        ):
            return cmap, stops[1]
        return cmap, None

    # Built-in two-color colormaps
    if name in _TWO_COLOR:
        stops = _TWO_COLOR[name]
        return Colormap(stops), stops[1]

    # Vispy built-in catalog (viridis, plasma, etc.)
    try:
        cmap = _vispy_get_colormap(name)
        return cmap, None
    except Exception:
        pass

    # Unknown name — fall back to white with a warning
    import warnings
    warnings.warn(f"Unknown colormap '{name}', falling back to 'White'.")
    stops = _TWO_COLOR["White"]
    return Colormap(stops), stops[1]


def _load_bundled():
    """Auto-register bundled colormaps from _resources/colormaps/*.csv."""
    cmap_dir = Path(__file__).parent / "_resources" / "colormaps"
    if not cmap_dir.is_dir():
        return
    for csv_path in sorted(cmap_dir.glob("*.csv")):
        rgb = np.loadtxt(csv_path, delimiter=",")
        rgba = np.ones((rgb.shape[0], 4), dtype=np.float32)
        rgba[:, :3] = rgb
        register(csv_path.stem, rgba)


_load_bundled()
