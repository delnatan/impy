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

# Vispy built-in catalog (perceptually uniform and other named colormaps),
# grouped by use — sequential, diverging (good for signed data with both
# positive and negative peaks — pair with a symmetric auto-contrast), and
# cyclic (good for phase, orientation, and any wrap-around quantity). These
# groupings also drive the submenus in colormap pickers; see `categories()`.
_VISPY_SEQUENTIAL = [
    "viridis", "plasma", "magma", "inferno", "cividis",
    "hot", "cool", "turbo", "gray", "gray_r",
]
_VISPY_DIVERGING = ["RdBu", "coolwarm", "diverging", "GrBu", "PuGr", "HiLo"]
_VISPY_CYCLIC = ["hsl", "husl"]
_VISPY_NAMED = _VISPY_SEQUENTIAL + _VISPY_DIVERGING + _VISPY_CYCLIC

# Names that produce a meaningful display only with a symmetric clim.
# Channel state may use this hint to anchor zero at the colormap centre
# even when one tail of the data dominates the percentile bracket.
DIVERGING_NAMES = frozenset(_VISPY_DIVERGING)

CYCLIC_NAMES = frozenset(_VISPY_CYCLIC)

# Emission wavelength (nm) breakpoints -> matching two-color channel colormap,
# in ascending order. Used to auto-assign a channel's default color from its
# emission wavelength metadata (when readers provide one) instead of cycling
# through a fixed palette. Blue/red emission map to Cyan/Magenta rather than
# literal Blue/Red -- pure blue and red are dim/low-contrast composited on a
# black canvas, while cyan and magenta stay visible and pop against green
# and orange neighbors.
_WAVELENGTH_COLORMAPS = [
    (500, "Cyan"),    # blue emission, e.g. DAPI ~460nm, CFP ~475nm
    (560, "Green"),   # green emission, e.g. GFP ~509nm, YFP ~527nm
    (600, "Orange"),  # yellow/orange emission, e.g. Cy3/TRITC ~570nm
]
_WAVELENGTH_FALLBACK = "Magenta"  # red/far-red emission, e.g. Cy5 ~670nm


def is_diverging(name: str) -> bool:
    """Whether ``name`` denotes a diverging colormap (zero-centred)."""
    return name in DIVERGING_NAMES


def is_cyclic(name: str) -> bool:
    """Whether ``name`` denotes a cyclic colormap."""
    return name in CYCLIC_NAMES

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
    return [name for _, group in categories() for name in group]


def categories() -> list:
    """Group colormap names into display categories, in display order.

    Returns a list of ``(category_label, [names...])`` pairs. Pickers that
    want submenus (rather than one flat list) iterate this instead of
    `names()`.
    """
    return [
        ("Channel Colors", list(_TWO_COLOR)),
        ("Sequential", list(_VISPY_SEQUENTIAL)),
        ("Diverging", list(_VISPY_DIVERGING)),
        ("Cyclic", list(_VISPY_CYCLIC)),
        ("Custom", list(_registry)),
    ]


def colormap_for_wavelength(wavelength_nm) -> "str | None":
    """Map an emission wavelength (nm) to a matching two-color colormap name.

    Returns ``None`` if *wavelength_nm* is missing or not a sane positive
    number, so callers can fall back to their own default (e.g. cycling a
    fixed palette). Expects an already-normalized float/None -- `load_image`
    coerces each channel's wavelength fields before this is ever called.
    """
    if wavelength_nm is None:
        return None
    try:
        wl = float(wavelength_nm)
    except (TypeError, ValueError):
        return None
    if not wl > 0:
        return None
    for breakpoint, name in _WAVELENGTH_COLORMAPS:
        if wl < breakpoint:
            return name
    return _WAVELENGTH_FALLBACK


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
