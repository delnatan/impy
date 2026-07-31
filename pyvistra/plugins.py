"""Lightweight plugin discovery for pyvistra.

Downstream packages (e.g. a PSF-computation or deconvolution library) can
wire themselves into pyvistra's menus and file-format registries without
pyvistra ever importing them by name. A plugin package declares itself in
its own ``pyproject.toml``:

    [project.entry-points."pyvistra.plugins"]
    mypackage = "mypackage._pyvistra_plugin:register"

and ``mypackage/_pyvistra_plugin.py`` exposes a zero-arg ``register()``
that calls the functions below to add menu items / shape context-menu
entries / file formats / colormaps. Nothing else is required --
``pip install mypackage`` is enough for pyvistra to pick it up.

Discovery is triggered lazily, from ``build_menus``/``build_proxy_menus``
(``ui/window.py``/``ui/workspace.py``) the first time any viewer actually
builds a menu bar -- not at import time. This keeps ``import pyvistra``
fast (no plugin's Qt/vispy/numeric dependencies load until the GUI is
actually starting) and guarantees every viewer class a plugin might want
to target (``ImageWindow``, ``OrthoViewer``, ``TiledViewer``, ...) is
already imported by the time ``register()`` runs.
"""

import importlib.metadata
import warnings

from .colormaps import register as register_colormap
from .io import register_input_format, register_output_format

__all__ = [
    "discover_plugins",
    "add_menu_item",
    "register_output_format",
    "register_input_format",
    "register_colormap",
    "register_shape_context_action",
    "shape_context_actions_for",
]

_discovered = False


def discover_plugins():
    """Load every entry point in the "pyvistra.plugins" group and call it.

    Idempotent -- safe to call from every menu-build site. A plugin's
    register() raising (missing optional dependency, bug, ...) only
    warns; it never blocks pyvistra from starting or other plugins from
    loading.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True
    for entry_point in importlib.metadata.entry_points(group="pyvistra.plugins"):
        try:
            entry_point.load()()
        except Exception as exc:
            warnings.warn(f"pyvistra plugin {entry_point.name!r} failed to load: {exc}")


def add_menu_item(target_cls, menu_title, item, submenu=None):
    """Append a MENU_SPEC-shaped *item* into *target_cls*.MENU_SPEC.

    Args:
        target_cls: A viewer class declaring a class-level ``MENU_SPEC``
            (e.g. ``pyvistra.ui.window.ImageWindow``,
            ``pyvistra.viewers.TiledViewer``).
        menu_title: Top-level menu to append into (e.g. ``"Image"``),
            created if not already present.
        item: A MENU_SPEC leaf dict (``{"label", "method", ...}``) or a
            nested ``{"label", "submenu": [...]}`` group.
        submenu: Optional label of an existing (or newly created) nested
            submenu within *menu_title* to append *item* into instead of
            appending directly to the top-level menu.

    Must be called from a plugin's ``register()``, before any window of
    *target_cls* has built its menu bar -- guaranteed by the
    discover_plugins() call sites in build_menus()/build_proxy_menus().
    Mutates the class's MENU_SPEC list in place, so both the embedded
    menu bar and the Workspace's mirrored bar pick up the addition with
    no further wiring.
    """
    spec = target_cls.MENU_SPEC
    for title, items in spec:
        if title != menu_title:
            continue
        if submenu is None:
            items.append(item)
            return
        for existing in items:
            if existing and existing.get("submenu") is not None and existing.get("label") == submenu:
                existing["submenu"].append(item)
                return
        items.append({"label": submenu, "submenu": [item]})
        return
    new_items = [item] if submenu is None else [{"label": submenu, "submenu": [item]}]
    spec.append((menu_title, new_items))


_shape_context_actions = []  # (target_cls, shape_types, label, method, tooltip)


def register_shape_context_action(target_cls, shape_types, label, method, tooltip=None):
    """Register a right-click context-menu entry for shapes of the given
    type(s) on *target_cls* (e.g. ``ImageWindow``).

    *method* is looked up on the target instance and called as
    ``method(layer, shape_id)`` when triggered -- same dispatch convention
    as a ``MENU_SPEC`` leaf's ``"method"`` key. *shape_types* is a single
    shape-type constant or a tuple of them (see ``pyvistra.data.shapes``).
    """
    if isinstance(shape_types, str):
        shape_types = (shape_types,)
    _shape_context_actions.append((target_cls, tuple(shape_types), label, method, tooltip))


def shape_context_actions_for(target_cls, shape_type):
    """Return ``[(label, method, tooltip), ...]`` registered for
    *target_cls* matching *shape_type*, in registration order."""
    return [
        (label, method, tooltip)
        for (cls, types, label, method, tooltip) in _shape_context_actions
        if cls is target_cls and shape_type in types
    ]
