"""Core viewer, toolbar, and managers for pyvistra.

Re-exports are lazy (PEP 562) so importing ``pyvistra.ui.manager`` from
a viewer or widget never drags in ``ui.window`` — that eager import was
the hub of every ui ↔ viewers ↔ widgets cycle.
"""

_LAZY_IMPORTS = {
    "ImageWindow": (".window", "ImageWindow"),
    "imshow": (".window", "imshow"),
    "run_app": (".window", "run_app"),
    "Toolbar": (".toolbar", "Toolbar"),
    "WindowManager": (".manager", "WindowManager"),
    "manager": (".manager", "manager"),
    "LayerManager": (".layer_manager", "LayerManager"),
    "get_layer_manager": (".layer_manager", "get_layer_manager"),
    "show_layer_manager": (".layer_manager", "show_layer_manager"),
}

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name):
    try:
        module_path, attr = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    import importlib

    value = getattr(importlib.import_module(module_path, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
