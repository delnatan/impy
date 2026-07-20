"""Specialized multi-panel image viewers for pyvistra.

Lazy re-exports (PEP 562): ``ui.window`` imports ``OrthoViewer`` from
here while the ortho/tiled/volume modules import ``ui.manager`` — lazy
resolution keeps that from being a cycle.
"""

_LAZY_IMPORTS = {
    "OrthoViewer": (".ortho", "OrthoViewer"),
    "VolumeViewer": (".volume", "VolumeViewer"),
    "TiledViewer": (".tiled", "TiledViewer"),
    "ZMontageViewer": (".zmontage", "ZMontageViewer"),
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
