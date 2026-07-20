"""File format readers for pyvistra.

Lazy re-exports (PEP 562): h5py / aicspylibczi only load when the
corresponding reader is actually used.
"""

_LAZY_IMPORTS = {
    "ImarisReader": (".imaris", "ImarisReader"),
    "ImarisWriter": (".imaris_writer", "ImarisWriter"),
    "save_imaris": (".imaris_writer", "save_imaris"),
    "CZIReader": (".czi", "CZIReader"),
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
