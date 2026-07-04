"""pyvistra - a light-weight image visualization tool

Based on vispy and PyQt (via qtpy)

Public names are re-exported lazily (PEP 562): ``import pyvistra`` does
not pull in Qt, vispy, or any reader library until the corresponding
attribute is first touched. This also means submodules can never form an
import cycle through the package root.
"""

from ._version import __version__

# Public name -> (relative module, attribute). None attribute means the
# submodule itself. Resolved on first access by __getattr__ below.
_LAZY_IMPORTS = {
    "colormaps": (".colormaps", None),
    # io
    "load_image": (".io", "load_image"),
    "load_sparse_labels": (".io", "load_sparse_labels"),
    "save_tiff": (".io", "save_tiff"),
    "save_imaris": (".io", "save_imaris"),
    "save_sparse_labels": (".io", "save_sparse_labels"),
    "normalize_to_5d": (".io", "normalize_to_5d"),
    "project_z_max": (".io", "project_z_max"),
    "build_z_projection_metadata": (".io", "build_z_projection_metadata"),
    "Imaris5DProxy": (".io", "Imaris5DProxy"),
    "Numpy5DProxy": (".io", "Numpy5DProxy"),
    "CZI5DProxy": (".io", "CZI5DProxy"),
    "ImageBuffer": (".io", "ImageBuffer"),
    # ui
    "ImageWindow": (".ui.window", "ImageWindow"),
    "Toolbar": (".ui.toolbar", "Toolbar"),
    "imshow": (".ui.window", "imshow"),
    "run_app": (".ui.window", "run_app"),
    "Workspace": (".ui.workspace", "Workspace"),
    "get_workspace": (".ui.workspace", "get_workspace"),
    "present_window": (".ui.workspace", "present_window"),
    # widgets
    "ImageOutputSelector": (".widgets", "ImageOutputSelector"),
    "PSFComputeDialog": (".widgets", "PSFComputeDialog"),
    "PupilComputeDialog": (".widgets", "PupilComputeDialog"),
    "ZProjectionDialog": (".widgets", "ZProjectionDialog"),
    # rois (legacy)
    "ROI": (".rois", "ROI"),
    "RectangleROI": (".rois", "RectangleROI"),
    "CircleROI": (".rois", "CircleROI"),
    "LineROI": (".rois", "LineROI"),
    "CoordinateROI": (".rois", "CoordinateROI"),
    "LaneROI": (".rois", "LaneROI"),
    # labels/masks
    "SparseLabels": (".data.labels", "SparseLabels"),
    "LabelOverlayVisual": (".visuals.labels", "LabelOverlayVisual"),
    # tracks / points
    "TrackTable": (".data.tracks", "TrackTable"),
    "PointTable": (".data.points", "PointTable"),
    # colors (smart coloring)
    "SMART_LABEL_PALETTE": (".data.colors", "SMART_LABEL_PALETTE"),
    "DEFAULT_LABEL_PALETTE": (".data.colors", "DEFAULT_LABEL_PALETTE"),
    "compute_smart_colors": (".data.colors", "compute_smart_colors"),
    "compute_adjacency_graph": (".data.colors", "compute_adjacency_graph"),
    # layers
    "Layer": (".layers", "Layer"),
    "LayerList": (".layers", "LayerList"),
    "UndoStack": (".layers", "UndoStack"),
    "ShapeData": (".data.shapes", "ShapeData"),
    "LayerManager": (".ui.layer_manager", "LayerManager"),
    "get_layer_manager": (".ui.layer_manager", "get_layer_manager"),
    "show_layer_manager": (".ui.layer_manager", "show_layer_manager"),
    # managers
    "WindowManager": (".ui.manager", "WindowManager"),
    "manager": (".ui.manager", "manager"),
    # gel analysis
    "GelAnalyzerWidget": (".apps.gel_analyzer", "GelAnalyzerWidget"),
    "get_gel_analyzer": (".apps.gel_analyzer", "get_gel_analyzer"),
    "show_gel_analyzer": (".apps.gel_analyzer", "show_gel_analyzer"),
    # viewers
    "OrthoViewer": (".viewers", "OrthoViewer"),
    "VolumeViewer": (".viewers", "VolumeViewer"),
    # readers/writers
    "ImarisReader": (".readers", "ImarisReader"),
    "ImarisWriter": (".readers", "ImarisWriter"),
}

__all__ = ["__version__"] + sorted(_LAZY_IMPORTS)


def __getattr__(name):
    try:
        module_path, attr = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    import importlib

    module = importlib.import_module(module_path, __name__)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value  # cache: __getattr__ won't fire again
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
