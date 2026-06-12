"""pyvistra - a light-weight image visualization tool

Based on vispy and PyQt (via qtpy)

"""

from ._version import __version__
from . import colormaps

from .ui.annotation_manager import (
    AnnotationManager,
    get_annotation_manager,
    show_annotation_manager,
)
from .data.colors import (
    DEFAULT_LABEL_PALETTE,
    SMART_LABEL_PALETTE,
    compute_adjacency_graph,
    compute_smart_colors,
)
from .apps.gel_analyzer import GelAnalyzerWidget, get_gel_analyzer, show_gel_analyzer
from .readers import ImarisReader, ImarisWriter
from .io import (
    CZI5DProxy,
    ImageBuffer,
    Imaris5DProxy,
    Numpy5DProxy,
    build_z_projection_metadata,
    load_image,
    load_sparse_labels,
    project_z_max,
    normalize_to_5d,
    save_imaris,
    save_sparse_labels,
    save_tiff,
)
from .visuals.labels import LabelOverlayVisual
from .data.labels import SparseLabels
from .ui.manager import WindowManager, manager
from .data.points import PointTable
from .viewers import OrthoViewer, VolumeViewer
from .rois import ROI, CircleROI, CoordinateROI, LaneROI, LineROI, RectangleROI
from .ui.toolbar import Toolbar
from .ui.layer_manager import LayerManager, get_layer_manager, show_layer_manager
from .data.tracks import TrackTable
from .data.shapes import ShapeData
from .layers import Layer, LayerList, UndoStack
from .ui.window import ImageWindow, imshow, run_app

from .widgets import (
    ImageOutputSelector,
    PSFComputeDialog,
    PupilComputeDialog,
    ZProjectionDialog,
)

__all__ = [
    "__version__",
    "colormaps",
    # io
    "load_image",
    "load_sparse_labels",
    "save_tiff",
    "save_imaris",
    "save_sparse_labels",
    "normalize_to_5d",
    "project_z_max",
    "build_z_projection_metadata",
    "Imaris5DProxy",
    "Numpy5DProxy",
    "CZI5DProxy",
    "ImageBuffer",
    # ui
    "ImageWindow",
    "Toolbar",
    "imshow",
    "run_app",
    # widgets
    "ImageOutputSelector",
    "PSFComputeDialog",
    "PupilComputeDialog",
    "ZProjectionDialog",
    # rois
    "ROI",
    "RectangleROI",
    "CircleROI",
    "LineROI",
    "CoordinateROI",
    "LaneROI",
    # labels/masks
    "SparseLabels",
    "LabelOverlayVisual",
    # tracks
    "TrackTable",
    # points
    "PointTable",
    # colors (smart coloring)
    "SMART_LABEL_PALETTE",
    "DEFAULT_LABEL_PALETTE",
    "compute_smart_colors",
    "compute_adjacency_graph",
    # layers
    "Layer",
    "LayerList",
    "UndoStack",
    "ShapeData",
    "LayerManager",
    "get_layer_manager",
    "show_layer_manager",
    # managers
    "AnnotationManager",
    "get_annotation_manager",
    "show_annotation_manager",
    "WindowManager",
    "manager",
    # gel analysis
    "GelAnalyzerWidget",
    "get_gel_analyzer",
    "show_gel_analyzer",
    # viewers
    "OrthoViewer",
    "VolumeViewer",
    # readers/writers
    "ImarisReader",
    "ImarisWriter",
]
