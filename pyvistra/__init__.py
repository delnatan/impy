"""pyvistra - a light-weight image visualization tool

Based on vispy and PyQt (via qtpy)

"""

from ._version import __version__

from .annotation_manager import (
    AnnotationManager,
    get_annotation_manager,
    show_annotation_manager,
)
from .colors import (
    DEFAULT_LABEL_PALETTE,
    SMART_LABEL_PALETTE,
    compute_adjacency_graph,
    compute_smart_colors,
)
from .gel_analyzer import GelAnalyzerWidget, get_gel_analyzer, show_gel_analyzer
from .imaris_reader import ImarisReader
from .imaris_writer import ImarisWriter
from .io import (
    CZI5DProxy,
    ImageBuffer,
    Imaris5DProxy,
    Numpy5DProxy,
    load_image,
    load_psf,
    load_sparse_labels,
    normalize_to_5d,
    save_imaris,
    save_psf,
    save_sparse_labels,
    save_tiff,
)
from .label_visual import LabelOverlayVisual
from .labels import SparseLabels
from .manager import WindowManager, manager
from .ortho import OrthoViewer
from .rois import ROI, CircleROI, CoordinateROI, LaneROI, LineROI, RectangleROI
from .toolbar import Toolbar
from .ui import ImageWindow, imshow, run_app
from .volume import VolumeViewer
from .widgets import (
    Denoise2DTimelapseDialog,
    ImageOutputSelector,
    PSFComputeDialog,
)

__all__ = [
    "__version__",
    # io
    "load_image",
    "load_psf",
    "load_sparse_labels",
    "save_tiff",
    "save_imaris",
    "save_psf",
    "save_sparse_labels",
    "normalize_to_5d",
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
    "Denoise2DTimelapseDialog",
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
    # colors (smart coloring)
    "SMART_LABEL_PALETTE",
    "DEFAULT_LABEL_PALETTE",
    "compute_smart_colors",
    "compute_adjacency_graph",
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
