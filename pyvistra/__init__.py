"""pyvistra - a light-weight image visualization tool

Based on vispy and PyQt (via qtpy)

"""

__version__ = "0.1.2"

from .imaris_reader import ImarisReader
from .io import (
    ImageBuffer,
    Imaris5DProxy,
    Numpy5DProxy,
    load_image,
    load_psf,
    load_sparse_labels,
    normalize_to_5d,
    save_psf,
    save_sparse_labels,
    save_tiff,
)
from .label_manager import LabelManager, get_label_manager
from .label_visual import LabelOverlayVisual
from .labels import SparseLabels
from .manager import WindowManager, manager
from .ortho import OrthoViewer
from .volume import VolumeViewer
from .roi_manager import ROIManager, get_roi_manager
from .gel_analyzer import GelAnalyzerWidget, get_gel_analyzer, show_gel_analyzer
from .rois import ROI, CircleROI, CoordinateROI, LineROI, RectangleROI, LaneROI
from .ui import ImageWindow, Toolbar, imshow, run_app
from .widgets import ImageOutputSelector, PSFComputeDialog

__all__ = [
    "__version__",
    # io
    "load_image",
    "load_psf",
    "load_sparse_labels",
    "save_tiff",
    "save_psf",
    "save_sparse_labels",
    "normalize_to_5d",
    "Imaris5DProxy",
    "Numpy5DProxy",
    "ImageBuffer",
    # ui
    "ImageWindow",
    "Toolbar",
    "imshow",
    "run_app",
    # widgets
    "ImageOutputSelector",
    "PSFComputeDialog",
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
    "LabelManager",
    "get_label_manager",
    # managers
    "ROIManager",
    "get_roi_manager",
    "WindowManager",
    "manager",
    # gel analysis
    "GelAnalyzerWidget",
    "get_gel_analyzer",
    "show_gel_analyzer",
    # viewers
    "OrthoViewer",
    "VolumeViewer",
    # readers
    "ImarisReader",
]
