"""pyvistra - a light-weight image visualization tool

Based on vispy and PyQt (via qtpy)

"""

__version__ = "0.1.2"

from .imaris_reader import ImarisReader
from .io import (
    Imaris5DProxy,
    Numpy5DProxy,
    load_image,
    normalize_to_5d,
    save_tiff,
)
from .manager import WindowManager, manager
from .ortho import OrthoViewer
from .volume import VolumeViewer
from .roi_manager import ROIManager, get_roi_manager
from .gel_analyzer import GelAnalyzerWidget, get_gel_analyzer, show_gel_analyzer
from .rois import ROI, CircleROI, CoordinateROI, LineROI, RectangleROI, LaneROI
from .ui import ImageWindow, Toolbar, imshow, run_app

__all__ = [
    "__version__",
    # io
    "load_image",
    "save_tiff",
    "normalize_to_5d",
    "Imaris5DProxy",
    "Numpy5DProxy",
    # ui
    "ImageWindow",
    "Toolbar",
    "imshow",
    "run_app",
    # rois
    "ROI",
    "RectangleROI",
    "CircleROI",
    "LineROI",
    "CoordinateROI",
    "LaneROI",
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
