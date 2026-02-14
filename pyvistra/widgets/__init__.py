"""
pyvistra.widgets - UI widgets for image visualization

This package contains dialog widgets and histogram components used by pyvistra.
"""

# Histogram widgets and helpers
from .histogram import (
    HANDLE_COLOR,
    HANDLE_WIDTH,
    TEXT_COLOR,
    WIDGET_BG,
    CompactHistogramWidget,
    HistogramWidget,
    compute_spinbox_params,
    configure_spinbox_for_range,
    format_value_adaptive,
)

# Dialog widgets
from .contrast import ContrastDialog
from .channel_panel import ChannelPanel, ChannelRow
from .metadata import MetadataDialog
from .transform import TransformDialog
from .alignment import AlignmentDialog
from .output_selector import ImageOutputSelector
from .psf_dialog import PSFComputeDialog
from .axes_dialog import AxesDialog
from .line_profile import LineProfileDialog, get_line_profile_dialog, line_profile_dialog_exists
from .denoise_dialog import Denoise2DTimelapseDialog
from .processing_helper import BufferProcessingRunner
from .overlay_settings import OverlaySettingsDialog

__all__ = [
    # Theme constants
    "WIDGET_BG",
    "TEXT_COLOR",
    "HANDLE_COLOR",
    "HANDLE_WIDTH",
    # Helper functions
    "compute_spinbox_params",
    "format_value_adaptive",
    "configure_spinbox_for_range",
    # Histogram widgets
    "HistogramWidget",
    "CompactHistogramWidget",
    # Dialog widgets
    "ContrastDialog",
    "ChannelRow",
    "ChannelPanel",
    "MetadataDialog",
    "TransformDialog",
    "AlignmentDialog",
    "ImageOutputSelector",
    "PSFComputeDialog",
    "AxesDialog",
    "Denoise2DTimelapseDialog",
    "OverlaySettingsDialog",
    "BufferProcessingRunner",
    "LineProfileDialog",
    "get_line_profile_dialog",
    "line_profile_dialog_exists",
]
