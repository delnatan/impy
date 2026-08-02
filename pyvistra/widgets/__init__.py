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
from .channel_panel import ChannelPanel, ChannelPopup, ChannelRow, show_channel_dock
from .metadata import MetadataDialog
from .transform import TransformDialog
from .alignment import AlignmentDialog
from .output_selector import ImageOutputSelector
from .region_selector import RegionSelector
from .source_selector import SourceSelector
from .file_list import FlaggableFileListWidget
from .axes_dialog import AxesDialog
from .line_profile import LineProfileDialog, get_line_profile_dialog, line_profile_dialog_exists
from .radial_profile_dialog import (
    RadialProfileDialog,
    get_radial_profile_dialog,
    radial_profile_dialog_exists,
)
from .convergence_plot import (
    ConvergencePlotDialog,
    ConvergencePlotWidget,
    get_convergence_comparison_dialog,
)
from .z_projection_dialog import ZProjectionDialog
from .image_math_dialog import ImageMathDialog
from .combine_images_dialog import CombineImagesDialog
from .fft_dialog import FFTDialog
from .processing_helper import BufferProcessingRunner
from .overlay_settings import OverlaySettingsDialog
from .zmontage_settings import ZMontageSettingsDialog
from .tiled_display_settings_dialog import TiledDisplaySettingsDialog
from .color_button import ColorButton
from .point_display_settings_dialog import PointDisplaySettingsDialog
from .track_display_settings_dialog import TrackDisplaySettingsDialog

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
    "ChannelRow",
    "ChannelPanel",
    "ChannelPopup",
    "show_channel_dock",
    "MetadataDialog",
    "TransformDialog",
    "AlignmentDialog",
    "ImageOutputSelector",
    "RegionSelector",
    "SourceSelector",
    "FlaggableFileListWidget",
    "AxesDialog",
    "ZProjectionDialog",
    "ImageMathDialog",
    "CombineImagesDialog",
    "FFTDialog",
    "OverlaySettingsDialog",
    "ZMontageSettingsDialog",
    "TiledDisplaySettingsDialog",
    "ColorButton",
    "PointDisplaySettingsDialog",
    "TrackDisplaySettingsDialog",
    "BufferProcessingRunner",
    "LineProfileDialog",
    "get_line_profile_dialog",
    "line_profile_dialog_exists",
    "RadialProfileDialog",
    "get_radial_profile_dialog",
    "radial_profile_dialog_exists",
    "ConvergencePlotWidget",
    "ConvergencePlotDialog",
    "get_convergence_comparison_dialog",
]
