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
from .channel_panel import ChannelPanel, ChannelRow, show_channel_dock
from .metadata import MetadataDialog
from .transform import TransformDialog
from .alignment import AlignmentDialog
from .output_selector import ImageOutputSelector
from .region_selector import RegionSelector
from .psf_dialog import PSFComputeDialog
from .pupil_dialog import PupilComputeDialog
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
from .deconvolution_dialog import DeconvolutionDialog
from .deconvolution_worker import NLCGDeconvolutionWorker
from .richardson_lucy_dialog import RichardsonLucyDialog
from .richardson_lucy_worker import RLDeconvolutionWorker
from .memsolve_dialog import MemsolveDeconvolutionDialog
from .memsolve_worker import MemsolveDeconvolutionWorker
from .psf_distillation_dialog import PSFDistillationDialog
from .psf_distillation_worker import PSFDistillationWorker
from .processing_helper import BufferProcessingRunner
from .overlay_settings import OverlaySettingsDialog
from .zmontage_settings import ZMontageSettingsDialog

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
    "show_channel_dock",
    "MetadataDialog",
    "TransformDialog",
    "AlignmentDialog",
    "ImageOutputSelector",
    "RegionSelector",
    "PSFComputeDialog",
    "PupilComputeDialog",
    "AxesDialog",
    "ZProjectionDialog",
    "ImageMathDialog",
    "CombineImagesDialog",
    "FFTDialog",
    "DeconvolutionDialog",
    "NLCGDeconvolutionWorker",
    "RichardsonLucyDialog",
    "RLDeconvolutionWorker",
    "MemsolveDeconvolutionDialog",
    "MemsolveDeconvolutionWorker",
    "PSFDistillationDialog",
    "PSFDistillationWorker",
    "OverlaySettingsDialog",
    "ZMontageSettingsDialog",
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
