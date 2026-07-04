"""Dialog-state -> deconlib NLCG inputs.

`NLCGDialogState` is the flat dataclass the deconvolution dialog reads its
widgets into. `prepare_inputs` turns it plus the cropped observation/PSF
into the numeric payload `NLCGDeconvolutionWorker` hands to deconlib's
`make_forward_model` / `nlcg_with_operator` / `process_tiles`.

Shape math (`visible_shape`, `padded_shape`, `valid_slices`), input
preparation (`prepare_inputs`), and the PSF-preset helper are generic
across every deconvolution engine and live in `decon_common` -- re-exported
here (unchanged names) so this module keeps being the single import site
`deconvolution_dialog.py` already uses (`from . import decon_nlcg as dnl`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .decon_common import (  # noqa: F401 (re-exported for `dnl.*` call sites)
    PreparedInputs,
    compact_psf_shape_for_data,
    estimate_tile_plan,
    log,
    output_5d_shape,
    output_shape,
    padded_shape,
    per_axis_zoom,
    prepare_inputs,
    valid_slices,
    visible_shape,
)


# --------------------------------------------------------------------------- #
# State container

@dataclass(frozen=True)
class NLCGDialogState:
    # Forward model — independent lateral (XY) / axial (Z) zoom factors.
    # >= 1.0; fractional values use deconlib's FractionalAreaDownsample.
    zoom_xy: float = 1.0
    zoom_z: float = 1.0

    # Regularization (optional; off by default per deconlib's own guidance —
    # early stopping alone is usually enough).
    regularizer_kind: str = "none"          # "none" | "gradient" | "hessian"
    reg_weight: float = 0.0

    # NLCG solver knobs (see deconlib.deconvolution.nlcg_with_operator).
    num_iter: int = 150
    background: float = 0.0
    eval_interval: int = 1
    slack: float = 1.25                     # discrepancy-principle multiplier
    tol: float = 1e-4                       # Eq. 17 fallback threshold (deconlib's own default)
    min_iter: int = 10
    restart_interval: int = 0               # 0 -> disabled (None)
    newton_iters: int = 3
    verbose: bool = False                   # print per-iteration diagnostics

    # Output region (single-volume only; tiled output is always cropped).
    crop_to_visible: bool = True

    # Tiling (large fields; see deconlib.deconvolution.process_tiles).
    tiled: bool = False
    tile_size: int = 256
    guard_px: int = 0                       # 0 -> deconlib default (half PSF width)
    min_z_slices: int = 48


# --------------------------------------------------------------------------- #
# Regularizer

def build_regularizer(state: NLCGDialogState, ndim: int, voxel_spacing: Tuple[float, ...]):
    """Return a deconlib `LinearOperator` regularizer, or None when disabled."""
    if state.regularizer_kind == "none" or state.reg_weight <= 0.0:
        return None

    from deconlib.deconvolution import Gradient2D, Gradient3D, Hessian2D, Hessian3D

    if ndim == 3:
        r = float(voxel_spacing[1]) / float(voxel_spacing[0])
        if state.regularizer_kind == "gradient":
            return Gradient3D(r=r)
        return Hessian3D(r=r)
    if state.regularizer_kind == "gradient":
        return Gradient2D()
    return Hessian2D()
