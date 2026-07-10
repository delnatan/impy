"""Dialog-state -> deconlib ER-Decon inputs.

`ERDeconDialogState` is the flat dataclass the ER-Decon dialog reads its
widgets into. Shape math and `prepare_inputs` are generic across every
deconvolution engine and live in `decon_common` -- re-exported here so
`erdecon_dialog.py`/`erdecon_worker.py` have a single import site,
mirroring `decon_nlcg.py`'s own re-export of the same names.

Unlike NLCG's optional gradient/Hessian regularizer, ER-Decon's
edge-preserving Hessian-log regularizer is intrinsic to the algorithm (it
is not a knob that can be switched off) -- see
`deconlib.deconvolution.erdecon_mlx` for the objective. `build_hessian`
below always returns a `Hessian2D`/`Hessian3D` operator, weighted by the
crop's actual (zoom-corrected) voxel spacing so the curvature threshold
`eps_reg` means the same physical thing on every axis -- the same
spacing-aware construction `decon_nlcg.build_regularizer` uses for its
own optional Hessian regularizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .decon_common import (  # noqa: F401 (re-exported for `der.*` call sites)
    PreparedInputs,
    compact_psf_shape_for_data,
    effective_voxel_spacing,
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
class ERDeconDialogState:
    # Forward model — independent lateral (XY) / axial (Z) zoom factors.
    # >= 1.0; fractional values use deconlib's FractionalAreaDownsample.
    zoom_xy: float = 1.0
    zoom_z: float = 1.0

    # Edge-preserving Hessian-log regularizer (always on; see module
    # docstring). `eps_reg` is an absolute curvature threshold in units of
    # |Hg|^2, not a fraction of `reg_weight`.
    reg_weight: float = 0.05
    eps_reg: float = 1e-2
    data_term: str = "gaussian"             # "gaussian" | "poisson"

    # Gauss-Newton-CG solver knobs (see
    # deconlib.deconvolution.erdecon_with_operator).
    num_iter: int = 50
    background: float = 0.0
    normalize: bool = True
    eval_interval: int = 5
    newton_tol: float = 1e-3                # primary convergence test
    tol: float = 0.0                        # secondary (data-misfit) test, off by default
    min_iter: int = 5
    cg_max_steps: int = 25
    cg_tol: float = 0.1
    ls_max_backtracks: int = 30
    ls_c1: float = 1e-4
    verbose: bool = False                   # print per-iteration diagnostics

    # Output region (single-volume only; tiled output is always cropped).
    crop_to_visible: bool = True

    # Tiling (large fields; see deconlib.deconvolution.process_tiles).
    tiled: bool = False
    tile_size: int = 256
    guard_px: int = 0                       # 0 -> deconlib default (half PSF width)
    min_z_slices: int = 48


# --------------------------------------------------------------------------- #
# Hessian regularizer (always on)

def build_hessian(state: ERDeconDialogState, ndim: int, voxel_spacing: Tuple[float, ...]):
    """Return a spacing-weighted deconlib `Hessian2D`/`Hessian3D` operator.

    Mirrors `decon_nlcg.build_regularizer`'s own Hessian construction
    (``r = dy / dz``) rather than `Hessian3D.from_spacing`, which raises on
    anisotropic lateral (Y/X) spacing -- a case this dialog otherwise
    handles fine.
    """
    from deconlib.deconvolution import Hessian2D, Hessian3D

    if ndim == 3:
        r = float(voxel_spacing[1]) / float(voxel_spacing[0])
        return Hessian3D(r=r)
    return Hessian2D()
