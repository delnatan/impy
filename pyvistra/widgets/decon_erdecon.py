"""Dialog-state -> deconlib ER-Decon inputs.

`ERDeconDialogState` is the flat dataclass the ER-Decon dialog reads its
widgets into. Shape math and `prepare_inputs` are generic across every
deconvolution engine and live in `decon_common` -- re-exported here so
`erdecon_dialog.py`/`erdecon_worker.py` have a single import site,
mirroring `decon_nlcg.py`'s own re-export of the same names.

Unlike NLCG's optional gradient/Hessian regularizer, ER-Decon's
edge-preserving log(eps + curvature) regularizer is intrinsic to the
algorithm (it is not a knob that can be switched off) -- see
`deconlib.deconvolution.erdecon_mlx` for the objective. `build_regularizer`
below returns the curvature operator: either the default `Hessian2D`/
`Hessian3D` (weighted by the crop's actual, zoom-corrected voxel spacing so
`eps_reg` means the same physical thing on every axis -- the same
spacing-aware construction `decon_nlcg.build_regularizer` uses for its own
optional Hessian regularizer), or -- when `regularizer_type="wavelet"` -- a
single-scale a-trous wavelet operator (`AtrousAnalysisOperator`), which
tests found dominates the Hessian stencil on both smoothness and accuracy
at `levels=1` (see `erdecon_mlx` module docstring / `test_erdecon.py`
`TestWaveletRegularizer`). The wavelet path does *not* correct for
lateral/axial physical anisotropy the way the Hessian path does -- it
dilates by the same pixel-index step on every axis -- so it is offered
only at `levels<=2` (the range `calibrate_noise_weights` validates; deeper
stacks can ring, see its docstring).

A third option, `regularizer_type="combined"`, stacks both operators via
`StackOperator` with `combine_channels=False` so each operator's channels
are thresholded independently against the shared `eps_reg` -- a spurious
curvature spike then has to fool the Hessian *and* the wavelet channels
simultaneously to escape regularization, not just one. This fixed a
real-dataset failure mode where the wavelet regularizer alone produced an
isolated near-delta "hot pixel" spike (see
`widefield_erdecon_realdata_demo.py`'s REGULARIZER section for the full
writeup). Both operators' channels are normalized to ~unit noise std via
`calibrate_operator_noise_weights` (probed on the actual padded/object
domain, not just the wavelet's own per-scale calibration) so the one
shared `eps_reg` reads as a comparable curvature-in-noise-sigma scale
regardless of which operator produced a given channel.
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

    # Quadratic-in-curvature IRLS floor (0 = off). Without it, once a
    # voxel's curvature crosses `eps_reg` the log penalty's weight keeps
    # falling to 0 as curvature grows further, so nothing pulls flux back
    # -- the optimizer can over-sharpen a real-but-modest bump into an
    # isolated near-delta "hot pixel" spike. See
    # `deconlib.deconvolution.erdecon_mlx.erdecon_with_operator`'s
    # `floor_frac` docstring.
    floor_frac: float = 0.0

    # Regularizer operator: default spacing-weighted Hessian2D/Hessian3D, a
    # single-scale a-trous wavelet operator, or both stacked together (see
    # module docstring).
    regularizer_type: str = "hessian"       # "hessian" | "wavelet" | "combined"
    wavelet_levels: int = 1

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
# Regularizer operator (always on)

def build_regularizer(
    state: ERDeconDialogState, prepared: PreparedInputs
) -> Tuple[object, bool]:
    """Return ``(operator, combine_channels)`` for `erdecon_with_operator`.

    Default (``regularizer_type="hessian"``): a spacing-weighted deconlib
    `Hessian2D`/`Hessian3D` operator. Mirrors `decon_nlcg.build_regularizer`'s
    own Hessian construction (``r = dy / dz``) rather than
    `Hessian3D.from_spacing`, which raises on anisotropic lateral (Y/X)
    spacing -- a case this dialog otherwise handles fine. `combine_channels`
    is True: the Hessian's stacked components are one physical tensor's
    unique entries and share one per-pixel curvature/weight.

    ``regularizer_type="wavelet"``: a single-scale (or `wavelet_levels`-scale,
    capped at 2) a-trous wavelet operator, noise-calibrated so `eps_reg`
    means the same curvature-in-noise-sigma at every scale. `combine_channels`
    is False here -- the wavelet channels are decorrelated scales, not one
    tensor's components, so each needs its own IRLS weight (see module
    docstring).

    ``regularizer_type="combined"``: both operators above, stacked via
    `StackOperator` and each independently noise-calibrated (via
    `calibrate_operator_noise_weights`, probed on the actual padded/object
    domain built from `prepared`) so the shared `eps_reg` reads as a
    comparable curvature scale on every stacked channel. `combine_channels`
    is False, same reasoning as the wavelet-only path.
    """
    ndim = prepared.y.ndim
    voxel_spacing = prepared.voxel_spacing

    def _hessian_operator():
        from deconlib.deconvolution import Hessian2D, Hessian3D

        if ndim == 3:
            r = float(voxel_spacing[1]) / float(voxel_spacing[0])
            return Hessian3D(r=r)
        return Hessian2D()

    def _wavelet_operator():
        from deconlib.deconvolution import (
            AtrousAnalysisOperator,
            AtrousTransform,
            calibrate_noise_weights,
        )

        levels = max(1, min(2, int(state.wavelet_levels)))
        weights = calibrate_noise_weights(levels=levels, ndim=ndim, n_trials=16)
        transform = AtrousTransform(levels=levels, weights=weights, backend="mlx")
        return AtrousAnalysisOperator(transform)

    if state.regularizer_type == "wavelet":
        return _wavelet_operator(), False

    if state.regularizer_type == "combined":
        from deconlib.deconvolution import (
            StackOperator,
            calibrate_operator_noise_weights,
        )

        hessian_op = _hessian_operator()
        wavelet_op = _wavelet_operator()
        padded = padded_shape(
            visible_shape(prepared.y.shape, prepared.zoom), prepared.psf.shape
        )
        hessian_weights = calibrate_operator_noise_weights(
            hessian_op, padded, n_trials=8
        )
        wavelet_weights = calibrate_operator_noise_weights(
            wavelet_op, padded, n_trials=8
        )
        regularizer = StackOperator(
            [hessian_op, wavelet_op], weights=[hessian_weights, wavelet_weights]
        )
        return regularizer, False

    return _hessian_operator(), True
