"""Dialog-state -> memsolve MaxEnt (MEM) inputs.

`MemsolveDialogState` is the flat dataclass the MEM dialog reads its
widgets into. `build_mem_problem` mirrors
`deconlib/scripts/memsolve_gaussian_icf.py`: a `GaussianICF` hidden->visible
operator, the same padded/visible forward model deconlib's other engines
use (`deconlib.deconvolution.make_forward_model`), and a flat MaxEnt prior.

Shape math is generic across every deconvolution engine and lives in
`decon_common` -- reused here for the dialog's recipe-preview labels
(the actual run instead asks `deconlib.deconvolution.make_forward_model`
for its `padded_shape`/`visible_shape`/`valid_slices`, so the two never
drift apart).

Unlike `richardson_lucy_with_operator`/NLCG, there is no separate additive
detector-background parameter here: memsolve's `LinearInverseProblem`
only accepts a plain linear `R`/`Rt` pair (wrapping `R` as
``lambda f: R(f) + background`` would make it affine, breaking
`mem.validate_problem`'s adjoint self-check), and a constant background
is naturally absorbed into the MaxEnt prior instead -- exactly what
`memsolve_gaussian_icf.py` does (no background handling at all, since its
flat prior already represents the baseline flux level). The shared
Detector/data group's Background field (from `SourcePSFPanelMixin`) is
disabled in the MEM dialog for this reason.

The prior (MaxEnt default model `m`) is `RCt(y)` -- the adjoint operator
applied to the observation -- rather than a flat value. `mem`'s hidden-space
MAP also initializes the starting iterate `h0` from the prior
(`_entropy_default`), so this simultaneously gives MaxEnt a non-flat default
model and a data-informed starting point, matching
`memsolve_gaussian_icf.py`. `RCt` is built from non-negative operators
(crop/downsample/convolve adjoints, Gaussian-blur ICF) applied to a
non-negative observation, so it is non-negative in exact arithmetic --- in
practice the FFT-based convolve/downsample adjoints can still ring very
slightly negative in float32, so it's floored at `_ADJOINT_PRIOR_FLOOR`
rather than the old constant-value prior's much larger padding floor.

Tiled (large-field) runs are different: they can't share one adjoint-based
prior across tiles (each tile only sees its own local data, so `RCt(y_tile)`
would give every tile a different absolute baseline and show up as seams
at tile boundaries once stitched). `memsolve_worker.MemsolveDeconvolutionWorker
._run_tiled` instead mirrors
`deconlib/scripts/tiled_memsolve_demo.py`: one flat prior value
(`mean(y) / prod(zoom)`), shared by every tile, built once alongside the
shared per-tile forward model/ICF (every tile reads a window of the same
shape). Because that prior isn't data-derived per tile, the dialog's tiled
path skips `build_mem_problem` entirely and instead prepares plain
`(y, psf, zoom, voxel_spacing)` via `decon_common.prepare_inputs` --
the worker does the shared-model/prior construction itself, once it knows
the real tile shape from `plan_tiles`.

`build_mem_problem` also accepts an optional `prior_source`: a plain array
at visible-grid resolution (`model.visible_shape`), e.g. read straight out
of another open window (`memsolve_dialog.py`'s prior-source combo lists
every open window the same way the PSF combo does). This is how the dialog
supports experimenting with priors deconlib itself doesn't build --
a low-pass-filtered version of the data, an over-regularized NLCG solution,
anything -- without teaching this module to compute any of those itself:
the caller does whatever it wants to produce a visible-grid array (via
another dialog, another window, external code) and this just embeds it via
`decon_common.embed_edge_padded`, which edge-extends it out to
`padded_shape` so the PSF-support margin isn't an artificial flat/zero
rim. Single-volume only, same as the default adjoint prior -- there's no
tiled equivalent yet (see the tiling paragraph above for why a per-tile
version of this is a separate, harder problem).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .decon_common import (  # noqa: F401 (re-exported for `dmem.*` call sites)
    PreparedInputs,
    _prepare_observation,
    _prepare_psf_kernel,
    compact_psf_shape_for_data,
    effective_voxel_spacing,
    embed_edge_padded,
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

# Numerical-noise floor for the single-volume path's RCt(y) prior -- see
# `build_mem_problem`'s comment. Much smaller than the tiled path's flat
# prior floor (`memsolve_worker._TILED_PADDING_PRIOR_VALUE`), since this
# only needs to clear float32 ringing, not mark a genuinely flux-free
# padding region.
_ADJOINT_PRIOR_FLOOR = 1e-6


@dataclass(frozen=True)
class MemsolveDialogState:
    # Forward model — independent lateral (XY) / axial (Z) zoom factors.
    zoom_xy: float = 1.0
    zoom_z: float = 1.0

    # Gaussian ICF (isotropic sigma, physical units matching voxel spacing).
    icf_gamma_um: float = 0.085

    # Data likelihood passed straight through to mem.LinearInverseProblem /
    # mem.MaxEntProblem ("poisson" | "gaussian"). "gaussian" omits a
    # per-pixel sigma (deconlib doesn't hand this dialog a live
    # noise-estimate map), so mem falls back to unit sigma -- an unweighted
    # least-squares chi2.
    likelihood: str = "poisson"

    # MaxEnt MAP solver knobs (see mem.maxent.MaxEntConfig).
    max_iter: int = 50
    tol_omega: float = 0.05
    aim: float = 1.0
    rate: float = 0.2
    omega_mode: str = "classic"             # "classic" | "auto"
    cg_epsilon: float = 1e-2
    cg_max_steps: int = 50
    n_probe_g: int = 1
    seed: int = 0
    # Include chi2/omega/alpha in each per-iteration status line (mirrors
    # RLDialogState.verbose); the line itself is emitted regardless, since
    # the worker drives the loop itself rather than depending on
    # MaxEntConfig.print_outer's stdout diagnostics.
    verbose: bool = False

    # Interval (outer iterations) for live-preview buffer writes. Each
    # preview costs an extra curvature-diagnostic evaluation (see
    # memsolve_worker.py), so this throttles that overhead the same way
    # RLDialogState.eval_interval throttles RL's cheaper per-iteration work.
    eval_interval: int = 1

    # Output region (single-volume only; tiled output is always cropped).
    crop_to_visible: bool = True

    # Tiling (large fields). memsolve's LinearInverseProblem doesn't compose
    # with deconlib.deconvolution.process_tiles's per-tile solver contract
    # (its RL/NLCG-shaped callback), so the worker drives its own per-tile
    # MAP loop instead -- see `MemsolveDeconvolutionWorker._run_tiled` and
    # `deconlib/scripts/tiled_memsolve_demo.py`.
    tiled: bool = False
    tile_size: int = 140
    guard_px: int = 0                       # 0 -> deconlib default (half PSF width)
    min_z_slices: int = 48


@dataclass(frozen=True)
class MemProblemInputs:
    """Numeric payload `MemsolveDeconvolutionWorker` hands to `mem`."""

    y: np.ndarray                           # cropped observation
    prior: np.ndarray                       # flat MaxEnt prior, padded_shape
    R: object                               # visible -> data
    Rt: object
    C: object                               # hidden -> visible (GaussianICF)
    Ct: object
    RC: object                              # combined hidden -> data
    RCt: object
    zoom: Tuple[float, ...]
    voxel_spacing: Tuple[float, ...]
    padded_shape: Tuple[int, ...]
    visible_shape: Tuple[int, ...]
    valid_slices: Tuple[slice, ...]


def build_mem_problem(
    *,
    state: MemsolveDialogState,
    y_obs: np.ndarray,
    psf_array: np.ndarray,
    psf_pixel_size_um: Tuple[float, ...],
    prior_source: Optional[np.ndarray] = None,
) -> MemProblemInputs:
    """Build the operator chain + prior `MemsolveDeconvolutionWorker` needs.

    Mirrors `memsolve_gaussian_icf.py`'s forward model exactly: padded
    visible -> convolve -> downsample -> crop -> data, plus a Gaussian ICF
    hidden -> visible operator.

    `prior_source`, if given, is a visible-grid array (shape ==
    `model.visible_shape`) used as the prior instead of the default
    `RCt(y)` -- see the module docstring's "prior_source" paragraph.
    """
    from deconlib.deconvolution import GaussianICF, as_numpy_op, make_forward_model

    ndim = y_obs.ndim
    if psf_array.ndim != ndim:
        raise ValueError(
            f"PSF ndim {psf_array.ndim} does not match observation ndim {ndim}"
        )

    zoom = per_axis_zoom(state, ndim)
    psf = _prepare_psf_kernel(psf_array)
    y = _prepare_observation(y_obs)
    voxel_spacing = tuple(float(s) for s in psf_pixel_size_um)

    model = make_forward_model(psf, y.shape, zoom)
    R, Rt = as_numpy_op(model.op)

    icf = GaussianICF(
        shape=model.padded_shape,
        sigmas=(float(state.icf_gamma_um),) * ndim,
        spacings=voxel_spacing,
        normalize=True,
    )
    C, Ct = as_numpy_op(icf)  # C == Ct (self-adjoint)

    def RC(h):
        return R(C(h))

    def RCt(u):
        return Ct(Rt(u))

    if prior_source is None:
        # Floored, not just cast: RCt is built from non-negative operators
        # applied to a non-negative observation, but the FFT-based
        # convolve/downsample adjoints can still ring very slightly negative
        # in float32 at the padded domain's edges -- `mem.validate_problem`
        # rejects a prior that isn't strictly positive everywhere, so clip
        # that noise floor rather than let it fail intermittently depending
        # on the data.
        prior = np.maximum(np.asarray(RCt(y), dtype=np.float32), _ADJOINT_PRIOR_FLOOR)
        prior_desc = "RCt(y)"
    else:
        prior = np.maximum(
            embed_edge_padded(prior_source, model.padded_shape, model.valid_slices),
            _ADJOINT_PRIOR_FLOOR,
        )
        prior_desc = "prior_source (edge-padded)"

    log.info(
        "build_mem_problem: y=%s psf=%s zoom=%s icf_gamma=%g voxel_spacing=%s "
        "padded_shape=%s prior=%s min/max=%.6g/%.6g",
        tuple(y.shape), tuple(psf.shape), zoom, state.icf_gamma_um,
        voxel_spacing, model.padded_shape, prior_desc,
        float(prior.min()), float(prior.max()),
    )

    return MemProblemInputs(
        y=y,
        prior=prior,
        R=R,
        Rt=Rt,
        C=C,
        Ct=Ct,
        RC=RC,
        RCt=RCt,
        zoom=zoom,
        voxel_spacing=voxel_spacing,
        padded_shape=model.padded_shape,
        visible_shape=model.visible_shape,
        valid_slices=model.valid_slices,
    )


def problem_output_5d_shape(
    problem_inputs: MemProblemInputs,
    *,
    crop_to_visible: bool,
    n_channels: int = 1,
    n_frames: int = 1,
) -> Tuple[int, int, int, int, int]:
    """(T, Z, C, Y, X) output shape for an already-built `MemProblemInputs`.

    Uses the real `visible_shape`/`padded_shape` the worker's forward model
    produced, rather than recomputing from `decon_common`'s dependency-free
    shape math, so the output buffer can never disagree with the actual run.
    """
    spatial = problem_inputs.visible_shape if crop_to_visible else problem_inputs.padded_shape
    C = max(1, int(n_channels))
    T = max(1, int(n_frames))
    if len(spatial) == 2:
        return (T, 1, C, spatial[0], spatial[1])
    return (T, spatial[0], C, spatial[1], spatial[2])
