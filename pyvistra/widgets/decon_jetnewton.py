"""Dialog-state -> deconlib jetnewton inputs.

``JetNewtonDialogState`` is the flat dataclass the jetnewton dialog reads its
widgets into. Shape math and ``prepare_inputs`` are generic across every
deconvolution engine and live in ``decon_common`` -- re-exported here so
``jetnewton_dialog.py``/``jetnewton_worker.py`` have a single import site.

jetnewton (``deconlib.deconvolution.jetnewton_mlx``) is the successor to and
full replacement for the earlier ER-Decon dialog -- same kind of log-penalty
curvature regularizer, but solved by an *exact*-Hessian active-set projected
Newton method instead of a Gauss-Newton surrogate. Per the module's own "UI
design notes" (see
``jetnewton_mlx`` docstring, and
``deconlib/scripts/widefield_jetnewton_realdata_demo.py``), the parameter
surface is deliberately much smaller than ER-Decon's:

* ``beta`` is the *only* genuine user-facing knob (regularization strength).
* ``s0`` (intensity scale), ``ell``/``kappa`` (PSF length scale) and ``eta``
  (penalty saturation threshold) are never typed in -- they are derived here
  from required acquisition inputs (``noise_sigma``, the PSF, pixel spacing)
  via :func:`calibrate`, exactly mirroring the demo script's "s0/ell
  bookkeeping" and "eta CALIBRATION" sections.
* Optimizer knobs (``cg_max_steps``, ``newton_tol``, ``tol``, active-set
  params, ...) are advanced/hidden with sane fixed defaults, not swept from
  the UI.
* There is no wavelet/combined regularizer option (unlike ER-Decon) and no
  missing-cone/OTF knob -- both were considered for jetnewton and dropped
  after synthetic ground-truth testing found curvature-only better or equal
  in every case tried (see ``jetnewton_mlx`` module docstring). Don't add
  either without repeating that validation.
* There is no ``normalize`` knob -- unlike ER-Decon, jetnewton's own
  non-dimensionalization (``x = s0 * x_tilde``) already makes the solve
  scale-invariant; no separate amplitude normalization is needed.
* No explicit ``init`` is passed to ``jetnewton_with_operator`` -- its
  default (an adjoint backprojection of the data, clipped to ``s0``) is a
  better starting guess than a flat background-level array, and the demo
  script's flat ``initial`` was only for parity with the ER-Decon demo, not
  a requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .decon_common import (  # noqa: F401 (re-exported for `jn.*` call sites)
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
class JetNewtonDialogState:
    # Forward model — independent lateral (XY) / axial (Z) zoom factors.
    zoom_xy: float = 1.0
    zoom_z: float = 1.0

    # Data term.
    data_term: str = "poisson"              # "poisson" | "gaussian"

    # Required acquisition inputs — camera/microscope preset, not tuned per
    # run. `background` is read from the shared Source/PSF tab's
    # `background_spin` (`SourcePSFPanelMixin`); `noise_sigma` is
    # jetnewton-specific and has its own field on this dialog's Solver tab.
    background: float = 0.0
    noise_sigma: float = 15.0               # measured data-space noise sigma

    # The one genuine UI knob (log-scale). See module docstring.
    beta: float = 1e-2

    # Solver knobs — primary.
    num_iter: int = 60
    eval_interval: int = 5
    verbose: bool = False

    # Advanced/hidden optimizer knobs — sane fixed defaults (see module
    # docstring and `jetnewton_with_operator`'s own docstring).
    cg_max_steps: int = 150
    newton_tol: float = 1e-4
    tol: float = 0.0
    min_iter: int = 3
    eps_bar: float = 1e-2
    freeze_tau: float = 1e-3
    freeze_delta: float = 1e-6
    ls_max_backtracks: int = 30
    ls_sigma: float = 1e-4

    # Output region (single-volume only; tiled output is always cropped).
    crop_to_visible: bool = True

    # Tiling (large fields; see deconlib.deconvolution.process_tiles).
    tiled: bool = False
    tile_size: int = 256
    guard_px: int = 0                       # 0 -> deconlib default (half PSF width)
    min_z_slices: int = 48


# --------------------------------------------------------------------------- #
# PSF length scales (ell) -- local estimate, not part of deconlib.

def estimate_psf_length_scales(
    psf_arr: np.ndarray,
    spacing: Tuple[float, ...],
    tail_threshold: float = 0.01,
) -> Tuple[float, ...]:
    """Per-axis PSF radius of gyration (physical units).

    Thresholds the tail first so long low-amplitude diffraction wings don't
    inflate it. ``psf_arr`` is corner-origin (DC at index 0, this codebase's
    convention -- ``_crop_observation_and_psf`` already fftshifts any PSF
    window not flagged ``psf_dc_corner``), but the coordinate grid here
    assumes a centered array -- ``fftshift`` first to align them. Skipping
    this makes the estimated radius of gyration tens of times too large (the
    peak sits ~n/2 pixels from where the moment calculation assumes it is),
    which squares into a huge ``kappa**2`` blowup and silently breaks the
    regularizer's non-dimensionalization -- see
    [[jetnewton_projected_newton]].
    """
    psf_np = np.fft.fftshift(np.asarray(psf_arr, dtype=np.float64))
    psf_np = np.where(psf_np >= tail_threshold * psf_np.max(), psf_np, 0.0)
    total = psf_np.sum()
    coords = [
        (np.arange(n) - (n - 1) / 2.0) * h for n, h in zip(psf_np.shape, spacing)
    ]
    grids = np.meshgrid(*coords, indexing="ij")
    centroid = [float((psf_np * g).sum() / total) for g in grids]
    ell = []
    for g, c in zip(grids, centroid):
        var = float((psf_np * (g - c) ** 2).sum() / total)
        ell.append(float(np.sqrt(var)))
    return tuple(ell)


# --------------------------------------------------------------------------- #
# Calibration (s0, ell/kappa, eta) -- always auto-computed, never a UI knob.

@dataclass(frozen=True)
class Calibration:
    """Everything jetnewton needs beyond `beta`, derived from required inputs."""

    ell: Tuple[float, ...]
    kappa: Tuple[float, ...]
    s0: float
    eta: float
    noise_floor: dict                       # {"mean", "median", "p1", "p99"}


def calibrate(state: JetNewtonDialogState, prepared: PreparedInputs):
    """Return ``(hessian, Calibration)`` for `jetnewton_with_operator`.

    Mirrors the demo script's "PSF LENGTH SCALES" / "eta CALIBRATION"
    sections exactly: ``ell`` from :func:`estimate_psf_length_scales`,
    ``kappa = ell / voxel_spacing`` via ``AnisotropicHessian2D``/``3D``, `s0`
    from the measured data-space noise sigma converted to visible-space
    (``1/sqrt(prod(zoom))``, correct for a standard deviation under the
    flux-summing downsample -- do NOT derive this from `background`, an
    unrelated camera baseline clamp), and `eta` from
    ``estimate_penalty_noise_floor`` probed on the padded/object domain this
    crop's forward model actually builds.
    """
    from deconlib.deconvolution import (
        AnisotropicHessian2D,
        AnisotropicHessian3D,
        estimate_penalty_noise_floor,
    )

    ell = estimate_psf_length_scales(prepared.psf, prepared.voxel_spacing)
    if prepared.y.ndim == 3:
        hessian = AnisotropicHessian3D.from_lengths(ell, prepared.voxel_spacing)
    else:
        hessian = AnisotropicHessian2D.from_lengths(ell, prepared.voxel_spacing)

    visible = visible_shape(prepared.y.shape, prepared.zoom)
    padded = padded_shape(visible, prepared.psf.shape)
    probe = estimate_penalty_noise_floor(hessian, padded, n_trials=8)
    eta = probe["curvature"]["median"]

    s0 = float(state.noise_sigma / np.sqrt(float(np.prod(prepared.zoom))))

    return hessian, Calibration(
        ell=ell, kappa=hessian.kappa, s0=s0, eta=eta,
        noise_floor=probe["curvature"],
    )
