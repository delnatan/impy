"""Dialog-state -> deconlib PSF-distillation-via-NLCG inputs.

PSF distillation swaps the roles of PSF and data relative to ordinary
deconvolution. The usual forward model is::

    reconstruction (unknown object) -> convolve(known PSF) -> data

For PSF distillation the knowns and unknowns trade places: bead positions
and amplitudes are detected first (matched filter against a theoretical
initial PSF), giving a fixed *ideal point-source* comb -- a sum of delta
functions, one per bead, scaled by its estimated amplitude. That comb is
now the fixed kernel; the unknown being reconstructed is the PSF itself::

    reconstruction (unknown PSF) -> convolve(fixed bead comb) -> data

Convolution is symmetric in its two arguments, so this swapped operator is
exactly as valid an input to ``nlcg_with_operator`` as the usual one -- see
``deconlib/scripts/psf_distillation_nlcg_demo.py``, which this module (and
:class:`~.psf_distillation_worker.PSFDistillationWorker`) mirrors directly.

``PSFDistillDialogState`` is the flat dataclass the PSF distillation dialog
reads its widgets into. ``prepare_inputs`` runs bead detection and builds
the fixed bead-comb kernel from the cropped observation and initial PSF;
the resulting :class:`PreparedDistillInputs` is the numeric payload handed
to ``PSFDistillationWorker``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Shared logger for the PSF distillation stack (mirrors decon_nlcg.log).
log = logging.getLogger("pyvistra.psf_distill")
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[psf-distill] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


# --------------------------------------------------------------------------- #
# State container

@dataclass(frozen=True)
class PSFDistillDialogState:
    # Point-source detection -- matched filter against the initial PSF.
    min_separation: int = 15
    min_intensity: float = 100.0

    # Object generation -- the fixed bead-comb kernel. Folding the bead's
    # physical extent + pixel integration into the kernel lets NLCG recover
    # the pure optical PSF rather than PSF-convolved-with-measurement-blur.
    bead_diameter: float = 0.0      # um; 0 disables the bead-extent term
    pixel_integration: bool = True

    # Regularization (optional; off by default). Note the effective weight
    # scale is *not* the same as deconvolution's: the reconstruction here is
    # a unit-flux PSF (values ~0.01-0.05), not photon counts in the
    # thousands, so beta typically needs to be ~1e3-1e6 (vs. ~1e-5-1e-2 for
    # deconvolution's object domain) to compete with the Poisson data term.
    regularizer_kind: str = "none"  # "none" | "gradient" | "hessian"
    reg_weight: float = 0.0

    # NLCG solver knobs (see deconlib.deconvolution.nlcg_with_operator).
    num_iter: int = 150
    background: float = 0.0
    eval_interval: int = 5
    slack: float = 0.0              # discrepancy principle is unreliable here
    tol: float = 1e-4               # Eq. 17 fallback threshold (deconlib default)
    min_iter: int = 20
    restart_interval: int = 0       # 0 -> disabled (None)
    newton_iters: int = 3
    verbose: bool = False           # print per-iteration diagnostics

    # Output.
    center_output: bool = True      # ifftshift for display; else DC-at-corner


@dataclass(frozen=True)
class PreparedDistillInputs:
    """Numeric payload the worker hands to deconlib."""

    y: np.ndarray                   # cropped bead observation, data space
    init_psf: np.ndarray            # positive, corner-origin initial PSF (matched-filter template + NLCG init)
    object_comb_eff: np.ndarray     # fixed kernel: bead comb, optionally measurement-blurred; matches y.shape
    voxel_spacing: Tuple[float, ...]
    positions: np.ndarray           # detected bead positions, shape (n_beads, ndim)
    amplitudes: np.ndarray          # matched-filter bead amplitudes, shape (n_beads,)


# --------------------------------------------------------------------------- #
# Shape math (local, dependency-free -- same rationale as decon_nlcg)

def padded_shape(
    data_shape: Tuple[int, ...], psf_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Minimum wrap-free linear-convolution domain (data_shape + psf_shape - 1).

    The worker rounds this up per-axis to an FFT-friendly size via
    deconlib's ``fast_padded_shape``; this is only for the dialog preview.
    """
    return tuple(d + p - 1 for d, p in zip(data_shape, psf_shape))


def output_5d_shape(psf_shape: Tuple[int, ...]) -> Tuple[int, int, int, int, int]:
    """Return the (T, Z, C, Y, X) shape of the distilled PSF output."""
    if len(psf_shape) == 2:
        return (1, 1, 1, psf_shape[0], psf_shape[1])
    return (1, psf_shape[0], 1, psf_shape[1], psf_shape[2])


# PSF preset helper -- reused as-is from decon_nlcg (generic, dependency-free).
from .decon_nlcg import compact_psf_shape_for_data  # noqa: E402,F401


# --------------------------------------------------------------------------- #
# Regularizer

def build_regularizer(state: PSFDistillDialogState, ndim: int, voxel_spacing: Tuple[float, ...]):
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


# --------------------------------------------------------------------------- #
# Input preparation

def _prepare_psf_kernel(psf_array: np.ndarray) -> np.ndarray:
    """Return a finite, nonnegative initial PSF (matched-filter template + NLCG init)."""
    psf = np.asarray(psf_array, dtype=np.float32)
    if not np.all(np.isfinite(psf)):
        raise ValueError("Initial PSF contains NaN or Inf values")

    min_value = float(psf.min()) if psf.size else 0.0
    negative_count = int(np.count_nonzero(psf < 0))
    if negative_count:
        log.warning(
            "initial PSF contains %d negative values (min %.6g); clipping "
            "to zero before bead detection",
            negative_count, min_value,
        )
        psf = np.maximum(psf, 0.0)

    total = float(np.sum(psf, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Initial PSF must have positive finite flux after clipping")
    return np.ascontiguousarray(psf, dtype=np.float32)


def _prepare_observation(y_obs: np.ndarray) -> np.ndarray:
    """Validate the bead image and enforce nonnegative support (Poisson-only)."""
    y = np.ascontiguousarray(y_obs, dtype=np.float32)
    if not np.all(np.isfinite(y)):
        raise ValueError("Observation contains NaN or Inf values")

    negative_count = int(np.count_nonzero(y < 0))
    if negative_count:
        log.warning(
            "observation contains %d negative values (min %.6g); "
            "clipping to zero for Poisson NLCG",
            negative_count, float(y.min()),
        )
        y = np.maximum(y, 0.0).astype(np.float32, copy=False)
    return y


def detect_beads(
    *,
    state: PSFDistillDialogState,
    y_obs: np.ndarray,
    init_psf_array: np.ndarray,
):
    """Run only the point-source detection stage (matched filter + peak-find).

    Separated from :func:`prepare_inputs` so the dialog can show detected
    positions as a points layer for visual review/editing *before*
    committing to a distillation run -- see the "Find Beads" button in
    ``psf_distillation_dialog.py``. Returns a
    ``deconlib.psf.distillation.BeadDetectionResult`` (``.positions``,
    ``.amplitudes``, ``.peak_intensities``); ``.positions`` is in
    `y_obs`-array-index order (``(y, x)`` for 2D, ``(z, y, x)`` for 3D).
    """
    from deconlib.psf.distillation import detect_beads as _detect_beads

    ndim = y_obs.ndim
    if init_psf_array.ndim != ndim:
        raise ValueError(
            f"Initial PSF ndim {init_psf_array.ndim} does not match "
            f"observation ndim {ndim}"
        )

    y = _prepare_observation(y_obs)
    init_psf = _prepare_psf_kernel(init_psf_array)

    result = _detect_beads(
        y, state.background, init_psf,
        min_separation=state.min_separation,
        min_intensity=state.min_intensity,
    )
    if result.positions.shape[0] == 0:
        raise ValueError(
            "No beads detected -- lower min intensity or check the "
            "background estimate."
        )
    log.info(
        "detect_beads: y=%s init_psf=%s beads=%d min_separation=%d min_intensity=%g",
        tuple(y.shape), tuple(init_psf.shape), result.positions.shape[0],
        state.min_separation, state.min_intensity,
    )
    return result


def prepare_inputs(
    *,
    state: PSFDistillDialogState,
    y_obs: np.ndarray,
    init_psf_array: np.ndarray,
    pixel_size_um: Tuple[float, ...],
    positions: np.ndarray,
) -> PreparedDistillInputs:
    """Build the (y, init_psf, object_comb_eff, ...) payload from reviewed beads.

    `y_obs` is the cropped bead observation (2D or 3D). `init_psf_array` is
    the theoretical/measured PSF used as NLCG's starting point -- it must be
    sampled at `pixel_size_um`, the same spacing as `y_obs` (no
    forward-model zoom in this problem). `positions` (shape ``(n_beads,
    ndim)``, `y_obs`-array-index order) comes from the points layer the user
    reviewed after :func:`detect_beads` -- amplitudes are always recomputed
    fresh from the current PSF/data here rather than trusted from an earlier
    detection pass, so manually added/moved points are handled correctly.
    """
    from scipy import fft as sfft

    from deconlib.psf.distillation import (
        fft_convolve,
        make_measurement_otf,
        matched_filter_amplitudes,
    )

    ndim = y_obs.ndim
    if init_psf_array.ndim != ndim:
        raise ValueError(
            f"Initial PSF ndim {init_psf_array.ndim} does not match "
            f"observation ndim {ndim}"
        )

    y = _prepare_observation(y_obs)
    init_psf = _prepare_psf_kernel(init_psf_array)

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != ndim:
        raise ValueError(
            f"positions must have shape (n_beads, {ndim}), got {positions.shape}"
        )
    positions = np.round(positions)

    int_pos = positions.astype(int)
    in_bounds = np.ones(len(int_pos), dtype=bool)
    for d, n in enumerate(y.shape):
        in_bounds &= (int_pos[:, d] >= 0) & (int_pos[:, d] < n)
    n_dropped = int((~in_bounds).sum())
    if n_dropped:
        log.warning(
            "%d bead point(s) fall outside the cropped region and will be "
            "ignored", n_dropped,
        )
    positions = positions[in_bounds]
    if positions.shape[0] == 0:
        raise ValueError("No beads within the cropped region to distill.")

    amplitudes = matched_filter_amplitudes(y, positions, init_psf, state.background)

    int_pos = positions.astype(int)
    idx = tuple(int_pos[:, d] for d in range(int_pos.shape[1]))
    object_comb = np.zeros(y.shape, dtype=np.float32)
    np.add.at(object_comb, idx, amplitudes.astype(np.float32))

    voxel_spacing = tuple(float(s) for s in pixel_size_um)
    if state.bead_diameter > 0 or state.pixel_integration:
        measurement_otf = make_measurement_otf(
            init_psf.shape, voxel_spacing,
            bead_diameter=state.bead_diameter,
            pixel_integration=state.pixel_integration,
        )
        k_meas = sfft.irfftn(measurement_otf, s=init_psf.shape).real.astype(np.float32)
        object_comb_eff = fft_convolve(object_comb, k_meas).astype(np.float32)
    else:
        object_comb_eff = object_comb

    log.info(
        "prepare_inputs: y=%s init_psf=%s beads=%d voxel_spacing=%s "
        "bead_diameter=%s pixel_integration=%s regularizer=%s(weight=%s)",
        tuple(y.shape), tuple(init_psf.shape), len(positions), voxel_spacing,
        state.bead_diameter, state.pixel_integration,
        state.regularizer_kind, state.reg_weight,
    )

    return PreparedDistillInputs(
        y=y,
        init_psf=init_psf,
        object_comb_eff=object_comb_eff,
        voxel_spacing=voxel_spacing,
        positions=positions,
        amplitudes=amplitudes,
    )
