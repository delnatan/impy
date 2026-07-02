"""Dialog-state -> deconlib NLCG inputs.

`NLCGDialogState` is the flat dataclass the deconvolution dialog reads its
widgets into. `prepare_inputs` turns it plus the cropped observation/PSF
into the numeric payload `NLCGDeconvolutionWorker` hands to deconlib's
`make_forward_model` / `nlcg_with_operator` / `process_tiles`.

Shape math (`visible_shape`, `padded_shape`, `valid_slices`) is a local,
dependency-free mirror of deconlib's own `compute_visible_shape` /
`compute_padded_shape` / `get_valid_slices` (see
`deconlib/deconvolution/shapes.py`). It exists so dialog preview/output-shape
computation doesn't require `deconlib`/`mlx` to be installed — neither is a
dependency of pyvistra itself, and the dialog only imports them lazily at
run time, same as the PSF dialog.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Single shared logger for the deconvolution stack. Attach a stderr handler
# on first import so messages show up without extra setup; silence with
# ``logging.getLogger("pyvistra.decon").setLevel(logging.WARNING)``.
log = logging.getLogger("pyvistra.decon")
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[decon] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


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


@dataclass(frozen=True)
class PreparedInputs:
    """Numeric payload the worker hands to deconlib."""

    y: np.ndarray                           # cropped observation, data space
    psf: np.ndarray                         # positive, corner-origin, visible-grid spacing
    zoom: Tuple[float, ...]                 # per-axis, matches y.ndim
    voxel_spacing: Tuple[float, ...]        # visible-grid spacing (regularizer r)


# --------------------------------------------------------------------------- #
# Per-axis helpers

def per_axis_zoom(state: NLCGDialogState, ndim: int) -> Tuple[float, ...]:
    """Per-axis zoom factor: `(fz, fy, fx)` (3D) or `(fy, fx)` (2D)."""
    fxy = max(float(state.zoom_xy), 1.0)
    fz = max(float(state.zoom_z), 1.0)
    if ndim == 3:
        return (fz, fxy, fxy)
    return (fxy, fxy)


# --------------------------------------------------------------------------- #
# Shape math (local mirror of deconlib.deconvolution.shapes)

def visible_shape(data_shape: Tuple[int, ...], zoom: Tuple[float, ...]) -> Tuple[int, ...]:
    """Reconstruction shape at visible pixel spacing (data_shape * zoom)."""
    return tuple(max(1, int(round(d * z))) for d, z in zip(data_shape, zoom))


def padded_shape(
    visible: Tuple[int, ...], psf_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Reconstruction domain: visible + PSF margins (M-1 per axis, split symmetric)."""
    return tuple(v + (m - 1) for v, m in zip(visible, psf_shape))


def valid_slices(
    padded: Tuple[int, ...], visible: Tuple[int, ...]
) -> Tuple[slice, ...]:
    """Slices cropping `padded_shape` back down to `visible_shape`."""
    slices = []
    for p, v in zip(padded, visible):
        total_pad = p - v
        before = total_pad // 2
        slices.append(slice(before, before + v))
    return tuple(slices)


def output_shape(
    state: NLCGDialogState,
    data_shape: Tuple[int, ...],
    psf_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Spatial output shape (same ndim as `data_shape`).

    Tiled runs always output the cropped/visible shape (``process_tiles``
    stitches owned cores in visible space). Single-volume runs output the
    visible shape when ``crop_to_visible`` else the full padded/object
    domain.
    """
    ndim = len(data_shape)
    zoom = per_axis_zoom(state, ndim)
    visible = visible_shape(data_shape, zoom)

    if state.tiled or state.crop_to_visible:
        return visible
    return padded_shape(visible, psf_shape)


def output_5d_shape(
    state: NLCGDialogState,
    data_shape: Tuple[int, ...],
    psf_shape: Tuple[int, ...],
    n_channels: int = 1,
) -> Tuple[int, int, int, int, int]:
    """Return the (T, Z, C, Y, X) shape of the deconvolved output."""
    out = output_shape(state, data_shape, psf_shape)
    C = max(1, int(n_channels))
    if len(data_shape) == 2:
        return (1, 1, C, out[0], out[1])
    return (1, out[0], C, out[1], out[2])


def estimate_tile_plan(
    state: NLCGDialogState, data_shape: Tuple[int, ...]
) -> Tuple[int, Tuple[int, ...]]:
    """Rough ``(n_tiles, tile_shape)`` preview, without importing deconlib.

    Mirrors the axis-splitting logic of ``plan_tiles``/``optimal_tile_size``
    closely enough for a dialog preview (tile axes are always the last two;
    Z is only tiled when it exceeds ``min_z_slices``). Doesn't need to be
    pixel-exact — the worker calls the real ``plan_tiles`` at run time.
    """
    ndim = len(data_shape)
    guard = max(int(state.guard_px), 0)
    tile_size = max(int(state.tile_size), 1)

    tile_axes = {ndim - 1, ndim - 2}
    if ndim >= 3 and data_shape[ndim - 3] > int(state.min_z_slices):
        tile_axes.add(ndim - 3)

    n_tiles = 1
    tile_shape = []
    for i, n in enumerate(data_shape):
        if i not in tile_axes or n <= tile_size:
            tile_shape.append(n)
            continue
        n_axis_tiles = -(-n // tile_size)          # ceil(n / tile_size)
        core = -(-n // n_axis_tiles)               # balanced core, <= tile_size
        n_tiles *= n_axis_tiles
        tile_shape.append(min(core + 2 * guard, n))
    return n_tiles, tuple(tile_shape)


# --------------------------------------------------------------------------- #
# PSF preset helper (unchanged behavior from the old decon_recipe module)

def compact_psf_shape_for_data(
    data_shape: Tuple[int, ...],
    factor: Tuple[float, ...],
    *,
    min_shape: Optional[Tuple[int, ...]] = None,
    max_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[int, ...]:
    """Return a compact odd PSF support shape at fine-grid sampling.

    The PSF support is not the detector field and not the object domain. This
    helper chooses a conservative compact default, bounded by the selected
    data region so bead distillation can still use the same preset.
    """
    if len(data_shape) != len(factor):
        raise ValueError("data_shape and factor must have the same ndim")
    default_min = (17,) * len(data_shape)
    default_max = (65,) * len(data_shape)
    min_shape = default_min if min_shape is None else tuple(min_shape)
    max_shape = default_max if max_shape is None else tuple(max_shape)
    if len(min_shape) != len(data_shape) or len(max_shape) != len(data_shape):
        raise ValueError("min_shape/max_shape must match data_shape ndim")

    out = []
    for d, f, n_min, n_max in zip(data_shape, factor, min_shape, max_shape):
        # Roughly one quarter of the detector span on the fine grid, clipped to
        # a compact default support and made odd so there is a clear centre.
        target = int(round(float(d) * max(float(f), 1.0) / 4.0))
        n = max(int(n_min), min(int(n_max), target))
        n = min(n, int(round(float(d) * max(float(f), 1.0))))
        n = max(1, n)
        if n > 1 and n % 2 == 0:
            n -= 1
        out.append(n)
    return tuple(out)


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


# --------------------------------------------------------------------------- #
# Input preparation

def _prepare_psf_kernel(psf_array: np.ndarray) -> np.ndarray:
    """Return a finite, nonnegative PSF for the (strictly positive) NLCG model."""
    psf = np.asarray(psf_array, dtype=np.float32)
    if not np.all(np.isfinite(psf)):
        raise ValueError("PSF contains NaN or Inf values")

    min_value = float(psf.min()) if psf.size else 0.0
    negative_count = int(np.count_nonzero(psf < 0))
    if negative_count:
        log.warning(
            "PSF contains %d negative values (min %.6g); clipping to zero "
            "before deconvolution",
            negative_count, min_value,
        )
        psf = np.maximum(psf, 0.0)

    total = float(np.sum(psf, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("PSF must have positive finite flux after clipping")
    return np.ascontiguousarray(psf, dtype=np.float32)


def _prepare_observation(y_obs: np.ndarray) -> np.ndarray:
    """Validate data and enforce nonnegative support (NLCG is Poisson-only)."""
    y = np.ascontiguousarray(y_obs, dtype=np.float32)
    if not np.all(np.isfinite(y)):
        raise ValueError("Observation contains NaN or Inf values")

    negative_count = int(np.count_nonzero(y < 0))
    if negative_count:
        log.warning(
            "observation contains %d negative values (min %.6g); "
            "clipping to zero for Poisson deconvolution",
            negative_count, float(y.min()),
        )
        y = np.maximum(y, 0.0).astype(np.float32, copy=False)
    return y


def prepare_inputs(
    *,
    state: NLCGDialogState,
    y_obs: np.ndarray,
    psf_array: np.ndarray,
    psf_pixel_size_um: Tuple[float, ...],
) -> PreparedInputs:
    """Build the (y, psf, zoom, voxel_spacing) payload for the NLCG worker.

    `y_obs` is the cropped observation (2D or 3D, ndim matches PSF).
    `psf_array` is at visible-grid sampling (data spacing / zoom), DC-at-corner.
    """
    ndim = y_obs.ndim
    if psf_array.ndim != ndim:
        raise ValueError(
            f"PSF ndim {psf_array.ndim} does not match observation ndim {ndim}"
        )

    zoom = per_axis_zoom(state, ndim)
    psf = _prepare_psf_kernel(psf_array)
    y = _prepare_observation(y_obs)
    voxel_spacing = tuple(float(s) for s in psf_pixel_size_um)

    log.info(
        "prepare_inputs: y=%s psf=%s zoom=%s voxel_spacing=%s "
        "regularizer=%s(weight=%s) tiled=%s",
        tuple(y.shape), tuple(psf.shape), zoom, voxel_spacing,
        state.regularizer_kind, state.reg_weight, state.tiled,
    )

    return PreparedInputs(y=y, psf=psf, zoom=zoom, voxel_spacing=voxel_spacing)
