"""Shared, algorithm-agnostic deconvolution plumbing.

Shape math (`visible_shape`, `padded_shape`, `valid_slices`, ...),
input validation/preparation (`prepare_inputs`), and buffer-write /
diagnostics helpers used by every deconvolution dialog (NLCG,
Richardson-Lucy, MaxEnt/memsolve) live here so the three engine-specific
modules (`decon_nlcg.py`, `decon_rl.py`, `decon_mem.py`) only need to add
their own solver-knob dataclass and worker.

None of this depends on any particular dialog state's algorithm-specific
fields (regularizer, MEM knobs, ...) -- every function here only reads
the handful of fields every engine's dialog state shares: `zoom_xy`,
`zoom_z`, `crop_to_visible`, `tiled`, `tile_size`, `guard_px`,
`min_z_slices`.

This module is a dependency-free (no `deconlib`/`mlx`) mirror of
deconlib's own `deconlib.deconvolution.shapes` conventions, so dialog
preview/output-shape computation works without those (optional, heavy)
dependencies installed -- they're only imported lazily, at run time, by
the workers.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Tuple

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


@dataclass(frozen=True)
class PreparedInputs:
    """Numeric payload a deconvolution worker hands to deconlib."""

    y: np.ndarray                           # cropped observation, data space
    psf: np.ndarray                         # positive, corner-origin, visible-grid spacing
    zoom: Tuple[float, ...]                 # per-axis, matches y.ndim
    voxel_spacing: Tuple[float, ...]        # visible-grid spacing (regularizer r)


# --------------------------------------------------------------------------- #
# Per-axis helpers

def per_axis_zoom(state, ndim: int) -> Tuple[float, ...]:
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
    state,
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
    state,
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
    state, data_shape: Tuple[int, ...]
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
# PSF preset helper

# Fixed lateral (Y, X) PSF support default, in pixels. PSF tails grow more
# prominent at high defocus, and a flat default covers the typical range of
# pixel sizes without needing to scale with the selected detector region.
_LATERAL_PSF_DEFAULT_PX = 128


def compact_psf_shape_for_data(
    data_shape: Tuple[int, ...],
    factor: Tuple[float, ...],
) -> Tuple[int, ...]:
    """Return a default PSF support shape at fine-grid sampling.

    The PSF support is not the detector field and not the object domain --
    this only picks a sane default (editable afterward in the Compute PSF
    dialog). Axis treatment is not uniform:

    * Axial (Z, present only for a 3D `data_shape`, at index 0) matches the
      (zoomed) data's actual depth exactly. A PSF axially narrower than the
      reconstructed volume clips its own tails, producing blocky,
      discontinuous artifacts along Z.
    * Lateral (Y, X) uses the fixed `_LATERAL_PSF_DEFAULT_PX` default
      regardless of the data/ROI size or zoom.
    """
    if len(data_shape) != len(factor):
        raise ValueError("data_shape and factor must have the same ndim")

    is_3d = len(data_shape) == 3
    out = []
    for i, (d, f) in enumerate(zip(data_shape, factor)):
        if is_3d and i == 0:
            out.append(max(1, int(round(float(d) * max(float(f), 1.0)))))
        else:
            out.append(_LATERAL_PSF_DEFAULT_PX)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Input preparation

def _prepare_psf_kernel(psf_array: np.ndarray) -> np.ndarray:
    """Return a finite, nonnegative PSF for a (strictly positive) solver."""
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
    """Validate data and enforce nonnegative support (Poisson-only solvers)."""
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
    state,
    y_obs: np.ndarray,
    psf_array: np.ndarray,
    psf_pixel_size_um: Tuple[float, ...],
) -> PreparedInputs:
    """Build the (y, psf, zoom, voxel_spacing) payload a worker hands to deconlib.

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
        "prepare_inputs: y=%s psf=%s zoom=%s voxel_spacing=%s tiled=%s",
        tuple(y.shape), tuple(psf.shape), zoom, voxel_spacing, state.tiled,
    )

    return PreparedInputs(y=y, psf=psf, zoom=zoom, voxel_spacing=voxel_spacing)


# --------------------------------------------------------------------------- #
# Buffer / diagnostics helpers (shared by every worker)

def mean_poisson_i_divergence(
    observed: np.ndarray, model: np.ndarray, eps: float = 1e-6
) -> float:
    """Mean Poisson I-divergence I(data || model), for verbose status text.

    Mirrors ``deconlib.deconvolution.rl_mlx.poisson_i_divergence`` (a small,
    stable numpy formula, kept local rather than importing an internal
    module path).
    """
    obs = np.asarray(observed, dtype=np.float64)
    mod = np.maximum(np.asarray(model, dtype=np.float64), eps)
    obs_safe = np.maximum(obs, eps)
    return float(np.mean(obs * np.log(obs_safe / mod) - obs + mod))


def raise_if_nonfinite(arr: np.ndarray, label: str) -> None:
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count or inf_count:
        raise ValueError(
            f"{label} contains {nan_count} NaN and {inf_count} Inf values; "
            "the solver diverged before producing a usable result"
        )


def write_to_buffer(
    buffer, arr: np.ndarray, *, channel: int = 0, log_stats: bool = True
) -> None:
    channel = max(0, int(channel))
    if channel >= buffer.shape[2]:
        raise ValueError(
            f"output channel {channel} is outside buffer channel count {buffer.shape[2]}"
        )

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count or inf_count:
        log.error(
            "result contains %d NaN and %d Inf values out of %d — "
            "the solver diverged. Common causes: PSF / data spacing "
            "mismatch, negative pixel values in the data, or a PSF that "
            "is not at the same fine-grid sampling the model expects.",
            nan_count, inf_count, arr.size,
        )
    elif log_stats:
        log.info("result stats: min=%.6g max=%.6g mean=%.6g sum=%.6g",
                 float(arr.min()), float(arr.max()),
                 float(arr.mean()), float(arr.sum()))

    if arr.ndim == 2:
        target = buffer.shape[-2:]
        if arr.shape != target:
            log.error("buffer/result shape mismatch: result=%s buffer=%s",
                      arr.shape, buffer.shape)
        buffer[0, 0, channel, :, :] = arr
    elif arr.ndim == 3:
        target = (buffer.shape[1], buffer.shape[3], buffer.shape[4])
        if arr.shape != target:
            log.error("buffer/result shape mismatch: result=%s buffer=%s",
                      arr.shape, buffer.shape)
        buffer[0, :, channel, :, :] = arr
    else:
        raise ValueError(f"unexpected result ndim {arr.ndim}")
