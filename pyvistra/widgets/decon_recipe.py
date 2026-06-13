"""Dialog-state → deconlib recipe / config translation.

`DecondialogState` is the flat-bag dataclass the deconvolution dialog
reads its widgets into. The `build_*` helpers turn it into the deconlib
types that `run_deconvolution_workflow` / `run_richardson_lucy` consume.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Single shared logger for the deconvolution stack. Attach a stderr
# handler on first import so the messages show up in the console
# without anything extra; users can silence with
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
class DecondialogState:
    algorithm: str                          # "memsolve_mem" | "richardson_lucy"

    # Forward model — independent lateral (XY) and axial (Z) controls.
    # deconlib's recipe registry currently derives detector domains from
    # integer super-res factors, so the dialog exposes whole-number factors.
    super_res_xy: float = 1.0
    super_res_z: float = 1.0
    pad_xy: int = 0                         # symmetric per lateral axis
    pad_z: int = 0                          # symmetric on Z

    # ICF (MEM only)
    icf_mode: str = "off"                   # "off" | "fixed" | "sweep" | "wavelet"
    icf_sigma_um: Optional[float] = None    # used when icf_mode == "fixed"
    icf_sweep_sigmas_um: Tuple[float, ...] = ()
    icf_refine: bool = True
    wavelet_levels: int = 3
    wavelet_kernel: str = "b3spline"        # "b3spline" | "triangle"
    wavelet_prior_scale: float = 5.0
    wavelet_allow_poisson: bool = False

    # Likelihood (MEM)
    likelihood: str = "poisson"             # "poisson" | "gaussian"
    sigma_gaussian: Optional[float] = None

    # MEM solver knobs
    max_iter: int = 60
    tol_omega: float = 0.05
    rate: float = 0.3
    omega_mode: str = "auto"                # "auto" | "classic"
    cg_epsilon: float = 1e-2
    cg_max_steps: int = 30
    n_probe_g: int = 1
    map_space: str = "hidden"               # "hidden" | "data"

    # MEM posterior
    posterior_n_samples: int = 0
    posterior_seed: int = 0

    # RL knobs
    rl_num_iter: int = 50
    rl_background: float = 0.0
    rl_eval_interval: int = 10

    # Output region (applies to both MEM and RL). "full" keeps the entire
    # hidden-grid reconstruction (including the detector-pad border);
    # "valid" crops to the measured-detector field on the fine grid.
    return_region: str = "full"             # "full" | "valid"


# --------------------------------------------------------------------------- #
# Builders

def build_recipe(state: DecondialogState, ndim: int):
    """Translate `state` into a `deconlib.ForwardRecipe`.

    The recipe's `super_res_factor` is informational (the actual hidden
    shape lives in the geometry); we round per-axis floats to ints for
    serialization. The recipe's `detector_padding` IS consumed by the
    builder, so it's already integer.
    """
    from deconlib import ForwardRecipe

    factor_tuple = _per_axis_factor(state, ndim)
    pad_tuple = _per_axis_pad(state, ndim)
    _validate_integral_super_res(factor_tuple)

    is_super_res = any(abs(f - 1.0) > 1e-6 for f in factor_tuple)
    has_pad = any(p > 0 for p in pad_tuple)
    if is_super_res:
        kind = "super_res_idc"
        srf: Tuple[int, ...] = tuple(int(round(f)) for f in factor_tuple)
        pad: Tuple[int, ...] = pad_tuple
    else:
        kind = "fft_conv"
        srf = ()
        pad = pad_tuple if has_pad else ()

    icf = None
    if (state.algorithm == "memsolve_mem"
            and state.icf_mode == "fixed"
            and state.icf_sigma_um is not None):
        icf = {
            "kind": "gaussian",
            "sigmas_um": (float(state.icf_sigma_um),) * ndim,
        }

    return ForwardRecipe(
        kind=kind,
        super_res_factor=srf,
        detector_padding=pad,
        psf_source="embedded",
        icf=icf,
    )


def build_mem_config(state: DecondialogState):
    """Build a `mem.MaxEntConfig` from the dialog state."""
    import mem

    return mem.MaxEntConfig(
        max_iter=state.max_iter,
        tol_omega=state.tol_omega,
        rate=state.rate,
        omega_mode=state.omega_mode,
        cg_epsilon=state.cg_epsilon,
        cg_max_steps=state.cg_max_steps,
        n_probe_g=state.n_probe_g,
    )


def build_rl_config(state: DecondialogState):
    """Build a `RichardsonLucyConfig` from the dialog state.

    ``return_region="valid"`` is only meaningful when the reconstruction
    extends beyond the detector field. For plain ``fft_conv`` we silently
    fall back to ``"full"`` so the dialog's shared toggle is harmless.
    """
    from deconlib import RichardsonLucyConfig

    factor_tuple = _per_axis_factor(state, ndim=2)  # axis count irrelevant here
    is_super_res = any(abs(f - 1.0) > 1e-6 for f in factor_tuple)
    pad_any = state.pad_xy > 0 or state.pad_z > 0
    # "valid" cropping is meaningful whenever there is something to crop —
    # super-res, detector padding, or both. fft_conv (no sr, no pad) is the
    # only case where the result is already the data field.
    region = state.return_region if (is_super_res or pad_any) else "full"
    return RichardsonLucyConfig(
        num_iter=state.rl_num_iter,
        background=state.rl_background,
        eval_interval=state.rl_eval_interval,
        return_region=region,
    )


def build_icf_sweep(state: DecondialogState):
    """Return an `IcfSweep` when `state` requests one, else None."""
    if state.icf_mode != "sweep" or not state.icf_sweep_sigmas_um:
        return None
    from deconlib import IcfSweep
    return IcfSweep(
        sigmas_um=tuple(float(s) for s in state.icf_sweep_sigmas_um),
        refine=bool(state.icf_refine),
    )


def build_posterior(state: DecondialogState):
    """Return a `mem.PosteriorConfig` when sampling is requested, else None."""
    if state.posterior_n_samples <= 0:
        return None
    import mem
    return mem.PosteriorConfig(
        n_samples=int(state.posterior_n_samples),
        seed=int(state.posterior_seed),
    )


def build_wavelet_config(state: DecondialogState):
    """Return a `WaveletMemConfig` when wavelet MEM is requested, else None."""
    if state.icf_mode != "wavelet":
        return None
    from deconlib import WaveletMemConfig
    return WaveletMemConfig(
        levels=max(1, int(state.wavelet_levels)),
        kernel=str(state.wavelet_kernel),
        prior_scale=max(float(state.wavelet_prior_scale), 1e-6),
        allow_poisson=bool(state.wavelet_allow_poisson),
    )


# --------------------------------------------------------------------------- #
# Geometry / PSF preparation

@dataclass(frozen=True)
class PreparedInputs:
    """The numeric inputs `run_*` workflow drivers consume.

    Geometry matches deconlib's recipe-builder contracts. PSFs may be compact
    corner-origin kernels; deconlib embeds them into the computed FFT canvas.

    ``output_slices`` is set when the dialog requested ``return_region
    == "valid"`` and the hidden-grid reconstruction needs cropping
    back to the detector field. MEM workers apply it before writing,
    while RL gets the same crop performed inside deconlib via the
    config.
    """
    recipe: object              # ForwardRecipe
    psf: object                 # deconlib.Psf
    optics: object              # deconlib.Optics
    geometry: object            # BundleGeometry
    y: np.ndarray
    prior: np.ndarray
    sigma: Optional[np.ndarray]
    output_slices: Optional[Tuple[slice, ...]] = None


def prepare_inputs(
    *,
    state: DecondialogState,
    y_obs: np.ndarray,
    psf_array: np.ndarray,
    psf_pixel_size_um: Tuple[float, ...],
    optics,
    require_psf_match: bool = True,
) -> PreparedInputs:
    """Build recipe + Psf + Geometry + y/prior/sigma.

    `y_obs` is the cropped observation (2D or 3D, ndim matches PSF).
    `psf_array` is at the hidden-grid sampling (i.e. data spacing divided
    by the chosen lateral / axial super-res factor), DC-at-corner.
    `psf_pixel_size_um` matches `psf_array.ndim`.

    When `require_psf_match` is True, the PSF's shape must match the computed
    hidden-grid shape. The dialog disables that legacy guard so compact PSFs
    can use deconlib's linear-convolution padding path.
    """
    from deconlib import BundleGeometry, Psf

    ndim = y_obs.ndim
    if psf_array.ndim != ndim:
        raise ValueError(
            f"PSF ndim {psf_array.ndim} does not match observation ndim {ndim}"
        )

    data_shape = tuple(y_obs.shape)
    recipe = build_recipe(state, ndim)

    factor_tuple = _per_axis_factor(state, ndim)
    pad_tuple = _per_axis_pad(state, ndim)
    hidden_shape = tuple(
        int(round((d + 2 * p) * f))
        for d, p, f in zip(data_shape, pad_tuple, factor_tuple)
    )
    visible_shape = hidden_shape

    log.info(
        "prepare_inputs: algorithm=%s y=%s psf=%s "
        "factor_xyz=%s pad_xyz=%s recipe.kind=%s "
        "→ hidden=%s visible=%s data=%s",
        state.algorithm, tuple(y_obs.shape), tuple(psf_array.shape),
        factor_tuple, pad_tuple, recipe.kind,
        hidden_shape, visible_shape, data_shape,
    )
    if require_psf_match and any(p > 0 for p in pad_tuple):
        require_psf_match = False
    target_psf_shape = hidden_shape
    if require_psf_match and tuple(psf_array.shape) != target_psf_shape:
        log.error(
            "PSF/hidden-shape mismatch — supplied %s, need %s",
            tuple(psf_array.shape), target_psf_shape,
        )
        raise ValueError(
            f"PSF shape {tuple(psf_array.shape)} does not match the expected "
            f"hidden-grid shape {target_psf_shape}. Re-compute the PSF for "
            "the current super-res factors and detector padding (use the "
            "“Compute PSF…” button)."
        )
    psf_padded = np.asarray(psf_array, dtype=np.float32)

    # The PSF is supplied at hidden-grid sampling (this is what the
    # "Compute PSF…" preset produces, dividing the data spacing by the
    # super-res factor). `voxel_spacing` is therefore just the PSF's
    # spacing — no further factor adjustment.
    voxel_spacing = tuple(float(s) for s in psf_pixel_size_um)

    geometry = BundleGeometry(
        hidden_shape=hidden_shape,
        visible_shape=visible_shape,
        data_shape=data_shape,
        voxel_spacing=voxel_spacing,
    )

    psf = Psf(
        psf=psf_padded,
        optics=optics,
        pixel_size=voxel_spacing,
    )

    y = np.ascontiguousarray(y_obs, dtype=np.float32)
    # Keep the default model flat, but preserve the observed total
    # intensity when the hidden grid has more pixels due to super-res
    # and/or detector padding.
    prior_value = max(float(np.sum(y)) / max(float(np.prod(hidden_shape)), 1.0), 1e-3)
    prior = np.full(hidden_shape, prior_value, dtype=np.float32)
    sigma = None
    if state.likelihood == "gaussian" and state.sigma_gaussian is not None:
        sigma = np.full(y.shape, float(state.sigma_gaussian), dtype=np.float32)

    log.info(
        "  spacing=%s likelihood=%s sigma=%s prior_mean=%s prior_sum=%s "
        "icf=%s super_res_factor(recipe)=%s detector_padding(recipe)=%s",
        voxel_spacing, state.likelihood,
        None if sigma is None else float(state.sigma_gaussian),
        float(prior_value), float(prior.sum(dtype=np.float64)),
        recipe.icf, recipe.super_res_factor, recipe.detector_padding,
    )
    slices = valid_slices_for_state(state, data_shape)
    keep_full = (slices is None
                 or all(s == slice(0, h) for s, h in zip(slices, hidden_shape)))
    output_slices = None if (state.return_region == "full" or keep_full) else slices

    return PreparedInputs(
        recipe=recipe, psf=psf, optics=optics, geometry=geometry,
        y=y, prior=prior, sigma=sigma, output_slices=output_slices,
    )


def valid_slices_for_state(
    state: DecondialogState, data_shape: Tuple[int, ...]
) -> Optional[Tuple[slice, ...]]:
    """Hidden-grid slices that crop a MAP/restored result to the detector field.

    Mirrors deconlib's ``_finite_detector_valid_slices`` so that MEM and
    RL crop the same way. Returns ``None`` when no recipe-level
    super-res is active (fft_conv keeps hidden == data, no crop needed).
    """
    ndim = len(data_shape)
    factor_tuple = _per_axis_factor(state, ndim)
    pad_tuple = _per_axis_pad(state, ndim)
    is_super_res = any(abs(f - 1.0) > 1e-6 for f in factor_tuple)
    has_pad = any(p > 0 for p in pad_tuple)
    if not (is_super_res or has_pad):
        return None

    slices: list = []
    for d, p, f in zip(data_shape, pad_tuple, factor_tuple):
        padded = d + 2 * p
        hidden = int(round(padded * f))
        scale = hidden / padded if padded else 1.0
        start = max(0, min(hidden, int(round(p * scale))))
        stop = max(start, min(hidden, int(round((p + d) * scale))))
        slices.append(slice(start, stop))
    return tuple(slices)


def expected_hidden_shape(state: DecondialogState, data_shape: Tuple[int, ...]
                          ) -> Tuple[int, ...]:
    """Return the hidden-grid shape implied by `state` for `data_shape`."""
    ndim = len(data_shape)
    factor_tuple = _per_axis_factor(state, ndim)
    pad_tuple = _per_axis_pad(state, ndim)
    is_super_res = any(abs(f - 1.0) > 1e-6 for f in factor_tuple)
    has_pad = any(p > 0 for p in pad_tuple)
    if not (is_super_res or has_pad):
        return tuple(data_shape)
    return tuple(
        int(round((d + 2 * p) * f))
        for d, p, f in zip(data_shape, pad_tuple, factor_tuple)
    )


def output_5d_shape(state: DecondialogState, data_shape: Tuple[int, ...], n_channels: int = 1
                    ) -> Tuple[int, int, int, int, int]:
    """Return the (T, Z, C, Y, X) shape of the deconvolved output.

    When ``state.return_region == "valid"`` and the reconstruction
    extends beyond the detector field, the output is cropped to the
    detector field on the fine grid — i.e. ``round(data * factor)`` per
    axis — so the output buffer must be sized to that crop, not to
    ``hidden_shape``.
    Applies to both MEM (cropped manually in the worker) and RL
    (cropped by deconlib via ``RichardsonLucyConfig.return_region``).
    """
    ndim = len(data_shape)
    factor_tuple = _per_axis_factor(state, ndim)
    pad_tuple = _per_axis_pad(state, ndim)
    is_super_res = any(abs(f - 1.0) > 1e-6 for f in factor_tuple)
    has_pad = any(p > 0 for p in pad_tuple)

    if not (is_super_res or has_pad):
        out = tuple(data_shape)
    elif state.return_region == "valid":
        out = tuple(int(round(d * f)) for d, f in zip(data_shape, factor_tuple))
    else:
        out = expected_hidden_shape(state, data_shape)

    C = max(1, int(n_channels))
    if ndim == 2:
        return (1, 1, C, out[0], out[1])
    return (1, out[0], C, out[1], out[2])


# --------------------------------------------------------------------------- #
# Internal helpers

def _per_axis_factor(state: DecondialogState, ndim: int) -> Tuple[float, ...]:
    """Per-axis super-res factor: `(fz, fy, fx)` (3D) or `(fy, fx)` (2D)."""
    fxy = max(float(state.super_res_xy), 1.0)
    fz = max(float(state.super_res_z), 1.0)
    if ndim == 3:
        return (fz, fxy, fxy)
    return (fxy, fxy)


def _per_axis_pad(state: DecondialogState, ndim: int) -> Tuple[int, ...]:
    """Per-axis detector pad: `(pz, py, px)` (3D) or `(py, px)` (2D)."""
    pxy = max(int(state.pad_xy), 0)
    pz = max(int(state.pad_z), 0)
    if ndim == 3:
        return (pz, pxy, pxy)
    return (pxy, pxy)


def _validate_integral_super_res(factors: Tuple[float, ...]) -> None:
    if any(abs(float(f) - round(float(f))) > 1e-6 for f in factors):
        pretty = " × ".join(f"{float(f):g}" for f in factors)
        raise ValueError(
            "Super-res factors must be whole numbers with the current "
            f"deconlib recipe builder (got {pretty}). Choose 1, 2, 3, ... "
            "or disable super-res."
        )


def _fit_psf_corner(psf: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape a DC-at-corner PSF to `target_shape`.

    Per-axis: zero-pad if smaller, crop from the corner if larger. The
    `prepare_inputs` shape gate normally ensures these are equal; this
    helper exists so any residual disagreement degrades gracefully
    instead of throwing a confusing broadcast error.
    """
    if tuple(psf.shape) == tuple(target_shape):
        return psf
    out = np.zeros(target_shape, dtype=psf.dtype)
    src_slices = tuple(slice(0, min(s, t)) for s, t in zip(psf.shape, target_shape))
    dst_slices = tuple(slice(0, min(s, t)) for s, t in zip(psf.shape, target_shape))
    out[dst_slices] = psf[src_slices]
    return out
