"""Dialog-state → deconlib recipe / config translation.

`DecondialogState` is the flat-bag dataclass the deconvolution dialog
reads its widgets into. The `build_*` helpers turn it into the deconlib
types that `run_deconvolution_workflow` / `run_richardson_lucy` consume.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple

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
    # Non-integer factors are implemented through deconlib's fractional
    # integrated-detector operator; the shape is derived from data + margin.
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

    Integer super-resolution uses deconlib's built-in ``super_res_idc``
    recipe. Non-integer factors use pyvistra's registered
    ``fractional_idc`` recipe, which derives detector integration from
    geometry instead of from an integer ``super_res_factor``.
    """
    from deconlib import ForwardRecipe

    factor_tuple = _per_axis_factor(state, ndim)
    pad_tuple = _per_axis_pad(state, ndim)

    is_super_res = any(abs(f - 1.0) > 1e-6 for f in factor_tuple)
    has_pad = any(p > 0 for p in pad_tuple)
    if is_super_res:
        is_integral = all(abs(f - round(f)) <= 1e-6 for f in factor_tuple)
        if is_integral:
            kind = "super_res_idc"
            srf: Tuple[int, ...] = tuple(int(round(f)) for f in factor_tuple)
        else:
            _ensure_fractional_idc_recipe_registered()
            kind = "fractional_idc"
            srf = ()
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


_FRACTIONAL_IDC_REGISTERED = False


def _ensure_fractional_idc_recipe_registered() -> None:
    """Register pyvistra's geometry-driven fractional IDC recipe once.

    deconlib's built-in ``super_res_idc`` persists integer
    ``super_res_factor`` values and reconstructs the detector domain by
    exact division. This recipe uses the same low-level operator but takes
    ``visible_shape`` and the low-resolution finite-detector padding from
    ``BundleGeometry``/``ForwardRecipe`` instead, allowing ratios such as
    1.25 or 1.5.
    """
    global _FRACTIONAL_IDC_REGISTERED
    if _FRACTIONAL_IDC_REGISTERED:
        return

    try:
        from deconlib import register_recipe
    except ImportError:
        # Unit tests may patch in a tiny fake deconlib module containing
        # only ForwardRecipe. In the real workflow, prepare/run imports
        # deconlib and this registration succeeds before the recipe is used.
        return

    @register_recipe("fractional_idc")
    def _build_fractional_idc(args):
        from deconlib.deconvolution.composition import as_numpy_op, compose
        from deconlib.deconvolution.linops_mlx import (
            FiniteDetector,
            IntegratedDetectorConvolver,
        )

        visible_shape = tuple(args.geometry.visible_shape)
        data_shape = tuple(args.geometry.data_shape)
        recipe_padding = tuple(args.recipe.detector_padding or ())
        if recipe_padding:
            detector_domain_shape = tuple(
                int(d) + 2 * int(p) for d, p in zip(data_shape, recipe_padding)
            )
        else:
            detector_domain_shape = data_shape

        detector_padding = _detector_padding_from_geometry(
            data_shape=data_shape,
            detector_domain_shape=detector_domain_shape,
            recipe_padding=recipe_padding,
        )
        psf_arr = _resolve_psf_array(args)
        idc = IntegratedDetectorConvolver(
            kernel=psf_arr,
            output_shape=detector_domain_shape,
            normalize=True,
            signal_shape=visible_shape,
        )

        if any(before or after for before, after in detector_padding):
            detector = FiniteDetector(
                detector_shape=data_shape,
                padding=detector_padding,
            )
            blur_op = compose(detector, idc)
        else:
            blur_op = idc
        R, Rt = as_numpy_op(blur_op)

        icf_ops = _make_icf_ops(
            args.recipe.icf,
            visible_shape,
            tuple(args.geometry.voxel_spacing),
        )
        _validate_icf_hidden_shape(args.geometry.hidden_shape, icf_ops, visible_shape)
        return {"R": R, "Rt": Rt, "blur_op": blur_op, **icf_ops}

    _FRACTIONAL_IDC_REGISTERED = True


def _normalize_detector_padding(
    padding: Tuple[Any, ...],
    ndim: int,
) -> Tuple[Tuple[int, int], ...]:
    if not padding:
        return tuple((0, 0) for _ in range(ndim))
    if len(padding) != ndim:
        raise ValueError("detector_padding length must match data_shape ndim")
    pairs = []
    for item in padding:
        if isinstance(item, (tuple, list, np.ndarray)):
            raise ValueError("detector_padding entries must be symmetric integers")
        pad_i = int(item)
        if pad_i < 0:
            raise ValueError("detector_padding values must be non-negative")
        pairs.append((pad_i, pad_i))
    return tuple(pairs)


def _detector_padding_from_geometry(
    *,
    data_shape: Tuple[int, ...],
    detector_domain_shape: Tuple[int, ...],
    recipe_padding: Tuple[Any, ...],
) -> Tuple[Tuple[int, int], ...]:
    if len(data_shape) != len(detector_domain_shape):
        raise ValueError("data_shape and detector domain shape must have same ndim")
    if any(v < d for d, v in zip(data_shape, detector_domain_shape)):
        raise ValueError("detector domain shape cannot be smaller than data_shape")

    explicit = _normalize_detector_padding(recipe_padding, len(data_shape))
    if any(before or after for before, after in explicit):
        expected = tuple(
            int(d) + before + after
            for d, (before, after) in zip(data_shape, explicit)
        )
        if expected != detector_domain_shape:
            raise ValueError(
                "detector_padding implies detector domain shape "
                f"{expected}, but geometry implies {detector_domain_shape}"
            )
        return explicit

    inferred = []
    for d, v in zip(data_shape, detector_domain_shape):
        extra = int(v) - int(d)
        before = extra // 2
        inferred.append((before, extra - before))
    return tuple(inferred)


def _resolve_psf_array(args) -> np.ndarray:
    if args.psf is None:
        raise ValueError(
            f"recipe.kind={args.recipe.kind!r} requires an embedded PSF"
        )
    return np.asarray(args.psf.psf, dtype=np.float32)


def _make_icf_ops(
    spec: Optional[dict],
    shape: Tuple[int, ...],
    spacings: Tuple[float, ...],
) -> dict[str, Any]:
    if spec is None:
        return {"C": None, "Ct": None}
    kind = spec.get("kind")
    if kind == "gaussian":
        sigmas = tuple(float(s) for s in spec["sigmas_um"])
        if len(sigmas) != len(shape):
            raise ValueError(
                f"ICF sigmas_um has {len(sigmas)} entries; expected {len(shape)}"
            )
        if len(spacings) != len(shape):
            raise ValueError(
                f"voxel_spacing has {len(spacings)} entries; expected {len(shape)}"
            )
        from deconlib.deconvolution import GaussianICF, as_numpy_op

        icf_mlx = GaussianICF(
            shape=shape,
            sigmas=sigmas,
            spacings=spacings,
            normalize=True,
        )
        C, Ct = as_numpy_op(icf_mlx)
        return {"C": C, "Ct": Ct}
    if kind == "atrous":
        from deconlib.deconvolution import AtrousTransform, as_numpy_op

        levels = int(spec["levels"])
        kernel = str(spec.get("kernel", "b3spline"))
        axes = spec.get("axes")
        axes_tuple = None if axes is None else tuple(int(a) for a in axes)
        weights = spec.get("weights")
        weights_arr = None if weights is None else np.asarray(weights, dtype=float)
        transform = AtrousTransform(
            levels=levels,
            kernel=kernel,  # type: ignore[arg-type]
            axes=axes_tuple,
            weights=weights_arr,
        )
        C, Ct = as_numpy_op(transform)
        return {
            "C": C,
            "Ct": Ct,
            "entropy": "positive_negative",
            "hidden_shape": transform.hidden_shape(shape),
        }
    raise ValueError(f"unknown ICF kind {kind!r}")


def _validate_icf_hidden_shape(
    hidden_shape: Tuple[int, ...],
    icf_ops: dict[str, Any],
    visible_shape: Tuple[int, ...],
) -> None:
    expected = tuple(icf_ops.get("hidden_shape", visible_shape))
    if tuple(hidden_shape) != expected:
        raise ValueError(
            f"ICF expects hidden_shape {expected}, got {tuple(hidden_shape)}"
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

    is_super_res = (
        abs(float(state.super_res_xy) - 1.0) > 1e-6
        or abs(float(state.super_res_z) - 1.0) > 1e-6
    )
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


def compact_psf_shape_for_data(
    data_shape: Tuple[int, ...],
    factor: Tuple[float, ...],
    *,
    min_shape: Tuple[int, ...] | None = None,
    max_shape: Tuple[int, ...] | None = None,
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


def psf_support_radius(
    psf_array: np.ndarray,
    *,
    dc_corner: bool = True,
    energy_fraction: float = 0.995,
) -> Tuple[int, ...]:
    """Estimate per-axis positive PSF support radius in PSF pixels."""
    psf = np.asarray(psf_array, dtype=np.float32)
    if psf.ndim == 0:
        return ()
    if not np.all(np.isfinite(psf)):
        raise ValueError("PSF contains NaN or Inf values")
    h = np.maximum(psf, 0.0)
    total = float(np.sum(h, dtype=np.float64))
    if total <= 0.0:
        return tuple(0 for _ in range(psf.ndim))

    frac = min(max(float(energy_fraction), 0.0), 1.0)
    radii: list[int] = []
    for axis, n in enumerate(psf.shape):
        reduce_axes = tuple(i for i in range(psf.ndim) if i != axis)
        marginal = np.sum(h, axis=reduce_axes, dtype=np.float64)
        idx = np.arange(n)
        if dc_corner:
            dist = np.minimum(idx, n - idx)
        else:
            dist = np.abs(idx - (n // 2))
        order = np.argsort(dist, kind="stable")
        cumulative = np.cumsum(marginal[order])
        threshold = frac * total
        hit = int(np.searchsorted(cumulative, threshold, side="left"))
        hit = min(hit, len(order) - 1)
        radii.append(int(dist[order[hit]]))
    return tuple(radii)


def object_margin_from_psf_support(
    psf_radius: Tuple[int, ...],
    factor: Tuple[float, ...],
    data_shape: Tuple[int, ...],
    *,
    max_fraction: float = 0.10,
    max_margin: int = 32,
) -> Tuple[int, ...]:
    """Convert fine-grid PSF radii to bounded detector-pixel object margins."""
    if not (len(psf_radius) == len(factor) == len(data_shape)):
        raise ValueError("psf_radius, factor, and data_shape must have same ndim")
    out = []
    for r, f, d in zip(psf_radius, factor, data_shape):
        raw = int(np.ceil(max(int(r), 0) / max(float(f), 1.0)))
        cap = max(0, min(int(max_margin), int(np.floor(max_fraction * int(d)))))
        out.append(min(raw, cap))
    return tuple(out)


def linear_convolution_shape(
    signal_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Minimum canvas shape for non-circular convolution.

    This is a computational canvas: it is large enough for a signal of
    ``signal_shape`` and a PSF/kernel of ``kernel_shape`` to convolve
    without wrap-around. It does not add unknown object pixels; the
    modeled object/visible domain is still ``signal_shape``.
    """
    if len(signal_shape) != len(kernel_shape):
        raise ValueError(
            "signal_shape and kernel_shape must have the same dimensionality"
        )
    return tuple(
        max(1, int(n)) + max(1, int(m)) - 1
        for n, m in zip(signal_shape, kernel_shape)
    )


def _prepare_psf_kernel(psf_array: np.ndarray) -> np.ndarray:
    """Return a finite, nonnegative, unit-flux PSF for positive solvers."""
    psf = np.asarray(psf_array, dtype=np.float32)
    if not np.all(np.isfinite(psf)):
        raise ValueError("PSF contains NaN or Inf values")

    min_value = float(psf.min()) if psf.size else 0.0
    negative_count = int(np.count_nonzero(psf < 0))
    if negative_count:
        log.warning(
            "PSF contains %d negative values (min %.6g); clipping to zero "
            "before normalization for positive deconvolution",
            negative_count, min_value,
        )
        psf = np.maximum(psf, 0.0)

    total = float(np.sum(psf, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("PSF must have positive finite flux after clipping")
    return np.ascontiguousarray(psf / total, dtype=np.float32)


def _prepare_observation(state: DecondialogState, y_obs: np.ndarray) -> np.ndarray:
    """Validate data and enforce nonnegative support for Poisson solvers."""
    y = np.ascontiguousarray(y_obs, dtype=np.float32)
    if not np.all(np.isfinite(y)):
        raise ValueError("Observation contains NaN or Inf values")

    uses_poisson = (
        state.algorithm == "richardson_lucy"
        or state.likelihood == "poisson"
    )
    if uses_poisson:
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
    psf_padded = _prepare_psf_kernel(psf_array)

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

    y = _prepare_observation(state, y_obs)
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
    detector field on the fine grid, so the output buffer must be sized
    to that crop, not to ``hidden_shape``.
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
        slices = valid_slices_for_state(state, data_shape)
        if slices is None:
            out = tuple(data_shape)
        else:
            out = tuple(int(s.stop) - int(s.start) for s in slices)
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
