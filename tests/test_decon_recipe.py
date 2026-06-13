from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyvistra.widgets.decon_recipe import (
    DecondialogState,
    build_recipe,
    output_5d_shape,
    prepare_inputs,
)


@dataclass
class _FakeBundleGeometry:
    hidden_shape: tuple[int, ...]
    visible_shape: tuple[int, ...]
    data_shape: tuple[int, ...]
    voxel_spacing: tuple[float, ...]


@dataclass
class _FakePsf:
    psf: np.ndarray
    optics: object
    pixel_size: tuple[float, ...]


@dataclass
class _FakeForwardRecipe:
    kind: str
    super_res_factor: tuple[int, ...] = ()
    detector_padding: tuple[int, ...] = ()
    psf_source: str = "embedded"
    icf: dict | None = None


def _fake_deconlib_module():
    return SimpleNamespace(
        BundleGeometry=_FakeBundleGeometry,
        ForwardRecipe=_FakeForwardRecipe,
        Psf=_FakePsf,
    )


def test_prepare_inputs_preserves_total_intensity_in_super_res_prior():
    state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=2.0,
        super_res_z=1.0,
        pad_xy=0,
        pad_z=0,
    )
    y_obs = np.full((4, 4), 5.0, dtype=np.float32)
    psf_array = np.zeros((8, 8), dtype=np.float32)
    optics = object()

    with patch.dict("sys.modules", {"deconlib": _fake_deconlib_module()}):
        prepared = prepare_inputs(
            state=state,
            y_obs=y_obs,
            psf_array=psf_array,
            psf_pixel_size_um=(0.05, 0.05),
            optics=optics,
        )

    assert prepared.prior.shape == (8, 8)
    np.testing.assert_allclose(prepared.prior.sum(dtype=np.float64), y_obs.sum(dtype=np.float64))
    np.testing.assert_allclose(float(prepared.prior.mean()), float(y_obs.mean()) / 4.0)


def test_prepare_inputs_keeps_mean_when_hidden_shape_matches_data():
    state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=1.0,
        super_res_z=1.0,
        pad_xy=0,
        pad_z=0,
    )
    y_obs = np.full((4, 4), 5.0, dtype=np.float32)
    psf_array = np.zeros((4, 4), dtype=np.float32)
    optics = object()

    with patch.dict("sys.modules", {"deconlib": _fake_deconlib_module()}):
        prepared = prepare_inputs(
            state=state,
            y_obs=y_obs,
            psf_array=psf_array,
            psf_pixel_size_um=(0.1, 0.1),
            optics=optics,
        )

    np.testing.assert_allclose(float(prepared.prior.mean()), float(y_obs.mean()))
    np.testing.assert_allclose(prepared.prior.sum(dtype=np.float64), y_obs.sum(dtype=np.float64))


def test_output_5d_shape_can_preserve_source_channel_count():
    state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=2.0,
        super_res_z=1.0,
        pad_xy=0,
        pad_z=0,
    )

    assert output_5d_shape(state, (5, 6), n_channels=3) == (1, 1, 3, 10, 12)
    assert output_5d_shape(state, (4, 5, 6), n_channels=3) == (1, 4, 3, 10, 12)


def test_padding_only_recipe_uses_fft_conv_finite_detector():
    state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=1.0,
        super_res_z=1.0,
        pad_xy=3,
        pad_z=0,
    )

    with patch.dict("sys.modules", {"deconlib": _fake_deconlib_module()}):
        recipe = build_recipe(state, ndim=2)

    assert recipe.kind == "fft_conv"
    assert recipe.super_res_factor == ()
    assert recipe.detector_padding == (3, 3)


def test_fractional_super_res_fails_before_geometry_mismatch():
    state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=1.2,
        super_res_z=1.0,
        pad_xy=16,
        pad_z=4,
    )

    with patch.dict("sys.modules", {"deconlib": _fake_deconlib_module()}):
        try:
            build_recipe(state, ndim=3)
        except ValueError as exc:
            assert "whole numbers" in str(exc)
        else:
            raise AssertionError("expected fractional super-res to be rejected")


def test_prepare_inputs_preserves_compact_psf_on_padded_domain():
    state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=1.0,
        super_res_z=1.0,
        pad_xy=2,
        pad_z=0,
    )
    y_obs = np.full((4, 4), 5.0, dtype=np.float32)
    psf_array = np.zeros((3, 3), dtype=np.float32)
    optics = object()

    with patch.dict("sys.modules", {"deconlib": _fake_deconlib_module()}):
        prepared = prepare_inputs(
            state=state,
            y_obs=y_obs,
            psf_array=psf_array,
            psf_pixel_size_um=(0.1, 0.1),
            optics=optics,
        )

    assert prepared.recipe.kind == "fft_conv"
    assert prepared.geometry.hidden_shape == (8, 8)
    assert prepared.psf.psf.shape == (3, 3)
