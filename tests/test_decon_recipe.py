from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyvistra.widgets.decon_recipe import (
    DecondialogState,
    build_recipe,
    build_rl_config,
    compact_psf_shape_for_data,
    expected_hidden_shape,
    linear_convolution_shape,
    object_margin_from_psf_support,
    output_5d_shape,
    prepare_inputs,
    psf_support_radius,
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


@dataclass
class _FakeRichardsonLucyConfig:
    num_iter: int
    background: float
    eval_interval: int
    return_region: str


def _fake_deconlib_module():
    return SimpleNamespace(
        BundleGeometry=_FakeBundleGeometry,
        ForwardRecipe=_FakeForwardRecipe,
        Psf=_FakePsf,
        RichardsonLucyConfig=_FakeRichardsonLucyConfig,
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
    psf_array = np.ones((8, 8), dtype=np.float32)
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
    psf_array = np.ones((4, 4), dtype=np.float32)
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


def test_fractional_super_res_uses_fractional_idc_recipe():
    state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=1.25,
        super_res_z=1.0,
        pad_xy=16,
        pad_z=4,
    )

    with patch.dict("sys.modules", {"deconlib": _fake_deconlib_module()}):
        recipe = build_recipe(state, ndim=3)

    assert recipe.kind == "fractional_idc"
    assert recipe.super_res_factor == ()
    assert recipe.detector_padding == (4, 16, 16)


def test_prepare_inputs_preserves_compact_psf_on_padded_domain():
    state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=1.0,
        super_res_z=1.0,
        pad_xy=2,
        pad_z=0,
    )
    y_obs = np.full((4, 4), 5.0, dtype=np.float32)
    psf_array = np.ones((3, 3), dtype=np.float32)
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
    np.testing.assert_allclose(float(prepared.psf.psf.sum()), 1.0)


def test_prepare_inputs_clips_negative_psf_and_normalizes_flux():
    state = DecondialogState(
        algorithm="richardson_lucy",
        super_res_xy=1.0,
        super_res_z=1.0,
    )
    y_obs = np.ones((2, 2), dtype=np.float32)
    psf_array = np.array([[2.0, -1.0], [0.0, 2.0]], dtype=np.float32)

    with patch.dict("sys.modules", {"deconlib": _fake_deconlib_module()}):
        prepared = prepare_inputs(
            state=state,
            y_obs=y_obs,
            psf_array=psf_array,
            psf_pixel_size_um=(0.1, 0.1),
            optics=object(),
        )

    assert np.all(prepared.psf.psf >= 0.0)
    np.testing.assert_allclose(float(prepared.psf.psf.sum()), 1.0)
    np.testing.assert_allclose(prepared.psf.psf[0, 1], 0.0)


def test_prepare_inputs_rejects_zero_flux_psf():
    state = DecondialogState(
        algorithm="richardson_lucy",
        super_res_xy=1.0,
        super_res_z=1.0,
    )

    with patch.dict("sys.modules", {"deconlib": _fake_deconlib_module()}):
        try:
            prepare_inputs(
                state=state,
                y_obs=np.ones((2, 2), dtype=np.float32),
                psf_array=np.zeros((2, 2), dtype=np.float32),
                psf_pixel_size_um=(0.1, 0.1),
                optics=object(),
            )
        except ValueError as exc:
            assert "positive finite flux" in str(exc)
        else:
            raise AssertionError("expected ValueError for zero-flux PSF")


def test_prepare_inputs_clips_negative_poisson_observation():
    state = DecondialogState(
        algorithm="richardson_lucy",
        super_res_xy=1.0,
        super_res_z=1.0,
    )
    y_obs = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)

    with patch.dict("sys.modules", {"deconlib": _fake_deconlib_module()}):
        prepared = prepare_inputs(
            state=state,
            y_obs=y_obs,
            psf_array=np.ones((2, 2), dtype=np.float32),
            psf_pixel_size_um=(0.1, 0.1),
            optics=object(),
        )

    assert np.all(prepared.y >= 0.0)
    np.testing.assert_allclose(prepared.y[0, 1], 0.0)


def test_compact_psf_shape_is_not_roi_or_object_canvas():
    shape = compact_psf_shape_for_data(
        (56, 176, 166),
        (1.0, 1.2, 1.2),
    )

    assert shape == (17, 53, 49)
    assert shape != (56, 211, 199)
    assert shape != (112, 518, 506)


def test_psf_support_radius_uses_mass_not_array_half_size():
    yx = np.zeros((256, 256), dtype=np.float32)
    yx[0, 0] = 1.0
    yx[1, 0] = 0.01
    yx[-1, 0] = 0.01

    radius = psf_support_radius(yx, dc_corner=True, energy_fraction=0.99)

    assert radius == (1, 0)
    assert radius != (128, 128)


def test_object_margin_converts_fine_psf_radius_to_bounded_detector_pixels():
    margin = object_margin_from_psf_support(
        (28, 128, 128),
        (1.0, 1.2, 1.2),
        (56, 176, 166),
        max_margin=16,
    )

    assert margin == (5, 16, 16)


def test_object_margin_and_linear_convolution_canvas_are_separate_shapes():
    state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=1.0,
        super_res_z=1.0,
        pad_xy=67,
        pad_z=0,
    )

    object_shape = expected_hidden_shape(state, (128, 128))

    assert object_shape == (262, 262)
    assert linear_convolution_shape(object_shape, (31, 31)) == (292, 292)
    assert linear_convolution_shape((263, 263), (31, 31)) == (293, 293)


def test_valid_output_crops_object_margin_after_super_resolution():
    full_state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=2.0,
        super_res_z=1.0,
        pad_xy=2,
        pad_z=0,
        return_region="full",
    )
    valid_state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=2.0,
        super_res_z=1.0,
        pad_xy=2,
        pad_z=0,
        return_region="valid",
    )

    assert output_5d_shape(full_state, (4, 4)) == (1, 1, 1, 16, 16)
    assert output_5d_shape(valid_state, (4, 4)) == (1, 1, 1, 8, 8)


def test_valid_output_uses_fractional_crop_slice_length():
    state = DecondialogState(
        algorithm="memsolve_mem",
        super_res_xy=1.25,
        super_res_z=1.0,
        pad_xy=1,
        pad_z=0,
        return_region="valid",
    )

    assert output_5d_shape(state, (5, 5)) == (1, 1, 1, 7, 7)


def test_rl_config_honors_z_only_super_resolution_for_valid_region():
    state = DecondialogState(
        algorithm="richardson_lucy",
        super_res_xy=1.0,
        super_res_z=1.5,
        pad_xy=0,
        pad_z=0,
        return_region="valid",
    )

    with patch.dict("sys.modules", {"deconlib": _fake_deconlib_module()}):
        config = build_rl_config(state)

    assert config.return_region == "valid"
