from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyvistra.widgets.decon_recipe import DecondialogState, prepare_inputs


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
