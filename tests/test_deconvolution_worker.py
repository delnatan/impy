from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyvistra.widgets.decon_recipe import PreparedInputs
from pyvistra.widgets.deconvolution_worker import (
    MemDeconvolutionWorker,
    _build_blur_op_from_recipe,
    _raise_if_nonfinite,
    _write_to_buffer,
)


@dataclass
class _FakeOperatorFactoryArgs:
    psf: object
    optics: object
    geometry: object
    recipe: object
    likelihood: str


def _prepared_inputs():
    return PreparedInputs(
        recipe=object(),
        psf=SimpleNamespace(psf=np.zeros((34, 269, 269), dtype=np.float32)),
        optics=object(),
        geometry=SimpleNamespace(
            hidden_shape=(34, 269, 269),
            visible_shape=(34, 269, 269),
        ),
        y=np.zeros((34, 128, 128), dtype=np.float32),
        prior=np.ones((34, 269, 269), dtype=np.float32),
        sigma=None,
        output_slices=(slice(0, 34), slice(32, 237), slice(32, 237)),
    )


def test_mem_preview_writes_tensor_visible_array_before_cropping():
    prepared = _prepared_inputs()
    buffer = np.zeros((1, 34, 1, 205, 205), dtype=np.float32)
    worker = MemDeconvolutionWorker(
        prepared=prepared,
        map_config=None,
        icf_sweep=None,
        posterior=None,
        buffer=buffer,
    )

    preview = np.arange(
        np.prod(prepared.geometry.visible_shape), dtype=np.float32
    ).reshape(prepared.geometry.visible_shape)
    worker._write_preview(preview, preview_space="visible")

    expected = preview[prepared.output_slices]
    np.testing.assert_array_equal(buffer[0, :, 0, :, :], expected)


def test_mem_preview_rejects_flattened_visible_array():
    prepared = _prepared_inputs()
    buffer = np.zeros((1, 34, 1, 205, 205), dtype=np.float32)
    worker = MemDeconvolutionWorker(
        prepared=prepared,
        map_config=None,
        icf_sweep=None,
        posterior=None,
        buffer=buffer,
    )

    bad = np.zeros(np.prod(prepared.geometry.visible_shape), dtype=np.float32)
    try:
        worker._write_preview(bad, preview_space="visible")
    except ValueError as exc:
        assert "MEM preview shape mismatch" in str(exc)
    else:
        raise AssertionError("expected ValueError for flattened preview payload")


def test_write_to_buffer_targets_requested_channel():
    buffer = np.zeros((1, 2, 3, 4, 5), dtype=np.float32)
    arr = np.ones((2, 4, 5), dtype=np.float32)

    _write_to_buffer(buffer, arr, channel=2)

    np.testing.assert_array_equal(buffer[0, :, 2, :, :], arr)
    assert not np.any(buffer[0, :, 0, :, :])
    assert not np.any(buffer[0, :, 1, :, :])


def test_raise_if_nonfinite_reports_solver_divergence():
    arr = np.array([1.0, np.nan, np.inf], dtype=np.float32)

    try:
        _raise_if_nonfinite(arr, "Richardson-Lucy result")
    except ValueError as exc:
        assert "1 NaN and 1 Inf" in str(exc)
        assert "solver diverged" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-finite result")


def test_build_blur_op_from_recipe_uses_public_registry():
    blur_op = object()
    seen = {}

    def builder(args):
        seen["args"] = args
        return {"blur_op": blur_op}

    fake_deconlib = SimpleNamespace(
        OperatorFactoryArgs=_FakeOperatorFactoryArgs,
        RECIPE_REGISTRY={"example": builder},
    )
    recipe = SimpleNamespace(kind="example")
    psf = object()
    optics = object()
    geometry = object()

    with patch.dict("sys.modules", {"deconlib": fake_deconlib}):
        result = _build_blur_op_from_recipe(
            recipe,
            psf=psf,
            optics=optics,
            geometry=geometry,
            likelihood="poisson",
        )

    assert result is blur_op
    assert seen["args"].recipe is recipe
    assert seen["args"].psf is psf
    assert seen["args"].optics is optics
    assert seen["args"].geometry is geometry
    assert seen["args"].likelihood == "poisson"


def test_build_blur_op_from_recipe_rejects_missing_blur_op():
    fake_deconlib = SimpleNamespace(
        OperatorFactoryArgs=_FakeOperatorFactoryArgs,
        RECIPE_REGISTRY={"example": lambda _args: {}},
    )

    with patch.dict("sys.modules", {"deconlib": fake_deconlib}):
        try:
            _build_blur_op_from_recipe(
                SimpleNamespace(kind="example"),
                psf=object(),
                optics=object(),
                geometry=object(),
                likelihood="poisson",
            )
        except ValueError as exc:
            assert "does not expose an MLX blur_op" in str(exc)
        else:
            raise AssertionError("expected ValueError for recipe without blur_op")
