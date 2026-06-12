from types import SimpleNamespace

import numpy as np

from pyvistra.widgets.decon_recipe import PreparedInputs
from pyvistra.widgets.deconvolution_worker import MemDeconvolutionWorker


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
