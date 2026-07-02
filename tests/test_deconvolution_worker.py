import numpy as np
import pytest

from pyvistra.widgets.deconvolution_worker import (
    _mean_poisson_i_divergence,
    _raise_if_nonfinite,
    _write_to_buffer,
)


def test_mean_poisson_i_divergence_is_zero_for_perfect_model():
    data = np.array([1.0, 5.0, 10.0], dtype=np.float32)
    assert _mean_poisson_i_divergence(data, data) == pytest.approx(0.0, abs=1e-6)


def test_mean_poisson_i_divergence_is_positive_for_mismatched_model():
    data = np.array([1.0, 5.0, 10.0], dtype=np.float32)
    model = np.array([2.0, 5.0, 10.0], dtype=np.float32)
    assert _mean_poisson_i_divergence(data, model) > 0.0


def test_write_to_buffer_targets_requested_channel():
    buffer = np.zeros((1, 2, 3, 4, 5), dtype=np.float32)
    arr = np.ones((2, 4, 5), dtype=np.float32)

    _write_to_buffer(buffer, arr, channel=2)

    np.testing.assert_array_equal(buffer[0, :, 2, :, :], arr)
    assert not np.any(buffer[0, :, 0, :, :])
    assert not np.any(buffer[0, :, 1, :, :])


def test_write_to_buffer_handles_2d_result():
    buffer = np.zeros((1, 1, 2, 4, 5), dtype=np.float32)
    arr = np.full((4, 5), 3.0, dtype=np.float32)

    _write_to_buffer(buffer, arr, channel=1)

    np.testing.assert_array_equal(buffer[0, 0, 1, :, :], arr)


def test_raise_if_nonfinite_reports_solver_divergence():
    arr = np.array([1.0, np.nan, np.inf], dtype=np.float32)

    try:
        _raise_if_nonfinite(arr, "NLCG result")
    except ValueError as exc:
        assert "1 NaN and 1 Inf" in str(exc)
        assert "solver diverged" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-finite result")
