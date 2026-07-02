import numpy as np

from pyvistra.widgets.decon_nlcg import (
    NLCGDialogState,
    compact_psf_shape_for_data,
    output_5d_shape,
    output_shape,
    padded_shape,
    per_axis_zoom,
    prepare_inputs,
    valid_slices,
    visible_shape,
)


def test_per_axis_zoom_broadcasts_and_floors_at_one():
    state = NLCGDialogState(zoom_xy=2.0, zoom_z=0.5)
    assert per_axis_zoom(state, ndim=3) == (1.0, 2.0, 2.0)
    assert per_axis_zoom(state, ndim=2) == (2.0, 2.0)


def test_visible_shape_scales_by_zoom():
    assert visible_shape((4, 4), (2.0, 2.0)) == (8, 8)
    assert visible_shape((4, 4), (1.0, 1.0)) == (4, 4)


def test_padded_shape_adds_psf_margin():
    assert padded_shape((8, 8), (3, 3)) == (10, 10)
    assert padded_shape((8, 8), (1, 1)) == (8, 8)


def test_valid_slices_crops_symmetric_margin():
    slices = valid_slices((10, 10), (8, 8))
    assert slices == (slice(1, 9), slice(1, 9))


def test_output_shape_single_volume_full_vs_cropped():
    full_state = NLCGDialogState(zoom_xy=2.0, zoom_z=1.0, crop_to_visible=False)
    cropped_state = NLCGDialogState(zoom_xy=2.0, zoom_z=1.0, crop_to_visible=True)

    assert output_shape(full_state, (4, 4), (3, 3)) == (10, 10)
    assert output_shape(cropped_state, (4, 4), (3, 3)) == (8, 8)


def test_output_shape_tiled_always_cropped():
    state = NLCGDialogState(zoom_xy=2.0, zoom_z=1.0, crop_to_visible=False, tiled=True)
    assert output_shape(state, (4, 4), (3, 3)) == (8, 8)


def test_output_5d_shape_preserves_channel_count():
    state = NLCGDialogState(zoom_xy=2.0, zoom_z=1.0, crop_to_visible=True)
    assert output_5d_shape(state, (5, 6), (3, 3), n_channels=3) == (1, 1, 3, 10, 12)
    assert output_5d_shape(state, (4, 5, 6), (1, 3, 3), n_channels=3) == (1, 4, 3, 10, 12)


def test_compact_psf_shape_is_not_roi_or_object_canvas():
    shape = compact_psf_shape_for_data(
        (56, 176, 166),
        (1.0, 1.2, 1.2),
    )

    assert shape == (17, 53, 49)
    assert shape != (56, 211, 199)
    assert shape != (112, 518, 506)


def test_prepare_inputs_clips_negative_psf_and_keeps_values():
    state = NLCGDialogState(zoom_xy=1.0, zoom_z=1.0)
    y_obs = np.ones((2, 2), dtype=np.float32)
    psf_array = np.array([[2.0, -1.0], [0.0, 2.0]], dtype=np.float32)

    prepared = prepare_inputs(
        state=state,
        y_obs=y_obs,
        psf_array=psf_array,
        psf_pixel_size_um=(0.1, 0.1),
    )

    assert np.all(prepared.psf >= 0.0)
    np.testing.assert_allclose(prepared.psf[0, 1], 0.0)
    np.testing.assert_allclose(prepared.psf[0, 0], 2.0)


def test_prepare_inputs_rejects_zero_flux_psf():
    state = NLCGDialogState(zoom_xy=1.0, zoom_z=1.0)

    try:
        prepare_inputs(
            state=state,
            y_obs=np.ones((2, 2), dtype=np.float32),
            psf_array=np.zeros((2, 2), dtype=np.float32),
            psf_pixel_size_um=(0.1, 0.1),
        )
    except ValueError as exc:
        assert "positive finite flux" in str(exc)
    else:
        raise AssertionError("expected ValueError for zero-flux PSF")


def test_prepare_inputs_clips_negative_poisson_observation():
    state = NLCGDialogState(zoom_xy=1.0, zoom_z=1.0)
    y_obs = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)

    prepared = prepare_inputs(
        state=state,
        y_obs=y_obs,
        psf_array=np.ones((2, 2), dtype=np.float32),
        psf_pixel_size_um=(0.1, 0.1),
    )

    assert np.all(prepared.y >= 0.0)
    np.testing.assert_allclose(prepared.y[0, 1], 0.0)


def test_prepare_inputs_rejects_ndim_mismatch():
    state = NLCGDialogState()
    try:
        prepare_inputs(
            state=state,
            y_obs=np.ones((4, 4), dtype=np.float32),
            psf_array=np.ones((4, 4, 4), dtype=np.float32),
            psf_pixel_size_um=(0.1, 0.1, 0.1),
        )
    except ValueError as exc:
        assert "does not match observation ndim" in str(exc)
    else:
        raise AssertionError("expected ValueError for PSF/observation ndim mismatch")
