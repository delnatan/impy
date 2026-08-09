import numpy as np
import pytest

from pyvistra.data.property_filter import (
    PropertyFilterSpec,
    PropertyRange,
    SelectionEffect,
    compute_mask,
    resolve_row_effects,
)


def test_empty_spec_returns_all_true():
    mask = compute_mask(5, {}, PropertyFilterSpec())
    assert mask.dtype == bool
    assert mask.tolist() == [True] * 5

    # Non-empty properties dict, still an empty spec: still all True.
    mask = compute_mask(3, {"amplitude": np.array([1.0, 2.0, 3.0])}, PropertyFilterSpec())
    assert mask.tolist() == [True, True, True]


def test_single_range_filters_in_and_out_of_bounds():
    properties = {"amplitude": np.array([0.0, 5.0, 10.0, 15.0, 20.0])}
    spec = PropertyFilterSpec(ranges=(PropertyRange("amplitude", min_value=5.0, max_value=15.0),))

    mask = compute_mask(5, properties, spec)

    assert mask.tolist() == [False, True, True, True, False]


def test_multiple_ranges_and_combine():
    properties = {
        "amplitude": np.array([10.0, 10.0, 100.0, 100.0]),
        "background": np.array([1.0, 50.0, 1.0, 50.0]),
    }
    spec = PropertyFilterSpec(
        ranges=(
            PropertyRange("amplitude", min_value=50.0),
            PropertyRange("background", max_value=10.0),
        )
    )

    mask = compute_mask(4, properties, spec)

    # Only the row with high amplitude AND low background passes.
    assert mask.tolist() == [False, False, True, False]


def test_nan_excludes_row_when_range_active():
    properties = {"amplitude": np.array([1.0, np.nan, 3.0])}
    spec = PropertyFilterSpec(ranges=(PropertyRange("amplitude", min_value=0.0, max_value=10.0),))

    mask = compute_mask(3, properties, spec)

    assert mask.tolist() == [True, False, True]


def test_nan_irrelevant_when_range_inactive():
    properties = {"amplitude": np.array([1.0, np.nan, 3.0])}
    # No range references "amplitude" at all.
    spec = PropertyFilterSpec(ranges=())

    mask = compute_mask(3, properties, spec)

    assert mask.tolist() == [True, True, True]


def test_missing_column_is_noop():
    properties = {"amplitude": np.array([1.0, 2.0, 3.0])}
    spec = PropertyFilterSpec(ranges=(PropertyRange("sigma", min_value=0.0, max_value=1.0),))

    mask = compute_mask(3, properties, spec)

    assert mask.tolist() == [True, True, True]


def test_all_nan_column_all_excluded_when_active():
    properties = {"amplitude": np.array([np.nan, np.nan, np.nan])}
    spec = PropertyFilterSpec(ranges=(PropertyRange("amplitude", min_value=0.0),))

    mask = compute_mask(3, properties, spec)

    assert mask.tolist() == [False, False, False]


def test_empty_table():
    properties = {"amplitude": np.array([], dtype=np.float64)}
    spec = PropertyFilterSpec(ranges=(PropertyRange("amplitude", min_value=0.0, max_value=1.0),))

    mask = compute_mask(0, properties, spec)

    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_unbounded_one_side():
    properties = {"amplitude": np.array([1.0, 5.0, 10.0])}

    lower_only = PropertyFilterSpec(ranges=(PropertyRange("amplitude", min_value=5.0),))
    assert compute_mask(3, properties, lower_only).tolist() == [False, True, True]

    upper_only = PropertyFilterSpec(ranges=(PropertyRange("amplitude", max_value=5.0),))
    assert compute_mask(3, properties, upper_only).tolist() == [True, True, False]


def test_duplicate_name_rejected():
    with pytest.raises(ValueError):
        PropertyFilterSpec(
            ranges=(
                PropertyRange("amplitude", min_value=0.0),
                PropertyRange("amplitude", max_value=10.0),
            )
        )


def test_resolve_row_effects_empty_spec_is_noop():
    properties = {"amplitude": np.array([1.0, 2.0, 3.0])}
    visible, color_override = resolve_row_effects(
        3, properties, PropertyFilterSpec(), SelectionEffect()
    )

    assert visible.tolist() == [True, True, True]
    assert color_override is None


def test_resolve_row_effects_hidden_hides_selected_rows():
    properties = {"amplitude": np.array([0.0, 5.0, 10.0, 15.0])}
    spec = PropertyFilterSpec(ranges=(PropertyRange("amplitude", min_value=5.0, max_value=10.0),))
    effect = SelectionEffect(hidden=True)

    visible, color_override = resolve_row_effects(4, properties, spec, effect)

    # Selected rows (amplitude in [5, 10]) are hidden; the rest stay visible.
    assert visible.tolist() == [True, False, False, True]
    assert color_override is None


def test_resolve_row_effects_color_overrides_selected_rows_only():
    properties = {"amplitude": np.array([0.0, 5.0, 10.0, 15.0])}
    spec = PropertyFilterSpec(ranges=(PropertyRange("amplitude", min_value=5.0, max_value=10.0),))
    effect = SelectionEffect(color=(1.0, 0.0, 0.0, 1.0))

    visible, color_override = resolve_row_effects(4, properties, spec, effect)

    assert visible.tolist() == [True, True, True, True]
    assert color_override is not None
    assert np.isnan(color_override[0]).all()
    assert color_override[1].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert color_override[2].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert np.isnan(color_override[3]).all()


def test_resolve_row_effects_hidden_and_color_together():
    properties = {"amplitude": np.array([0.0, 5.0, 10.0])}
    spec = PropertyFilterSpec(ranges=(PropertyRange("amplitude", min_value=5.0),))
    effect = SelectionEffect(hidden=True, color=(0.0, 1.0, 0.0, 1.0))

    visible, color_override = resolve_row_effects(3, properties, spec, effect)

    assert visible.tolist() == [True, False, False]
    assert color_override[1].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert color_override[2].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert np.isnan(color_override[0]).all()


def test_selection_effect_is_noop():
    assert SelectionEffect().is_noop()
    assert not SelectionEffect(hidden=True).is_noop()
    assert not SelectionEffect(color=(1.0, 1.0, 1.0, 1.0)).is_noop()
