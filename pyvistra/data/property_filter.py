"""Generic numeric range filtering over columnar property data.

Both :class:`~pyvistra.data.points.PointTable` and
:class:`~pyvistra.data.tracks.TrackTable` carry an arbitrary
``properties: dict[str, np.ndarray]`` bag (e.g. Gaussian-fit results such as
amplitude, background, sigma). This module is the one place that turns a set
of per-column range constraints into a boolean row mask, so that
``PointLayerVisual`` and ``TrackLayerVisual`` share a single filtering
mechanism instead of each hand-rolling their own.

This module is Qt-free and vispy-free, and has no dependency on
``PointTable``/``TrackTable`` — it only needs "a dict of same-length arrays",
so it composes with any future columnar table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PropertyRange:
    """A closed numeric range filter over one property column.

    ``min_value``/``max_value`` of ``None`` means unbounded on that side;
    both ``None`` means the range is inert (matches every row). There is no
    separate "enabled" flag — encoding "off" as "no bound" means a range can
    never be enabled with stale or contradictory bounds.
    """

    name: str
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class PropertyFilterSpec:
    """An ordered set of :class:`PropertyRange` constraints, combined by AND.

    A row must satisfy every active range to pass. At most one range may
    reference a given property name (checked here, once, so every caller
    building a spec doesn't have to dedupe by hand).
    """

    ranges: tuple[PropertyRange, ...] = field(default_factory=tuple)

    def __post_init__(self):
        names = [r.name for r in self.ranges]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate property names in filter spec: {names}")

    def is_empty(self) -> bool:
        return len(self.ranges) == 0


def ranges_from_tuples(value) -> tuple[PropertyRange, ...]:
    """Coerce a widget-friendly ``(name, min, max)`` tuple sequence into
    :class:`PropertyRange` objects.

    ``PropertyFilterWidget.get_filters()`` returns plain, hashable tuples
    rather than dataclasses; this is the one place that boundary conversion
    happens, e.g. in ``PropertyInspectorPanel`` when building a selection.
    """
    return tuple(
        item if isinstance(item, PropertyRange) else PropertyRange(*item)
        for item in value
    )


def compute_mask(
    n_rows: int,
    properties: dict[str, np.ndarray],
    spec: PropertyFilterSpec,
) -> np.ndarray:
    """Boolean mask (len == n_rows), True = row passes every active range.

    Ranges naming a column absent from ``properties`` are ignored (no-op) —
    this can legitimately happen if a layer's data was swapped for a table
    lacking that column, and silently excluding every row (or raising) would
    be worse than ignoring the stale range.

    NaN in a filtered column excludes that row for that range whenever the
    range is active. This isn't a special case: NumPy's ``<``/``>`` already
    evaluate to False against NaN, so an active range simply fails a NaN
    value for free. It's also the safe default — a failed fit (NaN
    amplitude) shouldn't bypass a filter meant to hide low-quality data.
    """
    if spec.is_empty():
        return np.ones(n_rows, dtype=bool)

    mask = np.ones(n_rows, dtype=bool)
    for rng in spec.ranges:
        if rng.min_value is None and rng.max_value is None:
            continue
        arr = properties.get(rng.name)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if rng.min_value is not None:
            mask &= arr >= rng.min_value
        if rng.max_value is not None:
            mask &= arr <= rng.max_value
    return mask


@dataclass(frozen=True)
class SelectionEffect:
    """What happens to rows selected by a :class:`PropertyFilterSpec`.

    A selection on its own is inert — it only becomes visible through an
    effect. ``hidden`` removes selected rows from view; ``color`` overrides
    their rendered color. Either, both, or neither may be active. ``shape``
    is reserved for a future per-row marker override and is unused for now,
    kept here so adding it later doesn't require a breaking dataclass change
    for existing callers.
    """

    hidden: bool = False
    color: tuple[float, float, float, float] | None = None
    shape: object | None = None

    def is_noop(self) -> bool:
        return not self.hidden and self.color is None


def resolve_row_effects(
    n_rows: int,
    properties: dict[str, np.ndarray],
    spec: PropertyFilterSpec,
    effect: SelectionEffect,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resolve a selection + effect into per-row rendering instructions.

    Returns ``(visible, color_override)``:

    - ``visible``: bool mask, len ``n_rows``. Rows outside the selection are
      always visible — the effect only acts on the selected subset, unlike
      :func:`compute_mask` which is itself a visibility predicate.
    - ``color_override``: ``None`` if ``effect.color`` is unset, otherwise an
      ``(n_rows, 4)`` float array with NaN for non-selected rows and
      ``effect.color`` for selected rows.
    """
    selected = compute_mask(n_rows, properties, spec)

    visible = ~selected if effect.hidden else np.ones(n_rows, dtype=bool)

    color_override = None
    if effect.color is not None:
        color_override = np.full((n_rows, 4), np.nan, dtype=np.float64)
        color_override[selected] = effect.color

    return visible, color_override
