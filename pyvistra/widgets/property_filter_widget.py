"""PropertyFilterWidget — reusable numeric range-filter editor.

Embedded by :class:`~pyvistra.widgets.property_inspector_panel.PropertyInspectorPanel`
as the "Selection" section: build a subset of rows by property range, then
apply an effect (hide/highlight) to it. Round-trips to/from the same
plain-tuple shape :func:`~pyvistra.data.property_filter.ranges_from_tuples`
coerces into a :class:`~pyvistra.data.property_filter.PropertyFilterSpec`.

Each row also renders the property's distribution with
:class:`~pyvistra.widgets.histogram.CompactHistogramWidget` — the same
draggable-handle histogram the channel-adjustment panel uses for contrast —
so picking a range is population-level inspection, not blind spinbox entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import colors as tokens
from .histogram import CompactHistogramWidget

_HISTOGRAM_COLOR = tokens.ACCENT


def numeric_property_infos(properties: dict[str, np.ndarray]) -> list["PropertyRangeInfo"]:
    """Numeric property columns as :class:`PropertyRangeInfo` (name, observed
    range, and the raw column for a filter row's histogram).

    Non-numeric columns can't have a min/max range, so they're simply never
    offered as filter/inspection candidates rather than raising or
    misbehaving. Shared by the display-settings dialogs (color-by-property)
    and the Property Inspector dock (selection + table columns).
    """
    infos = []
    for name in sorted(properties):
        arr = np.asarray(properties[name])
        if not np.issubdtype(arr.dtype, np.number):
            continue
        finite = arr[np.isfinite(arr)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        infos.append(PropertyRangeInfo(name, lo, hi, values=arr))
    return infos


@dataclass(frozen=True)
class PropertyRangeInfo:
    """Describes one filterable numeric column for widget construction.

    ``data_min``/``data_max`` seed sane default spinbox bounds/step only —
    a user-typed filter threshold is not clamped to the observed range, so
    filters can be set ahead of new data arriving. ``values`` is the raw
    column (the same array ``compute_mask`` filters); when given, the row
    renders it as a histogram instead of a pair of blind spinboxes. It's
    excluded from equality/repr because NumPy array equality is ambiguous
    for dataclass ``__eq__``.
    """

    name: str
    data_min: float
    data_max: float
    values: np.ndarray | None = field(default=None, repr=False, compare=False)


class _FilterRow(QWidget):
    """One property-name + min/max range row, with a draggable histogram.

    "No Min"/"No Max" encode an unbounded side (see
    :class:`~pyvistra.data.property_filter.PropertyRange`). The histogram
    always needs two finite handle positions to draw, so an unbounded side
    is displayed pinned to the data extreme on that side — dragging that
    handle away from the edge is treated as the user wanting a concrete
    bound, so it clears the corresponding checkbox.
    """

    rangeChanged = Signal()

    def __init__(self, available, initial=None, on_remove=None, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)

        self.property_combo = QComboBox()
        for info in available:
            self.property_combo.addItem(info.name, info)
        top_row.addWidget(self.property_combo, 1)

        remove_btn = QPushButton("✕")
        remove_btn.setFixedWidth(24)
        remove_btn.setToolTip("Remove this property row")
        if on_remove is not None:
            remove_btn.clicked.connect(lambda: on_remove(self))
        top_row.addWidget(remove_btn)
        outer.addLayout(top_row)

        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)

        self.no_min = QCheckBox("No Min")
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1e12, 1e12)
        self.min_spin.setDecimals(4)

        self.histogram = CompactHistogramWidget()

        self.no_max = QCheckBox("No Max")
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1e12, 1e12)
        self.max_spin.setDecimals(4)

        bottom_row.addWidget(self.no_min)
        bottom_row.addWidget(self.min_spin)
        bottom_row.addWidget(self.histogram, 1)
        bottom_row.addWidget(self.max_spin)
        bottom_row.addWidget(self.no_max)
        outer.addLayout(bottom_row)

        self.no_min.toggled.connect(self._on_no_min_toggled)
        self.no_max.toggled.connect(self._on_no_max_toggled)
        self.min_spin.valueChanged.connect(lambda _: self._sync_histogram_clim())
        self.max_spin.valueChanged.connect(lambda _: self._sync_histogram_clim())
        self.histogram.climChanged.connect(self._on_histogram_clim_changed)
        self.property_combo.currentIndexChanged.connect(lambda _: self._on_property_changed())

        self.no_min.toggled.connect(self.rangeChanged)
        self.no_max.toggled.connect(self.rangeChanged)
        self.min_spin.valueChanged.connect(self.rangeChanged)
        self.max_spin.valueChanged.connect(self.rangeChanged)
        self.histogram.climChanged.connect(self.rangeChanged)
        self.property_combo.currentIndexChanged.connect(self.rangeChanged)

        name, min_value, max_value = (
            initial if initial is not None else (available[0].name if available else "", None, None)
        )
        idx = self.property_combo.findText(name)
        self.property_combo.setCurrentIndex(0 if idx < 0 else idx)

        self.no_min.setChecked(min_value is None)
        if min_value is not None:
            self.min_spin.setValue(float(min_value))
        self.no_max.setChecked(max_value is None)
        if max_value is not None:
            self.max_spin.setValue(float(max_value))

        self._on_property_changed()

    def _on_property_changed(self):
        info = self.property_combo.currentData()
        if info is None:
            return
        span = max(float(info.data_max) - float(info.data_min), 1e-6)
        step = span / 100.0
        self.min_spin.setSingleStep(step)
        self.max_spin.setSingleStep(step)
        if info.values is not None:
            self.histogram.set_data(info.values, _HISTOGRAM_COLOR)
        self._sync_histogram_clim()

    def _effective_min(self) -> float:
        info = self.property_combo.currentData()
        if self.no_min.isChecked() and info is not None:
            return float(info.data_min)
        return self.min_spin.value()

    def _effective_max(self) -> float:
        info = self.property_combo.currentData()
        if self.no_max.isChecked() and info is not None:
            return float(info.data_max)
        return self.max_spin.value()

    def _sync_histogram_clim(self):
        self.histogram.blockSignals(True)
        self.histogram.set_clim(self._effective_min(), self._effective_max())
        self.histogram.blockSignals(False)
        self.histogram.update()

    def _on_no_min_toggled(self, checked):
        self.min_spin.setEnabled(not checked)
        self._sync_histogram_clim()

    def _on_no_max_toggled(self, checked):
        self.max_spin.setEnabled(not checked)
        self._sync_histogram_clim()

    def _on_histogram_clim_changed(self, vmin, vmax):
        # Only clear "unbounded" on the side actually being dragged — a
        # center-drag (both handles) clears both, but dragging just the
        # max handle shouldn't silently bound an unrelated unbounded min.
        dragging = self.histogram._dragging
        if dragging in ("min", "center"):
            self.no_min.setChecked(False)
        if dragging in ("max", "center"):
            self.no_max.setChecked(False)
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)

    def get_range(self) -> tuple[str, float | None, float | None]:
        info = self.property_combo.currentData()
        name = info.name if info is not None else self.property_combo.currentText()
        min_value = None if self.no_min.isChecked() else self.min_spin.value()
        max_value = None if self.no_max.isChecked() else self.max_spin.value()
        return (name, min_value, max_value)


class PropertyFilterWidget(QWidget):
    """Editable list of numeric range rows (property combo + histogram +
    min/max) over named properties — doubles as a population-level
    property inspector, since an unbounded row is a no-op filter.

    Gating the histogram behind an explicit "add a filter" action would
    make inspection feel like a filtering commitment it isn't, so when
    there's nothing to restore (no ``initial_filters``) and at least one
    numeric property is available, one unbounded row is seeded up front —
    the distribution is visible the moment the dialog opens, with nothing
    filtered yet. "Add Property" appends further rows, e.g. to inspect or
    filter a second column at the same time.
    """

    changed = Signal()

    def __init__(
        self,
        available_properties: list[PropertyRangeInfo],
        initial_filters=(),
        parent=None,
    ):
        super().__init__(parent)
        self._available = list(available_properties)
        self._rows: list[_FilterRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._rows_layout = QVBoxLayout()
        layout.addLayout(self._rows_layout)

        self._add_button = QPushButton("Add Property")
        self._add_button.setEnabled(bool(self._available))
        # clicked() emits a bool (the button's checked state); a direct
        # connection would pass it through as _add_row's `initial` arg,
        # overriding its None default. Discard it explicitly.
        self._add_button.clicked.connect(lambda: self._add_row())
        layout.addWidget(self._add_button)

        self.set_filters(initial_filters)

    def _add_row(self, initial=None):
        if not self._available:
            return
        if initial is None:
            used = {row.get_range()[0] for row in self._rows}
            unused = [info.name for info in self._available if info.name not in used]
            default_name = unused[0] if unused else self._available[0].name
            initial = (default_name, None, None)
        row = _FilterRow(self._available, initial=initial, on_remove=self._remove_row)
        row.rangeChanged.connect(self.changed)
        self._rows.append(row)
        self._rows_layout.addWidget(row)
        self.changed.emit()

    def _remove_row(self, row: _FilterRow):
        self._rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self.changed.emit()

    def get_filters(self) -> tuple[tuple[str, float | None, float | None], ...]:
        # Keyed by name so a widget in any state always returns at most one
        # range per property, matching PropertyFilterSpec's invariant.
        ranges: dict[str, tuple[str, float | None, float | None]] = {}
        for row in self._rows:
            ranges[row.get_range()[0]] = row.get_range()
        return tuple(ranges.values())

    def set_filters(self, filters) -> None:
        for row in list(self._rows):
            self._remove_row(row)
        filters = list(filters)
        if not filters and self._available:
            # Nothing saved to restore: seed one unbounded (no-op) row so
            # the histogram is visible for inspection immediately, without
            # requiring an "Add Property" click first.
            filters = [(self._available[0].name, None, None)]
        for f in filters:
            self._add_row(initial=tuple(f))
