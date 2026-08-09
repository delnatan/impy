"""PropertyInspectorPanel — dockable table + histogram property inspector.

This is the dedicated, non-modal "application" for exploring a points/tracks
layer's properties: a read-only table of every row plus a selection editor
(reusing :class:`~pyvistra.widgets.property_filter_widget.PropertyFilterWidget`)
that builds a :class:`~pyvistra.data.property_filter.PropertyFilterSpec` from
property ranges. An effect (hide selected rows, and/or give them a highlight
color) is applied immediately to the layer's visual as the selection changes
— no OK button, mirroring the live-update pattern in
:class:`~pyvistra.widgets.channel_panel.ChannelPanel`.

This is intentionally separate from the display-settings dialogs
(``point_display_settings_dialog.py``/``track_display_settings_dialog.py``):
those are pure visual styling, this is data exploration. Selection state
here is transient — it lives only on the visual for as long as the dock is
open, and is not persisted into ``layer.style`` or project save/load.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..data.property_filter import PropertyFilterSpec, SelectionEffect, ranges_from_tuples
from .color_button import ColorButton
from .dock_utils import make_dock_float_resize_resilient
from .property_filter_widget import PropertyFilterWidget, numeric_property_infos


class _PropertyTableModel(QAbstractTableModel):
    """Read-only table over one layer's rows: an id column, base columns
    (t/x/y/z for points; n_frames for tracks), then one column per property.
    """

    def __init__(
        self,
        ids: np.ndarray,
        base_columns: dict[str, np.ndarray],
        properties: dict[str, np.ndarray],
        parent=None,
    ):
        super().__init__(parent)
        self._columns: list[tuple[str, np.ndarray]] = [("id", np.asarray(ids))]
        for name, arr in base_columns.items():
            self._columns.append((name, np.asarray(arr)))
        for name in sorted(properties):
            self._columns.append((name, np.asarray(properties[name])))
        self._n_rows = len(ids)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._n_rows

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        _name, arr = self._columns[index.column()]
        value = arr[index.row()]
        if isinstance(value, (np.floating, float)):
            return f"{float(value):.4g}"
        if hasattr(value, "item"):
            value = value.item()
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._columns[section][0]
        return str(section + 1)


def _rows_from_points(table) -> _PropertyTableModel:
    base = {"t": table.t, "x": table.x, "y": table.y}
    if table.z is not None:
        base["z"] = table.z
    return _PropertyTableModel(table.point_id, base, table.properties)


def _rows_from_tracks(table) -> _PropertyTableModel:
    track_ids = table.track_ids
    n_frames = np.array(
        [sl.stop - sl.start for _, sl in table.iter_track_slices()], dtype=np.int64
    )
    aggregated = {
        name: table.aggregate_property(arr) for name, arr in table.properties.items()
    }
    return _PropertyTableModel(track_ids, {"n_frames": n_frames}, aggregated)


def _selection_properties(layer_type: str, table) -> dict[str, np.ndarray]:
    """Properties dict in the shape the layer's selection spec is evaluated
    against — raw per-row for points, per-track aggregated for tracks."""
    if layer_type == "tracks":
        return {name: table.aggregate_property(arr) for name, arr in table.properties.items()}
    return dict(table.properties)


class PropertyInspectorPanel(QWidget):
    """Table + selection + effect controls for one points/tracks layer."""

    def __init__(self, window, layer, parent=None):
        super().__init__(parent)
        self.window = window
        self.layer_name = layer.name
        self.layer_type = layer.layer_type

        table = layer.data.table if layer.layer_type == "points" else layer.data
        available_properties = numeric_property_infos(_selection_properties(layer.layer_type, table))
        model = _rows_from_points(table) if layer.layer_type == "points" else _rows_from_tracks(table)

        layout = QVBoxLayout(self)

        self.table_view = QTableView()
        self.table_view.setModel(model)
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table_view, 1)

        layout.addWidget(QLabel("Selection"))
        self.filter_widget = PropertyFilterWidget(available_properties, ())
        layout.addWidget(self.filter_widget)

        effect_row = QHBoxLayout()
        self.hide_checkbox = QCheckBox("Hide selected")
        self.highlight_checkbox = QCheckBox("Highlight selected")
        self.color_button = ColorButton((1.0, 0.85, 0.0, 1.0), title="Highlight Color")
        self.color_button.setEnabled(False)
        effect_row.addWidget(self.hide_checkbox)
        effect_row.addWidget(self.highlight_checkbox)
        effect_row.addWidget(self.color_button)
        effect_row.addStretch(1)
        layout.addLayout(effect_row)

        self.filter_widget.changed.connect(self._apply_selection)
        self.hide_checkbox.toggled.connect(self._apply_selection)
        self.highlight_checkbox.toggled.connect(self._on_highlight_toggled)
        self.color_button.clicked.connect(self._apply_selection)

        self.window.layer_removed.connect(self._on_layer_removed)

    def _on_highlight_toggled(self, checked: bool) -> None:
        self.color_button.setEnabled(checked)
        self._apply_selection()

    def _current_layer(self):
        try:
            return self.window.layers[self.layer_name]
        except KeyError:
            return None

    def _apply_selection(self, *_args) -> None:
        layer = self._current_layer()
        if layer is None or layer.visual is None:
            return
        spec = PropertyFilterSpec(ranges=ranges_from_tuples(self.filter_widget.get_filters()))
        effect = SelectionEffect(
            hidden=self.hide_checkbox.isChecked(),
            color=self.color_button.rgba if self.highlight_checkbox.isChecked() else None,
        )
        layer.visual.set_property_selection(spec, effect)
        canvas = getattr(self.window, "canvas", None)
        if canvas is not None:
            canvas.update()

    def _on_layer_removed(self, removed_layer) -> None:
        if getattr(removed_layer, "name", None) != self.layer_name:
            return
        docks = getattr(self.window, "_property_inspector_docks", None)
        if docks is not None:
            docks.pop(self.layer_name, None)
        dock = self.parent()
        if isinstance(dock, QDockWidget):
            dock.close()


def show_property_inspector_dock(window, layer) -> None:
    """Show (creating on first use) a Property Inspector dock for *layer*.

    One dock per (window, layer) pair, tracked in
    ``window._property_inspector_docks``. Re-invoking for the same layer
    raises the existing dock rather than creating a duplicate.
    """
    docks = getattr(window, "_property_inspector_docks", None)
    if docks is None:
        docks = {}
        window._property_inspector_docks = docks

    dock = docks.get(layer.name)
    if dock is None:
        panel = PropertyInspectorPanel(window, layer)
        dock = QDockWidget(f"Inspect: {layer.name}", window)
        dock.setObjectName(f"property_inspector_dock_{layer.name}")
        dock.setWidget(panel)
        window.addDockWidget(Qt.RightDockWidgetArea, dock)
        make_dock_float_resize_resilient(dock, window)
        docks[layer.name] = dock

    dock.show()
    dock.raise_()
