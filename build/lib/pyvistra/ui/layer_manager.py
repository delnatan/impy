"""LayerManager — unified layer panel for all annotation types.

Provides a single tree view showing shapes, points, tracks, and labels
layers from the window's LayerList. Shape layers expand to show one row
per shape with per-shape context-menu operations and a double-click that
opens the numerical properties dialog.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..data.shapes import (
    AddShape,
    RemoveShape,
    SetShapeLabel,
)
from ..data.point_commands import RemovePoint
from .manager import manager


# Type icons for display
_TYPE_LABELS = {
    "shapes": "Shapes",
    "points": "Points",
    "tracks": "Tracks",
    "labels": "Labels",
}

# Roles for tree-item payload
_ROLE_LAYER_NAME = Qt.UserRole
_ROLE_SHAPE_ID = Qt.UserRole + 1
_ROLE_KIND = Qt.UserRole + 2  # "layer" or "shape"


def _shape_summary(rec) -> tuple[str, str]:
    """Return (display label, type/anchor column) for a shape record."""
    label = rec.label or f"Shape {rec.shape_id}"
    from ..data.shapes import ALL_FRAMES
    t_str = "*" if rec.t == ALL_FRAMES else str(rec.t)
    z_str = "*" if rec.z == ALL_FRAMES else str(rec.z)
    return label, f"{rec.shape_type} · t={t_str} z={z_str}"


class LayerManager(QWidget):
    """Unified layer management panel.

    Shows all layers from the active window's LayerList in a tree,
    grouped by type. Shape layers expand to one row per shape.
    """

    _instance: LayerManager | None = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Layer Manager")
        self.setMinimumWidth(280)
        self._current_window = None
        self._connected_windows: set = set()
        # Per-layer-data unsubscribe handles, keyed by id(layer.data).
        self._data_unsubs: dict[int, callable] = {}
        self._setup_ui()
        self._connect_manager()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Window selector label
        self._window_label = QLabel("No window")
        layout.addWidget(self._window_label)

        # Layer tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Layer", "Type"])
        self._tree.setColumnCount(2)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._tree)

        # Buttons
        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add Layer")
        self._add_btn.clicked.connect(self._on_add_layer)
        btn_row.addWidget(self._add_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._on_remove_layer)
        btn_row.addWidget(self._remove_btn)

        self._undo_btn = QPushButton("Undo")
        self._undo_btn.clicked.connect(self._on_undo)
        btn_row.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("Redo")
        self._redo_btn.clicked.connect(self._on_redo)
        btn_row.addWidget(self._redo_btn)

        layout.addLayout(btn_row)

    def _connect_manager(self):
        manager.window_registered.connect(self._on_manager_window_registered)
        manager.window_unregistered.connect(self._on_manager_window_unregistered)
        # Pick up windows that already exist when this panel is first built.
        for win in manager.get_all().values():
            self._connect_window(win)
            if self._current_window is None and self._has_layers(win):
                self._set_active_window(win)

    def _on_manager_window_registered(self, window):
        self._connect_window(window)
        if self._current_window is None and self._has_layers(window):
            self._set_active_window(window)

    def _on_manager_window_unregistered(self, window):
        self._disconnect_window(window)
        if self._current_window is window:
            remaining = next(
                (w for w in manager.get_all().values() if self._has_layers(w)),
                None,
            )
            self._set_active_window(remaining)

    @staticmethod
    def _has_layers(window) -> bool:
        """True only for viewers that own a LayerList (i.e. ImageWindow)."""
        return hasattr(window, "layers") and hasattr(window, "layer_added")

    def _connect_window(self, window):
        if window in self._connected_windows or not hasattr(window, "layer_added"):
            return
        window.window_activated.connect(self._on_window_activated)
        window.layer_added.connect(self._refresh)
        window.layer_removed.connect(self._refresh)
        self._connected_windows.add(window)

    def _disconnect_window(self, window):
        if window not in self._connected_windows:
            return
        try:
            window.window_activated.disconnect(self._on_window_activated)
            window.layer_added.disconnect(self._refresh)
            window.layer_removed.disconnect(self._refresh)
        except (TypeError, RuntimeError):
            pass
        self._connected_windows.discard(window)

    def _on_window_activated(self, window):
        if self._has_layers(window):
            self._set_active_window(window)

    def _set_active_window(self, window):
        if self._current_window is window:
            return
        self._current_window = window
        if window is not None:
            self._window_label.setText(window.windowTitle())
        else:
            self._window_label.setText("No window")
        self._refresh()

    # ------------------------------------------------------------------
    # Data subscriptions (one per shape layer in the active window)
    # ------------------------------------------------------------------
    def _clear_data_subscriptions(self) -> None:
        for unsub in list(self._data_unsubs.values()):
            try:
                unsub()
            except Exception:
                pass
        self._data_unsubs.clear()

    def _ensure_data_subscriptions(self, layers) -> None:
        """Subscribe to every shape and point layer's data. Idempotent."""
        live_ids = set()
        for layer in layers:
            if layer.layer_type not in ("shapes", "points"):
                continue
            data = layer.data
            if not hasattr(data, "subscribe"):
                continue
            key = id(data)
            live_ids.add(key)
            if key in self._data_unsubs:
                continue

            def _cb(event_kind, item_id, _layer=layer):
                self._refresh()

            self._data_unsubs[key] = data.subscribe(_cb)

        # Drop subscriptions for layers no longer present.
        for key in list(self._data_unsubs.keys()):
            if key not in live_ids:
                try:
                    self._data_unsubs[key]()
                except Exception:
                    pass
                del self._data_unsubs[key]

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------
    def _refresh(self, *_args):
        """Rebuild the tree from the current window's LayerList."""
        self._tree.blockSignals(True)
        self._tree.clear()

        if self._current_window is None or not self._has_layers(self._current_window):
            self._clear_data_subscriptions()
            self._tree.blockSignals(False)
            return

        layers = list(self._current_window.layers)
        self._ensure_data_subscriptions(layers)

        for layer in layers:
            item = QTreeWidgetItem()
            item.setText(0, layer.name)
            item.setText(1, _TYPE_LABELS.get(layer.layer_type, layer.layer_type))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if layer.visible else Qt.Unchecked)
            item.setData(0, _ROLE_LAYER_NAME, layer.name)
            item.setData(0, _ROLE_KIND, "layer")
            self._tree.addTopLevelItem(item)

            if layer.layer_type == "shapes":
                self._populate_shape_children(item, layer)
            elif layer.layer_type == "points":
                self._populate_point_children(item, layer)

        self._tree.blockSignals(False)

    def _populate_shape_children(self, parent_item: QTreeWidgetItem, layer) -> None:
        # One child row per shape, ordered by shape_id.
        shape_ids = sorted(layer.data.shape_ids)
        for sid in shape_ids:
            rec = layer.data.get(sid)
            label, summary = _shape_summary(rec)
            child = QTreeWidgetItem()
            child.setText(0, label)
            child.setText(1, summary)
            child.setData(0, _ROLE_LAYER_NAME, layer.name)
            child.setData(0, _ROLE_SHAPE_ID, sid)
            child.setData(0, _ROLE_KIND, "shape")
            if sid in layer.selected_ids:
                child.setSelected(True)
            parent_item.addChild(child)
        parent_item.setExpanded(True)

    def _populate_point_children(self, parent_item: QTreeWidgetItem, layer) -> None:
        # PointDataHolder exposes the current table; sort by point_id.
        tbl = getattr(layer.data, "table", None)
        if tbl is None or tbl.n_rows == 0:
            return
        # Cap the number of rows shown to keep the panel responsive.
        max_rows = 500
        order = sorted(range(tbl.n_rows), key=lambda i: int(tbl.point_id[i]))
        for i, idx in enumerate(order):
            if i >= max_rows:
                more = QTreeWidgetItem()
                more.setText(
                    0,
                    f"… and {tbl.n_rows - max_rows} more points",
                )
                more.setData(0, _ROLE_KIND, "info")
                parent_item.addChild(more)
                break
            pid = int(tbl.point_id[idx])
            x = float(tbl.x[idx]); y = float(tbl.y[idx])
            t = int(tbl.t[idx])
            child = QTreeWidgetItem()
            child.setText(0, f"Point {pid}")
            child.setText(1, f"t={t}  ({x:.1f}, {y:.1f})")
            child.setData(0, _ROLE_LAYER_NAME, layer.name)
            child.setData(0, _ROLE_SHAPE_ID, pid)  # reuse role for point_id
            child.setData(0, _ROLE_KIND, "point")
            parent_item.addChild(child)
        parent_item.setExpanded(False)

    # ------------------------------------------------------------------
    # Tree-item interactions
    # ------------------------------------------------------------------
    def _on_item_changed(self, item, column):
        """Handle checkbox toggle for visibility on layer rows."""
        if self._current_window is None:
            return
        if column != 0:
            return
        if item.data(0, _ROLE_KIND) != "layer":
            return
        name = item.data(0, _ROLE_LAYER_NAME)
        if name is None:
            return
        visible = item.checkState(0) == Qt.Checked
        try:
            layer = self._current_window.layers[name]
            layer.set_visible(visible)
            self._current_window.canvas.update()
        except KeyError:
            pass

    def _resolve_shape_item(self, item):
        """Return (layer, shape_id) for a shape-row item, or (None, None)."""
        if item is None or self._current_window is None:
            return None, None
        if item.data(0, _ROLE_KIND) != "shape":
            return None, None
        name = item.data(0, _ROLE_LAYER_NAME)
        sid = item.data(0, _ROLE_SHAPE_ID)
        try:
            layer = self._current_window.layers[name]
        except KeyError:
            return None, None
        if sid not in layer.data:
            return None, None
        return layer, int(sid)

    def _on_item_clicked(self, item, _column):
        layer, sid = self._resolve_shape_item(item)
        if layer is None:
            return
        # Select just this shape on the canvas.
        if hasattr(self._current_window, "_select_shape"):
            self._current_window._select_shape(layer, sid)
        self._current_window.canvas.update()

    def _on_item_double_clicked(self, item, _column):
        layer, sid = self._resolve_shape_item(item)
        if layer is not None:
            self._open_properties_dialog(layer, sid)
            return
        layer, pid = self._resolve_point_item(item)
        if layer is not None:
            self._open_point_properties_dialog(layer, pid)

    def _resolve_point_item(self, item):
        if item is None or self._current_window is None:
            return None, None
        if item.data(0, _ROLE_KIND) != "point":
            return None, None
        name = item.data(0, _ROLE_LAYER_NAME)
        pid = item.data(0, _ROLE_SHAPE_ID)
        try:
            layer = self._current_window.layers[name]
        except KeyError:
            return None, None
        if layer.layer_type != "points":
            return None, None
        return layer, int(pid)

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------
    def _on_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if item is None:
            return
        kind = item.data(0, _ROLE_KIND)
        global_pos = self._tree.mapToGlobal(pos)
        if kind == "layer":
            self._show_layer_menu(item, global_pos)
        elif kind == "shape":
            self._show_shape_menu(item, global_pos)
        elif kind == "point":
            self._show_point_menu(item, global_pos)

    def _show_point_menu(self, item, global_pos) -> None:
        layer, pid = self._resolve_point_item(item)
        if layer is None:
            return
        menu = QMenu(self)
        act_props = menu.addAction("Edit Properties…")
        act_intensity = menu.addAction("Intensity & Profiles…")
        menu.addSeparator()
        act_delete = menu.addAction("Delete")
        chosen = menu.exec_(global_pos)
        if chosen is act_props:
            self._open_point_properties_dialog(layer, pid)
        elif chosen is act_intensity:
            from ..widgets.point_intensity_dialog import PointIntensityDialog
            dlg = PointIntensityDialog(
                self._current_window, layer, pid, parent=self._current_window
            )
            dlg.show()
            dlg.raise_()
        elif chosen is act_delete:
            layer.undo_stack.push(RemovePoint(pid), layer.data)

    def _open_point_properties_dialog(self, layer, pid: int) -> None:
        from ..widgets.point_properties_dialog import PointPropertiesDialog

        dlg = PointPropertiesDialog(
            self._current_window, layer, pid, parent=self._current_window
        )
        dlg.show()
        dlg.raise_()

    def _show_layer_menu(self, item, global_pos) -> None:
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        remove_action = menu.addAction("Remove")
        chosen = menu.exec_(global_pos)
        name = item.data(0, _ROLE_LAYER_NAME)
        if chosen is rename_action:
            self._rename_layer(name)
        elif chosen is remove_action:
            self._remove_layer(name)

    def _show_shape_menu(self, item, global_pos) -> None:
        layer, sid = self._resolve_shape_item(item)
        if layer is None:
            return
        menu = QMenu(self)
        act_props = menu.addAction("Edit Properties…")
        act_rename = menu.addAction("Rename…")
        act_duplicate = menu.addAction("Duplicate")
        menu.addSeparator()
        act_delete = menu.addAction("Delete")

        from .manager import compatible_windows
        copy_actions: dict = {}
        if self._current_window is not None:
            targets = compatible_windows(self._current_window)
            if targets:
                menu.addSeparator()
                send_menu = menu.addMenu("Send Copy To Window")
                for tgt in targets:
                    title = tgt.windowTitle() or "window"
                    label = title
                    try:
                        s_src = (self._current_window.meta or {}).get("scale") or ()
                        s_tgt = (tgt.meta or {}).get("scale") or ()
                        if s_src and s_tgt and tuple(s_src) != tuple(s_tgt):
                            label = f"{title}  ⚠ scale differs"
                    except Exception:
                        pass
                    action = send_menu.addAction(label)
                    copy_actions[action] = tgt

        chosen = menu.exec_(global_pos)
        if chosen is act_props:
            self._open_properties_dialog(layer, sid)
        elif chosen is act_rename:
            self._rename_shape(layer, sid)
        elif chosen is act_duplicate:
            self._duplicate_shape(layer, sid)
        elif chosen is act_delete:
            self._delete_shape(layer, sid)
        elif chosen in copy_actions:
            self._current_window._copy_shape_to_window(
                layer, sid, copy_actions[chosen]
            )

    # ------------------------------------------------------------------
    # Per-shape actions
    # ------------------------------------------------------------------
    def _open_properties_dialog(self, layer, sid: int) -> None:
        from ..widgets.shape_properties_dialog import ShapePropertiesDialog

        dlg = ShapePropertiesDialog(
            self._current_window, layer, sid, parent=self._current_window
        )
        dlg.show()
        dlg.raise_()

    def _rename_shape(self, layer, sid: int) -> None:
        current = layer.data.get_label(sid)
        new_label, ok = QInputDialog.getText(
            self, "Rename Shape", "Label:", text=current
        )
        if not ok:
            return
        new_label = new_label.strip()
        if new_label == current:
            return
        layer.undo_stack.push(SetShapeLabel(sid, new_label), layer.data)

    def _duplicate_shape(self, layer, sid: int) -> None:
        rec = layer.data.get(sid)
        new_params = rec.params.copy()
        # Offset by +10 px to keep the duplicate visible & graspable.
        new_params[0:4:2] += 10.0
        new_params[1:4:2] += 10.0
        new_label = f"{rec.label} copy" if rec.label else ""
        layer.undo_stack.push(
            AddShape(
                rec.shape_type,
                new_params,
                t=rec.t,
                z=rec.z,
                label=new_label,
                properties=dict(rec.properties),
            ),
            layer.data,
        )

    def _delete_shape(self, layer, sid: int) -> None:
        layer.selected_ids.discard(sid)
        layer.undo_stack.push(RemoveShape(sid), layer.data)

    # ------------------------------------------------------------------
    # Layer-level actions (unchanged)
    # ------------------------------------------------------------------
    def _on_add_layer(self):
        if self._current_window is None:
            return
        # For now, default to adding a shape layer
        name, ok = QInputDialog.getText(self, "Add Layer", "Layer name:")
        if ok and name:
            self._current_window.add_shape_layer(name)

    def _on_remove_layer(self):
        item = self._tree.currentItem()
        if item is None:
            return
        # Walk up to the layer row if a shape row is selected.
        if item.data(0, _ROLE_KIND) == "shape":
            item = item.parent()
            if item is None:
                return
        name = item.data(0, _ROLE_LAYER_NAME)
        if name:
            self._remove_layer(name)

    def _remove_layer(self, name):
        if self._current_window is None:
            return
        try:
            layer = self._current_window.layers[name]
        except KeyError:
            return

        if layer.layer_type == "shapes":
            self._current_window.remove_shape_layer(name)
        elif layer.layer_type == "points":
            self._current_window.remove_point_layer(name)
        elif layer.layer_type == "tracks":
            self._current_window.remove_track_layer(name)
        elif layer.layer_type == "labels":
            self._current_window.remove_mask_layer(name)
        self._refresh()

    def _rename_layer(self, old_name):
        if self._current_window is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename Layer", "New name:", text=old_name)
        if ok and new_name and new_name != old_name:
            try:
                self._current_window.layers.rename(old_name, new_name)
                self._refresh()
            except (KeyError, ValueError) as e:
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Rename Error", str(e))

    def _on_undo(self):
        if self._current_window is None:
            return
        # Find active shape layer and undo
        layer = self._current_window.layers.active("shapes")
        if layer is not None and layer.undo_stack.can_undo:
            layer.undo_stack.undo(layer.data)

    def _on_redo(self):
        if self._current_window is None:
            return
        layer = self._current_window.layers.active("shapes")
        if layer is not None and layer.undo_stack.can_redo:
            layer.undo_stack.redo(layer.data)


# Singleton access
_layer_manager: LayerManager | None = None


def get_layer_manager() -> LayerManager:
    """Get or create the singleton LayerManager."""
    global _layer_manager
    if _layer_manager is None:
        _layer_manager = LayerManager()
    return _layer_manager


def show_layer_manager(window=None):
    """Show the layer manager panel."""
    lm = get_layer_manager()
    if window is not None:
        lm._connect_window(window)
        lm._set_active_window(window)
    lm.show()
    lm.raise_()
    return lm


def layer_manager_exists() -> bool:
    """Check if a layer manager instance exists."""
    return _layer_manager is not None
