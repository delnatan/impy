"""LayerManager — unified layer panel for all annotation types.

Provides a single tree view showing shapes, points, tracks, and labels
layers from the window's LayerList. Delegates type-specific operations
to the appropriate inspector panels.

This is designed to work alongside the existing AnnotationManager during
migration. Once migration is complete, AnnotationManager can be removed.
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

from .manager import manager


# Type icons for display
_TYPE_LABELS = {
    "shapes": "Shapes",
    "points": "Points",
    "tracks": "Tracks",
    "labels": "Labels",
}


class LayerManager(QWidget):
    """Unified layer management panel.

    Shows all layers from the active window's LayerList in a tree,
    grouped by type. Supports add/remove/rename/visibility/reorder
    operations uniformly.
    """

    _instance: LayerManager | None = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Layer Manager")
        self.setMinimumWidth(280)
        self._current_window = None
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
        manager.window_activated.connect(self._on_window_activated)
        manager.window_removed.connect(self._on_window_removed)

    def _on_window_activated(self, window):
        if self._current_window is window:
            return
        # Disconnect old signals
        if self._current_window is not None:
            try:
                self._current_window.layer_added.disconnect(self._refresh)
                self._current_window.layer_removed.disconnect(self._refresh)
            except (TypeError, RuntimeError):
                pass
        self._current_window = window
        if window is not None:
            self._window_label.setText(window.windowTitle())
            window.layer_added.connect(self._refresh)
            window.layer_removed.connect(self._refresh)
        else:
            self._window_label.setText("No window")
        self._refresh()

    def _on_window_removed(self, name):
        if self._current_window is not None:
            title = self._current_window.windowTitle()
            if title == name:
                self._current_window = None
                self._window_label.setText("No window")
                self._refresh()

    def _refresh(self, *_args):
        """Rebuild the tree from the current window's LayerList."""
        self._tree.blockSignals(True)
        self._tree.clear()

        if self._current_window is None:
            self._tree.blockSignals(False)
            return

        layers = self._current_window.layers
        for layer in layers:
            item = QTreeWidgetItem()
            item.setText(0, layer.name)
            item.setText(1, _TYPE_LABELS.get(layer.layer_type, layer.layer_type))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if layer.visible else Qt.Unchecked)
            item.setData(0, Qt.UserRole, layer.name)
            self._tree.addTopLevelItem(item)

        self._tree.blockSignals(False)

    def _on_item_changed(self, item, column):
        """Handle checkbox toggle for visibility."""
        if self._current_window is None:
            return
        if column != 0:
            return
        name = item.data(0, Qt.UserRole)
        if name is None:
            return
        visible = item.checkState(0) == Qt.Checked
        try:
            layer = self._current_window.layers[name]
            layer.set_visible(visible)
            self._current_window.canvas.update()
        except KeyError:
            pass

    def _on_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if item is None:
            return
        name = item.data(0, Qt.UserRole)
        if name is None:
            return

        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        remove_action = menu.addAction("Remove")

        action = menu.exec_(self._tree.mapToGlobal(pos))
        if action == rename_action:
            self._rename_layer(name)
        elif action == remove_action:
            self._remove_layer(name)

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
        name = item.data(0, Qt.UserRole)
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
            if layer.visual is not None:
                layer.visual.update(
                    layer.data, layer.selected_ids,
                    self._current_window.t_idx, self._current_window.z_idx,
                )
            self._current_window.layer_data_changed.emit(layer)
            self._current_window.canvas.update()

    def _on_redo(self):
        if self._current_window is None:
            return
        layer = self._current_window.layers.active("shapes")
        if layer is not None and layer.undo_stack.can_redo:
            layer.undo_stack.redo(layer.data)
            if layer.visual is not None:
                layer.visual.update(
                    layer.data, layer.selected_ids,
                    self._current_window.t_idx, self._current_window.z_idx,
                )
            self._current_window.layer_data_changed.emit(layer)
            self._current_window.canvas.update()


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
        lm._on_window_activated(window)
    lm.show()
    lm.raise_()
    return lm


def layer_manager_exists() -> bool:
    """Check if a layer manager instance exists."""
    return _layer_manager is not None
