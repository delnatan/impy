"""
LabelManager - Qt widget for managing SparseLabels across ImageWindows.

Follows the ROIManager pattern with hide/show singleton behavior.
"""

import os

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..data.labels import SparseLabels
from .manager import manager


class LabelManager(QWidget):
    """
    Manages SparseLabels across multiple ImageWindows.

    Uses hide/show pattern rather than create/destroy.
    Once instantiated, persists for application lifetime.

    Signals:
        label_added: Emitted when a new label is created (label_id, window)
        active_label_changed: Emitted when active label changes (label_id)
    """

    label_added = Signal(int, object)
    active_label_changed = Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Label Manager")
        self.resize(320, 450)
        self.active_window = None
        self._connected_windows = set()
        self._is_shutting_down = False

        self._setup_ui()

        # Connect to WindowManager for immediate window tracking
        manager.window_registered.connect(self._on_manager_window_registered)

        # Connect to existing windows
        for window in manager.get_all().values():
            self._connect_window(window)

    def _setup_ui(self):
        """Build the UI layout."""
        self.layout = QVBoxLayout(self)

        # Window Selection
        win_layout = QHBoxLayout()
        win_layout.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.window_combo.setMinimumContentsLength(10)
        self.window_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.window_combo.currentIndexChanged.connect(
            self.on_window_combo_changed
        )
        win_layout.addWidget(self.window_combo)
        self.layout.addLayout(win_layout)

        # Labels List
        self.layout.addWidget(QLabel("Labels:"))
        self.label_list = QListWidget()
        self.label_list.itemClicked.connect(self.on_item_clicked)
        self.label_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.layout.addWidget(self.label_list)

        # Active Label & Brush Size
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Active:"))
        self.active_label_spin = QSpinBox()
        self.active_label_spin.setRange(1, 9999)
        self.active_label_spin.setValue(1)
        self.active_label_spin.valueChanged.connect(
            self._on_active_label_changed
        )
        control_layout.addWidget(self.active_label_spin)

        control_layout.addWidget(QLabel("Brush:"))
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 100)
        self.brush_size_spin.setValue(5)
        self.brush_size_spin.setSuffix(" px")
        self.brush_size_spin.valueChanged.connect(self._on_brush_size_changed)
        control_layout.addWidget(self.brush_size_spin)

        self.layout.addLayout(control_layout)

        # Preserve Labels Checkbox
        self.preserve_checkbox = QCheckBox("Preserve existing labels")
        self.preserve_checkbox.setChecked(True)
        self.preserve_checkbox.setToolTip(
            "When checked, painting only affects background pixels.\n"
            "Existing labels are protected from overwriting."
        )
        self.preserve_checkbox.toggled.connect(self._on_preserve_changed)
        self.layout.addWidget(self.preserve_checkbox)

        # Visibility and Opacity
        opacity_layout = QHBoxLayout()

        self.show_checkbox = QCheckBox("Show (V)")
        self.show_checkbox.setChecked(True)
        self.show_checkbox.setToolTip(
            "Toggle label overlay visibility\nKeyboard shortcut: V"
        )
        self.show_checkbox.toggled.connect(self._on_show_changed)
        opacity_layout.addWidget(self.show_checkbox)

        opacity_layout.addWidget(QLabel("Opacity:"))
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setValue(50)
        self.opacity_spin.setSuffix("%")
        self.opacity_spin.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_spin)
        opacity_layout.addStretch()
        self.layout.addLayout(opacity_layout)

        # Action Buttons Row 1
        btn_layout1 = QHBoxLayout()

        self.btn_new = QPushButton("New")
        self.btn_new.clicked.connect(self.new_label)
        btn_layout1.addWidget(self.btn_new)

        self.btn_merge = QPushButton("Merge")
        self.btn_merge.clicked.connect(self.merge_labels)
        btn_layout1.addWidget(self.btn_merge)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self.delete_label)
        btn_layout1.addWidget(self.btn_delete)

        self.btn_rename = QPushButton("Rename")
        self.btn_rename.clicked.connect(self.rename_label)
        btn_layout1.addWidget(self.btn_rename)

        self.layout.addLayout(btn_layout1)

        # Action Buttons Row 2
        btn_layout2 = QHBoxLayout()

        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.save_labels)
        btn_layout2.addWidget(self.btn_save)

        # Load button with menu for format selection
        self.btn_load = QPushButton("Load")
        load_menu = QMenu(self)
        load_menu.addAction("Load Zarr Folder...", self.load_labels_zarr)
        load_menu.addAction("Load NPZ File...", self.load_labels_npz)
        load_menu.addSeparator()
        load_menu.addAction("From Dense Image...", self.load_labels_from_dense)
        self.btn_load.setMenu(load_menu)
        btn_layout2.addWidget(self.btn_load)

        self.layout.addLayout(btn_layout2)

    # ---- Signal Handlers for UI ----

    def _on_active_label_changed(self, value):
        """Handle active label spinbox change."""
        if self.active_window and hasattr(self.active_window, "_active_label"):
            self.active_window._active_label = value
        self.active_label_changed.emit(value)

    def _on_brush_size_changed(self, value):
        """Handle brush size spinbox change."""
        if self.active_window and hasattr(self.active_window, "_brush_size"):
            self.active_window._brush_size = value

    def _on_preserve_changed(self, checked):
        """Handle preserve labels checkbox change."""
        if self.active_window and hasattr(
            self.active_window, "_preserve_labels"
        ):
            self.active_window._preserve_labels = checked

    def _on_show_changed(self, checked):
        """Handle show labels checkbox change."""
        if self.active_window and hasattr(self.active_window, "label_overlay"):
            if self.active_window.label_overlay:
                self.active_window.label_overlay.visible = checked
                self.active_window.canvas.update()

    def _on_opacity_changed(self, value):
        """Handle opacity spinbox change."""
        if self.active_window and hasattr(self.active_window, "label_overlay"):
            if self.active_window.label_overlay:
                self.active_window.label_overlay.set_opacity(value / 100.0)
                self.active_window.canvas.update()

    # ---- Window Management ----

    def _on_manager_window_registered(self, window):
        """Handle WindowManager.window_registered signal."""
        if self._is_shutting_down:
            return
        self._connect_window(window)
        if self.isVisible():
            self.refresh_windows()

    def showEvent(self, event):
        """Refresh when shown."""
        super().showEvent(event)
        self.refresh_windows()
        if not self.active_window and self.window_combo.count() > 0:
            wid = self.window_combo.itemData(0)
            win = manager.get(wid)
            if win:
                self.set_active_window(win)
        self.refresh_list()

    def closeEvent(self, event):
        """Hide instead of destroy (singleton pattern)."""
        if self._is_shutting_down:
            super().closeEvent(event)
        else:
            event.ignore()
            self.hide()

    def cleanup(self):
        """Prepare for application shutdown."""
        self._is_shutting_down = True

        try:
            manager.window_registered.disconnect(
                self._on_manager_window_registered
            )
        except (TypeError, RuntimeError):
            pass

        for window in list(self._connected_windows):
            self._disconnect_window(window)

        self.active_window = None
        self.label_list.clear()
        self.window_combo.clear()

    def _connect_window(self, window):
        """Connect to an ImageWindow's signals."""
        if window in self._connected_windows:
            return

        # Only connect to label-capable windows
        if not hasattr(window, "label_changed"):
            return

        window.window_shown.connect(self._on_window_shown)
        window.window_closing.connect(self._on_window_closing)
        window.window_activated.connect(self._on_window_activated)
        window.label_changed.connect(self._on_label_changed)

        self._connected_windows.add(window)

    def _disconnect_window(self, window):
        """Disconnect from an ImageWindow's signals."""
        if window not in self._connected_windows:
            return

        try:
            window.window_shown.disconnect(self._on_window_shown)
            window.window_closing.disconnect(self._on_window_closing)
            window.window_activated.disconnect(self._on_window_activated)
            window.label_changed.disconnect(self._on_label_changed)
        except (TypeError, RuntimeError):
            pass

        self._connected_windows.discard(window)

    def _on_window_shown(self, window):
        if self._is_shutting_down:
            return
        self.refresh_windows()

    def _on_window_closing(self, window):
        self._disconnect_window(window)
        if self._is_shutting_down:
            return
        self.remove_window(window)

    def _on_window_activated(self, window):
        if self._is_shutting_down:
            return
        self.set_active_window(window)

    def _on_label_changed(self, labels):
        """Handle label_changed signal from window."""
        if self._is_shutting_down:
            return
        self.refresh_list()

    def refresh_windows(self):
        """Populate the window combo box."""
        self.window_combo.blockSignals(True)
        self.window_combo.clear()

        windows = manager.get_all()
        for wid, win in windows.items():
            # Only show label-capable windows
            if not hasattr(win, "label_changed"):
                continue
            title = win.windowTitle()
            self.window_combo.addItem(title, userData=wid)
            self._connect_window(win)

        if self.active_window:
            idx = self.window_combo.findData(self.active_window.window_id)
            if idx >= 0:
                self.window_combo.setCurrentIndex(idx)

        self.window_combo.blockSignals(False)

    def on_window_combo_changed(self, index):
        if index < 0:
            return
        wid = self.window_combo.itemData(index)
        win = manager.get(wid)
        if win:
            self.set_active_window(win)

    def set_active_window(self, window):
        if self.active_window == window:
            return

        self.active_window = window
        self.setWindowTitle(f"Label Manager - Window {window.window_id}")
        self.refresh_windows()
        self.refresh_list()

        # Sync UI with window state
        if hasattr(window, "_active_label"):
            self.active_label_spin.blockSignals(True)
            self.active_label_spin.setValue(window._active_label)
            self.active_label_spin.blockSignals(False)

        if hasattr(window, "_brush_size"):
            self.brush_size_spin.blockSignals(True)
            self.brush_size_spin.setValue(window._brush_size)
            self.brush_size_spin.blockSignals(False)

        if hasattr(window, "_preserve_labels"):
            self.preserve_checkbox.blockSignals(True)
            self.preserve_checkbox.setChecked(window._preserve_labels)
            self.preserve_checkbox.blockSignals(False)

        if hasattr(window, "label_overlay") and window.label_overlay:
            self.opacity_spin.blockSignals(True)
            self.opacity_spin.setValue(
                int(window.label_overlay.get_opacity() * 100)
            )
            self.opacity_spin.blockSignals(False)

            self.show_checkbox.blockSignals(True)
            self.show_checkbox.setChecked(window.label_overlay.visible)
            self.show_checkbox.blockSignals(False)

    def remove_window(self, window):
        """Handle window closure."""
        if self.active_window == window:
            self.active_window = None
            self.setWindowTitle("Label Manager")
            self.label_list.clear()

        self.refresh_windows()

        if not self.active_window and self.window_combo.count() > 0:
            self.window_combo.setCurrentIndex(0)

    def refresh_list(self):
        """Refresh the label list from active window."""
        self.label_list.clear()

        if not self.active_window:
            return

        labels = getattr(self.active_window, "labels", None)
        if labels is None:
            return

        overlay = getattr(self.active_window, "label_overlay", None)

        for label in labels:
            area = labels.area(label)
            item = QListWidgetItem(f"Label {label} ({area} px)")
            item.setData(Qt.UserRole, label)

            # Show color indicator
            if overlay:
                color = overlay.get_label_color(label)
                qcolor = QColor.fromRgbF(color[0], color[1], color[2])
                item.setBackground(qcolor)

                # Checkable for visibility
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                visible = overlay.get_label_visible(label)
                item.setCheckState(Qt.Checked if visible else Qt.Unchecked)

            self.label_list.addItem(item)

    def on_item_clicked(self, item):
        """Handle label list item click."""
        if not self.active_window:
            return

        label = item.data(Qt.UserRole)

        # Set as active label
        self.active_label_spin.setValue(label)

        # Handle visibility checkbox
        overlay = getattr(self.active_window, "label_overlay", None)
        if overlay:
            visible = item.checkState() == Qt.Checked
            overlay.set_label_visible(label, visible)
            self.active_window.canvas.update()

    def on_item_double_clicked(self, item):
        """Handle double-click to change color."""
        if not self.active_window:
            return

        label = item.data(Qt.UserRole)
        overlay = getattr(self.active_window, "label_overlay", None)

        if overlay:
            current_color = overlay.get_label_color(label)
            qcolor = QColor.fromRgbF(
                current_color[0], current_color[1], current_color[2]
            )

            new_color = QColorDialog.getColor(
                qcolor, self, "Select Label Color"
            )
            if new_color.isValid():
                rgba = (
                    new_color.redF(),
                    new_color.greenF(),
                    new_color.blueF(),
                    current_color[3],  # Keep alpha
                )
                overlay.set_label_color(label, rgba)
                self.refresh_list()
                self.active_window.canvas.update()

    # ---- Label Actions ----

    def new_label(self):
        """Create a new label with next available ID."""
        if not self.active_window:
            return

        labels = getattr(self.active_window, "labels", None)
        if labels is None:
            # Create new SparseLabels if none exists
            Y, X = self.active_window.Y, self.active_window.X
            if self.active_window.Z > 1:
                shape = (self.active_window.Z, Y, X)
            else:
                shape = (Y, X)
            labels = SparseLabels(shape)
            self.active_window.labels = labels

            # Create overlay if needed
            if hasattr(self.active_window, "_ensure_label_overlay"):
                self.active_window._ensure_label_overlay()

        # Find next available label
        existing = set(labels.labels)
        new_id = 1
        while new_id in existing:
            new_id += 1

        # Set as active label
        self.active_label_spin.setValue(new_id)
        self.label_added.emit(new_id, self.active_window)

    def delete_label(self):
        """Delete selected label."""
        item = self.label_list.currentItem()
        if not item or not self.active_window:
            return

        label = item.data(Qt.UserRole)
        labels = getattr(self.active_window, "labels", None)

        if labels and label in labels:
            labels.remove(label)
            self.active_window.label_changed.emit(labels)

            overlay = getattr(self.active_window, "label_overlay", None)
            if overlay:
                overlay.refresh()

            self.refresh_list()
            self.active_window.canvas.update()

    def merge_labels(self):
        """Merge selected labels (multi-select with Ctrl)."""
        if not self.active_window:
            return

        selected_items = self.label_list.selectedItems()
        if len(selected_items) < 2:
            return

        labels_to_merge = [item.data(Qt.UserRole) for item in selected_items]
        labels = getattr(self.active_window, "labels", None)

        if labels:
            new_label = labels.merge(labels_to_merge)
            self.active_window.label_changed.emit(labels)

            overlay = getattr(self.active_window, "label_overlay", None)
            if overlay:
                overlay.refresh()

            self.refresh_list()
            self.active_window.canvas.update()

            # Set merged label as active
            self.active_label_spin.setValue(new_label)

    def rename_label(self):
        """Rename (relabel) selected label."""
        item = self.label_list.currentItem()
        if not item or not self.active_window:
            return

        old_label = item.data(Qt.UserRole)
        labels = getattr(self.active_window, "labels", None)

        if not labels or old_label not in labels:
            return

        new_label, ok = QInputDialog.getInt(
            self,
            "Rename Label",
            f"Enter new ID for label {old_label}:",
            value=old_label,
            min=1,
            max=99999,
        )

        if ok and new_label != old_label:
            try:
                labels.relabel(old_label, new_label)
                self.active_window.label_changed.emit(labels)

                overlay = getattr(self.active_window, "label_overlay", None)
                if overlay:
                    overlay.refresh()

                self.refresh_list()
                self.active_window.canvas.update()

                self.active_label_spin.setValue(new_label)
            except ValueError as e:
                from qtpy.QtWidgets import QMessageBox

                QMessageBox.warning(self, "Rename Failed", str(e))

    def save_labels(self):
        """Save labels to file."""
        if not self.active_window:
            return

        labels = getattr(self.active_window, "labels", None)
        if not labels or labels.n_objects == 0:
            return

        # Default path
        default_dir = "."
        default_name = "labels.sparse.zarr"

        if self.active_window.filepath:
            default_dir = os.path.dirname(self.active_window.filepath)
            base_name = os.path.basename(self.active_window.filepath)
            name_without_ext = os.path.splitext(base_name)[0]
            default_name = f"{name_without_ext}.sparse.zarr"

        default_path = os.path.join(default_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Labels",
            default_path,
            "Zarr Labels (*.sparse.zarr);;NPZ Labels (*.sparse.npz)",
        )

        if not path:
            return

        # Ensure proper extension
        if not (path.endswith(".sparse.zarr") or path.endswith(".sparse.npz")):
            path += ".sparse.zarr"

        labels.save(path)

    def _get_expected_shape(self):
        """Get expected label shape for active window."""
        Y, X = self.active_window.Y, self.active_window.X
        if self.active_window.Z > 1:
            return (self.active_window.Z, Y, X)
        return (Y, X)

    def _apply_loaded_labels(self, labels):
        """Apply loaded labels to active window with validation."""
        expected_shape = self._get_expected_shape()

        if labels.shape != expected_shape:
            QMessageBox.warning(
                self,
                "Shape Mismatch",
                f"Label shape {labels.shape} does not match "
                f"image shape {expected_shape}",
            )
            return False

        self.active_window.labels = labels

        # Update overlay
        if hasattr(self.active_window, "_ensure_label_overlay"):
            self.active_window._ensure_label_overlay()

        overlay = getattr(self.active_window, "label_overlay", None)
        if overlay:
            overlay.set_labels(labels)
            overlay.refresh()

        self.active_window.label_changed.emit(labels)
        self.refresh_list()
        self.active_window.canvas.update()
        return True

    def load_labels_zarr(self):
        """Load labels from .sparse.zarr folder."""
        if not self.active_window:
            return

        default_dir = "."
        if self.active_window.filepath:
            default_dir = os.path.dirname(self.active_window.filepath)

        # Use directory picker for zarr
        path = QFileDialog.getExistingDirectory(
            self,
            "Select .sparse.zarr Folder",
            default_dir,
        )

        if not path:
            return

        if not path.endswith(".sparse.zarr"):
            QMessageBox.warning(
                self,
                "Invalid Folder",
                "Please select a folder ending with .sparse.zarr",
            )
            return

        try:
            labels = SparseLabels.from_file(path)
            self._apply_loaded_labels(labels)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    def load_labels_npz(self):
        """Load labels from .sparse.npz file."""
        if not self.active_window:
            return

        default_dir = "."
        if self.active_window.filepath:
            default_dir = os.path.dirname(self.active_window.filepath)

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load NPZ Labels",
            default_dir,
            "NPZ Labels (*.sparse.npz);;All Files (*)",
        )

        if not path:
            return

        try:
            labels = SparseLabels.from_file(path)
            self._apply_loaded_labels(labels)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    def load_labels_from_dense(self):
        """Load labels from a dense label image (e.g., from segmentation)."""
        if not self.active_window:
            return

        default_dir = "."
        if self.active_window.filepath:
            default_dir = os.path.dirname(self.active_window.filepath)

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Dense Label Image",
            default_dir,
            "Image Files (*.tif *.tiff *.png);;All Files (*)",
        )

        if not path:
            return

        try:
            import numpy as np
            from tifffile import imread

            # Load the dense label image
            if path.endswith(('.tif', '.tiff')):
                dense_image = imread(path)
            else:
                from PIL import Image
                dense_image = np.asarray(Image.open(path))
                if dense_image.ndim == 3:
                    dense_image = dense_image[:, :, 0]
                dense_image = dense_image.astype(np.uint16)

            # Convert to sparse
            labels = SparseLabels.from_dense(dense_image)
            self._apply_loaded_labels(labels)

        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    # ---- Keyboard Shortcuts ----

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_Delete:
            self.delete_label()
        elif event.key() == Qt.Key_N:
            self.new_label()
        else:
            super().keyPressEvent(event)


# Global singleton
_label_manager_instance = None


def get_label_manager():
    """Get the global LabelManager instance (creates if needed)."""
    global _label_manager_instance
    if _label_manager_instance is None:
        _label_manager_instance = LabelManager()
    return _label_manager_instance


def label_manager_exists():
    """Check if LabelManager singleton has been created."""
    return _label_manager_instance is not None
