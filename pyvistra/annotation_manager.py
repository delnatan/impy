"""
AnnotationManager - Unified widget for managing ROI groups, mask layers, and track layers.

Replaces ROIManager + LabelManager with a single widget that manages:
- ROI groups: named collections of ROIs with show/hide
- Mask layers: named SparseLabels instances with their own visuals
- Track layers: named TrackTable instances with time-aware track rendering
"""

import json
import os
import csv

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAction,
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
    QMenuBar,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .labels import SparseLabels
from .manager import manager
from .rois import (
    CircleROI,
    CoordinateROI,
    LaneROI,
    LineROI,
    RectangleROI,
)
from .tracks import TrackTable


class AnnotationManager(QWidget):
    """
    Manages ROI groups, mask layers, and track layers across multiple ImageWindows.

    Uses hide/show pattern (singleton). Provides the same `active_window`
    and `refresh_list()` interface that GelAnalyzer depends on.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotation Manager")
        self.resize(320, 550)
        self.active_window = None
        self._connected_windows = set()
        self._is_shutting_down = False
        self._selected_group = None  # (type, name) e.g. ("roi", "Default") or ("mask", "Labels-1")

        self._setup_ui()

        # Connect to WindowManager for immediate window tracking
        manager.window_registered.connect(self._on_manager_window_registered)

        # Connect to existing windows
        for window in manager.get_all().values():
            self._connect_window(window)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self._install_shortcuts()

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
        layout.addLayout(win_layout)

        # Groups Tree
        layout.addWidget(QLabel("Groups:"))
        self.group_tree = QTreeWidget()
        self.group_tree.setHeaderHidden(True)
        self.group_tree.setMaximumHeight(160)
        self.group_tree.itemClicked.connect(self._on_group_clicked)
        self.group_tree.itemChanged.connect(self._on_group_check_changed)
        layout.addWidget(self.group_tree)

        # Group buttons
        group_btn_layout = QHBoxLayout()
        btn_add_group = QPushButton("+ ROI Group")
        btn_add_group.clicked.connect(self._add_roi_group)
        group_btn_layout.addWidget(btn_add_group)

        btn_add_mask = QPushButton("+ Mask Layer")
        btn_add_mask.clicked.connect(self._add_mask_layer)
        group_btn_layout.addWidget(btn_add_mask)

        btn_add_track = QPushButton("+ Track Layer")
        btn_add_track.clicked.connect(self._add_track_layer)
        group_btn_layout.addWidget(btn_add_track)

        btn_delete_group = QPushButton("Delete")
        btn_delete_group.clicked.connect(self._delete_selected_group)
        group_btn_layout.addWidget(btn_delete_group)
        layout.addLayout(group_btn_layout)

        # Items List
        layout.addWidget(QLabel("Items:"))
        self.item_list = QListWidget()
        self.item_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.item_list)

        # Label controls (for mask layers)
        self.label_controls = QWidget()
        label_ctrl_layout = QHBoxLayout(self.label_controls)
        label_ctrl_layout.setContentsMargins(0, 0, 0, 0)

        label_ctrl_layout.addWidget(QLabel("Active:"))
        self.active_label_spin = QSpinBox()
        self.active_label_spin.setRange(1, 9999)
        self.active_label_spin.setValue(1)
        self.active_label_spin.valueChanged.connect(
            self._on_active_label_changed
        )
        label_ctrl_layout.addWidget(self.active_label_spin)

        label_ctrl_layout.addWidget(QLabel("Brush:"))
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 100)
        self.brush_size_spin.setValue(5)
        self.brush_size_spin.setSuffix(" px")
        self.brush_size_spin.valueChanged.connect(self._on_brush_size_changed)
        label_ctrl_layout.addWidget(self.brush_size_spin)

        layout.addWidget(self.label_controls)
        self.label_controls.setVisible(False)

        # Preserve labels checkbox
        self.preserve_checkbox = QCheckBox("Preserve existing labels")
        self.preserve_checkbox.setChecked(True)
        self.preserve_checkbox.toggled.connect(self._on_preserve_changed)
        layout.addWidget(self.preserve_checkbox)
        self.preserve_checkbox.setVisible(False)

        # Fill all Z slices checkbox (cookie-cut)
        self.fill_all_z_checkbox = QCheckBox("Fill all Z slices")
        self.fill_all_z_checkbox.setChecked(False)
        self.fill_all_z_checkbox.toggled.connect(self._on_fill_all_z_changed)
        layout.addWidget(self.fill_all_z_checkbox)
        self.fill_all_z_checkbox.setVisible(False)

        # Opacity control
        self.opacity_widget = QWidget()
        opacity_layout = QHBoxLayout(self.opacity_widget)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setValue(50)
        self.opacity_spin.setSuffix("%")
        self.opacity_spin.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_spin)
        opacity_layout.addStretch()
        layout.addWidget(self.opacity_widget)
        self.opacity_widget.setVisible(False)

        # Action Buttons
        btn_layout = QHBoxLayout()

        self.btn_delete = QPushButton("Delete Item")
        self.btn_delete.clicked.connect(self.delete_selected_item)
        btn_layout.addWidget(self.btn_delete)

        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.save_annotations)
        btn_layout.addWidget(self.btn_save)

        self.btn_load = QPushButton("Load")
        load_menu = QMenu(self)
        load_menu.addAction("Load ROIs (JSON)...", self.load_rois)
        load_menu.addAction("Load Tracks (CSV/JSON)...", self.load_tracks)
        load_menu.addAction("Load Mask (Zarr)...", self.load_mask_zarr)
        load_menu.addAction("Load Mask (NPZ)...", self.load_mask_npz)
        load_menu.addSeparator()
        load_menu.addAction("From Dense Image...", self.load_mask_from_dense)
        self.btn_load.setMenu(load_menu)
        btn_layout.addWidget(self.btn_load)

        layout.addLayout(btn_layout)

        # Menu Bar
        self.menu_bar = QMenuBar()
        layout.setMenuBar(self.menu_bar)

        # Analysis Menu (same as ROIManager)
        from .analysis import crop_image, measure_intensity

        analysis_menu = self.menu_bar.addMenu("Analysis")

        action_crop = QAction("Crop Image", self)
        action_crop.triggered.connect(lambda: self._run_analysis(crop_image))
        analysis_menu.addAction(action_crop)

        action_measure = QAction("Measure Intensity", self)
        action_measure.triggered.connect(
            lambda: self._run_analysis(measure_intensity)
        )
        analysis_menu.addAction(action_measure)

        action_line_profile = QAction("Line Profile...", self)
        action_line_profile.triggered.connect(self._show_line_profile_dialog)
        analysis_menu.addAction(action_line_profile)

        analysis_menu.addSeparator()

        from .gel_analyzer import show_gel_analyzer

        action_gel = QAction("Gel Analyzer...", self)
        action_gel.triggered.connect(lambda: show_gel_analyzer(self))
        analysis_menu.addAction(action_gel)

        # Lanes Menu
        from .analysis import align_lanes

        lanes_menu = self.menu_bar.addMenu("Lanes")
        action_align = QAction("Align Lanes", self)
        action_align.triggered.connect(self._align_lanes_action)
        lanes_menu.addAction(action_align)

    def _install_shortcuts(self):
        """Install manager-local shortcuts that work with child focus."""
        self._shortcut_delete = QShortcut(Qt.Key_Delete, self)
        self._shortcut_delete.setContext(Qt.WidgetWithChildrenShortcut)
        self._shortcut_delete.activated.connect(self.delete_selected_item)

        self._shortcut_flip = QShortcut(Qt.Key_F, self)
        self._shortcut_flip.setContext(Qt.WidgetWithChildrenShortcut)
        self._shortcut_flip.activated.connect(self._flip_selected_coordinate_roi)

    # ---- Window Management ----

    def _on_manager_window_registered(self, window):
        if self._is_shutting_down:
            return
        self._connect_window(window)
        if self.isVisible():
            self.refresh_windows()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_windows()
        if not self.active_window and self.window_combo.count() > 0:
            wid = self.window_combo.itemData(0)
            win = manager.get(wid)
            if win:
                self.set_active_window(win)
        self.refresh_list()

    def closeEvent(self, event):
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
        self.item_list.clear()
        self.group_tree.clear()
        self.window_combo.clear()

    def _connect_window(self, window):
        if window in self._connected_windows:
            return
        if not hasattr(window, "roi_added"):
            return

        window.window_shown.connect(self._on_window_shown)
        window.window_closing.connect(self._on_window_closing)
        window.window_activated.connect(self._on_window_activated)
        window.roi_added.connect(self._on_roi_added)
        window.roi_removed.connect(self._on_roi_removed)
        window.roi_selection_changed.connect(self._on_roi_selection_changed)
        window.label_changed.connect(self._on_label_changed)

        if hasattr(window, "mask_layer_added"):
            window.mask_layer_added.connect(self._on_mask_layer_added)
        if hasattr(window, "mask_layer_removed"):
            window.mask_layer_removed.connect(self._on_mask_layer_removed)
        if hasattr(window, "track_layer_added"):
            window.track_layer_added.connect(self._on_track_layer_added)
        if hasattr(window, "track_layer_removed"):
            window.track_layer_removed.connect(self._on_track_layer_removed)

        self._connected_windows.add(window)

    def _disconnect_window(self, window):
        if window not in self._connected_windows:
            return
        try:
            window.window_shown.disconnect(self._on_window_shown)
            window.window_closing.disconnect(self._on_window_closing)
            window.window_activated.disconnect(self._on_window_activated)
            window.roi_added.disconnect(self._on_roi_added)
            window.roi_removed.disconnect(self._on_roi_removed)
            window.roi_selection_changed.disconnect(
                self._on_roi_selection_changed
            )
            window.label_changed.disconnect(self._on_label_changed)
            if hasattr(window, "mask_layer_added"):
                window.mask_layer_added.disconnect(self._on_mask_layer_added)
            if hasattr(window, "mask_layer_removed"):
                window.mask_layer_removed.disconnect(
                    self._on_mask_layer_removed
                )
            if hasattr(window, "track_layer_added"):
                window.track_layer_added.disconnect(self._on_track_layer_added)
            if hasattr(window, "track_layer_removed"):
                window.track_layer_removed.disconnect(
                    self._on_track_layer_removed
                )
        except (TypeError, RuntimeError):
            pass
        self._connected_windows.discard(window)

    # ---- Signal Handlers ----

    def _on_window_shown(self, window):
        if self._is_shutting_down:
            return
        self.refresh_windows()

    def _on_window_closing(self, window):
        self._disconnect_window(window)
        if self._is_shutting_down:
            return
        self._remove_window(window)

    def _on_window_activated(self, window):
        if self._is_shutting_down:
            return
        self.set_active_window(window)

    def _on_roi_added(self, roi):
        if self._is_shutting_down:
            return
        self.refresh_list()

    def _on_roi_removed(self, roi):
        if self._is_shutting_down:
            return
        self.refresh_list()

    def _on_roi_selection_changed(self, roi):
        if self._is_shutting_down:
            return
        self._select_roi_in_list(roi)

    def _on_label_changed(self, labels):
        if self._is_shutting_down:
            return
        self.refresh_list()

    def _on_mask_layer_added(self, name):
        if self._is_shutting_down:
            return
        self.refresh_list()

    def _on_mask_layer_removed(self, name):
        if self._is_shutting_down:
            return
        self.refresh_list()

    def _on_track_layer_added(self, name):
        if self._is_shutting_down:
            return
        self.refresh_list()

    def _on_track_layer_removed(self, name):
        if self._is_shutting_down:
            return
        self.refresh_list()

    def refresh_windows(self):
        self.window_combo.blockSignals(True)
        self.window_combo.clear()

        windows = manager.get_all()
        for wid, win in windows.items():
            if not hasattr(win, "roi_added"):
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
        self.setWindowTitle(f"Annotation Manager - Window {window.window_id}")
        self.refresh_windows()
        self.refresh_list()

        # Sync label controls
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
        if hasattr(window, "_mask_propagate_z"):
            self.fill_all_z_checkbox.blockSignals(True)
            self.fill_all_z_checkbox.setChecked(window._mask_propagate_z)
            self.fill_all_z_checkbox.blockSignals(False)

    def _remove_window(self, window):
        if self.active_window == window:
            self.active_window = None
            self.setWindowTitle("Annotation Manager")
            self.item_list.clear()
            self.group_tree.clear()
        self.refresh_windows()
        if not self.active_window and self.window_combo.count() > 0:
            self.window_combo.setCurrentIndex(0)

    # ---- Refresh ----

    def refresh_list(self):
        """Refresh both the group tree and items list."""
        self._ensure_valid_selected_group()
        self._refresh_group_tree()
        self._refresh_items()

    def _ensure_valid_selected_group(self):
        """
        Ensure selected group is valid for the active window.

        On first open, default to the active ROI group so pre-existing ROIs
        are immediately visible in the items list.
        """
        if not self.active_window:
            self._selected_group = None
            return

        w = self.active_window
        roi_groups = w.get_roi_groups()
        mask_groups = set(w._mask_layers.keys())
        track_groups = set(w._track_layers.keys())

        if self._selected_group is not None:
            gtype, gname = self._selected_group
            if gtype == "roi" and gname in roi_groups:
                return
            if gtype == "mask" and gname in mask_groups:
                return
            if gtype == "track" and gname in track_groups:
                return

        preferred_roi = getattr(w, "_active_roi_group", None)
        if preferred_roi in roi_groups:
            self._selected_group = ("roi", preferred_roi)
            return

        if roi_groups:
            self._selected_group = ("roi", next(iter(roi_groups.keys())))
            return

        if mask_groups:
            self._selected_group = ("mask", next(iter(mask_groups)))
            return

        if track_groups:
            self._selected_group = ("track", next(iter(track_groups)))
            return

        self._selected_group = None

    def _refresh_group_tree(self):
        self.group_tree.blockSignals(True)
        self.group_tree.clear()

        if not self.active_window:
            self.group_tree.blockSignals(False)
            return

        w = self.active_window

        # ROI groups
        roi_groups = w.get_roi_groups()
        for name, rois in roi_groups.items():
            visible = w._roi_groups_visible.get(name, True)
            item = QTreeWidgetItem(
                self.group_tree,
                [f"[shapes] {name} ({len(rois)} ROIs)"],
            )
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)
            item.setData(0, Qt.UserRole, ("roi", name))
            # Highlight active group
            if name == w._active_roi_group:
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            if self._selected_group == ("roi", name):
                self.group_tree.setCurrentItem(item)

        # Mask layers
        for name, entry in w._mask_layers.items():
            n_labels = entry["labels"].n_objects if entry["labels"] else 0
            visible = entry["visible"]
            item = QTreeWidgetItem(
                self.group_tree,
                [f"[mask] {name} ({n_labels} labels)"],
            )
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)
            item.setData(0, Qt.UserRole, ("mask", name))
            # Highlight active mask layer
            if name == w._active_mask_layer:
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            if self._selected_group == ("mask", name):
                self.group_tree.setCurrentItem(item)

        # Track layers
        for name, entry in w._track_layers.items():
            tracks = entry["tracks"]
            n_tracks = tracks.n_tracks if tracks is not None else 0
            visible = entry["visible"]
            item = QTreeWidgetItem(
                self.group_tree,
                [f"[track] {name} ({n_tracks} tracks)"],
            )
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)
            item.setData(0, Qt.UserRole, ("track", name))
            if name == w._active_track_layer:
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            if self._selected_group == ("track", name):
                self.group_tree.setCurrentItem(item)

        self.group_tree.blockSignals(False)

    def _refresh_items(self):
        self.item_list.clear()
        if not self.active_window or not self._selected_group:
            return

        gtype, gname = self._selected_group
        w = self.active_window

        if gtype == "roi":
            for i, roi in enumerate(w.rois):
                if getattr(roi, "group", "Default") == gname:
                    item = QListWidgetItem(
                        f"{roi.name} ({roi.__class__.__name__})"
                    )
                    item.setData(Qt.UserRole, ("roi", roi))
                    self.item_list.addItem(item)
        elif gtype == "mask":
            if gname in w._mask_layers:
                labels = w._mask_layers[gname]["labels"]
                overlay = w._mask_layers[gname]["visual"]
                if labels:
                    for label in labels:
                        area = labels.area(label)
                        item = QListWidgetItem(f"Label {label} ({area} px)")
                        item.setData(Qt.UserRole, ("label", label))
                        # Show color
                        if overlay:
                            color = overlay.get_label_color(label)
                            qcolor = QColor.fromRgbF(
                                color[0], color[1], color[2]
                            )
                            item.setBackground(qcolor)
                        self.item_list.addItem(item)
        elif gtype == "track":
            if gname in w._track_layers:
                tracks = w._track_layers[gname]["tracks"]
                if tracks is not None:
                    for track_id in tracks.track_ids:
                        track = tracks.get_track(int(track_id))
                        if track is None:
                            continue
                        n_points = len(track["t"])
                        t0 = int(track["t"][0]) if n_points else 0
                        t1 = int(track["t"][-1]) if n_points else 0
                        item = QListWidgetItem(
                            f"Track {int(track_id)} ({n_points} pts, t={t0}-{t1})"
                        )
                        item.setData(Qt.UserRole, ("track", int(track_id)))
                        self.item_list.addItem(item)

    # ---- Group Tree Interactions ----

    def _on_group_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data is None:
            return
        gtype, gname = data
        self._selected_group = (gtype, gname)

        if not self.active_window:
            return

        # Set active group/layer
        if gtype == "roi":
            self.active_window._active_roi_group = gname
        elif gtype == "mask":
            self.active_window.set_active_mask_layer(gname)
        elif gtype == "track":
            self.active_window.set_active_track_layer(gname)

        # Show/hide label controls
        is_mask = gtype == "mask"
        self.label_controls.setVisible(is_mask)
        self.preserve_checkbox.setVisible(is_mask)
        self.opacity_widget.setVisible(is_mask)
        self.fill_all_z_checkbox.setVisible(
            is_mask and self.active_window.Z > 1
        )

        # Sync opacity if mask
        if is_mask and gname in self.active_window._mask_layers:
            visual = self.active_window._mask_layers[gname]["visual"]
            if visual:
                self.opacity_spin.blockSignals(True)
                self.opacity_spin.setValue(int(visual.get_opacity() * 100))
                self.opacity_spin.blockSignals(False)

        self._refresh_items()
        self._refresh_group_tree()

    def _on_group_check_changed(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data is None:
            return
        gtype, gname = data
        visible = item.checkState(0) == Qt.Checked

        if not self.active_window:
            return

        if gtype == "roi":
            self.active_window.set_group_visible(gname, visible)
        elif gtype == "mask":
            self.active_window.set_mask_layer_visible(gname, visible)
        elif gtype == "track":
            self.active_window.set_track_layer_visible(gname, visible)

    def _add_roi_group(self):
        if not self.active_window:
            return
        name, ok = QInputDialog.getText(self, "New ROI Group", "Group name:")
        if ok and name:
            self.active_window.add_roi_group(name)
            self.refresh_list()

    def _add_mask_layer(self):
        if not self.active_window:
            return
        # Auto-generate name
        existing = set(self.active_window._mask_layers.keys())
        idx = 1
        while f"Labels-{idx}" in existing:
            idx += 1
        name = f"Labels-{idx}"

        name, ok = QInputDialog.getText(
            self, "New Mask Layer", "Layer name:", text=name
        )
        if ok and name:
            self.active_window.add_mask_layer(name)
            self.refresh_list()

    def _add_track_layer(self):
        if not self.active_window:
            return
        existing = set(self.active_window._track_layers.keys())
        idx = 1
        while f"Tracks-{idx}" in existing:
            idx += 1
        name = f"Tracks-{idx}"

        name, ok = QInputDialog.getText(
            self, "New Track Layer", "Layer name:", text=name
        )
        if ok and name:
            self.active_window.add_track_layer(name)
            self.refresh_list()

    def _delete_selected_group(self):
        if not self.active_window or not self._selected_group:
            return
        gtype, gname = self._selected_group

        if gtype == "roi":
            self.active_window.remove_roi_group(gname)
        elif gtype == "mask":
            self.active_window.remove_mask_layer(gname)
        elif gtype == "track":
            self.active_window.remove_track_layer(gname)

        self._selected_group = None
        self.refresh_list()

    # ---- Items Interactions ----

    def _on_item_clicked(self, item):
        if not self.active_window:
            return
        data = item.data(Qt.UserRole)
        if data is None:
            return

        item_type, value = data

        if item_type == "roi":
            # Select this ROI, deselect others
            for r in self.active_window.rois:
                r.select(r is value)
            self.active_window.canvas.update()
        elif item_type == "label":
            # Set as active label
            self.active_label_spin.setValue(value)

    def _select_roi_in_list(self, roi):
        """Sync list selection to match canvas selection."""
        self.item_list.blockSignals(True)
        found = False
        for i in range(self.item_list.count()):
            item = self.item_list.item(i)
            data = item.data(Qt.UserRole)
            if data and data[0] == "roi" and data[1] is roi:
                self.item_list.setCurrentItem(item)
                found = True
                break
        if not found:
            self.item_list.clearSelection()
        self.item_list.blockSignals(False)

    def delete_selected_item(self):
        item = self.item_list.currentItem()
        if not item or not self.active_window:
            return
        data = item.data(Qt.UserRole)
        if data is None:
            return

        item_type, value = data

        if item_type == "roi":
            self.active_window.remove_roi(value)
        elif item_type == "label":
            if self._selected_group:
                _, gname = self._selected_group
                if gname in self.active_window._mask_layers:
                    labels = self.active_window._mask_layers[gname]["labels"]
                    if labels and value in labels:
                        labels.remove(value)
                        visual = self.active_window._mask_layers[gname][
                            "visual"
                        ]
                        if visual:
                            visual.refresh()
                        self.active_window.label_changed.emit(labels)
        elif item_type == "track":
            if self._selected_group and self._selected_group[0] == "track":
                _, gname = self._selected_group
                self.active_window.remove_track_from_layer(gname, value)

        self.refresh_list()
        self.active_window.canvas.update()

    # ---- Label Controls ----

    def _on_active_label_changed(self, value):
        if self.active_window and hasattr(self.active_window, "_active_label"):
            self.active_window._active_label = value

    def _on_brush_size_changed(self, value):
        if self.active_window and hasattr(self.active_window, "_brush_size"):
            self.active_window._brush_size = value

    def _on_preserve_changed(self, checked):
        if self.active_window and hasattr(
            self.active_window, "_preserve_labels"
        ):
            self.active_window._preserve_labels = checked

    def _on_fill_all_z_changed(self, checked):
        if self.active_window and hasattr(
            self.active_window, "_mask_propagate_z"
        ):
            self.active_window._mask_propagate_z = checked

    def _on_opacity_changed(self, value):
        if not self.active_window or not self._selected_group:
            return
        gtype, gname = self._selected_group
        if gtype == "mask" and gname in self.active_window._mask_layers:
            visual = self.active_window._mask_layers[gname]["visual"]
            if visual:
                visual.set_opacity(value / 100.0)
                self.active_window.canvas.update()

    # ---- Save/Load ----

    def _get_default_dir(self):
        if self.active_window and self.active_window.filepath:
            return os.path.dirname(self.active_window.filepath)
        return "."

    def _get_base_name(self):
        if self.active_window and self.active_window.filepath:
            base = os.path.basename(self.active_window.filepath)
            return os.path.splitext(base)[0]
        return "annotations"

    def save_annotations(self):
        """Save selected group or all annotations."""
        if not self.active_window:
            return

        if self._selected_group:
            gtype, gname = self._selected_group
            if gtype == "roi":
                self._save_roi_group(gname)
            elif gtype == "mask":
                self._save_mask_layer(gname)
        else:
            self._save_all_rois()

    def _save_roi_group(self, group_name):
        default_name = f"{self._get_base_name()}_{group_name}.json"
        default_path = os.path.join(self._get_default_dir(), default_name)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI Group", default_path, "JSON Files (*.json)"
        )
        if not path:
            return

        w = self.active_window
        rois_in_group = [
            r for r in w.rois if getattr(r, "group", "Default") == group_name
        ]

        data = {
            "format": "pyvistra_roi_group",
            "version": 1,
            "group": group_name,
            "rois": [roi.to_dict() for roi in rois_in_group],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_all_rois(self):
        default_name = f"{self._get_base_name()}.json"
        default_path = os.path.join(self._get_default_dir(), default_name)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save All ROIs", default_path, "JSON Files (*.json)"
        )
        if not path:
            return

        w = self.active_window
        roi_groups = w.get_roi_groups()

        data = {
            "format": "pyvistra_annotations",
            "version": 1,
            "roi_groups": {},
        }
        for gname, rois in roi_groups.items():
            if rois:
                data["roi_groups"][gname] = [r.to_dict() for r in rois]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_mask_layer(self, layer_name):
        default_name = f"{self._get_base_name()}_{layer_name}.sparse.zarr"
        default_path = os.path.join(self._get_default_dir(), default_name)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask Layer",
            default_path,
            "Zarr Labels (*.sparse.zarr);;NPZ Labels (*.sparse.npz);;TIFF Image (*.tif)",
        )
        if not path:
            return
        if not (
            path.endswith(".sparse.zarr")
            or path.endswith(".sparse.npz")
            or path.endswith(".tif")
            or path.endswith(".tiff")
        ):
            path += ".sparse.zarr"

        w = self.active_window
        if layer_name in w._mask_layers:
            labels = w._mask_layers[layer_name]["labels"]
            if labels and labels.n_objects > 0:
                if path.endswith((".tif", ".tiff")):
                    import tifffile

                    dense = labels.to_dense(dtype=np.uint16)
                    tifffile.imwrite(path, dense)
                else:
                    labels.save(path)

    def load_rois(self):
        """Load ROIs from JSON (backward compatible with old format)."""
        if not self.active_window:
            return

        default_name = f"{self._get_base_name()}.json"
        default_path = os.path.join(self._get_default_dir(), default_name)

        path, _ = QFileDialog.getOpenFileName(
            self, "Load ROIs", default_path, "JSON Files (*.json)"
        )
        if not path:
            return

        with open(path, "r") as f:
            data = json.load(f)

        w = self.active_window

        # Detect format
        if isinstance(data, list):
            # Old format: plain array of ROI dicts → "Default" group
            self._load_roi_list(data, "Default")
        elif isinstance(data, dict):
            fmt = data.get("format", "")
            if fmt == "pyvistra_roi_group":
                group = data.get("group", "Default")
                w.add_roi_group(group)
                self._load_roi_list(data.get("rois", []), group)
            elif fmt == "pyvistra_annotations":
                for gname, roi_list in data.get("roi_groups", {}).items():
                    w.add_roi_group(gname)
                    self._load_roi_list(roi_list, gname)
            else:
                # Unknown dict format, try treating as old format
                self._load_roi_list([data], "Default")

        self.refresh_list()
        w.canvas.update()

    def _load_roi_list(self, roi_list, group_name):
        """Load a list of ROI dicts into a specific group."""
        w = self.active_window
        cls_map = {
            "CoordinateROI": CoordinateROI,
            "RectangleROI": RectangleROI,
            "CircleROI": CircleROI,
            "LineROI": LineROI,
            "LaneROI": LaneROI,
        }

        for item in roi_list:
            cls_name = item.get("type", "")
            cls = cls_map.get(cls_name)
            if cls is None:
                continue

            roi = cls(w.view, name=item.get("name", ""))
            roi.group = item.get("group", group_name)
            roi.from_dict(item.get("data", {}))
            w.rois.append(roi)
            w.roi_added.emit(roi)

    def load_mask_zarr(self):
        if not self.active_window:
            return
        path = QFileDialog.getExistingDirectory(
            self, "Select .sparse.zarr Folder", self._get_default_dir()
        )
        if not path or not path.endswith(".sparse.zarr"):
            return
        self._load_mask_from_path(path)

    def load_mask_npz(self):
        if not self.active_window:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load NPZ Labels",
            self._get_default_dir(),
            "NPZ Labels (*.sparse.npz);;All Files (*)",
        )
        if not path:
            return
        self._load_mask_from_path(path)

    def load_mask_from_dense(self):
        if not self.active_window:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Dense Label Image",
            self._get_default_dir(),
            "Image Files (*.tif *.tiff *.png);;All Files (*)",
        )
        if not path:
            return
        try:
            import numpy as np
            from tifffile import imread

            if path.endswith((".tif", ".tiff")):
                dense_image = imread(path)
            else:
                from matplotlib.pyplot import imread as mpl_imread

                dense_image = mpl_imread(path)
                if dense_image.ndim == 3:
                    dense_image = dense_image[:, :, 0]
                dense_image = dense_image.astype(np.uint16)

            labels = SparseLabels.from_dense(dense_image)
            self._apply_loaded_mask(labels)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    def load_tracks(self):
        """Load tracks from CSV/JSON and apply to selected or new track layer."""
        if not self.active_window:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Tracks",
            self._get_default_dir(),
            "Track Tables (*.csv *.json);;CSV Files (*.csv);;JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                tracks = self._load_tracks_from_csv(path)
            elif path.lower().endswith(".json"):
                tracks = self._load_tracks_from_json(path)
            else:
                raise ValueError(
                    "Unsupported track format. Use .csv or .json."
                )
            self._apply_loaded_tracks(tracks)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    def _load_tracks_from_csv(self, path):
        required = {"track_id", "t", "x", "y"}
        track_id = []
        t = []
        x = []
        y = []
        z = []
        has_z = False

        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row")
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Missing required columns: {', '.join(sorted(missing))}"
                )
            has_z = "z" in reader.fieldnames
            for row in reader:
                track_id.append(int(row["track_id"]))
                t.append(int(float(row["t"])))
                x.append(float(row["x"]))
                y.append(float(row["y"]))
                if has_z:
                    z_val = row.get("z", "")
                    z.append(float(z_val) if z_val != "" else 0.0)

        return TrackTable.from_arrays(
            track_id=track_id,
            t=t,
            x=x,
            y=y,
            z=z if has_z else None,
        )

    def _load_tracks_from_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            rows = data.get("tracks", data.get("rows", None))
            if rows is None:
                # Accept dict-of-columns style
                if {"track_id", "t", "x", "y"}.issubset(data.keys()):
                    return TrackTable.from_arrays(
                        track_id=data["track_id"],
                        t=data["t"],
                        x=data["x"],
                        y=data["y"],
                        z=data.get("z"),
                    )
                raise ValueError(
                    "JSON must be a list of rows or contain a 'tracks'/'rows' list"
                )
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError("Invalid JSON track format")

        if not rows:
            return TrackTable.from_arrays(track_id=[], t=[], x=[], y=[])

        required = {"track_id", "t", "x", "y"}
        for key in required:
            if key not in rows[0]:
                raise ValueError(
                    f"Missing required key '{key}' in JSON row records"
                )

        track_id = [int(r["track_id"]) for r in rows]
        t = [int(float(r["t"])) for r in rows]
        x = [float(r["x"]) for r in rows]
        y = [float(r["y"]) for r in rows]
        use_z = any("z" in r for r in rows)
        z = [float(r.get("z", 0.0)) for r in rows] if use_z else None

        return TrackTable.from_arrays(
            track_id=track_id,
            t=t,
            x=x,
            y=y,
            z=z,
        )

    def _apply_loaded_tracks(self, tracks):
        """Apply loaded tracks to selected track layer or create a new layer."""
        w = self.active_window
        if self._selected_group and self._selected_group[0] == "track":
            _, gname = self._selected_group
            if gname in w._track_layers:
                w.set_track_layer_tracks(gname, tracks)
                self.refresh_list()
                w.canvas.update()
                return

        existing = set(w._track_layers.keys())
        idx = 1
        while f"Tracks-{idx}" in existing:
            idx += 1
        name = f"Tracks-{idx}"
        w.add_track_layer(name, tracks=tracks)
        self._selected_group = ("track", name)
        self.refresh_list()
        w.canvas.update()

    def _load_mask_from_path(self, path):
        try:
            labels = SparseLabels.from_file(path)
            self._apply_loaded_mask(labels)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    def _apply_loaded_mask(self, labels):
        """Apply loaded labels to active mask layer or create new one."""
        w = self.active_window
        Y, X = w.Y, w.X
        expected = (w.Z, Y, X) if w.Z > 1 else (Y, X)

        if labels.shape != expected:
            QMessageBox.warning(
                self,
                "Shape Mismatch",
                f"Label shape {labels.shape} does not match "
                f"image shape {expected}",
            )
            return

        if self._selected_group and self._selected_group[0] == "mask":
            _, gname = self._selected_group
            if gname in w._mask_layers:
                w._mask_layers[gname]["labels"] = labels
                visual = w._mask_layers[gname]["visual"]
                if visual:
                    visual.set_labels(labels)
                w.label_changed.emit(labels)
                self.refresh_list()
                w.canvas.update()
                return

        # Create new mask layer
        existing = set(w._mask_layers.keys())
        idx = 1
        while f"Labels-{idx}" in existing:
            idx += 1
        w.add_mask_layer(f"Labels-{idx}", labels=labels)
        self.refresh_list()
        w.canvas.update()

    # ---- Analysis ----

    def _run_analysis(self, func):
        from .analysis import crop_image

        item = self.item_list.currentItem()
        if not item or not self.active_window:
            print("No item selected")
            return

        data = item.data(Qt.UserRole)
        if not data or data[0] != "roi":
            print("Select an ROI first")
            return

        roi = data[1]
        data_5d = self.active_window.img_data

        if func == crop_image:
            func(data_5d, roi)
        else:
            cache = self.active_window.renderer.current_slice_cache
            if cache is None:
                print("No image data available")
                return
            if cache.ndim == 3:
                c = self.active_window.c_idx
                data_2d = cache[c] if c < cache.shape[0] else cache[0]
            else:
                data_2d = cache
            func(data_2d, roi)

    def _show_line_profile_dialog(self):
        from .widgets import get_line_profile_dialog

        dialog = get_line_profile_dialog()
        if self.active_window:
            dialog.active_window = self.active_window
        dialog.show()
        dialog.raise_()
        if self.active_window:
            for roi in self.active_window.rois:
                if roi.selected and isinstance(roi, LineROI):
                    dialog._update_profile(roi)
                    break

    def _align_lanes_action(self):
        from .analysis import align_lanes

        if not self.active_window:
            return
        count = align_lanes(self.active_window.rois, reference="top")
        if count > 0:
            self.active_window.canvas.update()

    # ---- Keyboard Shortcuts ----

    def _flip_selected_coordinate_roi(self):
        item = self.item_list.currentItem()
        if not item or not self.active_window:
            return
        data = item.data(Qt.UserRole)
        if data and data[0] == "roi" and isinstance(data[1], CoordinateROI):
            data[1].flip()
            self.active_window.canvas.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.delete_selected_item()
        elif event.key() == Qt.Key_F:
            self._flip_selected_coordinate_roi()
        else:
            super().keyPressEvent(event)


# Global singleton
_annotation_manager_instance = None


def get_annotation_manager():
    global _annotation_manager_instance
    if _annotation_manager_instance is None:
        _annotation_manager_instance = AnnotationManager()
    return _annotation_manager_instance


def annotation_manager_exists():
    """Check if AnnotationManager singleton has been created."""
    return _annotation_manager_instance is not None
