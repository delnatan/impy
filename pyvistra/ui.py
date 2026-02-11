import heapq
import sys
from collections import OrderedDict

import numpy as np
from qtpy import API_NAME
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider
from vispy import app, scene

from .io import Imaris5DProxy, Numpy5DProxy, load_image, normalize_to_5d
from .label_visual import LabelOverlayVisual
from .labels import SparseLabels
from .manager import manager
from .ortho import OrthoViewer
from .rois import CircleROI, CoordinateROI, LaneROI, LineROI, RectangleROI
from .visuals import CompositeImageVisual
from .widgets import (
    AlignmentDialog,
    AxesDialog,
    ChannelPanel,
    ContrastDialog,
    MetadataDialog,
    TransformDialog,
)

try:
    app.use_app(API_NAME)
except Exception:
    app.use_app("pyqt5")


class ImageWindow(QMainWindow):
    """Main image viewer window with ROI support."""

    # Signals for decoupled communication
    window_activated = Signal(object)  # Emits self when window becomes active
    window_shown = Signal(object)  # Emits self when window is shown
    window_closing = Signal(object)  # Emits self when window is closing
    roi_added = Signal(object)  # Emits the ROI that was added
    roi_removed = Signal(object)  # Emits the ROI that was removed
    roi_selection_changed = Signal(object)  # Emits the selected ROI (or None)
    roi_modified = Signal(object)  # Emits ROI when it's being modified (dragged)
    label_changed = Signal(object)  # Emits SparseLabels when labels change
    mask_layer_added = Signal(str)  # Emits mask layer name when added
    mask_layer_removed = Signal(str)  # Emits mask layer name when removed

    def __init__(self, data_or_path, title="Image", meta=None):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)

        # 1. Load/Set Data
        if isinstance(data_or_path, str):
            self.filepath = data_or_path
            self.img_data, self.meta = load_image(self.filepath)
            filename = self.meta.get("filename", "Image")
        else:
            self.filepath = None
            self.meta = meta or {}

            # Accept any 5D proxy-like object (Imaris5DProxy, Numpy5DProxy, etc.)
            if isinstance(data_or_path, (Imaris5DProxy, Numpy5DProxy)):
                self.img_data = data_or_path
            elif (
                hasattr(data_or_path, "shape")
                and hasattr(data_or_path, "ndim")
                and data_or_path.ndim == 5
            ):
                # Generic 5D proxy-like object
                self.img_data = data_or_path
            elif isinstance(data_or_path, np.ndarray):
                self.img_data = normalize_to_5d(data_or_path)
            else:
                raise ValueError(
                    "data must be a 5D proxy, numpy array, or filepath string"
                )

            filename = self.meta.get("filename", title)

        # Register with Manager
        self.window_id = manager.register(self)

        self.T, self.Z, self.C, self.Y, self.X = self.img_data.shape

        # Title
        sz, sy, sx = self.meta.get("scale", (1.0, 1.0, 1.0))
        title_str = f"[{self.window_id}] {filename} "
        title_str += f"[{self.X}x{self.Y} px] "
        if self.filepath:
            title_str += f"[{sx:.2f} x {sy:.2f} \u00b5m]"
        self.setWindowTitle(title_str)
        self.resize(700, 750)  # Taller for extra controls

        # 2. Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # 3. Vispy Canvas
        self.canvas = scene.SceneCanvas(keys=None, bgcolor="black", show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"
        self.view.camera.aspect = 1

        self.layout.addWidget(self.canvas.native, 1)

        # 4. Info Bar
        self.info_label = QLabel("Hover over image")
        self.info_label.setStyleSheet(
            "background-color: #333; color: #EEE; padding: 4px;"
        )
        self.info_label.setFixedHeight(25)
        self.layout.addWidget(self.info_label, 0)

        # 5. Visuals
        is_rgb = self.meta.get("is_rgb", False)
        self.renderer = CompositeImageVisual(
            self.view, self.img_data, is_rgb=is_rgb
        )
        self.renderer.reset_camera(self.img_data.shape)

        # 6. Controls Area (Sliders + Mode)
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(10, 10, 10, 10)
        self.controls_layout.setSpacing(5)
        self.layout.addWidget(self.controls_widget, 0)

        self.t_idx = 0
        self.z_idx = 0
        self.c_idx = 0  # Active channel index for Single mode

        self._setup_controls()

        # 7. Menu & Dialogs
        self.contrast_dialog = None
        self.channel_panel = None
        self.transform_dialog = None
        self._alignment_dialog = None  # Shared singleton
        self._setup_menu()

        # 8. ROI State
        self.rois = []
        self._next_roi_id = 0
        self._freed_roi_ids = []  # min-heap of freed IDs for reuse
        self.drawing_roi = None
        self.start_pos = None
        # ROI grouping
        self._roi_groups_visible = {"Default": True}
        self._active_roi_group = "Default"
        # Editing State
        self.dragging_roi = None
        self.drag_handle = None
        self.last_pos = None
        # SPACE bar temporary pointer mode
        self._space_held_previous_tool = None

        # 9. Label/Mask State (multiple named mask layers)
        self._mask_layers = OrderedDict()  # name → {"labels": SparseLabels, "visual": LabelOverlayVisual, "visible": True}
        self._active_mask_layer = None  # name of active mask layer
        self._active_label = 1
        self._brush_size = 5
        self._preserve_labels = True  # Protect existing labels by default
        self._painting = False
        self._contour_mode = False  # Right-click contour fill mode
        self._contour_start = None  # (x, y) of contour start
        self._contour_marker = None  # Visual marker for start point
        self._stroke_points = []  # Accumulated path for contour
        self._contour_max_dist = 0.0  # Max distance from contour start

        # 10. Events
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
        self.canvas.events.key_press.connect(self._on_vispy_key_press)

        # Focus policy
        self.setFocusPolicy(Qt.StrongFocus)

        # Initial Draw
        self.update_view()

    def showEvent(self, event):
        super().showEvent(event)
        self.window_shown.emit(self)

    def closeEvent(self, event):
        manager.unregister(self)
        self.window_closing.emit(self)

        # Cleanup data buffers/proxies (ImageBuffer, Imaris5DProxy)
        if hasattr(self.img_data, "release"):
            try:
                self.img_data.release()
            except Exception:
                pass
        elif hasattr(self.img_data, "close"):
            try:
                self.img_data.close()
            except Exception:
                pass

        super().closeEvent(event)

    def focusInEvent(self, event):
        self.window_activated.emit(self)
        super().focusInEvent(event)

    def _on_vispy_key_press(self, event):
        """Block certain keys from reaching VisPy's camera handles"""
        if event.key == "Backspace":
            # prevent PanZoom camera from resetting view on Backspace
            event.handled = True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.renderer.reset_camera(self.img_data.shape)
            self.canvas.update()
        elif event.key() == Qt.Key_F:
            # Flip selected CoordinateROI
            for roi in self.rois:
                if roi.selected and isinstance(roi, CoordinateROI):
                    roi.flip()
                    self.canvas.update()
                    break
        elif event.key() == Qt.Key_L:
            # Toggle ROI labels visibility
            from .rois import ROI

            show = ROI.toggle_labels()
            # Update visibility for all ROIs in all windows
            for w in manager.get_all().values():
                for roi in w.rois:
                    roi.label_visual.visible = show
                w.canvas.update()
        elif event.key() == Qt.Key_Escape:
            # Cancel contour mode if active
            if self._contour_mode:
                self._cancel_contour()
                return
            # Deselect all ROIs
            for roi in self.rois:
                roi.select(False)
            self.canvas.update()
            self.roi_selection_changed.emit(None)
        elif event.key() == Qt.Key_Space:
            # Temporarily switch to pointer mode for panning/zooming
            if self._space_held_previous_tool is None:
                self._space_held_previous_tool = manager.active_tool
                manager.active_tool = "pointer"
                self.update_cursor()
        # Label/Mask shortcuts
        elif event.key() == Qt.Key_B:
            # Brush tool
            manager.active_tool = "brush"
            self.update_cursor()
        elif event.key() == Qt.Key_E:
            # Eraser tool
            manager.active_tool = "eraser"
            self.update_cursor()
        elif event.key() == Qt.Key_N:
            # New label (next available ID)
            if self.labels is not None:
                existing = set(self.labels.labels)
                new_id = 1
                while new_id in existing:
                    new_id += 1
                self.active_label = new_id
        elif event.key() == Qt.Key_P:
            # Toggle preserve labels
            self.preserve_labels = not self.preserve_labels
        elif event.key() == Qt.Key_V:
            # Toggle label overlay visibility
            if self.label_overlay is not None:
                self.label_overlay.visible = not self.label_overlay.visible
                self.canvas.update()
        elif event.key() == Qt.Key_BracketLeft:
            # Decrease brush size
            self.brush_size = max(1, self.brush_size - 1)
        elif event.key() == Qt.Key_BracketRight:
            # Increase brush size
            self.brush_size = min(100, self.brush_size + 1)
        elif event.key() in (
            Qt.Key_1,
            Qt.Key_2,
            Qt.Key_3,
            Qt.Key_4,
            Qt.Key_5,
            Qt.Key_6,
            Qt.Key_7,
            Qt.Key_8,
            Qt.Key_9,
        ):
            # Quick label selection 1-9
            label_num = event.key() - Qt.Key_0
            self.active_label = label_num
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            # Restore previous tool when SPACE is released
            if self._space_held_previous_tool is not None:
                manager.active_tool = self._space_held_previous_tool
                self._space_held_previous_tool = None
                self.update_cursor()
        else:
            super().keyReleaseEvent(event)

    def get_data(self):
        """Return the current image data."""
        return self.img_data

    def set_data(self, new_data, metadata=None):
        """Update the image data in place.

        Args:
            new_data: 5D array-like (T, Z, C, Y, X)
            metadata: Optional dict to replace window's metadata entirely.
                      If provided, replaces self.meta.
        """
        if new_data.ndim != 5:
            # Try to reshape or warn? For now assume 5D or compatible
            pass

        self.img_data = new_data
        # Update renderer data
        self.renderer.data = new_data

        # Update metadata if provided
        if metadata is not None:
            self.meta = metadata

        self.renderer.update_slice(self.t_idx, self.z_idx)
        self.canvas.update()

    def _setup_menu(self):
        menubar = self.menuBar()

        # Adjust Menu
        adjust_menu = menubar.addMenu("Adjust")
        bc_action = QAction("Brightness/Contrast", self)
        bc_action.setShortcut("Shift+C")
        bc_action.triggered.connect(self.show_contrast_dialog)
        adjust_menu.addAction(bc_action)

        channels_action = QAction("Channels...", self)
        channels_action.setShortcut("Shift+H")
        channels_action.triggered.connect(self.show_channel_panel)
        adjust_menu.addAction(channels_action)

        adjust_menu.addSeparator()

        line_profile_action = QAction("Line Profile...", self)
        line_profile_action.setShortcut("Shift+K")
        line_profile_action.triggered.connect(self.show_line_profile)
        adjust_menu.addAction(line_profile_action)

        # Image Menu
        image_menu = menubar.addMenu("Image")
        info_action = QAction("Image Info", self)
        info_action.setShortcut("Shift+I")
        info_action.triggered.connect(self.show_metadata_dialog)
        image_menu.addAction(info_action)

        ortho_action = QAction("Ortho View", self)
        ortho_action.triggered.connect(self.show_ortho_view)
        image_menu.addAction(ortho_action)

        volume_action = QAction("3D Volume View", self)
        volume_action.triggered.connect(self.show_volume_view)
        image_menu.addAction(volume_action)

        image_menu.addSeparator()

        transform_action = QAction("Transform...", self)
        transform_action.setShortcut("Shift+T")
        transform_action.triggered.connect(self.show_transform_dialog)
        image_menu.addAction(transform_action)

        align_action = QAction("Align Images...", self)
        align_action.triggered.connect(self.show_alignment_dialog)
        image_menu.addAction(align_action)

        image_menu.addSeparator()

        axes_action = QAction("Reorder Axes...", self)
        axes_action.triggered.connect(self.show_axes_dialog)
        image_menu.addAction(axes_action)

    # ---- ROI ID Management ----

    def _get_next_roi_id(self):
        """Get next available ROI ID, reusing freed IDs when possible."""
        if self._freed_roi_ids:
            return heapq.heappop(self._freed_roi_ids)
        roi_id = self._next_roi_id
        self._next_roi_id += 1
        return roi_id

    def _free_roi_id(self, roi):
        """Return an ROI's ID to the pool for reuse."""
        try:
            heapq.heappush(self._freed_roi_ids, int(roi.name))
        except ValueError:
            pass  # Non-numeric name, ignore

    def remove_roi(self, roi):
        """Remove an ROI from this window, freeing its ID."""
        if roi in self.rois:
            self._free_roi_id(roi)
            roi.remove()
            self.rois.remove(roi)
            self.roi_removed.emit(roi)

    # ---- Label/Mask Methods ----

    # ---- Mask Layer Properties (backward compat) ----

    @property
    def labels(self):
        """Active mask layer's SparseLabels (backward compat)."""
        if self._active_mask_layer and self._active_mask_layer in self._mask_layers:
            return self._mask_layers[self._active_mask_layer]["labels"]
        return None

    @labels.setter
    def labels(self, value):
        """Set labels on active mask layer, creating one if needed."""
        if value is None:
            return
        if not self._mask_layers:
            self.add_mask_layer("Labels-1", labels=value)
        elif self._active_mask_layer and self._active_mask_layer in self._mask_layers:
            self._mask_layers[self._active_mask_layer]["labels"] = value
            visual = self._mask_layers[self._active_mask_layer]["visual"]
            if visual is not None:
                visual.set_labels(value)

    @property
    def label_overlay(self):
        """Active mask layer's visual (backward compat)."""
        if self._active_mask_layer and self._active_mask_layer in self._mask_layers:
            return self._mask_layers[self._active_mask_layer]["visual"]
        return None

    def set_labels(self, labels):
        """Set SparseLabels for this window (backward compat)."""
        if labels is None:
            return
        if not self._mask_layers:
            self.add_mask_layer("Labels-1", labels=labels)
        else:
            self.labels = labels
            self._ensure_label_overlay()
        self.label_changed.emit(labels)
        self.canvas.update()

    def get_labels(self):
        """Return the current SparseLabels instance."""
        return self.labels

    def _ensure_label_overlay(self):
        """Ensure a mask layer with visual exists. Auto-creates 'Labels-1' if none."""
        if not self._mask_layers:
            self.add_mask_layer("Labels-1")
            return
        if self._active_mask_layer and self._active_mask_layer in self._mask_layers:
            entry = self._mask_layers[self._active_mask_layer]
            if entry["visual"] is None:
                entry["visual"] = LabelOverlayVisual(
                    self.view,
                    shape_yx=(self.Y, self.X),
                    scale=self.renderer.scale,
                )
            if entry["labels"] is not None:
                entry["visual"].set_labels(entry["labels"])

    # ---- Multiple Mask Layers API ----

    def add_mask_layer(self, name, labels=None):
        """Create a named mask layer with its own SparseLabels and visual."""
        if name in self._mask_layers:
            return
        if labels is None:
            if self.Z > 1:
                shape = (self.Z, self.Y, self.X)
            else:
                shape = (self.Y, self.X)
            labels = SparseLabels(shape)
        visual = LabelOverlayVisual(
            self.view,
            shape_yx=(self.Y, self.X),
            scale=self.renderer.scale,
        )
        visual.set_labels(labels)
        self._mask_layers[name] = {
            "labels": labels,
            "visual": visual,
            "visible": True,
        }
        if self._active_mask_layer is None:
            self._active_mask_layer = name
        self.mask_layer_added.emit(name)

    def remove_mask_layer(self, name):
        """Remove a named mask layer and clean up its visual."""
        if name not in self._mask_layers:
            return
        entry = self._mask_layers.pop(name)
        if entry["visual"] is not None:
            entry["visual"].remove()
        if self._active_mask_layer == name:
            self._active_mask_layer = next(iter(self._mask_layers), None)
        self.mask_layer_removed.emit(name)
        self.canvas.update()

    def set_mask_layer_visible(self, name, visible):
        """Toggle visibility of a specific mask layer."""
        if name not in self._mask_layers:
            return
        self._mask_layers[name]["visible"] = visible
        visual = self._mask_layers[name]["visual"]
        if visual is not None:
            visual.visible = visible
        self.canvas.update()

    def set_active_mask_layer(self, name):
        """Switch which mask layer receives painting."""
        if name in self._mask_layers:
            self._active_mask_layer = name

    def _get_brush_coords(self, cx, cy):
        """
        Generate circular brush mask centered at (cx, cy).

        Returns:
            Tuple of (y_coords, x_coords) arrays
        """
        radius = self.brush_size
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        mask = xx**2 + yy**2 <= radius**2
        ys, xs = np.where(mask)

        # Offset to image coordinates
        y_coords = ys + int(cy) - radius
        x_coords = xs + int(cx) - radius

        # Bounds check
        valid = (
            (y_coords >= 0)
            & (y_coords < self.Y)
            & (x_coords >= 0)
            & (x_coords < self.X)
        )

        return y_coords[valid], x_coords[valid]

    def _filter_existing_labels(self, coords):
        """
        Remove coordinates that belong to any existing label.

        Args:
            coords: Tuple of (y, x) or (z, y, x) coordinate arrays

        Returns:
            Filtered coordinate tuple with occupied pixels removed
        """
        if self.labels is None or self.labels.n_objects == 0:
            return coords

        # Build occupancy mask for current slice
        occupied = np.zeros((self.Y, self.X), dtype=bool)

        for label in self.labels:
            label_coords = self.labels.coords(label)
            if self.labels.ndim == 3:
                # Filter to current Z slice
                z_coords, y_coords, x_coords = label_coords
                z_mask = z_coords == self.z_idx
                y_slice = y_coords[z_mask]
                x_slice = x_coords[z_mask]
            else:
                y_slice, x_slice = label_coords

            # Bounds check before setting
            valid = (
                (y_slice >= 0)
                & (y_slice < self.Y)
                & (x_slice >= 0)
                & (x_slice < self.X)
            )
            occupied[y_slice[valid], x_slice[valid]] = True

        # Filter out occupied pixels
        if len(coords) == 3:
            # 3D coords
            zs, ys, xs = coords
            free = ~occupied[ys, xs]
            return (zs[free], ys[free], xs[free])
        else:
            # 2D coords
            ys, xs = coords
            free = ~occupied[ys, xs]
            return (ys[free], xs[free])

    def _paint_stroke(self, x, y, erase=False):
        """
        Paint or erase at the given position.

        Args:
            x, y: Image coordinates
            erase: If True, erase pixels instead of adding
        """
        if self.labels is None:
            return

        brush_coords = self._get_brush_coords(x, y)
        if len(brush_coords[0]) == 0:
            return

        # Convert to proper dimensionality
        if self.labels.ndim == 3:
            z_coords = np.full(len(brush_coords[0]), self.z_idx, dtype=np.int32)
            coords = (z_coords, brush_coords[0], brush_coords[1])
        else:
            coords = brush_coords

        if erase:
            # Remove from active label
            if self.active_label in self.labels:
                try:
                    self.labels.remove_pixels(self.active_label, coords)
                except KeyError:
                    pass  # Label was completely erased
        else:
            # Add to active label
            if self.preserve_labels:
                coords = self._filter_existing_labels(coords)
                if len(coords[0]) == 0:
                    return

            self.labels.add_pixels(self.active_label, coords)

        # Update visuals
        if self.label_overlay:
            self.label_overlay.refresh()
        self.canvas.update()

    def _show_contour_marker(self, x, y):
        """Show visual marker at contour start point."""
        if self._contour_marker is None:
            self._contour_marker = scene.visuals.Markers(parent=self.view.scene)

        # Green circle marker at start point
        self._contour_marker.set_data(
            pos=np.array([[x, y]]),
            face_color=(0.2, 1.0, 0.2, 0.8),
            edge_color=(1.0, 1.0, 1.0, 1.0),
            edge_width=2,
            size=max(15, self.brush_size * 2),
            symbol="o",
        )
        self._contour_marker.visible = True
        self.canvas.update()

    def _hide_contour_marker(self):
        """Hide the contour start marker."""
        if self._contour_marker is not None:
            self._contour_marker.visible = False
            self.canvas.update()

    def _finish_contour(self):
        """Finalize contour mode; fill if closed near start."""
        if not self._contour_mode or self._contour_start is None:
            self._cancel_contour()
            return

        if len(self._stroke_points) < 10:
            self._cancel_contour()
            return

        # Check if we closed near the start
        end = np.array(self._stroke_points[-1])
        start = np.array(self._contour_start)
        distance = np.linalg.norm(end - start)

        # Threshold: close if within 3x brush radius or 15 pixels
        threshold = max(self.brush_size * 3, 15)
        if distance < threshold:
            self._fill_closed_contour()

        self._cancel_contour()
        if self.labels:
            self.label_changed.emit(self.labels)

    def _cancel_contour(self):
        """Cancel contour mode without filling."""
        self._contour_mode = False
        self._contour_start = None
        self._stroke_points = []
        self._contour_max_dist = 0.0
        self._hide_contour_marker()

    def _finish_stroke(self):
        """Finalize a regular brush stroke (no auto-fill)."""
        self._stroke_points = []
        if self.labels:
            self.label_changed.emit(self.labels)

    def _fill_closed_contour(self):
        """Fill interior of closed stroke path."""
        if self.labels is None:
            return

        try:
            from skimage.draw import polygon, line
        except ImportError:
            return

        # Close the contour by appending the start point
        if self._stroke_points:
            start = self._stroke_points[0]
            self._stroke_points.append(start)

            # Paint the connecting segment from last point back to start
            if len(self._stroke_points) >= 2:
                last = self._stroke_points[-2]
                r0, c0 = int(last[1]), int(last[0])
                r1, c1 = int(start[1]), int(start[0])
                rr_line, cc_line = line(r0, c0, r1, c1)
                for r, c in zip(rr_line, cc_line):
                    self._paint_stroke(c, r, erase=False)

        # Extract contour coordinates
        ys = np.array([p[1] for p in self._stroke_points])
        xs = np.array([p[0] for p in self._stroke_points])

        # Get all interior pixels
        rr, cc = polygon(ys, xs, shape=(self.Y, self.X))

        if len(rr) == 0:
            return

        # Convert to proper dimensionality
        if self.labels.ndim == 3:
            z_coords = np.full(len(rr), self.z_idx, dtype=np.int32)
            fill_coords = (z_coords, rr, cc)
        else:
            fill_coords = (rr, cc)

        if self.preserve_labels:
            fill_coords = self._filter_existing_labels(fill_coords)
            if len(fill_coords[0]) == 0:
                return

        self.labels.add_pixels(self.active_label, fill_coords)

        # Update visuals
        if self.label_overlay:
            self.label_overlay.refresh()
        self.canvas.update()

    @property
    def active_label(self) -> int:
        return self._active_label

    @active_label.setter
    def active_label(self, value: int):
        self._active_label = value

    @property
    def brush_size(self) -> int:
        return self._brush_size

    @brush_size.setter
    def brush_size(self, value: int):
        self._brush_size = value

    @property
    def preserve_labels(self) -> bool:
        return self._preserve_labels

    @preserve_labels.setter
    def preserve_labels(self, value: bool):
        self._preserve_labels = value

    # ---- ROI Grouping ----

    def get_roi_groups(self):
        """Return dict mapping group name → list of ROIs."""
        groups = {}
        for roi in self.rois:
            g = getattr(roi, "group", "Default")
            groups.setdefault(g, []).append(roi)
        # Include empty groups that exist in visibility dict
        for name in self._roi_groups_visible:
            if name not in groups:
                groups[name] = []
        return groups

    def set_group_visible(self, name, visible):
        """Toggle visibility of all ROIs in a group."""
        self._roi_groups_visible[name] = visible
        for roi in self.rois:
            if getattr(roi, "group", "Default") == name:
                roi.set_visible(visible)
        self.canvas.update()

    def add_roi_group(self, name):
        """Create a new empty ROI group."""
        if name not in self._roi_groups_visible:
            self._roi_groups_visible[name] = True

    def remove_roi_group(self, name):
        """Delete a group and all its ROIs."""
        rois_to_remove = [
            r for r in self.rois if getattr(r, "group", "Default") == name
        ]
        for roi in rois_to_remove:
            self.remove_roi(roi)
        self._roi_groups_visible.pop(name, None)
        if self._active_roi_group == name:
            self._active_roi_group = "Default"
            self._roi_groups_visible.setdefault("Default", True)

    def show_metadata_dialog(self):
        dlg = MetadataDialog(self.meta, parent=self)
        dlg.exec_()

    def show_ortho_view(self):
        # Copy colormap settings from current renderer
        colormaps = {}
        for c in range(self.renderer.num_channels):
            colormaps[c] = self.renderer.get_colormap_name(c)

        # Acquire reference if proxy supports ref counting (for shared file handles)
        data = self.img_data
        if hasattr(data, "acquire"):
            data = data.acquire()

        self.ortho_viewer = OrthoViewer(
            data,
            self.meta,
            title=f"Ortho View - {self.windowTitle()}",
            channel_colormaps=colormaps,
        )
        self.ortho_viewer.show()

    def show_volume_view(self):
        """Open 3D volume rendering view."""
        from .volume import VolumeViewer

        # Acquire reference if proxy supports ref counting (for shared HDF5 files)
        data = self.img_data
        if hasattr(data, "acquire"):
            data = data.acquire()

        self.volume_viewer = VolumeViewer(
            data,
            self.meta,
            title=f"3D Volume - {self.windowTitle()}",
            channel=self.c_idx if hasattr(self, "c_idx") else 0,
            time=self.t_idx if hasattr(self, "t_idx") else 0,
        )
        self.volume_viewer.show()

    def update_cursor(self):
        tool = manager.active_tool
        if tool == "pointer":
            self.view.camera.interactive = True
        else:
            self.view.camera.interactive = False

    def show_contrast_dialog(self):
        if self.contrast_dialog is None:
            self.contrast_dialog = ContrastDialog(self, parent=self)
        self.contrast_dialog.show()
        self.contrast_dialog.raise_()
        self.contrast_dialog.refresh_ui()

    def show_channel_panel(self):
        if self.channel_panel is None:
            self.channel_panel = ChannelPanel(self, parent=self)
        self.channel_panel.show()
        self.channel_panel.raise_()
        self.channel_panel.refresh_ui()

    def show_transform_dialog(self):
        if self.transform_dialog is None:
            self.transform_dialog = TransformDialog(self, parent=self)
        self.transform_dialog.show()
        self.transform_dialog.raise_()
        self.transform_dialog.refresh_ui()

    def show_alignment_dialog(self):
        if self._alignment_dialog is None:
            self._alignment_dialog = AlignmentDialog(parent=self)
        self._alignment_dialog.show()
        self._alignment_dialog.raise_()

    def show_line_profile(self):
        """Show the line profile dialog."""
        from .widgets import get_line_profile_dialog

        dialog = get_line_profile_dialog()
        dialog.active_window = self
        dialog.show()
        dialog.raise_()
        # Trigger update if a LineROI is selected
        for roi in self.rois:
            if roi.selected and isinstance(roi, LineROI):
                dialog._update_profile(roi)
                break

    def show_axes_dialog(self):
        """Show dialog to reorder axes for ambiguous TIFF dimensions."""
        raw_shape = self.meta.get("raw_shape")
        if raw_shape is None:
            from qtpy.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Reorder Axes",
                "Axes reordering is only available for TIFF/PNG/JPEG images.",
            )
            return

        dlg = AxesDialog(raw_shape, parent=self)
        if dlg.exec_():
            dims = dlg.get_dims_string()
            self.reorder_axes(dims)

    def reorder_axes(self, dims):
        """
        Reorder axes using a new dimension string.

        Args:
            dims: Dimension string (e.g., 'tyx', 'zyx', 'zcyx', 'tzyx')
        """
        if not self.filepath:
            print("Cannot reorder axes: no source file available")
            return

        # Re-load and re-normalize with new dims
        new_data, new_meta = load_image(self.filepath, dims=dims)

        # Update data and dimensions
        self.img_data = new_data
        self.meta = new_meta
        self.T, self.Z, self.C, self.Y, self.X = self.img_data.shape

        # Reset indices
        self.t_idx = 0
        self.z_idx = 0
        self.c_idx = 0

        # Remove old renderer layers
        for layer in self.renderer.layers:
            layer.parent = None

        # Create new renderer with updated data
        is_rgb = self.meta.get("is_rgb", False)
        self.renderer = CompositeImageVisual(
            self.view, self.img_data, is_rgb=is_rgb
        )
        self.renderer.reset_camera(self.img_data.shape)

        # Rebuild controls
        self._rebuild_controls()

        # Update view
        self.update_view()

    def _rebuild_controls(self):
        """Rebuild control widgets after axes reorder."""
        # Remove all widgets from controls_layout
        while self.controls_layout.count():
            item = self.controls_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Recursively delete layout items
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        # Reset widget references
        self.mode_combo = None
        self.channel_row_widget = None
        self.c_slider = None
        self.z_slider = None
        self.z_label = None
        self.chk_proj = None
        self.z_range_slider = None
        self.z_range_slider_widget = None
        self.z_range_slider_min_label = None
        self.z_range_slider_max_label = None

        # Rebuild controls
        self._setup_controls()

    def set_tool(self, tool_name):
        """
        Set the active tool (e.g. 'pointer', 'rect', 'circle', 'line', 'coordinate').
        """
        valid_tools = ["pointer", "coordinate", "rect", "circle", "line"]
        if tool_name not in valid_tools:
            print(f"Invalid tool: {tool_name}. Valid tools: {valid_tools}")
            return

        manager.active_tool = tool_name

        # Update cursors in all windows
        for w in manager.get_all().values():
            w.update_cursor()

    def _setup_controls(self):
        # -- Mode Selector (Only if Multi-channel) --
        if self.C > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Mode:"))

            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Composite", "Single Channel"])
            self.mode_combo.currentIndexChanged.connect(self.on_mode_change)
            row.addWidget(self.mode_combo)
            row.addStretch()
            self.controls_layout.addLayout(row)

            # -- Channel Slider (Initially Hidden) --
            self.channel_row_widget = QWidget()
            c_layout = QHBoxLayout(self.channel_row_widget)
            c_layout.setContentsMargins(0, 0, 0, 0)

            c_layout.addWidget(QLabel("Channel"))
            self.c_slider = QSlider(Qt.Horizontal)
            self.c_slider.setRange(0, self.C - 1)
            self.c_slider.valueChanged.connect(self.on_channel_change)
            c_layout.addWidget(self.c_slider)

            self.controls_layout.addWidget(self.channel_row_widget)
            self.channel_row_widget.setVisible(False)  # Default is Composite

        # -- Time Slider --
        if self.T > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Time"))
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, self.T - 1)
            sl.valueChanged.connect(self.on_time_change)
            row.addWidget(sl)
            self.controls_layout.addLayout(row)

        # -- Z Slider --
        if self.Z > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Z-Pos"))

            # Standard Slider
            self.z_slider = QSlider(Qt.Horizontal)
            self.z_slider.setRange(0, self.Z - 1)
            self.z_slider.valueChanged.connect(self.on_z_change)
            row.addWidget(self.z_slider)

            self.z_label = QLabel("0")
            self.z_label.setFixedWidth(30)  # Fixed width to prevent jumping
            self.z_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(self.z_label)

            # Projection Controls
            self.chk_proj = QCheckBox("Max Proj")
            self.chk_proj.toggled.connect(self.toggle_z_projection)
            self.z_range_slider_widget = QWidget()
            self.z_range_slider_layout = QHBoxLayout()
            self.z_range_slider_layout.setContentsMargins(0, 0, 0, 0)
            self.z_range_slider = QRangeSlider(Qt.Horizontal)
            self.z_range_slider_min_label = QLabel("0")
            self.z_range_slider_max_label = QLabel(f"{self.Z - 1}")
            self.z_range_slider.setRange(0, self.Z - 1)
            self.z_range_slider.setValue((0, self.Z - 1))
            self.z_range_slider.barIsVisible = True
            self.z_range_slider.barIsEnabled = True
            self.z_range_slider.barIsEnabled = False

            self.z_range_slider_layout.addWidget(self.z_range_slider_min_label)
            self.z_range_slider_layout.addWidget(self.z_range_slider)
            self.z_range_slider_layout.addWidget(self.z_range_slider_max_label)
            self.z_range_slider_widget.setLayout(self.z_range_slider_layout)
            self.z_range_slider_widget.setVisible(False)

            self.z_range_slider.valueChanged.connect(self.on_z_proj_change)
            row.addWidget(self.z_range_slider_widget)
            row.addWidget(self.chk_proj)
            self.controls_layout.addLayout(row)

    def on_mode_change(self, index):
        mode = "composite" if index == 0 else "single"

        # Toggle Channel Slider Visibility
        if self.C > 1:
            self.channel_row_widget.setVisible(mode == "single")

        # Update Renderer
        self.renderer.set_mode(mode)
        self.canvas.update()

    def on_channel_change(self, val):
        self.c_idx = val
        self.renderer.set_active_channel(val)
        self.canvas.update()

        # If Contrast Dialog is open, sync it to this channel
        if self.contrast_dialog and self.contrast_dialog.isVisible():
            self.contrast_dialog.combo.setCurrentIndex(val)
            self.contrast_dialog.refresh_ui()

    def on_time_change(self, val):
        self.t_idx = val
        self.update_view()

    def toggle_z_projection(self, checked):
        self.z_slider.setVisible(not checked)
        self.z_range_slider_widget.setVisible(checked)
        self.update_view()

    def on_z_proj_change(self, val):
        # update z-min/max labels
        self.z_range_slider_min_label.setText(str(val[0]))
        self.z_range_slider_max_label.setText(str(val[1]))
        self.update_view()

    def on_z_change(self, val):
        self.z_idx = val
        if hasattr(self, "z_label") and self.z_label is not None:
            self.z_label.setText(str(val))
        self.update_view()

    def update_view(self):
        if hasattr(self, "chk_proj") and self.chk_proj is not None and self.chk_proj.isChecked():
            mn, mx = self.z_range_slider.value()
            z_slice = slice(mn, mx + 1)
            self.renderer.update_slice(self.t_idx, z_slice)
        else:
            self.renderer.update_slice(self.t_idx, self.z_idx)

        # Update all mask layers for 3D data
        for entry in self._mask_layers.values():
            if entry["visual"] and entry["labels"] and entry["labels"].ndim == 3:
                entry["visual"].update_slice(self.z_idx)

        self.canvas.update()
        if self.contrast_dialog and self.contrast_dialog.isVisible():
            self.contrast_dialog.refresh_ui()
        if self.channel_panel and self.channel_panel.isVisible():
            self.channel_panel.refresh_ui()

    def _map_event_to_image(self, event):
        tr = self.canvas.scene.node_transform(self.renderer.layers[0])
        pos = tr.map(event.pos)
        return pos[0], pos[1]

    def on_mouse_press(self, event):
        tool = manager.active_tool
        x, y = self._map_event_to_image(event)

        # Notify that this window is now active
        self.window_activated.emit(self)

        # Ctrl+Click (or Cmd+Click on Mac): delete label under cursor
        if event.button == 1 and tool in ("pointer", "brush", "eraser"):
            has_ctrl_or_cmd = (
                "Control" in event.modifiers or "Meta" in event.modifiers
            )
            if has_ctrl_or_cmd and self.label_overlay is not None:
                iy, ix = int(y), int(x)
                label_id = self.label_overlay.label_at(iy, ix)
                if label_id > 0 and self.labels is not None:
                    self.labels.remove(label_id)
                    self.label_overlay.refresh()
                    self.label_changed.emit(self.labels)
                    self.canvas.update()
                    return

        if tool == "pointer":
            # Hit Test ROIs (Reverse order to select top-most)
            hit_roi = None
            hit_handle = None

            for roi in reversed(self.rois):
                res = roi.hit_test((x, y))
                if res:
                    hit_roi = roi
                    hit_handle = res
                    break

            # Handle modifier clicks on LaneROIs
            if isinstance(hit_roi, LaneROI):
                # Ctrl+Click (or Cmd+Click on Mac) on a marker: remove it
                has_ctrl_or_cmd = (
                    "Control" in event.modifiers or "Meta" in event.modifiers
                )
                if has_ctrl_or_cmd:
                    if (
                        isinstance(hit_handle, tuple)
                        and hit_handle[0] == "marker"
                    ):
                        marker_idx = hit_handle[1]
                        hit_roi.remove_marker(marker_idx)
                        # Trigger callback for live MW update
                        if hit_roi._on_markers_changed:
                            hit_roi._on_markers_changed()
                        self.canvas.update()
                        return

                # Shift+Click anywhere in lane: add a marker
                # Works on body, center, or even on existing marker position
                if "Shift" in event.modifiers:
                    if hit_handle in ("body", "center") or isinstance(
                        hit_handle, tuple
                    ):
                        x_min, x_max, y_min, y_max = hit_roi._get_bounds()
                        y_local = y - y_min
                        hit_roi.add_marker(y_local)
                        # Trigger callback for live MW update
                        if hit_roi._on_markers_changed:
                            hit_roi._on_markers_changed()
                        self.canvas.update()
                        return

            # Update Selection
            for roi in self.rois:
                roi.select(roi is hit_roi)

            # Notify about selection change
            self.roi_selection_changed.emit(hit_roi)

            if hit_roi:
                self.dragging_roi = hit_roi
                self.drag_handle = hit_handle
                self.last_pos = (x, y)
                # Disable camera panning while dragging ROI
                self.view.camera.interactive = False
                self.canvas.update()
            else:
                self.canvas.update()
            return

        # Handle brush/eraser tools
        if tool in ("brush", "eraser"):
            # Ensure labels and overlay exist
            if self.labels is None:
                if self.Z > 1:
                    shape = (self.Z, self.Y, self.X)
                else:
                    shape = (self.Y, self.X)
                self.labels = SparseLabels(shape)
            self._ensure_label_overlay()

            # Right-click: toggle contour fill mode (brush only)
            if event.button == 2 and tool == "brush":
                if self._contour_mode:
                    # Already in contour mode - cancel it
                    self._cancel_contour()
                else:
                    # Start contour mode
                    self._contour_mode = True
                    self._contour_start = (x, y)
                    self._contour_max_dist = 0.0
                    self._stroke_points = [(x, y)]
                    self._show_contour_marker(x, y)
                    self.view.camera.interactive = False
                    # Initial paint at start
                    self._paint_stroke(x, y, erase=False)
                return

            # Left-click: normal painting (only if not in contour mode)
            if event.button == 1 and not self._contour_mode:
                self._painting = True
                self._stroke_points = [(x, y)]
                self.view.camera.interactive = False
                # Initial paint
                self._paint_stroke(x, y, erase=(tool == "eraser"))
            return

        # Handle shape drawing tools (rect, circle, line, coordinate)
        if tool in ("rect", "circle", "line", "coordinate"):
            self.start_pos = (x, y)

            # Get unique ROI ID (reuses freed IDs via heapq)
            roi_index = str(self._get_next_roi_id())

            if tool == "coordinate":
                self.drawing_roi = CoordinateROI(self.view, name=roi_index)
            elif tool == "rect":
                self.drawing_roi = RectangleROI(self.view, name=roi_index)
            elif tool == "circle":
                self.drawing_roi = CircleROI(self.view, name=roi_index)
            elif tool == "line":
                self.drawing_roi = LineROI(self.view, name=roi_index)

            if self.drawing_roi:
                self.drawing_roi.group = self._active_roi_group
                self.rois.append(self.drawing_roi)
                self.roi_added.emit(self.drawing_roi)
                self.drawing_roi.update((x, y), (x, y))
                self.view.camera.interactive = False
                self.canvas.update()

    def on_mouse_move(self, event):
        # 1. Update Info Label (always)
        if self.renderer.layers:
            x, y = self._map_event_to_image(event)
            ix, iy = int(x), int(y)
            if 0 <= ix < self.X and 0 <= iy < self.Y:
                cache = self.renderer.current_slice_cache
                if cache is not None:
                    vals = []
                    for c in range(cache.shape[0]):
                        try:
                            val = cache[c, iy, ix]
                            vals.append(f"{val:.1f}")
                        except IndexError:
                            pass
                    val_str = ", ".join(vals)
                    info_text = f"X: {ix}  Y: {iy}  Val: [{val_str}]"
                    if self.label_overlay is not None:
                        lid = self.label_overlay.label_at(iy, ix)
                        if lid > 0:
                            info_text += f"  Label: {lid}"
                    self.info_label.setText(info_text)
            else:
                self.info_label.setText("")

        # 2. Contour Fill Mode (no button held - just mouse movement)
        if self._contour_mode:
            x, y = self._map_event_to_image(event)
            self._stroke_points.append((x, y))
            self._paint_stroke(x, y, erase=False)

            # Check if we've returned close to start (auto-close)
            start = np.array(self._contour_start)
            current = np.array([x, y])
            distance = np.linalg.norm(current - start)
            self._contour_max_dist = max(self._contour_max_dist, distance)

            min_travel = 50.0
            threshold = max(self.brush_size * 3, 15)
            if self._contour_max_dist > min_travel and distance < threshold:
                self._finish_contour()
            return

        # 3. Brush/Eraser Painting (left-click drag)
        if self._painting:
            tool = manager.active_tool
            x, y = self._map_event_to_image(event)
            self._stroke_points.append((x, y))
            self._paint_stroke(x, y, erase=(tool == "eraser"))
            return

        # 4. ROI Editing
        if self.dragging_roi and event.button == 1:
            x, y = self._map_event_to_image(event)
            dx = x - self.last_pos[0]
            dy = y - self.last_pos[1]

            if self.drag_handle == "center":
                self.dragging_roi.move((dx, dy))
            else:
                self.dragging_roi.adjust(self.drag_handle, (x, y))

            self.last_pos = (x, y)
            self.roi_modified.emit(self.dragging_roi)
            self.canvas.update()
            return

        # 3. Update Drawing
        if self.drawing_roi and event.button == 1:
            x, y = self._map_event_to_image(event)
            end_pos = (x, y)

            # Shift key constrains LineROI to horizontal/vertical
            if (
                isinstance(self.drawing_roi, LineROI)
                and "Shift" in event.modifiers
            ):
                sx, sy = self.start_pos
                dx = abs(x - sx)
                dy = abs(y - sy)
                if dx > dy:
                    # Horizontal line
                    end_pos = (x, sy)
                else:
                    # Vertical line
                    end_pos = (sx, y)

            self.drawing_roi.update(self.start_pos, end_pos)
            self.canvas.update()

    def on_mouse_release(self, event):
        # Contour mode is handled by mouse movement, not release
        # (auto-closes when returning to start marker)

        # Handle brush/eraser release (left-click)
        if self._painting:
            self._painting = False
            self._finish_stroke()
            # Re-enable camera panning
            tool = manager.active_tool
            if tool in ("brush", "eraser"):
                self.view.camera.interactive = False  # Keep disabled for paint tools
            return

        if self.dragging_roi:
            # Emit final modification signal
            self.roi_modified.emit(self.dragging_roi)
            # Notify ROI that drag ended (for LaneROI marker callbacks)
            if hasattr(self.dragging_roi, "end_marker_drag"):
                self.dragging_roi.end_marker_drag()
            self.dragging_roi = None
            self.drag_handle = None
            self.last_pos = None
            # Re-enable camera panning if in pointer mode
            if manager.active_tool == "pointer":
                self.view.camera.interactive = True

        if self.drawing_roi:
            self.drawing_roi = None
            self.start_pos = None
            # Re-enable camera after drawing
            self.view.camera.interactive = False  # Keep disabled for draw tools


from .toolbar import Toolbar  # Re-export for backward compatibility


def imshow(
    data,
    meta_or_title=None,
    dims=None,
    *,
    title=None,
    scale=None,
    colormap=None,
):
    """
    Convenience function to show an image.

    Args:
        data: Image data (numpy array or 5D proxy from load_image).
        meta_or_title: Either a metadata dict from load_image(), or a string title.
        dims (str): Dimension order string (e.g. 'tyx', 'zcyx').
                    Only used for numpy arrays. If None, heuristics are used.
        title (str): Window title (keyword-only, for backward compatibility).
                     Ignored if meta_or_title is provided.
        scale (tuple): Pixel spacing as (z, y, x) in physical units (e.g. microns).
                       Used for proper aspect ratio in OrthoViewer and VolumeViewer.
                       Overrides scale from metadata if both are provided.
        colormap (str or dict): Colormap name or dict mapping channel indices to names.
                                Available colormaps: "viridis", "plasma", "magma", "inferno",
                                "cividis", "hot", "cool", "coolwarm", "turbo", "gray",
                                "Orange", "Green", "Cyan", "Magenta", "Yellow", "White",
                                "Red", "Pure Green", "Blue".

    Examples:
        # From load_image (recommended)
        img, meta = load_image("my_image.ims")
        imshow(img, meta)

        # From numpy array
        imshow(my_array, "My Title")
        imshow(my_array, title="My Title")
        imshow(my_array, dims="zcyx")

        # With explicit pixel spacing (z=0.5um, y/x=0.1um)
        imshow(my_array, dims="zyx", scale=(0.5, 0.1, 0.1))

        # With title and scale
        imshow(my_array, "My Title", dims="zcyx", scale=(0.5, 0.1, 0.1))

        # With colormap (single colormap for all channels)
        imshow(my_array, colormap="viridis")

        # With per-channel colormaps
        imshow(my_array, colormap={0: "Green", 1: "Magenta"})
    """
    # Ensure QApplication exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Apply Theme
    from .theme import DARK_THEME

    app.setStyleSheet(DARK_THEME)

    # Handle backward compatibility: title= keyword argument
    if meta_or_title is None and title is not None:
        meta_or_title = title

    # Determine title and metadata from second argument
    meta = None
    title_str = "Image"

    if isinstance(meta_or_title, dict):
        meta = meta_or_title
        title_str = meta.get("filename", "Image")
    elif isinstance(meta_or_title, str):
        title_str = meta_or_title

    # If scale is provided, merge it into metadata
    if scale is not None:
        if meta is None:
            meta = {}
        else:
            meta = meta.copy()  # Don't mutate original
        meta["scale"] = scale

    # Handle different data types
    if isinstance(data, (Imaris5DProxy, Numpy5DProxy)):
        # Already a 5D proxy, pass directly
        pass
    elif hasattr(data, "shape") and hasattr(data, "ndim") and data.ndim == 5:
        # Generic 5D proxy-like object
        pass
    elif isinstance(data, np.ndarray):
        # Numpy array, normalize to 5D
        data = normalize_to_5d(data, dims=dims)
    else:
        raise ValueError(
            "data must be a 5D proxy, numpy array, or other array-like object"
        )

    viewer = ImageWindow(data, title=title_str, meta=meta)

    # Apply colormap(s) if specified
    if colormap is not None:
        if isinstance(colormap, str):
            # Apply same colormap to all channels
            for c in range(viewer.renderer.num_channels):
                viewer.renderer.set_colormap(c, colormap)
        elif isinstance(colormap, dict):
            # Apply per-channel colormaps
            for c, cmap_name in colormap.items():
                viewer.renderer.set_colormap(c, cmap_name)
        viewer.canvas.update()

    viewer.show()

    return viewer


def run_app():
    """
    Start the Qt event loop. Use this when running from a script
    to ensure windows are visible and interactive.
    """
    app = QApplication.instance()
    if app:
        from .theme import DARK_THEME

        app.setStyleSheet(DARK_THEME)
        app.exec_()
