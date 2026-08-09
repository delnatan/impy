import os
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np
from qtpy import API_NAME
from qtpy.QtCore import QPoint, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider
from vispy import app, scene

from .. import colors as tokens
from ..io import (
    Imaris5DProxy,
    Numpy5DProxy,
    load_image,
    normalize_to_5d,
    save_imaris,
    save_tiff,
)
from ..visuals.labels import LabelOverlayVisual
from ..data.labels import SparseLabels
from ..data.shapes import (
    ShapeData,
    ALL_FRAMES,
    RECTANGLE,
    CIRCLE,
    LINE,
    POLYLINE,
    EVT_EDITED,
    AddShape,
    RemoveShape,
    MoveShape,
    AdjustHandle,
    AddVertex,
    RemoveVertex,
    SetPolylineFlags,
    SetShapeLabel,
    SetShapeParams,
    SetShapeVertices,
    _apply_handle_adjustment,
    crop_rect,
    integer_square_from_corner,
    polyline_is_closed,
    rect_opposite_corner,
    snap_rectangle_params,
)
from ..layers.base import Layer, LayerList
from .manager import manager
from .playback import PlaybackController
from .workspace import present_window
from ..visuals.overlays import ScaleTimestampOverlay
from ..viewers import OrthoViewer
from ..data.points import PointTable
from ..data.slice_loader import SliceLoader
from ..data.view_state import ViewState
from ..data.channel_state import ChannelDisplayList
from ..visuals.points import DEFAULT_STYLE as POINT_DEFAULT_STYLE, PointLayerVisual
from ..visuals.shapes import ShapeLayerVisual
from ..rois import PointROI
from ..data.tracks import TrackTable
from ..visuals.tracks import DEFAULT_STYLE as TRACK_DEFAULT_STYLE, TrackLayerVisual
from ..visuals.image import CompositeImageVisual
from ..widgets import (
    AlignmentDialog,
    AxesDialog,
    CombineImagesDialog,
    FFTDialog,
    ImageMathDialog,
    MetadataDialog,
    OverlaySettingsDialog,
    TransformDialog,
    ZProjectionDialog,
)

try:
    app.use_app(API_NAME)
except Exception:
    app.use_app("pyqt5")


# Declarative menu structure for ImageWindow's File/Adjust/Image/View
# menus: (menu_title, [item, ...]) where each item is either None (a
# separator) or a dict with "label", "method" (name looked up on the
# target at trigger time), and optional "shortcut"/"tooltip"/"enabled".
# "enabled" may be a bool or a zero-arg callable evaluated once when the
# action is built; a falsy result disables (greys out) the action -- the
# documented convention for a plugin to grey out its own menu item when
# its optional backend isn't installed (see pyvistra/plugins.py). Single
# source of truth for both ImageWindow's own embedded menu bar
# (build_menus(..., target=self)) and the Workspace shell's persistent
# mirrored bar (build_menus(..., target=get_active_window) — see
# ui/workspace.py), so the two never drift apart.
MENU_SPEC = [
    ("File", [
        {"label": "Save as TIFF...", "shortcut": "Ctrl+S", "method": "save_as_tiff"},
        {"label": "Save as Imaris...", "shortcut": "Ctrl+Shift+S", "method": "save_as_imaris"},
        {
            "label": "Save As...",
            "shortcut": "Ctrl+Alt+S",
            "method": "save_as_any",
            "tooltip": "Save to any registered format (.tif, .ims, .psf.h5, .pupil.h5, ...)",
        },
        {
            "label": "Save Channel...",
            "method": "_save_channel_dialog",
            "tooltip": (
                "Extract one channel (and an optional T/Z subrange) and route "
                "it to a new window, an existing window, or a file."
            ),
        },
        None,
        {"label": "Save Snapshot...", "method": "_save_snapshot_dialog"},
        {"label": "Export Frames...", "method": "_export_frames_dialog"},
    ]),
    ("Adjust", [
        {"label": "Channels && Contrast...", "shortcut": "Shift+C", "method": "show_channel_panel"},
        None,
        {"label": "Line Profile...", "shortcut": "Shift+K", "method": "show_line_profile"},
    ]),
    ("Image", [
        {"label": "Image Info", "shortcut": "Shift+I", "method": "show_metadata_dialog"},
        {"label": "Ortho View", "method": "show_ortho_view"},
        {"label": "3D Volume View", "method": "show_volume_view"},
        {"label": "Z-Montage View...", "method": "show_zmontage_view"},
        None,
        {"label": "Transform...", "shortcut": "Shift+T", "method": "show_transform_dialog"},
        {"label": "Align Images...", "method": "show_alignment_dialog"},
        {"label": "Z Projection...", "method": "show_z_projection_dialog"},
        {"label": "Image Math...", "method": "show_image_math_dialog"},
        {"label": "Combine Images...", "method": "show_combine_images_dialog"},
        {"label": "FFT...", "method": "show_fft_dialog"},
        None,
        {"label": "Reorder Axes...", "method": "show_axes_dialog"},
    ]),
    ("View", [
        {"label": "Overlay Settings...", "method": "show_overlay_settings_dialog"},
        None,
        {"label": "Compare With...", "method": "compare_with_dialog"},
    ]),
]


def _build_menu_items(menu, items, target, actions):
    """Populate *menu* from *items*, recursing into nested ``"submenu"``
    entries. Appends every dispatchable leaf action to *actions*."""
    for item in items:
        if item is None:
            menu.addSeparator()
            continue
        if "submenu" in item:
            submenu = menu.addMenu(item["label"])
            _build_menu_items(submenu, item["submenu"], target, actions)
            continue
        action = QAction(item["label"], target)
        if "shortcut" in item:
            action.setShortcut(item["shortcut"])
        if "tooltip" in item:
            action.setToolTip(item["tooltip"])
        enabled = item.get("enabled", True)
        action.setEnabled(enabled() if callable(enabled) else enabled)
        action.triggered.connect(getattr(target, item["method"]))
        menu.addAction(action)
        actions.append(action)


def build_menus(menubar, spec, target):
    """Populate *menubar* from *spec*, connecting each action to
    ``getattr(target, item["method"])``. Returns the flat list of leaf
    (non-separator) QActions created, in spec order. Items may nest a
    ``"submenu"`` list instead of a ``"method"`` to group related actions
    -- their leaves are folded into the same flat list.

    Used for ImageWindow's own embedded menu bar (``target=self``,
    bound once at construction). The Workspace shell's persistent
    mirrored bar uses ``build_proxy_menus`` in ``ui/workspace.py``
    instead, which retargets dynamically rather than binding to a
    single fixed object.
    """
    from ..plugins import discover_plugins

    discover_plugins()

    actions = []
    for menu_title, items in spec:
        menu = menubar.addMenu(menu_title)
        _build_menu_items(menu, items, target, actions)
    return actions


# Fixed regardless of how many of the Mode/Channel, Time, and Z rows a
# given image actually needs, so two windows with different dimensionality
# get the same canvas height whenever their outer window height matches
# (floating windows sized alike, or docked side-by-side in a Workspace
# split) -- otherwise the canvas (the layout's only stretch=1 item) eats
# whatever the controls panel doesn't use, and that varies per image.
_CONTROLS_PANEL_HEIGHT = 140


class ImageWindow(QMainWindow):
    """Main image viewer window with ROI support."""

    # Every viewer class the Workspace hosts declares a class-level
    # MENU_SPEC; the workspace mirrors the active tab's spec onto its
    # persistent menu bar (see ui/workspace.py).
    MENU_SPEC = MENU_SPEC

    # Signals for decoupled communication
    window_activated = Signal(object)  # Emits self when window becomes active
    window_shown = Signal(object)  # Emits self when window is shown
    window_closing = Signal(object)  # Emits self when window is closing
    roi_added = Signal(object)  # Emits the ROI that was added
    roi_removed = Signal(object)  # Emits the ROI that was removed
    roi_selection_changed = Signal(object)  # Emits the selected ROI (or None)
    roi_modified = Signal(object)  # Emits ROI when it's being modified (dragged)
    view_changed = Signal(object)  # Emits self when displayed slice/channel changes
    # Emits the new channel index precisely on change. t/z have this via
    # ViewState.subscribe(); channel has no such observable, so anything
    # that needs to react to *just* a channel change (as opposed to
    # view_changed's coarser "slice or channel changed") uses this instead.
    channel_changed = Signal(int)
    label_changed = Signal(object)  # Emits SparseLabels when labels change
    mask_layer_added = Signal(str)  # Emits mask layer name when added
    mask_layer_removed = Signal(str)  # Emits mask layer name when removed
    track_layer_added = Signal(str)  # Emits track layer name when added
    track_layer_removed = Signal(str)  # Emits track layer name when removed
    point_layer_added = Signal(str)  # Emits point layer name when added
    point_layer_removed = Signal(str)  # Emits point layer name when removed
    # New layer system signals
    layer_added = Signal(object)  # Emits Layer when added
    layer_removed = Signal(object)  # Emits Layer when removed
    shape_selection_changed = Signal(object, object)  # Emits (Layer, shape_id) or (None, None)

    # Internal — marshals ImageBuffer change notifications from any worker
    # thread onto the GUI thread (Qt.QueuedConnection in __init__).
    _buffer_dirty = Signal(object)
    # Internal — marshals SliceLoader deliveries (key, plane) from the
    # loader thread onto the GUI thread (Qt.QueuedConnection in __init__).
    _slice_ready = Signal(object, object)

    def __init__(self, data_or_path, title="Image", meta=None, filepath=None):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)

        # 1. Load/Set Data
        if isinstance(data_or_path, str):
            self.filepath = data_or_path
            self.img_data, self.meta = load_image(self.filepath)
            filename = self.meta.get("filename", "Image")
        else:
            self.filepath = filepath
            self.meta = meta or {}

            # Accept any 5D proxy-like object (Imaris5DProxy, Numpy5DProxy, etc.)
            if isinstance(data_or_path, (Imaris5DProxy, Numpy5DProxy)):
                self.img_data = data_or_path
            elif isinstance(data_or_path, np.ndarray):
                # Plain ndarray, even if already 5D -- always wrap it so
                # img_data is refcounted (.acquire()/.release()) like every
                # other proxy, instead of falling through to the generic
                # duck-typed branch below (which a bare ndarray also matches).
                self.img_data = normalize_to_5d(data_or_path)
            elif (
                hasattr(data_or_path, "shape")
                and hasattr(data_or_path, "ndim")
                and data_or_path.ndim == 5
            ):
                # Generic 5D proxy-like object
                self.img_data = data_or_path
            else:
                raise ValueError(
                    "data must be a 5D proxy, numpy array, or filepath string"
                )

            filename = self.meta.get("filename", title)

        # Register with Manager
        self.window_id = manager.register(self)

        # Generic flag a PSF-producing plugin can set so a PSF-picker it
        # provides can annotate windows that are actually PSFs (such a
        # picker should still list every open window, not just
        # is_psf-flagged ones -- a PSF loaded from disk is just as valid).
        self.is_psf = False

        # "real" (default) or "frequency" -- set on FFT output metadata
        # (see fft_dialog.py) so unit-aware consumers (line profile, radial
        # profile) know pixel indices mean cycles/unit, not distance.
        self.pixel_space = self.meta.get("space", "real")

        self.T, self.Z, self.C, self.Y, self.X = self.img_data.shape

        # Title
        sz, sy, sx = self.meta.get("scale", (1.0, 1.0, 1.0))
        title_str = f"[{self.window_id}] {filename} "
        title_str += f"[{self.X}x{self.Y} px] "
        if self.filepath:
            title_str += f"[{sx:.2f} x {sy:.2f} \u00b5m]"
        self.setWindowTitle(title_str)

        # 2. Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # 3. Vispy Canvas
        self._build_canvas()

        # 4. Info Bar
        self.info_label = QLabel("Hover over image")
        self.info_label.setStyleSheet(
            f"background-color: {tokens.BG_SURFACE}; color: {tokens.TEXT_PRIMARY}; padding: 4px;"
        )
        self.info_label.setFixedHeight(25)
        self.layout.addWidget(self.info_label, 0)

        # 5. Visuals
        is_rgb = self.meta.get("is_rgb", False)
        self.renderer = CompositeImageVisual(
            self.view,
            self.img_data,
            is_rgb=is_rgb,
            channels_meta=self.meta.get("channels"),
        )
        self.renderer.reset_camera(self.img_data.shape)

        # 6. Controls Area (Sliders + Mode)
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(10, 10, 10, 10)
        self.controls_layout.setSpacing(5)
        self.controls_widget.setFixedHeight(_CONTROLS_PANEL_HEIGHT)
        self.layout.addWidget(self.controls_widget, 0)

        # Navigation state (t, z, projection). Sliders write into it;
        # its subscription drives label sync + update_view, so any writer
        # (slider, playback, scripting) triggers exactly one redraw.
        # Default z to the middle slice for 3D stacks -- opening on z=0
        # is rarely the useful view for volumetric data.
        self.view_state = ViewState(z=self.Z // 2)
        self._suspend_view_updates = False
        self.c_idx = 0  # Active channel index for Single mode

        # Control references are optional (depend on C/T/Z dimensionality).
        # Initialize eagerly so playback/control methods are safe when T == 1.
        self._init_control_refs()
        # Subscribe only after control refs exist — the handler touches them.
        self.view_state.subscribe(self._on_view_state_changed)

        # Timelapse playback state
        self._playback = PlaybackController(
            frame_count=lambda: self.T,
            time_index=lambda: self.t_idx,
            advance_time=self._advance_time_index,
            timestamps=lambda: self.meta.get("timestamps", []),
            parent=self,
        )

        self._init_overlay()

        self._setup_controls()

        # 7. Menu & Dialogs
        self.transform_dialog = None
        self.z_projection_dialog = None
        self._image_math_dialog = None
        self._combine_images_dialog = None
        self._fft_dialog = None
        self._alignment_dialog = None  # Shared singleton
        self._setup_menu()

        # Initial window size: fit the canvas to the image's XY aspect
        # ratio (within a screen-relative bounding box) rather than a flat
        # default, so e.g. a wide mosaic and a tall crop don't both open at
        # the same square-ish size. Done here (not earlier) so the menu bar
        # is already built and its sizeHint() is accurate.
        canvas_w, canvas_h = self._compute_canvas_size()
        menubar_h = self.menuBar().sizeHint().height()
        total_h = canvas_h + self.info_label.height() + _CONTROLS_PANEL_HEIGHT + menubar_h
        self.resize(canvas_w, total_h)

        # 8. ROI State (focused-point editing only; see PointROI)
        self.start_pos = None
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
        self._mask_propagate_z = False  # Fill all Z slices (cookie-cut)

        # Ad hoc read-only overlays keyed by caller-chosen string (e.g. a
        # LineProfileDialog compare series). Lets dialogs in widgets/ draw a
        # vispy visual here without importing vispy themselves.
        self._external_overlays = {}

        # 10. Track State (multiple named track layers)
        self._track_layers = OrderedDict()  # name -> {"tracks": TrackTable, "visual": TrackLayerVisual, "visible": True}
        self._active_track_layer = None

        # 11. Point State (multiple named point layers)
        self._point_layers = OrderedDict()  # name -> {"points": PointTable, "visual": PointLayerVisual, "visible": True}
        self._active_point_layer = None
        self._focused_point_layer = None
        self._focused_point_id = None
        self._focused_point_roi: PointROI | None = None

        # 12. Unified Layer System
        self.layers = LayerList()
        self._drawing_shape_layer = None  # Layer being drawn into
        self._drawing_shape_id = None  # Shape ID being drawn
        self._drawing_shape_cmd = None  # AddShape command, to retract by identity
        # Multi-click polyline drawing state.
        self._polyline_drawing_layer = None
        self._polyline_drawing_id = None
        # Point tool drag state.
        self._point_dragging_layer = None  # layer name (str) or None
        self._point_dragging_id = None
        self._point_drag_start_xy: tuple[float, float] | None = None

        # Interactive edit (pointer drag) on an existing shape entry.
        # _editing_shape_handle is None for body drag, else handle name.
        self._editing_shape_layer = None
        self._editing_shape_id = None
        self._editing_shape_handle = None
        self._editing_shape_start_params = None
        self._editing_shape_start_pos = None
        # For POLYLINE edits the original vertex array snapshot.
        self._editing_shape_start_vertices = None
        self._dragging_gel_marker_layer = None
        self._dragging_gel_marker_shape_id = None
        self._dragging_gel_marker_idx = None

        # Live buffer updates: dispatch from any thread to the GUI thread.
        self._buffer_unsubscribe = None
        self._buffer_dirty.connect(
            self._on_buffer_dirty, Qt.QueuedConnection
        )
        self._subscribe_to_buffer(self.img_data)

        # Async slice loading for lazy (file/zarr-backed) sources: reads
        # happen on a worker thread, delivered here via queued signal.
        # In-memory sources keep the synchronous path (loader is None).
        self._slice_loader = None
        # Set by _request_auto_contrast when the first slice for a lazy
        # source hasn't arrived yet; _on_slice_ready fires the deferred
        # auto-contrast once it does (see _request_auto_contrast).
        self._pending_auto_contrast = False
        self._slice_ready.connect(self._on_slice_ready, Qt.QueuedConnection)
        self._replace_slice_loader(self.img_data)

        # 13. Events
        self._connect_canvas_events()
        manager.tool_changed.connect(self._on_active_tool_changed)

        # 14. Hover tooltip (on-canvas Text visual for point info)
        self._build_hover_label()

        # Focus policy
        self.setFocusPolicy(Qt.StrongFocus)

        # Initial Draw + one-shot auto-contrast so default clim matches data.
        self.update_view()
        self._request_auto_contrast()
        self.canvas.update()

    # ------------------------------------------------------------------
    # Navigation state (ViewState-backed)
    # ------------------------------------------------------------------

    @property
    def t_idx(self):
        return self.view_state.t

    @t_idx.setter
    def t_idx(self, val):
        self.view_state.set_t(val)

    @property
    def z_idx(self):
        return self.view_state.z

    @z_idx.setter
    def z_idx(self, val):
        self.view_state.set_z(val)

    def _on_view_state_changed(self, field):
        """Sync nav widgets to ViewState, then redraw.

        Fires once per actual state change (setters no-op on unchanged
        values), so a slider drag and a programmatic ``window.t_idx = 5``
        take the same single-redraw path.
        """
        vs = self.view_state
        if field == "t":
            if self.t_slider is not None and self.t_slider.value() != vs.t:
                self.t_slider.blockSignals(True)
                self.t_slider.setValue(vs.t)
                self.t_slider.blockSignals(False)
            if self.t_spin is not None and self.t_spin.value() != vs.t:
                self.t_spin.blockSignals(True)
                self.t_spin.setValue(vs.t)
                self.t_spin.blockSignals(False)
        elif field == "z":
            if self.z_slider is not None and self.z_slider.value() != vs.z:
                self.z_slider.blockSignals(True)
                self.z_slider.setValue(vs.z)
                self.z_slider.blockSignals(False)
            if self.z_spin is not None and self.z_spin.value() != vs.z:
                self.z_spin.blockSignals(True)
                self.z_spin.setValue(vs.z)
                self.z_spin.blockSignals(False)
        if not self._suspend_view_updates:
            self.update_view()

    def showEvent(self, event):
        super().showEvent(event)
        self.window_shown.emit(self)
        # Qt doesn't repaint hidden widgets, so canvas.update() calls made
        # while this window was a non-current tab (e.g. a docked split-tab
        # group) are no-ops -- the vispy scene state itself is current, but
        # nothing forces a redraw of it. Catch up now that we're visible.
        self.canvas.update()

    def closeEvent(self, event):
        manager.unregister(self)
        self.window_closing.emit(self)

        self._playback.stop()

        if hasattr(self, "overlay") and self.overlay is not None:
            self.overlay.remove()
        for entry in self._track_layers.values():
            visual = entry.get("visual")
            if visual is not None:
                visual.remove()
        for entry in self._point_layers.values():
            visual = entry.get("visual")
            if visual is not None:
                visual.remove()
        if self._focused_point_roi is not None:
            self._focused_point_roi.remove()
            self._focused_point_roi = None

        for overlay in self._external_overlays.values():
            overlay.parent = None
        self._external_overlays.clear()

        # Stop receiving buffer change notifications before tearing down.
        if self._buffer_unsubscribe is not None:
            self._buffer_unsubscribe()
            self._buffer_unsubscribe = None

        # Stop the slice-loader thread before releasing the data source.
        if self._slice_loader is not None:
            self._slice_loader.close()
            self._slice_loader = None

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
            if self.overlay is not None:
                self.overlay.update()
            self.canvas.update()
        elif event.key() == Qt.Key_L:
            # Toggle ROI labels visibility
            from ..rois import ROI

            ROI.toggle_labels()
            for w in manager.get_all().values():
                w.canvas.update()
        elif event.key() == Qt.Key_Escape:
            # Cancel contour mode if active
            if self._contour_mode:
                self._cancel_contour()
                return
            # Cancel an in-progress polyline drawing.
            if self._polyline_drawing_id is not None and self._polyline_drawing_layer is not None:
                layer = self._polyline_drawing_layer
                sid = self._polyline_drawing_id
                if layer.undo_stack.can_undo:
                    layer.undo_stack.undo(layer.data)
                else:
                    if sid in layer.data:
                        layer.data.remove(sid)
                self._polyline_drawing_layer = None
                self._polyline_drawing_id = None
                return
            # Deselect all ROIs and shape-layer entries
            self._clear_shape_selection()
            self.canvas.update()
            self.roi_selection_changed.emit(None)
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            # Delete currently-selected shape (undoable).
            if self._current_shape_selection() is not None:
                self.delete_selected_shape()
                return
            super().keyPressEvent(event)
        elif event.key() == Qt.Key_F2:
            # Rename currently-selected shape.
            if self._current_shape_selection() is not None:
                self.rename_selected_shape()
                return
            super().keyPressEvent(event)
        elif event.key() in (
            Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down
        ) and self._current_shape_selection() is not None:
            # Nudge selected shape. Shift = ×10.
            step = 10.0 if (event.modifiers() & Qt.ShiftModifier) else 1.0
            dx = dy = 0.0
            if event.key() == Qt.Key_Left:
                dx = -step
            elif event.key() == Qt.Key_Right:
                dx = step
            elif event.key() == Qt.Key_Up:
                dy = -step
            elif event.key() == Qt.Key_Down:
                dy = step
            self.nudge_selected_shape(dx, dy)
            return
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            # Open properties dialog for selected shape.
            if self._current_shape_selection() is not None:
                self.edit_selected_shape_properties()
                return
            super().keyPressEvent(event)
        elif event.key() == Qt.Key_Space:
            # Temporarily switch to pointer mode for panning/zooming
            if self._space_held_previous_tool is None:
                self._space_held_previous_tool = manager.active_tool
                manager.set_active_tool("pointer")
                self.update_cursor()
        # Label/Mask shortcuts
        elif event.key() == Qt.Key_B:
            # Brush tool
            manager.set_active_tool("brush")
            self.update_cursor()
        elif event.key() == Qt.Key_E:
            # Eraser tool
            manager.set_active_tool("eraser")
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
                manager.set_active_tool(self._space_held_previous_tool)
                self._space_held_previous_tool = None
                self.update_cursor()
        else:
            super().keyReleaseEvent(event)

    def _build_canvas(self):
        """(Re)build the vispy canvas/view/camera and add it to the layout.

        Split out of __init__ so `_rebuild_canvas_for_float` can build a
        brand-new canvas (fresh GL context) rather than reparenting the
        live one -- reparenting a QOpenGLWidget from an embedded tab to a
        top-level window corrupts its GL context on this platform and
        crashes (see Workspace.float_window).

        Inserted at index 0 rather than appended: at __init__ time the
        layout is empty so the two are equivalent, but
        `_rebuild_canvas_for_float` calls this after info_label and
        controls_widget are already in the layout, and an append would
        land the canvas below them instead of restoring it to the top.
        """
        self.canvas = scene.SceneCanvas(keys=None, bgcolor="black", show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"
        self.view.camera.aspect = 1
        self.layout.insertWidget(0, self.canvas.native, 1)

    def _connect_canvas_events(self):
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
        self.canvas.events.key_press.connect(self._on_vispy_key_press)
        self.canvas.events.mouse_release.connect(self._on_view_transform_event)
        self.canvas.events.mouse_wheel.connect(self._on_view_transform_event)
        self.canvas.events.resize.connect(self._on_view_transform_event)

    def _build_hover_label(self):
        self._hover_label = scene.visuals.Text(
            text="",
            pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.95),
            font_size=9,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.view.scene,
        )
        self._hover_label.order = 10_100
        self._hover_label.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )
        self._hover_label.visible = False

    def _rebuild_canvas_for_float(self):
        """Tear down the live GL canvas and build a fresh one in place.

        Reparenting a QOpenGLWidget across a top-level-window boundary
        (workspace tab <-> floating window, either direction) corrupts
        its GL context on this platform (GL errors followed by a hard
        crash) -- see Workspace.float_window and Workspace.add_window.
        The only reliable fix is to never move the live canvas: discard
        it and rebuild every visual from its underlying data (data/ and
        visuals/ are already kept separate for exactly this reason).
        """
        saved_rect = self.view.camera.rect
        overlay_config = self.overlay.get_config()
        # CompositeImageVisual.__init__ always seeds self.display with
        # fresh defaults (default clim for the dtype, default per-channel
        # colormap) -- it has no way to be constructed with existing
        # state. Save the user's current clim/gamma/colormap/visibility
        # per channel here and reapply after rebuilding below, or a
        # float/redock silently reverts any contrast adjustment the user
        # made via the Channels && Contrast panel.
        saved_channel_states = [
            self.renderer.display[c] for c in range(len(self.renderer.display))
        ]

        # Transient/interactive visuals: drop them, they lazily recreate
        # themselves on next use (focused-point editing, contour drawing,
        # cross-window line-profile overlays).
        self.clear_focused_point()
        self._contour_marker = None
        self._external_overlays.clear()

        # Persistent, data-backed visuals: dispose the GL objects but
        # keep the data (dicts/Layer entries) so they can be rebuilt.
        self.overlay.remove()
        for entry in self._track_layers.values():
            visual = entry.get("visual")
            if visual is not None:
                visual.remove()
        for entry in self._point_layers.values():
            visual = entry.get("visual")
            if visual is not None:
                visual.remove()
        for entry in self._mask_layers.values():
            visual = entry.get("visual")
            if visual is not None:
                visual.remove()
        for layer in self.layers.by_type("shapes"):
            if layer.visual is not None:
                layer.visual.remove()
        for layer_visual in self.renderer.layers:
            layer_visual.parent = None

        self.layout.removeWidget(self.canvas.native)
        self.canvas.close()

        # Fresh GL context, then rebuild everything that renders into it.
        self._build_canvas()
        self._connect_canvas_events()

        is_rgb = self.meta.get("is_rgb", False)
        self.renderer = CompositeImageVisual(
            self.view,
            self.img_data,
            is_rgb=is_rgb,
            channels_meta=self.meta.get("channels"),
        )
        for c, state in enumerate(saved_channel_states):
            if c >= self.renderer.num_channels:
                break
            self.renderer.set_clim(c, *state.clim)
            self.renderer.set_gamma(c, state.gamma)
            self.renderer.set_colormap(c, state.colormap_name)
            self.renderer.set_channel_visible(c, state.visible)
        # reset_camera() also sets camera.flip -- the fresh PanZoomCamera
        # built in _build_canvas() defaults to no flip, so this must run
        # unconditionally or the Y axis renders un-flipped (image appears
        # upside-down / lateral coords inverted) whenever a saved_rect
        # exists, which is the common case (any window that was already
        # showing something before it floated).
        self.renderer.reset_camera(self.img_data.shape)
        if saved_rect is not None:
            self.view.camera.rect = saved_rect

        self._build_hover_label()

        self._init_overlay()
        self.overlay.set_config(overlay_config)

        for name, entry in self._mask_layers.items():
            visual = LabelOverlayVisual(
                self.view,
                shape_yx=(self.Y, self.X),
                scale=self.renderer.scale,
            )
            entry["visual"] = visual
            if entry["labels"] is not None:
                visual.set_labels(entry["labels"])
                if entry["labels"].ndim == 3:
                    visual.update_slice(self.z_idx)
            visual.visible = entry["visible"]
            if name in self.layers:
                self.layers[name].visual = visual

        for name, entry in self._point_layers.items():
            visual = PointLayerVisual(self.view, **entry["style"])
            entry["visual"] = visual
            visual.set_points(entry["points"])
            visual.set_time_z(self.t_idx, self.z_idx)
            visual.set_selected_point_ids(entry["selected_ids"])
            visual.visible = entry["visible"]
            if name in self.layers:
                self.layers[name].visual = visual

        for name, entry in self._track_layers.items():
            visual = TrackLayerVisual(self.view, **entry["style"])
            entry["visual"] = visual
            visual.set_tracks(entry["tracks"])
            visual.set_time_z(self.t_idx, self.z_idx)
            visual.visible = entry["visible"]
            if name in self.layers:
                self.layers[name].visual = visual

        for layer in self.layers.by_type("shapes"):
            visual = ShapeLayerVisual(self.view.scene)
            visual.update(layer.data, layer.selected_ids, self.t_idx, self.z_idx)
            layer.visual = visual

        # Whichever ChannelPanel currently displays this window's state --
        # its own private popup, or the workspace's shared popup if that's
        # what's targeting it right now -- needs to resubscribe to the
        # just-rebuilt renderer's display, or it keeps listening to the
        # orphaned old one.
        active_panel = getattr(self, "_active_channel_panel", None)
        if active_panel is not None:
            active_panel.rebind_renderer()

        self.canvas.update()

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
            raise ValueError("new_data must be 5D (T, Z, C, Y, X)")

        old_shape = (self.T, self.Z, self.C, self.Y, self.X)
        old_is_rgb = self.meta.get("is_rgb", False)

        # Update data and metadata
        if self.img_data is not None and hasattr(self.img_data, "release"):
            self.img_data.release()
        self.img_data = new_data
        if metadata is not None:
            self.meta = metadata
            self.pixel_space = self.meta.get("space", "real")
        self._subscribe_to_buffer(new_data)
        self._replace_slice_loader(new_data)

        self.T, self.Z, self.C, self.Y, self.X = self.img_data.shape
        new_shape = (self.T, self.Z, self.C, self.Y, self.X)
        new_is_rgb = self.meta.get("is_rgb", False)

        # Clamp indices to valid bounds for the new data. Suspend
        # subscription redraws: the renderer isn't rebuilt yet, and
        # update_view() runs explicitly below.
        self._suspend_view_updates = True
        try:
            self.view_state.clamp(self.T, self.Z)
        finally:
            self._suspend_view_updates = False
        self.c_idx = min(max(0, self.c_idx), self.C - 1)

        needs_rebuild = (old_shape != new_shape) or (old_is_rgb != new_is_rgb)
        if needs_rebuild:
            # Remove old renderer layers, then rebuild renderer and controls.
            for layer in self.renderer.layers:
                layer.parent = None

            self.renderer = CompositeImageVisual(
                self.view,
                self.img_data,
                is_rgb=new_is_rgb,
                channels_meta=self.meta.get("channels"),
            )
            self.renderer.reset_camera(self.img_data.shape)
            self._rebuild_controls()
        else:
            self.renderer.data = new_data

        self._sync_overlay_spacing()
        self._playback.refresh_realtime_ui()

        # Sync sliders to clamped indices without emitting callbacks.
        c_slider = getattr(self, "c_slider", None)
        if c_slider is not None:
            c_slider.blockSignals(True)
            c_slider.setValue(self.c_idx)
            c_slider.blockSignals(False)
        t_slider = getattr(self, "t_slider", None)
        if t_slider is not None:
            t_slider.blockSignals(True)
            t_slider.setValue(self.t_idx)
            t_slider.blockSignals(False)
        z_slider = getattr(self, "z_slider", None)
        if z_slider is not None:
            z_slider.blockSignals(True)
            z_slider.setValue(self.z_idx)
            z_slider.blockSignals(False)
        c_spin = getattr(self, "c_spin", None)
        if c_spin is not None:
            c_spin.blockSignals(True)
            c_spin.setValue(self.c_idx)
            c_spin.blockSignals(False)
        t_spin = getattr(self, "t_spin", None)
        if t_spin is not None:
            t_spin.blockSignals(True)
            t_spin.setValue(self.t_idx)
            t_spin.blockSignals(False)
        z_spin = getattr(self, "z_spin", None)
        if z_spin is not None:
            z_spin.blockSignals(True)
            z_spin.setValue(self.z_idx)
            z_spin.blockSignals(False)

        self.update_view()
        if needs_rebuild:
            self._request_auto_contrast()
            self.canvas.update()

    def _setup_menu(self):
        # Kept so the workspace shell can disarm these specific
        # shortcuts while this window's menu bar is hidden (docked) —
        # see workspace.add_window / float_window — without touching
        # unrelated shortcuts (tool keys, playback, etc).
        self._menu_actions = build_menus(self.menuBar(), MENU_SPEC, self)

    def _default_save_path(self, ext):
        """Build a default save path for Save As."""
        if self.filepath:
            root, _ = os.path.splitext(self.filepath)
            return f"{root}_export{ext}"

        filename = self.meta.get("filename", "image")
        root, _ = os.path.splitext(filename)
        if not root:
            root = "image"
        return f"{root}{ext}"

    def _save_image_file(self, filepath):
        """Save current image data to TIFF or Imaris depending on extension."""
        ext = os.path.splitext(filepath)[1].lower()
        scale = self.meta.get("scale", (1.0, 1.0, 1.0))

        if ext in (".tif", ".tiff"):
            save_tiff(filepath, self.img_data, scale=scale, metadata=self.meta)
            return

        if ext == ".ims":
            save_imaris(filepath, self.img_data, metadata=self.meta)
            return

        raise ValueError(f"Unsupported output format: {ext}")

    def _save_with_dialog(self, title, default_ext, valid_exts, file_filter):
        """Prompt for output path for a specific format and save current image."""
        default_path = self._default_save_path(default_ext)
        filepath, _ = QFileDialog.getSaveFileName(
            self, title, default_path, file_filter
        )
        if not filepath:
            return

        current_ext = os.path.splitext(filepath)[1].lower()
        allowed_exts = [e.lower() for e in valid_exts]
        if current_ext not in allowed_exts:
            filepath += default_ext

        try:
            self._save_image_file(filepath)
            QMessageBox.information(
                self,
                "Save Complete",
                f"Saved image to:\n{filepath}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Could not save image:\n{exc}",
            )

    def save_as_tiff(self):
        """Save image as TIFF."""
        self._save_with_dialog(
            title="Save Image as TIFF",
            default_ext=".tif",
            valid_exts=[".tif", ".tiff"],
            file_filter="TIFF files (*.tif *.tiff)",
        )

    def save_as_imaris(self):
        """Save image as Imaris .ims."""
        self._save_with_dialog(
            title="Save Image as Imaris",
            default_ext=".ims",
            valid_exts=[".ims"],
            file_filter="Imaris files (*.ims)",
        )

    def save_as_any(self):
        """Save via any registered output format (TIFF, Imaris, PSF, Pupil, …).

        The format list is pulled from :func:`pyvistra.io.available_output_formats`,
        so adding a new format anywhere (incl. downstream libraries) makes it
        available from this menu with no extra wiring.
        """
        from pyvistra.io import available_output_formats, get_output_format

        formats = available_output_formats()
        if not formats:
            QMessageBox.warning(
                self, "Save As", "No output formats are registered."
            )
            return

        # Sort by extension length descending so multi-dot extensions like
        # ``.psf.h5`` win over ``.h5`` (if both were ever registered).
        formats_sorted = sorted(formats, key=lambda le: -len(le[1]))
        filters = [f"{label} (*{ext})" for label, ext in formats_sorted]
        file_filter = ";;".join(filters)
        default_ext = formats_sorted[0][1]
        default_path = self._default_save_path(default_ext)

        filepath, selected_filter = QFileDialog.getSaveFileName(
            self, "Save As", default_path, file_filter
        )
        if not filepath:
            return

        # Resolve the target extension: prefer the actual filename suffix if
        # it matches a registered format; otherwise fall back to the one
        # implied by the chosen filter.
        ext = next(
            (e for _, e in formats_sorted if filepath.lower().endswith(e.lower())),
            None,
        )
        if ext is None:
            ext = next(
                (e for label, e in formats_sorted if f"(*{e})" in selected_filter),
                default_ext,
            )
            filepath += ext

        fmt = get_output_format(ext)
        if fmt is None:
            QMessageBox.critical(
                self, "Save Failed", f"No saver registered for {ext!r}."
            )
            return
        _, saver = fmt

        try:
            saver(filepath, self.img_data, self.meta)
        except Exception as exc:
            QMessageBox.critical(
                self, "Save Failed", f"Could not save {ext} file:\n{exc}"
            )
            return

        QMessageBox.information(
            self, "Save Complete", f"Saved image to:\n{filepath}"
        )

    # ---- Snapshot & Frame Export ----

    def snapshot(self, path=None, scale=1):
        """Save or return a screenshot of the current canvas view.

        Parameters
        ----------
        path : str or None
            If given, save image to this path and return the path.
            If None, return the RGBA array (H, W, 4) uint8.
        scale : int
            Upscale factor (e.g. 2 for 2× resolution). The canvas is
            rendered at its native size and then resized with Lanczos.
        """
        from PIL import Image

        rgba = self.canvas.render()
        if scale > 1:
            h, w = rgba.shape[:2]
            img = Image.fromarray(rgba)
            img = img.resize((w * scale, h * scale), Image.LANCZOS)
            rgba = np.array(img)
        if path is None:
            return rgba
        Image.fromarray(rgba).save(path)
        return path

    def export_frames(self, output_dir, prefix="frame", scale=1,
                      frame_range=None, fmt="png"):
        """Export frames as numbered image files.

        For timelapse data (T > 1), iterates over timepoints.
        For single-timepoint Z-stacks (T == 1, Z > 1), iterates over Z slices.

        Parameters
        ----------
        output_dir : str
            Directory to write frames into (created if needed).
        prefix : str
            Filename prefix. Files will be {prefix}_{NNNN}.{fmt}
        scale : int
            Render resolution multiplier.
        frame_range : tuple(int, int) or None
            (start, end) range, inclusive. Refers to T or Z depending on
            the data shape. None = all frames.
        fmt : str
            Image format, "png" (default) or "tiff".
        """
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)

        # Decide axis: Z-stack when there's no time dimension
        animate_z = self.T == 1 and self.Z > 1
        n_frames = self.Z if animate_z else self.T
        axis_label = "Z" if animate_z else "T"

        if frame_range is None:
            f_start, f_end = 0, n_frames - 1
        else:
            f_start, f_end = int(frame_range[0]), int(frame_range[1])

        ext = "tif" if fmt.lower() in ("tif", "tiff") else fmt.lower()
        total = f_end - f_start + 1

        for i, idx in enumerate(range(f_start, f_end + 1)):
            if animate_z:
                self.on_z_change(idx)
            else:
                self.on_time_change(idx)
            QApplication.processEvents()
            rgba = self.snapshot(scale=scale)
            filename = f"{prefix}_{idx:04d}.{ext}"
            filepath = os.path.join(output_dir, filename)
            Image.fromarray(rgba).save(filepath)
            print(
                f"\r  Exported {axis_label}={idx} ({i + 1}/{total}): {filename}",
                end="", flush=True,
            )

        print()
        print(f"Frames saved to: {output_dir}")
        print(f"To make a movie:")
        print(
            f"  ffmpeg -framerate 10 -i {output_dir}/{prefix}_%04d.{ext}"
            f" -c:v libx264 -pix_fmt yuv420p output.mp4"
        )

    def _save_snapshot_dialog(self):
        """Open file dialog and save a snapshot."""
        default_path = self._default_save_path(".png")
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", default_path,
            "PNG files (*.png);;JPEG files (*.jpg *.jpeg);;TIFF files (*.tif *.tiff)",
        )
        if filepath:
            self.snapshot(filepath)

    def _export_frames_dialog(self):
        """Open directory dialog and export timelapse frames."""
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Frames"
        )
        if not output_dir:
            return
        self.export_frames(output_dir)

    # ---- Label/Mask Methods ----

    # ---- Active mask layer (used by LabelManager) ----

    @property
    def labels(self):
        """SparseLabels of the active mask layer (or None)."""
        if self._active_mask_layer and self._active_mask_layer in self._mask_layers:
            return self._mask_layers[self._active_mask_layer]["labels"]
        return None

    @labels.setter
    def labels(self, value):
        """Set labels on the active mask layer, creating one if needed."""
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
        """LabelOverlayVisual of the active mask layer (or None)."""
        if self._active_mask_layer and self._active_mask_layer in self._mask_layers:
            return self._mask_layers[self._active_mask_layer]["visual"]
        return None

    def set_labels(self, labels):
        """Set SparseLabels on the active mask layer."""
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
                if entry["labels"].ndim == 3:
                    entry["visual"].update_slice(self.z_idx)

    # ---- Shared named-layer-registry bookkeeping ----
    #
    # _mask_layers/_track_layers/_point_layers are parallel OrderedDicts
    # (name -> entry dict) with an identical add/remove/visible/active
    # skeleton around type-specific entry construction. These three
    # helpers hold that skeleton once; per-type add_*_layer/remove_*_layer
    # build the type-specific entry/visual/unified-Layer data and delegate
    # the rest here. Data-replacement (set_*_layer_tracks/points), style,
    # selection, and focus-point methods stay separate per type -- point
    # layers in particular reach into the mouse-drag handlers elsewhere in
    # this file, so that logic is deliberately left alone.

    def _register_layer_entry(
        self, registry, active_attr, name, entry, *, added_signal, layer_type,
        layer_data, visual, style=None,
    ):
        """Store *entry* in *registry*, promote it to active if none is
        active yet, emit *added_signal*, and mirror it into the unified
        ``self.layers`` LayerList. Caller must already have guarded against
        a duplicate *name* and built *entry*/*visual*/*layer_data*."""
        registry[name] = entry
        if getattr(self, active_attr) is None:
            setattr(self, active_attr, name)
        added_signal.emit(name)
        if name not in self.layers:
            kwargs = {} if style is None else {"style": style}
            layer = Layer(
                name=name, layer_type=layer_type, data=layer_data, visual=visual,
                **kwargs,
            )
            self.layers.add(layer)
            self.layer_added.emit(layer)

    def _unregister_layer_entry(
        self, registry, active_attr, name, *, removed_signal, after_pop=None
    ):
        """Pop *name* from *registry*, tear down its visual, reassign the
        active layer if needed, emit *removed_signal*, and remove the
        mirrored entry from ``self.layers``. *after_pop* (if given) runs
        immediately after the pop, before the visual is torn down -- for
        type-specific cleanup that must happen first (e.g. unsubscribing).
        Returns the popped entry."""
        entry = registry.pop(name)
        if after_pop is not None:
            after_pop(entry)
        visual = entry.get("visual")
        if visual is not None:
            visual.remove()
        if getattr(self, active_attr) == name:
            setattr(self, active_attr, next(iter(registry), None))
        removed_signal.emit(name)
        if name in self.layers:
            removed = self.layers.remove(name)
            self.layer_removed.emit(removed)
        self.canvas.update()
        return entry

    def _set_layer_visible(self, registry, name, visible):
        """Toggle visibility for a named layer entry in *registry*."""
        if name not in registry:
            return
        visible = bool(visible)
        registry[name]["visible"] = visible
        visual = registry[name]["visual"]
        if visual is not None:
            visual.visible = visible
        self.canvas.update()

    def _set_active_layer(self, registry, active_attr, name):
        """Switch the active layer name for *registry*. Returns whether
        *name* was found (and thus actually switched)."""
        if name not in registry:
            return False
        setattr(self, active_attr, name)
        return True

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
        if labels.ndim == 3:
            visual.update_slice(self.z_idx)
        self._register_layer_entry(
            self._mask_layers, "_active_mask_layer", name,
            {"labels": labels, "visual": visual, "visible": True},
            added_signal=self.mask_layer_added,
            layer_type="labels", layer_data=labels, visual=visual,
        )

    def remove_mask_layer(self, name):
        """Remove a named mask layer and clean up its visual."""
        if name not in self._mask_layers:
            return
        self._unregister_layer_entry(
            self._mask_layers, "_active_mask_layer", name,
            removed_signal=self.mask_layer_removed,
        )

    def set_mask_layer_visible(self, name, visible):
        """Toggle visibility of a specific mask layer."""
        self._set_layer_visible(self._mask_layers, name, visible)

    def set_active_mask_layer(self, name):
        """Switch which mask layer receives painting."""
        self._set_active_layer(self._mask_layers, "_active_mask_layer", name)

    # ---- Multiple Track Layers API ----

    def add_track_layer(self, name, tracks=None, trail_window=30):
        """Create a named track layer with its own visual."""
        if name in self._track_layers:
            return

        if tracks is None:
            tracks = TrackTable.from_arrays(track_id=[], t=[], x=[], y=[])
        elif not isinstance(tracks, TrackTable):
            raise TypeError("tracks must be a TrackTable or None")

        style = dict(TRACK_DEFAULT_STYLE)
        style["trail_window"] = trail_window
        visual = TrackLayerVisual(self.view, **style)
        visual.set_tracks(tracks)
        visual.set_time_z(self.t_idx, self.z_idx)
        self._register_layer_entry(
            self._track_layers, "_active_track_layer", name,
            {"tracks": tracks, "visual": visual, "visible": True, "style": style},
            added_signal=self.track_layer_added,
            layer_type="tracks", layer_data=tracks, visual=visual, style=style,
        )
        self.canvas.update()

    def set_track_layer_tracks(self, name, tracks):
        """Replace data for a specific track layer."""
        if name not in self._track_layers:
            return
        if not isinstance(tracks, TrackTable):
            raise TypeError("tracks must be a TrackTable")

        self._track_layers[name]["tracks"] = tracks
        if name in self.layers:
            self.layers[name].data = tracks
        visual = self._track_layers[name]["visual"]
        if visual is not None:
            visual.set_tracks(tracks)
            visual.set_time_z(self.t_idx, self.z_idx)
        self.canvas.update()

    def set_track_layer_style(self, name, **style):
        """Update style for a specific track layer."""
        if name not in self._track_layers:
            return
        entry = self._track_layers[name]
        visual = entry["visual"]
        if visual is None:
            return
        visual.set_style(**style)
        entry["style"].update(style)
        self.canvas.update()

    def remove_track_layer(self, name):
        """Remove a named track layer and clean up its visual."""
        if name not in self._track_layers:
            return
        self._unregister_layer_entry(
            self._track_layers, "_active_track_layer", name,
            removed_signal=self.track_layer_removed,
        )

    def set_track_layer_visible(self, name, visible):
        """Toggle visibility of a specific track layer."""
        self._set_layer_visible(self._track_layers, name, visible)

    def set_active_track_layer(self, name):
        """Switch which track layer is active."""
        self._set_active_layer(self._track_layers, "_active_track_layer", name)

    def remove_track_from_layer(self, layer_name, track_id):
        """Remove a single trajectory from a track layer by track ID."""
        if layer_name not in self._track_layers:
            return
        tracks = self._track_layers[layer_name]["tracks"]
        updated = tracks.remove_track(track_id)
        self.set_track_layer_tracks(layer_name, updated)

    # ---- Multiple Point Layers API ----

    def add_point_layer(self, name, points=None):
        """Create a named point layer with its own visual.

        Internally wraps the :class:`PointTable` in a
        :class:`PointDataHolder` so that the unified layer commands
        (``AddPoint``/``RemovePoint``/``MovePoint``) operate on the same
        in-place object. The legacy ``self._point_layers[name]["points"]``
        attribute keeps pointing at the current immutable table — kept in
        sync via a :meth:`PointDataHolder.subscribe` callback.
        """
        if name in self._point_layers:
            return

        from ..data.point_commands import PointDataHolder

        if points is None:
            points = PointTable.from_arrays(x=[], y=[])
        elif not isinstance(points, PointTable):
            raise TypeError("points must be a PointTable or None")

        holder = PointDataHolder(points)
        style = dict(POINT_DEFAULT_STYLE)
        visual = PointLayerVisual(self.view, **style)
        visual.set_points(holder.table)
        visual.set_time_z(self.t_idx, self.z_idx)
        # Registered with the holder as the unified-LayerList data (not the
        # table) so the standard AddPoint/RemovePoint/MovePoint commands
        # operate on the same in-place object.
        self._register_layer_entry(
            self._point_layers, "_active_point_layer", name,
            {
                "points": holder.table,
                "holder": holder,
                "visual": visual,
                "visible": True,
                "selected_ids": set(),
                "style": style,
            },
            added_signal=self.point_layer_added,
            layer_type="points", layer_data=holder, visual=visual, style=style,
        )

        def _on_point_event(_kind, _pid, _name=name):
            entry = self._point_layers.get(_name)
            if entry is None:
                return
            entry["points"] = entry["holder"].table
            v = entry["visual"]
            if v is not None:
                v.set_points(entry["points"])
                v.set_time_z(self.t_idx, self.z_idx)
                v.set_selected_point_ids(entry["selected_ids"])
            self.canvas.update()

        self._point_layers[name]["_unsubscribe"] = holder.subscribe(_on_point_event)

        self.canvas.update()

    def set_point_layer_points(self, name, points):
        """Replace data for a specific point layer."""
        if name not in self._point_layers:
            return
        if not isinstance(points, PointTable):
            raise TypeError("points must be a PointTable")

        self._point_layers[name]["points"] = points
        visual = self._point_layers[name]["visual"]
        if visual is not None:
            visual.set_points(points)
            visual.set_time_z(self.t_idx, self.z_idx)
            visual.set_selected_point_ids(self._point_layers[name]["selected_ids"])
        self.canvas.update()

    def remove_point_layer(self, name):
        """Remove a named point layer and clean up its visual."""
        if name not in self._point_layers:
            return

        def _unsubscribe(entry):
            unsub = entry.get("_unsubscribe")
            if unsub is not None:
                try:
                    unsub()
                except Exception:
                    pass

        self._unregister_layer_entry(
            self._point_layers, "_active_point_layer", name,
            removed_signal=self.point_layer_removed, after_pop=_unsubscribe,
        )
        if self._focused_point_layer == name:
            self.clear_focused_point()

    def set_point_layer_visible(self, name, visible):
        """Toggle visibility of a specific point layer."""
        self._set_layer_visible(self._point_layers, name, visible)

    def set_active_point_layer(self, name):
        """Switch which point layer is active."""
        if self._set_active_layer(self._point_layers, "_active_point_layer", name):
            self.canvas.update()

    def remove_point_from_layer(self, layer_name, point_id):
        """Remove a single point from a point layer by point ID."""
        if layer_name not in self._point_layers:
            return
        points = self._point_layers[layer_name]["points"]
        updated = points.remove_point(point_id)
        self.set_point_layer_points(layer_name, updated)
        if self._focused_point_layer == layer_name and int(self._focused_point_id or -1) == int(point_id):
            self.clear_focused_point()

    def set_point_layer_style(self, name, **style):
        """Update style for a specific point layer."""
        if name not in self._point_layers:
            return
        entry = self._point_layers[name]
        visual = entry["visual"]
        if visual is None:
            return
        visual.set_style(**style)
        entry["style"].update(style)
        if self._focused_point_layer == name and self._focused_point_id is not None:
            self.focus_point(name, self._focused_point_id)
        self.canvas.update()

    def select_points_in_layer(self, name, point_ids):
        """Select a set of point IDs in a point layer."""
        if name not in self._point_layers:
            return
        ids = {int(v) for v in point_ids}
        entry = self._point_layers[name]
        entry["selected_ids"] = ids
        visual = entry["visual"]
        if visual is not None:
            visual.set_selected_point_ids(ids)
        self.canvas.update()

    # ------------------------------------------------------------------
    # Shape layer methods (new unified layer system)
    # ------------------------------------------------------------------

    def add_shape_layer(self, name, data=None):
        """Create a named shape layer using the unified layer system.

        Parameters
        ----------
        name : str
            Unique layer name.
        data : ShapeData, optional
            Pre-populated shape data. If None, an empty ShapeData is created.
        """
        if name in self.layers:
            return
        if data is None:
            data = ShapeData()
        visual = ShapeLayerVisual(self.view.scene)
        layer = Layer(name=name, layer_type="shapes", data=data, visual=visual)
        self.layers.add(layer)
        visual.update(data, layer.selected_ids, self.t_idx, self.z_idx)

        # Auto-refresh on any data mutation. Subscribers fire on the caller's
        # thread; shapes always mutate on the GUI thread, so a direct call is
        # safe. The unsubscribe handle is stashed in ``layer.style`` so the
        # subscription is dropped on remove.
        def _on_shape_event(_event_kind, _shape_id, _layer=layer):
            if str(_event_kind).startswith("gel_marker_"):
                return
            if _layer.visual is None:
                return
            _layer.visual.update(
                _layer.data, _layer.selected_ids, self.t_idx, self.z_idx
            )
            self.canvas.update()

        layer.style["_unsubscribe_data"] = data.subscribe(_on_shape_event)

        self.layer_added.emit(layer)
        self.canvas.update()
        return layer

    def remove_shape_layer(self, name):
        """Remove a shape layer by name."""
        if name not in self.layers:
            return
        layer = self.layers.remove(name)
        unsub = layer.style.pop("_unsubscribe_data", None)
        if unsub is not None:
            try:
                unsub()
            except Exception:
                pass
        if layer.visual is not None:
            layer.visual.remove()
        self.layer_removed.emit(layer)
        self.canvas.update()

    def _refresh_shape_layers(self):
        """Rebuild visuals for all shape layers at current time/z."""
        for layer in self.layers.by_type("shapes"):
            if layer.visual is not None:
                layer.visual.update(
                    layer.data, layer.selected_ids, self.t_idx, self.z_idx
                )

    def _active_shape_layer(self):
        """Get the currently active shape layer, or None."""
        return self.layers.active("shapes")

    def _hit_test_shape_layers(self, x, y):
        """Hit-test shape-layer entries at (x, y) on the current t/z slice.

        Tests layers in reverse order (top-most first). For each layer,
        first tries handles of currently-selected shapes (so an edge grab
        wins over a body click), then tests bodies.

        Returns
        -------
        (layer, shape_id, handle) or None
            handle is a string for handle hits, None for body hits.
        """
        layers = list(self.layers.by_type("shapes"))
        for layer in reversed(layers):
            if not layer.visible:
                continue
            data = layer.data
            # Handle hits on selected shapes take priority.
            for sid in layer.selected_ids:
                if sid not in data:
                    continue
                rec = data.get(sid)
                if not self._shape_visible_in_current_slice(rec):
                    continue
                handle = data.hit_test_handle((x, y), sid)
                if handle is not None:
                    return layer, sid, handle
            sid = data.hit_test((x, y), t=self.t_idx, z=self.z_idx)
            if sid is not None:
                return layer, sid, None
        return None

    def _shape_visible_in_current_slice(self, rec) -> bool:
        """Return whether a shape should be interactive in this T/Z view."""
        return (
            (rec.t == self.t_idx or rec.t == ALL_FRAMES)
            and (rec.z == self.z_idx or rec.z == ALL_FRAMES)
        )

    def _select_shape(self, layer, shape_id):
        """Make shape_id the sole selection on its layer, deselect elsewhere."""
        for lyr in self.layers.by_type("shapes"):
            if lyr is layer:
                lyr.selected_ids = {shape_id}
            else:
                lyr.selected_ids = set()
            if lyr.visual is not None:
                lyr.visual.update(
                    lyr.data, lyr.selected_ids, self.t_idx, self.z_idx
                )
        self.shape_selection_changed.emit(layer, shape_id)

    def _clear_shape_selection(self):
        """Deselect all shapes across all shape layers."""
        had_selection = any(lyr.selected_ids for lyr in self.layers.by_type("shapes"))
        for lyr in self.layers.by_type("shapes"):
            if not lyr.selected_ids:
                continue
            lyr.selected_ids = set()
            if lyr.visual is not None:
                lyr.visual.update(
                    lyr.data, lyr.selected_ids, self.t_idx, self.z_idx
                )
        if had_selection:
            self.shape_selection_changed.emit(None, None)

    def _hit_test_gel_marker(self, rec, x: float, y: float) -> int | None:
        """Return gel marker index under the pointer for a gel-lane shape."""
        if rec.shape_type != RECTANGLE or not rec.properties.get("gel_lane"):
            return None
        markers = rec.properties.get("gel_markers") or []
        if not markers:
            return None
        x1, y1, x2, y2 = rec.params[:4]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min = min(y1, y2)
        if not (x_min <= x <= x_max):
            return None
        for i, marker in enumerate(markers):
            try:
                y_global = y_min + float(marker.get("y_local", 0.0))
            except (TypeError, ValueError):
                continue
            if abs(y - y_global) < 5:
                return i
        return None

    def _add_gel_marker(self, layer, shape_id: int, y: float) -> None:
        rec = layer.data.get(shape_id)
        x1, y1, x2, y2 = rec.params[:4]
        y_min, y_max = min(y1, y2), max(y1, y2)
        y_local = max(0.0, min(float(y - y_min), float(y_max - y_min)))
        markers = list(rec.properties.get("gel_markers") or [])
        color = rec.properties.get("gel_marker_color", "#00BCD4")
        markers.append({"y_local": y_local, "label": "", "color": color})
        rec.properties["gel_lane"] = True
        rec.properties["gel_markers"] = markers
        layer.data._emit(EVT_EDITED, shape_id)
        self.canvas.update()

    def _remove_gel_marker(self, layer, shape_id: int, marker_idx: int) -> None:
        rec = layer.data.get(shape_id)
        markers = list(rec.properties.get("gel_markers") or [])
        if 0 <= marker_idx < len(markers):
            markers.pop(marker_idx)
            rec.properties["gel_markers"] = markers
            layer.data._emit("gel_marker_changed", shape_id)
            self.canvas.update()

    def _move_gel_marker(self, layer, shape_id: int, marker_idx: int, y: float) -> None:
        rec = layer.data.get(shape_id)
        markers = list(rec.properties.get("gel_markers") or [])
        if not (0 <= marker_idx < len(markers)):
            return
        x1, y1, x2, y2 = rec.params[:4]
        y_min, y_max = min(y1, y2), max(y1, y2)
        y_local = max(0.0, min(float(y - y_min), float(y_max - y_min)))
        marker = dict(markers[marker_idx])
        marker["y_local"] = y_local
        markers[marker_idx] = marker
        rec.properties["gel_markers"] = markers
        layer.data._emit("gel_marker_moved", shape_id)
        self.canvas.update()

    def _polyline_segment_insert_index(self, rec, x, y, tolerance=8.0):
        """If ``(x, y)`` is near a polyline segment, return the index at
        which a new vertex should be inserted. Otherwise return ``None``.
        """
        if rec.shape_type != POLYLINE or rec.vertices is None:
            return None
        verts = rec.vertices
        n = len(verts)
        if n < 2:
            return None
        closed = polyline_is_closed(rec)
        best_idx = None
        best_dist = float(tolerance)
        segments = n - 1 + (1 if closed else 0)
        for i in range(segments):
            ax, ay = float(verts[i, 0]), float(verts[i, 1])
            j = (i + 1) % n
            bx, by = float(verts[j, 0]), float(verts[j, 1])
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            if l2 == 0:
                continue
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / l2))
            qx = ax + t * dx
            qy = ay + t * dy
            d = float(np.hypot(x - qx, y - qy))
            if d < best_dist:
                best_dist = d
                best_idx = i + 1
        return best_idx

    def _current_shape_selection(self):
        """Return ``(layer, shape_id)`` if exactly one shape is selected, else None."""
        hit = None
        for lyr in self.layers.by_type("shapes"):
            for sid in lyr.selected_ids:
                if hit is not None:
                    return None  # multiple selections — ambiguous
                hit = (lyr, sid)
        return hit

    def _refresh_shape_layer_visual(self, layer):
        if layer.visual is not None:
            layer.visual.update(
                layer.data, layer.selected_ids, self.t_idx, self.z_idx
            )

    def _finalize_shape_drawing(self) -> None:
        """Clear active draw state and select the newly-created shape.

        A click with no drag (press and release at the same point, e.g. a
        fast click where no intermediate mouse-move ever arrives) leaves a
        zero-extent rect/circle/line: invisible on canvas but still listed
        in the layer manager. Discard it instead of keeping that stray
        shape around.
        """
        if self._drawing_shape_layer is None:
            return
        layer = self._drawing_shape_layer
        shape_id = self._drawing_shape_id
        cmd = self._drawing_shape_cmd
        self._drawing_shape_layer = None
        self._drawing_shape_id = None
        self._drawing_shape_cmd = None
        self.start_pos = None
        if shape_id is None or shape_id not in layer.data:
            return
        rec = layer.data.get(shape_id)
        x1, y1, x2, y2 = rec.params[:4]
        if np.hypot(x2 - x1, y2 - y1) < 1e-6:
            # Retract by identity, not a blind top-of-stack pop: if some
            # other command got pushed onto this layer in between (e.g. a
            # keyboard nudge/delete of a different, still-selected shape
            # while the mouse was held down), pop_if_top() is a no-op and
            # we leave the degenerate shape in place rather than risk
            # reverting the wrong command.
            layer.undo_stack.pop_if_top(cmd, layer.data)
            return
        self._select_shape(layer, shape_id)

    def _finalize_shape_edit(self) -> None:
        """Record the live shape edit as a single undoable command."""
        if self._editing_shape_id is None or self._editing_shape_layer is None:
            return
        layer = self._editing_shape_layer
        shape_id = self._editing_shape_id
        rec = layer.data.get(shape_id)
        old_params = self._editing_shape_start_params
        new_params = rec.params.copy()
        params_changed = (
            old_params is not None and not np.array_equal(old_params, new_params)
        )
        old_verts = self._editing_shape_start_vertices
        new_verts = rec.vertices.copy() if rec.vertices is not None else None
        verts_changed = (
            old_verts is not None and new_verts is not None
            and not np.array_equal(old_verts, new_verts)
        )
        if params_changed:
            layer.undo_stack.push_executed(
                SetShapeParams(shape_id, old_params, new_params)
            )
        if verts_changed:
            layer.undo_stack.push_executed(
                SetShapeVertices(shape_id, old_verts, new_verts)
            )
        self._editing_shape_layer = None
        self._editing_shape_id = None
        self._editing_shape_handle = None
        self._editing_shape_start_params = None
        self._editing_shape_start_vertices = None
        self._editing_shape_start_pos = None

    def _finalize_polyline_drawing(self) -> None:
        """Finish a polyline preview when leaving the polyline tool."""
        if self._polyline_drawing_id is None or self._polyline_drawing_layer is None:
            return
        layer = self._polyline_drawing_layer
        sid = self._polyline_drawing_id
        try:
            rec = layer.data.get(sid)
        except KeyError:
            self._polyline_drawing_layer = None
            self._polyline_drawing_id = None
            return
        if rec.vertices is not None:
            if len(rec.vertices) >= 3:
                rec.vertices = rec.vertices[:-1].astype(np.float32, copy=False)
                layer.data._emit(EVT_EDITED, sid)
                self._select_shape(layer, sid)
            elif layer.undo_stack.can_undo:
                layer.undo_stack.undo(layer.data)
            elif sid in layer.data:
                layer.data.remove(sid)
        self._polyline_drawing_layer = None
        self._polyline_drawing_id = None

    def _on_active_tool_changed(self, tool_name: str) -> None:
        """Keep transient shape state consistent when shortcuts/toolbars switch tools."""
        self._finalize_shape_drawing()
        self._finalize_shape_edit()
        if tool_name != "polyline":
            self._finalize_polyline_drawing()
        self.update_cursor()

    def delete_selected_shape(self):
        """Remove the currently-selected shape (undoable). No-op if none."""
        sel = self._current_shape_selection()
        if sel is None:
            return
        layer, sid = sel
        layer.undo_stack.push(RemoveShape(sid), layer.data)
        layer.selected_ids.discard(sid)
        self._refresh_shape_layer_visual(layer)
        self.canvas.update()

    def rename_selected_shape(self):
        """Prompt for a new label for the selected shape (undoable)."""
        sel = self._current_shape_selection()
        if sel is None:
            return
        layer, sid = sel
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
        self._refresh_shape_layer_visual(layer)
        self.canvas.update()

    def nudge_selected_shape(self, dx: float, dy: float) -> None:
        """Translate the currently-selected shape by (dx, dy) pixels."""
        sel = self._current_shape_selection()
        if sel is None or (dx == 0 and dy == 0):
            return
        layer, sid = sel
        layer.undo_stack.push(MoveShape(sid, dx, dy), layer.data)

    def edit_selected_shape_properties(self) -> None:
        """Open the numerical properties dialog for the selected shape."""
        sel = self._current_shape_selection()
        if sel is None:
            return
        layer, sid = sel
        from ..widgets.shape_properties_dialog import ShapePropertiesDialog

        dlg = ShapePropertiesDialog(self, layer, sid, parent=self)
        dlg.show()
        dlg.raise_()

    def _reset_vispy_press_state(self):
        """Clear vispy's tracked ``press_event``.

        QMenu.exec_() and modal dialogs swallow the mouse release that
        belongs to the press which opened them, so vispy's backend never
        sees the matching release and ``_vispy_mouse_data['press_event']``
        stays populated. After that, every ``mouse_move`` overrides
        ``event.button`` with the stale press's button (see
        ``vispy.app.base.BaseCanvasBackend._vispy_mouse_move``), which
        breaks any drag handler that filters on ``event.button == 1``.

        Must be scheduled (not called synchronously) when invoked from
        inside an ``on_mouse_press`` handler: vispy's backend assigns
        ``press_event = ev`` *after* the handler returns
        (``vispy/app/base.py:184``), so a synchronous reset gets
        immediately overwritten. ``QTimer.singleShot(0, ...)`` defers the
        clear to the next event-loop tick, after vispy has finished
        dispatching the press.
        """
        def _clear():
            backend = getattr(self.canvas, "_backend", None)
            mouse_data = getattr(backend, "_vispy_mouse_data", None)
            if isinstance(mouse_data, dict):
                mouse_data["press_event"] = None
                mouse_data["last_event"] = None

        QTimer.singleShot(0, _clear)

    def _show_shape_context_menu(self, layer, shape_id, global_pos):
        """Right-click menu for a shape."""
        rec = layer.data.get(shape_id)
        menu = QMenu(self)
        act_props = menu.addAction("Edit Properties…")
        act_rename = menu.addAction("Rename…")
        menu.addSeparator()
        act_delete = menu.addAction("Delete")

        # Send copy to another window (XY-compatible).
        from .manager import compatible_windows
        targets = compatible_windows(self)
        copy_actions: dict = {}
        if targets:
            menu.addSeparator()
            copy_menu = menu.addMenu("Send Copy To Window")
            for tgt in targets:
                title = tgt.windowTitle() or "window"
                label = title
                # Annotate scale mismatch — copy is still allowed.
                try:
                    s_src = (self.meta or {}).get("scale") or ()
                    s_tgt = (tgt.meta or {}).get("scale") or ()
                    if s_src and s_tgt and tuple(s_src) != tuple(s_tgt):
                        label = f"{title}  ⚠ scale differs"
                except Exception:
                    pass
                action = copy_menu.addAction(label)
                copy_actions[action] = tgt

        act_crop = None
        act_stats = None
        act_kymo = None
        act_close = None
        smooth_actions: dict = {}
        if rec.shape_type in (RECTANGLE, CIRCLE, POLYLINE):
            menu.addSeparator()
            if rec.shape_type == RECTANGLE:
                act_crop = menu.addAction("Crop…")
            act_stats = menu.addAction("Region Statistics…")
        act_profile = None
        if rec.shape_type in (LINE, POLYLINE):
            menu.addSeparator()
            act_profile = menu.addAction("Line Profile…")
            # Kymograph requires a time-series.
            if self.img_data.shape[0] > 1:
                act_kymo = menu.addAction("Kymograph…")
        act_radial_profile = None
        if rec.shape_type == CIRCLE:
            menu.addSeparator()
            act_radial_profile = menu.addAction("Radial Profile…")
        if rec.shape_type == POLYLINE:
            menu.addSeparator()
            currently_closed = polyline_is_closed(rec)
            act_close = menu.addAction(
                "Open path" if currently_closed else "Close path"
            )
            smooth_menu = menu.addMenu("Smoothness")
            current_s = float(rec.properties.get("smoothness", 0.0))
            for value, label in (
                (0.0, "0 — straight segments"),
                (0.5, "0.5"),
                (2.0, "2"),
                (5.0, "5"),
            ):
                action = smooth_menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(abs(current_s - value) < 1e-6)
                smooth_actions[action] = value

        # Plugin-contributed entries (see pyvistra.plugins.register_shape_context_action).
        from ..plugins import shape_context_actions_for
        plugin_entries = shape_context_actions_for(ImageWindow, rec.shape_type)
        plugin_actions: dict = {}
        if plugin_entries:
            menu.addSeparator()
            for label, method, tooltip in plugin_entries:
                action = menu.addAction(label)
                if tooltip:
                    action.setToolTip(tooltip)
                plugin_actions[action] = method

        chosen = menu.exec_(global_pos)
        if chosen is act_props:
            # Make sure the right shape is the single selection.
            self._select_shape(layer, shape_id)
            self.edit_selected_shape_properties()
        elif chosen is act_rename:
            self._select_shape(layer, shape_id)
            self.rename_selected_shape()
        elif chosen is act_delete:
            self._select_shape(layer, shape_id)
            self.delete_selected_shape()
        elif act_crop is not None and chosen is act_crop:
            self._crop_with_rect(layer, shape_id)
        elif chosen in plugin_actions:
            self._select_shape(layer, shape_id)
            getattr(self, plugin_actions[chosen])(layer, shape_id)
        elif act_stats is not None and chosen is act_stats:
            from ..widgets.region_statistics_dialog import RegionStatisticsDialog
            dlg = RegionStatisticsDialog(self, layer, shape_id, parent=self)
            dlg.show()
            dlg.raise_()
        elif act_kymo is not None and chosen is act_kymo:
            from ..widgets.kymograph_dialog import KymographDialog
            dlg = KymographDialog(self, layer, shape_id, parent=self)
            dlg.show()
            dlg.raise_()
        elif act_profile is not None and chosen is act_profile:
            from ..widgets import get_line_profile_dialog
            dlg = get_line_profile_dialog()
            dlg.set_shape_source(self, layer, shape_id)
            dlg.show()
            dlg.raise_()
        elif act_radial_profile is not None and chosen is act_radial_profile:
            from ..widgets import get_radial_profile_dialog
            dlg = get_radial_profile_dialog()
            dlg.set_shape_source(self, layer, shape_id)
            dlg.show()
            dlg.raise_()
        elif act_close is not None and chosen is act_close:
            layer.undo_stack.push(
                SetPolylineFlags(shape_id, closed=not polyline_is_closed(rec)),
                layer.data,
            )
        elif chosen in smooth_actions:
            layer.undo_stack.push(
                SetPolylineFlags(shape_id, smoothness=smooth_actions[chosen]),
                layer.data,
            )
        elif chosen in copy_actions:
            self._copy_shape_to_window(layer, shape_id, copy_actions[chosen])

    def _copy_shape_to_window(self, src_layer, shape_id, target_window):
        """Clone a shape into ``target_window``'s active shape layer.

        If the target has no shape layer, a default ``"Shapes"`` one is
        created. The copy is independent — no live link is established.
        Anchors (``t``, ``z``) are translated through the ``ALL_FRAMES``
        sentinel if applicable; otherwise they're carried verbatim.
        """
        if target_window is None or src_layer is None:
            return
        if not hasattr(target_window, "add_shape_layer"):
            return
        target_layer = target_window.layers.active("shapes")
        if target_layer is None:
            target_window.add_shape_layer("Shapes")
            target_layer = target_window.layers.active("shapes")
            if target_layer is None:
                return

        rec = src_layer.data.get(shape_id)
        verts = None if rec.vertices is None else rec.vertices.copy()
        cmd = AddShape(
            rec.shape_type,
            rec.params.copy(),
            t=rec.t,
            z=rec.z,
            label=rec.label,
            properties=dict(rec.properties),
            vertices=verts,
        )
        target_layer.undo_stack.push(cmd, target_layer.data)
        target_window.raise_()
        target_window.activateWindow()

    def _crop_with_rect(self, layer, shape_id):
        """Crop ``self.img_data`` to a rectangle's XY bounds and route the
        result through ``ImageOutputSelector`` (reuse window / new window /
        save). Supports optional T, Z, and C subranges.
        """
        from qtpy.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QGroupBox,
            QSpinBox,
        )
        from ..widgets.output_selector import ImageOutputSelector

        rec = layer.data.get(shape_id)
        try:
            # Get the XY cropped data first to determine available dimensions
            xy_cropped = crop_rect(self.img_data, rec)
        except ValueError as e:
            QMessageBox.warning(self, "Crop", str(e))
            return

        T, Z, C, Yc, Xc = xy_cropped.shape
        
        meta_channels = (self.meta or {}).get("channels") or []
        
        dlg = QDialog(self)
        dlg.setWindowTitle("Crop region")
        dlg_layout = QVBoxLayout(dlg)
        label = rec.label or f"shape {shape_id}"
        dlg_layout.addWidget(
            QLabel(f"Cropping {Yc}×{Xc} (Y×X) from '{label}' — T={T}, Z={Z}, C={C}")
        )

        # Channel range
        c_group = QGroupBox(f"Channel range (C = {C})")
        c_form = QFormLayout(c_group)
        c_from = QSpinBox()
        c_from.setRange(0, C - 1)
        c_from.setValue(0)
        c_to = QSpinBox()
        c_to.setRange(0, C - 1)
        c_to.setValue(C - 1)
        c_form.addRow("From:", c_from)
        c_form.addRow("To:", c_to)
        c_group.setEnabled(C > 1)
        dlg_layout.addWidget(c_group)

        # T-range
        t_group = QGroupBox(f"Time range (T = {T})")
        t_form = QFormLayout(t_group)
        t_from = QSpinBox()
        t_from.setRange(0, T - 1)
        t_from.setValue(0)
        t_to = QSpinBox()
        t_to.setRange(0, T - 1)
        t_to.setValue(T - 1)
        t_form.addRow("From:", t_from)
        t_form.addRow("To:", t_to)
        t_group.setEnabled(T > 1)
        dlg_layout.addWidget(t_group)

        # Z-range
        z_group = QGroupBox(f"Z range (Z = {Z})")
        z_form = QFormLayout(z_group)
        z_from = QSpinBox()
        z_from.setRange(0, Z - 1)
        z_from.setValue(0)
        z_to = QSpinBox()
        z_to.setRange(0, Z - 1)
        z_to.setValue(Z - 1)
        z_form.addRow("From:", z_from)
        z_form.addRow("To:", z_to)
        z_group.setEnabled(Z > 1)
        dlg_layout.addWidget(z_group)

        # Output destination.
        default_title = f"{self.windowTitle()} [crop]"
        selector = ImageOutputSelector(
            parent=dlg, default_title=default_title, formats=[".tif", ".ims"]
        )
        dlg_layout.addWidget(selector)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        if dlg.exec_() != QDialog.Accepted:
            return

        c0, c1 = sorted((int(c_from.value()), int(c_to.value())))
        t0, t1 = sorted((int(t_from.value()), int(t_to.value())))
        z0, z1 = sorted((int(z_from.value()), int(z_to.value())))

        # Apply the full 5D crop: T, Z, C, Y, X
        try:
            cropped = np.asarray(xy_cropped[t0:t1 + 1, z0:z1 + 1, c0:c1 + 1, :, :])
        except Exception as e:
            QMessageBox.warning(self, "Crop", f"Could not crop: {e}")
            return

        # Build channel metadata pared down to the selected range.
        metadata = dict(self.meta or {})
        if meta_channels and c0 <= c1 and c1 < len(meta_channels):
            metadata["channels"] = [dict(meta_channels[c]) for c in range(c0, c1 + 1)]
        base_name = metadata.get("filename") or self.windowTitle() or "crop"
        
        # Build a descriptive suffix for the crop
        suffix_parts = []
        if t0 > 0 or t1 < T - 1:
            suffix_parts.append(f"T{t0}-{t1}")
        if z0 > 0 or z1 < Z - 1:
            suffix_parts.append(f"Z{z0}-{z1}")
        if c0 > 0 or c1 < C - 1:
            if c0 == c1:
                ch_name = ""
                if meta_channels and c0 < len(meta_channels):
                    ch_name = (meta_channels[c0] or {}).get("name") or ""
                suffix_parts.append(f"ch{c0}" + (f" {ch_name}" if ch_name else ""))
            else:
                suffix_parts.append(f"C{c0}-{c1}")
        
        suffix = " ".join(suffix_parts)
        if suffix:
            metadata["filename"] = f"{base_name} [{suffix}]"
        else:
            metadata["filename"] = f"{base_name} [crop]"

        selector.send(cropped, metadata)

    def _save_channel_dialog(self):
        """Extract one channel (and an optional T/Z subrange) and route the
        result through ``ImageOutputSelector``.

        The output keeps the 5D ``(T, Z, C, Y, X)`` shape with ``C=1`` so it
        round-trips through every existing saver/window.
        """
        from qtpy.QtWidgets import (
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QGroupBox,
            QSpinBox,
        )
        from ..widgets.output_selector import ImageOutputSelector

        T, Z, C, _Y, _X = self.img_data.shape
        if C <= 0:
            QMessageBox.warning(self, "Save Channel", "No channels to save.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Save Channel")
        layout = QVBoxLayout(dlg)

        # Channel picker — use metadata names if available.
        form = QFormLayout()
        channel_combo = QComboBox()
        meta_channels = (self.meta or {}).get("channels") or []
        for c in range(C):
            name = ""
            if c < len(meta_channels) and isinstance(meta_channels[c], dict):
                name = meta_channels[c].get("name") or ""
            label = f"Channel {c}" + (f" — {name}" if name else "")
            channel_combo.addItem(label, c)
        # Default to the currently-displayed channel.
        channel_combo.setCurrentIndex(min(self.c_idx, C - 1))
        form.addRow("Channel:", channel_combo)
        layout.addLayout(form)

        # T-range
        t_group = QGroupBox(f"Time range (T = {T})")
        t_form = QFormLayout(t_group)
        t_from = QSpinBox(); t_from.setRange(0, T - 1); t_from.setValue(0)
        t_to = QSpinBox(); t_to.setRange(0, T - 1); t_to.setValue(T - 1)
        t_form.addRow("From:", t_from)
        t_form.addRow("To:", t_to)
        t_group.setEnabled(T > 1)
        layout.addWidget(t_group)

        # Z-range
        z_group = QGroupBox(f"Z range (Z = {Z})")
        z_form = QFormLayout(z_group)
        z_from = QSpinBox(); z_from.setRange(0, Z - 1); z_from.setValue(0)
        z_to = QSpinBox(); z_to.setRange(0, Z - 1); z_to.setValue(Z - 1)
        z_form.addRow("From:", z_from)
        z_form.addRow("To:", z_to)
        z_group.setEnabled(Z > 1)
        layout.addWidget(z_group)

        # Output destination.
        default_title = f"{self.windowTitle()} [ch{channel_combo.currentData()}]"
        selector = ImageOutputSelector(
            parent=dlg, default_title=default_title, formats=[".tif", ".ims"]
        )
        layout.addWidget(selector)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec_() != QDialog.Accepted:
            return

        c = int(channel_combo.currentData())
        t0, t1 = sorted((int(t_from.value()), int(t_to.value())))
        z0, z1 = sorted((int(z_from.value()), int(z_to.value())))

        # Slice with [c:c+1] to keep the channel axis intact (length 1).
        try:
            subset = np.asarray(
                self.img_data[t0:t1 + 1, z0:z1 + 1, c:c + 1, :, :]
            )
        except Exception as e:
            QMessageBox.warning(self, "Save Channel", f"Could not extract: {e}")
            return

        # Build channel metadata pared down to the single selected channel.
        metadata = dict(self.meta or {})
        if meta_channels and c < len(meta_channels):
            metadata["channels"] = [dict(meta_channels[c])]
        base_name = metadata.get("filename") or self.windowTitle() or "image"
        ch_name = ""
        if meta_channels and c < len(meta_channels):
            ch_name = (meta_channels[c] or {}).get("name") or ""
        suffix = f"ch{c}" + (f" {ch_name}" if ch_name else "")
        metadata["filename"] = f"{base_name} [{suffix}]"

        selector.send(subset, metadata)

    def clear_focused_point(self):
        if self._focused_point_roi is not None:
            self._focused_point_roi.remove()
            self._focused_point_roi = None
        self._focused_point_layer = None
        self._focused_point_id = None
        self.canvas.update()

    def focus_point(self, layer_name, point_id):
        """Create/update transient PointROI for focused point editing."""
        if layer_name not in self._point_layers:
            self.clear_focused_point()
            return
        entry = self._point_layers[layer_name]
        points = entry["points"]
        if points is None:
            self.clear_focused_point()
            return
        row = points.get_point(int(point_id))
        if row is None:
            self.clear_focused_point()
            return

        self._focused_point_layer = layer_name
        self._focused_point_id = int(point_id)
        self.select_points_in_layer(layer_name, {int(point_id)})

        if self._focused_point_roi is None:
            self._focused_point_roi = PointROI(
                self.view,
                name=f"point:{int(point_id)}",
                point_id=int(point_id),
                on_change=self._on_focused_point_roi_changed,
            )
            self._focused_point_roi.group = "_point_focus"

        style = entry.get("style", {})
        box_size_data = float(style.get("box_size_data", 9.0))
        self._focused_point_roi.point_id = int(point_id)
        self._focused_point_roi.set_name(f"Point {int(point_id)}")
        self._focused_point_roi.set_from_point(row["x"], row["y"], box_size_data)
        self._focused_point_roi.select(True)
        self.canvas.update()

    def _on_focused_point_roi_changed(self, state):
        if self._focused_point_layer is None or self._focused_point_id is None:
            return
        if self._focused_point_layer not in self._point_layers:
            return
        entry = self._point_layers[self._focused_point_layer]
        points = entry["points"]
        updated = points.update_point(
            int(self._focused_point_id),
            x=state.get("x"),
            y=state.get("y"),
        )
        entry["points"] = updated
        box_size = float(state.get("box_size_data", entry["style"].get("box_size_data", 9.0)))
        entry["style"]["box_size_data"] = box_size
        visual = entry["visual"]
        if visual is not None:
            visual.set_points(updated)
            visual.set_style(box_size_data=box_size)
            visual.set_selected_point_ids(entry["selected_ids"])
            visual.set_time_z(self.t_idx, self.z_idx)
        self.canvas.update()

    def _get_point_row(self, layer_name: str, point_id: int):
        if layer_name not in self._point_layers:
            return None
        points = self._point_layers[layer_name]["points"]
        if points is None:
            return None
        return points.get_point(int(point_id))

    def _format_hover_text(self, row: dict, template: str) -> str:
        try:
            return template.format(**row)
        except Exception:
            return str(row.get("point_id", ""))

    def _nearest_point(self, x: float, y: float, *, radius_px=8.0):
        if self._active_point_layer is None:
            return None
        if self._active_point_layer not in self._point_layers:
            return None
        entry = self._point_layers[self._active_point_layer]
        points = entry["points"]
        if points is None or points.n_rows == 0:
            return None
        s = points.get_time_slice(self.t_idx)
        if s is None:
            return None
        px_per_data = 1.0
        visual = entry.get("visual")
        if visual is not None and hasattr(visual, "_px_per_data"):
            px_per_data = max(float(visual._px_per_data()), 1e-6)
        best = None
        best_px = float("inf")
        for i in range(s.start, s.stop):
            if points.z is not None:
                ztol = float(entry["style"].get("z_tolerance", 0.5))
                if abs(float(points.z[i]) - float(self.z_idx)) > ztol:
                    continue
            dx = float(points.x[i]) - float(x)
            dy = float(points.y[i]) - float(y)
            dpx = (dx * dx + dy * dy) ** 0.5 * px_per_data
            if dpx < best_px:
                best_px = dpx
                best = int(points.point_id[i])
        if best is None or best_px > float(radius_px):
            return None
        return {"layer": self._active_point_layer, "point_id": best, "distance_px": best_px}

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

    def show_line_overlay(self, key, points, color=(1.0, 1.0, 0.0, 0.9), width=2):
        """Show/update a read-only polyline overlay identified by ``key``.

        Lets dialogs outside ``ui/`` (e.g. ``LineProfileDialog``) indicate a
        path sourced from another window without importing vispy themselves
        (``widgets/`` stays vispy-free per the layering rule). ``points`` is
        an (N, 2+) array of image-pixel coordinates; fewer than 2 points
        hides the overlay.
        """
        overlay = self._external_overlays.get(key)
        if overlay is None:
            overlay = scene.visuals.Line(pos=np.zeros((2, 2)), parent=self.view.scene)
            overlay.set_gl_state(
                preset="translucent", blend=True, depth_test=False
            )
            overlay.order = 100
            self._external_overlays[key] = overlay

        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            overlay.visible = False
            self.canvas.update()
            return

        overlay.set_data(pos=pts[:, :2], color=color, width=width)
        overlay.visible = True
        self.canvas.update()

    def hide_line_overlay(self, key):
        """Remove the overlay previously shown via ``show_line_overlay``."""
        overlay = self._external_overlays.pop(key, None)
        if overlay is not None:
            overlay.parent = None
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
        """Finalize contour mode and fill the enclosed region."""
        if not self._contour_mode or self._contour_start is None:
            self._cancel_contour()
            return

        if len(self._stroke_points) < 10:
            self._cancel_contour()
            return

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
            if self._mask_propagate_z:
                # Cookie-cut: fill across all Z slices
                # Build combined mask: brush strokes + polygon interior
                mask = np.zeros((self.Y, self.X), dtype=bool)
                for px, py in self._stroke_points:
                    by, bx = self._get_brush_coords(px, py)
                    if len(by) > 0:
                        mask[by, bx] = True
                mask[rr, cc] = True
                rr_all, cc_all = np.where(mask)

                n_z = self.labels.shape[0]
                n_px = len(rr_all)
                z_coords = np.repeat(np.arange(n_z, dtype=np.int32), n_px)
                rr_tiled = np.tile(rr_all, n_z)
                cc_tiled = np.tile(cc_all, n_z)
                fill_coords = (z_coords, rr_tiled, cc_tiled)
            else:
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

    def show_metadata_dialog(self):
        dlg = MetadataDialog(self.meta, parent=self)
        dlg.exec_()

    def show_ortho_view(self):
        # Copy full per-channel display state (clim/gamma/colormap/visible)
        # from the current renderer so the derived view matches what's on
        # screen instead of resetting to auto/default contrast.
        channel_display = ChannelDisplayList.from_states(
            self.renderer.display[c] for c in range(self.renderer.num_channels)
        )

        # Acquire reference if proxy supports ref counting (for shared file handles)
        data = self.img_data
        if hasattr(data, "acquire"):
            data = data.acquire()

        self.ortho_viewer = OrthoViewer(
            data,
            self.meta,
            title=f"Ortho View - {self.windowTitle()}",
            channel_display=channel_display,
            t_idx=self.t_idx,
            z_idx=self.z_idx,
        )
        present_window(self.ortho_viewer)

    def show_volume_view(self):
        """Open 3D volume rendering view."""
        from ..viewers import VolumeViewer

        # Copy full per-channel display state from the current renderer.
        channel_display = ChannelDisplayList.from_states(
            self.renderer.display[c] for c in range(self.renderer.num_channels)
        )

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
            channel_display=channel_display,
        )
        present_window(self.volume_viewer)

    def show_zmontage_view(self):
        """Open the Z-montage view (every Z-slice tiled into a grid)."""
        from ..viewers import ZMontageViewer

        # Copy full per-channel display state from the current renderer.
        channel_display = ChannelDisplayList.from_states(
            self.renderer.display[c] for c in range(self.renderer.num_channels)
        )

        # Acquire reference if proxy supports ref counting (for shared file handles)
        data = self.img_data
        if hasattr(data, "acquire"):
            data = data.acquire()

        self.zmontage_viewer = ZMontageViewer(
            data,
            self.meta,
            title=f"Z-Montage - {self.windowTitle()}",
            channel_display=channel_display,
            t_idx=self.t_idx if hasattr(self, "t_idx") else 0,
        )
        present_window(self.zmontage_viewer)

    def update_cursor(self):
        tool = manager.active_tool
        if tool == "pointer":
            self.view.camera.interactive = True
        else:
            self.view.camera.interactive = False

    def show_channel_panel(self):
        from .workspace import show_channel_panel as _show_channel_panel

        _show_channel_panel(self)

    def _show_singleton_dialog(self, attr_name, factory):
        """Show a dialog cached as ``self.<attr_name>``, building it once
        via zero-arg *factory* on first use and reusing it afterwards."""
        dlg = getattr(self, attr_name)
        if dlg is None:
            dlg = factory()
            setattr(self, attr_name, dlg)
        dlg.show()
        dlg.raise_()
        return dlg

    def show_transform_dialog(self):
        dlg = self._show_singleton_dialog(
            "transform_dialog", lambda: TransformDialog(self, parent=self)
        )
        dlg.refresh_ui()

    def show_alignment_dialog(self):
        self._show_singleton_dialog(
            "_alignment_dialog", lambda: AlignmentDialog(parent=self)
        )

    def show_z_projection_dialog(self):
        self._show_singleton_dialog(
            "z_projection_dialog", lambda: ZProjectionDialog(self, parent=self)
        )

    def show_image_math_dialog(self):
        self._show_singleton_dialog(
            "_image_math_dialog", lambda: ImageMathDialog(self, parent=self)
        )

    def show_combine_images_dialog(self):
        self._show_singleton_dialog(
            "_combine_images_dialog", lambda: CombineImagesDialog(self, parent=self)
        )

    def show_fft_dialog(self):
        self._show_singleton_dialog(
            "_fft_dialog", lambda: FFTDialog(self, parent=self)
        )

    def show_line_profile(self):
        """Show the line profile dialog."""
        from ..widgets import get_line_profile_dialog

        dialog = get_line_profile_dialog()
        dialog.active_window = self
        dialog.show()
        dialog.raise_()

    def compare_with_dialog(self):
        """Pick a second window and dock it side by side with this one as
        an explicit comparison pair (see ``ui/comparison.py``)."""
        from .workspace import get_workspace

        get_workspace().compare_with_dialog(self)

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
        if self.img_data is not None and hasattr(self.img_data, "release"):
            self.img_data.release()
        self.img_data = new_data
        self.meta = new_meta
        self._replace_slice_loader(new_data)
        self.T, self.Z, self.C, self.Y, self.X = self.img_data.shape

        # Reset indices (suspended: renderer/controls rebuilt below,
        # followed by an explicit update_view)
        self._suspend_view_updates = True
        try:
            self.view_state.set_t(0)
            self.view_state.set_z(self.Z // 2)
        finally:
            self._suspend_view_updates = False
        self.c_idx = 0

        # Remove old renderer layers
        for layer in self.renderer.layers:
            layer.parent = None

        # Create new renderer with updated data
        is_rgb = self.meta.get("is_rgb", False)
        self.renderer = CompositeImageVisual(
            self.view,
            self.img_data,
            is_rgb=is_rgb,
            channels_meta=self.meta.get("channels"),
        )
        self.renderer.reset_camera(self.img_data.shape)
        self._sync_overlay_spacing()

        # Rebuild controls
        self._rebuild_controls()

        # Update view
        self.update_view()
        self._request_auto_contrast()
        self.canvas.update()

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
        self._init_control_refs()

        # Rebuild controls
        self._setup_controls()

        # Fresh controls start with projection unchecked and a full-range
        # slider; sync ViewState so update_view agrees with the widgets.
        self._suspend_view_updates = True
        try:
            self.view_state.set_z_projection(False)
            self.view_state.set_z_range(0, self.Z - 1)
        finally:
            self._suspend_view_updates = False

    def set_tool(self, tool_name):
        """
        Set the active tool (e.g. 'pointer', 'rect', 'circle', 'line').
        """
        valid_tools = [
            "pointer", "rect", "circle", "line",
            "polyline", "point", "brush", "eraser",
        ]
        if tool_name not in valid_tools:
            print(f"Invalid tool: {tool_name}. Valid tools: {valid_tools}")
            return

        manager.set_active_tool(tool_name)

        # Update cursors in all windows
        for w in manager.get_all().values():
            w.update_cursor()

    def _compute_canvas_size(self):
        """Fit the canvas to the image's XY aspect ratio within a
        screen-relative bounding box, scaling small images up to a minimum
        visible size. Extreme aspect ratios legitimately end up capped on
        one axis and small on the other -- that's correct, not a bug.
        """
        aspect = self.X / self.Y

        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            max_w, max_h = int(avail.width() * 0.7), int(avail.height() * 0.7)
        else:
            max_w, max_h = 900, 720  # offscreen/headless fallback

        min_long_side = 500  # don't let small images open tiny

        long_side = max(self.X, self.Y, min_long_side)
        if self.X >= self.Y:
            canvas_w, canvas_h = long_side, long_side / aspect
        else:
            canvas_h, canvas_w = long_side, long_side * aspect

        scale = min(max_w / canvas_w, max_h / canvas_h, 1.0)
        return int(canvas_w * scale), int(canvas_h * scale)

    def _init_control_refs(self):
        """Initialize dynamic control references to safe defaults."""
        self.mode_combo = None
        self.channel_row_widget = None
        self.c_slider = None
        self.c_spin = None
        self.t_slider = None
        self.t_spin = None
        self.z_slider = None
        self.z_spin = None
        self.chk_proj = None
        self.z_range_slider = None
        self.z_range_slider_widget = None
        self.z_range_slider_min_label = None
        self.z_range_slider_max_label = None

    def _setup_controls(self):
        # -- Channel Slider + Mode Selector (Only if Multi-channel) --
        if self.C > 1:
            row = QHBoxLayout()

            # -- Mode Selector (always visible when C > 1) --
            row.addWidget(QLabel("Mode:"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Composite", "Single Channel"])
            self.mode_combo.currentIndexChanged.connect(self.on_mode_change)
            row.addWidget(self.mode_combo)

            # -- Channel Slider (Initially Hidden) --
            self.channel_row_widget = QWidget()
            c_layout = QHBoxLayout(self.channel_row_widget)
            c_layout.setContentsMargins(0, 0, 0, 0)

            c_layout.addWidget(QLabel("Channel"))
            self.c_slider = QSlider(Qt.Horizontal)
            self.c_slider.setRange(0, self.C - 1)
            self.c_slider.setValue(self.c_idx)
            # Arrow keys are the shape-nudge shortcut (see keyPressEvent) --
            # a slider holding keyboard focus would otherwise steal them to
            # step its own value instead, silently moving off the shape's
            # t/z plane and making it look like it "disappeared".
            self.c_slider.setFocusPolicy(Qt.NoFocus)
            self.c_slider.setMaximumWidth(120)
            self.c_slider.setPageStep(1)
            self.c_slider.setTickPosition(QSlider.TicksBelow)
            self.c_slider.setTickInterval(1)
            self.c_slider.valueChanged.connect(self.on_channel_change)
            c_layout.addWidget(self.c_slider)

            self.c_spin = QSpinBox()
            self.c_spin.setRange(0, self.C - 1)
            self.c_spin.setValue(self.c_idx)
            self.c_spin.setFocusPolicy(Qt.ClickFocus)
            self.c_spin.setFixedWidth(50)
            self.c_spin.valueChanged.connect(self.on_channel_change)
            c_layout.addWidget(self.c_spin)

            row.addWidget(self.channel_row_widget)
            self.channel_row_widget.setVisible(False)  # Default is Composite
            row.addStretch()

            self.controls_layout.addLayout(row)

        # -- Time Controls --
        has_multiple_timepoints = self.T > 1
        if has_multiple_timepoints:
            row = QHBoxLayout()
            row.addWidget(QLabel("Time"))

            self.t_slider = QSlider(Qt.Horizontal)
            self.t_slider.setRange(0, self.T - 1)
            self.t_slider.setValue(self.t_idx)
            self.t_slider.setFocusPolicy(Qt.NoFocus)
            self.t_slider.setMaximumWidth(150)
            self.t_slider.setPageStep(1)
            self.t_slider.setTickPosition(QSlider.TicksBelow)
            self.t_slider.setTickInterval(max(1, (self.T - 1) // 20))
            self.t_slider.valueChanged.connect(self.on_time_change)
            row.addWidget(self.t_slider)

            self.t_spin = QSpinBox()
            self.t_spin.setRange(0, self.T - 1)
            self.t_spin.setValue(self.t_idx)
            self.t_spin.setFocusPolicy(Qt.ClickFocus)
            self.t_spin.setFixedWidth(50)
            self.t_spin.valueChanged.connect(self.on_time_change)
            row.addWidget(self.t_spin)

            play_button = QPushButton("Play")
            play_button.setCheckable(True)
            play_button.toggled.connect(self._playback.toggle)
            row.addWidget(play_button)

            playback_fps_spin = QDoubleSpinBox()
            playback_fps_spin.setRange(0.01, 120.0)
            playback_fps_spin.setDecimals(2)
            playback_fps_spin.setSingleStep(0.25)
            playback_fps_spin.setValue(self._playback.fps)
            playback_fps_spin.setSuffix(" fps")
            # ClickFocus (not NoFocus like c_slider above): unlike a slider,
            # this needs direct keyboard text entry, but only when the user
            # clicks into it -- Tab traversal still won't land here and
            # steal arrow keys from the shape-nudge shortcut.
            playback_fps_spin.setFocusPolicy(Qt.ClickFocus)
            playback_fps_spin.valueChanged.connect(self._playback.set_fps)
            row.addWidget(playback_fps_spin)

            playback_realtime_btn = QPushButton("Realtime")
            playback_realtime_btn.clicked.connect(self._playback.set_realtime_fps)
            row.addWidget(playback_realtime_btn)

            self._playback.bind_widgets(
                play_button, playback_fps_spin, playback_realtime_btn
            )

            self.controls_layout.addLayout(row)

        # -- Z Slider --
        if self.Z > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Z-Pos"))

            # Standard Slider
            self.z_slider = QSlider(Qt.Horizontal)
            self.z_slider.setRange(0, self.Z - 1)
            self.z_slider.setValue(self.z_idx)
            self.z_slider.setFocusPolicy(Qt.NoFocus)
            self.z_slider.setMaximumWidth(150)
            self.z_slider.setPageStep(1)
            self.z_slider.setTickPosition(QSlider.TicksBelow)
            self.z_slider.setTickInterval(max(1, (self.Z - 1) // 20))
            self.z_slider.valueChanged.connect(self.on_z_change)
            row.addWidget(self.z_slider)

            self.z_spin = QSpinBox()
            self.z_spin.setRange(0, self.Z - 1)
            self.z_spin.setValue(self.z_idx)
            self.z_spin.setFocusPolicy(Qt.ClickFocus)
            self.z_spin.setFixedWidth(50)
            self.z_spin.valueChanged.connect(self.on_z_change)
            row.addWidget(self.z_spin)

            # Projection Controls
            self.chk_proj = QCheckBox("Max Proj")
            self.chk_proj.toggled.connect(self.toggle_z_projection)
            self.z_range_slider_widget = QWidget()
            self.z_range_slider_layout = QHBoxLayout()
            self.z_range_slider_layout.setContentsMargins(0, 0, 0, 0)
            self.z_range_slider = QRangeSlider(Qt.Horizontal)
            self.z_range_slider.setFocusPolicy(Qt.NoFocus)  # see c_slider above
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

        self.controls_layout.addStretch(1)

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
        if self.c_slider is not None and self.c_slider.value() != val:
            self.c_slider.blockSignals(True)
            self.c_slider.setValue(val)
            self.c_slider.blockSignals(False)
        if self.c_spin is not None and self.c_spin.value() != val:
            self.c_spin.blockSignals(True)
            self.c_spin.setValue(val)
            self.c_spin.blockSignals(False)
        self.renderer.set_active_channel(val)
        self.canvas.update()
        self.view_changed.emit(self)
        self.channel_changed.emit(val)

    def on_time_change(self, val):
        self.view_state.set_t(val)  # label + redraw via subscription
        self._playback.on_time_changed()

    def _advance_time_index(self, idx):
        if self.t_slider is not None:
            self.t_slider.setValue(idx)
        else:
            self.on_time_change(idx)

    def toggle_z_projection(self, checked):
        self.z_slider.setVisible(not checked)
        self.z_range_slider_widget.setVisible(checked)
        self.view_state.set_z_projection(checked)  # redraw via subscription

    def on_z_proj_change(self, val):
        # update z-min/max labels
        self.z_range_slider_min_label.setText(str(val[0]))
        self.z_range_slider_max_label.setText(str(val[1]))
        self.view_state.set_z_range(val[0], val[1])  # redraw via subscription

    def on_z_change(self, val):
        self.view_state.set_z(val)  # label + redraw via subscription

    def _replace_slice_loader(self, source):
        """(Re)create the async slice loader for *source*.

        In-memory sources (plain arrays) slice in microseconds; going
        through the worker thread would only add a frame of latency, so
        they keep the synchronous update_view path (loader stays None).
        """
        if self._slice_loader is not None:
            self._slice_loader.close()
            self._slice_loader = None
        if not isinstance(source, (np.ndarray, Numpy5DProxy)):
            self._slice_loader = SliceLoader(source, self._emit_slice_ready)

    def _emit_slice_ready(self, key, plane):
        # Runs on the loader thread. The window may be mid-teardown
        # (WA_DeleteOnClose); emitting on a dead QObject raises.
        try:
            self._slice_ready.emit(key, plane)
        except RuntimeError:
            pass

    def _current_slice_key(self):
        vs = self.view_state
        if vs.z_projection:
            mn, mx = vs.z_range if vs.z_range is not None else (0, self.Z - 1)
            return SliceLoader.key_for(vs.t, slice(mn, mx + 1))
        return SliceLoader.key_for(vs.t, vs.z)

    def _on_slice_ready(self, key, plane):
        if key != self._current_slice_key():
            return  # stale: view moved on while this frame loaded
        self.renderer.set_slice(plane)
        if self._pending_auto_contrast:
            self._pending_auto_contrast = False
            self.renderer.auto_contrast()
        self.canvas.update()
        self.view_changed.emit(self)

    def _request_auto_contrast(self):
        """One-shot auto-contrast, deferred until a slice is actually cached.

        For in-memory sources ``update_view()`` slices synchronously, so
        the cache is already populated and this fires immediately. For
        lazy (file/zarr-backed) sources the first ``update_view()`` call
        is a cache miss — the async loader hasn't delivered a plane yet
        — so calling ``renderer.auto_contrast()`` right away would
        silently no-op on an empty cache and leave the full-dtype
        default clim (e.g. 0-65535 for uint16), which reads as solid
        black for any image without a huge dynamic range. Defer to
        ``_on_slice_ready`` instead, which applies it once the first
        real plane arrives.
        """
        if self.renderer.current_slice_cache is not None:
            self.renderer.auto_contrast()
        else:
            self._pending_auto_contrast = True

    def _subscribe_to_buffer(self, data):
        if self._buffer_unsubscribe is not None:
            self._buffer_unsubscribe()
            self._buffer_unsubscribe = None
        if hasattr(data, "subscribe"):
            self._buffer_unsubscribe = data.subscribe(
                lambda key: self._buffer_dirty.emit(key)
            )

    def _on_buffer_dirty(self, key):
        if self._slice_loader is not None:
            # Buffer contents changed under the cache.
            self._slice_loader.invalidate()
        if self._key_touches_current_slice(key):
            self.update_view()
        else:
            self.canvas.update()

    def _key_touches_current_slice(self, key):
        """Conservative check: returns True if a buffer write at *key* could
        affect the currently displayed (t, z) slice. Unknown keys are
        treated as overlapping (better to over-refresh than miss updates).
        """
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            return True
        if len(key) < 2:
            return True

        def hits(axis_val, current, size):
            if isinstance(axis_val, slice):
                start, stop, step = axis_val.indices(size)
                return (
                    start <= current < stop
                    and (current - start) % step == 0
                )
            try:
                return int(axis_val) == current
            except (TypeError, ValueError):
                return True

        return hits(key[0], self.t_idx, self.T) and hits(
            key[1], self.z_idx, self.Z
        )

    def update_view(self):
        vs = self.view_state
        if vs.z_projection:
            mn, mx = vs.z_range if vs.z_range is not None else (0, self.Z - 1)
            z_key = slice(mn, mx + 1)
        else:
            z_key = vs.z

        if self._slice_loader is None:
            # In-memory source: slice synchronously.
            self.renderer.update_slice(vs.t, z_key)
            pixels_ready = True
        else:
            plane = self._slice_loader.request(vs.t, z_key)
            if plane is not None:
                self.renderer.set_slice(plane)
            # else: keep showing the current frame; _on_slice_ready
            # displays the new one as soon as the worker has it.
            pixels_ready = plane is not None

        # Update all mask layers for 3D data
        for entry in self._mask_layers.values():
            if entry["visual"] and entry["labels"] and entry["labels"].ndim == 3:
                entry["visual"].update_slice(self.z_idx)
        for entry in self._track_layers.values():
            visual = entry.get("visual")
            if visual is not None:
                visual.set_time_z(self.t_idx, self.z_idx)
        for entry in self._point_layers.values():
            visual = entry.get("visual")
            if visual is not None:
                visual.set_time_z(self.t_idx, self.z_idx)
        self._refresh_shape_layers()
        self._sync_focused_point_visibility()

        if self.overlay is not None:
            self.overlay.update()
        self.canvas.update()
        if pixels_ready:
            # Channel panel auto-refreshes via view_changed + display
            # subscription. On a lazy-source cache miss the pixel data
            # hasn't actually arrived yet (current_slice_cache still holds
            # the previous frame) -- emitting here would have every
            # view_changed consumer that samples pixel data (line/radial
            # profile, region stats, point intensity, channel histograms)
            # transiently read the wrong frame. _on_slice_ready emits this
            # same signal once the real plane lands instead.
            self.view_changed.emit(self)

    def _sync_focused_point_visibility(self):
        if self._focused_point_roi is None:
            return
        if self._focused_point_id is None:
            return
        if self._focused_point_layer not in self._point_layers:
            self.clear_focused_point()
            return
        row = self._get_point_row(self._focused_point_layer, self._focused_point_id)
        if row is None:
            self.clear_focused_point()
            return
        show = int(row.get("t", 0)) == int(self.t_idx)
        if "z" in row:
            ztol = float(
                self._point_layers[self._focused_point_layer]["style"].get(
                    "z_tolerance", 0.5
                )
            )
            show = show and abs(float(row["z"]) - float(self.z_idx)) <= ztol
        self._focused_point_roi.set_visible(show)

    def _init_overlay(self):
        _, _, sx = self.meta.get("scale", (1.0, 1.0, 1.0))
        self.overlay = ScaleTimestampOverlay(
            self.view,
            axis_spacing_um=sx,
            world_units_are_um=False,
            get_time_index=lambda: self.t_idx,
            get_timestamps=lambda: self.meta.get("timestamps", []),
        )

    def _sync_overlay_spacing(self):
        if self.overlay is None:
            return
        _, _, sx = self.meta.get("scale", (1.0, 1.0, 1.0))
        self.overlay.set_axis_spacing_um(sx)

    def _on_view_transform_event(self, _event):
        for entry in self._point_layers.values():
            visual = entry.get("visual")
            if visual is not None:
                visual.refresh()
        if self.overlay is not None:
            self.overlay.update()
            self.canvas.update()

    def show_overlay_settings_dialog(self):
        if self.overlay is None:
            return
        dlg = OverlaySettingsDialog(self.overlay.get_config(), parent=self)
        if dlg.exec_():
            self.overlay.set_config(dlg.get_config())
            self.canvas.update()

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
            # Right-click on a shape-layer entry → context menu (Rename/Delete).
            # Routed before the drag-init path so right-click never starts a drag.
            if event.button == 2:
                shape_hit = self._hit_test_shape_layers(x, y)
                if shape_hit is not None:
                    layer, shape_id, _handle = shape_hit
                    self._select_shape(layer, shape_id)
                    self.canvas.update()
                    ex, ey = int(event.pos[0]), int(event.pos[1])
                    global_pos = self.canvas.native.mapToGlobal(QPoint(ex, ey))
                    self._show_shape_context_menu(layer, shape_id, global_pos)
                    self._reset_vispy_press_state()
                    return

            # Hit Test ROIs
            hit_roi = None
            hit_handle = None

            if self._focused_point_roi is not None:
                res = self._focused_point_roi.hit_test((x, y))
                if res:
                    hit_roi = self._focused_point_roi
                    hit_handle = res

            if hit_roi is None:
                near = self._nearest_point(x, y, radius_px=10.0)
                if near is not None:
                    self.focus_point(near["layer"], near["point_id"])
                    hit_roi = self._focused_point_roi
                    hit_handle = "center"

            # Update Selection
            if self._focused_point_roi is not None:
                self._focused_point_roi.select(self._focused_point_roi is hit_roi)

            # Notify about selection change
            self.roi_selection_changed.emit(hit_roi)

            if hit_roi:
                self.dragging_roi = hit_roi
                self.drag_handle = hit_handle
                self.last_pos = (x, y)
                # Disable camera panning while dragging ROI
                self.view.camera.interactive = False
                self._clear_shape_selection()
                self.canvas.update()
                return

            # No legacy ROI hit — try shape-layer entries.
            shape_hit = self._hit_test_shape_layers(x, y)
            if shape_hit is not None:
                layer, shape_id, handle = shape_hit
                rec = layer.data.get(shape_id)
                has_alt = "Alt" in event.modifiers
                has_ctrl_or_cmd = (
                    "Control" in event.modifiers or "Meta" in event.modifiers
                )
                gel_marker_idx = self._hit_test_gel_marker(rec, x, y)

                if rec.shape_type == RECTANGLE and rec.properties.get("gel_lane"):
                    if has_ctrl_or_cmd and gel_marker_idx is not None:
                        self._select_shape(layer, shape_id)
                        self._remove_gel_marker(layer, shape_id, gel_marker_idx)
                        return
                    if "Shift" in event.modifiers:
                        self._select_shape(layer, shape_id)
                        self._add_gel_marker(layer, shape_id, y)
                        return
                    if gel_marker_idx is not None:
                        self._select_shape(layer, shape_id)
                        self._dragging_gel_marker_layer = layer
                        self._dragging_gel_marker_shape_id = shape_id
                        self._dragging_gel_marker_idx = gel_marker_idx
                        self.view.camera.interactive = False
                        self.canvas.update()
                        return

                # Polyline-specific shortcuts before drag init.
                if rec.shape_type == POLYLINE:
                    # Alt+click a vertex: remove it.
                    if (
                        handle is not None
                        and handle.startswith("v")
                        and has_alt
                        and rec.vertices is not None
                        and len(rec.vertices) > 2
                    ):
                        try:
                            idx = int(handle[1:])
                        except ValueError:
                            idx = -1
                        if 0 <= idx < len(rec.vertices):
                            self._select_shape(layer, shape_id)
                            layer.undo_stack.push(
                                RemoveVertex(shape_id, idx), layer.data
                            )
                            return
                    # Body click near a segment: insert vertex (skip on Alt).
                    if handle is None and not has_alt:
                        insert_idx = self._polyline_segment_insert_index(
                            rec, x, y
                        )
                        if insert_idx is not None:
                            self._select_shape(layer, shape_id)
                            layer.undo_stack.push(
                                AddVertex(shape_id, insert_idx, x, y),
                                layer.data,
                            )
                            return

                self._select_shape(layer, shape_id)
                self._editing_shape_layer = layer
                self._editing_shape_id = shape_id
                self._editing_shape_handle = handle  # None == body drag
                self._editing_shape_start_params = rec.params.copy()
                self._editing_shape_start_vertices = (
                    rec.vertices.copy() if rec.vertices is not None else None
                )
                self._editing_shape_start_pos = (x, y)
                self.view.camera.interactive = False
                self.canvas.update()
                return

            # Nothing hit — clear all selections.
            self._clear_shape_selection()
            self.clear_focused_point()
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
                    # Already in contour mode - finish and fill
                    self._finish_contour()
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

        # Shape drawing tools (rect, circle, line) populate the active
        # ShapeData layer via AddShape.
        if tool in ("rect", "circle", "line"):
            # If the press lands on a HANDLE of a selected shape, route to
            # the edit path so the user can grab handles of the just-drawn
            # shape without switching tools. Body hits fall through to a
            # new draw — you can't visually distinguish "inside an existing
            # shape" from "empty area," but handles are visible targets.
            shape_hit = self._hit_test_shape_layers(x, y)
            if shape_hit is not None and shape_hit[2] is not None:
                layer, shape_id, handle = shape_hit
                rec = layer.data.get(shape_id)
                self._select_shape(layer, shape_id)
                self._editing_shape_layer = layer
                self._editing_shape_id = shape_id
                self._editing_shape_handle = handle
                self._editing_shape_start_params = rec.params.copy()
                self._editing_shape_start_vertices = (
                    rec.vertices.copy() if rec.vertices is not None else None
                )
                self._editing_shape_start_pos = (x, y)
                self.view.camera.interactive = False
                self.canvas.update()
                return
            layer = self._active_shape_layer()
            if layer is None:
                layer = self.add_shape_layer("Shapes")
            shape_type = {
                "rect": RECTANGLE,
                "circle": CIRCLE,
                "line": LINE,
            }[tool]
            self.start_pos = (x, y)
            cmd = AddShape(
                shape_type,
                [x, y, x, y],
                t=self.t_idx,
                z=self.z_idx,
                label=f"Shape {len(layer.data)}",
            )
            layer.undo_stack.push(cmd, layer.data)
            self._drawing_shape_layer = layer
            self._drawing_shape_id = cmd.shape_id
            self._drawing_shape_cmd = cmd
            layer.visual.update(layer.data, layer.selected_ids, self.t_idx, self.z_idx)
            self.view.camera.interactive = False
            self.canvas.update()

        # Multi-click polyline drawing.
        if tool == "polyline":
            if event.button == 1:
                layer = self._polyline_drawing_layer
                if layer is None:
                    layer = self._active_shape_layer() or self.add_shape_layer("Shapes")
                if self._polyline_drawing_id is None:
                    cmd = AddShape(
                        POLYLINE,
                        vertices=[(x, y), (x, y)],
                        t=self.t_idx,
                        z=self.z_idx,
                        label=f"Shape {len(layer.data)}",
                    )
                    layer.undo_stack.push(cmd, layer.data)
                    self._polyline_drawing_layer = layer
                    self._polyline_drawing_id = cmd.shape_id
                else:
                    rec = layer.data.get(self._polyline_drawing_id)
                    new_verts = np.vstack(
                        [rec.vertices, [[x, y]]]
                    ).astype(np.float32, copy=False)
                    rec.vertices = new_verts
                    layer.data._emit("edited", self._polyline_drawing_id)
                self.view.camera.interactive = False
                return
            if event.button == 2 and self._polyline_drawing_id is not None:
                # Right-click finishes — drop the trailing preview vertex.
                layer = self._polyline_drawing_layer
                rec = layer.data.get(self._polyline_drawing_id)
                if rec.vertices is not None and len(rec.vertices) >= 3:
                    rec.vertices = rec.vertices[:-1].astype(
                        np.float32, copy=False
                    )
                    layer.data._emit("edited", self._polyline_drawing_id)
                self._polyline_drawing_layer = None
                self._polyline_drawing_id = None
                return

        # Point tool — drop, drag, or Alt-remove.
        if tool == "point" and event.button == 1:
            has_alt = "Alt" in event.modifiers
            # Find active point layer (or create one).
            layer = self.layers.active("points")
            layer_name = layer.name if layer is not None else None
            if layer_name is None or layer_name not in self._point_layers:
                self.add_point_layer("Points")
                layer = self.layers.active("points")
                layer_name = layer.name if layer is not None else None
            holder = self._point_layers[layer_name]["holder"]

            near = self._nearest_point(x, y, radius_px=10.0)
            if near is not None and near["layer"] == layer_name:
                pid = int(near["point_id"])
                if has_alt:
                    from ..data.point_commands import RemovePoint
                    layer.undo_stack.push(RemovePoint(pid), holder)
                    return
                # Start drag.
                row = self._get_point_row(layer_name, pid)
                if row is not None:
                    self._point_dragging_layer = layer_name
                    self._point_dragging_id = pid
                    self._point_drag_start_xy = (
                        float(row.get("x", x)), float(row.get("y", y))
                    )
                    self.view.camera.interactive = False
                return

            # No hit — drop a new point.
            from ..data.point_commands import AddPoint
            layer.undo_stack.push(
                AddPoint(x=float(x), y=float(y), t=self.t_idx), holder
            )
            return

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
                    near = self._nearest_point(x, y, radius_px=10.0)
                    if near is not None:
                        row = self._get_point_row(near["layer"], near["point_id"])
                        if row is not None:
                            info_text += f"  Point: {row.get('point_id')}"
                            if "amplitude" in row:
                                info_text += f" amp={row.get('amplitude')}"
                            self.info_label.setToolTip(str(row))
                            # On-canvas hover tooltip
                            entry = self._point_layers.get(near["layer"])
                            visual = entry["visual"] if entry else None
                            template = visual.label_template if visual else "{point_id}"
                            hover_text = self._format_hover_text(row, template)
                            offset = 5.0 / max(visual._px_per_data(), 1e-6) if visual else 5.0
                            self._hover_label.text = hover_text
                            self._hover_label.pos = np.array(
                                [[x + offset, y - offset, 0.0]], dtype=np.float32
                            )
                            self._hover_label.visible = True
                        else:
                            self.info_label.setToolTip("")
                            self._hover_label.visible = False
                    else:
                        self.info_label.setToolTip("")
                        self._hover_label.visible = False
                    self.info_label.setText(info_text)
            else:
                self.info_label.setText("")
                self._hover_label.visible = False

        # 1b. Polyline drawing — update preview vertex (no button held).
        if self._polyline_drawing_id is not None and self._polyline_drawing_layer is not None:
            x, y = self._map_event_to_image(event)
            layer = self._polyline_drawing_layer
            try:
                rec = layer.data.get(self._polyline_drawing_id)
            except KeyError:
                self._polyline_drawing_layer = None
                self._polyline_drawing_id = None
            else:
                if rec.vertices is not None and len(rec.vertices) > 0:
                    rec.vertices[-1, 0] = x
                    rec.vertices[-1, 1] = y
                    layer.data._emit("edited", self._polyline_drawing_id)

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

        # 5. Shape Layer Drawing Update
        if self._drawing_shape_layer is not None and self._drawing_shape_id is not None and event.button == 1:
            x, y = self._map_event_to_image(event)
            layer = self._drawing_shape_layer
            sx, sy = self.start_pos
            rec = layer.data.get(self._drawing_shape_id)
            if rec.shape_type == RECTANGLE and "Shift" in event.modifiers:
                x, y = integer_square_from_corner(sx, sy, x, y)
            layer.data.update(self._drawing_shape_id, [sx, sy, x, y])
            layer.visual.update(layer.data, layer.selected_ids, self.t_idx, self.z_idx)
            self.canvas.update()

        # 5a. Gel marker drag on a shape-backed lane.
        if (
            self._dragging_gel_marker_layer is not None
            and self._dragging_gel_marker_shape_id is not None
            and self._dragging_gel_marker_idx is not None
            and event.button == 1
        ):
            _x, y = self._map_event_to_image(event)
            self._move_gel_marker(
                self._dragging_gel_marker_layer,
                self._dragging_gel_marker_shape_id,
                self._dragging_gel_marker_idx,
                y,
            )
            return

        # 5b. Point-tool drag — update x/y of the dragged point in place
        # (no command pushed yet; we push a single MovePoint on release).
        if (
            self._point_dragging_id is not None
            and self._point_dragging_layer is not None
            and event.button == 1
        ):
            x, y = self._map_event_to_image(event)
            entry = self._point_layers.get(self._point_dragging_layer)
            if entry is not None:
                holder = entry["holder"]
                tbl = holder.table
                idx = tbl._id_to_index.get(int(self._point_dragging_id))
                if idx is not None:
                    tbl.x[idx] = float(x)
                    tbl.y[idx] = float(y)
                    visual = entry["visual"]
                    if visual is not None:
                        visual.set_points(tbl)
                        visual.set_time_z(self.t_idx, self.z_idx)
                    self.canvas.update()

        # 6. Shape Layer Edit Drag (move body or adjust handle)
        if self._editing_shape_id is not None and event.button == 1:
            x, y = self._map_event_to_image(event)
            layer = self._editing_shape_layer
            rec = layer.data.get(self._editing_shape_id)
            start_params = self._editing_shape_start_params
            start_verts = self._editing_shape_start_vertices
            sx, sy = self._editing_shape_start_pos
            if self._editing_shape_handle is None:
                # Body drag: translate the params (rect-style) and any
                # polyline vertices by (dx, dy).
                dx, dy = x - sx, y - sy
                rec.params[0] = start_params[0] + dx
                rec.params[1] = start_params[1] + dy
                rec.params[2] = start_params[2] + dx
                rec.params[3] = start_params[3] + dy
                if rec.shape_type == RECTANGLE:
                    rec.params[:] = snap_rectangle_params(rec.params)
                if start_verts is not None:
                    rec.vertices = (start_verts + np.array([dx, dy], dtype=np.float32))
            else:
                # Handle drag: reset to start, then apply current pointer.
                rec.params[:] = start_params
                if start_verts is not None:
                    rec.vertices = start_verts.copy()
                nx, ny = x, y
                if rec.shape_type == RECTANGLE and "Shift" in event.modifiers:
                    anchor = rect_opposite_corner(
                        start_params, self._editing_shape_handle
                    )
                    if anchor is not None:
                        nx, ny = integer_square_from_corner(anchor[0], anchor[1], x, y)
                _apply_handle_adjustment(rec, self._editing_shape_handle, nx, ny)
            # Notify subscribers (e.g. region-stats / line-profile dialogs)
            # so derived calculations refresh live during the drag.
            layer.data._emit(EVT_EDITED, self._editing_shape_id)
            layer.visual.update(
                layer.data, layer.selected_ids, self.t_idx, self.z_idx
            )
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
            self.dragging_roi = None
            self.drag_handle = None
            self.last_pos = None
            # Re-enable camera panning if in pointer mode
            if manager.active_tool == "pointer":
                self.view.camera.interactive = True

        # Shape layer drawing finalization
        if self._drawing_shape_layer is not None:
            self._finalize_shape_drawing()
            self.view.camera.interactive = manager.active_tool == "pointer"

        if self._dragging_gel_marker_layer is not None:
            layer = self._dragging_gel_marker_layer
            shape_id = self._dragging_gel_marker_shape_id
            if shape_id is not None:
                layer.data._emit("gel_marker_released", shape_id)
            self._dragging_gel_marker_layer = None
            self._dragging_gel_marker_shape_id = None
            self._dragging_gel_marker_idx = None
            if manager.active_tool == "pointer":
                self.view.camera.interactive = True

        # Point drag finalization — push a single MovePoint command.
        if (
            self._point_dragging_id is not None
            and self._point_dragging_layer is not None
        ):
            entry = self._point_layers.get(self._point_dragging_layer)
            pid = int(self._point_dragging_id)
            sx, sy = self._point_drag_start_xy or (0.0, 0.0)
            self._point_dragging_layer = None
            self._point_dragging_id = None
            self._point_drag_start_xy = None
            if entry is not None:
                holder = entry["holder"]
                tbl = holder.table
                idx = tbl._id_to_index.get(pid)
                if idx is not None:
                    final_x = float(tbl.x[idx])
                    final_y = float(tbl.y[idx])
                    if final_x != sx or final_y != sy:
                        # Roll back the in-place mutation, then push a
                        # MovePoint command so undo/redo are clean.
                        tbl.x[idx] = sx
                        tbl.y[idx] = sy
                        from ..data.point_commands import MovePoint
                        layer = self.layers[self._point_dragging_layer] if (
                            self._point_dragging_layer and
                            self._point_dragging_layer in self.layers
                        ) else None
                        # Use stored entry's layer instead of stale name.
                        for lyr in self.layers.by_type("points"):
                            if lyr.data is holder:
                                layer = lyr
                                break
                        if layer is not None:
                            layer.undo_stack.push(
                                MovePoint(pid, final_x, final_y), holder
                            )
            if manager.active_tool == "pointer":
                self.view.camera.interactive = True

        # Shape layer edit finalization — push a single undoable command
        # for the whole drag (live mutation already applied).
        if self._editing_shape_id is not None:
            self._finalize_shape_edit()
            if manager.active_tool == "pointer":
                self.view.camera.interactive = True


def imshow(
    data,
    meta_or_title=None,
    dims=None,
    *,
    title=None,
    scale=None,
    colormap=None,
    floating=False,
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
        floating (bool): Show as an independent top-level window instead
                         of a workspace tab.
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
        # AA_ShareOpenGLContexts must be set before the QApplication is
        # constructed: it keeps vispy canvases' GL resources valid when a
        # viewer moves between top-level windows (workspace tab float/dock),
        # which otherwise segfaults.
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
        app = QApplication(sys.argv)

    # Apply Theme
    from ..theme import DARK_THEME

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

    return present_window(viewer, floating=floating)


def run_app():
    """
    Start the Qt event loop. Use this when running from a script
    to ensure windows are visible and interactive.
    """
    app = QApplication.instance()
    if app:
        from ..theme import DARK_THEME

        app.setStyleSheet(DARK_THEME)
        app.exec_()
