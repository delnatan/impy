import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAction,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from vispy import scene
from vispy.visuals.transforms import STTransform

from ..visuals.overlays import ScaleTimestampOverlay
from ..visuals.image import CompositeImageVisual, MultiViewChannelProxy
from ..widgets import (
    ChannelPanel,
    MetadataDialog,
    OverlaySettingsDialog,
)


class TransposedProxy:
    """
    Wraps a 5D data object and presents a transposed view.
    Handles mapping slices to the original data and transposing the result.
    """

    def __init__(self, data, perm):
        self.data = data
        self.perm = perm  # e.g. (0, 4, 2, 3, 1)

        # Calculate new shape
        self.shape = tuple(data.shape[p] for p in perm)
        self.dtype = data.dtype
        self.ndim = 5

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Pad key
        if len(key) < 5:
            key = key + (slice(None),) * (5 - len(key))

        # 1. Map key to original dimensions
        original_key = [slice(None)] * 5
        for i, p in enumerate(self.perm):
            original_key[p] = key[i]
        original_key = tuple(original_key)

        # 2. Get data
        res = self.data[original_key]

        # 3. Transpose result if necessary
        # Identify which dimensions remain
        # The result 'res' will have dimensions corresponding to those
        # indices in 'original_key' that were slices (not integers).
        # They will appear in the order of the original data (0..4).

        # We want them in the order they appeared in 'key'.

        # Indices in 'key' that are slices
        kept_dims_view = [i for i, k in enumerate(key) if isinstance(k, slice)]

        # Corresponding indices in 'original data'
        target_dims = [self.perm[i] for i in kept_dims_view]

        # The actual dimensions present in 'res' are 'target_dims' sorted.
        present_dims = sorted(target_dims)

        # Map from original dim index to position in 'res'
        dim_to_pos = {d: i for i, d in enumerate(present_dims)}

        # Construct permutation for 'res'
        # We want the k-th dimension of output to correspond to target_dims[k]
        res_perm = [dim_to_pos[d] for d in target_dims]

        # Only transpose if permutation is not identity
        if res_perm != list(range(len(res_perm))):
            res = res.transpose(res_perm)

        return res


class OrthoViewer(QMainWindow):
    # Emitted whenever the displayed orthoslice triplet changes (crosshair
    # move, time step, set_data, …). ChannelPanel & line-profile widgets
    # listen for this to refresh per-channel histograms / profiles.
    view_changed = Signal(object)

    def __init__(self, data, meta=None, title="Ortho View", channel_colormaps=None):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(1000, 800)

        self.data = data  # (T, Z, C, Y, X)
        self.meta = meta or {}
        self._channel_colormaps = channel_colormaps  # Store for later application
        self.T, self.Z, self.C, self.Y, self.X = self.data.shape

        # Scale (z, y, x)
        self.scale = self.meta.get("scale", (1.0, 1.0, 1.0))
        sz, sy, sx = self.scale

        # Current Position (in pixels)
        self.cz = self.Z // 2
        self.cy = self.Y // 2
        self.cx = self.X // 2
        self.ct = 0

        # Camera sync flag (must be initialized before reset_cameras)
        self._syncing_cameras = False

        # Store full extents for zoom calculations (initialized in reset_cameras)
        self._full_x = self.X * sx
        self._full_y = self.Y * sy
        self._full_z = self.Z * sz

        # -- Layout --
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Grid for Views
        self.grid = QGridLayout()
        self.grid.setSpacing(0)
        self.main_layout.addLayout(self.grid, 1)

        # Crosshair visibility state
        self.crosshair_visible = True

        # Calculate physical dimensions for layout stretches
        phys_x = int(self.X * sx)
        phys_y = int(self.Y * sy)
        phys_z = int(self.Z * sz)

        # Ensure at least 1 to avoid errors
        phys_x = max(1, phys_x)
        phys_y = max(1, phys_y)
        phys_z = max(1, phys_z)

        # Set stretches
        # Col 0: X, Col 1: Z
        self.grid.setColumnStretch(0, phys_x)
        self.grid.setColumnStretch(1, phys_z)

        # Row 0: Y, Row 1: Z
        self.grid.setRowStretch(0, phys_y)
        self.grid.setRowStretch(1, phys_z)

        # Enforce minimum size for side views to avoid "unusable thinness"
        # when Z dimension is very small relative to X/Y.
        self.grid.setColumnMinimumWidth(1, 150)
        self.grid.setRowMinimumHeight(1, 150)

        # -- 1. Create Canvases --
        # YX View (Top-Left)
        self.canvas_yx = scene.SceneCanvas(keys=None, bgcolor="black")
        self.view_yx = self.canvas_yx.central_widget.add_view()
        self.view_yx.padding = 0
        self.view_yx.camera = "panzoom"
        self.view_yx.camera.aspect = 1
        self.grid.addWidget(self.canvas_yx.native, 0, 0)

        # ZY View (Top-Right) - Rotated: Y vertical, Z horizontal
        self.canvas_zy = scene.SceneCanvas(keys=None, bgcolor="black")
        self.view_zy = self.canvas_zy.central_widget.add_view()
        self.view_zy.padding = 0
        self.view_zy.camera = "panzoom"
        self.view_zy.camera.aspect = 1
        self.grid.addWidget(self.canvas_zy.native, 0, 1)

        # ZX View (Bottom-Left) - Rotated: Z vertical, X horizontal
        self.canvas_zx = scene.SceneCanvas(keys=None, bgcolor="black")
        self.view_zx = self.canvas_zx.central_widget.add_view()
        self.view_zx.padding = 0
        self.view_zx.camera = "panzoom"
        self.view_zx.camera.aspect = 1
        self.grid.addWidget(self.canvas_zx.native, 1, 0)

        # -- 2. Create Visuals --
        # YX: (T, Z, C, Y, X) -> Slice Z -> (C, Y, X)
        self.vis_yx = CompositeImageVisual(
            self.view_yx, self.data, scale=(sy, sx)
        )

        # ZY: Need (T, X, C, Y, Z). Slice X -> (C, Y, Z)
        # Transpose data: (0, 4, 2, 3, 1) -> T, X, C, Y, Z
        data_zy = TransposedProxy(self.data, (0, 4, 2, 3, 1))
        self.vis_zy = CompositeImageVisual(
            self.view_zy, data_zy, scale=(sy, sz)
        )

        # ZX: Need (T, Y, C, Z, X). Slice Y -> (C, Z, X)
        # Transpose data: (0, 3, 2, 1, 4) -> T, Y, C, Z, X
        data_zx = TransposedProxy(self.data, (0, 3, 2, 1, 4))
        self.vis_zx = CompositeImageVisual(
            self.view_zx, data_zx, scale=(sz, sx)
        )

        self.proxy = MultiViewChannelProxy([self.vis_yx, self.vis_zy, self.vis_zx])
        self._init_overlays()

        # Apply colormaps from source window if provided
        if self._channel_colormaps:
            for channel_idx, cmap_name in self._channel_colormaps.items():
                self.proxy.set_colormap(channel_idx, cmap_name)

        # -- 3. Crosshairs --
        # Use cyan color for good contrast against typical fluorescence colormaps
        crosshair_color = (0, 0.8, 1, 0.7)  # Cyan with 70% opacity

        # YX View: V-Line at X, H-Line at Y
        self.line_yx_v = scene.visuals.InfiniteLine(
            pos=self.cx,
            color=crosshair_color,
            vertical=True,
            parent=self.view_yx.scene,
        )
        self.line_yx_h = scene.visuals.InfiniteLine(
            pos=self.cy,
            color=crosshair_color,
            vertical=False,
            parent=self.view_yx.scene,
        )
        self.line_yx_v.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )
        self.line_yx_h.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )
        # ZY View: V-Line at Z, H-Line at Y
        self.line_zy_v = scene.visuals.InfiniteLine(
            pos=self.cz,
            color=crosshair_color,
            vertical=True,
            parent=self.view_zy.scene,
        )
        self.line_zy_h = scene.visuals.InfiniteLine(
            pos=self.cy,
            color=crosshair_color,
            vertical=False,
            parent=self.view_zy.scene,
        )
        self.line_zy_v.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )
        self.line_zy_h.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )

        # ZX View: V-Line at X, H-Line at Z
        self.line_zx_v = scene.visuals.InfiniteLine(
            pos=self.cx,
            color=crosshair_color,
            vertical=True,
            parent=self.view_zx.scene,
        )
        self.line_zx_h = scene.visuals.InfiniteLine(
            pos=self.cz,
            color=crosshair_color,
            vertical=False,
            parent=self.view_zx.scene,
        )
        self.line_zx_v.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )
        self.line_zx_h.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )

        # -- 4. Info Bar --
        self.info_label = QLabel("Hover over image")
        self.info_label.setStyleSheet(
            "background-color: #333; color: #EEE; padding: 4px;"
        )
        self.info_label.setFixedHeight(25)
        self.main_layout.addWidget(self.info_label, 0)

        # -- 5. Controls --
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.main_layout.addWidget(self.controls_widget)

        # Store event handlers for proper cleanup on close
        self._event_handlers = []

        self._setup_controls()
        self._setup_menu()
        self._setup_events()

        # Initial Update
        self.update_views()
        self.reset_cameras()

        # Camera Synchronization
        # Connect to mouse events to sync cameras after zoom/pan operations
        self._syncing_cameras = False

        def make_camera_handler(view_name):
            def handler(e):
                self._on_camera_change(view_name)

            return handler

        for canvas, view_name in [
            (self.canvas_yx, "yx"),
            (self.canvas_zy, "zy"),
            (self.canvas_zx, "zx"),
        ]:
            wheel_handler = make_camera_handler(view_name)
            release_handler = make_camera_handler(view_name)
            resize_handler = make_camera_handler(view_name)
            canvas.events.mouse_wheel.connect(wheel_handler)
            canvas.events.mouse_release.connect(release_handler)
            canvas.events.resize.connect(resize_handler)
            self._event_handlers.append(
                (canvas.events.mouse_wheel, wheel_handler)
            )
            self._event_handlers.append(
                (canvas.events.mouse_release, release_handler)
            )
            self._event_handlers.append((canvas.events.resize, resize_handler))

    def _setup_controls(self):
        # Mode (Composite/Single)
        if self.C > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Mode:"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Composite", "Single Channel"])
            self.mode_combo.currentIndexChanged.connect(self.on_mode_change)
            row.addWidget(self.mode_combo)
            row.addStretch()
            self.controls_layout.addLayout(row)

            # Channel Slider
            self.channel_row = QWidget()
            l = QHBoxLayout(self.channel_row)
            l.setContentsMargins(0, 0, 0, 0)
            l.addWidget(QLabel("Channel"))
            self.c_slider = QSlider(Qt.Horizontal)
            self.c_slider.setRange(0, self.C - 1)
            self.c_slider.valueChanged.connect(self.on_channel_change)
            l.addWidget(self.c_slider)
            self.controls_layout.addWidget(self.channel_row)
            self.channel_row.setVisible(False)

        # Time Slider
        if self.T > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Time"))
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, self.T - 1)
            sl.valueChanged.connect(self.on_time_change)
            row.addWidget(sl)
            self.controls_layout.addLayout(row)

    def _setup_menu(self):
        menubar = self.menuBar()

        # Adjust
        adjust_menu = menubar.addMenu("Adjust")
        channels_action = QAction("Channels && Contrast...", self)
        channels_action.setShortcut("Shift+C")
        channels_action.triggered.connect(self.show_channel_panel)
        adjust_menu.addAction(channels_action)

        # View
        view_menu = menubar.addMenu("View")
        crosshair_action = QAction("Toggle Crosshair", self)
        crosshair_action.setShortcut("H")
        crosshair_action.triggered.connect(self.toggle_crosshair)
        view_menu.addAction(crosshair_action)

        reset_view_action = QAction("Reset View", self)
        reset_view_action.setShortcut("A")
        reset_view_action.triggered.connect(self.reset_cameras)
        view_menu.addAction(reset_view_action)

        overlay_action = QAction("Overlay Settings...", self)
        overlay_action.triggered.connect(self.show_overlay_settings_dialog)
        view_menu.addAction(overlay_action)

        # Image
        image_menu = menubar.addMenu("Image")
        info_action = QAction("Image Info", self)
        info_action.triggered.connect(self.show_metadata_dialog)
        image_menu.addAction(info_action)

    def _setup_events(self):
        def make_click_handler(view_name):
            def handler(e):
                self.on_shift_click(e, view_name)

            return handler

        def make_move_handler(view_name):
            def handler(e):
                self.on_mouse_move(e, view_name)

            return handler

        for canvas, view_name in [
            (self.canvas_yx, "yx"),
            (self.canvas_zy, "zy"),
            (self.canvas_zx, "zx"),
        ]:
            click_handler = make_click_handler(view_name)
            move_handler = make_move_handler(view_name)
            canvas.events.mouse_press.connect(click_handler)
            canvas.events.mouse_move.connect(move_handler)
            self._event_handlers.append(
                (canvas.events.mouse_press, click_handler)
            )
            self._event_handlers.append(
                (canvas.events.mouse_move, move_handler)
            )

    def on_shift_click(self, e, view_name):
        if e.button != 1:
            return

        # only respond when 'Shift' is held
        if "Shift" not in e.modifiers:
            return

        # Map to visual coordinates
        # We use the inverse transform to go from Canvas (Screen) -> Local (Pixel Indices)
        # This handles the STTransform (Scale) automatically.
        if view_name == "yx":
            tr_to_data = self.vis_yx.layers[0].get_transform(
                map_from="canvas", map_to="visual"
            )
            pos = tr_to_data.map(e.pos)

            x = int(pos[0])
            y = int(pos[1])

            self.cx = np.clip(x, 0, self.X - 1)
            self.cy = np.clip(y, 0, self.Y - 1)

        elif view_name == "zy":
            tr_to_data = self.vis_zy.layers[0].get_transform(
                map_from="canvas", map_to="visual"
            )
            pos = tr_to_data.map(e.pos)

            # ZY View Local Coords: (Z, Y) (because we transposed data to (T, X, C, Y, Z) -> sliced X -> (C, Y, Z) -> Vispy sees (Z, Y))
            # Wait, Vispy Image visual sees (col, row) = (x, y).
            # Data is (C, Y, Z).
            # Vispy Image interprets (Y, Z) as (row, col)?
            # Vispy Image(data):
            # If data is (H, W), Vispy displays it with W along X and H along Y.
            # Here data slice is (C, Y, Z) -> (Y, Z) spatial.
            # So Y is height (y-axis), Z is width (x-axis).
            # So local pos[0] is Z, pos[1] is Y.

            z = int(pos[0])
            y = int(pos[1])

            self.cz = np.clip(z, 0, self.Z - 1)
            self.cy = np.clip(y, 0, self.Y - 1)

        elif view_name == "zx":
            tr_to_data = self.vis_zx.layers[0].get_transform(
                map_from="canvas", map_to="visual"
            )
            pos = tr_to_data.map(e.pos)

            # ZX View Data Slice: (C, Z, X) -> (Z, X) spatial.
            # Z is height (y-axis), X is width (x-axis).
            # So local pos[0] is X, pos[1] is Z.

            x = int(pos[0])
            z = int(pos[1])

            self.cx = np.clip(x, 0, self.X - 1)
            self.cz = np.clip(z, 0, self.Z - 1)

        self.update_views()

    def on_mouse_move(self, e, view_name):
        """Update status bar with position and intensity at cursor."""
        # Map canvas coordinates to data coordinates
        if view_name == "yx":
            tr_to_data = self.vis_yx.layers[0].get_transform(
                map_from="canvas", map_to="visual"
            )
            pos = tr_to_data.map(e.pos)
            x, y = int(pos[0]), int(pos[1])
            z = self.cz  # Current Z slice

            if 0 <= x < self.X and 0 <= y < self.Y:
                cache = self.vis_yx.current_slice_cache
                if cache is not None:
                    vals = self._get_intensity_values(cache, y, x)
                    self.info_label.setText(
                        f"X: {x}  Y: {y}  Z: {z}  Val: [{vals}]"
                    )
            else:
                self.info_label.setText("")

        elif view_name == "zy":
            tr_to_data = self.vis_zy.layers[0].get_transform(
                map_from="canvas", map_to="visual"
            )
            pos = tr_to_data.map(e.pos)
            z, y = int(pos[0]), int(pos[1])
            x = self.cx  # Current X slice

            if 0 <= z < self.Z and 0 <= y < self.Y:
                cache = self.vis_zy.current_slice_cache
                if cache is not None:
                    vals = self._get_intensity_values(cache, y, z)
                    self.info_label.setText(
                        f"X: {x}  Y: {y}  Z: {z}  Val: [{vals}]"
                    )
            else:
                self.info_label.setText("")

        elif view_name == "zx":
            tr_to_data = self.vis_zx.layers[0].get_transform(
                map_from="canvas", map_to="visual"
            )
            pos = tr_to_data.map(e.pos)
            x, z = int(pos[0]), int(pos[1])
            y = self.cy  # Current Y slice

            if 0 <= x < self.X and 0 <= z < self.Z:
                cache = self.vis_zx.current_slice_cache
                if cache is not None:
                    vals = self._get_intensity_values(cache, z, x)
                    self.info_label.setText(
                        f"X: {x}  Y: {y}  Z: {z}  Val: [{vals}]"
                    )
            else:
                self.info_label.setText("")

    def _get_intensity_values(self, cache, row, col):
        """Extract intensity values from cache for all channels."""
        vals = []
        for c in range(cache.shape[0]):
            try:
                val = cache[c, row, col]
                vals.append(f"{val:.1f}")
            except IndexError:
                pass
        return ", ".join(vals)

    def toggle_crosshair(self):
        """Toggle visibility of crosshair lines in all views."""
        self.crosshair_visible = not self.crosshair_visible
        visible = self.crosshair_visible

        # YX view crosshairs
        self.line_yx_v.visible = visible
        self.line_yx_h.visible = visible

        # ZY view crosshairs
        self.line_zy_v.visible = visible
        self.line_zy_h.visible = visible

        # ZX view crosshairs
        self.line_zx_v.visible = visible
        self.line_zx_h.visible = visible

        self.canvas_yx.update()
        self.canvas_zy.update()
        self.canvas_zx.update()

    def update_views(self):
        # Update Slices
        # YX View: Slice at Z
        self.vis_yx.update_slice(self.ct, self.cz)

        # ZY View: Slice at X
        self.vis_zy.update_slice(self.ct, self.cx)

        # ZX View: Slice at Y
        self.vis_zx.update_slice(self.ct, self.cy)

        # Update Crosshairs (Scaled positions with half-pixel offset to center on pixel)
        sx, sy, sz = self.scale[2], self.scale[1], self.scale[0]

        # YX: V at X, H at Y
        self.line_yx_v.set_data(pos=(self.cx + 0.5) * sx)
        self.line_yx_h.set_data(pos=(self.cy + 0.5) * sy)

        # ZY: V at Z, H at Y
        self.line_zy_v.set_data(pos=(self.cz + 0.5) * sz)
        self.line_zy_h.set_data(pos=(self.cy + 0.5) * sy)

        # ZX: V at X, H at Z
        self.line_zx_v.set_data(pos=(self.cx + 0.5) * sx)
        self.line_zx_h.set_data(pos=(self.cz + 0.5) * sz)

        self._update_overlays()
        self.canvas_yx.update()
        self.canvas_zy.update()
        self.canvas_zx.update()

        # Notify subscribers (ChannelPanel, line-profile, …) that the
        # displayed slices changed so they can re-pull from
        # renderer.current_slice_cache.
        self.view_changed.emit(self)

    def reset_cameras(self):
        sx, sy, sz = self.scale[2], self.scale[1], self.scale[0]

        # Temporarily disable sync during reset
        self._syncing_cameras = True

        # Store full extents for zoom factor calculations
        self._full_x = self.X * sx
        self._full_y = self.Y * sy
        self._full_z = self.Z * sz

        # YX
        self.view_yx.camera.rect = (0, 0, self._full_x, self._full_y)
        self.view_yx.camera.flip = (False, True, False)

        # ZY
        self.view_zy.camera.rect = (0, 0, self._full_z, self._full_y)
        self.view_zy.camera.flip = (False, True, False)

        # ZX
        self.view_zx.camera.rect = (0, 0, self._full_x, self._full_z)
        self.view_zx.camera.flip = (False, True, False)

        self._syncing_cameras = False

        # Force canvas updates
        self._update_overlays()
        self.canvas_yx.update()
        self.canvas_zy.update()
        self.canvas_zx.update()

    def _on_camera_change(self, source_view):
        """Sync camera axes across views when one view's camera changes."""
        if self._syncing_cameras:
            return
        self._syncing_cameras = True

        try:
            if source_view == "yx":
                # YX changed: sync Y to ZY, X to ZX, and zoom Z proportionally
                yx_rect = self.view_yx.camera.rect

                # Calculate zoom factor from YX (use Y axis as reference)
                zoom_factor = (
                    self._full_y / yx_rect.height
                    if yx_rect.height > 0
                    else 1.0
                )

                # Sync Y axis to ZY, and scale Z axis by same zoom factor
                zy_rect = self.view_zy.camera.rect
                new_z_width = self._full_z / zoom_factor
                # Center the Z axis around current center
                z_center = zy_rect.left + zy_rect.width / 2
                new_z_left = z_center - new_z_width / 2
                new_zy_rect = (
                    new_z_left,
                    yx_rect.bottom,
                    new_z_width,
                    yx_rect.height,
                )
                self.view_zy.camera.rect = new_zy_rect

                # Sync X axis to ZX, and scale Z axis by same zoom factor
                zx_rect = self.view_zx.camera.rect
                new_z_height = self._full_z / zoom_factor
                z_center = zx_rect.bottom + zx_rect.height / 2
                new_z_bottom = z_center - new_z_height / 2
                new_zx_rect = (
                    yx_rect.left,
                    new_z_bottom,
                    yx_rect.width,
                    new_z_height,
                )
                self.view_zx.camera.rect = new_zx_rect

            elif source_view == "zy":
                # ZY changed: sync Y to YX, Z to ZX
                zy_rect = self.view_zy.camera.rect

                # Calculate zoom factor from ZY (use Y axis as reference)
                zoom_factor = (
                    self._full_y / zy_rect.height
                    if zy_rect.height > 0
                    else 1.0
                )

                # Sync Y axis to YX, scale X by same zoom factor
                yx_rect = self.view_yx.camera.rect
                new_x_width = self._full_x / zoom_factor
                x_center = yx_rect.left + yx_rect.width / 2
                new_x_left = x_center - new_x_width / 2
                new_yx_rect = (
                    new_x_left,
                    zy_rect.bottom,
                    new_x_width,
                    zy_rect.height,
                )
                self.view_yx.camera.rect = new_yx_rect

                # Sync Z axis to ZX
                zx_rect = self.view_zx.camera.rect
                new_zx_rect = (
                    new_x_left,
                    zy_rect.left,
                    new_x_width,
                    zy_rect.width,
                )
                self.view_zx.camera.rect = new_zx_rect

            elif source_view == "zx":
                # ZX changed: sync X to YX, Z to ZY
                zx_rect = self.view_zx.camera.rect

                # Calculate zoom factor from ZX (use X axis as reference)
                zoom_factor = (
                    self._full_x / zx_rect.width if zx_rect.width > 0 else 1.0
                )

                # Sync X axis to YX, scale Y by same zoom factor
                yx_rect = self.view_yx.camera.rect
                new_y_height = self._full_y / zoom_factor
                y_center = yx_rect.bottom + yx_rect.height / 2
                new_y_bottom = y_center - new_y_height / 2
                new_yx_rect = (
                    zx_rect.left,
                    new_y_bottom,
                    zx_rect.width,
                    new_y_height,
                )
                self.view_yx.camera.rect = new_yx_rect

                # Sync Z axis to ZY
                zy_rect = self.view_zy.camera.rect
                new_zy_rect = (
                    zx_rect.bottom,
                    new_y_bottom,
                    zx_rect.height,
                    new_y_height,
                )
                self.view_zy.camera.rect = new_zy_rect
        finally:
            self._syncing_cameras = False

        self._update_overlays()

    def on_time_change(self, val):
        self.ct = val
        self.update_views()

    def on_mode_change(self, idx):
        mode = "composite" if idx == 0 else "single"
        self.channel_row.setVisible(mode == "single")
        self.proxy.set_mode(mode)
        self.canvas_yx.update()
        self.canvas_zy.update()
        self.canvas_zx.update()

    def on_channel_change(self, val):
        self.proxy.set_active_channel(val)
        self.canvas_yx.update()
        self.canvas_zy.update()
        self.canvas_zx.update()

    def show_channel_panel(self):
        if not hasattr(self, "channel_panel") or self.channel_panel is None:
            self.channel_panel = ChannelPanel(self, parent=self)
        self.channel_panel.show()
        self.channel_panel.raise_()
        self.channel_panel.refresh_ui()

    @property
    def renderer(self):
        return self.proxy

    @property
    def img_data(self):
        """Compatibility alias for widgets expecting the main viewer API."""
        return self.data

    @img_data.setter
    def img_data(self, value):
        self.data = value

    @property
    def canvas(self):
        # ChannelPanel calls viewer.canvas.update()
        # We return the YX canvas as a proxy.
        # Ideally, we should update ALL canvases.
        # We can wrap this in a dummy object or just return one canvas
        # and ensure our proxy updates the others.
        # Since MultiViewChannelProxy updates all visuals, calling update() on one canvas
        # might not be enough if they are on different canvases?
        # Actually, they ARE on different canvases.
        # So we need a CanvasProxy that updates all 3.
        return self._canvas_proxy

    @property
    def _canvas_proxy(self):
        class CanvasProxy:
            def __init__(self, viewer):
                self.viewer = viewer

            def update(self):
                self.viewer.canvas_yx.update()
                self.viewer.canvas_zy.update()
                self.viewer.canvas_zx.update()

        return CanvasProxy(self)

    def show_metadata_dialog(self):
        dlg = MetadataDialog(self.meta, parent=self)
        dlg.exec_()

    def show_overlay_settings_dialog(self):
        dlg = OverlaySettingsDialog(self._overlay_config, parent=self)
        if dlg.exec_():
            self._overlay_config = dlg.get_config()
            for overlay in self._overlays.values():
                overlay.set_config(self._overlay_config)
            self.canvas_yx.update()
            self.canvas_zy.update()
            self.canvas_zx.update()

    def closeEvent(self, event):
        # Disconnect all event handlers to prevent callbacks to dead objects
        for event_emitter, handler in self._event_handlers:
            try:
                event_emitter.disconnect(handler)
            except Exception:
                pass
        self._event_handlers.clear()

        for overlay in self._overlays.values():
            overlay.remove()

        # Unparent crosshair visuals from scene
        for line in [
            self.line_yx_v,
            self.line_yx_h,
            self.line_zy_v,
            self.line_zy_h,
            self.line_zx_v,
            self.line_zx_h,
        ]:
            try:
                line.parent = None
            except Exception:
                pass

        # Release data proxy ref count (or close if no ref counting)
        if hasattr(self.data, "release"):
            try:
                self.data.release()
            except Exception:
                pass
        elif hasattr(self.data, "close"):
            try:
                self.data.close()
            except Exception:
                pass

        # Clear data reference to help GC after proxies are released.
        self.data = None
        super().closeEvent(event)

    def _init_overlays(self):
        sz, _sy, sx = self.scale
        self._overlay_config = dict(ScaleTimestampOverlay.DEFAULT_CONFIG)

        self._overlays = {
            "yx": ScaleTimestampOverlay(
                self.view_yx,
                axis_spacing_um=sx,
                world_units_are_um=True,
                get_time_index=lambda: self.ct,
                get_timestamps=lambda: self.meta.get("timestamps", []),
                config=self._overlay_config,
            ),
            "zy": ScaleTimestampOverlay(
                self.view_zy,
                axis_spacing_um=sz,
                world_units_are_um=True,
                get_time_index=lambda: self.ct,
                get_timestamps=lambda: self.meta.get("timestamps", []),
                config=self._overlay_config,
            ),
            "zx": ScaleTimestampOverlay(
                self.view_zx,
                axis_spacing_um=sx,
                world_units_are_um=True,
                get_time_index=lambda: self.ct,
                get_timestamps=lambda: self.meta.get("timestamps", []),
                config=self._overlay_config,
            ),
        }

    def _update_overlays(self):
        for overlay in self._overlays.values():
            overlay.update()
