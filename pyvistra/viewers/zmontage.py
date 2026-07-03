"""
Z-Montage Viewer for pyvistra.

Tiles every Z-slice of a single loaded image into a grid within one shared
vispy canvas/camera, so panning and zooming moves the whole montage as one
image. Contrast this with TiledViewer, which shows N separate *files* each
in its own canvas -- here there is exactly one image, and Z is the axis
being "distributed" across tiles while T/channel/mode stay slide-able.
"""

import math

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAction,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from vispy import scene

from ..contrast import compute_percentile_clim
from ..visuals.image import CompositeImageVisual, MultiViewChannelProxy
from ..widgets import ChannelPanel, MetadataDialog, ZMontageSettingsDialog


class ZMontageViewer(QMainWindow):
    """Grid of every Z-slice of one image, sharing a single view/camera."""

    # Emitted when the displayed timepoint changes, so ChannelPanel can
    # refresh its histograms (mirrors OrthoViewer/ImageWindow).
    view_changed = Signal(object)

    DEFAULT_CONFIG = {
        "show_labels": True,
        "label_anchor": "top-left",
        "label_font_size": 10.0,
        "label_color": (1.0, 1.0, 1.0, 0.8),
        "gap_fraction": 0.05,
    }

    def __init__(
        self,
        data,
        meta=None,
        title="Z-Montage View",
        channel_colormaps=None,
        t_idx=0,
    ):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(1000, 800)

        self.data = data  # (T, Z, C, Y, X)
        self.img_data = data  # ChannelPanel reads this for dtype bounds
        self.meta = meta or {}
        self.T, self.Z, self.C, self.Y, self.X = self.data.shape

        self.scale = self.meta.get("scale", (1.0, 1.0, 1.0))
        _, sy, sx = self.scale

        self.t_idx = min(t_idx, self.T - 1)
        self.mode = "composite"
        self.channel_idx = 0

        self._config = dict(self.DEFAULT_CONFIG)
        self._gap = 0.0
        self._cols = max(1, math.ceil(math.sqrt(self.Z)))
        self._rows = max(1, math.ceil(self.Z / self._cols))

        self.channel_panel = None

        # -- Layout --
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # -- Toolbar: mode, channel, time --
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)

        toolbar_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Composite", "Single Channel"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        toolbar_layout.addWidget(self.mode_combo)

        toolbar_layout.addSpacing(10)

        self.channel_widget = QWidget()
        channel_layout = QHBoxLayout(self.channel_widget)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.addWidget(QLabel("Channel:"))
        self.channel_slider = QSlider(Qt.Horizontal)
        self.channel_slider.setRange(0, max(0, self.C - 1))
        self.channel_slider.setFixedWidth(80)
        self.channel_slider.valueChanged.connect(self._on_channel_changed)
        channel_layout.addWidget(self.channel_slider)
        self.channel_label = QLabel("0")
        self.channel_label.setFixedWidth(20)
        channel_layout.addWidget(self.channel_label)
        self.channel_widget.setVisible(False)
        toolbar_layout.addWidget(self.channel_widget)

        toolbar_layout.addSpacing(20)

        self.time_widget = QWidget()
        time_layout = QHBoxLayout(self.time_widget)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, self.T - 1)
        self.time_slider.setValue(self.t_idx)
        self.time_slider.setFixedWidth(100)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        time_layout.addWidget(self.time_slider)
        self.time_label = QLabel(str(self.t_idx))
        self.time_label.setFixedWidth(25)
        time_layout.addWidget(self.time_label)
        self.time_widget.setVisible(self.T > 1)
        toolbar_layout.addWidget(self.time_widget)

        toolbar_layout.addStretch()

        main_layout.addWidget(toolbar)

        # -- Canvas: one shared view/camera for the whole grid --
        self.canvas = scene.SceneCanvas(keys=None, bgcolor="black")
        self.view = self.canvas.central_widget.add_view()
        self.view.padding = 0
        self.view.camera = "panzoom"
        self.view.camera.aspect = 1
        main_layout.addWidget(self.canvas.native, 1)

        # -- Build the grid: one CompositeImageVisual per Z-slice --
        self.z_renderers = []
        for z in range(self.Z):
            renderer = CompositeImageVisual(
                self.view, self.data, scale=(sy, sx)
            )
            renderer.update_slice(self.t_idx, z)
            self.z_renderers.append(renderer)

        self.renderer = MultiViewChannelProxy(self.z_renderers)

        if channel_colormaps:
            for channel_idx, cmap_name in channel_colormaps.items():
                self.renderer.set_colormap(channel_idx, cmap_name)

        self._z_labels = []
        self._layout_tiles()
        self._reset_view()

        self._setup_menu()

    def _layout_tiles(self):
        """Position every tile and (re)build its Z label from ``self._config``."""
        _, sy, sx = self.scale
        tile_w = self.X * sx
        tile_h = self.Y * sy
        self._gap = self._config["gap_fraction"] * max(tile_w, tile_h)

        for label in self._z_labels:
            label.parent = None
        self._z_labels = []

        for z, renderer in enumerate(self.z_renderers):
            row, col = divmod(z, self._cols)
            tx = col * (tile_w + self._gap)
            ty = row * (tile_h + self._gap)
            renderer.set_transform(translate_x=tx, translate_y=ty)

            if self._config["show_labels"]:
                self._z_labels.append(
                    self._make_label(z, tx, ty, tile_w, tile_h)
                )

    def _make_label(self, z, tx, ty, tile_w, tile_h):
        anchor = self._config["label_anchor"]
        pad = 2.0
        anchor_x = "right" if "right" in anchor else "left"
        anchor_y = "bottom" if "bottom" in anchor else "top"
        x = tx + tile_w - pad if anchor_x == "right" else tx + pad
        y = ty + tile_h - pad if anchor_y == "bottom" else ty + pad

        label = scene.visuals.Text(
            text=f"z={z}",
            color=self._config["label_color"],
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            font_size=self._config["label_font_size"],
            parent=self.view.scene,
        )
        label.pos = (x, y)
        label.order = 10_000
        return label

    def _setup_menu(self):
        menubar = self.menuBar()

        adjust_menu = menubar.addMenu("Adjust")

        channels_action = QAction("Channels && Contrast...", self)
        channels_action.setShortcut("Shift+C")
        channels_action.triggered.connect(self.show_channel_panel)
        adjust_menu.addAction(channels_action)

        auto_action = QAction("Auto Contrast", self)
        auto_action.setShortcut("C")
        auto_action.triggered.connect(self._auto_contrast)
        adjust_menu.addAction(auto_action)

        view_menu = menubar.addMenu("View")

        reset_action = QAction("Reset View", self)
        reset_action.setShortcut("A")
        reset_action.triggered.connect(self._reset_view)
        view_menu.addAction(reset_action)

        settings_action = QAction("Montage Settings...", self)
        settings_action.triggered.connect(self.show_montage_settings)
        view_menu.addAction(settings_action)

        image_menu = menubar.addMenu("Image")

        info_action = QAction("Image Info", self)
        info_action.setShortcut("Shift+I")
        info_action.triggered.connect(self.show_metadata_dialog)
        image_menu.addAction(info_action)

    # -- Controls --

    def _on_mode_changed(self, index):
        self.mode = "composite" if index == 0 else "single"
        self.channel_widget.setVisible(self.mode == "single" and self.C > 1)
        self.renderer.set_mode(self.mode)
        if self.mode == "single":
            self.renderer.set_active_channel(self.channel_idx)
        self.canvas.update()

    def _on_channel_changed(self, value):
        self.channel_idx = value
        self.channel_label.setText(str(value))
        self.renderer.set_active_channel(value)
        self.canvas.update()

    def _on_time_changed(self, value):
        self.t_idx = value
        self.time_label.setText(str(value))
        for z, renderer in enumerate(self.z_renderers):
            renderer.update_slice(value, z)
        self.canvas.update()
        self.view_changed.emit(self)

    def _auto_contrast(self, pct_low=0.5, pct_high=99.5):
        """Percentile-based auto-contrast, aggregated across every tile.

        Aggregating (rather than auto-contrasting each tile independently)
        keeps relative intensity comparable across Z, which is the point of
        looking at a Z-montage in the first place.
        """
        for c in range(self.C):
            planes = [
                r.current_slice_cache[c]
                for r in self.z_renderers
                if r.current_slice_cache is not None
                and c < r.current_slice_cache.shape[0]
            ]
            if not planes:
                continue
            combined = np.concatenate([p.ravel() for p in planes])
            mn, mx = compute_percentile_clim(combined, pct_low, pct_high)
            self.renderer.set_clim(c, mn, mx)
        self.canvas.update()

    def _reset_view(self):
        _, sy, sx = self.scale
        width = self._cols * (self.X * sx + self._gap) - self._gap
        height = self._rows * (self.Y * sy + self._gap) - self._gap
        self.view.camera.rect = (0, 0, max(width, 1), max(height, 1))
        self.view.camera.flip = (False, True, False)
        self.canvas.update()

    def show_montage_settings(self):
        dlg = ZMontageSettingsDialog(self._config, parent=self)
        if dlg.exec_():
            self._config = dlg.get_config()
            self._layout_tiles()
            self._reset_view()

    def show_channel_panel(self):
        if self.channel_panel is None:
            self.channel_panel = ChannelPanel(self, parent=self)
        self.channel_panel.show()
        self.channel_panel.raise_()

    def show_metadata_dialog(self):
        dlg = MetadataDialog(self.meta, parent=self)
        dlg.exec_()

    def closeEvent(self, event):
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
        super().closeEvent(event)
