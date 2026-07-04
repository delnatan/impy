"""Unified channel-control panel.

One floating dialog handles every per-channel visual adjustment:
colormap, contrast (min/max + compact histogram), gamma, and
visibility. The header hosts panel-wide actions (auto-contrast,
tighten/loosen).

Subscribes to ``renderer.display`` (a :class:`ChannelDisplayList`) so
state mutations from any source — programmatic or another widget —
update the UI automatically. The panel writes user input back through
the same list, which propagates to the underlying vispy layers.

This module replaces the old ``ContrastDialog`` + ``ChannelPanel``
pair. The single-channel "big histogram" view is intentionally
omitted; the per-row compact histogram covers the common case.
"""

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyvistra import colormaps as _colormaps
from pyvistra.contrast import compute_percentile_clim
from .histogram import (
    CompactHistogramWidget,
    configure_spinbox_for_range,
    get_safe_contrast_bounds,
)


class ChannelRow(QWidget):
    """One row per channel: visibility, colormap swatch, min/max + histogram, gamma."""

    visibilityChanged = Signal(int, bool)  # channel_idx, visible
    climChanged = Signal(int, float, float)  # channel_idx, vmin, vmax
    colormapChanged = Signal(int, str)  # channel_idx, colormap_name
    gammaChanged = Signal(int, float)  # channel_idx, gamma

    def __init__(
        self, channel_idx, channel_name, color, data_dtype=None, parent=None
    ):
        super().__init__(parent)
        self.channel_idx = channel_idx
        self.data_dtype = data_dtype

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self.chk_visible = QCheckBox()
        self.chk_visible.setChecked(True)
        self.chk_visible.setToolTip("Toggle channel visibility")
        self.chk_visible.toggled.connect(self._on_visibility_changed)
        layout.addWidget(self.chk_visible)

        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(20, 20)
        self.color_btn.setCursor(Qt.PointingHandCursor)
        self.color_btn.setToolTip("Change colormap")
        self._update_color_swatch(color)
        self.color_btn.clicked.connect(self._show_colormap_menu)
        layout.addWidget(self.color_btn)

        self.name_label = QLabel(channel_name)
        self.name_label.setFixedWidth(40)
        self.name_label.setStyleSheet("color: #EEE; font-size: 11px;")
        self.name_label.setToolTip(channel_name)
        layout.addWidget(self.name_label)

        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(1)
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setSingleStep(10)
        self.min_spin.setKeyboardTracking(False)
        self.min_spin.setFixedWidth(65)
        self.min_spin.setToolTip("Minimum intensity")
        self.min_spin.valueChanged.connect(self._on_min_changed)
        layout.addWidget(self.min_spin)

        self.histogram = CompactHistogramWidget()
        self.histogram.climChanged.connect(self._on_histogram_clim_changed)
        layout.addWidget(self.histogram, 1)

        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(1)
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setSingleStep(10)
        self.max_spin.setKeyboardTracking(False)
        self.max_spin.setFixedWidth(65)
        self.max_spin.setToolTip("Maximum intensity")
        self.max_spin.valueChanged.connect(self._on_max_changed)
        layout.addWidget(self.max_spin)

        gamma_label = QLabel("γ")
        gamma_label.setStyleSheet("color: #AAA; font-size: 10px;")
        gamma_label.setFixedWidth(10)
        layout.addWidget(gamma_label)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 4.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setFixedWidth(50)
        self.gamma_spin.setToolTip("Gamma correction")
        self.gamma_spin.valueChanged.connect(self._on_gamma_changed)
        layout.addWidget(self.gamma_spin)

        self.current_colormap = "White"

    def _update_color_swatch(self, color):
        self.color_btn.setStyleSheet(
            f"background-color: {color}; border: 1px solid #555; border-radius: 3px;"
        )

    def _on_visibility_changed(self, checked):
        self.visibilityChanged.emit(self.channel_idx, checked)

    def _on_min_changed(self, value):
        max_val = self.max_spin.value()
        if value < max_val:
            self.climChanged.emit(self.channel_idx, value, max_val)
            self.histogram.blockSignals(True)
            self.histogram.set_clim(value, max_val)
            self.histogram.blockSignals(False)

    def _on_max_changed(self, value):
        min_val = self.min_spin.value()
        if value > min_val:
            self.climChanged.emit(self.channel_idx, min_val, value)
            self.histogram.blockSignals(True)
            self.histogram.set_clim(min_val, value)
            self.histogram.blockSignals(False)

    def _on_histogram_clim_changed(self, vmin, vmax):
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        self.climChanged.emit(self.channel_idx, vmin, vmax)

    def _on_gamma_changed(self, value):
        self.gammaChanged.emit(self.channel_idx, value)

    def _show_colormap_menu(self):
        from qtpy.QtWidgets import QMenu

        menu = QMenu(self)
        for cmap_name in _colormaps.names():
            action = menu.addAction(cmap_name)
            action.triggered.connect(
                lambda checked, name=cmap_name: self._on_colormap_selected(name)
            )
        menu.exec_(
            self.color_btn.mapToGlobal(self.color_btn.rect().bottomLeft())
        )

    def _on_colormap_selected(self, cmap_name):
        self.current_colormap = cmap_name
        self.colormapChanged.emit(self.channel_idx, cmap_name)

    def set_data(self, data_slice, color):
        """Update histogram data and swatch color."""
        self._update_color_swatch(color)
        self.histogram.set_data(data_slice, color)

        data_min = self.histogram.data_min
        data_max = self.histogram.data_max
        configure_spinbox_for_range(self.min_spin, data_min, data_max)
        configure_spinbox_for_range(self.max_spin, data_min, data_max)
        dtype = self.data_dtype if self.data_dtype is not None else data_slice.dtype
        hard_min, hard_max = get_safe_contrast_bounds(dtype, data_min, data_max)
        self.min_spin.setRange(hard_min, hard_max)
        self.max_spin.setRange(hard_min, hard_max)

    def set_clim(self, vmin, vmax):
        """Update contrast displays without emitting signals."""
        self.histogram.blockSignals(True)
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.histogram.set_clim(vmin, vmax)
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.histogram.blockSignals(False)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)

    def set_visible_state(self, visible):
        self.chk_visible.blockSignals(True)
        self.chk_visible.setChecked(visible)
        self.chk_visible.blockSignals(False)

    def set_gamma(self, gamma):
        self.gamma_spin.blockSignals(True)
        self.gamma_spin.setValue(gamma)
        self.gamma_spin.blockSignals(False)


class ChannelPanel(QWidget):
    """Per-channel controls + global contrast actions.

    Plain widget: viewers embed it in a ``QDockWidget`` via
    :func:`show_channel_dock`. Callers that want it floating set window
    flags themselves (see the tiled viewer's per-tile panel).
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Channels")
        self.resize(520, min(220 + viewer.C * 60, 540))

        # Auto-contrast tighten/loosen state.
        self._pct_low = 0.5
        self._pct_high = 99.98
        self._pct_step = 0.01

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        if hasattr(viewer, "window_id"):
            wid_label = QLabel(f"<b>Window: {viewer.window_id}</b>")
            wid_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(wid_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        self.rows_layout = QVBoxLayout(scroll_content)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        self.channel_rows = []
        self._setup_channel_rows()

        # Contrast action row.
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_loosen = QPushButton("-")
        btn_loosen.setFixedWidth(30)
        btn_loosen.setToolTip("Loosen contrast (expand percentile range)")
        btn_loosen.clicked.connect(lambda: self._adjust_pct(-1))
        btn_layout.addWidget(btn_loosen)

        btn_auto = QPushButton("Auto Contrast All")
        btn_auto.clicked.connect(self._auto_contrast_all)
        btn_layout.addWidget(btn_auto)

        btn_tighten = QPushButton("+")
        btn_tighten.setFixedWidth(30)
        btn_tighten.setToolTip("Tighten contrast (shrink percentile range)")
        btn_tighten.clicked.connect(lambda: self._adjust_pct(1))
        btn_layout.addWidget(btn_tighten)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Subscribe to display state changes (clim/gamma/colormap/visible).
        # Renderers without a display attribute fall back to polling.
        self._display_unsubscribe = None
        display = getattr(self.viewer.renderer, "display", None)
        if display is not None:
            self._display_unsubscribe = display.subscribe(
                self._on_display_changed
            )

        # Refresh histograms when the displayed slice changes. Debounced:
        # scrubbing the T/Z sliders fires view_changed per tick, and an
        # np.histogram per channel per tick makes scrubbing stutter.
        self._hist_timer = QTimer(self)
        self._hist_timer.setSingleShot(True)
        self._hist_timer.setInterval(150)
        self._hist_timer.timeout.connect(self._refresh_histograms)
        self._hist_stale = False
        if hasattr(self.viewer, "view_changed"):
            self.viewer.view_changed.connect(self._on_view_changed)

        self.refresh_ui()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_channel_rows(self):
        n_channels = self.viewer.C
        meta_channels = self.viewer.meta.get("channels", [])

        for c in range(n_channels):
            if c < len(meta_channels) and "name" in meta_channels[c]:
                ch_name = meta_channels[c]["name"]
            else:
                ch_name = f"Ch {c + 1}"

            color = self._swatch_color(c)
            img_data = getattr(self.viewer, "img_data", None)
            row = ChannelRow(
                c, ch_name, color, data_dtype=getattr(img_data, "dtype", None)
            )
            row.visibilityChanged.connect(self._on_visibility_changed)
            row.climChanged.connect(self._on_clim_changed)
            row.colormapChanged.connect(self._on_colormap_changed)
            row.gammaChanged.connect(self._on_gamma_changed)

            self.channel_rows.append(row)
            self.rows_layout.addWidget(row)

        self.rows_layout.addStretch()

    def _swatch_color(self, c_idx):
        colors = self.viewer.renderer.channel_colors
        if not colors:
            return "#888888"
        return colors[c_idx % len(colors)]

    # ------------------------------------------------------------------
    # User input → renderer
    # ------------------------------------------------------------------

    def _on_visibility_changed(self, channel_idx, visible):
        self.viewer.renderer.set_channel_visible(channel_idx, visible)
        self.viewer.canvas.update()

    def _on_clim_changed(self, channel_idx, vmin, vmax):
        self.viewer.renderer.set_clim(channel_idx, vmin, vmax)
        self.viewer.canvas.update()

    def _on_colormap_changed(self, channel_idx, cmap_name):
        self.viewer.renderer.set_colormap(channel_idx, cmap_name)
        self.viewer.canvas.update()

    def _on_gamma_changed(self, channel_idx, gamma):
        self.viewer.renderer.set_gamma(channel_idx, gamma)
        self.viewer.canvas.update()

    # ------------------------------------------------------------------
    # Display state subscription
    # ------------------------------------------------------------------

    def _on_display_changed(self, channel_idx, field):
        """Called when any channel's display state mutates externally."""
        if channel_idx >= len(self.channel_rows):
            return
        row = self.channel_rows[channel_idx]
        if field == "clim":
            vmin, vmax = self.viewer.renderer.get_clim(channel_idx)
            row.set_clim(vmin, vmax)
        elif field == "gamma":
            row.set_gamma(self.viewer.renderer.get_gamma(channel_idx))
        elif field == "colormap_name":
            row._update_color_swatch(self._swatch_color(channel_idx))
            row.histogram.color = QColor(self._swatch_color(channel_idx))
            row.histogram.update()
        elif field == "visible":
            row.set_visible_state(
                self.viewer.renderer.get_channel_visible(channel_idx)
            )

    def _on_view_changed(self, _viewer):
        """Schedule a histogram refresh after a slice/time change."""
        if not self.isVisible():
            self._hist_stale = True
            return
        self._hist_timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        if self._hist_stale:
            self._hist_stale = False
            self._refresh_histograms()

    # ------------------------------------------------------------------
    # Contrast actions
    # ------------------------------------------------------------------

    def _auto_contrast_all(self):
        cache = self.viewer.renderer.current_slice_cache
        if cache is None:
            return
        self._pct_low = 0.5
        self._pct_high = 99.98
        for c in range(len(self.channel_rows)):
            if c < cache.shape[0]:
                mn, mx = compute_percentile_clim(
                    cache[c], self._pct_low, self._pct_high
                )
                self.viewer.renderer.set_clim(c, mn, mx)
        self.viewer.canvas.update()

    def _adjust_pct(self, direction):
        """Tighten (+1) or loosen (-1) the percentile range, then apply."""
        if direction > 0:
            self._pct_low += self._pct_step
            self._pct_high -= self._pct_step
        else:
            self._pct_low -= self._pct_step
            self._pct_high += self._pct_step
        self._pct_low = max(0.0, min(self._pct_low, 49.0))
        self._pct_high = max(51.0, min(self._pct_high, 100.0))

        cache = self.viewer.renderer.current_slice_cache
        if cache is None:
            return
        for c in range(len(self.channel_rows)):
            if c < cache.shape[0]:
                mn, mx = compute_percentile_clim(
                    cache[c], self._pct_low, self._pct_high
                )
                self.viewer.renderer.set_clim(c, mn, mx)
        self.viewer.canvas.update()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_ui(self):
        """Refresh all rows from current renderer state. Kept for callers
        that explicitly invalidate the panel (e.g. after a data swap)."""
        self._refresh_histograms()
        for c, row in enumerate(self.channel_rows):
            vmin, vmax = self.viewer.renderer.get_clim(c)
            row.set_clim(vmin, vmax)
            row.set_visible_state(self.viewer.renderer.get_channel_visible(c))
            row.set_gamma(self.viewer.renderer.get_gamma(c))

    def _refresh_histograms(self):
        self._hist_timer.stop()
        cache = self.viewer.renderer.current_slice_cache
        if cache is None:
            return
        for c, row in enumerate(self.channel_rows):
            if c < cache.shape[0]:
                row.set_data(cache[c], self._swatch_color(c))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._display_unsubscribe is not None:
            self._display_unsubscribe()
            self._display_unsubscribe = None
        if hasattr(self.viewer, "view_changed"):
            try:
                self.viewer.view_changed.disconnect(self._on_view_changed)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)


def show_channel_dock(
    window, panel_factory=None, title="Channels & Contrast"
):
    """Show (creating on first use) a channel panel docked in *window*.

    *window* must be a ``QMainWindow``. The panel is stored at
    ``window.channel_panel`` and its dock at ``window._channel_dock``, so
    existing ``window.channel_panel.refresh_ui()`` call sites keep
    working. The dock is closable, movable, and floatable — closing hides
    it; the same menu action re-shows it.

    *panel_factory* builds the panel widget (default:
    ``ChannelPanel(window)``).
    """
    if getattr(window, "channel_panel", None) is None:
        panel = (
            panel_factory(window) if panel_factory else ChannelPanel(window)
        )
        dock = QDockWidget(title, window)
        dock.setObjectName("channel_dock")
        dock.setWidget(panel)
        window.addDockWidget(Qt.RightDockWidgetArea, dock)
        window.channel_panel = panel
        window._channel_dock = dock
    window._channel_dock.show()
    window._channel_dock.raise_()
    window.channel_panel.refresh_ui()
