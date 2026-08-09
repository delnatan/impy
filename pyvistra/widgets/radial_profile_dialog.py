"""
Radial Profile Dialog - Circle-driven intensity profiles.

Two modes, both keyed off a CIRCLE shape's center + radius:
  - Radial average: mean intensity per 1-px-wide ring, from center out to
    the circle's radius. Useful for e.g. FFT power spectra.
  - Perimeter: intensity sampled around the circle's circumference at its
    current radius, as a function of angle. Useful for spotting anisotropy
    (e.g. diffraction spots) at a fixed spatial frequency.

Reuses ``LineProfileWidget`` for plotting (it's a generic x/y multi-series
canvas); everything shape/sampling-specific lives here.
"""

import csv
from collections import OrderedDict

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .. import colors as tokens
from ..data import calibration
from ..data.calibration import window_is_frequency_space
from .line_profile import LineProfileWidget
from .window_series_mixin import WindowSeriesMixin
from ..ui.comparison import paired_window
from ..data.shapes import CIRCLE, EVT_EDITED, EVT_REMOVED, EVT_BULK

# Singleton instance
_radial_profile_dialog = None

_MODE_RADIAL = "radial"
_MODE_PERIMETER = "perimeter"


def radial_profile(image_2d, cx, cy, radius):
    """Bin pixel-to-center distance into 1-px shells and average.

    Pure numpy: takes a 2D array and returns ``(radii, profile)``, both
    length ``floor(max(1, radius)) + 1``. Restricted to the bounding box
    around the circle for efficiency rather than scanning the whole image.
    """
    H, W = image_2d.shape
    max_r = max(1.0, float(radius))
    y0 = max(0, int(np.floor(cy - max_r)))
    y1 = min(H, int(np.ceil(cy + max_r)) + 1)
    x0 = max(0, int(np.floor(cx - max_r)))
    x1 = min(W, int(np.ceil(cx + max_r)) + 1)
    if y1 <= y0 or x1 <= x0:
        return None, None

    n_bins = int(np.floor(max_r)) + 1
    sub = np.asarray(image_2d[y0:y1, x0:x1], dtype=np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - cx, yy - cy)
    r_bin = r.astype(np.int64)
    # The crop is a square bounding box, so its corners are up to
    # radius*sqrt(2) away -- outside the actual circle. Exclude them rather
    # than clipping them into the outermost bin, which would otherwise
    # contaminate the edge-ring average with background pixels that were
    # never inside the requested radius.
    in_range = r_bin < n_bins
    r_bin = r_bin[in_range]
    vals = sub[in_range]

    sums = np.bincount(r_bin, weights=vals, minlength=n_bins)
    counts = np.bincount(r_bin, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = sums / counts
    radii = np.arange(n_bins, dtype=float)
    return radii, profile


def get_radial_profile_dialog():
    """Get or create the singleton RadialProfileDialog instance."""
    global _radial_profile_dialog
    if _radial_profile_dialog is None:
        _radial_profile_dialog = RadialProfileDialog()
    return _radial_profile_dialog


def radial_profile_dialog_exists():
    """Check if the radial profile dialog has been created."""
    return _radial_profile_dialog is not None


class RadialProfileDialog(WindowSeriesMixin, QDialog):
    """Floating dialog comparing radial/perimeter intensity profiles of a
    CIRCLE shape across one or more windows. Window/series bookkeeping is
    shared with LineProfileDialog via :class:`WindowSeriesMixin`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Radial Profile")
        self.setWindowFlags(Qt.Tool)
        self.resize(760, 460)

        self.active_window = None
        self.source_window = None
        self.current_circle_data = None  # {cx, cy, radius}

        self._source_shape_layer = None
        self._source_shape_id = None
        self._source_shape_unsub = None

        # wid -> {window, channel, visible, label, color}
        self.series_config = OrderedDict()
        self._computed_series = []

        self._is_shutting_down = False

        self._setup_ui()
        self._init_window_series()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        controls = QHBoxLayout()

        controls.addWidget(QLabel("Profile:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Radial average", _MODE_RADIAL)
        self.mode_combo.addItem("Perimeter (angle)", _MODE_PERIMETER)
        self.mode_combo.currentIndexChanged.connect(lambda _i: self._refresh_profiles())
        controls.addWidget(self.mode_combo)

        controls.addSpacing(12)

        btn_add_active = QPushButton("Add Active Window")
        btn_add_active.clicked.connect(self._add_active_window_series)
        controls.addWidget(btn_add_active)

        btn_add_window = QPushButton("Add Window...")
        btn_add_window.clicked.connect(self._add_window_series_dialog)
        controls.addWidget(btn_add_window)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._remove_selected_series)
        controls.addWidget(btn_remove)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_series)
        controls.addWidget(btn_clear)

        controls.addStretch()

        btn_export = QPushButton("Export...")
        btn_export.clicked.connect(self._export_profiles)
        controls.addWidget(btn_export)

        layout.addLayout(controls)

        layout.addWidget(QLabel("Compared Series:"))
        self.series_list = QListWidget()
        self.series_list.itemChanged.connect(self._on_series_item_changed)
        self.series_list.itemSelectionChanged.connect(self._on_series_selection_changed)
        layout.addWidget(self.series_list, 0)

        channel_row = QHBoxLayout()
        self.all_channels_cb = QCheckBox("All Channels")
        self.all_channels_cb.setChecked(False)
        self.all_channels_cb.toggled.connect(self._on_all_channels_toggled)
        channel_row.addWidget(self.all_channels_cb)
        channel_row.addWidget(QLabel("Channel:"))
        self.series_channel_spin = QSpinBox()
        self.series_channel_spin.setRange(1, 1)
        self.series_channel_spin.setEnabled(False)
        self.series_channel_spin.valueChanged.connect(self._on_selected_channel_changed)
        channel_row.addWidget(self.series_channel_spin)
        channel_row.addStretch()
        layout.addLayout(channel_row)

        self.profile_widget = LineProfileWidget()
        layout.addWidget(self.profile_widget, 1)

        self.status_label = QLabel("Select a circle shape to start")
        self.status_label.setStyleSheet(f"color: {tokens.TEXT_SECONDARY}; font-size: 10px;")
        layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Window bookkeeping (mirrors LineProfileDialog)
    # ------------------------------------------------------------------
    # -- WindowSeriesMixin hooks --------------------------------------
    def _connect_extra_series_signals(self, window):
        window.view_changed.connect(self._on_view_changed)

    def _disconnect_extra_series_signals(self, window):
        window.view_changed.disconnect(self._on_view_changed)

    def _on_source_window_closed(self):
        self.current_circle_data = None

    def _has_shape_source(self) -> bool:
        return self.current_circle_data is not None

    def _reset_shape_source_data(self):
        self.current_circle_data = None

    def _empty_status_text(self) -> str:
        return "Select a circle shape to start"

    # ------------------------------------------------------------------
    # Circle shape source
    # ------------------------------------------------------------------
    def set_shape_source(self, window, layer, shape_id):
        """Use a CIRCLE shape-layer record as the profile source.

        Subscribes to the layer so the profile updates live as the circle
        is edited (center/radius handle drag).
        """
        if shape_id not in layer.data:
            return
        if not self._extract_shape_circle_data(layer, shape_id):
            return

        self.source_window = window
        self.active_window = window
        self._subscribe_to_shape_source(layer, shape_id)
        self._add_series_for_window(window)
        # If the source window is one half of an explicit comparison pair
        # (see ui/comparison.py), add the paired window as a series too
        # instead of making the user click "Add Window..." for it.
        partner = paired_window(window)
        if partner is not None:
            self._add_series_for_window(partner)
        self._refresh_profiles()

    def _extract_shape_circle_data(self, layer, shape_id):
        if shape_id not in layer.data:
            return False
        rec = layer.data.get(shape_id)
        if rec.shape_type != CIRCLE:
            return False
        p = rec.params
        cx, cy, ex, ey = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        self.current_circle_data = {
            "cx": cx, "cy": cy, "radius": float(np.hypot(ex - cx, ey - cy)),
        }
        return True

    def _subscribe_to_shape_source(self, layer, shape_id):
        self._unsubscribe_from_shape_source()
        self._source_shape_layer = layer
        self._source_shape_id = shape_id

        def _on_shape_event(event_kind, sid):
            if self._is_shutting_down:
                return
            if sid != self._source_shape_id and event_kind != EVT_BULK:
                return
            if event_kind == EVT_REMOVED:
                self._unsubscribe_from_shape_source()
                self.current_circle_data = None
                self.profile_widget.clear()
                self.status_label.setText("Select a circle shape to start")
                return
            if event_kind in (EVT_EDITED, EVT_BULK):
                if self._extract_shape_circle_data(
                    self._source_shape_layer, self._source_shape_id
                ):
                    self._refresh_profiles()

        self._source_shape_unsub = layer.data.subscribe(_on_shape_event)

    def _unsubscribe_from_shape_source(self):
        if self._source_shape_unsub is not None:
            try:
                self._source_shape_unsub()
            except Exception:
                pass
        self._source_shape_unsub = None
        self._source_shape_layer = None
        self._source_shape_id = None

    # ------------------------------------------------------------------
    # Sampling + plotting
    # ------------------------------------------------------------------
    def _refresh_profiles(self):
        if self.current_circle_data is None:
            self.profile_widget.clear()
            self._computed_series = []
            self.status_label.setText("Select a circle shape to start")
            return

        cx = self.current_circle_data["cx"]
        cy = self.current_circle_data["cy"]
        radius = self.current_circle_data["radius"]
        mode = self.mode_combo.currentData()

        is_frequency_space = window_is_frequency_space(self.source_window)
        src_scale_yx = calibration.window_scale_yx(self.source_window)
        has_phys = src_scale_yx is not None

        if has_phys:
            # Physical-space center/radius: the coordinate space shared
            # across windows with different pixel sizes (see the
            # per-window conversion below). Circles don't have a natural
            # anisotropic generalization (that would make them ellipses),
            # so radius uses the same averaged sx/sy approximation as the
            # rest of this dialog.
            cx_phys, cy_phys = calibration.points_px_to_phys(
                np.array([[cx, cy]]), src_scale_yx
            )[0]
            radius_phys = calibration.radius_px_to_phys(radius, src_scale_yx)
        else:
            cx_phys = cy_phys = radius_phys = None

        if mode == _MODE_RADIAL:
            x_axis_label = (
                "Spatial freq. (1/um)" if is_frequency_space else "Radius (um)"
            ) if has_phys else (
                "Spatial freq. (cycles/px)" if is_frequency_space else "Radius (px)"
            )
        else:
            x_axis_label = "Angle (degrees)"

        all_channels = self.all_channels_cb.isChecked()
        plot_series = []
        computed_series = []
        stale_wids = []
        skipped_labels = []
        color_idx = 0

        for wid, cfg in self.series_config.items():
            window = cfg.get("window")
            if window is None:
                stale_wids.append(wid)
                continue

            visible = bool(cfg.get("visible", True))
            label = f"[{window.window_id}] {window.windowTitle()}"
            cfg["label"] = label

            if has_phys:
                win_is_freq = window_is_frequency_space(window)
                if win_is_freq != is_frequency_space:
                    # Real-space distance and spatial frequency aren't the
                    # same physical quantity -- skip rather than mislabel.
                    skipped_labels.append(label)
                    continue
                win_scale_yx = calibration.window_scale_yx(window) or (1.0, 1.0)
                radial_scale_w = 0.5 * (win_scale_yx[0] + win_scale_yx[1])
                # Convert the shared physical circle into *this* window's
                # own pixel grid -- reusing the source window's raw pixel
                # center/radius would sample the wrong physical region
                # whenever pixel sizes differ between windows.
                cx_w, cy_w = calibration.points_phys_to_px(
                    np.array([[cx_phys, cy_phys]]), win_scale_yx
                )[0]
                radius_w = calibration.radius_phys_to_px(radius_phys, win_scale_yx)
            else:
                cx_w, cy_w, radius_w = cx, cy, radius
                radial_scale_w = None

            channels = range(self._num_channels(window)) if all_channels else [int(cfg.get("channel", 0))]

            for ch in channels:
                if mode == _MODE_RADIAL:
                    x_raw, profile, ch_used = self._sample_radial(window, cx_w, cy_w, radius_w, ch)
                    if x_raw is None:
                        continue
                    x_plot = x_raw * radial_scale_w if radial_scale_w is not None else x_raw
                else:
                    n_samples_w = max(64, int(np.ceil(2 * np.pi * max(radius_w, 1.0))))
                    profile, ch_used = self._sample_perimeter(window, cx_w, cy_w, radius_w, ch, n_samples_w)
                    if profile is None:
                        continue
                    x_raw = x_plot = np.degrees(
                        np.linspace(0.0, 2 * np.pi, n_samples_w, endpoint=False)
                    )

                color = self._get_window_channel_color(window, ch_used, color_idx)
                color_idx += 1
                ch_label = f"{label} Ch{ch_used + 1}"

                plot_series.append({
                    "label": ch_label, "color": color,
                    "distances": x_plot, "values": profile, "visible": visible,
                })
                computed_series.append({
                    "window_id": window.window_id, "window_title": window.windowTitle(),
                    "channel": ch_used, "color": color, "visible": visible,
                    "mode": mode, "x_raw": x_raw, "x_plot": x_plot, "values": profile,
                })

            if not all_channels and channels:
                first_ch = list(channels)[0]
                cfg["color"] = self._get_window_channel_color(window, first_ch, 0)
                cfg["channel"] = first_ch

        for wid in stale_wids:
            self.series_config.pop(wid, None)
        if stale_wids:
            self._refresh_series_list()

        self._computed_series = computed_series
        self.profile_widget.set_x_axis_label(x_axis_label)
        self.profile_widget.set_series(plot_series)

        n_visible = sum(1 for s in plot_series if s.get("visible", True))
        status = (
            f"Circle: r={radius:.1f}px center=({cx:.0f}, {cy:.0f}) | "
            f"Series: {n_visible}/{len(plot_series)}"
        )
        if skipped_labels:
            status += f" | {len(skipped_labels)} skipped (mismatched pixel space)"
        self.status_label.setText(status)

    def _sample_radial(self, window, cx, cy, radius, channel_idx):
        image_2d, channel_used = self._get_channel_slice(window, channel_idx)
        if image_2d is None:
            return None, None, channel_idx
        radii, profile = radial_profile(image_2d, cx, cy, radius)
        if radii is None:
            return None, None, channel_used
        return radii, profile, channel_used

    def _sample_perimeter(self, window, cx, cy, radius, channel_idx, n_samples):
        image_2d, channel_used = self._get_channel_slice(window, channel_idx)
        if image_2d is None:
            return None, channel_idx

        theta = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
        xs = cx + radius * np.cos(theta)
        ys = cy + radius * np.sin(theta)
        path_xy = np.column_stack([xs, ys])
        profile = calibration.sample_along_path(image_2d, path_xy)
        return profile, channel_used

    @staticmethod
    def _get_channel_slice(window, channel_idx):
        cache = window.renderer.current_slice_cache
        if cache is None:
            return None, channel_idx
        if cache.ndim == 2:
            return cache, 0
        if cache.ndim == 3:
            channel_used = int(np.clip(channel_idx, 0, max(0, cache.shape[0] - 1)))
            return cache[channel_used], channel_used
        return None, channel_idx

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_profiles(self):
        visible = [s for s in self._computed_series if s.get("visible", True)]
        if not visible:
            self.status_label.setText("No profile data to export")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Radial Profiles", "radial_profiles.csv",
            "CSV Files (*.csv);;TSV Files (*.tsv)",
        )
        if not path:
            return
        delimiter = "\t" if (path.lower().endswith(".tsv") or "TSV" in selected_filter) else ","

        mode = visible[0]["mode"]
        x_col = "radius_px" if mode == _MODE_RADIAL else "angle_deg"
        is_frequency_space = window_is_frequency_space(self.source_window)
        x_phys_col = None
        if mode == _MODE_RADIAL and not np.array_equal(
            visible[0]["x_raw"], visible[0]["x_plot"]
        ):
            x_phys_col = "spatial_freq_1_per_um" if is_frequency_space else "radius_um"

        # Windows with different pixel sizes get their own radius_w/n_samples
        # (see _refresh_profiles), so series can have different lengths --
        # a single shared x column only works when they all match.
        same_length = len({len(s["x_raw"]) for s in visible}) == 1

        try:
            with open(path, "w", newline="") as f:
                f.write(f"# Radial Profile ({mode})\n")
                f.write(f"# center_xy: ({self.current_circle_data['cx']:.2f}, {self.current_circle_data['cy']:.2f})\n")
                f.write(f"# radius_px: {self.current_circle_data['radius']:.2f}\n")
                f.write(f"# all_channels: {self.all_channels_cb.isChecked()}\n#\n")

                writer = csv.writer(f, delimiter=delimiter)

                if same_length:
                    x_raw = np.asarray(visible[0]["x_raw"], dtype=float)
                    x_plot = np.asarray(visible[0]["x_plot"], dtype=float)
                    col_labels = [
                        f"[{s['window_id']}] {s['window_title']} Ch{s['channel'] + 1}"
                        for s in visible
                    ]
                    header = [x_col]
                    if x_phys_col is not None:
                        header.append(x_phys_col)
                    header.extend(col_labels)
                    writer.writerow(header)

                    for i in range(len(x_raw)):
                        row = [f"{float(x_raw[i]):.4f}"]
                        if x_phys_col is not None:
                            row.append(f"{float(x_plot[i]):.6g}")
                        for s in visible:
                            vals = np.asarray(s["values"], dtype=float)
                            row.append(f"{float(vals[i]):.6g}")
                        writer.writerow(row)
                else:
                    # Long format: one row per (series, sample) triple.
                    header = ["series", x_col]
                    if x_phys_col is not None:
                        header.append(x_phys_col)
                    header.append("value")
                    writer.writerow(header)

                    for s in visible:
                        label = f"[{s['window_id']}] {s['window_title']} Ch{s['channel'] + 1}"
                        x_raw_s = np.asarray(s["x_raw"], dtype=float)
                        x_plot_s = np.asarray(s["x_plot"], dtype=float)
                        vals_s = np.asarray(s["values"], dtype=float)
                        for i in range(len(x_raw_s)):
                            row = [label, f"{float(x_raw_s[i]):.4f}"]
                            if x_phys_col is not None:
                                row.append(f"{float(x_plot_s[i]):.6g}")
                            row.append(f"{float(vals_s[i]):.6g}")
                            writer.writerow(row)
        except Exception as e:
            self.status_label.setText(f"Export failed: {e}")
            return

        self.status_label.setText(f"Exported {len(visible)} series to {path}")

