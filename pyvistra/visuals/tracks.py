from __future__ import annotations

import colorsys

import numpy as np
from vispy import scene

from .. import colormaps as cmap
from ..data.tracks import TrackTable

DEFAULT_STYLE = {
    "trail_window": 30,
    "line_width": 2.0,
    "opacity": 0.9,
    "color_mode": "auto",
    "fixed_color": (1.0, 1.0, 1.0, 1.0),
    "color_property": None,
    "colormap_name": "viridis",
    "color_clim": None,
    "show_heads": True,
    "head_size": 8.0,
    "min_track_length": 1,
}


class TrackLayerVisual:
    """Renders time-aware polylines for tracking overlays."""

    def __init__(
        self,
        view,
        *,
        trail_window=30,
        line_width=2.0,
        opacity=0.9,
        color_mode="auto",
        fixed_color=(1.0, 1.0, 1.0, 1.0),
        color_property=None,
        colormap_name="viridis",
        color_clim=None,
        show_heads=True,
        head_size=8.0,
        min_track_length=1,
    ):
        self.view = view
        self._tracks: TrackTable | None = None
        self._visible = True
        self._trail_window = trail_window
        self._line_width = float(line_width)
        self._opacity = float(opacity)
        self._color_mode = str(color_mode)
        self._fixed_color = tuple(fixed_color)
        self._color_property = color_property
        self._colormap_name = str(colormap_name)
        self._color_clim = None if color_clim is None else tuple(color_clim)
        self._show_heads = bool(show_heads)
        self._head_size = float(head_size)
        self._min_track_length = int(min_track_length)
        self._t_idx = 0
        self._z_idx = 0

        self._line = scene.visuals.Line(
            pos=np.zeros((0, 2), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.0),
            connect="segments",
            width=self._line_width,
            parent=self.view.scene,
        )
        self._heads = scene.visuals.Markers(parent=self.view.scene)

        self._line.order = 5_000
        self._heads.order = 5_001

        self._line.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )
        self._heads.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )

    def _clear_line(self):
        # VisPy Line can fail on empty per-vertex color arrays; use a scalar RGBA.
        self._line.set_data(
            pos=np.zeros((0, 2), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.0),
            connect="segments",
            width=self._line_width,
        )

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = bool(value)
        self._line.visible = self._visible
        self._heads.visible = self._visible

    @property
    def style(self) -> dict[str, object]:
        return {
            "trail_window": self._trail_window,
            "line_width": self._line_width,
            "opacity": self._opacity,
            "color_mode": self._color_mode,
            "fixed_color": self._fixed_color,
            "color_property": self._color_property,
            "colormap_name": self._colormap_name,
            "color_clim": self._color_clim,
            "show_heads": self._show_heads,
            "head_size": self._head_size,
            "min_track_length": self._min_track_length,
        }

    def set_tracks(self, tracks: TrackTable | None):
        self._tracks = tracks
        self.refresh()

    def set_time_z(self, t_idx: int, z_idx: int = 0):
        self._t_idx = int(t_idx)
        self._z_idx = int(z_idx)
        self.refresh()

    def set_trail_window(self, trail_window: int | None):
        self._trail_window = trail_window
        self.refresh()

    def set_line_width(self, line_width: float):
        self._line_width = float(line_width)
        self.refresh()

    def set_opacity(self, opacity: float):
        self._opacity = float(np.clip(opacity, 0.0, 1.0))
        self.refresh()

    def set_style(self, **kwargs):
        if "trail_window" in kwargs:
            self._trail_window = kwargs["trail_window"]
        if "line_width" in kwargs:
            self._line_width = float(kwargs["line_width"])
        if "opacity" in kwargs:
            self._opacity = float(np.clip(kwargs["opacity"], 0.0, 1.0))
        if "color_mode" in kwargs:
            self._color_mode = str(kwargs["color_mode"])
        if "fixed_color" in kwargs:
            self._fixed_color = tuple(kwargs["fixed_color"])
        if "color_property" in kwargs:
            self._color_property = kwargs["color_property"]
        if "colormap_name" in kwargs:
            self._colormap_name = str(kwargs["colormap_name"])
        if "color_clim" in kwargs:
            clim = kwargs["color_clim"]
            self._color_clim = None if clim is None else tuple(clim)
        if "show_heads" in kwargs:
            self._show_heads = bool(kwargs["show_heads"])
        if "head_size" in kwargs:
            self._head_size = float(kwargs["head_size"])
        if "min_track_length" in kwargs:
            self._min_track_length = int(kwargs["min_track_length"])
        self.refresh()

    def remove(self):
        self._line.parent = None
        self._heads.parent = None

    def refresh(self):
        if not self._visible or self._tracks is None or self._tracks.n_rows == 0:
            self._clear_line()
            self._heads.set_data(pos=np.zeros((0, 2), dtype=np.float32), size=0)
            return

        segment_positions = []
        segment_colors = []
        head_positions = []
        head_colors = []

        track_colors = self._build_track_colors()

        for track_id, track_slice in self._tracks.iter_track_slices():
            if (track_slice.stop - track_slice.start) < self._min_track_length:
                continue

            t = self._tracks.t[track_slice]
            x = self._tracks.x[track_slice]
            y = self._tracks.y[track_slice]
            z = self._tracks.z[track_slice] if self._tracks.has_z else None

            in_time = t <= self._t_idx
            if self._trail_window is not None:
                in_time &= t >= (self._t_idx - int(self._trail_window))

            idx = np.where(in_time)[0]
            if idx.size == 0:
                continue

            if z is not None:
                # 2D canvas display: only show points close to current z slice.
                z_ok = np.abs(z - self._z_idx) <= 0.5
                idx = idx[z_ok[idx]]
                if idx.size == 0:
                    continue

            tx = x[idx]
            ty = y[idx]
            tt = t[idx]

            rgb = track_colors.get(int(track_id), (1.0, 1.0, 1.0))

            if idx.size >= 2:
                if tt[-1] > tt[0]:
                    age_norm = (tt[1:] - tt[0]) / float(tt[-1] - tt[0])
                else:
                    age_norm = np.ones(idx.size - 1, dtype=np.float32)

                for i in range(idx.size - 1):
                    p0 = np.array([tx[i], ty[i]], dtype=np.float32)
                    p1 = np.array([tx[i + 1], ty[i + 1]], dtype=np.float32)
                    alpha = (0.2 + 0.8 * float(age_norm[i])) * self._opacity
                    color = (rgb[0], rgb[1], rgb[2], alpha)
                    segment_positions.extend([p0, p1])
                    segment_colors.extend([color, color])

            if self._show_heads:
                head_positions.append(np.array([tx[-1], ty[-1]], dtype=np.float32))
                head_colors.append((rgb[0], rgb[1], rgb[2], self._opacity))

        if segment_positions:
            self._line.set_data(
                pos=np.asarray(segment_positions, dtype=np.float32),
                color=np.asarray(segment_colors, dtype=np.float32),
                connect="segments",
                width=self._line_width,
            )
        else:
            self._clear_line()

        if head_positions:
            self._heads.set_data(
                pos=np.asarray(head_positions, dtype=np.float32),
                face_color=np.asarray(head_colors, dtype=np.float32),
                edge_color=(1.0, 1.0, 1.0, min(1.0, self._opacity + 0.1)),
                edge_width=1.0,
                symbol="disc",
                size=self._head_size,
            )
        else:
            self._heads.set_data(pos=np.zeros((0, 2), dtype=np.float32), size=0)

    def _build_track_colors(self) -> dict[int, tuple[float, float, float]]:
        """Resolve one RGB color per track_id under the current color mode.

        "property" colors by the per-track mean of a properties column
        (one color per whole trajectory, not per-vertex/time-varying) —
        falls back to "auto" if the property isn't present on the table.
        """
        tracks = self._tracks

        if self._color_mode == "fixed":
            rgb = tuple(float(c) for c in self._fixed_color[:3])
            return {int(tid): rgb for tid in tracks.track_ids}

        if (
            self._color_mode == "property"
            and self._color_property is not None
            and self._color_property in tracks.properties
        ):
            prop = tracks.properties[self._color_property]
            means: dict[int, float] = {}
            for tid, sl in tracks.iter_track_slices():
                vals = np.asarray(prop[sl], dtype=np.float64)
                means[tid] = float(np.nanmean(vals)) if vals.size else 0.0

            if self._color_clim is not None:
                lo, hi = self._color_clim
            elif means:
                vals = np.array(list(means.values()), dtype=np.float64)
                lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            else:
                lo, hi = 0.0, 1.0
            span = (hi - lo) if hi > lo else 1.0

            colormap, _ = cmap.get(self._colormap_name)
            tids = list(means.keys())
            norm = np.clip(
                (np.array([means[t] for t in tids], dtype=np.float64) - lo) / span,
                0.0, 1.0,
            )
            rgba = colormap.map(norm)
            return {int(tid): tuple(rgba[i, :3]) for i, tid in enumerate(tids)}

        # "auto" (or "property" mode with an unset/missing property column):
        # deterministic hue-by-track-id, unchanged from the original default.
        return {
            int(tid): colorsys.hsv_to_rgb((int(tid) * 0.61803398875) % 1.0, 0.8, 0.95)
            for tid in tracks.track_ids
        }
