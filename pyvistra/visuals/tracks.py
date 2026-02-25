from __future__ import annotations

import colorsys

import numpy as np
from vispy import scene

from ..data.tracks import TrackTable


class TrackLayerVisual:
    """Renders time-aware polylines for tracking overlays."""

    def __init__(self, view, *, trail_window=30, line_width=2.0, opacity=0.9):
        self.view = view
        self._tracks: TrackTable | None = None
        self._visible = True
        self._trail_window = trail_window
        self._line_width = float(line_width)
        self._opacity = float(opacity)
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

    def set_opacity(self, opacity: float):
        self._opacity = float(np.clip(opacity, 0.0, 1.0))
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

        for track_id, track_slice in self._tracks.iter_track_slices():
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

            if idx.size >= 2:
                if tt[-1] > tt[0]:
                    age_norm = (tt[1:] - tt[0]) / float(tt[-1] - tt[0])
                else:
                    age_norm = np.ones(idx.size - 1, dtype=np.float32)

                rgb = self._color_for_track(track_id)
                for i in range(idx.size - 1):
                    p0 = np.array([tx[i], ty[i]], dtype=np.float32)
                    p1 = np.array([tx[i + 1], ty[i + 1]], dtype=np.float32)
                    alpha = (0.2 + 0.8 * float(age_norm[i])) * self._opacity
                    color = (rgb[0], rgb[1], rgb[2], alpha)
                    segment_positions.extend([p0, p1])
                    segment_colors.extend([color, color])

            head_positions.append(np.array([tx[-1], ty[-1]], dtype=np.float32))
            hr, hg, hb = self._color_for_track(track_id)
            head_colors.append((hr, hg, hb, self._opacity))

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
                size=8,
            )
        else:
            self._heads.set_data(pos=np.zeros((0, 2), dtype=np.float32), size=0)

    def _color_for_track(self, track_id: int):
        hue = (int(track_id) * 0.61803398875) % 1.0
        return colorsys.hsv_to_rgb(hue, 0.8, 0.95)
