from __future__ import annotations

import numpy as np
from vispy import scene

from ..data.points import PointTable


class PointLayerVisual:
    """Renders localization-style points with crosshair markers and optional circle."""

    def __init__(
        self,
        view,
        *,
        size=9.0,
        min_screen_px=3.0,
        max_screen_px=80.0,
        edge_width=1.2,
        layer_color=(1.0, 1.0, 0.0, 1.0),
        selected_color=(1.0, 0.3, 0.3, 1.0),
        show_circle=False,
        circle_radius_px=4.0,
        z_tolerance=0.5,
        label_visible=False,
        label_template="{point_id}",
        label_font_size=9,
        dense_label_threshold=2000,
    ):
        self.view = view
        self._points: PointTable | None = None
        self._visible = True
        self._selected_point_ids: set[int] = set()

        self._t_idx = 0
        self._z_idx = 0

        self._size = float(size)
        self._min_screen_px = float(min_screen_px)
        self._max_screen_px = float(max_screen_px)
        self._edge_width = float(edge_width)
        self._layer_color = tuple(layer_color)
        self._selected_color = tuple(selected_color)
        self._show_circle = bool(show_circle)
        self._circle_radius_px = float(circle_radius_px)
        self._z_tolerance = float(z_tolerance)
        self._label_visible = bool(label_visible)
        self._label_template = str(label_template)
        self._label_font_size = int(label_font_size)
        self._dense_label_threshold = int(dense_label_threshold)

        self._cross_line = scene.visuals.Line(
            pos=np.zeros((0, 2), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.0),
            connect="segments",
            width=max(1.0, self._edge_width),
            parent=self.view.scene,
        )
        self._circle_line = scene.visuals.Line(
            pos=np.zeros((0, 2), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.0),
            connect="segments",
            width=max(1.0, self._edge_width),
            parent=self.view.scene,
        )
        self._labels = scene.visuals.Text(
            text=[],
            pos=np.zeros((0, 3), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.9),
            font_size=self._label_font_size,
            anchor_x="left",
            anchor_y="bottom",
            parent=self.view.scene,
        )

        self._cross_line.order = 5_100
        self._circle_line.order = 5_101
        self._labels.order = 5_102

        for visual in (self._cross_line, self._circle_line, self._labels):
            visual.set_gl_state(
                preset="translucent",
                blend=True,
                blend_func=("src_alpha", "one_minus_src_alpha"),
                depth_test=False,
            )

        self.refresh()

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = bool(value)
        self.refresh()

    @property
    def style(self) -> dict[str, object]:
        return {
            "size": self._size,
            "min_screen_px": self._min_screen_px,
            "max_screen_px": self._max_screen_px,
            "edge_width": self._edge_width,
            "layer_color": self._layer_color,
            "selected_color": self._selected_color,
            "show_circle": self._show_circle,
            "circle_radius_px": self._circle_radius_px,
            "z_tolerance": self._z_tolerance,
            "label_visible": self._label_visible,
            "label_template": self._label_template,
            "label_font_size": self._label_font_size,
            "dense_label_threshold": self._dense_label_threshold,
        }

    def set_points(self, points: PointTable | None):
        self._points = points
        self.refresh()

    def set_time_z(self, t_idx: int, z_idx: int = 0):
        self._t_idx = int(t_idx)
        self._z_idx = int(z_idx)
        self.refresh()

    def set_selected_point_ids(self, ids: set[int]):
        self._selected_point_ids = {int(v) for v in ids}
        self.refresh()

    def set_label_fields(self, template: str):
        self._label_template = str(template)
        self.refresh()

    def set_style(self, **kwargs):
        if "size" in kwargs:
            self._size = float(kwargs["size"])
        if "min_screen_px" in kwargs:
            self._min_screen_px = float(kwargs["min_screen_px"])
        if "max_screen_px" in kwargs:
            self._max_screen_px = float(kwargs["max_screen_px"])
        if "edge_width" in kwargs:
            self._edge_width = float(kwargs["edge_width"])
        if "layer_color" in kwargs:
            self._layer_color = tuple(kwargs["layer_color"])
        if "selected_color" in kwargs:
            self._selected_color = tuple(kwargs["selected_color"])
        if "show_circle" in kwargs:
            self._show_circle = bool(kwargs["show_circle"])
        if "circle_radius_px" in kwargs:
            self._circle_radius_px = float(kwargs["circle_radius_px"])
        if "z_tolerance" in kwargs:
            self._z_tolerance = float(kwargs["z_tolerance"])
        if "label_visible" in kwargs:
            self._label_visible = bool(kwargs["label_visible"])
        if "label_template" in kwargs:
            self._label_template = str(kwargs["label_template"])
        if "label_font_size" in kwargs:
            self._label_font_size = int(kwargs["label_font_size"])
            self._labels.font_size = self._label_font_size
        if "dense_label_threshold" in kwargs:
            self._dense_label_threshold = int(kwargs["dense_label_threshold"])
        self.refresh()

    def remove(self):
        self._cross_line.parent = None
        self._circle_line.parent = None
        self._labels.parent = None

    def _clear_line(self, line):
        line.set_data(
            pos=np.zeros((0, 2), dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.0),
            connect="segments",
            width=max(1.0, self._edge_width),
        )

    def _clear(self):
        self._clear_line(self._cross_line)
        self._clear_line(self._circle_line)
        self._labels.text = []
        self._labels.pos = np.zeros((0, 3), dtype=np.float32)
        self._labels.visible = False

    def _px_per_data(self) -> float:
        cam = getattr(self.view, "camera", None)
        rect = getattr(cam, "rect", None)
        if rect is None:
            return 1.0
        w = max(float(rect.width), 1e-9)
        h = max(float(rect.height), 1e-9)
        canvas_size = getattr(self.view.canvas, "size", (1, 1))
        cw = max(float(canvas_size[0]), 1.0)
        ch = max(float(canvas_size[1]), 1.0)
        sx = cw / w
        sy = ch / h
        return 0.5 * (sx + sy)

    def _format_label(self, row: dict[str, object]) -> str:
        try:
            return self._label_template.format(**row)
        except Exception:
            return str(row.get("point_id", ""))

    def _filtered_rows(self):
        if self._points is None:
            return []
        s = self._points.get_time_slice(self._t_idx)
        if s is None:
            return []

        rows = []
        for i in range(s.start, s.stop):
            if self._points.z is not None:
                if abs(float(self._points.z[i]) - self._z_idx) > self._z_tolerance:
                    continue
            row = {
                "point_id": int(self._points.point_id[i]),
                "t": int(self._points.t[i]),
                "x": float(self._points.x[i]),
                "y": float(self._points.y[i]),
            }
            if self._points.z is not None:
                row["z"] = float(self._points.z[i])
            for key, arr in self._points.properties.items():
                value = arr[i]
                row[key] = value.item() if hasattr(value, "item") else value
            rows.append(row)
        return rows

    def refresh(self):
        if not self._visible or self._points is None or self._points.n_rows == 0:
            self._clear()
            return

        rows = self._filtered_rows()
        if not rows:
            self._clear()
            return

        pos = np.array([[r["x"], r["y"]] for r in rows], dtype=np.float32)
        selected_mask = np.array(
            [int(r["point_id"]) in self._selected_point_ids for r in rows],
            dtype=bool,
        )

        px_per_data = max(self._px_per_data(), 1e-6)

        # Crosshair arm length in data coords, clamped by screen-px limits
        arm_data = self._size
        screen_size = arm_data * px_per_data
        if screen_size < self._min_screen_px:
            arm_data = self._min_screen_px / px_per_data
        elif screen_size > self._max_screen_px:
            arm_data = self._max_screen_px / px_per_data
        half_arm = 0.5 * arm_data

        # Circle radius converted from screen pixels to data coords
        circle_r_data = self._circle_radius_px / px_per_data

        cross_segments = []
        cross_colors = []
        circle_segments = []
        circle_colors = []

        # Pre-compute circle unit vectors (12-segment polygon)
        n_segs = 12
        angles = np.linspace(0, 2 * np.pi, n_segs + 1)
        cx = np.cos(angles)
        cy = np.sin(angles)

        for i, (x, y) in enumerate(pos):
            color = self._selected_color if selected_mask[i] else self._layer_color

            # Crosshair: horizontal + vertical arms
            cross_segments.extend([
                [x - half_arm, y], [x + half_arm, y],
                [x, y - half_arm], [x, y + half_arm],
            ])
            cross_colors.extend([color] * 4)

            # Optional circle
            if self._show_circle:
                for j in range(n_segs):
                    circle_segments.extend([
                        [x + cx[j] * circle_r_data, y + cy[j] * circle_r_data],
                        [x + cx[j + 1] * circle_r_data, y + cy[j + 1] * circle_r_data],
                    ])
                    circle_colors.extend([color] * 2)

        if cross_segments:
            self._cross_line.set_data(
                pos=np.asarray(cross_segments, dtype=np.float32),
                color=np.asarray(cross_colors, dtype=np.float32),
                connect="segments",
                width=max(1.0, self._edge_width),
            )
        else:
            self._clear_line(self._cross_line)

        if self._show_circle and circle_segments:
            self._circle_line.set_data(
                pos=np.asarray(circle_segments, dtype=np.float32),
                color=np.asarray(circle_colors, dtype=np.float32),
                connect="segments",
                width=max(1.0, self._edge_width),
            )
        else:
            self._clear_line(self._circle_line)

        if self._label_visible and len(rows) <= self._dense_label_threshold:
            labels = [self._format_label(r) for r in rows]
            lpos = np.column_stack(
                [pos[:, 0] + 2.0, pos[:, 1] - 2.0, np.zeros(len(pos), dtype=np.float32)]
            )
            self._labels.text = labels
            self._labels.pos = lpos
            self._labels.visible = True
        else:
            self._labels.text = []
            self._labels.pos = np.zeros((0, 3), dtype=np.float32)
            self._labels.visible = False

        show = bool(self._visible)
        self._cross_line.visible = show
        self._circle_line.visible = show and self._show_circle
