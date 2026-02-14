from datetime import datetime

import numpy as np
from vispy import scene

DISTANCE_TO_UM = {
    "nm": 1e-3,
    "um": 1.0,
    "mm": 1e3,
}

TIME_TO_SECONDS = {
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
}


class ScaleTimestampOverlay:
    """Draws a scale bar + timestamp indicator in a SceneView."""

    DEFAULT_CONFIG = {
        "show_scale_bar": True,
        "scale_bar_length": 10.0,
        "scale_bar_unit": "um",
        "scale_anchor": "bottom-left",
        "scale_offset_x": 0.04,
        "scale_offset_y": 0.06,
        "scale_color": (1.0, 1.0, 1.0, 0.95),
        "show_timestamp": True,
        "timestamp_mode": "elapsed",  # elapsed | absolute | both
        "frame_interval_value": 1.0,
        "frame_interval_unit": "s",
        "timestamp_anchor": "top-left",
        "timestamp_offset_x": 0.04,
        "timestamp_offset_y": 0.06,
        "timestamp_color": (1.0, 1.0, 1.0, 0.95),
    }

    def __init__(
        self,
        view,
        *,
        axis_spacing_um=1.0,
        world_units_are_um=False,
        get_time_index=None,
        get_timestamps=None,
        config=None,
    ):
        self.view = view
        self.axis_spacing_um = float(axis_spacing_um) if axis_spacing_um else 1.0
        self.world_units_are_um = bool(world_units_are_um)
        self.get_time_index = get_time_index or (lambda: 0)
        self.get_timestamps = get_timestamps or (lambda: [])

        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        self._line = scene.visuals.Line(
            pos=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            color=(1, 1, 1, 0.95),
            width=3,
            parent=self.view.scene,
        )
        self._scale_text = scene.visuals.Text(
            text="",
            color=(1, 1, 1, 0.95),
            anchor_x="center",
            anchor_y="bottom",
            font_size=11,
            parent=self.view.scene,
        )
        self._time_text = scene.visuals.Text(
            text="",
            color=(1, 1, 1, 0.95),
            anchor_x="left",
            anchor_y="top",
            font_size=11,
            parent=self.view.scene,
        )

        self._line.order = 10_000
        self._scale_text.order = 10_001
        self._time_text.order = 10_001

        self._line.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )

        self.update()

    def set_config(self, config):
        self.config.update(config)
        self.update()

    def get_config(self):
        return dict(self.config)

    def set_axis_spacing_um(self, spacing_um):
        if spacing_um and spacing_um > 0:
            self.axis_spacing_um = float(spacing_um)
        self.update()

    def update(self):
        rect = getattr(self.view.camera, "rect", None)
        if rect is None or rect.width <= 0 or rect.height <= 0:
            self._line.visible = False
            self._scale_text.visible = False
            self._time_text.visible = False
            return

        self._update_scale_bar(rect)
        self._update_timestamp(rect)

    def remove(self):
        for visual in (self._line, self._scale_text, self._time_text):
            try:
                visual.parent = None
            except Exception:
                pass

    def _update_scale_bar(self, rect):
        if not self.config.get("show_scale_bar", True):
            self._line.visible = False
            self._scale_text.visible = False
            return

        length = float(self.config.get("scale_bar_length", 0.0) or 0.0)
        unit = str(self.config.get("scale_bar_unit", "um"))
        world_len = self._distance_to_world(length, unit)
        if world_len <= 0:
            self._line.visible = False
            self._scale_text.visible = False
            return

        anchor = str(self.config.get("scale_anchor", "bottom-left"))
        ox = float(self.config.get("scale_offset_x", 0.04))
        oy = float(self.config.get("scale_offset_y", 0.06))
        x0, y0 = self._anchor_point(rect, anchor, ox, oy)

        is_right = "right" in anchor
        is_top = "top" in anchor
        x1 = x0 - world_len if is_right else x0 + world_len
        line_color = self._normalize_color(
            self.config.get("scale_color", (1.0, 1.0, 1.0, 0.95))
        )

        self._line.set_data(
            pos=np.array([[x0, y0], [x1, y0]], dtype=np.float32),
            color=line_color,
        )
        self._line.visible = True

        self._scale_text.text = f"{length:g} {unit}"
        self._scale_text.color = line_color
        # Keep Text anchors fixed; adjust offset direction instead.
        text_dy = -rect.height * 0.03 if is_top else rect.height * 0.015
        self._scale_text.pos = ((x0 + x1) * 0.5, y0 + text_dy, 0)
        self._scale_text.visible = True

    def _update_timestamp(self, rect):
        if not self.config.get("show_timestamp", True):
            self._time_text.visible = False
            return

        t_text = self._format_timestamp_text()
        if not t_text:
            self._time_text.visible = False
            return

        anchor = str(self.config.get("timestamp_anchor", "top-left"))
        ox = float(self.config.get("timestamp_offset_x", 0.04))
        oy = float(self.config.get("timestamp_offset_y", 0.06))
        tx, ty = self._anchor_point(rect, anchor, ox, oy)
        color = self._normalize_color(
            self.config.get("timestamp_color", (1.0, 1.0, 1.0, 0.95))
        )

        self._time_text.color = color
        self._time_text.text = t_text
        self._time_text.pos = (tx, ty, 0)
        self._time_text.visible = True

    def _distance_to_world(self, value, unit):
        if value <= 0:
            return 0.0

        if unit == "px":
            if self.world_units_are_um:
                return value * self.axis_spacing_um
            return value

        um_value = value * DISTANCE_TO_UM.get(unit, 1.0)
        if self.world_units_are_um:
            return um_value

        if self.axis_spacing_um <= 0:
            return 0.0
        return um_value / self.axis_spacing_um

    def _format_timestamp_text(self):
        t_idx = int(self.get_time_index() or 0)
        timestamps = self._parse_timestamps(self.get_timestamps())

        current_dt = timestamps[t_idx] if 0 <= t_idx < len(timestamps) else None

        start_dt = None
        for dt in timestamps:
            if dt is not None:
                start_dt = dt
                break

        elapsed_sec = None
        if current_dt is not None and start_dt is not None:
            elapsed_sec = (current_dt - start_dt).total_seconds()
        else:
            dt_val = float(self.config.get("frame_interval_value", 1.0) or 1.0)
            dt_unit = str(self.config.get("frame_interval_unit", "s"))
            elapsed_sec = t_idx * dt_val * TIME_TO_SECONDS.get(dt_unit, 1.0)

        mode = str(self.config.get("timestamp_mode", "elapsed"))
        abs_text = None
        if current_dt is not None:
            abs_text = current_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        elapsed_text = None
        if elapsed_sec is not None:
            out_unit = str(self.config.get("frame_interval_unit", "s"))
            denom = TIME_TO_SECONDS.get(out_unit, 1.0)
            elapsed_val = elapsed_sec / denom if denom > 0 else elapsed_sec
            elapsed_text = f"t={elapsed_val:.4g} {out_unit}"

        if mode == "absolute":
            return abs_text or f"frame={t_idx}"
        if mode == "both":
            if abs_text and elapsed_text:
                return f"{abs_text} | {elapsed_text}"
            return abs_text or elapsed_text or f"frame={t_idx}"

        # elapsed (default)
        return elapsed_text or f"frame={t_idx}"

    def _parse_timestamps(self, timestamps):
        if not isinstance(timestamps, (list, tuple)):
            return []

        out = []
        for item in timestamps:
            if item is None:
                out.append(None)
                continue
            if isinstance(item, datetime):
                out.append(item)
                continue
            if isinstance(item, np.datetime64):
                try:
                    iso = np.datetime_as_string(item, unit="us")
                    out.append(datetime.fromisoformat(iso.replace("Z", "+00:00")))
                except Exception:
                    out.append(None)
                continue
            if isinstance(item, str):
                txt = item.strip()
                dt = None
                for fmt in (
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S",
                ):
                    try:
                        dt = datetime.strptime(txt, fmt)
                        break
                    except ValueError:
                        continue
                if dt is None:
                    try:
                        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
                    except ValueError:
                        dt = None
                out.append(dt)
                continue

            out.append(None)

        return out

    def _anchor_point(self, rect, anchor, ox, oy):
        ox = min(max(float(ox), 0.0), 1.0)
        oy = min(max(float(oy), 0.0), 1.0)
        anchor = str(anchor or "top-left")

        if "right" in anchor:
            x = rect.right - rect.width * ox
        else:
            x = rect.left + rect.width * ox

        if "bottom" in anchor:
            y = rect.bottom + rect.height * oy
        else:
            y = rect.top - rect.height * oy

        return x, y

    def _normalize_color(self, color):
        if (
            isinstance(color, (list, tuple))
            and len(color) == 4
            and all(isinstance(c, (int, float)) for c in color)
        ):
            return tuple(min(max(float(c), 0.0), 1.0) for c in color)
        return (1.0, 1.0, 1.0, 0.95)
