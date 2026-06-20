"""ShapeData — pure data model for annotation shapes (no vispy, no Qt).

Stores rect/circle/line shapes in a columnar layout.
Each shape has: shape_id, shape_type, t, z, params (N×8 float32), properties.

Param layout per shape_type:
    rectangle: [x1, y1, x2, y2, 0, 0, 0, 0]
    circle:    [cx, cy, ex, ey, 0, 0, 0, 0]   (center + edge point)
    line:      [x1, y1, x2, y2, 0, 0, 0, 0]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Shape type constants
RECTANGLE = "rectangle"
CIRCLE = "circle"
LINE = "line"
POLYLINE = "polyline"

VALID_SHAPE_TYPES = {RECTANGLE, CIRCLE, LINE, POLYLINE}

# Param slot count
N_PARAMS = 8

# Sentinel: t = -1 / z = -1 in a ShapeRecord means "render on every frame /
# every slice." See get_time_slice() and hit_test().
ALL_FRAMES = -1

# Change-notification event kinds (passed to subscribers)
EVT_ADDED = "added"
EVT_REMOVED = "removed"
EVT_EDITED = "edited"          # params/vertices mutated
EVT_LABEL = "label"
EVT_T_Z = "t_z"                 # time/z anchor changed
EVT_BULK = "bulk"               # large-scale rebuild (e.g. from_list)


@dataclass
class ShapeRecord:
    """A single shape entry.

    ``vertices`` is only used by :data:`POLYLINE` shapes — an ``(N, 2)`` float32
    array of control points. Polyline flags live in :attr:`properties`:
    ``closed`` (bool) and ``smoothness`` (float; 0 = straight segments, >0 =
    scipy ``splprep`` smoothing factor).
    """

    shape_id: int
    shape_type: str
    t: int
    z: int
    params: np.ndarray  # (8,) float32 — unused for POLYLINE
    label: str = ""
    properties: dict = field(default_factory=dict)
    vertices: np.ndarray | None = None  # (N, 2) float32; POLYLINE only


class ShapeData:
    """Columnar storage for annotation shapes.

    Designed for ~10s of shapes (not millions). Stores shapes in a list
    for simplicity and direct ID-based access.
    """

    def __init__(self) -> None:
        self._shapes: dict[int, ShapeRecord] = {}
        self._next_id: int = 0
        self._subscribers: list[Callable[[str, int], None]] = []

    # ------------------------------------------------------------------
    # Pub/sub — mirrors ImageBuffer.subscribe / ChannelDisplayList.
    # ------------------------------------------------------------------
    def subscribe(
        self, callback: Callable[[str, int], None]
    ) -> Callable[[], None]:
        """Register ``callback(event_kind, shape_id)`` for data changes.

        ``shape_id`` is -1 for bulk events. Returns an unsubscribe function.
        """
        self._subscribers.append(callback)

        def _unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def _emit(self, event_kind: str, shape_id: int = -1) -> None:
        for cb in list(self._subscribers):
            try:
                cb(event_kind, shape_id)
            except Exception:
                # Don't let one bad subscriber break the rest.
                pass

    def __len__(self) -> int:
        return len(self._shapes)

    def __contains__(self, shape_id: int) -> bool:
        return shape_id in self._shapes

    @property
    def shape_ids(self) -> list[int]:
        return list(self._shapes.keys())

    def add(
        self,
        shape_type: str,
        params: np.ndarray | tuple | list | None = None,
        *,
        t: int = 0,
        z: int = 0,
        label: str = "",
        properties: dict | None = None,
        shape_id: int | None = None,
        vertices: np.ndarray | list | None = None,
    ) -> int:
        """Add a shape. Returns the assigned shape_id.

        For POLYLINE shapes, pass ``vertices=[(x1, y1), (x2, y2), …]`` and
        leave ``params`` as ``None``. For all other types, supply ``params``
        and leave ``vertices`` as ``None``.
        """
        if shape_type not in VALID_SHAPE_TYPES:
            raise ValueError(
                f"Invalid shape type '{shape_type}', must be one of {VALID_SHAPE_TYPES}"
            )
        p = np.zeros(N_PARAMS, dtype=np.float32)
        if params is not None:
            p[: len(params)] = params
        p = _normalize_shape_params(shape_type, p)

        verts_arr: np.ndarray | None = None
        if shape_type == POLYLINE:
            if vertices is None:
                raise ValueError("POLYLINE shapes require 'vertices'")
            verts_arr = np.asarray(vertices, dtype=np.float32).reshape(-1, 2).copy()

        if shape_id is None:
            shape_id = self._next_id
            self._next_id += 1
        elif shape_id >= self._next_id:
            self._next_id = shape_id + 1

        self._shapes[shape_id] = ShapeRecord(
            shape_id=shape_id,
            shape_type=shape_type,
            t=t,
            z=z,
            params=p,
            label=label,
            properties=dict(properties) if properties else {},
            vertices=verts_arr,
        )
        self._emit(EVT_ADDED, shape_id)
        return shape_id

    def set_label(self, shape_id: int, label: str) -> None:
        """Update a shape's display label in place."""
        self._shapes[shape_id].label = label
        self._emit(EVT_LABEL, shape_id)

    def get_label(self, shape_id: int) -> str:
        """Return the shape's display label (empty string if unset)."""
        return self._shapes[shape_id].label

    def set_anchor(self, shape_id: int, *, t: int | None = None, z: int | None = None) -> None:
        """Update a shape's t/z anchor. Use ``ALL_FRAMES`` (-1) for sentinel."""
        rec = self._shapes[shape_id]
        if t is not None:
            rec.t = t
        if z is not None:
            rec.z = z
        self._emit(EVT_T_Z, shape_id)

    def remove(self, shape_id: int) -> ShapeRecord:
        """Remove and return a shape by ID."""
        rec = self._shapes.pop(shape_id)
        self._emit(EVT_REMOVED, shape_id)
        return rec

    def get(self, shape_id: int) -> ShapeRecord:
        """Get a shape by ID."""
        return self._shapes[shape_id]

    def update(self, shape_id: int, params: np.ndarray | tuple | list) -> None:
        """Update shape params in-place."""
        rec = self._shapes[shape_id]
        p = _normalize_shape_params(rec.shape_type, params)
        rec.params[:len(p)] = p
        self._emit(EVT_EDITED, shape_id)

    def get_time_slice(self, t: int, z: int | None = None) -> list[ShapeRecord]:
        """Return all shapes visible at time ``t`` (and optionally z).

        A record matches if ``rec.t == t`` or ``rec.t == ALL_FRAMES``. When
        ``z`` is given, the same rule applies to ``rec.z``.
        """
        out = []
        for s in self._shapes.values():
            if s.t != t and s.t != ALL_FRAMES:
                continue
            if z is not None and s.z != z and s.z != ALL_FRAMES:
                continue
            out.append(s)
        return out

    def hit_test(self, point: tuple[float, float], t: int = 0, z: int = 0, tolerance: float = 5.0) -> int | None:
        """Test if a point hits any shape. Returns shape_id or None.

        Checks shapes in reverse order (top-most first). Shapes anchored to
        ``ALL_FRAMES`` are eligible at every t/z.
        """
        px, py = point
        for shape_id in reversed(list(self._shapes.keys())):
            rec = self._shapes[shape_id]
            if rec.t != t and rec.t != ALL_FRAMES:
                continue
            if rec.z != z and rec.z != ALL_FRAMES:
                continue
            if _point_in_shape(px, py, rec, tolerance):
                return shape_id
        return None

    def hit_test_handle(
        self, point: tuple[float, float], shape_id: int, tolerance: float = 5.0
    ) -> str | None:
        """Test if a point hits a handle of a specific shape. Returns handle name or None."""
        rec = self._shapes[shape_id]
        px, py = point
        handles = get_handles(rec)
        for name, (hx, hy) in handles.items():
            if np.hypot(px - hx, py - hy) < tolerance:
                return name
        return None

    def set_vertices(self, shape_id: int, vertices: np.ndarray | list) -> None:
        """Replace a polyline's vertex array in place."""
        rec = self._shapes[shape_id]
        if rec.shape_type != POLYLINE:
            raise ValueError(
                f"set_vertices only valid on POLYLINE shapes, got {rec.shape_type!r}"
            )
        rec.vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 2).copy()
        self._emit(EVT_EDITED, shape_id)

    def to_list(self) -> list[dict]:
        """Serialize all shapes to a list of dicts."""
        result = []
        for rec in self._shapes.values():
            entry = {
                "shape_id": rec.shape_id,
                "shape_type": rec.shape_type,
                "t": rec.t,
                "z": rec.z,
                "params": rec.params.tolist(),
                "label": rec.label,
                "properties": rec.properties,
            }
            if rec.vertices is not None:
                entry["vertices"] = rec.vertices.tolist()
            result.append(entry)
        return result

    @classmethod
    def from_list(cls, data: list[dict]) -> "ShapeData":
        """Deserialize from a list of dicts."""
        sd = cls()
        # Suppress per-add events; emit one bulk event at the end.
        subs, sd._subscribers = sd._subscribers, []
        try:
            for d in data:
                props = dict(d.get("properties") or {})
                # Back-compat: label used to live in properties["name"].
                label = d.get("label", "")
                if not label and "name" in props:
                    label = str(props.pop("name"))
                sd.add(
                    shape_type=d["shape_type"],
                    params=d.get("params"),
                    t=d.get("t", 0),
                    z=d.get("z", 0),
                    label=label,
                    properties=props,
                    shape_id=d.get("shape_id"),
                    vertices=d.get("vertices"),
                )
        finally:
            sd._subscribers = subs
        sd._emit(EVT_BULK)
        return sd


# ---------------------------------------------------------------------------
# Geometry helpers (pure functions, no vispy)
# ---------------------------------------------------------------------------

def snap_rectangle_params(params: np.ndarray | tuple | list) -> np.ndarray:
    """Return rectangle params snapped to integer pixel coordinates."""
    p = np.asarray(params, dtype=np.float32).copy()
    n = min(4, p.size)
    if n:
        p[:n] = np.rint(p[:n])
    return p


def _normalize_shape_params(shape_type: str, params: np.ndarray | tuple | list) -> np.ndarray:
    p = np.asarray(params, dtype=np.float32).copy()
    if shape_type == RECTANGLE:
        return snap_rectangle_params(p)
    return p


def _set_record_params(rec: ShapeRecord, params: np.ndarray | tuple | list) -> None:
    p = _normalize_shape_params(rec.shape_type, params)
    rec.params[:len(p)] = p


def rectangle_bounds(
    rec: ShapeRecord, yx_shape: tuple[int, int] | tuple | None = None
) -> tuple[int, int, int, int]:
    """Return a rectangle's half-open integer bounds as ``(x0, y0, x1, y1)``.

    The bounds use the same exclusive end convention as numpy slices.
    Existing fractional rectangle payloads are rounded to the nearest
    integer coordinates before sorting/clamping.
    """
    if rec.shape_type != RECTANGLE:
        raise ValueError(
            f"rectangle_bounds requires a {RECTANGLE} shape, got {rec.shape_type!r}"
        )
    p = snap_rectangle_params(rec.params)
    x0, x1 = sorted((int(p[0]), int(p[2])))
    y0, y1 = sorted((int(p[1]), int(p[3])))
    if yx_shape is not None:
        H, W = int(yx_shape[-2]), int(yx_shape[-1])
        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0))
        y1 = max(0, min(H, y1))
    return x0, y0, x1, y1


def get_handles(rec: ShapeRecord) -> dict[str, tuple[float, float]]:
    """Return handle positions for a shape as {name: (x, y)}.

    For POLYLINE, returns one entry per control vertex with keys
    ``"v0"``, ``"v1"``, … so :func:`_apply_handle_adjustment` can route
    drags by index.
    """
    p = rec.params
    if rec.shape_type == RECTANGLE:
        x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
        l, r = min(x1, x2), max(x1, x2)
        t, b = min(y1, y2), max(y1, y2)
        return {"tl": (l, t), "tr": (r, t), "bl": (l, b), "br": (r, b)}
    elif rec.shape_type == CIRCLE:
        cx, cy, ex, ey = p[0], p[1], p[2], p[3]
        return {"center": (cx, cy), "edge": (ex, ey)}
    elif rec.shape_type == LINE:
        x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
        return {"p1": (x1, y1), "p2": (x2, y2)}
    elif rec.shape_type == POLYLINE and rec.vertices is not None:
        return {
            f"v{i}": (float(vx), float(vy))
            for i, (vx, vy) in enumerate(rec.vertices)
        }
    return {}


def polyline_is_closed(rec: ShapeRecord) -> bool:
    """Whether a polyline's path should be drawn as a closed loop."""
    return bool(rec.properties.get("closed", False))


def polyline_smoothness(rec: ShapeRecord) -> float:
    """Smoothness factor for spline fit (0 = straight segments)."""
    try:
        return float(rec.properties.get("smoothness", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _sample_polyline_path(
    verts: np.ndarray, *, closed: bool, smoothness: float,
    n_samples: int = 200,
) -> np.ndarray:
    """Return rendered path points as ``(M, 2)`` float32.

    Falls back to the raw control polyline (closed when requested) whenever
    ``smoothness`` is 0 or scipy is unavailable / the vertex count is too low
    for the requested fit.
    """
    n = len(verts)
    if n == 0:
        return np.empty((0, 2), dtype=np.float32)
    if n < 2:
        return verts.astype(np.float32, copy=True)

    if smoothness <= 0 or n < 4:
        path = verts.astype(np.float32, copy=True)
        if closed:
            path = np.vstack([path, path[0:1]])
        return path

    try:
        from scipy.interpolate import splev, splprep
    except ImportError:
        path = verts.astype(np.float32, copy=True)
        if closed:
            path = np.vstack([path, path[0:1]])
        return path

    try:
        tck, _u = splprep(
            [verts[:, 0], verts[:, 1]],
            s=float(smoothness),
            per=1 if closed else 0,
            k=min(3, n - 1),
        )
        u_eval = np.linspace(0.0, 1.0, int(n_samples))
        xs, ys = splev(u_eval, tck)
        path = np.column_stack([xs, ys]).astype(np.float32)
    except Exception:
        path = verts.astype(np.float32, copy=True)
        if closed:
            path = np.vstack([path, path[0:1]])
    return path


def get_outline(rec: ShapeRecord, n_circle_pts: int = 64) -> np.ndarray:
    """Return outline vertices as (N, 2) float32 array for rendering.

    For rectangles: 5 points (closed box).
    For circles: n_circle_pts points around the perimeter.
    For lines: 2 endpoints.
    """
    p = rec.params
    if rec.shape_type == RECTANGLE:
        x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
        l, r = min(x1, x2), max(x1, x2)
        t, b = min(y1, y2), max(y1, y2)
        return np.array([[l, t], [r, t], [r, b], [l, b], [l, t]], dtype=np.float32)
    elif rec.shape_type == CIRCLE:
        cx, cy, ex, ey = p[0], p[1], p[2], p[3]
        radius = np.hypot(ex - cx, ey - cy)
        theta = np.linspace(0, 2 * np.pi, n_circle_pts, endpoint=False)
        pts = np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])
        # Close the circle
        pts = np.vstack([pts, pts[0:1]])
        return pts.astype(np.float32)
    elif rec.shape_type == LINE:
        x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
        return np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    elif rec.shape_type == POLYLINE and rec.vertices is not None:
        return _sample_polyline_path(
            rec.vertices,
            closed=polyline_is_closed(rec),
            smoothness=polyline_smoothness(rec),
        )
    return np.empty((0, 2), dtype=np.float32)


def square_from_corner(
    fx: float, fy: float, x: float, y: float
) -> tuple[float, float]:
    """Constrain (x, y) so the rect anchored at (fx, fy) is a square.

    Uses the larger of |x-fx|, |y-fy| as the side length and preserves
    each delta's sign, so the square grows toward the pointer no matter
    the quadrant.
    """
    dx, dy = x - fx, y - fy
    s = max(abs(dx), abs(dy))
    sx = 1.0 if dx >= 0 else -1.0
    sy = 1.0 if dy >= 0 else -1.0
    return (fx + sx * s, fy + sy * s)


def integer_square_from_corner(
    fx: float, fy: float, x: float, y: float
) -> tuple[float, float]:
    """Return the opposite corner for an integer-snapped square bbox."""
    dx, dy = x - fx, y - fy
    side = int(np.rint(max(abs(dx), abs(dy))))
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    ax = int(np.rint(fx))
    ay = int(np.rint(fy))
    return float(ax + sx * side), float(ay + sy * side)


def rect_opposite_corner(
    params: np.ndarray, handle: str
) -> tuple[float, float] | None:
    """Return the anchor corner opposite a rectangle handle, or None.

    Handles are named in canonical orientation (tl/tr/bl/br) regardless
    of param order, so we derive l/r/t/b first.
    """
    l, r = float(min(params[0], params[2])), float(max(params[0], params[2]))
    t, b = float(min(params[1], params[3])), float(max(params[1], params[3]))
    return {
        "tl": (r, b),
        "tr": (l, b),
        "bl": (r, t),
        "br": (l, t),
    }.get(handle)


def crop_rect(source, rec: ShapeRecord) -> np.ndarray:
    """Cookie-cut the YX plane of a 5D source with a rectangle shape.

    Always keeps full T, Z, C; the rect's (x1, y1, x2, y2) is clamped
    to the source's YX shape, rounded to integer pixel indices,
    and applied as ``source[:, :, :, y0:y1, x0:x1]``. The result is
    realized as a numpy array — large proxies will be materialized.
    """
    if rec.shape_type != RECTANGLE:
        raise ValueError(
            f"crop_rect requires a {RECTANGLE} shape, got {rec.shape_type!r}"
        )
    x0, y0, x1, y1 = rectangle_bounds(rec, source.shape[-2:])

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"Crop region is empty after clamping: x=[{x0},{x1}), y=[{y0},{y1})"
        )

    return np.asarray(source[:, :, :, y0:y1, x0:x1])


def _point_in_shape(px: float, py: float, rec: ShapeRecord, tolerance: float) -> bool:
    """Test if point (px, py) is inside/on a shape."""
    p = rec.params
    if rec.shape_type == RECTANGLE:
        x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
        l, r = min(x1, x2), max(x1, x2)
        t, b = min(y1, y2), max(y1, y2)
        return l <= px <= r and t <= py <= b
    elif rec.shape_type == CIRCLE:
        cx, cy, ex, ey = p[0], p[1], p[2], p[3]
        radius = np.hypot(ex - cx, ey - cy)
        return np.hypot(px - cx, py - cy) <= radius
    elif rec.shape_type == LINE:
        x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
        # Distance from point to line segment
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 == 0:
            return np.hypot(px - x1, py - y1) < tolerance
        t_param = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / l2))
        proj_x = x1 + t_param * dx
        proj_y = y1 + t_param * dy
        return np.hypot(px - proj_x, py - proj_y) < tolerance
    elif rec.shape_type == POLYLINE and rec.vertices is not None:
        path = _sample_polyline_path(
            rec.vertices,
            closed=polyline_is_closed(rec),
            smoothness=polyline_smoothness(rec),
        )
        if len(path) < 2:
            return False
        # On-path hit: distance to nearest sampled segment.
        if _min_distance_to_polyline(px, py, path) < tolerance:
            return True
        # Interior hit for closed polylines.
        if polyline_is_closed(rec):
            return _point_in_polygon(px, py, path)
        return False
    return False


def _min_distance_to_polyline(px: float, py: float, path: np.ndarray) -> float:
    """Minimum distance from a point to any segment of a sampled polyline."""
    x1 = path[:-1, 0]; y1 = path[:-1, 1]
    x2 = path[1:, 0];  y2 = path[1:, 1]
    dx = x2 - x1; dy = y2 - y1
    l2 = dx * dx + dy * dy
    # Guard against zero-length segments.
    safe = np.where(l2 > 1e-12, l2, 1.0)
    t = ((px - x1) * dx + (py - y1) * dy) / safe
    t = np.clip(t, 0.0, 1.0)
    t = np.where(l2 > 1e-12, t, 0.0)
    qx = x1 + t * dx
    qy = y1 + t * dy
    return float(np.min(np.hypot(px - qx, py - qy)))


def _point_in_polygon(px: float, py: float, path: np.ndarray) -> bool:
    """Ray-cast point-in-polygon test (assumes path closes back to start)."""
    n = len(path)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(path[i, 0]), float(path[i, 1])
        xj, yj = float(path[j, 0]), float(path[j, 1])
        if ((yi > py) != (yj > py)):
            x_cross = (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
            if px < x_cross:
                inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Shape commands (for UndoStack)
# ---------------------------------------------------------------------------

class AddShape:
    """Command to add a shape."""

    def __init__(self, shape_type: str, params: Any = None, *, t: int = 0, z: int = 0,
                 label: str = "", properties: dict | None = None,
                 vertices: Any = None):
        self.shape_type = shape_type
        self.params = params
        self.t = t
        self.z = z
        self.label = label
        self.properties = properties
        self.vertices = vertices
        self._shape_id: int | None = None

    def execute(self, data: ShapeData) -> None:
        self._shape_id = data.add(
            self.shape_type, self.params, t=self.t, z=self.z,
            label=self.label, properties=self.properties,
            shape_id=self._shape_id, vertices=self.vertices,
        )

    def undo(self, data: ShapeData) -> None:
        if self._shape_id is not None:
            data.remove(self._shape_id)

    @property
    def shape_id(self) -> int | None:
        return self._shape_id


class RemoveShape:
    """Command to remove a shape."""

    def __init__(self, shape_id: int):
        self.shape_id = shape_id
        self._snapshot: ShapeRecord | None = None

    def execute(self, data: ShapeData) -> None:
        self._snapshot = data.remove(self.shape_id)

    def undo(self, data: ShapeData) -> None:
        if self._snapshot is not None:
            s = self._snapshot
            verts = None if s.vertices is None else s.vertices.copy()
            data.add(
                s.shape_type, s.params, t=s.t, z=s.z,
                label=s.label, properties=s.properties,
                shape_id=s.shape_id, vertices=verts,
            )


class SetShapeLabel:
    """Command to rename a shape's display label."""

    def __init__(self, shape_id: int, new_label: str):
        self.shape_id = shape_id
        self._new_label = str(new_label)
        self._old_label: str | None = None

    def execute(self, data: ShapeData) -> None:
        if self._old_label is None:
            self._old_label = data.get_label(self.shape_id)
        data.set_label(self.shape_id, self._new_label)

    def undo(self, data: ShapeData) -> None:
        if self._old_label is not None:
            data.set_label(self.shape_id, self._old_label)


class MoveShape:
    """Command to move a shape by delta. Translates the four-corner params
    and (for POLYLINE) every vertex."""

    def __init__(self, shape_id: int, dx: float, dy: float):
        self.shape_id = shape_id
        self.dx = dx
        self.dy = dy

    def _translate(self, data: ShapeData, sign: float) -> None:
        rec = data.get(self.shape_id)
        d = sign
        rec.params[0] += d * self.dx
        rec.params[1] += d * self.dy
        rec.params[2] += d * self.dx
        rec.params[3] += d * self.dy
        if rec.shape_type == RECTANGLE:
            _set_record_params(rec, rec.params)
        if rec.vertices is not None:
            rec.vertices[:, 0] += d * self.dx
            rec.vertices[:, 1] += d * self.dy
        data._emit(EVT_EDITED, self.shape_id)

    def execute(self, data: ShapeData) -> None:
        self._translate(data, 1.0)

    def undo(self, data: ShapeData) -> None:
        self._translate(data, -1.0)


class AdjustHandle:
    """Command to adjust a handle position."""

    def __init__(self, shape_id: int, handle_name: str, new_x: float, new_y: float):
        self.shape_id = shape_id
        self.handle_name = handle_name
        self.new_x = new_x
        self.new_y = new_y
        self._old_params: np.ndarray | None = None

    def execute(self, data: ShapeData) -> None:
        rec = data.get(self.shape_id)
        self._old_params = rec.params.copy()
        _apply_handle_adjustment(rec, self.handle_name, self.new_x, self.new_y)
        data._emit(EVT_EDITED, self.shape_id)

    def undo(self, data: ShapeData) -> None:
        if self._old_params is not None:
            rec = data.get(self.shape_id)
            _set_record_params(rec, self._old_params)
            data._emit(EVT_EDITED, self.shape_id)


class SetShapeParams:
    """Snapshot-based command for finalising interactive drag/resize.

    Captures the before/after params so a multi-frame drag becomes a
    single undoable step. Use with UndoStack.push_executed after the
    live mutation has already happened.
    """

    def __init__(self, shape_id: int, old_params: np.ndarray, new_params: np.ndarray):
        self.shape_id = shape_id
        self._old_params = np.array(old_params, dtype=np.float32).copy()
        self._new_params = np.array(new_params, dtype=np.float32).copy()

    def execute(self, data: ShapeData) -> None:
        _set_record_params(data.get(self.shape_id), self._new_params)
        data._emit(EVT_EDITED, self.shape_id)

    def undo(self, data: ShapeData) -> None:
        _set_record_params(data.get(self.shape_id), self._old_params)
        data._emit(EVT_EDITED, self.shape_id)


def _apply_handle_adjustment(rec: ShapeRecord, handle: str, nx: float, ny: float) -> None:
    """Apply a handle drag to shape params."""
    p = rec.params
    if rec.shape_type == RECTANGLE:
        x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
        l, r = min(x1, x2), max(x1, x2)
        t, b = min(y1, y2), max(y1, y2)
        if handle == "tl":
            l, t = nx, ny
        elif handle == "tr":
            r, t = nx, ny
        elif handle == "bl":
            l, b = nx, ny
        elif handle == "br":
            r, b = nx, ny
        p[0], p[1], p[2], p[3] = l, t, r, b
        p[:] = snap_rectangle_params(p)
    elif rec.shape_type == CIRCLE:
        if handle == "center":
            dx, dy = nx - p[0], ny - p[1]
            p[0] += dx
            p[1] += dy
            p[2] += dx
            p[3] += dy
        elif handle == "edge":
            p[2], p[3] = nx, ny
    elif rec.shape_type == LINE:
        if handle == "p1":
            p[0], p[1] = nx, ny
        elif handle == "p2":
            p[2], p[3] = nx, ny
    elif rec.shape_type == POLYLINE and rec.vertices is not None:
        if handle.startswith("v"):
            try:
                idx = int(handle[1:])
            except ValueError:
                return
            if 0 <= idx < len(rec.vertices):
                rec.vertices[idx, 0] = nx
                rec.vertices[idx, 1] = ny


# ---------------------------------------------------------------------------
# Polyline vertex commands
# ---------------------------------------------------------------------------

class AddVertex:
    """Insert a vertex into a POLYLINE at a given index."""

    def __init__(self, shape_id: int, index: int, x: float, y: float):
        self.shape_id = shape_id
        self.index = int(index)
        self.x = float(x)
        self.y = float(y)

    def execute(self, data: ShapeData) -> None:
        rec = data.get(self.shape_id)
        verts = rec.vertices if rec.vertices is not None else np.empty((0, 2), dtype=np.float32)
        idx = max(0, min(self.index, len(verts)))
        rec.vertices = np.insert(
            verts, idx, [[self.x, self.y]], axis=0
        ).astype(np.float32, copy=False)
        data._emit(EVT_EDITED, self.shape_id)

    def undo(self, data: ShapeData) -> None:
        rec = data.get(self.shape_id)
        if rec.vertices is None:
            return
        idx = max(0, min(self.index, len(rec.vertices) - 1))
        rec.vertices = np.delete(rec.vertices, idx, axis=0).astype(np.float32, copy=False)
        data._emit(EVT_EDITED, self.shape_id)


class RemoveVertex:
    """Remove a vertex from a POLYLINE by index."""

    def __init__(self, shape_id: int, index: int):
        self.shape_id = shape_id
        self.index = int(index)
        self._removed_xy: tuple[float, float] | None = None

    def execute(self, data: ShapeData) -> None:
        rec = data.get(self.shape_id)
        if rec.vertices is None or len(rec.vertices) <= self.index:
            return
        x, y = rec.vertices[self.index]
        self._removed_xy = (float(x), float(y))
        rec.vertices = np.delete(rec.vertices, self.index, axis=0).astype(np.float32, copy=False)
        data._emit(EVT_EDITED, self.shape_id)

    def undo(self, data: ShapeData) -> None:
        if self._removed_xy is None:
            return
        rec = data.get(self.shape_id)
        verts = rec.vertices if rec.vertices is not None else np.empty((0, 2), dtype=np.float32)
        rec.vertices = np.insert(
            verts, self.index, [list(self._removed_xy)], axis=0
        ).astype(np.float32, copy=False)
        data._emit(EVT_EDITED, self.shape_id)


class MoveVertex:
    """Set a polyline vertex to a new position (snapshot-based undo)."""

    def __init__(self, shape_id: int, index: int, new_x: float, new_y: float):
        self.shape_id = shape_id
        self.index = int(index)
        self.new_x = float(new_x)
        self.new_y = float(new_y)
        self._old_xy: tuple[float, float] | None = None

    def execute(self, data: ShapeData) -> None:
        rec = data.get(self.shape_id)
        if rec.vertices is None or self.index >= len(rec.vertices):
            return
        old = rec.vertices[self.index]
        self._old_xy = (float(old[0]), float(old[1]))
        rec.vertices[self.index, 0] = self.new_x
        rec.vertices[self.index, 1] = self.new_y
        data._emit(EVT_EDITED, self.shape_id)

    def undo(self, data: ShapeData) -> None:
        if self._old_xy is None:
            return
        rec = data.get(self.shape_id)
        if rec.vertices is None or self.index >= len(rec.vertices):
            return
        rec.vertices[self.index, 0] = self._old_xy[0]
        rec.vertices[self.index, 1] = self._old_xy[1]
        data._emit(EVT_EDITED, self.shape_id)


class SetPolylineFlags:
    """Toggle ``closed`` / ``smoothness`` on a POLYLINE."""

    def __init__(self, shape_id: int, *, closed: bool | None = None,
                 smoothness: float | None = None):
        self.shape_id = shape_id
        self._new_closed = closed
        self._new_smoothness = smoothness
        self._old_closed: bool | None = None
        self._old_smoothness: float | None = None

    def execute(self, data: ShapeData) -> None:
        rec = data.get(self.shape_id)
        if self._new_closed is not None:
            self._old_closed = bool(rec.properties.get("closed", False))
            rec.properties["closed"] = bool(self._new_closed)
        if self._new_smoothness is not None:
            self._old_smoothness = float(rec.properties.get("smoothness", 0.0))
            rec.properties["smoothness"] = float(self._new_smoothness)
        data._emit(EVT_EDITED, self.shape_id)

    def undo(self, data: ShapeData) -> None:
        rec = data.get(self.shape_id)
        if self._new_closed is not None and self._old_closed is not None:
            rec.properties["closed"] = self._old_closed
        if self._new_smoothness is not None and self._old_smoothness is not None:
            rec.properties["smoothness"] = self._old_smoothness
        data._emit(EVT_EDITED, self.shape_id)


class SetShapeVertices:
    """Snapshot-based command for finalising a polyline vertex-set replace.

    Use with ``UndoStack.push_executed`` after the live mutation has already
    happened — symmetric to :class:`SetShapeParams`.
    """

    def __init__(self, shape_id: int, old_vertices: np.ndarray, new_vertices: np.ndarray):
        self.shape_id = shape_id
        self._old = np.asarray(old_vertices, dtype=np.float32).reshape(-1, 2).copy()
        self._new = np.asarray(new_vertices, dtype=np.float32).reshape(-1, 2).copy()

    def execute(self, data: ShapeData) -> None:
        rec = data.get(self.shape_id)
        rec.vertices = self._new.copy()
        data._emit(EVT_EDITED, self.shape_id)

    def undo(self, data: ShapeData) -> None:
        rec = data.get(self.shape_id)
        rec.vertices = self._old.copy()
        data._emit(EVT_EDITED, self.shape_id)
