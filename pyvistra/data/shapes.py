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
from typing import Any

import numpy as np

# Shape type constants
RECTANGLE = "rectangle"
CIRCLE = "circle"
LINE = "line"

VALID_SHAPE_TYPES = {RECTANGLE, CIRCLE, LINE}

# Param slot count
N_PARAMS = 8


@dataclass
class ShapeRecord:
    """A single shape entry."""

    shape_id: int
    shape_type: str
    t: int
    z: int
    params: np.ndarray  # (8,) float32
    properties: dict = field(default_factory=dict)


class ShapeData:
    """Columnar storage for annotation shapes.

    Designed for ~10s of shapes (not millions). Stores shapes in a list
    for simplicity and direct ID-based access.
    """

    def __init__(self) -> None:
        self._shapes: dict[int, ShapeRecord] = {}
        self._next_id: int = 0

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
        params: np.ndarray | tuple | list,
        *,
        t: int = 0,
        z: int = 0,
        properties: dict | None = None,
        shape_id: int | None = None,
    ) -> int:
        """Add a shape. Returns the assigned shape_id."""
        if shape_type not in VALID_SHAPE_TYPES:
            raise ValueError(
                f"Invalid shape type '{shape_type}', must be one of {VALID_SHAPE_TYPES}"
            )
        p = np.zeros(N_PARAMS, dtype=np.float32)
        p[: len(params)] = params

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
            properties=dict(properties) if properties else {},
        )
        return shape_id

    def remove(self, shape_id: int) -> ShapeRecord:
        """Remove and return a shape by ID."""
        return self._shapes.pop(shape_id)

    def get(self, shape_id: int) -> ShapeRecord:
        """Get a shape by ID."""
        return self._shapes[shape_id]

    def update(self, shape_id: int, params: np.ndarray | tuple | list) -> None:
        """Update shape params in-place."""
        rec = self._shapes[shape_id]
        rec.params[:len(params)] = params

    def get_time_slice(self, t: int) -> list[ShapeRecord]:
        """Return all shapes at time t."""
        return [s for s in self._shapes.values() if s.t == t]

    def hit_test(self, point: tuple[float, float], t: int = 0, z: int = 0, tolerance: float = 5.0) -> int | None:
        """Test if a point hits any shape. Returns shape_id or None.

        Checks shapes in reverse order (top-most first).
        """
        px, py = point
        for shape_id in reversed(list(self._shapes.keys())):
            rec = self._shapes[shape_id]
            if rec.t != t or rec.z != z:
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

    def to_list(self) -> list[dict]:
        """Serialize all shapes to a list of dicts."""
        result = []
        for rec in self._shapes.values():
            result.append({
                "shape_id": rec.shape_id,
                "shape_type": rec.shape_type,
                "t": rec.t,
                "z": rec.z,
                "params": rec.params.tolist(),
                "properties": rec.properties,
            })
        return result

    @classmethod
    def from_list(cls, data: list[dict]) -> "ShapeData":
        """Deserialize from a list of dicts."""
        sd = cls()
        for d in data:
            sd.add(
                shape_type=d["shape_type"],
                params=d["params"],
                t=d.get("t", 0),
                z=d.get("z", 0),
                properties=d.get("properties"),
                shape_id=d.get("shape_id"),
            )
        return sd


# ---------------------------------------------------------------------------
# Geometry helpers (pure functions, no vispy)
# ---------------------------------------------------------------------------

def get_handles(rec: ShapeRecord) -> dict[str, tuple[float, float]]:
    """Return handle positions for a shape as {name: (x, y)}."""
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
    return {}


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
    return np.empty((0, 2), dtype=np.float32)


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
    return False


# ---------------------------------------------------------------------------
# Shape commands (for UndoStack)
# ---------------------------------------------------------------------------

class AddShape:
    """Command to add a shape."""

    def __init__(self, shape_type: str, params: Any, *, t: int = 0, z: int = 0,
                 properties: dict | None = None, name: str = ""):
        self.shape_type = shape_type
        self.params = params
        self.t = t
        self.z = z
        self.properties = properties
        self.name = name
        self._shape_id: int | None = None

    def execute(self, data: ShapeData) -> None:
        self._shape_id = data.add(
            self.shape_type, self.params, t=self.t, z=self.z,
            properties=self.properties, shape_id=self._shape_id,
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
            data.add(
                s.shape_type, s.params, t=s.t, z=s.z,
                properties=s.properties, shape_id=s.shape_id,
            )


class MoveShape:
    """Command to move a shape by delta."""

    def __init__(self, shape_id: int, dx: float, dy: float):
        self.shape_id = shape_id
        self.dx = dx
        self.dy = dy

    def execute(self, data: ShapeData) -> None:
        rec = data.get(self.shape_id)
        rec.params[0] += self.dx
        rec.params[1] += self.dy
        rec.params[2] += self.dx
        rec.params[3] += self.dy

    def undo(self, data: ShapeData) -> None:
        rec = data.get(self.shape_id)
        rec.params[0] -= self.dx
        rec.params[1] -= self.dy
        rec.params[2] -= self.dx
        rec.params[3] -= self.dy


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

    def undo(self, data: ShapeData) -> None:
        if self._old_params is not None:
            rec = data.get(self.shape_id)
            rec.params[:] = self._old_params


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
