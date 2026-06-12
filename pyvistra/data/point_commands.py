"""Commands for PointTable (immutable data model).

Since PointTable is frozen, commands work by replacing the entire table.
The layer must hold a mutable reference (list wrapper or attribute) that
the command can swap.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .points import PointTable

# Change-notification event kinds for PointDataHolder.subscribe.
PEVT_ADDED = "added"
PEVT_REMOVED = "removed"
PEVT_MOVED = "moved"
PEVT_BULK = "bulk"


class PointDataHolder:
    """Mutable wrapper around an immutable PointTable.

    Commands operate on this holder to swap the table in/out.

    Implements the same lightweight ``subscribe(callback)`` pub/sub as
    :class:`ShapeData` so UI consumers (visuals, layer panels, dialogs) can
    refresh on any change.
    """

    def __init__(self, table: PointTable | None = None):
        if table is None:
            table = PointTable.from_arrays(x=[], y=[])
        self.table = table
        self._subscribers: list[Callable[[str, int], None]] = []

    def __len__(self):
        return self.table.n_rows

    # ------------------------------------------------------------------
    # Pub/sub
    # ------------------------------------------------------------------
    def subscribe(
        self, callback: Callable[[str, int], None]
    ) -> Callable[[], None]:
        """Register ``callback(event_kind, point_id)``. Returns an unsubscribe."""
        self._subscribers.append(callback)

        def _unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def _emit(self, event_kind: str, point_id: int = -1) -> None:
        for cb in list(self._subscribers):
            try:
                cb(event_kind, point_id)
            except Exception:
                pass

    @property
    def point_ids(self):
        return self.table.point_id


class AddPoint:
    """Add a single point to the table."""

    def __init__(self, *, x: float, y: float, t: int = 0, z: float | None = None,
                 point_id: int | None = None, properties: dict | None = None):
        self.x = x
        self.y = y
        self.t = t
        self.z = z
        self.point_id = point_id
        self.properties = properties
        self._old_table: PointTable | None = None

    def execute(self, data: PointDataHolder) -> None:
        self._old_table = data.table
        # Auto-assign point_id if not given
        pid = self.point_id
        if pid is None:
            existing = data.table.point_id
            pid = int(existing.max()) + 1 if len(existing) > 0 else 1
            self.point_id = pid

        # Build new arrays by appending
        old = data.table
        n = old.n_rows
        new_pid = np.append(old.point_id, pid)
        new_t = np.append(old.t, self.t)
        new_x = np.append(old.x, self.x)
        new_y = np.append(old.y, self.y)
        new_z = None
        if old.z is not None or self.z is not None:
            old_z = old.z if old.z is not None else np.zeros(n, dtype=np.float32)
            new_z = np.append(old_z, self.z or 0.0)
        props = {}
        for k, v in old.properties.items():
            val = self.properties.get(k) if self.properties else None
            props[k] = np.append(v, val)
        if self.properties:
            for k, val in self.properties.items():
                if k not in props:
                    arr = np.empty(n + 1, dtype=object)
                    arr[:n] = None
                    arr[n] = val
                    props[k] = arr

        data.table = PointTable.from_arrays(
            point_id=new_pid, t=new_t, x=new_x, y=new_y, z=new_z,
            properties=props,
        )
        data._emit(PEVT_ADDED, int(pid))

    def undo(self, data: PointDataHolder) -> None:
        if self._old_table is not None:
            data.table = self._old_table
            data._emit(PEVT_REMOVED, int(self.point_id or -1))


class RemovePoint:
    """Remove a point by ID."""

    def __init__(self, point_id: int):
        self.point_id = point_id
        self._old_table: PointTable | None = None

    def execute(self, data: PointDataHolder) -> None:
        self._old_table = data.table
        data.table = data.table.remove_point(self.point_id)
        data._emit(PEVT_REMOVED, int(self.point_id))

    def undo(self, data: PointDataHolder) -> None:
        if self._old_table is not None:
            data.table = self._old_table
            data._emit(PEVT_ADDED, int(self.point_id))


class MovePoint:
    """Move a point to new coordinates."""

    def __init__(self, point_id: int, new_x: float, new_y: float):
        self.point_id = point_id
        self.new_x = new_x
        self.new_y = new_y
        self._old_table: PointTable | None = None

    def execute(self, data: PointDataHolder) -> None:
        self._old_table = data.table
        data.table = data.table.update_point(self.point_id, x=self.new_x, y=self.new_y)
        data._emit(PEVT_MOVED, int(self.point_id))

    def undo(self, data: PointDataHolder) -> None:
        if self._old_table is not None:
            data.table = self._old_table
            data._emit(PEVT_MOVED, int(self.point_id))
