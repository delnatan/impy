"""Commands for TrackTable (immutable data model).

Same pattern as point_commands: commands swap the frozen table in a holder.
"""

from __future__ import annotations

import numpy as np

from .tracks import TrackTable


class TrackDataHolder:
    """Mutable wrapper around an immutable TrackTable."""

    def __init__(self, table: TrackTable | None = None):
        if table is None:
            table = TrackTable.from_arrays(track_id=[], t=[], x=[], y=[])
        self.table = table

    def __len__(self):
        return len(self.table.track_id)


class AddTrack:
    """Add a complete track (all points for one track_id)."""

    def __init__(self, track_id: int, t: list, x: list, y: list, z: list | None = None):
        self.track_id = track_id
        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self._old_table: TrackTable | None = None

    def execute(self, data: TrackDataHolder) -> None:
        self._old_table = data.table
        old = data.table
        n_new = len(self.t)
        new_tid = np.append(old.track_id, np.full(n_new, self.track_id, dtype=np.int32))
        new_t = np.append(old.t, np.asarray(self.t, dtype=np.int32))
        new_x = np.append(old.x, np.asarray(self.x, dtype=np.float32))
        new_y = np.append(old.y, np.asarray(self.y, dtype=np.float32))
        new_z = None
        if old.z is not None or self.z is not None:
            old_z = old.z if old.z is not None else np.zeros(len(old.track_id), dtype=np.float32)
            z_arr = np.asarray(self.z, dtype=np.float32) if self.z is not None else np.zeros(n_new, dtype=np.float32)
            new_z = np.append(old_z, z_arr)
        data.table = TrackTable.from_arrays(
            track_id=new_tid, t=new_t, x=new_x, y=new_y, z=new_z,
        )

    def undo(self, data: TrackDataHolder) -> None:
        if self._old_table is not None:
            data.table = self._old_table


class RemoveTrack:
    """Remove all points of a track by track_id."""

    def __init__(self, track_id: int):
        self.track_id = track_id
        self._old_table: TrackTable | None = None

    def execute(self, data: TrackDataHolder) -> None:
        self._old_table = data.table
        old = data.table
        keep = old.track_id != self.track_id
        data.table = TrackTable.from_arrays(
            track_id=old.track_id[keep],
            t=old.t[keep],
            x=old.x[keep],
            y=old.y[keep],
            z=None if old.z is None else old.z[keep],
        )

    def undo(self, data: TrackDataHolder) -> None:
        if self._old_table is not None:
            data.table = self._old_table
