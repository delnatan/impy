from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackTable:
    """Columnar track container optimized for viewer-side queries.

    Rows are sorted by (track_id, t) and indexed per-track for efficient slicing.
    """

    track_id: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.track_id)
        if not (len(self.t) == len(self.x) == len(self.y) == n):
            raise ValueError("track_id, t, x, y must have the same length")
        if self.z is not None and len(self.z) != n:
            raise ValueError("z must have the same length as track_id")

        if n == 0:
            object.__setattr__(self, "_track_starts", np.array([], dtype=np.int64))
            object.__setattr__(self, "_track_counts", np.array([], dtype=np.int64))
            object.__setattr__(self, "_track_ids", np.array([], dtype=np.int32))
            return

        if not np.all(self.track_id[:-1] <= self.track_id[1:]):
            raise ValueError("TrackTable must be sorted by track_id")

        # Within a track, time must be non-decreasing.
        same_track = self.track_id[:-1] == self.track_id[1:]
        if np.any(self.t[:-1][same_track] > self.t[1:][same_track]):
            raise ValueError("TrackTable must be sorted by (track_id, t)")

        track_ids, starts, counts = np.unique(
            self.track_id, return_index=True, return_counts=True
        )
        object.__setattr__(self, "_track_starts", starts.astype(np.int64))
        object.__setattr__(self, "_track_counts", counts.astype(np.int64))
        object.__setattr__(self, "_track_ids", track_ids.astype(np.int32))

    @classmethod
    def from_arrays(
        cls,
        *,
        track_id,
        t,
        x,
        y,
        z=None,
    ) -> "TrackTable":
        track_id_arr = np.asarray(track_id, dtype=np.int32)
        t_arr = np.asarray(t, dtype=np.int32)
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        z_arr = None if z is None else np.asarray(z, dtype=np.float32)

        if len(track_id_arr) == 0:
            return cls(track_id_arr, t_arr, x_arr, y_arr, z_arr)

        order = np.lexsort((t_arr, track_id_arr))
        return cls(
            track_id_arr[order],
            t_arr[order],
            x_arr[order],
            y_arr[order],
            None if z_arr is None else z_arr[order],
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        *,
        track_id_col="track_id",
        t_col="t",
        x_col="x",
        y_col="y",
        z_col="z",
        use_z=True,
    ) -> "TrackTable":
        if track_id_col not in df.columns:
            raise ValueError(f"Missing required column: {track_id_col}")
        for col in (t_col, x_col, y_col):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        z_data = None
        if use_z and z_col in df.columns:
            z_data = df[z_col].to_numpy()

        return cls.from_arrays(
            track_id=df[track_id_col].to_numpy(),
            t=df[t_col].to_numpy(),
            x=df[x_col].to_numpy(),
            y=df[y_col].to_numpy(),
            z=z_data,
        )

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe()") from exc

        data = {
            "track_id": self.track_id,
            "t": self.t,
            "x": self.x,
            "y": self.y,
        }
        if self.z is not None:
            data["z"] = self.z
        return pd.DataFrame(data)

    @property
    def has_z(self) -> bool:
        return self.z is not None

    @property
    def n_rows(self) -> int:
        return int(len(self.track_id))

    @property
    def n_tracks(self) -> int:
        return int(len(self._track_ids))

    @property
    def track_ids(self) -> np.ndarray:
        return self._track_ids

    def get_track(self, track_id: int) -> dict[str, np.ndarray] | None:
        idx = np.searchsorted(self._track_ids, int(track_id))
        if idx >= len(self._track_ids) or self._track_ids[idx] != int(track_id):
            return None
        start = int(self._track_starts[idx])
        end = start + int(self._track_counts[idx])
        out = {
            "track_id": self.track_id[start:end],
            "t": self.t[start:end],
            "x": self.x[start:end],
            "y": self.y[start:end],
        }
        if self.z is not None:
            out["z"] = self.z[start:end]
        return out

    def remove_track(self, track_id: int) -> "TrackTable":
        keep = self.track_id != int(track_id)
        return TrackTable.from_arrays(
            track_id=self.track_id[keep],
            t=self.t[keep],
            x=self.x[keep],
            y=self.y[keep],
            z=None if self.z is None else self.z[keep],
        )

    def iter_track_slices(self):
        for i, tid in enumerate(self._track_ids):
            start = int(self._track_starts[i])
            end = start + int(self._track_counts[i])
            yield int(tid), slice(start, end)
