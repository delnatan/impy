from __future__ import annotations

import csv
from dataclasses import dataclass, field

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
    properties: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.track_id)
        if not (len(self.t) == len(self.x) == len(self.y) == n):
            raise ValueError("track_id, t, x, y must have the same length")
        if self.z is not None and len(self.z) != n:
            raise ValueError("z must have the same length as track_id")

        for key, values in self.properties.items():
            arr = np.asarray(values)
            if len(arr) != n:
                raise ValueError(
                    f"Property '{key}' must have the same length as track_id"
                )

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
        properties: dict[str, object] | None = None,
    ) -> "TrackTable":
        track_id_arr = np.asarray(track_id, dtype=np.int32)
        t_arr = np.asarray(t, dtype=np.int32)
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        z_arr = None if z is None else np.asarray(z, dtype=np.float32)

        n = len(track_id_arr)
        prop_arrays: dict[str, np.ndarray] = {}
        if properties:
            for key, values in properties.items():
                arr = np.asarray(values)
                if len(arr) != n:
                    raise ValueError(
                        f"Property '{key}' must have the same length as track_id"
                    )
                prop_arrays[key] = arr

        if n == 0:
            return cls(track_id_arr, t_arr, x_arr, y_arr, z_arr, prop_arrays)

        order = np.lexsort((t_arr, track_id_arr))
        sorted_props = {k: v[order] for k, v in prop_arrays.items()}
        return cls(
            track_id_arr[order],
            t_arr[order],
            x_arr[order],
            y_arr[order],
            None if z_arr is None else z_arr[order],
            sorted_props,
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

        excluded = {track_id_col, t_col, x_col, y_col, z_col}
        props = {
            str(col): df[col].to_numpy()
            for col in df.columns
            if col not in excluded
        }

        return cls.from_arrays(
            track_id=df[track_id_col].to_numpy(),
            t=df[t_col].to_numpy(),
            x=df[x_col].to_numpy(),
            y=df[y_col].to_numpy(),
            z=z_data,
            properties=props,
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
        data.update(self.properties)
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
        for key, values in self.properties.items():
            out[key] = values[start:end]
        return out

    def remove_track(self, track_id: int) -> "TrackTable":
        keep = self.track_id != int(track_id)
        return TrackTable.from_arrays(
            track_id=self.track_id[keep],
            t=self.t[keep],
            x=self.x[keep],
            y=self.y[keep],
            z=None if self.z is None else self.z[keep],
            properties={k: v[keep] for k, v in self.properties.items()},
        )

    def iter_track_slices(self):
        for i, tid in enumerate(self._track_ids):
            start = int(self._track_starts[i])
            end = start + int(self._track_counts[i])
            yield int(tid), slice(start, end)


def load_tracks_csv(path) -> TrackTable:
    """Load a TrackTable from a CSV file with ``track_id, t, x, y[, z]`` columns.

    Any additional columns are loaded as per-row properties.
    """
    from .points import _coerce_csv_scalar

    track_id = []
    t = []
    x = []
    y = []
    z = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        required = {"track_id", "t", "x", "y"} - set(reader.fieldnames)
        if required:
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(required))}"
            )

        has_z = "z" in reader.fieldnames
        prop_cols = [
            col
            for col in reader.fieldnames
            if col not in {"track_id", "t", "z", "x", "y"}
        ]
        properties = {col: [] for col in prop_cols}

        for row in reader:
            track_id.append(int(float(row["track_id"])))
            t.append(int(float(row["t"])))
            x.append(float(row["x"]))
            y.append(float(row["y"]))
            if has_z:
                z_val = row.get("z", "")
                z.append(float(z_val) if z_val != "" else 0.0)
            for col in prop_cols:
                properties[col].append(_coerce_csv_scalar(row.get(col, "")))

    return TrackTable.from_arrays(
        track_id=track_id,
        t=t,
        x=x,
        y=y,
        z=z if has_z else None,
        properties=properties,
    )


def save_tracks_csv(path, tracks: TrackTable) -> None:
    """Save a TrackTable to a CSV file with ``track_id, t, x, y[, z]`` columns.

    Any per-row properties are written as additional columns.
    """
    fieldnames = ["track_id", "t", "x", "y"]
    if tracks.z is not None:
        fieldnames.append("z")
    fieldnames.extend(sorted(tracks.properties.keys()))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(tracks.n_rows):
            row = {
                "track_id": int(tracks.track_id[i]),
                "t": int(tracks.t[i]),
                "x": float(tracks.x[i]),
                "y": float(tracks.y[i]),
            }
            if tracks.z is not None:
                row["z"] = float(tracks.z[i])
            for key, arr in tracks.properties.items():
                value = arr[i]
                row[key] = value.item() if hasattr(value, "item") else value
            writer.writerow(row)
