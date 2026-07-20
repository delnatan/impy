from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PointTable:
    """Columnar point container for localization-style annotations.

    Rows are sorted by (t, point_id) for efficient current-frame filtering.
    """

    point_id: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None
    properties: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.point_id)
        if not (len(self.t) == len(self.x) == len(self.y) == n):
            raise ValueError("point_id, t, x, y must have the same length")
        if self.z is not None and len(self.z) != n:
            raise ValueError("z must have the same length as point_id")

        if len(np.unique(self.point_id)) != n:
            raise ValueError("point_id values must be unique")

        for key, values in self.properties.items():
            arr = np.asarray(values)
            if len(arr) != n:
                raise ValueError(
                    f"Property '{key}' must have the same length as point_id"
                )

        if n > 0:
            if not np.all(self.t[:-1] <= self.t[1:]):
                raise ValueError("PointTable must be sorted by (t, point_id)")
            same_t = self.t[:-1] == self.t[1:]
            if np.any(self.point_id[:-1][same_t] > self.point_id[1:][same_t]):
                raise ValueError("PointTable must be sorted by (t, point_id)")

        id_to_index = {int(pid): i for i, pid in enumerate(self.point_id)}
        object.__setattr__(self, "_id_to_index", id_to_index)

        if n == 0:
            object.__setattr__(self, "_time_values", np.array([], dtype=np.int32))
            object.__setattr__(self, "_time_starts", np.array([], dtype=np.int64))
            object.__setattr__(self, "_time_counts", np.array([], dtype=np.int64))
            return

        time_values, starts, counts = np.unique(
            self.t, return_index=True, return_counts=True
        )
        object.__setattr__(self, "_time_values", time_values.astype(np.int32))
        object.__setattr__(self, "_time_starts", starts.astype(np.int64))
        object.__setattr__(self, "_time_counts", counts.astype(np.int64))

    @classmethod
    def from_arrays(
        cls,
        *,
        x,
        y,
        point_id=None,
        t=None,
        z=None,
        properties: dict[str, object] | None = None,
    ) -> "PointTable":
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        if len(x_arr) != len(y_arr):
            raise ValueError("x and y must have the same length")

        n = len(x_arr)
        if point_id is None:
            point_id_arr = np.arange(1, n + 1, dtype=np.int32)
        else:
            point_id_arr = np.asarray(point_id, dtype=np.int32)
            if len(point_id_arr) != n:
                raise ValueError("point_id must have the same length as x/y")

        if t is None:
            t_arr = np.zeros(n, dtype=np.int32)
        else:
            t_arr = np.asarray(t, dtype=np.int32)
            if len(t_arr) != n:
                raise ValueError("t must have the same length as x/y")

        z_arr = None if z is None else np.asarray(z, dtype=np.float32)
        if z_arr is not None and len(z_arr) != n:
            raise ValueError("z must have the same length as x/y")

        prop_arrays: dict[str, np.ndarray] = {}
        if properties:
            for key, values in properties.items():
                arr = np.asarray(values)
                if len(arr) != n:
                    raise ValueError(
                        f"Property '{key}' must have the same length as x/y"
                    )
                prop_arrays[key] = arr

        if n == 0:
            return cls(
                point_id=point_id_arr,
                t=t_arr,
                x=x_arr,
                y=y_arr,
                z=z_arr,
                properties=prop_arrays,
            )

        order = np.lexsort((point_id_arr, t_arr))
        sorted_props = {k: v[order] for k, v in prop_arrays.items()}

        return cls(
            point_id=point_id_arr[order],
            t=t_arr[order],
            x=x_arr[order],
            y=y_arr[order],
            z=None if z_arr is None else z_arr[order],
            properties=sorted_props,
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        *,
        x_col="x",
        y_col="y",
        point_id_col="point_id",
        t_col="t",
        z_col="z",
    ) -> "PointTable":
        for col in (x_col, y_col):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        excluded = {x_col, y_col, point_id_col, t_col, z_col}
        props = {
            str(col): df[col].to_numpy()
            for col in df.columns
            if col not in excluded
        }

        return cls.from_arrays(
            point_id=df[point_id_col].to_numpy() if point_id_col in df.columns else None,
            t=df[t_col].to_numpy() if t_col in df.columns else None,
            x=df[x_col].to_numpy(),
            y=df[y_col].to_numpy(),
            z=df[z_col].to_numpy() if z_col in df.columns else None,
            properties=props,
        )

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe()") from exc

        data: dict[str, np.ndarray] = {
            "point_id": self.point_id,
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
        return int(len(self.point_id))

    @property
    def point_ids(self) -> np.ndarray:
        return self.point_id

    def get_point(self, point_id: int) -> dict[str, object] | None:
        idx = self._id_to_index.get(int(point_id))
        if idx is None:
            return None

        out: dict[str, object] = {
            "point_id": int(self.point_id[idx]),
            "t": int(self.t[idx]),
            "x": float(self.x[idx]),
            "y": float(self.y[idx]),
        }
        if self.z is not None:
            out["z"] = float(self.z[idx])
        for key, values in self.properties.items():
            out[key] = values[idx].item() if hasattr(values[idx], "item") else values[idx]
        return out

    def get_time_slice(self, t_idx: int) -> slice | None:
        idx = np.searchsorted(self._time_values, int(t_idx))
        if idx >= len(self._time_values) or self._time_values[idx] != int(t_idx):
            return None
        start = int(self._time_starts[idx])
        end = start + int(self._time_counts[idx])
        return slice(start, end)

    def iter_time_slices(self):
        for i, t_val in enumerate(self._time_values):
            start = int(self._time_starts[i])
            end = start + int(self._time_counts[i])
            yield int(t_val), slice(start, end)

    def remove_point(self, point_id: int) -> "PointTable":
        keep = self.point_id != int(point_id)
        return PointTable.from_arrays(
            point_id=self.point_id[keep],
            t=self.t[keep],
            x=self.x[keep],
            y=self.y[keep],
            z=None if self.z is None else self.z[keep],
            properties={k: v[keep] for k, v in self.properties.items()},
        )

    def update_point(
        self,
        point_id: int,
        *,
        x=None,
        y=None,
        z=None,
        t=None,
        properties: dict[str, object] | None = None,
    ) -> "PointTable":
        idx = self._id_to_index.get(int(point_id))
        if idx is None:
            return self

        point_id_arr = self.point_id.copy()
        t_arr = self.t.copy()
        x_arr = self.x.copy()
        y_arr = self.y.copy()
        z_arr = None if self.z is None else self.z.copy()
        props = {k: v.copy() for k, v in self.properties.items()}

        if x is not None:
            x_arr[idx] = float(x)
        if y is not None:
            y_arr[idx] = float(y)
        if t is not None:
            t_arr[idx] = int(t)
        if z is not None:
            if z_arr is None:
                z_arr = np.zeros(self.n_rows, dtype=np.float32)
            z_arr[idx] = float(z)

        if properties:
            for key, value in properties.items():
                if key in props:
                    props[key][idx] = value
                else:
                    arr = np.empty(self.n_rows, dtype=object)
                    arr[:] = None
                    arr[idx] = value
                    props[key] = arr

        return PointTable.from_arrays(
            point_id=point_id_arr,
            t=t_arr,
            x=x_arr,
            y=y_arr,
            z=z_arr,
            properties=props,
        )


def _coerce_csv_scalar(value: str):
    text = value.strip()
    if text == "":
        return ""
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." not in text and "e" not in lower:
            return int(text)
        return float(text)
    except ValueError:
        return value


def load_points_csv(path) -> PointTable:
    x = []
    y = []
    point_id = []
    t = []
    z = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        missing = {"x", "y"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(missing))}"
            )

        has_point_id = "point_id" in reader.fieldnames
        has_t = "t" in reader.fieldnames
        has_z = "z" in reader.fieldnames
        prop_cols = [
            col
            for col in reader.fieldnames
            if col not in {"point_id", "t", "z", "x", "y"}
        ]
        properties = {col: [] for col in prop_cols}

        for row in reader:
            x.append(float(row["x"]))
            y.append(float(row["y"]))
            if has_point_id:
                point_id.append(int(float(row["point_id"])))
            if has_t:
                t.append(int(float(row["t"])))
            if has_z:
                z_val = row.get("z", "")
                z.append(float(z_val) if z_val != "" else 0.0)
            for col in prop_cols:
                properties[col].append(_coerce_csv_scalar(row.get(col, "")))

    return PointTable.from_arrays(
        point_id=point_id if has_point_id else None,
        t=t if has_t else None,
        x=x,
        y=y,
        z=z if has_z else None,
        properties=properties,
    )


def load_points_json(path) -> PointTable:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        rows = data.get("points", data.get("rows", None))
        if rows is None:
            if {"x", "y"}.issubset(data.keys()):
                excluded = {"point_id", "t", "z", "x", "y"}
                props = {
                    str(k): np.asarray(v)
                    for k, v in data.items()
                    if k not in excluded
                }
                return PointTable.from_arrays(
                    point_id=data.get("point_id"),
                    t=data.get("t"),
                    x=data["x"],
                    y=data["y"],
                    z=data.get("z"),
                    properties=props,
                )
            raise ValueError(
                "JSON must be a list of rows or contain a 'points'/'rows' list"
            )
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError("Invalid JSON point format")

    if not rows:
        return PointTable.from_arrays(x=[], y=[])

    if "x" not in rows[0] or "y" not in rows[0]:
        raise ValueError("Missing required keys 'x'/'y' in JSON row records")

    excluded = {"point_id", "t", "z", "x", "y"}
    prop_keys = sorted(
        {key for row in rows for key in row.keys() if key not in excluded}
    )

    props = {key: [row.get(key) for row in rows] for key in prop_keys}
    point_id = (
        [int(float(r["point_id"])) for r in rows]
        if all("point_id" in r for r in rows)
        else None
    )
    t = (
        [int(float(r["t"])) for r in rows]
        if any("t" in r for r in rows)
        else None
    )
    z = (
        [float(r.get("z", 0.0)) for r in rows]
        if any("z" in r for r in rows)
        else None
    )

    return PointTable.from_arrays(
        point_id=point_id,
        t=t,
        x=[float(r["x"]) for r in rows],
        y=[float(r["y"]) for r in rows],
        z=z,
        properties=props,
    )


def save_points_csv(path, points: PointTable) -> None:
    fieldnames = ["point_id", "t", "x", "y"]
    if points.z is not None:
        fieldnames.append("z")
    fieldnames.extend(sorted(points.properties.keys()))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(points.n_rows):
            row = {
                "point_id": int(points.point_id[i]),
                "t": int(points.t[i]),
                "x": float(points.x[i]),
                "y": float(points.y[i]),
            }
            if points.z is not None:
                row["z"] = float(points.z[i])
            for key, arr in points.properties.items():
                value = arr[i]
                row[key] = value.item() if hasattr(value, "item") else value
            writer.writerow(row)


def save_points_json(path, points: PointTable) -> None:
    rows = []
    for i in range(points.n_rows):
        row = {
            "point_id": int(points.point_id[i]),
            "t": int(points.t[i]),
            "x": float(points.x[i]),
            "y": float(points.y[i]),
        }
        if points.z is not None:
            row["z"] = float(points.z[i])
        for key, arr in points.properties.items():
            value = arr[i]
            row[key] = value.item() if hasattr(value, "item") else value
        rows.append(row)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"format": "pyvistra_points", "version": 1, "points": rows},
            f,
            indent=2,
        )
