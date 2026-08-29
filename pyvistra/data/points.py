from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PointTable:
    """Columnar point container for localization-style annotations.

    Rows are sorted by (t, point_id) for efficient current-frame filtering.

    ``pixel_index_coordinates`` declares which coordinate frame x/y are in,
    set once by whoever actually knows the answer — the code constructing
    this table — rather than guessed downstream from the data's shape or
    provenance:

    - ``False`` (default): x/y are already scene coordinates, e.g. captured
      directly from a mouse click (``ImageWindow._map_event_to_image``).
    - ``True``: x/y are array-index coordinates (integer detector output,
      *or* a sub-pixel-refined position still expressed in that same frame —
      e.g. spotfitlm's MLE Gaussian fit reports ``xc = xc_seed_int +
      offset``, so refinement doesn't change the frame, only the precision).
      vispy's ``Image`` places array index i's center at scene coordinate
      i+0.5, so this table's x/y need that same +0.5 to render/hit-test in
      the right place; see ``PointLayerVisual.pixel_offset``.
    """

    point_id: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None
    properties: dict[str, np.ndarray] = field(default_factory=dict)
    pixel_index_coordinates: bool = False

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
        pixel_index_coordinates: bool = False,
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
                pixel_index_coordinates=bool(pixel_index_coordinates),
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
            pixel_index_coordinates=bool(pixel_index_coordinates),
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
        pixel_index_coordinates: bool = False,
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
            pixel_index_coordinates=pixel_index_coordinates,
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
    def scene_offset(self) -> float:
        """Shift between this table's stored x/y and scene (display) x/y.

        0.5 when ``pixel_index_coordinates`` is set, else 0.0 — the single
        place both readers (``PointLayerVisual.pixel_offset``, hit-testing)
        and writers (mouse-driven ``AddPoint``/drag in ``ImageWindow``)
        derive the conversion from, so a click into an already-loaded
        pixel-index table is converted to this table's storage convention
        on the way in, not just compensated for on the way out.
        """
        return 0.5 if self.pixel_index_coordinates else 0.0

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
            pixel_index_coordinates=self.pixel_index_coordinates,
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
            pixel_index_coordinates=self.pixel_index_coordinates,
        )


_REQUIRED_COLUMNS = {"x", "y", "frame"}
_SPECIAL_COLUMNS = {"point_id", "frame", "x", "y", "z"}


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


def load_points_csv(path, *, pixel_index_coordinates: bool = False) -> PointTable:
    """Load a PointTable from a CSV file with required ``x, y, frame``
    columns (optional ``point_id``, ``z``). Any other column is carried
    through as a per-point feature — data available for filtering or for
    mapping onto the canvas (e.g. localizer output like ``flux``, ``sigma``,
    or a raw acquisition ``t`` in seconds, which is a feature here, not the
    frame index).

    ``pixel_index_coordinates`` must be supplied by the caller — it can't be
    inferred from the file (e.g. from whether x/y look integer-valued: a
    sub-pixel-refined position, such as from spotfitlm's MLE Gaussian fit,
    is still expressed in array-index units, not scene units — refinement
    only adds precision, it doesn't change the coordinate frame). Pass
    ``True`` for detections/localizations sourced from an array-index
    pipeline; leave ``False`` (default) for already-scene-space data.
    """
    x = []
    y = []
    point_id = []
    t = []
    z = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(missing))}"
            )

        has_point_id = "point_id" in reader.fieldnames
        has_z = "z" in reader.fieldnames
        prop_cols = [
            col for col in reader.fieldnames if col not in _SPECIAL_COLUMNS
        ]
        properties = {col: [] for col in prop_cols}

        for row in reader:
            x.append(float(row["x"]))
            y.append(float(row["y"]))
            t.append(int(float(row["frame"])))
            if has_point_id:
                point_id.append(int(float(row["point_id"])))
            if has_z:
                z_val = row.get("z", "")
                z.append(float(z_val) if z_val != "" else 0.0)
            for col in prop_cols:
                properties[col].append(_coerce_csv_scalar(row.get(col, "")))

    return PointTable.from_arrays(
        point_id=point_id if has_point_id else None,
        t=t,
        x=x,
        y=y,
        z=z if has_z else None,
        properties=properties,
        pixel_index_coordinates=pixel_index_coordinates,
    )


def load_points_json(path, *, pixel_index_coordinates: bool = False) -> PointTable:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        rows = data.get("points", data.get("rows", None))
        if rows is None:
            if _REQUIRED_COLUMNS.issubset(data.keys()):
                props = {
                    str(k): np.asarray(v)
                    for k, v in data.items()
                    if k not in _SPECIAL_COLUMNS
                }
                return PointTable.from_arrays(
                    point_id=data.get("point_id"),
                    t=data["frame"],
                    x=data["x"],
                    y=data["y"],
                    z=data.get("z"),
                    properties=props,
                    pixel_index_coordinates=pixel_index_coordinates,
                )
            raise ValueError(
                "JSON must be a list of rows or contain a 'points'/'rows' list"
            )
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError("Invalid JSON point format")

    if not rows:
        return PointTable.from_arrays(x=[], y=[], pixel_index_coordinates=pixel_index_coordinates)

    missing = _REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Missing required keys in JSON row records: {', '.join(sorted(missing))}"
        )

    prop_keys = sorted(
        {key for row in rows for key in row.keys() if key not in _SPECIAL_COLUMNS}
    )

    props = {key: [row.get(key) for row in rows] for key in prop_keys}
    point_id = (
        [int(float(r["point_id"])) for r in rows]
        if all("point_id" in r for r in rows)
        else None
    )
    t = [int(float(r["frame"])) for r in rows]
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
        pixel_index_coordinates=pixel_index_coordinates,
    )


def save_points_csv(path, points: PointTable) -> None:
    fieldnames = ["point_id", "frame", "x", "y"]
    if points.z is not None:
        fieldnames.append("z")
    fieldnames.extend(sorted(points.properties.keys()))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(points.n_rows):
            row = {
                "point_id": int(points.point_id[i]),
                "frame": int(points.t[i]),
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
            "frame": int(points.t[i]),
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


def load_points_parquet(path, *, pixel_index_coordinates: bool = False) -> PointTable:
    """Load a PointTable from a Parquet file with required ``x, y, frame``
    columns (optional ``point_id``, ``z``). Any other column is carried
    through as a per-point feature, same convention as :func:`load_points_csv`.

    Requires the optional ``pyarrow`` dependency (``pip install
    pyvistra[parquet]``).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read Parquet point files "
            "(pip install pyvistra[parquet])"
        ) from exc

    table = pq.read_table(path)
    columns = set(table.column_names)
    missing = _REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    has_point_id = "point_id" in columns
    has_z = "z" in columns
    prop_cols = [c for c in table.column_names if c not in _SPECIAL_COLUMNS]

    return PointTable.from_arrays(
        point_id=table.column("point_id").to_pylist() if has_point_id else None,
        t=table.column("frame").to_pylist(),
        x=table.column("x").to_pylist(),
        y=table.column("y").to_pylist(),
        z=table.column("z").to_pylist() if has_z else None,
        properties={col: table.column(col).to_pylist() for col in prop_cols},
        pixel_index_coordinates=pixel_index_coordinates,
    )


def save_points_parquet(path, points: PointTable) -> None:
    """Save a PointTable to a Parquet file with ``point_id, frame, x, y[,
    z]`` columns plus one column per feature.

    Requires the optional ``pyarrow`` dependency (``pip install
    pyvistra[parquet]``).
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to write Parquet point files "
            "(pip install pyvistra[parquet])"
        ) from exc

    data: dict[str, object] = {
        "point_id": points.point_id,
        "frame": points.t,
        "x": points.x,
        "y": points.y,
    }
    if points.z is not None:
        data["z"] = points.z
    data.update(points.properties)

    pq.write_table(pa.table(data), path)
