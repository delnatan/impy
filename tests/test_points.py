import json

import numpy as np
import pytest

from pyvistra.data.points import (
    PointTable,
    load_points_csv,
    load_points_json,
    load_points_parquet,
    save_points_csv,
    save_points_json,
    save_points_parquet,
)


def test_point_table_required_only_defaults():
    points = PointTable.from_arrays(x=[1.2, 3.4], y=[5.6, 7.8])

    assert points.n_rows == 2
    assert points.point_id.tolist() == [1, 2]
    assert points.t.tolist() == [0, 0]
    assert not points.has_z


def test_point_table_optional_fields_and_properties():
    points = PointTable.from_arrays(
        point_id=[5, 1],
        t=[2, 1],
        x=[10.0, 20.0],
        y=[30.0, 40.0],
        z=[3.2, 4.1],
        properties={"amp": [100, 200], "index": [9, 8]},
    )

    # Sorted by (t, point_id)
    assert points.t.tolist() == [1, 2]
    assert points.point_id.tolist() == [1, 5]
    assert points.z is not None
    assert np.allclose(points.z, [4.1, 3.2])
    assert points.properties["amp"].tolist() == [200, 100]


def test_point_table_property_length_validation():
    try:
        PointTable.from_arrays(
            x=[1.0, 2.0],
            y=[3.0, 4.0],
            properties={"amp": [1.0]},
        )
    except ValueError as e:
        assert "Property 'amp'" in str(e)
    else:
        raise AssertionError("Expected ValueError for mismatched property length")


def test_point_table_duplicate_point_id_rejected():
    try:
        PointTable.from_arrays(
            point_id=[1, 1],
            x=[1.0, 2.0],
            y=[3.0, 4.0],
        )
    except ValueError as e:
        assert "point_id values must be unique" in str(e)
    else:
        raise AssertionError("Expected ValueError for duplicate point_id")


def test_point_table_remove_point():
    points = PointTable.from_arrays(
        point_id=[1, 2, 3],
        t=[0, 0, 1],
        x=[10.0, 20.0, 30.0],
        y=[40.0, 50.0, 60.0],
    )

    updated = points.remove_point(2)

    assert updated.point_id.tolist() == [1, 3]
    assert updated.n_rows == 2
    assert updated.get_point(2) is None


def test_point_table_update_point_coordinates_and_property():
    points = PointTable.from_arrays(
        point_id=[1, 2],
        t=[0, 0],
        x=[10.0, 20.0],
        y=[30.0, 40.0],
        properties={"amplitude": [100.0, 200.0]},
    )

    updated = points.update_point(
        2,
        x=25.5,
        y=45.5,
        properties={"amplitude": 222.0, "note": "focus"},
    )

    row = updated.get_point(2)
    assert row is not None
    assert np.isclose(row["x"], 25.5)
    assert np.isclose(row["y"], 45.5)
    assert np.isclose(row["amplitude"], 222.0)
    assert row["note"] == "focus"


def test_point_table_dataframe_roundtrip_with_properties():
    points = PointTable.from_arrays(
        point_id=[11, 12],
        t=[1, 1],
        x=[2.5, 3.5],
        y=[4.5, 5.5],
        z=[0.3, 0.7],
        properties={"amplitude": [123.0, 456.0], "index": [0, 1]},
    )

    df = points.to_dataframe()
    restored = PointTable.from_dataframe(df)

    assert restored.point_id.tolist() == [11, 12]
    assert restored.t.tolist() == [1, 1]
    assert np.allclose(restored.x, [2.5, 3.5])
    assert np.allclose(restored.y, [4.5, 5.5])
    assert restored.z is not None
    assert np.allclose(restored.z, [0.3, 0.7])
    assert np.allclose(restored.properties["amplitude"], [123.0, 456.0])
    assert restored.properties["index"].tolist() == [0, 1]


def test_point_csv_json_roundtrip_preserves_properties(tmp_path):
    points = PointTable.from_arrays(
        point_id=[1, 2],
        t=[0, 1],
        x=[12.3, 45.6],
        y=[78.9, 10.2],
        z=[3.0, 4.0],
        properties={"amplitude": [111.0, 222.0], "label": ["a", "b"]},
    )

    csv_path = tmp_path / "points.csv"
    json_path = tmp_path / "points.json"

    save_points_csv(str(csv_path), points)
    save_points_json(str(json_path), points)

    csv_loaded = load_points_csv(str(csv_path))
    json_loaded = load_points_json(str(json_path))

    assert csv_loaded.n_rows == points.n_rows
    assert csv_loaded.point_id.tolist() == [1, 2]
    assert np.allclose(csv_loaded.properties["amplitude"], [111.0, 222.0])
    assert csv_loaded.properties["label"].tolist() == ["a", "b"]

    assert json_loaded.n_rows == points.n_rows
    assert json_loaded.point_id.tolist() == [1, 2]
    assert np.allclose(json_loaded.properties["amplitude"], [111.0, 222.0])
    assert json_loaded.properties["label"].tolist() == ["a", "b"]


def test_point_json_dict_of_columns(tmp_path):
    data = {
        "x": [1.0, 2.0],
        "y": [3.0, 4.0],
        "frame": [0, 1],
        "point_id": [9, 10],
        "amplitude": [10.5, 20.5],
    }

    path = tmp_path / "points_columns.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    points = load_points_json(str(path))

    assert points.point_id.tolist() == [9, 10]
    assert points.t.tolist() == [0, 1]
    assert np.allclose(points.properties["amplitude"], [10.5, 20.5])


def test_point_csv_requires_frame_column(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text("x,y\n1.0,2.0\n")

    with pytest.raises(ValueError, match="frame"):
        load_points_csv(str(path))


def test_point_csv_treats_extra_t_column_as_feature(tmp_path):
    # Regression case: a localizer file (e.g. spotsolve output) has both a
    # 'frame' column (the frame index) and a 't' column (real acquisition
    # time in seconds) — 't' must not be mistaken for the frame index.
    path = tmp_path / "points.csv"
    path.write_text("frame,x,y,t\n0,1.0,2.0,0.123\n1,3.0,4.0,0.456\n")

    points = load_points_csv(str(path))

    assert points.t.tolist() == [0, 1]
    assert np.allclose(points.properties["t"], [0.123, 0.456])


def test_point_parquet_roundtrip_preserves_properties(tmp_path):
    pytest.importorskip("pyarrow")

    points = PointTable.from_arrays(
        point_id=[1, 2],
        t=[0, 1],
        x=[12.3, 45.6],
        y=[78.9, 10.2],
        z=[3.0, 4.0],
        properties={"amplitude": [111.0, 222.0], "label": ["a", "b"]},
    )

    path = tmp_path / "points.parquet"
    save_points_parquet(str(path), points)
    loaded = load_points_parquet(str(path))

    assert loaded.n_rows == points.n_rows
    assert loaded.point_id.tolist() == [1, 2]
    assert loaded.t.tolist() == [0, 1]
    assert np.allclose(loaded.z, [3.0, 4.0])
    assert np.allclose(loaded.properties["amplitude"], [111.0, 222.0])
    assert loaded.properties["label"].tolist() == ["a", "b"]


# ---------------------------------------------------------------------------
# PointDataHolder subscribe + commands
# ---------------------------------------------------------------------------

from pyvistra.data.point_commands import (
    AddPoint,
    MovePoint,
    PEVT_ADDED,
    PEVT_MOVED,
    PEVT_REMOVED,
    PointDataHolder,
    RemovePoint,
)
from pyvistra.layers.commands import UndoStack


def test_point_holder_subscribe_fires_on_add_remove_move():
    holder = PointDataHolder()
    events = []
    unsub = holder.subscribe(lambda kind, pid: events.append((kind, int(pid))))

    stack = UndoStack()
    stack.push(AddPoint(x=1.0, y=2.0, t=0), holder)
    assert events[-1][0] == PEVT_ADDED
    new_pid = events[-1][1]

    stack.push(MovePoint(new_pid, 5.0, 6.0), holder)
    assert events[-1][0] == PEVT_MOVED

    stack.push(RemovePoint(new_pid), holder)
    assert events[-1][0] == PEVT_REMOVED

    # Undo restores the point.
    stack.undo(holder)
    assert events[-1][0] == PEVT_ADDED

    unsub()
    stack.push(AddPoint(x=7.0, y=8.0, t=0), holder)
    # No further events recorded after unsubscribe.
    last_kind = events[-1][0]
    assert last_kind == PEVT_ADDED  # still the last from before unsubscribe
