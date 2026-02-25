import json

import numpy as np

from pyvistra.ui.annotation_manager import AnnotationManager
from pyvistra.data.points import PointTable


def _mgr_stub():
    # Parser/save helpers do not rely on QWidget state.
    return AnnotationManager.__new__(AnnotationManager)


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
    mgr = _mgr_stub()
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

    mgr._save_points_to_csv(str(csv_path), points)
    mgr._save_points_to_json(str(json_path), points)

    csv_loaded = mgr._load_points_from_csv(str(csv_path))
    json_loaded = mgr._load_points_from_json(str(json_path))

    assert csv_loaded.n_rows == points.n_rows
    assert csv_loaded.point_id.tolist() == [1, 2]
    assert np.allclose(csv_loaded.properties["amplitude"], [111.0, 222.0])
    assert csv_loaded.properties["label"].tolist() == ["a", "b"]

    assert json_loaded.n_rows == points.n_rows
    assert json_loaded.point_id.tolist() == [1, 2]
    assert np.allclose(json_loaded.properties["amplitude"], [111.0, 222.0])
    assert json_loaded.properties["label"].tolist() == ["a", "b"]


def test_point_json_dict_of_columns(tmp_path):
    mgr = _mgr_stub()
    data = {
        "x": [1.0, 2.0],
        "y": [3.0, 4.0],
        "t": [0, 1],
        "point_id": [9, 10],
        "amplitude": [10.5, 20.5],
    }

    path = tmp_path / "points_columns.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    points = mgr._load_points_from_json(str(path))

    assert points.point_id.tolist() == [9, 10]
    assert points.t.tolist() == [0, 1]
    assert np.allclose(points.properties["amplitude"], [10.5, 20.5])
