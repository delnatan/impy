"""Tests for data/shapes.py (ShapeData, geometry helpers, commands)."""

import numpy as np
import pytest

from pyvistra.data.shapes import (
    CIRCLE,
    LINE,
    RECTANGLE,
    AddShape,
    AdjustHandle,
    MoveShape,
    RemoveShape,
    ShapeData,
    get_handles,
    get_outline,
)
from pyvistra.layers.commands import UndoStack


# ---------------------------------------------------------------------------
# ShapeData basics
# ---------------------------------------------------------------------------

def test_shape_data_add_and_get():
    sd = ShapeData()
    sid = sd.add(RECTANGLE, [10, 20, 50, 60])
    assert sid == 0
    assert len(sd) == 1
    rec = sd.get(sid)
    assert rec.shape_type == RECTANGLE
    np.testing.assert_allclose(rec.params[:4], [10, 20, 50, 60])


def test_shape_data_remove():
    sd = ShapeData()
    sid = sd.add(CIRCLE, [100, 100, 120, 100])
    sd.remove(sid)
    assert len(sd) == 0
    assert sid not in sd


def test_shape_data_update_params():
    sd = ShapeData()
    sid = sd.add(LINE, [0, 0, 10, 10])
    sd.update(sid, [5, 5, 15, 15])
    rec = sd.get(sid)
    np.testing.assert_allclose(rec.params[:4], [5, 5, 15, 15])


def test_shape_data_invalid_type():
    sd = ShapeData()
    with pytest.raises(ValueError, match="Invalid shape type"):
        sd.add("hexagon", [0, 0, 10, 10])


def test_shape_data_time_slice():
    sd = ShapeData()
    sd.add(RECTANGLE, [0, 0, 10, 10], t=0)
    sd.add(RECTANGLE, [0, 0, 10, 10], t=1)
    sd.add(LINE, [0, 0, 10, 10], t=0)
    assert len(sd.get_time_slice(0)) == 2
    assert len(sd.get_time_slice(1)) == 1
    assert len(sd.get_time_slice(2)) == 0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def test_get_handles_rectangle():
    sd = ShapeData()
    sid = sd.add(RECTANGLE, [10, 20, 50, 60])
    handles = get_handles(sd.get(sid))
    assert set(handles.keys()) == {"tl", "tr", "bl", "br"}
    assert handles["tl"] == (10.0, 20.0)
    assert handles["br"] == (50.0, 60.0)


def test_get_handles_circle():
    sd = ShapeData()
    sid = sd.add(CIRCLE, [50, 50, 70, 50])
    handles = get_handles(sd.get(sid))
    assert set(handles.keys()) == {"center", "edge"}
    assert handles["center"] == (50.0, 50.0)
    assert handles["edge"] == (70.0, 50.0)


def test_get_handles_line():
    sd = ShapeData()
    sid = sd.add(LINE, [0, 0, 100, 100])
    handles = get_handles(sd.get(sid))
    assert set(handles.keys()) == {"p1", "p2"}


def test_get_outline_rectangle():
    sd = ShapeData()
    sid = sd.add(RECTANGLE, [10, 20, 50, 60])
    outline = get_outline(sd.get(sid))
    assert outline.shape == (5, 2)  # closed box
    assert outline[0, 0] == outline[4, 0]  # first == last


def test_get_outline_circle():
    sd = ShapeData()
    sid = sd.add(CIRCLE, [50, 50, 70, 50])
    outline = get_outline(sd.get(sid))
    assert outline.shape[0] == 65  # 64 + 1 closing point
    # Check radius is ~20
    center = np.array([50, 50])
    radii = np.linalg.norm(outline[:-1] - center, axis=1)
    np.testing.assert_allclose(radii, 20.0, atol=0.5)


def test_get_outline_line():
    sd = ShapeData()
    sid = sd.add(LINE, [0, 0, 100, 100])
    outline = get_outline(sd.get(sid))
    assert outline.shape == (2, 2)


# ---------------------------------------------------------------------------
# Hit testing
# ---------------------------------------------------------------------------

def test_hit_test_rectangle():
    sd = ShapeData()
    sd.add(RECTANGLE, [10, 10, 50, 50])
    assert sd.hit_test((30, 30)) == 0  # inside
    assert sd.hit_test((0, 0)) is None  # outside


def test_hit_test_circle():
    sd = ShapeData()
    sd.add(CIRCLE, [50, 50, 70, 50])
    assert sd.hit_test((50, 50)) == 0  # center
    assert sd.hit_test((60, 50)) == 0  # inside
    assert sd.hit_test((100, 100)) is None  # outside


def test_hit_test_line():
    sd = ShapeData()
    sd.add(LINE, [0, 0, 100, 0])
    assert sd.hit_test((50, 0)) == 0  # on line
    assert sd.hit_test((50, 20)) is None  # too far


def test_hit_test_topmost():
    sd = ShapeData()
    sd.add(RECTANGLE, [0, 0, 100, 100])
    sd.add(RECTANGLE, [20, 20, 80, 80])
    # Should hit the second (top-most) shape
    assert sd.hit_test((50, 50)) == 1


def test_hit_test_handle():
    sd = ShapeData()
    sid = sd.add(RECTANGLE, [10, 10, 50, 50])
    assert sd.hit_test_handle((10, 10), sid) == "tl"
    assert sd.hit_test_handle((50, 50), sid) == "br"
    assert sd.hit_test_handle((30, 30), sid) is None


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_serialization_roundtrip():
    sd = ShapeData()
    sd.add(RECTANGLE, [10, 20, 50, 60], t=1, properties={"name": "ROI1"})
    sd.add(CIRCLE, [50, 50, 70, 50])
    sd.add(LINE, [0, 0, 100, 100])

    data = sd.to_list()
    sd2 = ShapeData.from_list(data)
    assert len(sd2) == 3
    assert sd2.get(0).properties.get("name") == "ROI1"
    np.testing.assert_allclose(sd2.get(0).params[:4], [10, 20, 50, 60])


# ---------------------------------------------------------------------------
# Commands with UndoStack
# ---------------------------------------------------------------------------

def test_add_shape_command():
    sd = ShapeData()
    stack = UndoStack()
    cmd = AddShape(RECTANGLE, [10, 20, 50, 60])
    stack.push(cmd, sd)
    assert len(sd) == 1
    assert cmd.shape_id == 0

    stack.undo(sd)
    assert len(sd) == 0

    stack.redo(sd)
    assert len(sd) == 1


def test_remove_shape_command():
    sd = ShapeData()
    sd.add(RECTANGLE, [10, 20, 50, 60])
    stack = UndoStack()
    cmd = RemoveShape(0)
    stack.push(cmd, sd)
    assert len(sd) == 0

    stack.undo(sd)
    assert len(sd) == 1
    np.testing.assert_allclose(sd.get(0).params[:4], [10, 20, 50, 60])


def test_move_shape_command():
    sd = ShapeData()
    sd.add(RECTANGLE, [10, 10, 50, 50])
    stack = UndoStack()
    cmd = MoveShape(0, dx=5, dy=10)
    stack.push(cmd, sd)
    np.testing.assert_allclose(sd.get(0).params[:4], [15, 20, 55, 60])

    stack.undo(sd)
    np.testing.assert_allclose(sd.get(0).params[:4], [10, 10, 50, 50])


def test_adjust_handle_command():
    sd = ShapeData()
    sd.add(RECTANGLE, [10, 10, 50, 50])
    stack = UndoStack()
    cmd = AdjustHandle(0, "br", 60, 60)
    stack.push(cmd, sd)
    np.testing.assert_allclose(sd.get(0).params[2:4], [60, 60])

    stack.undo(sd)
    np.testing.assert_allclose(sd.get(0).params[2:4], [50, 50])
