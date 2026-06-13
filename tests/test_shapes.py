"""Tests for data/shapes.py (ShapeData, geometry helpers, commands)."""

import numpy as np
import pytest

from pyvistra.data.shapes import (
    ALL_FRAMES,
    CIRCLE,
    EVT_ADDED,
    EVT_EDITED,
    EVT_LABEL,
    EVT_REMOVED,
    EVT_T_Z,
    LINE,
    RECTANGLE,
    AddShape,
    AdjustHandle,
    MoveShape,
    RemoveShape,
    SetShapeLabel,
    SetShapeParams,
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


def test_rectangle_params_snap_to_integer_pixels():
    sd = ShapeData()
    sid = sd.add(RECTANGLE, [10.2, 20.6, 50.7, 60.1])
    np.testing.assert_allclose(sd.get(sid).params[:4], [10, 21, 51, 60])

    sd.update(sid, [1.4, 2.5, 9.6, 10.2])
    np.testing.assert_allclose(sd.get(sid).params[:4], [1, 2, 10, 10])


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
    sd.add(RECTANGLE, [10, 20, 50, 60], t=1, label="ROI1")
    sd.add(CIRCLE, [50, 50, 70, 50])
    sd.add(LINE, [0, 0, 100, 100])

    data = sd.to_list()
    sd2 = ShapeData.from_list(data)
    assert len(sd2) == 3
    assert sd2.get(0).label == "ROI1"
    np.testing.assert_allclose(sd2.get(0).params[:4], [10, 20, 50, 60])


def test_legacy_properties_name_migrates_to_label():
    """Old payloads stored the name in properties['name']; from_list lifts it."""
    legacy = [
        {
            "shape_id": 0,
            "shape_type": RECTANGLE,
            "t": 0,
            "z": 0,
            "params": [0, 0, 10, 10, 0, 0, 0, 0],
            "properties": {"name": "legacy_roi"},
        }
    ]
    sd = ShapeData.from_list(legacy)
    assert sd.get(0).label == "legacy_roi"
    assert "name" not in sd.get(0).properties


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


def test_rectangle_commands_snap_to_integer_pixels():
    sd = ShapeData()
    sid = sd.add(RECTANGLE, [0, 0, 10, 10])

    AdjustHandle(sid, "br", 20.6, 30.2).execute(sd)
    np.testing.assert_allclose(sd.get(sid).params[:4], [0, 0, 21, 30])

    SetShapeParams(
        sid,
        sd.get(sid).params.copy(),
        np.array([2.2, 3.8, 12.6, 13.1, 0, 0, 0, 0], dtype=np.float32),
    ).execute(sd)
    np.testing.assert_allclose(sd.get(sid).params[:4], [2, 4, 13, 13])


# ---------------------------------------------------------------------------
# Subscribe API + sentinel
# ---------------------------------------------------------------------------

def test_subscribe_fires_on_add_edit_remove():
    sd = ShapeData()
    events = []
    unsub = sd.subscribe(lambda kind, sid: events.append((kind, sid)))

    sid = sd.add(RECTANGLE, [0, 0, 10, 10])
    sd.set_label(sid, "hello")
    sd.update(sid, [1, 1, 11, 11])
    sd.set_anchor(sid, t=3, z=5)
    sd.remove(sid)

    kinds = [k for k, _ in events]
    assert kinds == [EVT_ADDED, EVT_LABEL, EVT_EDITED, EVT_T_Z, EVT_REMOVED]
    assert all(s == sid for _, s in events)

    unsub()
    sd.add(RECTANGLE, [0, 0, 1, 1])
    assert len(events) == 5  # no further events after unsubscribe


def test_subscribe_fires_for_commands():
    sd = ShapeData()
    sid = sd.add(RECTANGLE, [0, 0, 10, 10])

    events = []
    sd.subscribe(lambda kind, s: events.append(kind))

    from pyvistra.layers.commands import UndoStack
    stack = UndoStack()
    stack.push(MoveShape(sid, 5, 5), sd)
    stack.push(AdjustHandle(sid, "br", 20, 20), sd)
    stack.push(SetShapeLabel(sid, "x"), sd)
    stack.undo(sd)
    stack.redo(sd)

    # Each command emits once on execute and once on undo/redo.
    assert events.count(EVT_EDITED) >= 2
    assert EVT_LABEL in events


def test_all_frames_sentinel_visible_at_every_t_z():
    sd = ShapeData()
    sd.add(RECTANGLE, [0, 0, 10, 10], t=0, z=0)
    sid_all = sd.add(LINE, [0, 0, 5, 5], t=ALL_FRAMES, z=ALL_FRAMES, label="ref")
    sd.add(CIRCLE, [5, 5, 8, 5], t=2, z=1)

    visible_t0 = {s.shape_id for s in sd.get_time_slice(0)}
    visible_t2 = {s.shape_id for s in sd.get_time_slice(2)}
    visible_t5 = {s.shape_id for s in sd.get_time_slice(5)}

    assert sid_all in visible_t0
    assert sid_all in visible_t2
    assert sid_all in visible_t5

    # With z filter, the t=2/z=1 circle excludes other z's; the sentinel
    # stays visible.
    visible_t2_z1 = {s.shape_id for s in sd.get_time_slice(2, z=1)}
    visible_t2_z0 = {s.shape_id for s in sd.get_time_slice(2, z=0)}
    assert sid_all in visible_t2_z1
    assert sid_all in visible_t2_z0


def test_hit_test_honors_all_frames_sentinel():
    sd = ShapeData()
    sd.add(RECTANGLE, [0, 0, 10, 10], t=ALL_FRAMES, z=ALL_FRAMES)
    # Hit at any (t, z) should resolve to the sentinel shape.
    assert sd.hit_test((5, 5), t=0, z=0) == 0
    assert sd.hit_test((5, 5), t=7, z=3) == 0


# ---------------------------------------------------------------------------
# POLYLINE
# ---------------------------------------------------------------------------

from pyvistra.data.shapes import (
    POLYLINE,
    AddVertex,
    MoveVertex,
    RemoveVertex,
    SetPolylineFlags,
)


def test_polyline_add_and_outline():
    sd = ShapeData()
    verts = [(0, 0), (10, 0), (10, 10), (0, 10)]
    sid = sd.add(POLYLINE, vertices=verts, label="poly")
    rec = sd.get(sid)
    assert rec.shape_type == POLYLINE
    assert rec.vertices.shape == (4, 2)

    # Open polyline: outline equals control points exactly when smoothness=0.
    outline = get_outline(rec)
    np.testing.assert_allclose(outline, np.asarray(verts, dtype=np.float32))


def test_polyline_closed_outline_appends_first():
    sd = ShapeData()
    sid = sd.add(
        POLYLINE,
        vertices=[(0, 0), (10, 0), (10, 10), (0, 10)],
        properties={"closed": True},
    )
    outline = get_outline(sd.get(sid))
    # Closed: outline is verts + first vertex appended.
    assert outline.shape == (5, 2)
    np.testing.assert_allclose(outline[0], outline[-1])


def test_polyline_hit_test_segment_and_interior():
    sd = ShapeData()
    sid = sd.add(POLYLINE, vertices=[(0, 0), (10, 0), (10, 10), (0, 10)])
    # Open polyline: on-segment hit.
    assert sd.hit_test((5, 0)) == sid
    # Open polyline: interior is NOT a hit.
    assert sd.hit_test((5, 5)) is None

    # Close it: interior becomes a hit.
    SetPolylineFlags(sid, closed=True).execute(sd)
    assert sd.hit_test((5, 5)) == sid


def test_polyline_handles_one_per_vertex():
    sd = ShapeData()
    sid = sd.add(POLYLINE, vertices=[(0, 0), (10, 0), (5, 10)])
    handles = get_handles(sd.get(sid))
    assert set(handles.keys()) == {"v0", "v1", "v2"}
    assert handles["v2"] == (5.0, 10.0)


def test_polyline_vertex_commands_roundtrip():
    sd = ShapeData()
    sid = sd.add(POLYLINE, vertices=[(0, 0), (10, 0), (10, 10)])

    from pyvistra.layers.commands import UndoStack
    stack = UndoStack()

    # Add vertex at index 1.
    stack.push(AddVertex(sid, 1, 5.0, 0.0), sd)
    assert len(sd.get(sid).vertices) == 4
    np.testing.assert_allclose(sd.get(sid).vertices[1], [5.0, 0.0])

    stack.undo(sd)
    assert len(sd.get(sid).vertices) == 3

    # Move vertex 0.
    stack.push(MoveVertex(sid, 0, -5.0, -5.0), sd)
    np.testing.assert_allclose(sd.get(sid).vertices[0], [-5.0, -5.0])
    stack.undo(sd)
    np.testing.assert_allclose(sd.get(sid).vertices[0], [0.0, 0.0])

    # Remove vertex 1.
    stack.push(RemoveVertex(sid, 1), sd)
    assert len(sd.get(sid).vertices) == 2
    stack.undo(sd)
    assert len(sd.get(sid).vertices) == 3


def test_polyline_move_shape_translates_vertices():
    sd = ShapeData()
    sid = sd.add(POLYLINE, vertices=[(0, 0), (10, 0), (10, 10)])
    MoveShape(sid, 5.0, 3.0).execute(sd)
    expected = np.array([[5, 3], [15, 3], [15, 13]], dtype=np.float32)
    np.testing.assert_allclose(sd.get(sid).vertices, expected)


def test_polyline_serialization_roundtrip():
    sd = ShapeData()
    sd.add(
        POLYLINE,
        vertices=[(1, 2), (3, 4), (5, 6)],
        properties={"closed": True, "smoothness": 1.5},
        label="path",
    )
    sd2 = ShapeData.from_list(sd.to_list())
    rec = sd2.get(0)
    assert rec.shape_type == POLYLINE
    np.testing.assert_allclose(rec.vertices, [[1, 2], [3, 4], [5, 6]])
    assert rec.properties["closed"] is True
    assert rec.properties["smoothness"] == 1.5
    assert rec.label == "path"


# ---------------------------------------------------------------------------
# Kymograph path sampling
# ---------------------------------------------------------------------------

def test_kymograph_sample_coords_line_evenly_spaced():
    from pyvistra.widgets.kymograph_dialog import _sample_coords

    sd = ShapeData()
    sid = sd.add(LINE, [0, 0, 10, 0])
    coords = _sample_coords(sd.get(sid), 11)
    np.testing.assert_allclose(coords[:, 0], np.linspace(0, 10, 11))
    np.testing.assert_allclose(coords[:, 1], 0.0)


def test_kymograph_sample_coords_polyline_arc_length():
    from pyvistra.widgets.kymograph_dialog import _sample_coords

    sd = ShapeData()
    # L-shape: (0,0) → (10,0) → (10,10). Total arc length = 20.
    sid = sd.add(POLYLINE, vertices=[(0, 0), (10, 0), (10, 10)])
    coords = _sample_coords(sd.get(sid), 21)
    # Midpoint of arc should sit exactly at the corner (10, 0).
    np.testing.assert_allclose(coords[10], [10.0, 0.0], atol=1e-5)
    # Endpoint matches the last vertex.
    np.testing.assert_allclose(coords[-1], [10.0, 10.0], atol=1e-5)
