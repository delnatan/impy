"""Tests for layers.base (Layer, LayerList) and layers.commands (UndoStack)."""

import pytest

from pyvistra.layers import Layer, LayerList, Command, UndoStack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MutableData:
    """Trivial mutable data model for testing."""
    def __init__(self):
        self.items = []


class AddItem:
    """Command that appends an item."""
    def __init__(self, value):
        self.value = value

    def execute(self, data):
        data.items.append(self.value)

    def undo(self, data):
        data.items.remove(self.value)


class RemoveItem:
    """Command that removes an item."""
    def __init__(self, value):
        self.value = value
        self._index = None

    def execute(self, data):
        self._index = data.items.index(self.value)
        data.items.remove(self.value)

    def undo(self, data):
        data.items.insert(self._index, self.value)


# ---------------------------------------------------------------------------
# UndoStack tests
# ---------------------------------------------------------------------------

def test_undo_stack_push_and_undo():
    stack = UndoStack()
    data = MutableData()
    stack.push(AddItem("a"), data)
    stack.push(AddItem("b"), data)
    assert data.items == ["a", "b"]

    stack.undo(data)
    assert data.items == ["a"]
    stack.undo(data)
    assert data.items == []


def test_undo_stack_redo():
    stack = UndoStack()
    data = MutableData()
    stack.push(AddItem("x"), data)
    stack.undo(data)
    assert data.items == []

    stack.redo(data)
    assert data.items == ["x"]


def test_undo_stack_push_clears_redo():
    stack = UndoStack()
    data = MutableData()
    stack.push(AddItem("a"), data)
    stack.push(AddItem("b"), data)
    stack.undo(data)
    assert stack.can_redo

    stack.push(AddItem("c"), data)
    assert not stack.can_redo
    assert data.items == ["a", "c"]


def test_undo_stack_empty_returns_false():
    stack = UndoStack()
    data = MutableData()
    assert not stack.undo(data)
    assert not stack.redo(data)


def test_undo_stack_max_size():
    stack = UndoStack(max_size=3)
    data = MutableData()
    for i in range(5):
        stack.push(AddItem(i), data)
    # Only last 3 commands kept
    assert len(stack._undo) == 3
    # Undo should work for the remaining 3
    for _ in range(3):
        assert stack.undo(data)
    assert not stack.undo(data)


def test_command_protocol():
    assert isinstance(AddItem("x"), Command)
    assert isinstance(RemoveItem("x"), Command)


# ---------------------------------------------------------------------------
# Layer tests
# ---------------------------------------------------------------------------

def _make_layer(name="test", layer_type="shapes"):
    return Layer(name=name, layer_type=layer_type, data=MutableData(), visual=None)


def test_layer_defaults():
    layer = _make_layer()
    assert layer.visible is True
    assert layer.locked is False
    assert layer.style == {}
    assert layer.selected_ids == set()
    assert isinstance(layer.undo_stack, UndoStack)


def test_layer_set_visible():
    layer = _make_layer()
    layer.set_visible(False)
    assert layer.visible is False


def test_layer_set_locked():
    layer = _make_layer()
    layer.set_locked(True)
    assert layer.locked is True
    layer.set_locked(False)
    assert layer.locked is False


# ---------------------------------------------------------------------------
# LayerList tests
# ---------------------------------------------------------------------------

def test_layer_list_add_and_iterate():
    ll = LayerList()
    l1 = _make_layer("shapes1", "shapes")
    l2 = _make_layer("points1", "points")
    ll.add(l1)
    ll.add(l2)
    assert len(ll) == 2
    assert list(ll) == [l1, l2]
    assert ll.names() == ["shapes1", "points1"]


def test_layer_list_duplicate_name_rejected():
    ll = LayerList()
    ll.add(_make_layer("a", "shapes"))
    with pytest.raises(ValueError, match="already exists"):
        ll.add(_make_layer("a", "points"))


def test_layer_list_invalid_type_rejected():
    ll = LayerList()
    with pytest.raises(ValueError, match="Invalid layer type"):
        ll.add(_make_layer("x", "invalid"))


def test_layer_list_remove():
    ll = LayerList()
    ll.add(_make_layer("a", "shapes"))
    ll.add(_make_layer("b", "shapes"))
    removed = ll.remove("a")
    assert removed.name == "a"
    assert "a" not in ll
    assert len(ll) == 1


def test_layer_list_remove_updates_active():
    ll = LayerList()
    ll.add(_make_layer("a", "shapes"))
    ll.add(_make_layer("b", "shapes"))
    assert ll.active("shapes").name == "a"
    ll.remove("a")
    assert ll.active("shapes").name == "b"


def test_layer_list_remove_last_clears_active():
    ll = LayerList()
    ll.add(_make_layer("a", "shapes"))
    ll.remove("a")
    assert ll.active("shapes") is None


def test_layer_list_rename():
    ll = LayerList()
    ll.add(_make_layer("old", "shapes"))
    ll.rename("old", "new")
    assert "new" in ll
    assert "old" not in ll
    assert ll["new"].name == "new"
    assert ll.active("shapes").name == "new"


def test_layer_list_rename_duplicate_rejected():
    ll = LayerList()
    ll.add(_make_layer("a", "shapes"))
    ll.add(_make_layer("b", "points"))
    with pytest.raises(ValueError, match="already exists"):
        ll.rename("a", "b")


def test_layer_list_by_type():
    ll = LayerList()
    ll.add(_make_layer("s1", "shapes"))
    ll.add(_make_layer("p1", "points"))
    ll.add(_make_layer("s2", "shapes"))
    shapes = ll.by_type("shapes")
    assert [l.name for l in shapes] == ["s1", "s2"]


def test_layer_list_set_active():
    ll = LayerList()
    ll.add(_make_layer("a", "shapes"))
    ll.add(_make_layer("b", "shapes"))
    ll.set_active("shapes", "b")
    assert ll.active("shapes").name == "b"


def test_layer_list_set_active_none():
    ll = LayerList()
    ll.add(_make_layer("a", "shapes"))
    ll.set_active("shapes", None)
    assert ll.active("shapes") is None


def test_layer_list_auto_active_on_first_add():
    ll = LayerList()
    ll.add(_make_layer("first", "points"))
    assert ll.active("points").name == "first"
    ll.add(_make_layer("second", "points"))
    assert ll.active("points").name == "first"  # unchanged
