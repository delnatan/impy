"""Tests for data.view_state (ViewState observable navigation state)."""

from pyvistra.data.view_state import ViewState


def test_defaults():
    vs = ViewState()
    assert vs.t == 0
    assert vs.z == 0
    assert vs.z_projection is False
    assert vs.z_range is None


def test_setters_notify_with_field():
    vs = ViewState()
    events = []
    vs.subscribe(lambda field: events.append(field))

    vs.set_t(3)
    vs.set_z(2)
    vs.set_z_projection(True)
    vs.set_z_range(1, 4)

    assert events == ["t", "z", "z_projection", "z_range"]
    assert vs.t == 3
    assert vs.z == 2
    assert vs.z_projection is True
    assert vs.z_range == (1, 4)


def test_unchanged_value_does_not_notify():
    vs = ViewState(t=3, z=2)
    events = []
    vs.subscribe(lambda field: events.append(field))

    vs.set_t(3)
    vs.set_z(2)
    vs.set_z_projection(False)

    assert events == []


def test_unsubscribe():
    vs = ViewState()
    events = []
    unsub = vs.subscribe(lambda field: events.append(field))
    vs.set_t(1)
    unsub()
    vs.set_t(2)
    assert events == ["t"]


def test_clamp_notifies_only_out_of_bounds_fields():
    vs = ViewState(t=9, z=1)
    vs.set_z_range(0, 9)
    events = []
    vs.subscribe(lambda field: events.append(field))

    vs.clamp(T=4, Z=6)

    assert vs.t == 3
    assert vs.z == 1
    assert vs.z_range == (0, 5)
    assert events == ["t", "z_range"]


def test_clamp_without_z_range():
    vs = ViewState(t=5, z=5)
    vs.clamp(T=2, Z=2)
    assert vs.t == 1
    assert vs.z == 1
    assert vs.z_range is None


def test_listener_exception_does_not_break_others():
    vs = ViewState()
    events = []

    def bad(field):
        raise RuntimeError("boom")

    vs.subscribe(bad)
    vs.subscribe(lambda field: events.append(field))
    vs.set_t(1)
    assert events == ["t"]
