"""Shared QDockWidget helpers.

Small, generic utilities for docks embedded in an ``ImageWindow`` — kept
separate from any one panel's module so panels don't depend on each other
for plumbing that has nothing to do with what they display.
"""

from __future__ import annotations

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QDockWidget


def nudge_window_size(window, *, deferred: bool = True) -> None:
    """Force *window* to redispatch a real resize event at its current size.

    Some of *window*'s children (notably vispy's GL canvas backend) only
    recompute their viewport on an actually-dispatched resize event, not on
    a layout recompute alone — resizing by one pixel and back forces a real
    one. *deferred* runs the nudge on the next event-loop tick, needed when
    the triggering geometry change (e.g. a reparent or dock float) hasn't
    been fully settled by the OS/window-manager yet.
    """

    def _nudge():
        size = window.size()
        window.resize(size.width() + 1, size.height())
        window.resize(size)

    if deferred:
        QTimer.singleShot(0, _nudge)
    else:
        _nudge()


def make_dock_float_resize_resilient(dock: QDockWidget, window) -> None:
    """Force *window* to redispatch a real resize event whenever *dock*
    changes float state (floated or redocked).

    ``QMainWindow``'s internal layout recomputes dock-area geometry on a
    float-state change, but vispy's canvas backend only recomputes its GL
    viewport on an actually-dispatched resize event — the recompute doesn't
    reliably trigger one for the central widget, so the canvas silently
    stops tracking the window's shape until something else forces a real
    resize. Same root cause and fix as ``Workspace.float_window``'s
    tab-detach nudge (:func:`nudge_window_size`).
    """
    dock.topLevelChanged.connect(lambda _floating: nudge_window_size(window))
