"""Explicit side-by-side comparison of exactly two windows.

A ``ComparisonPair`` is deliberately created by the user (see
``Workspace.compare_with_dialog``) to relate two open windows -- unlike
``Workspace``'s tab-group splitter, which is generic layout with no notion
of why two panes are side by side, a pair's whole reason to exist is the
relationship itself. ``ComparisonView`` is the two-pane widget that docks
as a single ordinary ``Workspace`` tab (see ``ui/workspace.py``); each side
holds exactly one window, permanently -- no tab bar per side and no further
splitting, which is what keeps a pair's lifecycle simple (created by
"Compare With...", ended by "Stop Comparing" or either member closing)
instead of inferred from tab-group emptiness the way a plain split is.
"""

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QSplitter, QVBoxLayout, QWidget


class ComparisonPair(QObject):
    """Owns the relationship between two windows compared side by side.

    Tracks which member last received focus (``active_member``) so menu
    dispatch can follow whichever pane the user actually clicked into, and
    ends itself the moment either member starts closing. Optionally keeps
    the two members' camera (pan/zoom) and/or T/Z/channel position in sync
    -- see ``set_link_pan_zoom``/``set_link_tzc``.
    """

    ended = Signal()

    def __init__(self, window_a, window_b, parent=None):
        super().__init__(parent)
        self.window_a = window_a
        self.window_b = window_b
        self.active_member = window_a
        # Set by _on_member_closing so Workspace._end_comparison knows which
        # member (if any) is mid-close and must not be re-docked -- its own
        # WA_DeleteOnClose teardown handles it.
        self.closing_member = None
        self._ended = False

        # Both link toggles default off (opt-in), matching TiledViewer's own
        # sync_pan_zoom default. The connections below are always live --
        # only these flags gate whether a change actually propagates, the
        # same shape as TiledViewer._on_tile_camera_changed's own
        # sync_pan_zoom/_syncing_camera check -- so toggling on/off never
        # has to (dis)connect anything.
        self.link_pan_zoom = False
        self.link_tzc = False
        self._syncing_view = False
        self._syncing_tzc = False

        # (emitter, slot) pairs to disconnect in end(); Qt signals and
        # vispy EventEmitters both expose connect()/disconnect(slot), so
        # one generic list covers both.
        self._connections = []
        self._view_state_unsubs = {}

        for window in (window_a, window_b):
            self._connect(window.window_activated, self._on_member_activated)
            self._connect(window.window_closing, self._on_member_closing)

            camera_handler = lambda event, w=window: self._on_camera_event(w)
            self._connect(window.canvas.events.mouse_wheel, camera_handler)
            self._connect(window.canvas.events.mouse_release, camera_handler)

            channel_handler = lambda c, w=window: self._on_channel_changed(w)
            self._connect(window.channel_changed, channel_handler)

            self._view_state_unsubs[window] = window.view_state.subscribe(
                lambda field, w=window: self._on_view_state_changed(w, field)
            )

    def _connect(self, emitter, slot):
        emitter.connect(slot)
        self._connections.append((emitter, slot))

    def other(self, window):
        """Return whichever member *window* is not."""
        return self.window_b if window is self.window_a else self.window_a

    def _on_member_activated(self, window):
        self.active_member = window

    def _on_member_closing(self, window):
        self.closing_member = window
        self.end()

    # ------------------------------------------------------------------
    # Linked pan/zoom
    # ------------------------------------------------------------------

    def set_link_pan_zoom(self, enabled):
        """Enable/disable keeping both members' camera view in sync.

        Turning it on immediately snaps the other member to whichever was
        last active, rather than waiting for the next pan/zoom.
        """
        self.link_pan_zoom = bool(enabled)
        if self.link_pan_zoom:
            self._sync_camera_from(self.active_member)

    def _on_camera_event(self, source):
        if not self.link_pan_zoom or self._syncing_view:
            return
        self._sync_camera_from(source)

    def _sync_camera_from(self, source):
        target = self.other(source)
        self._syncing_view = True
        try:
            target.view.camera.rect = source.view.camera.rect
            target.canvas.update()
        finally:
            self._syncing_view = False

    # ------------------------------------------------------------------
    # Linked T/Z/channel
    # ------------------------------------------------------------------

    def set_link_tzc(self, enabled):
        """Enable/disable keeping both members' T/Z/channel position in
        sync (independently of ``set_link_pan_zoom``)."""
        self.link_tzc = bool(enabled)
        if self.link_tzc:
            self._sync_tzc_from(self.active_member)

    def _on_view_state_changed(self, source, field):
        if not self.link_tzc or self._syncing_tzc or field not in ("t", "z"):
            return
        target = self.other(source)
        self._syncing_tzc = True
        try:
            vs = source.view_state
            if field == "t":
                target.view_state.set_t(min(vs.t, target.T - 1))
            else:
                target.view_state.set_z(min(vs.z, target.Z - 1))
        finally:
            self._syncing_tzc = False

    def _on_channel_changed(self, source):
        if not self.link_tzc or self._syncing_tzc:
            return
        target = self.other(source)
        self._syncing_tzc = True
        try:
            target.on_channel_change(min(source.c_idx, target.C - 1))
        finally:
            self._syncing_tzc = False

    def _sync_tzc_from(self, source):
        target = self.other(source)
        self._syncing_tzc = True
        try:
            target.view_state.set_t(min(source.t_idx, target.T - 1))
            target.view_state.set_z(min(source.z_idx, target.Z - 1))
            target.on_channel_change(min(source.c_idx, target.C - 1))
        finally:
            self._syncing_tzc = False

    def end(self):
        """Tear down the pair (idempotent)."""
        if self._ended:
            return
        self._ended = True
        for emitter, slot in self._connections:
            try:
                emitter.disconnect(slot)
            except Exception:
                # Already disconnected, or the window is mid-teardown.
                pass
        self._connections.clear()
        for unsub in self._view_state_unsubs.values():
            unsub()
        self._view_state_unsubs.clear()
        self.ended.emit()


class ComparisonView(QWidget):
    """Two-pane container hosting a ``ComparisonPair``'s windows side by side.

    Docks as a single ordinary ``Workspace`` tab -- from the tab group's
    point of view this is just another viewer widget, the same way
    ``OrthoViewer``/``VolumeViewer`` are.
    """

    def __init__(self, pair, parent=None):
        super().__init__(parent)
        self.pair = pair

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        controls = QHBoxLayout()
        controls.setContentsMargins(6, 4, 6, 4)
        link_pan_zoom_cb = QCheckBox("Link Pan/Zoom")
        link_pan_zoom_cb.toggled.connect(pair.set_link_pan_zoom)
        link_tzc_cb = QCheckBox("Link T/Z/C")
        link_tzc_cb.toggled.connect(pair.set_link_tzc)
        controls.addWidget(link_pan_zoom_cb)
        controls.addWidget(link_tzc_cb)
        controls.addStretch()
        layout.addLayout(controls)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(pair.window_a)
        self._splitter.addWidget(pair.window_b)
        layout.addWidget(self._splitter)

        # Both windows come from a _TabGroup (QTabWidget), which explicitly
        # hides every tab page that isn't the current one -- and leaves the
        # *outgoing* current page hidden too once its tab is removed. That
        # explicit hidden state isn't cleared just by reparenting into a new,
        # visible parent; QSplitter then allocates a hidden child exactly
        # zero space; both children stayed hidden here, so the pane you and
        # the user actually see was really a splitter with two 0-width
        # children -- the symptom was a blank comparison tab with no visible
        # canvas at all. Explicitly (re)show them once they're in place.
        pair.window_a.show()
        pair.window_b.show()

    def members(self):
        return (self.pair.window_a, self.pair.window_b)


def paired_window(window):
    """Return *window*'s ``ComparisonPair`` partner, or ``None`` if it isn't
    currently part of a pair.

    Shared lookup for anything that wants to act on "the other side of the
    comparison" -- e.g. shape-based tools (``LineProfileDialog``,
    ``RadialProfileDialog``) auto-adding the paired window as a comparison
    series alongside whichever window a line/circle was drawn in.
    """
    view = getattr(window, "_comparison_view", None)
    return view.pair.other(window) if view is not None else None
