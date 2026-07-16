"""Workspace shell: one main window, viewers as tabs, panels as docks.

``present_window(viewer)`` is the single entry point for showing any
viewer (``ImageWindow``, ``OrthoViewer``, ``VolumeViewer``, …): it adds
the viewer as a tab in the workspace singleton, creating the workspace
on first use. ``floating=True`` bypasses the shell and shows a plain
top-level window (the pre-workspace behaviour).

The central area is a horizontal splitter of tab groups. Tabs can be
split right (side-by-side comparison) or floated out via the tab-bar
context menu; a floated viewer can be re-adopted with
``present_window`` and empty tab groups collapse automatically.

Viewer lifecycle stays unchanged: closing a tab calls the viewer's
``close()`` (normal ``closeEvent`` → ``manager.unregister``), and a
viewer closing itself removes its tab via ``window_unregistered``.

Every viewer class declares a class-level ``MENU_SPEC`` (see
``window.MENU_SPEC`` for the format). While a viewer is docked, its own
``QMainWindow`` menu bar is hidden and its spec is mirrored onto the
workspace's persistent menu bar via ``build_proxy_menus``, dispatching
to whichever tab is currently active; only the active tab's top-level
menus are actually attached to the menu bar (``_refresh_mirrored_menus``
adds/removes them, it does not just call ``setVisible``), so the menu
bar always matches the selected tab. Floating a tab back out restores
its embedded menu bar. Hiding the embedded bars is load-bearing on
macOS: every docked ``QMainWindow``'s ``QMenuBar`` attaches to the same
native top-of-screen menu bar as the workspace's own, and whichever
registers last hijacks it — tab switches never re-evaluate it, leaving
the native bar stuck on a stale viewer's menus. Structurally
add/removing the mirrored top-level actions (rather than toggling
their visibility) matters for the same reason: on macOS, ``setVisible``
on an action already rendered into the native menu bar does not
reliably re-sync the underlying NSMenu, so a "hidden" menu can keep
showing — and stay clickable — after the active tab changes.
"""

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAction,
    QMainWindow,
    QMenu,
    QSplitter,
    QTabWidget,
)

from .manager import manager

_workspace = None


def _build_proxy_menu_items(menu, items, get_target, leaf_actions):
    """Populate *menu* from *items*, recursing into nested ``"submenu"``
    entries. Appends every dispatchable leaf action to *leaf_actions*."""
    for item in items:
        if item is None:
            menu.addSeparator()
            continue
        if "submenu" in item:
            submenu = menu.addMenu(item["label"])
            _build_proxy_menu_items(submenu, item["submenu"], get_target, leaf_actions)
            continue
        action = QAction(item["label"], menu)
        if "shortcut" in item:
            action.setShortcut(item["shortcut"])
        if "tooltip" in item:
            action.setToolTip(item["tooltip"])
        enabled = item.get("enabled", True)
        # Cached on the action itself so _refresh_mirrored_menus can AND
        # it with tab-match -- that method unconditionally re-enables
        # every leaf action of the active spec, which would otherwise
        # clobber an item disabled here (e.g. an optional dependency
        # that isn't installed) the next time its tab becomes active.
        action._pv_enabled = enabled() if callable(enabled) else enabled
        action.setEnabled(action._pv_enabled)
        method_name = item["method"]

        def _dispatch(checked=False, method_name=method_name):
            target = get_target()
            if target is not None:
                getattr(target, method_name)()

        action.triggered.connect(_dispatch)
        menu.addAction(action)
        leaf_actions.append(action)


def build_proxy_menus(menubar, spec, get_target):
    """Populate *menubar* from *spec* (see ``window.MENU_SPEC``), where
    each action dispatches to ``getattr(get_target(), item["method"])``
    at trigger time rather than a fixed bound method. Items may nest a
    ``"submenu"`` list instead of a ``"method"`` -- their leaves are
    folded into the same flat ``leaf_actions``.

    Returns ``(top_level_actions, leaf_actions)``. When ``get_target()``
    has nothing sensible to act on, hiding the top-level menu actions
    is not enough — a leaf action's own ``isEnabled()`` is untouched by
    its parent menu, so an enabled leaf action with a shortcut (e.g.
    Shift+C, which several viewer specs declare) stays "live" for Qt's
    shortcut matching even while its menu is hidden — colliding with
    the same shortcut mirrored for the viewer class that *is* active
    and making *both* ambiguous, so *neither* fires. See
    ``Workspace._refresh_mirrored_menus``.
    """
    from .. import plugins

    plugins.discover_plugins()

    top_level_actions = []
    leaf_actions = []
    for menu_title, items in spec:
        menu = menubar.addMenu(menu_title)
        top_level_actions.append(menu.menuAction())
        _build_proxy_menu_items(menu, items, get_target, leaf_actions)
    return top_level_actions, leaf_actions


class _TabGroup(QTabWidget):
    """One group of viewer tabs inside the workspace splitter."""

    def __init__(self, workspace):
        super().__init__()
        self._workspace = workspace
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setDocumentMode(True)
        self.tabCloseRequested.connect(self._close_tab)
        self.currentChanged.connect(self._activate_current)
        # currentChanged does not fire when the clicked tab is already
        # the current one in its group -- exactly what happens with two
        # single-tab groups side by side after a split. Without this,
        # clicking the "other" group's tab leaves the workspace's
        # notion of the active tab (and manager.active_window) stuck on
        # whichever tab last actually changed, so menu actions dispatch
        # to the wrong window.
        self.tabBar().tabBarClicked.connect(self._activate_current)
        self.tabBar().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tabBar().customContextMenuRequested.connect(self._tab_menu)

    def _close_tab(self, idx):
        widget = self.widget(idx)
        if widget is None:
            return
        if widget.close():
            # ImageWindows drop their tab via window_unregistered (and
            # WA_DeleteOnClose); other viewers need an explicit removal.
            i = self.indexOf(widget)
            if i >= 0:
                self.removeTab(i)
            self._workspace._collapse_empty_groups()

    def _activate_current(self, idx):
        widget = self.widget(idx)
        # Only ImageWindow instances register with `manager` — Ortho/
        # Volume/ZMontage/Tiled viewer tabs never do — so track the
        # workspace's own notion of "active tab" independently rather
        # than relying on manager.active_window, which would otherwise
        # keep pointing at the last *ImageWindow* tab forever.
        self._workspace._active_tab_widget = widget
        if widget is not None:
            manager.set_active_window(widget)
        self._workspace._refresh_mirrored_menus()

    def _tab_menu(self, pos):
        idx = self.tabBar().tabAt(pos)
        if idx < 0:
            return
        menu = QMenu(self)
        split_action = QAction("Split Right", menu)
        float_action = QAction("Float Window", menu)
        close_action = QAction("Close", menu)
        menu.addAction(split_action)
        menu.addAction(float_action)
        menu.addSeparator()
        menu.addAction(close_action)
        chosen = menu.exec_(self.tabBar().mapToGlobal(pos))
        if chosen is split_action:
            self._workspace.split_right(self.widget(idx))
        elif chosen is float_action:
            self._workspace.float_window(self.widget(idx))
        elif chosen is close_action:
            self._close_tab(idx)


class Workspace(QMainWindow):
    """Single main window hosting every viewer as a tab."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyvistra")
        self.resize(1200, 850)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self._splitter)
        self._active_tab_widget = None
        # spec id -> (spec, top_level_actions, leaf_actions); filled
        # lazily by _mirror_spec the first time a viewer class with
        # that MENU_SPEC is docked.
        self._mirrored = {}
        # Single shared Channels & Contrast popup for every docked tab,
        # created lazily by show_channel_panel() and retargeted across
        # tabs via ChannelPopup.rebind_viewer -- see that function's
        # docstring.
        self._channel_popup = None
        self._add_group()

        # A viewer that closes itself (or is closed via its tab) must
        # drop its tab; unregister fires for both paths.
        manager.window_unregistered.connect(self._drop_tab)

        self._setup_menu()

    # ------------------------------------------------------------------
    # Tab groups
    # ------------------------------------------------------------------

    def _add_group(self):
        group = _TabGroup(self)
        self._splitter.addWidget(group)
        return group

    def _groups(self):
        return [
            self._splitter.widget(i) for i in range(self._splitter.count())
        ]

    def _group_of(self, widget):
        for group in self._groups():
            if group.indexOf(widget) >= 0:
                return group
        return None

    def _active_group(self):
        for group in self._groups():
            if group.currentWidget() is not None and group.hasFocus():
                return group
        return self._groups()[-1]

    def _collapse_empty_groups(self):
        groups = self._groups()
        for group in groups:
            if group.count() == 0 and len(groups) > 1:
                # Do not setParent(None) here: removeTab() doesn't detach a
                # just-closed viewer that's still pending its deferred
                # WA_DeleteOnClose deletion, so reparenting this "empty"
                # group can drag a half-torn-down QMainWindow (with live
                # vispy/OpenGL children) out of the splitter and crash.
                # deleteLater() lets Qt's normal teardown handle it safely.
                group.deleteLater()
                groups = self._groups()
        # Closing a group's last tab fires currentChanged(-1) for *that*
        # group, which _activate_current blindly writes into the
        # workspace-wide _active_tab_widget -- even when another group
        # still has a perfectly good current tab. Left uncorrected, the
        # menu bar mirror stays blank for the surviving window until the
        # user clicks its tab. Re-derive the active tab whenever it no
        # longer points at a widget actually hosted by a group.
        if self._group_of(self._active_tab_widget) is None:
            self._reactivate_surviving_tab()
        self._refresh_mirrored_menus()

    def _reactivate_surviving_tab(self):
        """Pick a sensible active tab from whatever groups remain, skipping
        empty groups still pending their deferred deletion."""
        widget = None
        for group in self._groups():
            if group.hasFocus() and group.currentWidget() is not None:
                widget = group.currentWidget()
                break
        else:
            for group in reversed(self._groups()):
                if group.currentWidget() is not None:
                    widget = group.currentWidget()
                    break
        self._active_tab_widget = widget
        if widget is not None:
            manager.set_active_window(widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_window(self, window, title=None):
        """Adopt *window* (any QWidget viewer) as a tab."""
        if self._group_of(window) is not None:
            group = self._group_of(window)
            group.setCurrentWidget(window)
            return
        # A previously-floated window (re-adopted via present_window, per
        # the module docstring) is a live top-level QMainWindow; addTab()
        # below reparents it under the workspace, which is the same
        # cross-top-level reparent that corrupts a vispy QOpenGLWidget's
        # GL context in float_window() -- see _rebuild_canvas_for_float.
        # A window that has never been shown (the common case, fresh from
        # ImageWindow()) is not yet a live top-level window, so no rebuild
        # is needed for it.
        was_shown = window.isVisible()
        # Embedded viewers share one top-level window, so their QAction
        # shortcuts must be focus-scoped or every second tab makes
        # Ctrl+S & co. ambiguous. WidgetWithChildrenShortcut context
        # alone isn't enough: an action added via menu.addAction() has
        # only that QMenu as its associated widget, whose own
        # descendant tree is empty while the menu is closed, so the
        # shortcut would never match real focus (e.g. the vispy
        # canvas). Also registering *window* itself via addAction()
        # makes the window (and its real descendant tree) an
        # associated widget too, so "widget or its children have
        # focus" actually means the embedded viewer's canvas/controls.
        for action in window.findChildren(QAction):
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            window.addAction(action)
        # Every viewer class declares a MENU_SPEC, mirrored onto the
        # workspace's persistent bar (see build_proxy_menus): hide the
        # viewer's own embedded menu bar and disarm just its MENU_SPEC
        # shortcuts, so e.g. Ctrl+S doesn't fire both the hidden
        # embedded action and the proxy action. Hiding the embedded bar
        # is not optional on macOS: it would attach to the same native
        # top-of-screen menu bar as the workspace's own and hijack it
        # permanently (tab switches never re-evaluate native menu bar
        # ownership within one top-level window).
        # Set before _mirror_spec: the first time a given MENU_SPEC is
        # mirrored, _mirror_spec's own _refresh_mirrored_menus() call (see
        # below) needs this window's spec to already read as active, or it
        # adds the just-built top-level actions and immediately removes
        # them again (stale active tab), relying on the second
        # _refresh_mirrored_menus() call at the end of this method to
        # re-add them. Harmless in practice since both calls converge to
        # the same end state, but pointlessly churns the native menu bar
        # on the exact "first viewer of this class, single tab, right as
        # it opens" path -- avoid the churn instead of relying on it
        # resolving itself.
        self._active_tab_widget = window

        spec = getattr(type(window), "MENU_SPEC", None)
        if spec is not None:
            self._mirror_spec(spec)
            menu_actions = getattr(window, "_menu_actions", [])
            window._docked_menu_shortcuts = {
                action: action.shortcut()
                for action in menu_actions
                if not action.shortcut().isEmpty()
            }
            for action in window._docked_menu_shortcuts:
                action.setShortcut(QKeySequence())
            # Force the viewer's own menu bar off the native top-of-screen
            # bar permanently (not just hidden) before hiding it. A native
            # QMenuBar's visibility toggling doesn't reliably resync the
            # underlying NSMenu on macOS -- the same class of bug the
            # class docstring documents for the mirrored workspace bar --
            # so re-showing it later in float_window() can leave the
            # floated window with no visible menu at all. Once non-native
            # it renders as a plain in-window widget, whose visibility
            # toggles exactly as reliably as any other widget.
            window.menuBar().setNativeMenuBar(False)
            window.menuBar().setVisible(False)
        group = self._active_group()
        group.addTab(window, title or window.windowTitle() or "Image")
        group.setCurrentWidget(window)
        if was_shown and hasattr(window, "_rebuild_canvas_for_float"):
            window._rebuild_canvas_for_float()
        manager.set_active_window(window)
        self._refresh_mirrored_menus()

    def split_right(self, window):
        """Move *window*'s tab into a new tab group to the right."""
        group = self._group_of(window)
        if group is None:
            return
        title = group.tabText(group.indexOf(window))
        group.removeTab(group.indexOf(window))
        new_group = self._add_group()
        new_group.addTab(window, title)
        new_group.setCurrentWidget(window)
        self._collapse_empty_groups()

    def float_window(self, window):
        """Detach *window* into an ordinary top-level window."""
        group = self._group_of(window)
        if group is None:
            return
        group.removeTab(group.indexOf(window))
        # Restore this window's own menu bar and the shortcuts disarmed
        # in add_window, since it no longer shares the workspace's bar.
        for action, shortcut in getattr(
            window, "_docked_menu_shortcuts", {}
        ).items():
            action.setShortcut(shortcut)
        window._docked_menu_shortcuts = {}
        menu_bar = window.menuBar() if hasattr(window, "menuBar") else None
        if menu_bar is not None:
            menu_bar.setVisible(True)
        window.setParent(None)
        # ImageWindow owns a live vispy QOpenGLWidget canvas; reparenting
        # it from being a tab's child to a top-level window corrupts its
        # GL context on this platform (GL errors, then a hard crash) --
        # rebuild the canvas fresh instead of letting show() paint the
        # reparented one. Other docked viewer types don't have this
        # method and keep today's (unfixed) behavior.
        if hasattr(window, "_rebuild_canvas_for_float"):
            window._rebuild_canvas_for_float()
        window.show()
        window.raise_()
        # Reparenting out of the tab group back to a real top-level window
        # can leave QMainWindowLayout (and children that only recompute on
        # an actual geometry change, e.g. vispy's canvas backend) holding
        # sizes computed for the old embedded geometry -- nothing relayouts
        # until the user manually drags an edge. Defer a same-size nudge to
        # the next event-loop tick (after the OS has assigned the new
        # top-level window's real frame) so both the Qt layout and the
        # canvas recompute immediately instead of waiting on a manual
        # resize.
        def _nudge_size():
            size = window.size()
            window.resize(size.width() + 1, size.height())
            window.resize(size)

        QTimer.singleShot(0, _nudge_size)
        self._collapse_empty_groups()

    def _drop_tab(self, window):
        group = self._group_of(window)
        if group is not None:
            group.removeTab(group.indexOf(window))
            self._collapse_empty_groups()

    # ------------------------------------------------------------------
    # Menu / panels
    # ------------------------------------------------------------------

    def _setup_menu(self):
        # macOS's native top-of-screen menu bar syncs when the *key
        # window* changes, not reliably when one window's own menu bar
        # mutates its action list while staying focused -- which is
        # exactly what tab switching does here (see
        # _refresh_mirrored_menus). Rendering the bar as a normal
        # in-window widget instead of the native one sidesteps that
        # sync bug entirely: an inactive tab's menus reliably disappear
        # and the active tab's reliably appear, because it's just a
        # QMenuBar repainting like any other widget.
        self.menuBar().setNativeMenuBar(False)
        panels_menu = self.menuBar().addMenu("Panels")

        layers_action = QAction("Layers", self)
        layers_action.triggered.connect(self._show_layer_manager)
        panels_menu.addAction(layers_action)

        labels_action = QAction("Labels && Masks", self)
        labels_action.triggered.connect(self._show_label_manager)
        panels_menu.addAction(labels_action)

        # Viewer menus (window.MENU_SPEC & co.) are mirrored onto this
        # persistent bar lazily by add_window → _mirror_spec, the first
        # time a viewer class with that spec is docked.

    def _mirror_spec(self, spec):
        """Mirror *spec* onto the workspace menu bar (once per spec),
        dispatching each action to the active tab while that tab's
        class declares this spec."""
        if id(spec) in self._mirrored:
            return

        def _target():
            widget = self._active_tab_widget
            if getattr(type(widget), "MENU_SPEC", None) is spec:
                return widget
            return None

        top_actions, leaf_actions = build_proxy_menus(
            self.menuBar(), spec, _target
        )
        self._mirrored[id(spec)] = (spec, top_actions, leaf_actions)
        self._refresh_mirrored_menus()

    def _refresh_mirrored_menus(self):
        """Show only the active tab's mirrored menus on the workspace
        bar (called on every tab activation/addition/removal).

        Top-level menus are structurally added to/removed from the
        menu bar (not just hidden via ``setVisible``) — on macOS,
        toggling an already-native-rendered ``QMenuBar`` action's
        visibility does not reliably re-sync the underlying NSMenu, so
        a hidden menu can keep showing (and stay clickable) after the
        active tab changes. Adding/removing the action forces Qt to
        rebuild the native menu bar, which does sync correctly.
        """
        menubar = self.menuBar()
        bar_actions = menubar.actions()
        active_spec = getattr(
            type(self._active_tab_widget), "MENU_SPEC", None
        )
        for spec, top_actions, leaf_actions in self._mirrored.values():
            match = spec is active_spec
            for action in top_actions:
                in_bar = action in bar_actions
                if match and not in_bar:
                    menubar.addAction(action)
                elif not match and in_bar:
                    menubar.removeAction(action)
            for action in leaf_actions:
                # Leaf actions must be disabled too, not just hidden
                # with their parent menu — an enabled leaf action's
                # shortcut is still "live" for Qt's global shortcut
                # matching regardless of whether its menu is visible,
                # and would otherwise collide with the same shortcut
                # (e.g. Shift+C) mirrored for the viewer class whose
                # tab actually is active, making both ambiguous so
                # neither fires.
                action.setEnabled(match and getattr(action, "_pv_enabled", True))

    def _show_layer_manager(self):
        from .layer_manager import show_layer_manager

        show_layer_manager()

    def _show_label_manager(self):
        from .label_manager import get_label_manager

        get_label_manager().show()

    def closeEvent(self, event):
        global _workspace
        if self._channel_popup is not None:
            self._channel_popup.close()
            self._channel_popup = None
        # Close every hosted viewer so their closeEvents run.
        for group in self._groups():
            while group.count():
                widget = group.widget(0)
                group.removeTab(0)
                widget.close()
        _workspace = None
        super().closeEvent(event)


def workspace_exists():
    return _workspace is not None


def get_workspace():
    """Return the workspace singleton, creating it on first use."""
    global _workspace
    if _workspace is None:
        _workspace = Workspace()
    return _workspace


def present_window(window, floating=False, title=None):
    """Show *window* — as a workspace tab by default, or top-level when
    ``floating=True``. Returns *window* for chaining."""
    if floating:
        window.show()
        return window
    ws = get_workspace()
    ws.add_window(window, title=title)
    ws.show()
    ws.raise_()
    return window


def show_channel_panel(viewer):
    """Show *viewer*'s Channels & Contrast control as a compact popup.

    A docked tab (``not viewer.isWindow()``) shares the workspace's
    single ``ChannelPopup`` instance, retargeted via ``rebind_viewer`` --
    switching tabs and reopening it never accumulates extra windows, the
    same reasoning that motivated putting every viewer in one workspace
    to begin with. A floating window (``floating=True``, no workspace to
    share one with) gets its own private popup instead, created once and
    reused on every subsequent open.
    """
    from ..widgets.channel_panel import ChannelPopup

    if not viewer.isWindow():
        ws = get_workspace()
        if ws._channel_popup is None:
            ws._channel_popup = ChannelPopup(viewer, parent=ws)
        else:
            ws._channel_popup.rebind_viewer(viewer)
        popup = ws._channel_popup
    else:
        if getattr(viewer, "_channel_popup", None) is None:
            viewer._channel_popup = ChannelPopup(viewer, parent=viewer)
        popup = viewer._channel_popup
    popup.show()
    popup.raise_()
