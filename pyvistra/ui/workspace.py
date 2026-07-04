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
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAction,
    QMainWindow,
    QMenu,
    QSplitter,
    QTabWidget,
)

from .manager import manager

_workspace = None


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
        if widget is not None:
            manager.set_active_window(widget)

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
                group.setParent(None)
                group.deleteLater()
                groups = self._groups()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_window(self, window, title=None):
        """Adopt *window* (any QWidget viewer) as a tab."""
        if self._group_of(window) is not None:
            group = self._group_of(window)
            group.setCurrentWidget(window)
            return
        # Embedded viewers share one top-level window, so their QAction
        # shortcuts must be focus-scoped or every second tab makes
        # Ctrl+S & co. ambiguous.
        for action in window.findChildren(QAction):
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        group = self._active_group()
        group.addTab(window, title or window.windowTitle() or "Image")
        group.setCurrentWidget(window)
        manager.set_active_window(window)

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
        window.setParent(None)
        window.show()
        window.raise_()
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
        panels_menu = self.menuBar().addMenu("Panels")

        layers_action = QAction("Layers", self)
        layers_action.triggered.connect(self._show_layer_manager)
        panels_menu.addAction(layers_action)

        labels_action = QAction("Labels && Masks", self)
        labels_action.triggered.connect(self._show_label_manager)
        panels_menu.addAction(labels_action)

    def _show_layer_manager(self):
        from .layer_manager import show_layer_manager

        show_layer_manager()

    def _show_label_manager(self):
        from .label_manager import get_label_manager

        get_label_manager().show()

    def closeEvent(self, event):
        global _workspace
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
