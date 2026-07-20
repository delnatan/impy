"""Checkable file list with per-row status flags.

Generic building block for "point at a folder, review/skip individual
files, run something over the checked ones" dialogs (the shape a batch
processor needs) -- deliberately has no notion of what the files
*contain* or why one might be flagged; a caller inspects each file
itself and reports back via :meth:`set_flag`.

Lighter than :class:`~.thumbnail_grid.ThumbnailGridWidget` on purpose: no
decoding, no custom-painted grid, just a plain ``QListWidget`` with
checkable items and a small colored status square per row (theming for
selection/hover already comes from the app's global stylesheet, see
``theme.py``).
"""

import os

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QDragEnterEvent, QDropEvent, QIcon, QPixmap
from qtpy.QtWidgets import QListWidget, QListWidgetItem

_FLAG_COLORS = {
    "ok": None,
    "warn": "#e0a030",
    "error": "#f44444",
}

_ICON_SIZE = 10


def _flag_icon(color_hex):
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.transparent)
    if color_hex:
        from qtpy.QtGui import QPainter

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(color_hex))
        painter.drawEllipse(0, 0, _ICON_SIZE, _ICON_SIZE)
        painter.end()
    return QIcon(pixmap)


class FlaggableFileListWidget(QListWidget):
    """Checkable list of file paths with an optional per-row status flag.

    Signals
    -------
    filesChanged
        Emitted after :meth:`set_folder` / :meth:`add_files` repopulate
        the list (a good time for a caller to (re)validate).
    checkStateChanged
        Emitted whenever a row's checkbox is toggled (by the user or by
        :meth:`set_flag` auto-unchecking a row).
    """

    filesChanged = Signal()
    checkStateChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self._flags = {}  # path -> (level, message)
        self.itemChanged.connect(self._on_item_changed)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def set_folder(self, dir_path, extensions=None):
        """Populate from every supported file directly inside *dir_path*
        (not recursive -- a batch folder is expected to be flat)."""
        if extensions is None:
            from pyvistra.io import available_input_formats

            extensions = tuple(available_input_formats())
        else:
            extensions = tuple(extensions)

        try:
            names = sorted(os.listdir(dir_path))
        except OSError:
            names = []

        paths = [
            os.path.join(dir_path, name)
            for name in names
            if not name.startswith(".") and name.lower().endswith(extensions)
        ]
        self.add_files(paths, clear=True)

    def add_files(self, paths, clear=False):
        """Populate from an explicit path list (e.g. drag-and-drop)."""
        if clear:
            self.clear()
            self._flags = {}

        self.blockSignals(True)
        for path in paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setToolTip(path)
            self.addItem(item)
        self.blockSignals(False)
        self.filesChanged.emit()

    # ------------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------------

    def set_flag(self, path, level, message=""):
        """Mark *path* with a status *level* (``"ok"``/``"warn"``/``"error"``).

        ``"warn"``/``"error"`` auto-unchecks the row (advisory, not a hard
        block -- the checkbox stays user-togglable) and shows *message* as
        the row's tooltip. ``"ok"`` clears any prior flag/tooltip.
        """
        item = self._find_item(path)
        if item is None:
            return
        previous_level, _ = self._flags.get(path, ("ok", ""))
        self._flags[path] = (level, message)
        item.setIcon(_flag_icon(_FLAG_COLORS.get(level)))
        if level == "ok":
            item.setToolTip(path)
            if previous_level != "ok":
                item.setCheckState(Qt.Checked)
        else:
            item.setToolTip(message or level)
            item.setCheckState(Qt.Unchecked)

    def clear_flags(self):
        self._flags = {}
        for i in range(self.count()):
            item = self.item(i)
            item.setIcon(_flag_icon(None))
            item.setToolTip(item.data(Qt.UserRole))

    def flag_for(self, path):
        """Return ``(level, message)`` for *path*, or ``("ok", "")`` if unset."""
        return self._flags.get(path, ("ok", ""))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def paths(self):
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())]

    def checked_paths(self):
        return [
            self.item(i).data(Qt.UserRole)
            for i in range(self.count())
            if self.item(i).checkState() == Qt.Checked
        ]

    def _find_item(self, path):
        for i in range(self.count()):
            item = self.item(i)
            if item.data(Qt.UserRole) == path:
                return item
        return None

    def _on_item_changed(self, _item):
        self.checkStateChanged.emit()

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event: QDropEvent):
        from pyvistra.io import available_input_formats

        supported_ext = tuple(available_input_formats())
        dropped = [u.toLocalFile() for u in event.mimeData().urls()]

        paths = []
        for f in dropped:
            if os.path.isdir(f):
                try:
                    names = sorted(os.listdir(f))
                except OSError:
                    continue
                paths.extend(
                    os.path.join(f, name)
                    for name in names
                    if not name.startswith(".") and name.lower().endswith(supported_ext)
                )
            elif not os.path.basename(f).startswith(".") and f.lower().endswith(supported_ext):
                paths.append(f)

        if paths:
            self.add_files(sorted(paths), clear=True)
        event.acceptProposedAction()
