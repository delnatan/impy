"""Reusable input-source widget for processor dialogs.

Lets the user pick the input for a processor from either an open window
or a file on disk. Companion to :class:`~.output_selector.ImageOutputSelector`
(destination side) and :class:`~.region_selector.RegionSelector` (spatial
restriction) -- together they cover "pick an input image, optionally crop
it, run something, send the result somewhere", the shape of most plugin
processing dialogs (see pyvistra's ``pyvistra.plugins`` entry-points
protocol).

Usage
-----
.. code-block:: python

    selector = SourceSelector()
    layout.addWidget(selector)
    selector.set_default_window(my_window)
    # later, when running the processor:
    data, meta = selector.get_source()   # Readable5D, dict
    window = selector.selected_window()  # ImageWindow or None (file source)
"""

from __future__ import annotations

from typing import Optional, Tuple

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class SourceSelector(QGroupBox):
    """Group box exposing a window-or-file input source.

    Signals
    -------
    source_changed
        Emitted whenever the selected window changes or a file is loaded.
    """

    source_changed = Signal()

    _LOAD_FILE = "__file__"

    def __init__(self, parent=None, *, title: str = "Source"):
        super().__init__(title, parent)

        self._file_data = None  # Readable5D loaded from disk (refcounted)
        self._file_meta = None
        self._file_label = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        row = QHBoxLayout()
        row.addWidget(QLabel("Window:"))
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        row.addWidget(self.combo, 1)
        self.load_file_btn = QPushButton("Load File...")
        self.load_file_btn.clicked.connect(self._browse_file)
        row.addWidget(self.load_file_btn)
        outer.addLayout(row)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #888; font-size: 10px;")
        outer.addWidget(self.info_label)

        from pyvistra.ui.manager import manager

        self._manager = manager
        self._manager.window_registered.connect(self.refresh_windows)
        self._manager.window_unregistered.connect(self.refresh_windows)
        self.refresh_windows()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_default_window(self, window) -> None:
        """Preselect *window* in the combo (if it's registered)."""
        for i in range(self.combo.count()):
            if self._manager.get(self.combo.itemData(i)) is window:
                self.combo.setCurrentIndex(i)
                return

    def get_source(self) -> Tuple[object, dict]:
        """Return ``(data, meta)`` for the current selection.

        ``data`` is whatever the source window/file exposes as
        ``img_data`` -- callers that hold this across a thread hand-off
        should ``acquire()``/``release()`` it themselves (every 5D
        proxy/buffer supports this via ``RefCountMixin``).
        """
        data = self.combo.currentData()
        if data == self._LOAD_FILE:
            return self._file_data, self._file_meta or {}
        window = self._manager.get(data)
        if window is None:
            return None, {}
        return window.img_data, window.meta

    def selected_window(self):
        """Return the selected ``ImageWindow``, or ``None`` for a file source."""
        data = self.combo.currentData()
        if data == self._LOAD_FILE:
            return None
        return self._manager.get(data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def refresh_windows(self) -> None:
        previous = self.combo.currentData()

        self.combo.blockSignals(True)
        self.combo.clear()
        windows = self._manager.get_all()
        for wid, win in sorted(windows.items()):
            title = win.windowTitle()
            self.combo.addItem(f"[{wid}] {title[:30]}", wid)

        if self._file_data is not None:
            if windows:
                self.combo.insertSeparator(self.combo.count())
            self.combo.addItem(f"File: {self._file_label}", self._LOAD_FILE)

        idx = self.combo.findData(previous) if previous is not None else -1
        if idx < 0:
            idx = 0
        if self.combo.count():
            self.combo.setCurrentIndex(idx)

        self.combo.blockSignals(False)
        self._update_info()

    def _on_combo_changed(self, _index: int) -> None:
        self._update_info()
        self.source_changed.emit()

    def _update_info(self) -> None:
        data, meta = self.get_source()
        if data is None:
            self.info_label.setText("No source selected.")
            return
        shape = tuple(getattr(data, "shape", ()))
        dtype = getattr(data, "dtype", "")
        self.info_label.setText(f"shape={shape}  dtype={dtype}")

    def _browse_file(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Source Image")
        if not filepath:
            return

        from pyvistra.io import load_image

        if self._file_data is not None and hasattr(self._file_data, "release"):
            try:
                self._file_data.release()
            except Exception:
                pass

        data, meta = load_image(filepath)
        self._file_data = data
        self._file_meta = meta
        self._file_label = filepath.rsplit("/", 1)[-1]
        self.refresh_windows()

        idx = self.combo.findData(self._LOAD_FILE)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def closeEvent(self, event):
        if self._file_data is not None and hasattr(self._file_data, "release"):
            try:
                self._file_data.release()
            except Exception:
                pass
            self._file_data = None
        super().closeEvent(event)
