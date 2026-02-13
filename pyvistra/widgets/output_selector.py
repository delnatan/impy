from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class ImageOutputSelector(QWidget):
    """Reusable widget for selecting image output destination."""

    output_sent = Signal(object)  # Emits ImageWindow or filepath after send

    # Special item data values
    _NEW_WINDOW = "__new__"
    _SAVE_FILE = "__file__"

    def __init__(self, parent=None, default_title="Result", formats=None):
        """
        Args:
            parent: Parent widget
            default_title: Default title for new windows
            formats: List of supported formats for saving, e.g.
                     [("TIFF", ".tif"), ("PSF Zarr", ".psf.zarr")]
                     If None, defaults to [("TIFF", ".tif")]
        """
        super().__init__(parent)
        self._default_title = default_title
        self._formats = formats or [("TIFF", ".tif")]
        self._manager = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Row 1: Output dropdown
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Output:"))
        self._combo = QComboBox()
        self._combo.currentIndexChanged.connect(self._on_selection_changed)
        row1.addWidget(self._combo, 1)
        layout.addLayout(row1)

        # Row 2: Secondary input area (stacked widget)
        self._secondary = QStackedWidget()

        # Page 0: Empty (for existing window)
        self._secondary.addWidget(QWidget())

        # Page 1: Title field (for new window)
        title_page = QWidget()
        title_layout = QHBoxLayout(title_page)
        title_layout.setContentsMargins(0, 4, 0, 0)
        title_layout.addWidget(QLabel("Title:"))
        self._title_edit = QLineEdit(default_title)
        title_layout.addWidget(self._title_edit)
        self._secondary.addWidget(title_page)

        # Page 2: Format + path (for file)
        file_page = QWidget()
        file_layout = QHBoxLayout(file_page)
        file_layout.setContentsMargins(0, 4, 0, 0)
        file_layout.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        for name, ext in self._formats:
            self._format_combo.addItem(name, ext)
        file_layout.addWidget(self._format_combo)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path (optional, will prompt)")
        file_layout.addWidget(self._path_edit, 1)
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self._browse_btn)
        self._secondary.addWidget(file_page)

        layout.addWidget(self._secondary)

        # Initial population
        from pyvistra.manager import manager

        self._manager = manager
        self._manager.window_registered.connect(self.refresh_windows)
        self._manager.window_unregistered.connect(self.refresh_windows)
        self.refresh_windows()

    def refresh_windows(self):
        """Update list of available windows from WindowManager."""
        previous_selection = self._combo.currentData()

        self._combo.blockSignals(True)
        self._combo.clear()
        windows = self._manager.get_all()

        # Add existing windows
        for wid, win in sorted(windows.items()):
            title = win.windowTitle()
            display = f"[{wid}] {title[:30]}"
            self._combo.addItem(display, wid)

        # Separator if windows exist
        if windows:
            self._combo.insertSeparator(self._combo.count())

        # Special items
        self._combo.addItem("New Window", self._NEW_WINDOW)
        self._combo.addItem("Save to File", self._SAVE_FILE)

        # Restore previous selection if it still exists; otherwise default to New Window
        idx = self._combo.findData(previous_selection)
        if idx < 0:
            idx = self._combo.findData(self._NEW_WINDOW)
        self._combo.setCurrentIndex(idx)

        self._combo.blockSignals(False)
        self._on_selection_changed(self._combo.currentIndex())

    def _on_selection_changed(self, index):
        """Show/hide secondary inputs based on selection."""
        data = self._combo.currentData()
        if data == self._NEW_WINDOW:
            self._secondary.setCurrentIndex(1)  # Title field
        elif data == self._SAVE_FILE:
            self._secondary.setCurrentIndex(2)  # Format + path
        else:
            self._secondary.setCurrentIndex(0)  # Empty

    def get_selection_type(self):
        """Returns 'existing', 'new', or 'file'."""
        data = self._combo.currentData()
        if data == self._NEW_WINDOW:
            return "new"
        elif data == self._SAVE_FILE:
            return "file"
        else:
            return "existing"

    def send(self, data, metadata=None):
        """
        Send data to the selected destination.

        Args:
            data: 5D array-like (ImageBuffer, Numpy5DProxy, ndarray)
            metadata: dict with metadata (replaces window.meta entirely)

        Returns:
            ImageWindow: if sent to new/existing window
            str: filepath if saved to file
            None: if cancelled or error
        """
        selection = self._combo.currentData()
        metadata = metadata or {}

        if selection == self._NEW_WINDOW:
            return self._send_to_new_window(data, metadata)
        elif selection == self._SAVE_FILE:
            return self._send_to_file(data, metadata)
        else:
            return self._send_to_existing(selection, data, metadata)

    def _send_to_existing(self, wid, data, metadata):
        from pyvistra.manager import manager

        window = manager.get(wid)
        if window is None:
            return None
        window.set_data(data, metadata)
        # Update title to reflect new content
        filename = metadata.get("filename", "Result")
        window.setWindowTitle(f"[{wid}] {filename}")
        window.canvas.update()
        self.output_sent.emit(window)
        return window

    def _send_to_new_window(self, data, metadata):
        from pyvistra.ui import ImageWindow

        title = self._title_edit.text() or self._default_title
        metadata["filename"] = title
        viewer = ImageWindow(data, title=title, meta=metadata)
        viewer.show()
        self.output_sent.emit(viewer)
        self.refresh_windows()  # Update dropdown with new window
        return viewer

    def _send_to_file(self, data, metadata):
        from qtpy.QtWidgets import QFileDialog

        from pyvistra.io import save_psf, save_tiff

        filepath = self._path_edit.text()
        ext = self._format_combo.currentData()

        if not filepath:
            # Open file dialog
            filter_str = f"*{ext}"
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Output", f"output{ext}", filter_str
            )
            if not filepath:
                return None

        # Ensure correct extension
        if not filepath.endswith(ext):
            filepath += ext

        # Save based on format
        if ext == ".tif":
            scale = metadata.get("scale", (1.0, 1.0, 1.0))
            save_tiff(filepath, data, scale=scale)
        elif ext == ".psf.zarr":
            save_psf(filepath, data, metadata)
        elif ext == ".ims":
            from pyvistra.io import save_imaris

            save_imaris(filepath, data, metadata)

        self.output_sent.emit(filepath)
        return filepath

    def _browse_file(self):
        from qtpy.QtWidgets import QFileDialog

        ext = self._format_combo.currentData()
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Output", f"output{ext}", f"*{ext}"
        )
        if filepath:
            self._path_edit.setText(filepath)
