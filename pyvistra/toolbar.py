"""Main toolbar with drag-and-drop support for pyvistra."""

import os

from natsort import natsort_key
from qtpy.QtCore import Qt
from qtpy.QtGui import QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import (
    QAction,
    QActionGroup,
    QFileDialog,
    QLabel,
    QMainWindow,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from vispy import app

from .annotation_manager import annotation_manager_exists, get_annotation_manager
from .console import console_exists, get_console
from .manager import manager


class Toolbar(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyvistra v0.1 (prototype)")
        self.setGeometry(100, 100, 600, 100)  # Wider
        self.setAcceptDrops(True)
        self.open_windows = []
        self._psf_dialog = None  # Lazy singleton for PSF dialog

        # Central Widget with Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Label
        self.label = QLabel("Drag & Drop Images")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Tool Bar (Actual QToolBar)
        self.tools = QToolBar("Tools")
        self.addToolBar(self.tools)

        # Actions
        self.act_pointer = QAction("Pointer", self)
        self.act_pointer.setCheckable(True)
        self.act_pointer.setChecked(True)
        self.act_pointer.triggered.connect(lambda: self.set_tool("pointer"))

        self.act_coord = QAction("Coordinate", self)
        self.act_coord.setCheckable(True)
        self.act_coord.triggered.connect(lambda: self.set_tool("coordinate"))

        self.act_rect = QAction("Rectangle", self)
        self.act_rect.setCheckable(True)
        self.act_rect.triggered.connect(lambda: self.set_tool("rect"))

        self.act_circle = QAction("Circle", self)
        self.act_circle.setCheckable(True)
        self.act_circle.triggered.connect(lambda: self.set_tool("circle"))

        self.act_line = QAction("Line", self)
        self.act_line.setCheckable(True)
        self.act_line.triggered.connect(lambda: self.set_tool("line"))

        # Painting tools
        self.act_brush = QAction("Brush", self)
        self.act_brush.setCheckable(True)
        self.act_brush.triggered.connect(lambda: self.set_tool("brush"))

        self.act_eraser = QAction("Eraser", self)
        self.act_eraser.setCheckable(True)
        self.act_eraser.triggered.connect(lambda: self.set_tool("eraser"))

        self.tools.addAction(self.act_pointer)
        self.tools.addAction(self.act_coord)
        self.tools.addAction(self.act_rect)
        self.tools.addAction(self.act_circle)
        self.tools.addAction(self.act_line)

        # Separator before painting tools
        self.tools.addSeparator()
        self.tools.addAction(self.act_brush)
        self.tools.addAction(self.act_eraser)

        # Manager Button
        self.tools.addSeparator()
        self.act_annotation_mgr = QAction("Annotations", self)
        self.act_annotation_mgr.triggered.connect(self.show_annotation_manager)
        self.tools.addAction(self.act_annotation_mgr)

        # Python Console Button
        self.act_console = QAction("Console", self)
        self.act_console.triggered.connect(self.show_console)
        self.tools.addAction(self.act_console)

        # Group for exclusive tool selection
        group = QActionGroup(self)
        group.addAction(self.act_pointer)
        group.addAction(self.act_coord)
        group.addAction(self.act_rect)
        group.addAction(self.act_circle)
        group.addAction(self.act_line)
        group.addAction(self.act_brush)
        group.addAction(self.act_eraser)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        compute_psf_action = QAction("Compute PSF...", self)
        compute_psf_action.triggered.connect(self._show_psf_dialog)
        file_menu.addAction(compute_psf_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def set_tool(self, tool_name):
        manager.active_tool = tool_name
        # Update cursor in all windows
        for w in manager.get_all().values():
            w.update_cursor()

    def show_annotation_manager(self):
        mgr = get_annotation_manager()
        mgr.show()
        mgr.raise_()

    def show_console(self):
        console = get_console()
        console.show()
        console.raise_()

    def _show_psf_dialog(self):
        """Show the PSF computation dialog."""
        from .widgets import PSFComputeDialog

        if self._psf_dialog is None:
            self._psf_dialog = PSFComputeDialog(parent=self)
        self._psf_dialog.show()
        self._psf_dialog.raise_()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        # Collect supported image files
        supported_ext = {
            ".ims",
            ".czi",
            ".tif",
            ".tiff",
            ".png",
            ".jpg",
            ".jpeg",
        }
        image_files = []

        for f in files:
            # Check for .zarr directories (including .psf.zarr)
            if (
                f.endswith(".zarr/") or f.endswith(".psf.zarr/")
            ) and os.path.isdir(f):
                image_files.append(f)
            elif os.path.isdir(f):
                # Folder: collect all images recursively
                for root, dirs, names in os.walk(f):
                    # Skip hidden directories (e.g. .Spotlight-V100, .fseventsd)
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    # Check for .zarr directories (skip descending into them)
                    zarr_dirs = [d for d in dirs if d.endswith(".zarr/")]
                    for d in zarr_dirs:
                        image_files.append(os.path.join(root, d))
                        dirs.remove(d)  # Don't descend into zarr directories
                    for name in names:
                        if name.startswith("."):
                            continue
                        if os.path.splitext(name)[1].lower() in supported_ext:
                            image_files.append(os.path.join(root, name))
            elif (
                os.path.splitext(f)[1].lower() in supported_ext
                and not os.path.basename(f).startswith(".")
            ):
                image_files.append(f)

        # Sort by filename
        image_files.sort(key=natsort_key)

        if len(image_files) > 1:
            # Multiple files -> TiledViewer
            from .tiled_viewer import TiledViewer

            viewer = TiledViewer(image_files)
            viewer.show()
            self.open_windows.append(viewer)
        elif len(image_files) == 1:
            # Single file -> regular ImageWindow
            self.spawn_viewer(image_files[0])

    def open_file_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open file", ".")
        if fname:
            self.spawn_viewer(fname)

    def closeEvent(self, event):
        # Signal Annotation Manager to stop processing updates BEFORE closing windows
        if annotation_manager_exists():
            try:
                mgr = get_annotation_manager()
                mgr.cleanup()
            except Exception:
                pass

        # Signal Line Profile dialog to stop processing updates
        from .widgets import line_profile_dialog_exists, get_line_profile_dialog

        if line_profile_dialog_exists():
            try:
                dialog = get_line_profile_dialog()
                dialog.cleanup()
            except Exception:
                pass

        # Signal Console to stop processing updates
        if console_exists():
            try:
                console = get_console()
                if hasattr(console, "cleanup"):
                    console.cleanup()
            except Exception:
                pass

        # Now close all managed windows - their signals won't trigger ROI Manager updates
        windows = list(manager.get_all().values())
        for w in windows:
            w.close()

        # Quit Vispy's app to ensure clean OpenGL context shutdown
        try:
            app.quit()
        except Exception:
            pass

        super().closeEvent(event)

    def spawn_viewer(self, filepath):
        from .ui import ImageWindow

        try:
            viewer = ImageWindow(filepath)
            viewer.show()
            self.open_windows.append(viewer)
        except Exception as e:
            print(f"Error opening {filepath}: {e}")
