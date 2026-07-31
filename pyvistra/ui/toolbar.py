"""Main toolbar with drag-and-drop support for pyvistra."""

import os

from natsort import natsort_key
from qtpy.QtCore import QObject, QSize, Qt, QThread, Signal
from qtpy.QtGui import QDragEnterEvent, QDropEvent, QIcon
from qtpy.QtWidgets import (
    QAction,
    QActionGroup,
    QFileDialog,
    QLabel,
    QMainWindow,
    QProgressBar,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from vispy import app

from .._version import get_version
from ..apps.console import console_exists, get_console
from .manager import manager


class _ImageLoadWorker(QObject):
    """Runs load_image() on a worker thread so file-open/metadata-scan
    (and, for compressed TIFFs, a full eager array read) never blocks
    the GUI thread."""

    # img_data, meta, filepath, worker (self -- QObject.sender() is
    # unreliable for moveToThread()'d emitters on a queued connection,
    # so identify the pending-load entry to clean up explicitly)
    finished = Signal(object, object, str, object)
    error = Signal(str, str, object)  # message, filepath, worker

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        from ..io import load_image

        try:
            data, meta = load_image(self.filepath)
        except Exception as e:
            self.error.emit(str(e), self.filepath, self)
            return
        self.finished.emit(data, meta, self.filepath, self)


class Toolbar(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"pyvistra v{get_version()}")
        self.setGeometry(100, 100, 600, 100)  # Wider
        self.setAcceptDrops(True)
        self.open_windows = []
        self._pending_loads = {}  # worker -> thread, keeps them alive

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
        self.tools.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.tools.setIconSize(QSize(18, 18))
        self.tools.setStyleSheet(
            "QToolButton { padding: 3px; border-radius: 4px; }"
            "QToolButton:checked { background: rgba(45, 106, 79, 0.8); }"
            "QToolButton:hover { background: rgba(255, 255, 255, 0.2); }"
        )

        self._tool_labels = {
            "pointer": "Pan/Zoom",
            "rect": "Rectangle",
            "circle": "Circle",
            "line": "Line",
            "polyline": "Polyline",
            "point": "Point",
            "brush": "Brush",
            "eraser": "Eraser",
        }

        self.tool_actions = {}
        self.act_pointer = self._create_tool_action(
            "pointer",
            self._load_tool_icon(
                "pointer", self.style().standardIcon(QStyle.SP_ArrowUp)
            ),
            "Pan/Zoom",
            "Ctrl+1",
        )
        self.act_rect = self._create_tool_action(
            "rect",
            self._load_tool_icon(
                "rect",
                self.style().standardIcon(QStyle.SP_TitleBarNormalButton),
            ),
            "Rectangle",
            "Ctrl+2",
        )
        self.act_circle = self._create_tool_action(
            "circle",
            self._load_tool_icon(
                "circle",
                self.style().standardIcon(QStyle.SP_BrowserReload),
            ),
            "Circle",
            "Ctrl+3",
        )
        self.act_line = self._create_tool_action(
            "line",
            self._load_tool_icon(
                "line",
                self.style().standardIcon(QStyle.SP_LineEditClearButton),
            ),
            "Line",
            "Ctrl+4",
        )
        self.act_polyline = self._create_tool_action(
            "polyline",
            self._load_tool_icon(
                "polyline",
                self.style().standardIcon(QStyle.SP_ArrowForward),
            ),
            "Polyline (click vertices, right-click to finish)",
            "Ctrl+5",
        )
        self.act_point = self._create_tool_action(
            "point",
            self._load_tool_icon(
                "point",
                self.style().standardIcon(QStyle.SP_MediaStop),
            ),
            "Point (click to drop, Alt+click to remove)",
            "Ctrl+6",
        )
        self.act_brush = self._create_tool_action(
            "brush",
            self._load_tool_icon(
                "brush",
                self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            ),
            "Brush",
            "Ctrl+7",
        )
        self.act_eraser = self._create_tool_action(
            "eraser",
            self._load_tool_icon(
                "eraser",
                self.style().standardIcon(QStyle.SP_DialogDiscardButton),
            ),
            "Eraser",
            "Ctrl+8",
        )

        self.tools.addAction(self.act_pointer)
        self.tools.addAction(self.act_rect)
        self.tools.addAction(self.act_circle)
        self.tools.addAction(self.act_line)
        self.tools.addAction(self.act_polyline)
        self.tools.addAction(self.act_point)

        # Separator before painting tools
        self.tools.addSeparator()
        self.tools.addAction(self.act_brush)
        self.tools.addAction(self.act_eraser)

        # Layer Manager Button (unified shapes/points/tracks/labels panel)
        self.tools.addSeparator()
        self.act_layer_mgr = QAction(
            self._load_tool_icon(
                "layers",
                self.style().standardIcon(QStyle.SP_FileDialogInfoView),
            ),
            "",
            self,
        )
        self.act_layer_mgr.setToolTip("Layers")
        self.act_layer_mgr.triggered.connect(self.show_layer_manager)
        self.tools.addAction(self.act_layer_mgr)

        # Python Console Button
        self.act_console = QAction(
            self._load_tool_icon(
                "console",
                self.style().standardIcon(QStyle.SP_FileDialogListView),
            ),
            "",
            self,
        )
        self.act_console.setToolTip("Console")
        self.act_console.triggered.connect(self.show_console)
        self.tools.addAction(self.act_console)

        self.tools.addSeparator()
        self.mode_indicator = QLabel("Mode: Pan/Zoom")
        self.mode_indicator.setStyleSheet(
            "QLabel { color: #d2d2d2; font-size: 11px; padding: 0 4px; }"
        )
        self.tools.addWidget(self.mode_indicator)

        # Busy indicator: shown only while one or more _ImageLoadWorker
        # threads are in flight (see spawn_viewer/_pending_loads below).
        # Lives in the status bar rather than the QToolBar -- the toolbar
        # is narrow and already crowded with tool icons, so extra widgets
        # added to it can silently land behind the overflow chevron
        # instead of actually being seen. Indeterminate (min==max==0)
        # since there's no meaningful progress fraction for a single
        # load_image() call.
        self.loading_label = QLabel("")
        self.loading_label.setStyleSheet(
            "QLabel { color: #d2d2d2; font-size: 11px; padding: 0 4px; }"
        )
        self.loading_label.setVisible(False)
        self.statusBar().addWidget(self.loading_label)

        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)
        self.loading_bar.setMaximumWidth(80)
        self.loading_bar.setMaximumHeight(14)
        self.loading_bar.setTextVisible(False)
        self.loading_bar.setVisible(False)
        self.statusBar().addWidget(self.loading_bar)

        # Group for exclusive tool selection
        self.group = QActionGroup(self)
        self.group.setExclusive(True)
        self.group.addAction(self.act_pointer)
        self.group.addAction(self.act_rect)
        self.group.addAction(self.act_circle)
        self.group.addAction(self.act_line)
        self.group.addAction(self.act_polyline)
        self.group.addAction(self.act_point)
        self.group.addAction(self.act_brush)
        self.group.addAction(self.act_eraser)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        manager.tool_changed.connect(self._on_tool_changed)
        self._on_tool_changed(manager.active_tool)

    def _create_tool_action(self, tool_name, icon, tooltip, shortcut):
        action = QAction(icon, "", self)
        action.setCheckable(True)
        action.setShortcut(shortcut)
        action.setShortcutContext(Qt.ApplicationShortcut)
        action.setToolTip(f"{tooltip} ({shortcut})")
        action.triggered.connect(lambda: self.set_tool(tool_name))
        self.tool_actions[tool_name] = action
        return action

    def _load_tool_icon(self, name, fallback_icon):
        """Load custom toolbar icon if available, otherwise use fallback."""
        icon_dir = os.path.join(os.path.dirname(__file__), "..", "_resources", "icons")
        for ext in ("svg", "png"):
            path = os.path.join(icon_dir, f"{name}.{ext}")
            if os.path.exists(path):
                icon = QIcon(path)
                if not icon.isNull():
                    return icon
        return fallback_icon

    def set_tool(self, tool_name):
        manager.set_active_tool(tool_name)
        # Update cursor in all windows
        for w in manager.get_all().values():
            w.update_cursor()

    def _on_tool_changed(self, tool_name):
        action = self.tool_actions.get(tool_name)
        if action is not None and not action.isChecked():
            action.setChecked(True)
        label = self._tool_labels.get(tool_name, tool_name.title())
        self.mode_indicator.setText(f"Mode: {label}")

    def show_layer_manager(self):
        from .layer_manager import show_layer_manager

        show_layer_manager()

    def show_console(self):
        console = get_console()
        console.show()
        console.raise_()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        # Collect supported image files, driven by the same input-format
        # registry load_image() itself dispatches through (see io.py) --
        # matched via suffix rather than splitext so a plugin-registered
        # multi-dot extension (e.g. ".psf.h5") works without special-casing
        # here.
        from ..io import available_input_formats

        supported_ext = tuple(available_input_formats())

        def is_supported(name):
            return name.lower().endswith(supported_ext)

        image_files = []

        for f in files:
            # Check for .zarr directories (general Zarr arrays / OME-Zarr).
            if f.endswith(".zarr/") and os.path.isdir(f):
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
                        if is_supported(name):
                            image_files.append(os.path.join(root, name))
            elif not os.path.basename(f).startswith(".") and is_supported(f):
                image_files.append(f)

        # Sort by filename
        image_files.sort(key=natsort_key)

        if len(image_files) > 1:
            # Multiple files -> TiledViewer
            from ..viewers import TiledViewer

            viewer = TiledViewer(image_files)
            from .workspace import present_window

            present_window(viewer)
            self.open_windows.append(viewer)
        elif len(image_files) == 1:
            # Single file -> regular ImageWindow
            self.spawn_viewer(image_files[0])

    def open_file_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open file", ".")
        if fname:
            self.spawn_viewer(fname)

    def closeEvent(self, event):
        # Signal Line Profile dialog to stop processing updates
        from ..widgets import (
            get_line_profile_dialog,
            line_profile_dialog_exists,
        )

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
        worker = _ImageLoadWorker(filepath)
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_image_loaded)
        worker.error.connect(self._on_image_load_error)

        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._pending_loads[worker] = thread
        self._update_loading_indicator(os.path.basename(filepath))
        thread.start()

    def _update_loading_indicator(self, filepath_hint=None):
        """Show/hide the toolbar busy indicator based on how many
        _ImageLoadWorker threads are still in flight."""
        n = len(self._pending_loads)
        busy = n > 0
        if busy:
            if n == 1 and filepath_hint:
                text = f"Loading {filepath_hint}..."
            else:
                text = f"Loading {n} images..."
            self.loading_label.setText(text)
        self.loading_label.setVisible(busy)
        self.loading_bar.setVisible(busy)

    def _on_image_loaded(self, data, meta, filepath, worker):
        from .window import ImageWindow
        from .workspace import present_window

        self._pending_loads.pop(worker, None)
        self._update_loading_indicator()
        try:
            viewer = ImageWindow(data, meta=meta, filepath=filepath)
            present_window(viewer)
            self.open_windows.append(viewer)
        except Exception as e:
            print(f"Error opening {filepath}: {e}")

    def _on_image_load_error(self, message, filepath, worker):
        self._pending_loads.pop(worker, None)
        self._update_loading_indicator()
        print(f"Error opening {filepath}: {message}")
