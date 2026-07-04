"""Main toolbar with drag-and-drop support for pyvistra."""

import os

from natsort import natsort_key
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QDragEnterEvent, QDropEvent, QIcon
from qtpy.QtWidgets import (
    QAction,
    QActionGroup,
    QFileDialog,
    QLabel,
    QMainWindow,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from vispy import app

from .._version import get_version
from ..apps.console import console_exists, get_console
from .manager import manager


class Toolbar(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"pyvistra v{get_version()}")
        self.setGeometry(100, 100, 600, 100)  # Wider
        self.setAcceptDrops(True)
        self.open_windows = []
        self._psf_dialog = None  # Lazy singleton for PSF dialog
        self._pupil_dialog = None  # Lazy singleton for Pupil dialog

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

        compute_psf_action = QAction("Compute PSF...", self)
        compute_psf_action.triggered.connect(self._show_psf_dialog)
        file_menu.addAction(compute_psf_action)

        compute_pupil_action = QAction("Compute Pupil...", self)
        compute_pupil_action.triggered.connect(self._show_pupil_dialog)
        file_menu.addAction(compute_pupil_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        analyze_menu = menubar.addMenu("Analyze")
        gel_action = QAction("Gel Analyzer...", self)
        gel_action.triggered.connect(self.show_gel_analyzer)
        analyze_menu.addAction(gel_action)

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

    def show_gel_analyzer(self):
        from ..apps.gel_analyzer import show_gel_analyzer

        show_gel_analyzer(manager)

    def show_console(self):
        console = get_console()
        console.show()
        console.raise_()

    def _show_psf_dialog(self):
        """Show the PSF computation dialog."""
        from ..widgets import PSFComputeDialog

        if self._psf_dialog is None:
            self._psf_dialog = PSFComputeDialog(parent=self)
        self._psf_dialog.show()
        self._psf_dialog.raise_()

    def _show_pupil_dialog(self):
        """Show the Pupil computation dialog."""
        from ..widgets import PupilComputeDialog

        if self._pupil_dialog is None:
            self._pupil_dialog = PupilComputeDialog(parent=self)
        self._pupil_dialog.show()
        self._pupil_dialog.raise_()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        # Collect supported image files.
        #
        # Single extensions (matched via splitext) and compound suffixes
        # (matched via name.endswith) are kept separate because deconlib
        # artifacts use two-dot suffixes (``.psf.h5``, ``.pupil.h5``) that
        # splitext would collapse to a generic ``.h5``.
        supported_ext = {
            ".ims",
            ".czi",
            ".nd2",
            ".tif",
            ".tiff",
            ".png",
            ".jpg",
            ".jpeg",
        }
        supported_suffixes = (".psf.h5", ".pupil.h5")

        def is_supported(name):
            lower = name.lower()
            if lower.endswith(supported_suffixes):
                return True
            return os.path.splitext(lower)[1] in supported_ext

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
        from .window import ImageWindow
        from .workspace import present_window

        try:
            viewer = ImageWindow(filepath)
            present_window(viewer)
            self.open_windows.append(viewer)
        except Exception as e:
            print(f"Error opening {filepath}: {e}")
