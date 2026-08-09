"""
Interactive Python console for pyvistra.

Provides an embedded Python REPL with access to the application state,
similar to napari's built-in console.
"""

import code
import html
import sys
import traceback
from io import StringIO

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import (
    QFont,
    QFontDatabase,
    QFontMetrics,
    QKeyEvent,
    QTextCursor,
)
from qtpy.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import colors as tokens


class OutputCapture:
    """Context manager to capture stdout/stderr."""

    def __init__(self):
        self.stdout = StringIO()
        self.stderr = StringIO()
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self

    def __exit__(self, *args):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr

    def get_output(self):
        return self.stdout.getvalue()

    def get_error(self):
        return self.stderr.getvalue()


class ConsoleInput(QPlainTextEdit):
    """Input widget for the Python console with special key handling."""

    execute_requested = Signal(str)
    history_up = Signal()
    history_down = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText(
            ">>> Enter Python code (Shift+Enter for newline, Enter to execute)"
        )
        self._setup_font()

    def _setup_font(self):
        # Get system's default fixed-width font
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(10)
        self.setFont(font)

        # Set tab width to 4 spaces
        metrics = QFontMetrics(font)
        tab_width = metrics.horizontalAdvance(" ") * 4
        self.setTabStopDistance(tab_width)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if event.modifiers() & Qt.ShiftModifier:
                # Shift+Enter: insert newline
                super().keyPressEvent(event)
            else:
                # Enter: execute code
                code = self.toPlainText().strip()
                if code:
                    self.execute_requested.emit(code)
        elif event.key() == Qt.Key_Up and not self.toPlainText():
            # Up arrow on empty input: navigate history
            self.history_up.emit()
        elif event.key() == Qt.Key_Down and not self.toPlainText():
            # Down arrow on empty input: navigate history
            self.history_down.emit()
        else:
            super().keyPressEvent(event)


class ConsoleOutput(QPlainTextEdit):
    """Output widget for the Python console."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._setup_font()
        self._setup_style()

    def _setup_font(self):
        # Get system's default fixed-width font
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(10)
        self.setFont(font)

        # Set tab width to 4 spaces
        metrics = QFontMetrics(font)
        tab_width = metrics.horizontalAdvance(" ") * 4
        self.setTabStopDistance(tab_width)

    def _setup_style(self):
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {tokens.BG_ELEVATED};
                color: {tokens.TEXT_PRIMARY};
                border: none;
            }}
        """)

    def append_text(self, text, color=None):
        """Append text to the output with optional color."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)

        if color:
            # Insert colored text using HTML (escape special chars like < > &)
            escaped = html.escape(text).replace("\n", "<br>")
            html_str = f'<span style="color: {color};">{escaped}</span>'
            cursor.insertHtml(html_str)
            cursor.insertText("\n")
        else:
            cursor.insertText(text + "\n")

        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def append_input(self, code):
        """Append input code with prompt styling."""
        lines = code.split("\n")
        for i, line in enumerate(lines):
            prefix = ">>> " if i == 0 else "... "
            self.append_text(prefix + line, tokens.ACCENT)

    def append_output(self, text):
        """Append output text."""
        if text.strip():
            self.append_text(text.rstrip())

    def append_error(self, text):
        """Append error text in red."""
        if text.strip():
            self.append_text(text.rstrip(), tokens.DANGER)

    def append_result(self, text):
        """Append result text in green."""
        if text.strip():
            self.append_text(text.rstrip(), tokens.SUCCESS)


class PythonConsole(QWidget):
    """
    Interactive Python console widget for pyvistra.

    This widget uses a hide/show pattern rather than create/destroy.
    Once instantiated, it persists for the lifetime of the application.
    Calling close() will hide the widget rather than destroying it.

    Provides a REPL with access to application state including:
    - manager: WindowManager singleton
    - windows: dict of all open ImageWindow instances
    - np: numpy

    Usage:
        console = PythonConsole()
        console.show()
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Python Console")
        self.resize(700, 500)
        self._is_shutting_down = False

        # Command history
        self._history = []
        self._history_index = 0

        # Setup UI
        self._setup_ui()

        # Setup interpreter with namespace
        self._setup_interpreter()

        # Print welcome message
        self._print_welcome()

    def closeEvent(self, event):
        """Override close to hide instead of destroy.

        This implements the hide/show pattern for singleton widgets.
        """
        if self._is_shutting_down:
            super().closeEvent(event)
        else:
            event.ignore()
            self.hide()

    def cleanup(self):
        """Prepare for application shutdown."""
        self._is_shutting_down = True

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Header
        header = QLabel(
            "Python Console - Access pyvistra objects interactively"
        )
        header.setStyleSheet(f"color: {tokens.TEXT_FAINT}; font-size: 11px;")
        layout.addWidget(header)

        # Splitter for output/input
        splitter = QSplitter(Qt.Vertical)

        # Output area (larger)
        self.output = ConsoleOutput()
        splitter.addWidget(self.output)

        # Input area (smaller)
        self.input = ConsoleInput()
        self.input.setMaximumHeight(100)
        splitter.addWidget(self.input)

        # Set splitter sizes (80% output, 20% input)
        splitter.setSizes([400, 100])

        layout.addWidget(splitter)

        # Connect signals
        self.input.execute_requested.connect(self._execute)
        self.input.history_up.connect(self._history_prev)
        self.input.history_down.connect(self._history_next)

    def _setup_interpreter(self):
        """Setup the Python interpreter with pre-populated namespace."""
        import numpy as np

        # Build namespace with useful objects
        self.namespace = {
            "__name__": "__console__",
            "__doc__": None,
            "np": np,
            "numpy": np,
        }

        # Add pyvistra objects
        self._update_namespace()

        # Create interpreter
        self.interpreter = code.InteractiveInterpreter(self.namespace)

    def _update_namespace(self):
        """Update namespace with current pyvistra state."""
        from ..ui.manager import manager

        self.namespace["manager"] = manager
        self.namespace["windows"] = manager.get_all()

        # Add helper function to get current/active window
        def get_active_window():
            """Get the currently active ImageWindow, or None."""
            if manager.active_window is not None:
                return manager.active_window
            windows = manager.get_all()
            if windows:
                return list(windows.values())[0]
            return None

        self.namespace["active_window"] = get_active_window
        self.namespace["aw"] = get_active_window  # Short alias

        # Add imshow for convenience
        from ..ui import imshow

        self.namespace["imshow"] = imshow

        # Add io functions
        from ..io import load_image, save_tiff

        self.namespace["load_image"] = load_image
        self.namespace["save_tiff"] = save_tiff

        # Add reload helper for hot-reloading modules during prototyping
        import importlib

        def reload(module_or_name):
            """
            Reload a module by name or reference.

            Usage:
                reload('my_module')     # Reload from pyvistra.* or sys.path
                reload('mymodule')      # Reload mymodule from cwd
                reload(some_module)     # Reload a module object

            After reload, updated functions are available immediately.
            """
            if isinstance(module_or_name, str):
                name = module_or_name
                # Try pyvistra submodule first
                try:
                    mod = importlib.import_module(f".{name}", package="pyvistra")
                except ImportError:
                    # Try as top-level module (e.g., from cwd)
                    mod = importlib.import_module(name)
            else:
                mod = module_or_name

            reloaded = importlib.reload(mod)
            # Update namespace with module contents
            self.namespace[reloaded.__name__.split('.')[-1]] = reloaded
            print(f"Reloaded: {reloaded.__name__}")
            return reloaded

        self.namespace["reload"] = reload

        # lab scratch module removed; users can import their own modules

        # Add clear function (closure over self)
        def clear():
            """Clear the console output."""
            self.clear()

        self.namespace["clear"] = clear

    def _print_welcome(self):
        """Print welcome message with available objects."""
        welcome = """pyvistra Python Console
========================
Available objects:
  manager     - WindowManager (tracks all windows)
  windows     - dict of all ImageWindow instances
  aw()        - Get active window (shortcut for active_window())
  np          - numpy
  imshow()    - Display array as image
  load_image()- Load image file
  save_tiff() - Save as TIFF

Prototyping:
  reload()    - Hot-reload a module: reload('my_module')
  clear()     - Clear console output

Example:
  >>> w = aw()          # Get active window
  >>> data = w.img_data # Access 5D data (T,Z,C,Y,X)
  >>> w.layers          # Shapes/points/tracks/labels layers
  >>> reload('my_module')
"""
        self.output.append_text(welcome, tokens.TEXT_FAINT)

    def _execute(self, code_str):
        """Execute Python code and display results."""
        # Add to history
        if code_str and (not self._history or self._history[-1] != code_str):
            self._history.append(code_str)
        self._history_index = len(self._history)

        # Clear input
        self.input.clear()

        # Update namespace with current state
        self._update_namespace()

        # Show input in output
        self.output.append_input(code_str)

        # Capture output
        with OutputCapture() as capture:
            try:
                # Try to compile as expression first (for result display)
                try:
                    compiled = compile(code_str, "<console>", "eval")
                    result = eval(compiled, self.namespace)
                    if result is not None:
                        self.output.append_result(repr(result))
                except SyntaxError:
                    # Not an expression, execute as statements
                    compiled = compile(code_str, "<console>", "exec")
                    exec(compiled, self.namespace)
            except Exception:
                # Show traceback
                tb = traceback.format_exc()
                self.output.append_error(tb)

        # Show captured output
        stdout = capture.get_output()
        stderr = capture.get_error()

        if stdout:
            self.output.append_output(stdout)
        if stderr:
            self.output.append_error(stderr)

    def _history_prev(self):
        """Navigate to previous command in history."""
        if self._history and self._history_index > 0:
            self._history_index -= 1
            self.input.setPlainText(self._history[self._history_index])

    def _history_next(self):
        """Navigate to next command in history."""
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self.input.setPlainText(self._history[self._history_index])
        else:
            self._history_index = len(self._history)
            self.input.clear()

    def run_code(self, code_str):
        """
        Programmatically execute code in the console.

        Useful for running startup scripts or macros.
        """
        self._execute(code_str)

    def clear(self):
        """Clear the output area."""
        self.output.clear()
        self._print_welcome()


# Singleton pattern for console
_console_instance = None


def get_console():
    """Get or create the Python console singleton."""
    global _console_instance
    if _console_instance is None:
        _console_instance = PythonConsole()
    return _console_instance


def console_exists():
    """Check if console has been created without creating it."""
    return _console_instance is not None
