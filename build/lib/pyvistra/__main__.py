import sys

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication


def main():
    # Create the Qt Application FIRST (before importing any modules that create QObjects)
    # AA_ShareOpenGLContexts must be set before the QApplication is constructed:
    # it keeps vispy canvases' GL resources valid when a viewer moves between
    # top-level windows (workspace tab float/dock), which otherwise segfaults.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    qt_app = QApplication(sys.argv)

    # Now it's safe to import modules that create QObjects at module level
    from pyvistra.ui import Toolbar
    from pyvistra.theme import DARK_THEME

    # Apply global stylesheet
    qt_app.setStyleSheet(DARK_THEME)

    # Create the floating toolbar
    toolbar = Toolbar()
    toolbar.show()

    # Execute loop
    sys.exit(qt_app.exec_())


if __name__ == "__main__":
    main()
