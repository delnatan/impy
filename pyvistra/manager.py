from qtpy.QtCore import QObject, Signal


class WindowManager(QObject):
    """
    Manages all ImageWindow instances.

    Emits signals when windows are registered/unregistered so that
    other components (like ROIManager) can respond immediately.
    """

    # Signals for window lifecycle
    window_registered = Signal(object)    # Emits the window that was registered
    window_unregistered = Signal(object)  # Emits the window that was unregistered

    def __init__(self):
        super().__init__()
        self.windows = {}
        self._next_id = 1
        self.active_tool = "pointer"  # Global tool state
        self.paintbrush_radius = 5  # Global paintbrush brush radius (pixels)

    def register(self, window):
        """Register a window and return its assigned ID."""
        wid = self._next_id
        self.windows[wid] = window
        self._next_id += 1
        self.window_registered.emit(window)
        return wid

    def unregister(self, window):
        """Unregister a window instance."""
        # Find ID by value
        for wid, w in list(self.windows.items()):
            if w == window:
                del self.windows[wid]
                self.window_unregistered.emit(window)
                return

    def get(self, wid):
        """Get window by ID."""
        return self.windows.get(wid)

    def get_all(self):
        """Return dict of all windows."""
        return self.windows


# Global singleton instance
manager = WindowManager()
