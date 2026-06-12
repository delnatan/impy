from qtpy.QtCore import QObject, Signal


class WindowManager(QObject):
    """
    Manages all ImageWindow instances.

    Emits signals when windows are registered/unregistered so that
    other components (like AnnotationManager) can respond immediately.
    """

    # Signals for window lifecycle
    window_registered = Signal(object)    # Emits the window that was registered
    window_unregistered = Signal(object)  # Emits the window that was unregistered
    tool_changed = Signal(str)  # Emits active tool name when changed

    def __init__(self):
        super().__init__()
        self.windows = {}
        self._next_id = 1
        self.active_tool = "pointer"  # Global tool state

    def set_active_tool(self, tool_name):
        """Set the global active tool and emit update signal if it changed."""
        if tool_name == self.active_tool:
            return
        self.active_tool = tool_name
        self.tool_changed.emit(tool_name)

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


def compatible_windows(src):
    """Return open windows whose YX shape matches ``src``'s.

    Used by cross-window shape-copy and by analysis dialogs that need to
    sample matching pixel grids in several windows.

    Excludes ``src`` itself. Only considers windows that expose an
    ``img_data`` attribute with at least 2 spatial dims.
    """
    src_data = getattr(src, "img_data", None)
    if src_data is None or len(src_data.shape) < 2:
        return []
    src_yx = tuple(src_data.shape[-2:])
    out = []
    for w in manager.get_all().values():
        if w is src:
            continue
        data = getattr(w, "img_data", None)
        if data is None or len(data.shape) < 2:
            continue
        if tuple(data.shape[-2:]) == src_yx:
            out.append(w)
    return out


# Global singleton instance
manager = WindowManager()
