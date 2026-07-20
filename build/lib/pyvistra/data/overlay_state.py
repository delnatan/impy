"""Per-channel color/blend-mode/opacity state for the fast thumbnail
grid's Qt-only compositor.

This models something with no vispy/GPU equivalent -- flat-color,
alpha-composited channel overlays (as opposed to additive colormap
blending) -- so it lives separately from ``ChannelDisplayList``
(``data/channel_state.py``) rather than extending it. Same
subscribe/notify idiom; pure data, no Qt, no vispy.
"""

from dataclasses import dataclass, replace

ADDITIVE = "additive"
OVERLAY = "overlay"

# Cycled through for new channels; same hues as visuals/image.py's
# DEFAULT_CHANNEL_COLORMAPS so a fresh folder's fast-mode default look
# matches the vispy path's default look.
DEFAULT_COLORS = [
    "#ff8800",  # orange
    "#00ff44",  # green
    "#00eaff",  # cyan
    "#ff00ea",  # magenta
    "#ffee00",  # yellow
    "#ffffff",  # white
]


@dataclass
class OverlayChannelState:
    """Treat as immutable from outside OverlayStateList; mutations go
    through the list so subscribers get notified."""

    color_hex: str = "#ffffff"
    blend_mode: str = ADDITIVE
    opacity: float = 1.0


class OverlayStateList:
    """Ordered list of :class:`OverlayChannelState`, one per channel.

    Unlike ``ChannelDisplayList`` (which gets wholesale-replaced on a
    channel-count change so renderers can identity-compare), this list
    is resized *in place* via :meth:`resize` -- nothing needs identity
    comparison here, so callers that hold a reference never need it
    re-pushed.
    """

    def __init__(self, n_channels=0):
        self._states = [
            OverlayChannelState(
                color_hex=DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            )
            for i in range(n_channels)
        ]
        self._listeners = []

    def __len__(self):
        return len(self._states)

    def __getitem__(self, idx):
        return self._states[idx]

    def resize(self, n_channels):
        """Grow/shrink in place, preserving existing per-channel state."""
        if n_channels == len(self._states):
            return
        if n_channels < len(self._states):
            self._states = self._states[:n_channels]
        else:
            for i in range(len(self._states), n_channels):
                self._states.append(
                    OverlayChannelState(
                        color_hex=DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                    )
                )
        self._notify(-1, "resize")

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def set_color(self, idx, color_hex):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(self._states[idx], color_hex=color_hex)
            self._notify(idx, "color_hex")

    def set_blend_mode(self, idx, mode):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(self._states[idx], blend_mode=mode)
            self._notify(idx, "blend_mode")

    def set_opacity(self, idx, opacity):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(
                self._states[idx], opacity=float(opacity)
            )
            self._notify(idx, "opacity")

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def subscribe(self, callback):
        """Register ``callback(channel_idx, field)`` (``channel_idx`` is
        ``-1`` for a resize). Returns an unsubscribe function."""
        self._listeners.append(callback)

        def _unsubscribe():
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def _notify(self, idx, field):
        for cb in list(self._listeners):
            try:
                cb(idx, field)
            except Exception:
                pass
