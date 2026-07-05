"""Per-channel display state for image renderers.

A single ``ChannelDisplayList`` owns the contrast / gamma / colormap /
visibility for every channel of a multi-channel image. Renderers
(``CompositeImageVisual``, ``VolumeRendererProxy``, ``TiledViewer``,
``OrthoViewer``) delegate to it instead of carrying their own parallel
dicts. UI widgets subscribe to it via ``subscribe(callback)`` and
refresh on change — no more polling from ``update_view``.

This module is pure data: no Qt, no vispy. The display color for a
channel is derived from its colormap name via the colormap registry.
"""

from dataclasses import dataclass, field, replace

from .. import colormaps as _colormaps


@dataclass
class ChannelDisplayState:
    """Per-channel display parameters.

    Treat as immutable from outside ``ChannelDisplayList``; mutations go
    through the list so listeners get notified.
    """

    clim: tuple = (0.0, 1.0)
    gamma: float = 1.0
    colormap_name: str = "White"
    visible: bool = True

    def display_color(self):
        """Hex/CSS color string for UI swatches.

        Derived from the colormap. Multi-color colormaps (viridis, etc.)
        return ``None`` and callers should fall back to a neutral color.
        """
        _, color = _colormaps.get(self.colormap_name)
        return color


class ChannelDisplayList:
    """Ordered list of :class:`ChannelDisplayState`, one per channel.

    Mutations go through ``set_clim``/``set_gamma``/``set_colormap_name``/
    ``set_visible``. Each fires a ``(channel_idx, field)`` notification to
    every subscriber. ``field`` is one of ``"clim"``, ``"gamma"``,
    ``"colormap_name"``, ``"visible"``.

    Callbacks run on the caller's thread (no marshalling). Qt consumers
    that subscribe from the GUI thread can rely on synchronous delivery
    since renderer mutations originate on the GUI thread.
    """

    def __init__(self, n_channels, defaults=None):
        self._states = [
            replace(defaults) if defaults is not None else ChannelDisplayState()
            for _ in range(n_channels)
        ]
        self._listeners = []

    @classmethod
    def from_states(cls, states):
        """Build a new list by copying an iterable of
        :class:`ChannelDisplayState` (e.g. another window's ``display``).

        Used to hand off contrast/gamma/colormap/visibility settings to a
        derived viewer (ortho/volume/z-montage) at construction time.
        """
        obj = cls.__new__(cls)
        obj._states = [replace(s) for s in states]
        obj._listeners = []
        return obj

    def __len__(self):
        return len(self._states)

    def __getitem__(self, idx):
        return self._states[idx]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def set_clim(self, idx, vmin, vmax):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(
                self._states[idx], clim=(float(vmin), float(vmax))
            )
            self._notify(idx, "clim")

    def set_gamma(self, idx, gamma):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(self._states[idx], gamma=float(gamma))
            self._notify(idx, "gamma")

    def set_colormap_name(self, idx, name):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(self._states[idx], colormap_name=name)
            self._notify(idx, "colormap_name")

    def set_visible(self, idx, visible):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(
                self._states[idx], visible=bool(visible)
            )
            self._notify(idx, "visible")

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def subscribe(self, callback):
        """Register ``callback(channel_idx, field)``. Returns an
        unsubscribe function."""
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
