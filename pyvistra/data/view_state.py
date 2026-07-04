"""Navigation state for a 5D viewer.

A single ``ViewState`` owns the displayed position (t, z) and the
Z-projection mode/range of one viewer. Viewers delegate to it instead
of scattering ``t_idx``/``z_idx`` attributes and reading projection
state back out of widgets. UI widgets and render paths subscribe via
``subscribe(callback)`` and react on change — the same pattern as
:class:`ChannelDisplayList`.

This module is pure data: no Qt, no vispy.
"""


class ViewState:
    """Observable navigation state: t, z, z_projection, z_range.

    Mutations go through ``set_t``/``set_z``/``set_z_projection``/
    ``set_z_range``. Each fires a ``(field,)`` notification to every
    subscriber, with ``field`` one of ``"t"``, ``"z"``,
    ``"z_projection"``, ``"z_range"``. Setters no-op (no notification)
    when the value is unchanged.

    Callbacks run on the caller's thread (no marshalling), matching
    ``ChannelDisplayList`` semantics.
    """

    def __init__(self, t=0, z=0):
        self._t = int(t)
        self._z = int(z)
        self._z_projection = False
        self._z_range = None  # (lo, hi) inclusive, or None for full range
        self._listeners = []

    @property
    def t(self):
        return self._t

    @property
    def z(self):
        return self._z

    @property
    def z_projection(self):
        return self._z_projection

    @property
    def z_range(self):
        return self._z_range

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def set_t(self, t):
        t = int(t)
        if t != self._t:
            self._t = t
            self._notify("t")

    def set_z(self, z):
        z = int(z)
        if z != self._z:
            self._z = z
            self._notify("z")

    def set_z_projection(self, enabled):
        enabled = bool(enabled)
        if enabled != self._z_projection:
            self._z_projection = enabled
            self._notify("z_projection")

    def set_z_range(self, lo, hi):
        rng = (int(lo), int(hi))
        if rng != self._z_range:
            self._z_range = rng
            self._notify("z_range")

    def clamp(self, T, Z):
        """Clamp t and z into ``[0, T-1]`` / ``[0, Z-1]`` after a data
        swap. Notifies per changed field."""
        self.set_t(min(max(0, self._t), T - 1))
        self.set_z(min(max(0, self._z), Z - 1))
        if self._z_range is not None:
            lo, hi = self._z_range
            self.set_z_range(
                min(max(0, lo), Z - 1), min(max(0, hi), Z - 1)
            )

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def subscribe(self, callback):
        """Register ``callback(field)``. Returns an unsubscribe
        function."""
        self._listeners.append(callback)

        def _unsubscribe():
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def _notify(self, field):
        for cb in list(self._listeners):
            try:
                cb(field)
            except Exception:
                pass
