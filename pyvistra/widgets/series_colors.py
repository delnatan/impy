"""Small color helpers shared by the multi-window "series" dialogs
(line profile, radial profile) and :class:`WindowSeriesMixin`.

Split out from ``line_profile.py`` so ``window_series_mixin.py`` can use
them without an import cycle (the mixin is imported by both dialogs;
``line_profile.py`` used to be where these lived, which line_profile.py
itself still needs, but the mixin needs them too).
"""

from __future__ import annotations

from qtpy.QtGui import QColor

FALLBACK_COLORS = [
    "#66CCFF",
    "#FF9966",
    "#99FF99",
    "#FFCC66",
    "#CC99FF",
    "#FF6699",
    "#66FFCC",
    "#CCCCCC",
]


def _to_qcolor(value, default):
    if isinstance(value, QColor):
        return value

    # Handle rgb float triples/lists from renderer
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        try:
            r, g, b = value[:3]
            if max(r, g, b) <= 1.0:
                return QColor.fromRgbF(float(r), float(g), float(b))
            return QColor(int(r), int(g), int(b))
        except Exception:
            return default

    qcolor = QColor(value)
    if not qcolor.isValid():
        return default
    return qcolor
