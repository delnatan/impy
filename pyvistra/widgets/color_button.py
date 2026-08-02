"""ColorButton — a small reusable RGBA color-picker button.

Factors out the color-button pattern duplicated across ``overlay_settings.py``
and ``zmontage_settings.py`` (tuple<->QColor conversion, swatch styling, and
wiring a ``QColorDialog``) so new dialogs don't have to re-implement it.
"""

from __future__ import annotations

from qtpy.QtGui import QColor
from qtpy.QtWidgets import QColorDialog, QPushButton


def tuple_to_qcolor(color, default=(1.0, 1.0, 1.0, 1.0)) -> QColor:
    """Convert an RGBA float tuple to a ``QColor``, falling back to *default*."""
    if (
        isinstance(color, (list, tuple))
        and len(color) == 4
        and all(isinstance(c, (int, float)) for c in color)
    ):
        r, g, b, a = (min(max(float(c), 0.0), 1.0) for c in color)
        return QColor.fromRgbF(r, g, b, a)
    r, g, b, a = default
    return QColor.fromRgbF(r, g, b, a)


def qcolor_to_tuple(color: QColor) -> tuple[float, float, float, float]:
    """Convert a ``QColor`` to an RGBA float tuple."""
    return (color.redF(), color.greenF(), color.blueF(), color.alphaF())


class ColorButton(QPushButton):
    """A push button showing an RGBA swatch that opens ``QColorDialog`` on click."""

    def __init__(
        self,
        initial_rgba=(1.0, 1.0, 1.0, 1.0),
        *,
        title="Choose Color",
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        self._color = tuple_to_qcolor(initial_rgba)
        self._sync()
        self.clicked.connect(self._choose)

    @property
    def rgba(self) -> tuple[float, float, float, float]:
        return qcolor_to_tuple(self._color)

    def set_rgba(self, rgba) -> None:
        self._color = tuple_to_qcolor(rgba)
        self._sync()

    def _choose(self) -> None:
        color = QColorDialog.getColor(
            self._color, self, self._title, options=QColorDialog.ShowAlphaChannel,
        )
        if color.isValid():
            self._color = color
            self._sync()

    def _sync(self) -> None:
        self.setText(self._color.name(QColor.HexArgb))
        self.setStyleSheet(
            f"background-color: rgba({self._color.red()}, {self._color.green()}, "
            f"{self._color.blue()}, {self._color.alpha()});"
        )
