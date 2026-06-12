"""Reusable region-restriction widget for processor dialogs.

Lets the user restrict a processor's input to either (a) a rectangle ROI
on the source window and/or (b) a subset of Z planes. The widget is
*selector only* — it never owns the source window picker. The hosting
dialog calls :meth:`set_source` whenever the source window changes.

Usage
-----
.. code-block:: python

    selector = RegionSelector()
    layout.addWidget(selector)
    selector.set_source(my_window)
    # later, when running the processor:
    bbox = selector.bbox()              # None or (y0, x0, y1, x1)
    zrange = selector.z_range()         # None or (z0, z1) — inclusive z0, exclusive z1
    cropped = selector.crop_zyx(image)  # convenience, applies both

The widget reads rectangle entries from ``source.layers.by_type("shapes")``
and refreshes when the user changes the rectangle combo or when
:meth:`set_source` is re-called.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class RegionSelector(QGroupBox):
    """Group box exposing an optional rectangle bbox + Z range.

    Signals
    -------
    region_changed
        Emitted whenever the bbox or z range changes (combo selection,
        spinner edit, or :meth:`set_source` swapping in a new window).
    """

    region_changed = Signal()

    def __init__(self, parent=None, *, title: str = "Region (optional)"):
        super().__init__(title, parent)
        self._source = None  # ImageWindow or any object with .layers and .Z
        self._z_max = 0      # cached source Z size

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # --- Rectangle row ---
        rect_row = QHBoxLayout()
        rect_row.addWidget(QLabel("Rectangle:"))
        self.rect_combo = QComboBox()
        self.rect_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        rect_row.addWidget(self.rect_combo, 1)
        self.rect_refresh_btn = QPushButton("Refresh")
        self.rect_refresh_btn.clicked.connect(self._refresh_rectangles)
        rect_row.addWidget(self.rect_refresh_btn)
        outer.addLayout(rect_row)

        self.rect_info = QLabel("")
        self.rect_info.setStyleSheet("color: #888; font-size: 10px;")
        outer.addWidget(self.rect_info)

        # --- Z range row ---
        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z range:"))
        self.z_from_spin = QSpinBox()
        self.z_from_spin.setRange(0, 0)
        z_row.addWidget(self.z_from_spin)
        z_row.addWidget(QLabel("to"))
        self.z_to_spin = QSpinBox()
        self.z_to_spin.setRange(0, 0)
        z_row.addWidget(self.z_to_spin)
        self.z_full_btn = QPushButton("Full stack")
        self.z_full_btn.clicked.connect(self._reset_z_full)
        z_row.addWidget(self.z_full_btn)
        z_row.addStretch()
        outer.addLayout(z_row)

        self.z_info = QLabel("")
        self.z_info.setStyleSheet("color: #888; font-size: 10px;")
        outer.addWidget(self.z_info)

        # Signal wiring (after construction).
        self.rect_combo.currentIndexChanged.connect(self._on_rect_changed)
        self.z_from_spin.valueChanged.connect(self._on_z_changed)
        self.z_to_spin.valueChanged.connect(self._on_z_changed)

        self._refresh_rectangles()
        self._refresh_z()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_source(self, source) -> None:
        """Point the selector at a new source window.

        ``source`` is expected to be an ``ImageWindow``-like object with a
        ``layers`` attribute (``by_type("shapes")``) and ``Z`` int attribute.
        Pass ``None`` to clear.
        """
        self._source = source
        self._z_max = int(getattr(source, "Z", 0)) if source is not None else 0
        self._refresh_rectangles()
        self._refresh_z()
        self.region_changed.emit()

    def bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Return ``(y0, x0, y1, x1)`` or ``None`` for the full image.

        ``y1`` and ``x1`` are exclusive — directly usable as numpy slices.
        """
        data = self.rect_combo.currentData()
        if data is None:
            return None
        y0, x0, y1, x1 = data
        return int(y0), int(x0), int(y1), int(x1)

    def z_range(self) -> Optional[Tuple[int, int]]:
        """Return ``(z0, z1)`` (z1 exclusive) or ``None`` for the full stack."""
        if self._z_max <= 0:
            return None
        z0 = int(self.z_from_spin.value())
        z1 = int(self.z_to_spin.value()) + 1
        if z0 <= 0 and z1 >= self._z_max:
            return None
        return z0, z1

    def crop_zyx(self, image: np.ndarray) -> np.ndarray:
        """Apply bbox + z_range to a 3D ``(Z, Y, X)`` array.

        Returns a view (no copy). Returns ``image`` unchanged when both
        selectors are full-range.
        """
        zr = self.z_range()
        bb = self.bbox()
        if zr is None and bb is None:
            return image
        z_slice = slice(zr[0], zr[1]) if zr is not None else slice(None)
        if bb is None:
            return image[z_slice]
        y0, x0, y1, x1 = bb
        return image[z_slice, y0:y1, x0:x1]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _refresh_rectangles(self) -> None:
        """Repopulate the rectangle combo from the source window's shape layers."""
        self.rect_combo.blockSignals(True)
        self.rect_combo.clear()
        self.rect_combo.addItem("Full image", userData=None)

        if self._source is not None and hasattr(self._source, "layers"):
            for layer in self._source.layers.by_type("shapes"):
                data = layer.data
                for sid in data.shape_ids:
                    rec = data.get(sid)
                    if rec.shape_type != "rectangle":
                        continue
                    p = rec.params
                    x1, y1, x2, y2 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                    x0i = int(max(0, min(x1, x2)))
                    x1i = int(max(x1, x2))
                    y0i = int(max(0, min(y1, y2)))
                    y1i = int(max(y1, y2))
                    if y1i <= y0i or x1i <= x0i:
                        continue
                    bbox = (y0i, x0i, y1i, x1i)
                    shape_label = rec.label or f"rect #{sid}"
                    label = (
                        f"[{layer.name}] {shape_label}  "
                        f"({y1i - y0i}×{x1i - x0i})"
                    )
                    self.rect_combo.addItem(label, userData=bbox)

        self.rect_combo.blockSignals(False)
        self._on_rect_changed(self.rect_combo.currentIndex())

    def _refresh_z(self) -> None:
        """Reset Z spinbox ranges based on the source's Z size."""
        self.z_from_spin.blockSignals(True)
        self.z_to_spin.blockSignals(True)
        nz = max(1, self._z_max)
        self.z_from_spin.setRange(0, nz - 1)
        self.z_to_spin.setRange(0, nz - 1)
        self.z_from_spin.setValue(0)
        self.z_to_spin.setValue(nz - 1)
        self.z_from_spin.blockSignals(False)
        self.z_to_spin.blockSignals(False)
        self._update_z_info()

    def _reset_z_full(self) -> None:
        self._refresh_z()
        self.region_changed.emit()

    def _on_rect_changed(self, _index: int) -> None:
        bbox = self.bbox()
        if bbox is None:
            self.rect_info.setText("Full image will be used.")
        else:
            y0, x0, y1, x1 = bbox
            self.rect_info.setText(
                f"y={y0}..{y1}  x={x0}..{x1}  ({y1 - y0}×{x1 - x0} px)"
            )
        self.region_changed.emit()

    def _on_z_changed(self, _v: int) -> None:
        # Keep z_from <= z_to.
        if self.z_from_spin.value() > self.z_to_spin.value():
            sender = self.sender()
            other = self.z_to_spin if sender is self.z_from_spin else self.z_from_spin
            other.blockSignals(True)
            other.setValue(self.sender().value())
            other.blockSignals(False)
        self._update_z_info()
        self.region_changed.emit()

    def _update_z_info(self) -> None:
        zr = self.z_range()
        if zr is None:
            self.z_info.setText(f"Full stack ({self._z_max} planes).")
        else:
            z0, z1 = zr
            self.z_info.setText(f"Planes {z0}..{z1 - 1}  ({z1 - z0} of {self._z_max})")
