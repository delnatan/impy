"""PointPropertiesDialog — small modeless editor for one point.

Lets the user adjust ``x``, ``y``, and ``t`` for a single point in a point
layer. The dialog subscribes to the layer's :class:`PointDataHolder` so
external mutations (drag, undo) update the fields live.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..data.point_commands import MovePoint, PEVT_REMOVED

_RANGE = 1.0e7


class PointPropertiesDialog(QDialog):
    def __init__(self, window, layer, point_id: int, parent=None):
        super().__init__(parent if parent is not None else window)
        self.setWindowTitle(f"Point — {layer.name}")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self._window = window
        self._layer = layer
        self._point_id = int(point_id)
        self._suppress = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self._x_spin = QDoubleSpinBox(); self._x_spin.setRange(-_RANGE, _RANGE); self._x_spin.setDecimals(2)
        self._y_spin = QDoubleSpinBox(); self._y_spin.setRange(-_RANGE, _RANGE); self._y_spin.setDecimals(2)
        self._t_spin = QSpinBox(); self._t_spin.setRange(0, 1_000_000)
        self._x_spin.setKeyboardTracking(False)
        self._y_spin.setKeyboardTracking(False)
        form.addRow("x:", self._x_spin)
        form.addRow("y:", self._y_spin)
        form.addRow("t:", self._t_spin)
        outer.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

        self._x_spin.editingFinished.connect(self._commit_xy)
        self._y_spin.editingFinished.connect(self._commit_xy)
        # ``t`` is a structural change (resorts the table) — refuse to wire
        # it as an in-place edit; future work may add a SetPointFrame command.
        self._t_spin.setEnabled(False)

        self._populate()
        self._unsubscribe = layer.data.subscribe(self._on_event)

    def _row(self) -> dict | None:
        return self._layer.data.table.get_point(self._point_id)

    def _populate(self) -> None:
        row = self._row()
        if row is None:
            return
        self._suppress = True
        try:
            self._x_spin.setValue(float(row.get("x", 0.0)))
            self._y_spin.setValue(float(row.get("y", 0.0)))
            self._t_spin.setValue(int(row.get("t", 0)))
        finally:
            self._suppress = False

    def _commit_xy(self) -> None:
        if self._suppress:
            return
        row = self._row()
        if row is None:
            return
        nx = float(self._x_spin.value())
        ny = float(self._y_spin.value())
        if float(row["x"]) == nx and float(row["y"]) == ny:
            return
        self._layer.undo_stack.push(
            MovePoint(self._point_id, nx, ny), self._layer.data
        )

    def _on_event(self, kind, pid):
        if kind == PEVT_REMOVED and int(pid) == self._point_id:
            self.close()
            return
        if pid != self._point_id and pid != -1:
            return
        # Skip if a spinbox is being edited.
        focus = self.focusWidget()
        if isinstance(focus, (QDoubleSpinBox, QSpinBox)):
            return
        self._populate()

    def closeEvent(self, event):
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:
                pass
            self._unsubscribe = None
        super().closeEvent(event)
