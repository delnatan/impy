"""ShapePropertiesDialog — modeless numerical editor for a single shape.

Bound to a ``(Layer, shape_id)`` pair. The form is dispatched by shape type:

- **Rectangle**: ``x, y, width, height`` (positive width/height; x,y is the
  top-left corner of the axis-aligned bbox).
- **Circle**: ``cx, cy, radius``.
- **Line**: ``x1, y1, x2, y2``; readouts for length (px and µm when the
  parent window exposes a pixel scale) and angle.

Common header: label, time-anchor (``t``), z-anchor (``z``), and an "Apply to
all frames" checkbox that sets the ``ALL_FRAMES`` sentinel on both axes.

The dialog subscribes to the layer's ``ShapeData`` so external mutations
(drag-edit, undo) update the spinboxes live. Edits in the dialog are batched
per field — committed on focus-out / Enter — so each Apply lands as one
undoable ``SetShapeParams`` (or ``SetShapeLabel``) command.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..data.shapes import (
    ALL_FRAMES,
    CIRCLE,
    EVT_EDITED,
    EVT_REMOVED,
    LINE,
    POLYLINE,
    RECTANGLE,
    SetPolylineFlags,
    SetShapeLabel,
    SetShapeParams,
    SetShapeVertices,
    get_outline,
    polyline_is_closed,
    polyline_smoothness,
)

_COORD_RANGE = 1.0e7  # generous upper bound so we never clip user values


def _spin(decimals: int = 2, step: float = 1.0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(-_COORD_RANGE, _COORD_RANGE)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setKeyboardTracking(False)
    return s


class ShapePropertiesDialog(QDialog):
    """Modeless properties editor for one shape."""

    def __init__(self, window, layer, shape_id: int, parent=None):
        super().__init__(parent if parent is not None else window)
        self.setWindowTitle(f"Shape — {layer.name}")
        # Modeless; survive parent dialog interactions.
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self._window = window
        self._layer = layer
        self._shape_id = shape_id
        self._suppress = False
        self._unsubscribe = None

        # Track baseline params for the next snapshot-style commit so
        # multi-field edits land as a single SetShapeParams.
        rec = layer.data.get(shape_id)
        self._baseline_params = rec.params.copy()
        self._shape_type = rec.shape_type

        self._build_ui(rec)
        self._populate_from_record(rec)

        # Auto-update on data changes (and auto-close on removal).
        self._unsubscribe = layer.data.subscribe(self._on_data_event)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self, rec) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Header (label, t, z, all-frames)
        header = QFormLayout()
        header.setLabelAlignment(Qt.AlignRight)

        self._label_edit = QLineEdit()
        self._label_edit.editingFinished.connect(self._commit_label)
        header.addRow("Label:", self._label_edit)

        self._t_spin = QSpinBox()
        self._t_spin.setRange(0, 1_000_000)
        self._t_spin.valueChanged.connect(self._commit_anchor)
        header.addRow("t:", self._t_spin)

        self._z_spin = QSpinBox()
        self._z_spin.setRange(0, 1_000_000)
        self._z_spin.valueChanged.connect(self._commit_anchor)
        header.addRow("z:", self._z_spin)

        self._all_frames = QCheckBox("Apply to all frames (any t, any z)")
        self._all_frames.toggled.connect(self._on_all_frames_toggled)
        header.addRow("", self._all_frames)

        outer.addLayout(header)

        # Type label
        type_lbl = QLabel(f"<i>{rec.shape_type.capitalize()}</i>")
        type_lbl.setAlignment(Qt.AlignLeft)
        outer.addWidget(type_lbl)

        # Type-specific form
        coords_box = QWidget()
        coords_form = QFormLayout(coords_box)
        coords_form.setLabelAlignment(Qt.AlignRight)
        coords_form.setContentsMargins(0, 0, 0, 0)
        self._coord_spins: dict[str, QDoubleSpinBox] = {}
        self._readouts: dict[str, QLabel] = {}
        self._vertex_table: QTableWidget | None = None
        self._closed_check: QCheckBox | None = None
        self._smoothness_slider: QSlider | None = None

        if self._shape_type == POLYLINE:
            self._build_polyline_form(outer)
            self._snap_int = QCheckBox("Snap to integer pixels")
            outer.addWidget(self._snap_int)

            # Close button row.
            btn_row = QHBoxLayout()
            btn_row.addStretch(1)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.accept)
            btn_row.addWidget(close_btn)
            outer.addLayout(btn_row)
            return

        if self._shape_type == RECTANGLE:
            for key in ("x", "y", "width", "height"):
                spin = _spin()
                coords_form.addRow(f"{key}:", spin)
                self._coord_spins[key] = spin
                spin.editingFinished.connect(self._commit_coords)
        elif self._shape_type == CIRCLE:
            for key in ("cx", "cy", "radius"):
                spin = _spin()
                coords_form.addRow(f"{key}:", spin)
                self._coord_spins[key] = spin
                spin.editingFinished.connect(self._commit_coords)
        elif self._shape_type == LINE:
            for key in ("x1", "y1", "x2", "y2"):
                spin = _spin()
                coords_form.addRow(f"{key}:", spin)
                self._coord_spins[key] = spin
                spin.editingFinished.connect(self._commit_coords)
            self._readouts["length_px"] = QLabel("—")
            self._readouts["length_um"] = QLabel("—")
            self._readouts["angle"] = QLabel("—")
            coords_form.addRow("Length (px):", self._readouts["length_px"])
            coords_form.addRow("Length (µm):", self._readouts["length_um"])
            coords_form.addRow("Angle (°):", self._readouts["angle"])

        outer.addWidget(coords_box)

        # Snap-to-integer toggle (rectangle / line only — circle radius
        # snapping isn't well-defined and we leave it as-is).
        self._snap_int = QCheckBox("Snap to integer pixels")
        outer.addWidget(self._snap_int)

        # Close button row.
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Polyline form
    # ------------------------------------------------------------------
    def _build_polyline_form(self, outer: QVBoxLayout) -> None:
        # Vertex table — x, y columns. Edits commit on cellChanged.
        self._vertex_table = QTableWidget(0, 2)
        self._vertex_table.setHorizontalHeaderLabels(["x", "y"])
        self._vertex_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._vertex_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._vertex_table.setMinimumHeight(150)
        self._vertex_table.cellChanged.connect(self._on_vertex_cell_changed)
        outer.addWidget(self._vertex_table)

        # Add/remove buttons
        vbtn_row = QHBoxLayout()
        self._add_vertex_btn = QPushButton("Add vertex")
        self._add_vertex_btn.clicked.connect(self._on_add_vertex_row)
        self._remove_vertex_btn = QPushButton("Remove selected")
        self._remove_vertex_btn.clicked.connect(self._on_remove_vertex_row)
        vbtn_row.addWidget(self._add_vertex_btn)
        vbtn_row.addWidget(self._remove_vertex_btn)
        vbtn_row.addStretch(1)
        outer.addLayout(vbtn_row)

        # Flags
        flag_form = QFormLayout()
        self._closed_check = QCheckBox()
        self._closed_check.toggled.connect(self._commit_closed)
        flag_form.addRow("Closed:", self._closed_check)

        self._smoothness_slider = QSlider(Qt.Horizontal)
        self._smoothness_slider.setRange(0, 100)  # 0.0 .. 10.0 (×10)
        self._smoothness_slider.setTickInterval(10)
        self._smoothness_label = QLabel("0.0")
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(self._smoothness_slider, 1)
        smooth_row.addWidget(self._smoothness_label)
        self._smoothness_slider.valueChanged.connect(self._on_smoothness_slider)
        flag_form.addRow("Smoothness:", smooth_row)

        outer.addLayout(flag_form)

        # Readouts
        self._readouts["length_px"] = QLabel("—")
        self._readouts["length_um"] = QLabel("—")
        self._readouts["area"] = QLabel("—")
        read_form = QFormLayout()
        read_form.addRow("Length (px):", self._readouts["length_px"])
        read_form.addRow("Length (µm):", self._readouts["length_um"])
        read_form.addRow("Area (px²):", self._readouts["area"])
        outer.addLayout(read_form)

    def _on_add_vertex_row(self) -> None:
        if self._shape_id not in self._layer.data:
            return
        rec = self._layer.data.get(self._shape_id)
        if rec.vertices is None or len(rec.vertices) == 0:
            new_v = np.array([[0.0, 0.0]], dtype=np.float32)
        else:
            new_v = np.vstack([rec.vertices, rec.vertices[-1:]])
        self._push_vertex_replace(new_v)

    def _on_remove_vertex_row(self) -> None:
        if self._vertex_table is None or self._shape_id not in self._layer.data:
            return
        rec = self._layer.data.get(self._shape_id)
        if rec.vertices is None or len(rec.vertices) <= 2:
            return
        rows = sorted({idx.row() for idx in self._vertex_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        verts = rec.vertices.copy()
        for r in rows:
            if 0 <= r < len(verts):
                verts = np.delete(verts, r, axis=0)
        if len(verts) < 2:
            return
        self._push_vertex_replace(verts)

    def _on_vertex_cell_changed(self, row: int, col: int) -> None:
        if self._suppress or self._vertex_table is None:
            return
        if self._shape_id not in self._layer.data:
            return
        rec = self._layer.data.get(self._shape_id)
        verts = (
            rec.vertices.copy()
            if rec.vertices is not None
            else np.zeros((0, 2), dtype=np.float32)
        )
        if row >= len(verts):
            return
        item = self._vertex_table.item(row, col)
        if item is None:
            return
        try:
            v = float(item.text())
        except ValueError:
            return
        if self._snap_int is not None and self._snap_int.isChecked():
            v = float(round(v))
        if verts[row, col] == v:
            return
        verts[row, col] = v
        self._push_vertex_replace(verts)

    def _push_vertex_replace(self, new_verts: np.ndarray) -> None:
        rec = self._layer.data.get(self._shape_id)
        old = (
            rec.vertices.copy()
            if rec.vertices is not None
            else np.zeros((0, 2), dtype=np.float32)
        )
        if np.array_equal(old, new_verts):
            return
        # Apply the change live, then record as a SetShapeVertices command.
        rec.vertices = new_verts.astype(np.float32, copy=True)
        self._layer.data._emit(EVT_EDITED, self._shape_id)
        self._layer.undo_stack.push_executed(
            SetShapeVertices(self._shape_id, old, new_verts)
        )

    def _commit_closed(self, on: bool) -> None:
        if self._suppress or self._shape_id not in self._layer.data:
            return
        rec = self._layer.data.get(self._shape_id)
        if bool(rec.properties.get("closed", False)) == bool(on):
            return
        self._layer.undo_stack.push(
            SetPolylineFlags(self._shape_id, closed=bool(on)), self._layer.data
        )

    def _on_smoothness_slider(self, raw: int) -> None:
        value = raw / 10.0
        self._smoothness_label.setText(f"{value:.1f}")
        if self._suppress or self._shape_id not in self._layer.data:
            return
        rec = self._layer.data.get(self._shape_id)
        if abs(polyline_smoothness(rec) - value) < 1e-9:
            return
        self._layer.undo_stack.push(
            SetPolylineFlags(self._shape_id, smoothness=value), self._layer.data
        )

    def _populate_polyline_table(self, verts: np.ndarray | None) -> None:
        if self._vertex_table is None:
            return
        verts = verts if verts is not None else np.zeros((0, 2), dtype=np.float32)
        self._vertex_table.blockSignals(True)
        self._vertex_table.setRowCount(len(verts))
        for r in range(len(verts)):
            for c in (0, 1):
                item = QTableWidgetItem(f"{float(verts[r, c]):.2f}")
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self._vertex_table.setItem(r, c, item)
        self._vertex_table.blockSignals(False)

    def _update_polyline_readouts(self, rec) -> None:
        outline = get_outline(rec)
        if len(outline) >= 2:
            seg = np.diff(outline, axis=0)
            length_px = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        else:
            length_px = 0.0
        self._readouts["length_px"].setText(f"{length_px:.2f}")

        scale = None
        meta = getattr(self._window, "meta", None) or {}
        s = meta.get("scale")
        if s is not None and len(s) >= 3:
            sx, sy = float(s[2]), float(s[1])
            scale = 0.5 * (sx + sy)
        if scale is not None and scale > 0:
            self._readouts["length_um"].setText(f"{length_px * scale:.3f}")
        else:
            self._readouts["length_um"].setText("—")

        if polyline_is_closed(rec) and len(outline) >= 4:
            # Shoelace area (outline closes with first point appended).
            x = outline[:-1, 0]
            y = outline[:-1, 1]
            area = 0.5 * abs(
                float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            )
            self._readouts["area"].setText(f"{area:.2f}")
        else:
            self._readouts["area"].setText("—")

    # ------------------------------------------------------------------
    # Population (data → widgets)
    # ------------------------------------------------------------------
    def _populate_from_record(self, rec) -> None:
        self._suppress = True
        try:
            self._label_edit.setText(rec.label)

            all_frames = rec.t == ALL_FRAMES and rec.z == ALL_FRAMES
            self._all_frames.setChecked(all_frames)
            # When sentinel is on, hide the per-frame anchors.
            self._t_spin.setEnabled(not all_frames)
            self._z_spin.setEnabled(not all_frames)
            self._t_spin.setValue(0 if rec.t == ALL_FRAMES else int(rec.t))
            self._z_spin.setValue(0 if rec.z == ALL_FRAMES else int(rec.z))

            if self._shape_type == POLYLINE:
                self._populate_polyline_table(rec.vertices)
                if self._closed_check is not None:
                    self._closed_check.setChecked(polyline_is_closed(rec))
                if self._smoothness_slider is not None:
                    s = polyline_smoothness(rec)
                    self._smoothness_slider.setValue(int(round(s * 10)))
                    self._smoothness_label.setText(f"{s:.1f}")
                self._update_polyline_readouts(rec)
                self._baseline_params = rec.params.copy()
                return

            p = rec.params
            if self._shape_type == RECTANGLE:
                l, r = float(min(p[0], p[2])), float(max(p[0], p[2]))
                t, b = float(min(p[1], p[3])), float(max(p[1], p[3]))
                self._coord_spins["x"].setValue(l)
                self._coord_spins["y"].setValue(t)
                self._coord_spins["width"].setValue(r - l)
                self._coord_spins["height"].setValue(b - t)
            elif self._shape_type == CIRCLE:
                cx, cy, ex, ey = (float(p[0]), float(p[1]),
                                  float(p[2]), float(p[3]))
                self._coord_spins["cx"].setValue(cx)
                self._coord_spins["cy"].setValue(cy)
                self._coord_spins["radius"].setValue(float(np.hypot(ex - cx, ey - cy)))
            elif self._shape_type == LINE:
                self._coord_spins["x1"].setValue(float(p[0]))
                self._coord_spins["y1"].setValue(float(p[1]))
                self._coord_spins["x2"].setValue(float(p[2]))
                self._coord_spins["y2"].setValue(float(p[3]))
                self._update_line_readouts(p)
        finally:
            self._suppress = False

        self._baseline_params = rec.params.copy()

    def _update_line_readouts(self, p: np.ndarray) -> None:
        dx, dy = float(p[2] - p[0]), float(p[3] - p[1])
        length_px = float(np.hypot(dx, dy))
        self._readouts["length_px"].setText(f"{length_px:.2f}")

        scale = None
        meta = getattr(self._window, "meta", None) or {}
        s = meta.get("scale")
        if s is not None and len(s) >= 3:
            # scale is (sz, sy, sx) in µm. Use the average of sx, sy.
            sx, sy = float(s[2]), float(s[1])
            scale = 0.5 * (sx + sy)
        if scale is not None and scale > 0:
            self._readouts["length_um"].setText(f"{length_px * scale:.3f}")
        else:
            self._readouts["length_um"].setText("—")
        angle_deg = float(np.degrees(np.arctan2(-dy, dx))) if (dx or dy) else 0.0
        self._readouts["angle"].setText(f"{angle_deg:.2f}")

    # ------------------------------------------------------------------
    # Commits (widgets → data)
    # ------------------------------------------------------------------
    def _commit_label(self) -> None:
        if self._suppress:
            return
        if self._shape_id not in self._layer.data:
            return
        new_label = self._label_edit.text().strip()
        current = self._layer.data.get_label(self._shape_id)
        if new_label == current:
            return
        self._layer.undo_stack.push(
            SetShapeLabel(self._shape_id, new_label), self._layer.data
        )

    def _commit_anchor(self) -> None:
        if self._suppress:
            return
        if self._all_frames.isChecked():
            return
        if self._shape_id not in self._layer.data:
            return
        rec = self._layer.data.get(self._shape_id)
        new_t = int(self._t_spin.value())
        new_z = int(self._z_spin.value())
        if rec.t == new_t and rec.z == new_z:
            return
        self._layer.data.set_anchor(self._shape_id, t=new_t, z=new_z)

    def _on_all_frames_toggled(self, on: bool) -> None:
        if self._suppress:
            return
        if self._shape_id not in self._layer.data:
            return
        self._t_spin.setEnabled(not on)
        self._z_spin.setEnabled(not on)
        if on:
            self._layer.data.set_anchor(
                self._shape_id, t=ALL_FRAMES, z=ALL_FRAMES
            )
        else:
            self._layer.data.set_anchor(
                self._shape_id,
                t=int(self._t_spin.value()),
                z=int(self._z_spin.value()),
            )

    def _commit_coords(self) -> None:
        if self._suppress:
            return
        if self._shape_id not in self._layer.data:
            return
        rec = self._layer.data.get(self._shape_id)
        new_params = self._params_from_widgets(rec.params)
        if new_params is None:
            return
        if np.allclose(rec.params, new_params):
            return
        cmd = SetShapeParams(self._shape_id, self._baseline_params, new_params)
        self._layer.undo_stack.push(cmd, self._layer.data)
        self._baseline_params = new_params.copy()

    def _params_from_widgets(self, current: np.ndarray) -> np.ndarray | None:
        snap = self._snap_int.isChecked()

        def _snap(v: float) -> float:
            return float(round(v)) if snap else float(v)

        p = current.copy()
        if self._shape_type == RECTANGLE:
            x = _snap(self._coord_spins["x"].value())
            y = _snap(self._coord_spins["y"].value())
            w = _snap(max(0.0, self._coord_spins["width"].value()))
            h = _snap(max(0.0, self._coord_spins["height"].value()))
            p[0], p[1], p[2], p[3] = x, y, x + w, y + h
        elif self._shape_type == CIRCLE:
            cx = _snap(self._coord_spins["cx"].value())
            cy = _snap(self._coord_spins["cy"].value())
            r = max(0.0, self._coord_spins["radius"].value())
            p[0], p[1] = cx, cy
            # Place the edge point at (cx + r, cy) so the radius is exact.
            p[2], p[3] = cx + r, cy
        elif self._shape_type == LINE:
            p[0] = _snap(self._coord_spins["x1"].value())
            p[1] = _snap(self._coord_spins["y1"].value())
            p[2] = _snap(self._coord_spins["x2"].value())
            p[3] = _snap(self._coord_spins["y2"].value())
        else:
            return None
        return p

    # ------------------------------------------------------------------
    # Data → dialog
    # ------------------------------------------------------------------
    def _on_data_event(self, event_kind: str, shape_id: int) -> None:
        # Close if our shape was removed.
        if event_kind == EVT_REMOVED and shape_id == self._shape_id:
            self.close()
            return
        if shape_id != self._shape_id and shape_id != -1:
            return
        if self._shape_id not in self._layer.data:
            self.close()
            return
        # Skip if any spinbox currently has keyboard focus — avoid clobbering
        # the user's typing.
        focused = self.focusWidget()
        if isinstance(focused, (QDoubleSpinBox, QSpinBox, QLineEdit)):
            return
        rec = self._layer.data.get(self._shape_id)
        self._populate_from_record(rec)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:
                pass
            self._unsubscribe = None
        super().closeEvent(event)
