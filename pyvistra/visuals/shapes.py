"""ShapeLayerVisual — batched vispy renderer for ShapeData.

Renders all shapes in a single layer using:
  - Line visual for outlines
  - Markers visual for handles (when selected)
  - Text visual for labels
"""

from __future__ import annotations

import numpy as np
from vispy import scene

from ..data.shapes import ShapeData, get_handles, get_outline


class ShapeLayerVisual:
    """Renders ShapeData as batched vispy visuals.

    Parameters
    ----------
    parent : vispy SceneCanvas or ViewBox scene
        The parent node for all visuals.
    """

    # Default style
    DEFAULT_OUTLINE_COLOR = "yellow"
    DEFAULT_OUTLINE_WIDTH = 2
    HANDLE_SIZE = 10
    HANDLE_FACE_COLOR = "white"
    HANDLE_EDGE_COLOR = "blue"

    def __init__(self, parent):
        self._parent = parent

        # Outline visual (batched line segments)
        self._outline = scene.visuals.Line(
            pos=np.zeros((2, 2), dtype=np.float32),
            color=self.DEFAULT_OUTLINE_COLOR,
            width=self.DEFAULT_OUTLINE_WIDTH,
            connect="strip",
            parent=parent,
        )
        self._outline.visible = False

        # Handle markers (shown for selected shapes)
        self._handles = scene.visuals.Markers(
            parent=parent,
            face_color=self.HANDLE_FACE_COLOR,
            edge_color=self.HANDLE_EDGE_COLOR,
            size=self.HANDLE_SIZE,
        )
        self._handles.visible = False

        # Label text (one per shape)
        self._labels: list[scene.visuals.Text] = []

        self._visible = True

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, val: bool) -> None:
        self._visible = val
        self._outline.visible = val and self._outline_has_data
        self._handles.visible = val and self._handles_visible
        for lbl in self._labels:
            lbl.visible = val

    def update(self, data: ShapeData, selected_ids: set[int] | None = None,
               current_t: int = 0, current_z: int = 0) -> None:
        """Rebuild visuals from ShapeData for the current time/z slice."""
        if selected_ids is None:
            selected_ids = set()

        shapes = data.get_time_slice(current_t, current_z)

        # --- Outlines ---
        all_segments = []
        connects = []
        offset = 0
        for rec in shapes:
            outline = get_outline(rec)
            n = len(outline)
            if n < 2:
                continue
            all_segments.append(outline)
            # connect consecutive points within each shape, break between shapes
            seg_connect = np.ones(n, dtype=bool)
            seg_connect[-1] = False  # break at end of shape
            connects.append(seg_connect)
            offset += n

        if all_segments:
            pos = np.vstack(all_segments)
            connect = np.concatenate(connects)
            # Pad pos to 3D for vispy
            pos3d = np.zeros((len(pos), 3), dtype=np.float32)
            pos3d[:, :2] = pos
            self._outline.set_data(pos=pos3d, connect=connect)
            self._outline.visible = self._visible
            self._outline_has_data = True
        else:
            self._outline.visible = False
            self._outline_has_data = False

        # --- Handles for selected shapes ---
        handle_pts = []
        for rec in shapes:
            if rec.shape_id in selected_ids:
                handles = get_handles(rec)
                for hx, hy in handles.values():
                    handle_pts.append([hx, hy, 0])

        if handle_pts:
            self._handles.set_data(
                pos=np.array(handle_pts, dtype=np.float32),
                face_color=self.HANDLE_FACE_COLOR,
                size=self.HANDLE_SIZE,
            )
            self._handles.visible = self._visible
            self._handles_visible = True
        else:
            self._handles.visible = False
            self._handles_visible = False

        # --- Labels ---
        # Remove excess labels
        while len(self._labels) > len(shapes):
            lbl = self._labels.pop()
            lbl.parent = None

        # Add missing labels
        while len(self._labels) < len(shapes):
            lbl = scene.visuals.Text(
                text="",
                color="white",
                font_size=10,
                anchor_x="center",
                anchor_y="bottom",
                parent=self._parent,
            )
            self._labels.append(lbl)

        # Update label positions and text
        for i, rec in enumerate(shapes):
            lbl = self._labels[i]
            name = rec.label or rec.properties.get("name") or f"Shape {rec.shape_id}"
            lbl.text = name
            lbl.visible = self._visible
            # Position label above the shape
            outline = get_outline(rec)
            if len(outline) > 0:
                cx = outline[:, 0].mean()
                top_y = outline[:, 1].min() - 5
                lbl.pos = (cx, top_y, 0)

    def remove(self) -> None:
        """Remove all visuals from scene."""
        self._outline.parent = None
        self._handles.parent = None
        for lbl in self._labels:
            lbl.parent = None
        self._labels.clear()

    # Internal state
    _outline_has_data: bool = False
    _handles_visible: bool = False
