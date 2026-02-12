"""
Gel-specific ROI classes for pyvistra.

This module contains ROI classes specialized for gel electrophoresis analysis.
"""

import numpy as np
from vispy import scene

from .rois import ROI, RectangleROI


class LaneROI(RectangleROI):
    """
    Rectangle ROI specialized for gel lane analysis.

    Extends RectangleROI with:
    - Horizontal band markers with labels
    - Marker drag functionality (markers take priority over body)
    - Lock mode to prevent lane movement during analysis
    - Toggle for border and marker label visibility
    """

    # Default colors
    LADDER_COLOR = "#FFBB16"
    SAMPLE_COLOR = "#9C27B0"

    def __init__(self, view, name="Lane"):
        super().__init__(view, name)
        self.locked = False  # When True, body can't be moved
        self.show_marker_labels = True
        self._show_border = True
        self._markers = []  # List of dicts: {y_local, label, color}
        self._marker_visuals = []  # List of (Line, Text) tuples
        self._dragging_marker_idx = None
        self._on_markers_changed = None  # Callback when markers are adjusted

    @property
    def show_border(self):
        return self._show_border

    @show_border.setter
    def show_border(self, value):
        self._show_border = value
        self.rect.visible = value

    @property
    def markers(self):
        """Get list of marker dicts."""
        return self._markers

    def set_markers_changed_callback(self, callback):
        """Set callback to be called when markers are manually adjusted."""
        self._on_markers_changed = callback

    def set_markers(self, marker_data):
        """
        Set markers from list of dicts.

        Args:
            marker_data: List of dicts with keys:
                - y_local: float, position relative to lane top
                - label: str, text label (e.g., "250 kDa")
                - color: str, color name
        """
        self._clear_marker_visuals()
        self._markers = list(marker_data)
        self._create_marker_visuals()

    def clear_markers(self):
        """Remove all markers."""
        self._clear_marker_visuals()
        self._markers = []

    def add_marker(self, y_local, label="", color=None):
        """
        Add a new marker at the specified y_local position.

        Args:
            y_local: Position relative to lane top
            label: Text label for the marker
            color: Color for the marker (defaults to SAMPLE_COLOR)

        Returns:
            Index of the new marker
        """
        if color is None:
            color = self.SAMPLE_COLOR

        marker = {"y_local": float(y_local), "label": label, "color": color}
        self._markers.append(marker)

        # Create visual for new marker
        x_min, x_max, y_min, y_max = self._get_bounds()
        y_global = y_min + y_local

        line_pos = np.array(
            [[x_min, y_global, 0], [x_max, y_global, 0]], dtype=np.float32
        )
        line_visual = scene.visuals.Line(
            pos=line_pos, color=color, width=2, parent=self.view.scene
        )

        text_visual = scene.visuals.Text(
            text=label,
            color=color,
            font_size=8,
            anchor_x="right",
            anchor_y="center",
            parent=self.view.scene,
        )
        text_visual.pos = (x_min - 3, y_global, 0)
        text_visual.visible = self.show_marker_labels and bool(label)

        self._marker_visuals.append((line_visual, text_visual))
        self.visuals.extend([line_visual, text_visual])

        return len(self._markers) - 1

    def remove_marker(self, idx):
        """
        Remove a marker by index.

        Args:
            idx: Index of the marker to remove

        Returns:
            True if removed, False if index invalid
        """
        if idx < 0 or idx >= len(self._markers):
            return False

        # Remove visual
        if idx < len(self._marker_visuals):
            line_visual, text_visual = self._marker_visuals[idx]
            line_visual.parent = None
            text_visual.parent = None
            if line_visual in self.visuals:
                self.visuals.remove(line_visual)
            if text_visual in self.visuals:
                self.visuals.remove(text_visual)
            self._marker_visuals.pop(idx)

        # Remove data
        self._markers.pop(idx)
        return True

    def get_marker_positions(self):
        """Get list of marker y_local positions (sorted)."""
        return sorted(m["y_local"] for m in self._markers)

    def _get_sorted_marker_indices(self):
        """Get marker indices sorted by y_local position."""
        return sorted(
            range(len(self._markers)),
            key=lambda i: self._markers[i]["y_local"],
        )

    def update_marker_labels(self, labels):
        """
        Update marker labels in sorted y_local order.

        Args:
            labels: List of label strings, one per marker (in sorted y position order)
        """
        # Get indices sorted by y_local to match how positions are returned
        sorted_indices = self._get_sorted_marker_indices()

        for label_idx, marker_idx in enumerate(sorted_indices):
            if label_idx >= len(labels):
                break
            label = labels[label_idx]
            self._markers[marker_idx]["label"] = label
            if marker_idx < len(self._marker_visuals):
                _, text_visual = self._marker_visuals[marker_idx]
                text_visual.text = label
                text_visual.visible = self.show_marker_labels and bool(label)

    def _get_bounds(self):
        """Get lane bounds (x_min, x_max, y_min, y_max)."""
        if "p1" not in self.data:
            return 0, 0, 0, 0
        p1 = self.data["p1"]
        p2 = self.data["p2"]
        x_min = min(p1[0], p2[0])
        x_max = max(p1[0], p2[0])
        y_min = min(p1[1], p2[1])
        y_max = max(p1[1], p2[1])
        return x_min, x_max, y_min, y_max

    def _create_marker_visuals(self):
        """Create visuals for all markers."""
        x_min, x_max, y_min, y_max = self._get_bounds()

        for marker in self._markers:
            y_local = marker["y_local"]
            label = marker.get("label", "")
            color = marker.get("color", self.SAMPLE_COLOR)

            y_global = y_min + y_local

            # Create line visual
            line_pos = np.array(
                [[x_min, y_global, 0], [x_max, y_global, 0]], dtype=np.float32
            )

            line_visual = scene.visuals.Line(
                pos=line_pos, color=color, width=2, parent=self.view.scene
            )

            # Create label visual
            text_visual = scene.visuals.Text(
                text=label,
                color=color,
                font_size=8,
                anchor_x="right",
                anchor_y="center",
                parent=self.view.scene,
            )
            text_visual.pos = (x_min - 3, y_global, 0)
            text_visual.visible = self.show_marker_labels and bool(label)

            self._marker_visuals.append((line_visual, text_visual))
            self.visuals.extend([line_visual, text_visual])

    def _clear_marker_visuals(self):
        """Remove all marker visuals from scene."""
        for line_visual, text_visual in self._marker_visuals:
            line_visual.parent = None
            text_visual.parent = None
            if line_visual in self.visuals:
                self.visuals.remove(line_visual)
            if text_visual in self.visuals:
                self.visuals.remove(text_visual)
        self._marker_visuals = []

    def _update_marker_visual(self, idx):
        """Update visual for a single marker."""
        if idx >= len(self._markers) or idx >= len(self._marker_visuals):
            return

        marker = self._markers[idx]
        line_visual, text_visual = self._marker_visuals[idx]

        x_min, x_max, y_min, y_max = self._get_bounds()
        y_global = y_min + marker["y_local"]

        line_pos = np.array(
            [[x_min, y_global, 0], [x_max, y_global, 0]], dtype=np.float32
        )
        line_visual.set_data(pos=line_pos)
        text_visual.pos = (x_min - 3, y_global, 0)

    def set_marker_labels_visible(self, visible):
        """Toggle marker label visibility."""
        self.show_marker_labels = visible

        for line_visual, text_visual in self._marker_visuals:
            # Only show if there's actually a label
            marker_idx = self._marker_visuals.index((line_visual, text_visual))
            if marker_idx < len(self._markers):
                has_label = bool(self._markers[marker_idx].get("label", ""))
                text_visual.visible = visible and has_label

    def hit_test(self, point):
        """
        Test for hit on markers, handles, or body.

        Returns:
            - ('marker', idx) if marker hit
            - handle_id if handle hit
            - 'center' if body hit (and not locked, allows dragging)
            - 'body' if body hit (and locked, no dragging but in bounds)
            - None if no hit
        """
        px, py = point
        x_min, x_max, y_min, y_max = self._get_bounds()

        # 1. Check markers first (only if we have any)
        if self._markers and x_min <= px <= x_max:
            for i, marker in enumerate(self._markers):
                y_global = y_min + marker["y_local"]
                if abs(py - y_global) < 5:  # 5 pixel tolerance
                    return ("marker", i)

        # 2. Check handles (parent class)
        hid = ROI.hit_test(
            self, point
        )  # Call grandparent to skip RectangleROI body check
        if hid:
            return hid

        # 3. Check body
        if x_min <= px <= x_max and y_min <= py <= y_max:
            if self.locked:
                return "body"  # In bounds but can't move
            else:
                return "center"  # In bounds and can move

        return None

    def adjust(self, handle_id, new_pos):
        """Adjust marker or handle position."""
        if isinstance(handle_id, tuple) and handle_id[0] == "marker":
            idx = handle_id[1]
            self._move_marker(idx, new_pos[1])
            return

        # Otherwise, normal rectangle adjustment
        super().adjust(handle_id, new_pos)

    def _move_marker(self, idx, new_y_global):
        """Move a marker to a new global y position."""
        if idx >= len(self._markers):
            return

        x_min, x_max, y_min, y_max = self._get_bounds()

        # Clamp to lane bounds
        new_y_local = new_y_global - y_min
        new_y_local = max(0, min(new_y_local, y_max - y_min))

        self._markers[idx]["y_local"] = new_y_local
        self._update_marker_visual(idx)

    def end_marker_drag(self):
        """Called when marker dragging ends. Triggers callback."""
        if self._on_markers_changed:
            self._on_markers_changed()

    def move(self, delta):
        """Move the lane (if not locked)."""
        if self.locked:
            return
        super().move(delta)
        # Also move markers visually
        self._refresh_marker_visuals()

    def update(self, p1, p2):
        """Update lane bounds and refresh marker visuals."""
        super().update(p1, p2)
        self._refresh_marker_visuals()

    def _refresh_marker_visuals(self):
        """Refresh all marker visuals after lane move/resize."""
        for i in range(len(self._markers)):
            self._update_marker_visual(i)

    def set_visible(self, visible):
        """Set visibility of lane and optionally markers."""
        # Handle border separately
        if visible:
            self.rect.visible = self._show_border
        else:
            self.rect.visible = False

        # Handle other visuals (handles, label)
        if self.handle_visual:
            self.handle_visual.visible = visible and self.selected
        if self.label_visual:
            self.label_visual.visible = visible and ROI.show_labels

        # Marker visuals stay visible even when border is hidden
        for line_visual, text_visual in self._marker_visuals:
            line_visual.visible = visible
            marker_idx = self._marker_visuals.index((line_visual, text_visual))
            if marker_idx < len(self._markers):
                has_label = bool(self._markers[marker_idx].get("label", ""))
                text_visual.visible = (
                    visible and self.show_marker_labels and has_label
                )

    def to_dict(self):
        """Serialize lane including markers."""
        d = super().to_dict()
        d["data"]["markers"] = self._markers
        d["data"]["locked"] = self.locked
        return d

    def from_dict(self, data):
        """Deserialize lane including markers."""
        # Extract markers before calling parent
        markers = data.pop("markers", [])
        locked = data.pop("locked", False)

        super().from_dict(data)

        self.locked = locked
        if markers:
            self.set_markers(markers)
