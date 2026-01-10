import numpy as np
from vispy import scene


class ROI:
    # Class-level flag to control label visibility for all ROIs
    show_labels = True

    def __init__(self, view, name="ROI"):
        self.view = view
        self.name = name
        self.visuals = []
        self.data = {}  # Store geometry data for serialization

        # Editing State
        self.selected = False
        self.handle_visual = scene.visuals.Markers(
            parent=self.view.scene,
            face_color="white",
            edge_color="blue",
            size=12,
        )
        self.handle_visual.visible = False
        self.visuals.append(self.handle_visual)
        self.handle_points = {}  # id -> (x, y)

        # Label visual
        self.label_visual = scene.visuals.Text(
            text=self.name,
            color="white",
            font_size=10,
            anchor_x="center",
            anchor_y="bottom",
            parent=self.view.scene,
        )
        self.label_visual.visible = ROI.show_labels
        self.visuals.append(self.label_visual)

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}'>"

    def set_visible(self, visible):
        for v in self.visuals:
            # Don't show handles if not selected, even if ROI is visible
            if v is self.handle_visual:
                v.visible = visible and self.selected
            elif v is self.label_visual:
                v.visible = visible and ROI.show_labels
            else:
                v.visible = visible

    def set_name(self, name):
        """Update the ROI name and label."""
        self.name = name
        self.label_visual.text = name

    def _update_label_position(self):
        """Update label position. Override in subclasses."""
        pass

    @classmethod
    def toggle_labels(cls):
        """Toggle label visibility for all ROIs."""
        cls.show_labels = not cls.show_labels
        return cls.show_labels

    def remove(self):
        for v in self.visuals:
            v.parent = None
        self.visuals = []

    def select(self, active):
        self.selected = active
        self.handle_visual.visible = active
        if active:
            self._update_handles()

    def _update_handles(self):
        """Update the positions of the handle visual based on current geometry."""
        pass

    def hit_test(self, point):
        """
        Return handle_id if hit, 'center' if body hit, or None.
        point: (x, y) in data coordinates.
        """
        # 1. Check handles
        if self.selected:
            for hid, pos in self.handle_points.items():
                dist = np.linalg.norm(np.array(point) - np.array(pos))
                # Threshold depends on zoom, but let's assume data coords for now.
                # Ideally we project to screen coords for hit testing, but we don't have easy access to transform here?
                # We can approximate.
                if (
                    dist < 5
                ):  # 5 units tolerance? Might be too small/large depending on image scale.
                    return hid
        return None

    def move(self, delta):
        """Move the entire ROI by delta (dx, dy)."""
        pass

    def adjust(self, handle_id, new_pos):
        """Move a specific handle to new_pos."""
        pass

    def to_dict(self):
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "data": self.data,
        }

    def from_dict(self, data):
        self.data = data
        self._update_visuals_from_data()

    def _update_visuals_from_data(self):
        pass


class CoordinateROI(ROI):
    def __init__(self, view, name="Coordinate"):
        super().__init__(view, name)
        self.origin = None
        self.flipped = False

        # Visuals
        self.line = scene.visuals.Line(
            pos=np.zeros((3, 2)),
            color=["red", "green"],
            width=2,
            connect="segments",
            parent=self.view.scene,
        )
        self.marker = scene.visuals.Markers(
            pos=np.zeros((1, 3)),
            symbol="x",
            edge_color="blue",
            edge_width=2,
            size=8,
            parent=self.view.scene,
        )
        # Arrowhead for primary vector
        self.arrow = scene.visuals.Arrow(
            pos=np.zeros((2, 3)),
            color="red",
            width=4,
            arrow_size=20,
            arrow_type="stealth",
            parent=self.view.scene,
        )

        self.visuals.extend([self.line, self.marker, self.arrow])

    def update(self, p1, p2):
        """
        p1: Origin (x, y)
        p2: End of Anterior (Primary) vector (x, y)
        """
        self.origin = np.array(p1)
        anterior_vec = np.array(p2) - self.origin

        # Orthogonal vector (Dorsal) (-y, x)
        # If flipped, we negate it (or just rotate the other way)
        if self.flipped:
            dorsal_vec = np.array([anterior_vec[1], -anterior_vec[0]])
        else:
            dorsal_vec = np.array([-anterior_vec[1], anterior_vec[0]])

        # Calculate end points
        anterior_end = self.origin + anterior_vec
        dorsal_end = self.origin + dorsal_vec

        # Store data with new terminology
        self.data = {
            "origin": p1,
            "anterior": tuple(anterior_end),
            "dorsal": tuple(dorsal_end),
            "flipped": self.flipped,
        }

        # Dorsal Line (Green)
        dorsal_line_3d = np.zeros((2, 3))
        dorsal_line_3d[0, :2] = self.origin
        dorsal_line_3d[1, :2] = dorsal_end
        self.line.set_data(pos=dorsal_line_3d, color="green")

        # Anterior Arrow (Red)
        arrow_pos = np.zeros((2, 3))
        arrow_pos[0, :2] = self.origin
        arrow_pos[1, :2] = anterior_end
        self.arrow.set_data(pos=arrow_pos, color="red")

        marker_pos = np.zeros((1, 3))
        marker_pos[0, :2] = self.origin
        self.marker.set_data(
            pos=marker_pos, symbol="x", edge_color="blue", edge_width=2, size=8
        )

        self._update_label_position()

        if self.selected:
            self._update_handles()

    def _update_label_position(self):
        if self.origin is None:
            return
        # Position label slightly above and to the right of origin
        ox, oy = self.origin
        self.label_visual.pos = (ox + 10, oy - 10, 0)

    def _update_visuals_from_data(self):
        if "origin" in self.data and "anterior" in self.data:
            self.flipped = self.data.get("flipped", False)
            self.update(self.data["origin"], self.data["anterior"])

    def flip(self):
        """Flip the dorsal vector direction."""
        self.flipped = not self.flipped
        if "origin" in self.data and "anterior" in self.data:
            self.update(self.data["origin"], self.data["anterior"])

    def _update_handles(self):
        if "origin" not in self.data or "anterior" not in self.data:
            return

        self.handle_points = {
            "origin": self.data["origin"],
            "anterior": self.data["anterior"],
        }

        pts = list(self.handle_points.values())
        self.handle_visual.set_data(
            pos=np.array(pts), face_color="white", size=10
        )

    def hit_test(self, point):
        # 1. Check handles
        hid = super().hit_test(point)
        if hid:
            return hid

        # 2. Check lines? For now just handles are enough for adjustment.
        # Maybe check proximity to the main line for moving?
        if "origin" in self.data and "anterior" in self.data:
            p1 = np.array(self.data["origin"])
            p2 = np.array(self.data["anterior"])
            p = np.array(point)

            # Distance from point to segment
            # Project p onto line p1-p2
            l2 = np.sum((p1 - p2) ** 2)
            if l2 == 0:
                return None
            t = np.dot(p - p1, p2 - p1) / l2
            t = max(0, min(1, t))
            projection = p1 + t * (p2 - p1)
            dist = np.linalg.norm(p - projection)

            if dist < 5:
                return "center"

        return None

    def move(self, delta):
        if "origin" in self.data:
            dx, dy = delta
            origin = self.data["origin"]
            anterior = self.data["anterior"]

            new_origin = (origin[0] + dx, origin[1] + dy)
            new_anterior = (anterior[0] + dx, anterior[1] + dy)
            self.update(new_origin, new_anterior)

    def adjust(self, handle_id, new_pos):
        if "origin" not in self.data:
            return

        origin = self.data["origin"]
        anterior = self.data["anterior"]

        if handle_id == "origin":
            self.update(new_pos, anterior)
        elif handle_id == "anterior":
            self.update(origin, new_pos)


class RectangleROI(ROI):
    def __init__(self, view, name="Rectangle"):
        super().__init__(view, name)
        self.rect = scene.visuals.Rectangle(
            center=(0, 0, 0),
            width=1,
            height=1,
            border_color="yellow",
            color=(1, 1, 0, 0.1),
            parent=self.view.scene,
        )
        self.rect.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )
        self.visuals.append(self.rect)

    def update(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        self.data = {"p1": p1, "p2": p2}

        # Rectangle center is center of box
        cx = x + w / 2
        cy = y + h / 2

        # Ensure non-zero width/height to avoid Vispy errors
        w = max(w, 1e-6)
        h = max(h, 1e-6)

        self.rect.center = (cx, cy, 0)
        self.rect.width = w
        self.rect.height = h

        self._update_label_position()

        if self.selected:
            self._update_handles()

    def _update_label_position(self):
        if "p1" not in self.data:
            return
        p1 = self.data["p1"]
        p2 = self.data["p2"]
        # Center x, top y (with small offset above)
        cx = (p1[0] + p2[0]) / 2
        top_y = min(p1[1], p2[1]) - 5  # 5 pixels above
        self.label_visual.pos = (cx, top_y, 0)

    def _update_handles(self):
        if "p1" not in self.data:
            return

        p1 = self.data["p1"]
        p2 = self.data["p2"]
        x1, y1 = p1
        x2, y2 = p2

        # Define 4 corners
        # We need to know which is which to keep p1/p2 logic consistent?
        # Actually, p1 and p2 are just diagonal corners.
        # Let's define handles for all 4 corners to allow free resizing.
        # But for simplicity, let's just show p1 and p2?
        # No, users expect 4 corners.

        # Let's normalize
        l, r = min(x1, x2), max(x1, x2)
        t, b = min(y1, y2), max(y1, y2)

        self.handle_points = {
            "tl": (l, t),
            "tr": (r, t),
            "bl": (l, b),
            "br": (r, b),
        }

        pts = list(self.handle_points.values())
        self.handle_visual.set_data(
            pos=np.array(pts), face_color="white", size=10
        )

    def hit_test(self, point):
        # 1. Check handles
        hid = super().hit_test(point)
        if hid:
            return hid

        # 2. Check body (inside rect)
        if "p1" in self.data:
            p1 = self.data["p1"]
            p2 = self.data["p2"]
            x1, y1 = p1
            x2, y2 = p2
            l, r = min(x1, x2), max(x1, x2)
            t, b = min(y1, y2), max(y1, y2)

            px, py = point
            if l <= px <= r and t <= py <= b:
                return "center"

        return None

    def move(self, delta):
        if "p1" in self.data:
            dx, dy = delta
            p1 = self.data["p1"]
            p2 = self.data["p2"]

            new_p1 = (p1[0] + dx, p1[1] + dy)
            new_p2 = (p2[0] + dx, p2[1] + dy)
            self.update(new_p1, new_p2)

    def adjust(self, handle_id, new_pos):
        # handle_id is tl, tr, bl, br
        # We need to update p1/p2 such that the rect matches the new corner
        # This implies p1/p2 might swap.

        if "p1" not in self.data:
            return

        # Current bounds
        p1 = self.data["p1"]
        p2 = self.data["p2"]
        l, r = min(p1[0], p2[0]), max(p1[0], p2[0])
        t, b = min(p1[1], p2[1]), max(p1[1], p2[1])

        nx, ny = new_pos

        if handle_id == "tl":
            l, t = nx, ny
        elif handle_id == "tr":
            r, t = nx, ny
        elif handle_id == "bl":
            l, b = nx, ny
        elif handle_id == "br":
            r, b = nx, ny

        # Reconstruct p1, p2
        self.update((l, t), (r, b))

    def _update_visuals_from_data(self):
        if "p1" in self.data and "p2" in self.data:
            self.update(self.data["p1"], self.data["p2"])

    def get_region(self, data):
        """
        Extract rectangular region from data.

        Args:
            data: Array with shape (..., Y, X)

        Returns:
            Cropped array with shape (..., height, width)
        """
        x1, y1 = self.data["p1"]
        x2, y2 = self.data["p2"]

        # Normalize to min/max
        xmin, xmax = int(min(x1, x2)), int(max(x1, x2))
        ymin, ymax = int(min(y1, y2)), int(max(y1, y2))

        # Clamp to bounds
        Y, X = data.shape[-2:]
        xmin, xmax = max(0, xmin), min(X, xmax)
        ymin, ymax = max(0, ymin), min(Y, ymax)

        return data[..., ymin:ymax, xmin:xmax]


class CircleROI(ROI):
    def __init__(self, view, name="Circle"):
        super().__init__(view, name)
        self.circle = scene.visuals.Ellipse(
            center=(0, 0, 0),
            radius=1,
            border_color="cyan",
            color=(0, 1, 1, 0.1),
            parent=self.view.scene,
        )
        self.circle.set_gl_state(
            preset="translucent",
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
            depth_test=False,
        )
        self.visuals.append(self.circle)

    def update(self, p1, p2):
        # p1 is center, p2 defines radius
        cx, cy = p1
        dx = p2[0] - cx
        dy = p2[1] - cy
        radius = np.sqrt(dx**2 + dy**2)

        self.data = {"center": p1, "edge": p2}

        self.circle.center = (cx, cy, 0)
        self.circle.radius = max(radius, 1e-6)

        self._update_label_position()

        if self.selected:
            self._update_handles()

    def _update_label_position(self):
        if "center" not in self.data:
            return
        cx, cy = self.data["center"]
        # Calculate radius from edge point
        ex, ey = self.data.get("edge", (cx, cy))
        radius = np.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
        # Position label above the circle
        top_y = cy - radius - 5
        self.label_visual.pos = (cx, top_y, 0)

    def _update_handles(self):
        if "center" not in self.data:
            return

        center = self.data["center"]
        edge = self.data["edge"]

        self.handle_points = {"center": center, "edge": edge}

        pts = list(self.handle_points.values())
        self.handle_visual.set_data(
            pos=np.array(pts), face_color="white", size=10
        )

    def hit_test(self, point):
        # 1. Check handles
        hid = super().hit_test(point)
        if hid:
            return hid

        # 2. Check body (inside circle)
        if "center" in self.data:
            cx, cy = self.data["center"]
            px, py = point
            dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            if dist <= self.circle.radius:
                return "center"

        return None

    def move(self, delta):
        if "center" in self.data:
            dx, dy = delta
            cx, cy = self.data["center"]
            ex, ey = self.data["edge"]

            new_center = (cx + dx, cy + dy)
            new_edge = (ex + dx, ey + dy)
            self.update(new_center, new_edge)

    def adjust(self, handle_id, new_pos):
        if "center" not in self.data:
            return

        center = self.data["center"]
        edge = self.data["edge"]

        if handle_id == "center":
            # Moving center moves the whole circle? Or just center (changing radius)?
            # Usually center handle moves the object. But we have 'move' for that.
            # If user drags center handle, they expect move.
            # But here adjust is called when dragging a handle.
            # Let's make center handle move the circle.
            dx = new_pos[0] - center[0]
            dy = new_pos[1] - center[1]
            self.move((dx, dy))
        elif handle_id == "edge":
            # Change radius
            self.update(center, new_pos)

    def _update_visuals_from_data(self):
        if "center" in self.data and "edge" in self.data:
            self.update(self.data["center"], self.data["edge"])

    def get_region(self, data):
        """
        Extract circular region from data.

        Args:
            data: Array with shape (..., Y, X)

        Returns:
            tuple: (region, mask) where region is bounding box
                   and mask is boolean array for circle
        """
        cx, cy = self.data["center"]
        ex, ey = self.data["edge"]
        radius = np.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)

        # Bounding box
        xmin, xmax = int(cx - radius), int(cx + radius + 1)
        ymin, ymax = int(cy - radius), int(cy + radius + 1)

        # Clamp to bounds
        Y, X = data.shape[-2:]
        xmin, xmax = max(0, xmin), min(X, xmax)
        ymin, ymax = max(0, ymin), min(Y, ymax)

        region = data[..., ymin:ymax, xmin:xmax]

        # Create circular mask
        h, w = ymax - ymin, xmax - xmin
        yy, xx = np.ogrid[:h, :w]
        local_cx, local_cy = cx - xmin, cy - ymin
        mask = ((xx - local_cx) ** 2 + (yy - local_cy) ** 2) <= radius**2

        return region, mask


class LineROI(ROI):
    def __init__(self, view, name="Line"):
        super().__init__(view, name)
        self.line = scene.visuals.Line(
            pos=np.zeros((2, 3)),
            color="magenta",
            width=2,
            parent=self.view.scene,
        )
        self.visuals.append(self.line)

    def update(self, p1, p2):
        self.data = {"p1": p1, "p2": p2}

        pos = np.zeros((2, 3))
        pos[0, :2] = p1
        pos[1, :2] = p2
        self.line.set_data(pos=pos)

        self._update_label_position()

        if self.selected:
            self._update_handles()

    def _update_label_position(self):
        if "p1" not in self.data:
            return
        p1 = self.data["p1"]
        p2 = self.data["p2"]
        # Midpoint of line, slightly above
        mx = (p1[0] + p2[0]) / 2
        my = min(p1[1], p2[1]) - 5
        self.label_visual.pos = (mx, my, 0)

    def _update_handles(self):
        if "p1" not in self.data:
            return

        p1 = self.data["p1"]
        p2 = self.data["p2"]

        self.handle_points = {"p1": p1, "p2": p2}

        pts = list(self.handle_points.values())
        self.handle_visual.set_data(
            pos=np.array(pts), face_color="white", size=10
        )

    def hit_test(self, point):
        # 1. Check handles
        hid = super().hit_test(point)
        if hid:
            return hid

        # 2. Check proximity to line
        if "p1" in self.data:
            p1 = np.array(self.data["p1"])
            p2 = np.array(self.data["p2"])
            p = np.array(point)

            l2 = np.sum((p1 - p2) ** 2)
            if l2 == 0:
                return None
            t = np.dot(p - p1, p2 - p1) / l2
            t = max(0, min(1, t))
            projection = p1 + t * (p2 - p1)
            dist = np.linalg.norm(p - projection)

            if dist < 5:
                return "center"

        return None

    def move(self, delta):
        if "p1" in self.data:
            dx, dy = delta
            p1 = self.data["p1"]
            p2 = self.data["p2"]

            new_p1 = (p1[0] + dx, p1[1] + dy)
            new_p2 = (p2[0] + dx, p2[1] + dy)
            self.update(new_p1, new_p2)

    def adjust(self, handle_id, new_pos):
        if "p1" not in self.data:
            return

        p1 = self.data["p1"]
        p2 = self.data["p2"]

        if handle_id == "p1":
            self.update(new_pos, p2)
        elif handle_id == "p2":
            self.update(p1, new_pos)

    def _update_visuals_from_data(self):
        if "p1" in self.data and "p2" in self.data:
            self.update(self.data["p1"], self.data["p2"])

    def get_profile(self, data, num_points=None):
        """
        Extract intensity profile along line.

        Args:
            data: Array with shape (..., Y, X)
            num_points: Number of samples (default: line length)

        Returns:
            Array with shape (..., num_points)
        """
        from scipy.ndimage import map_coordinates

        x1, y1 = self.data["p1"]
        x2, y2 = self.data["p2"]

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if num_points is None:
            num_points = max(2, int(np.ceil(length)))

        xs = np.linspace(x1, x2, num_points)
        ys = np.linspace(y1, y2, num_points)
        coords = np.array([ys, xs])  # scipy uses (row, col) order

        # Handle multi-dimensional data
        if data.ndim == 2:
            return map_coordinates(data, coords, order=1)
        else:
            # For (C, Y, X) or similar, extract per channel
            result = []
            for i in range(data.shape[0]):
                result.append(map_coordinates(data[i], coords, order=1))
            return np.stack(result)


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
    LADDER_COLOR = "cyan"
    SAMPLE_COLOR = "orange"

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

    def get_marker_positions(self):
        """Get list of marker y_local positions (sorted)."""
        return sorted(m["y_local"] for m in self._markers)

    def update_marker_labels(self, labels):
        """
        Update marker labels.

        Args:
            labels: List of label strings, one per marker (in order)
        """
        for i, label in enumerate(labels):
            if i < len(self._markers):
                self._markers[i]["label"] = label
                if i < len(self._marker_visuals):
                    _, text_visual = self._marker_visuals[i]
                    text_visual.text = label
                    # Also update visibility based on whether label is non-empty
                    text_visual.visible = self.show_marker_labels and bool(
                        label
                    )

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
                anchor_x="left",
                anchor_y="center",
                parent=self.view.scene,
            )
            text_visual.pos = (x_max + 3, y_global, 0)
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
        text_visual.pos = (x_max + 3, y_global, 0)

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
            - 'center' if body hit (and not locked)
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

        # 3. Check body (only if not locked)
        if not self.locked:
            if x_min <= px <= x_max and y_min <= py <= y_max:
                return "center"

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


class FreehandROI(ROI):
    """
    Freehand line ROI for tracing arbitrary paths.

    Stores a sequence of (x, y) points that define the path.
    """
    def __init__(self, view, name="Freehand"):
        super().__init__(view, name)
        self.points = []  # List of (x, y) tuples
        self.line = scene.visuals.Line(
            pos=np.zeros((0, 3)),
            color="magenta",
            width=2,
            connect="strip",
            parent=self.view.scene,
        )
        self.visuals.append(self.line)

    def add_point(self, point):
        """Add a point to the freehand line during drawing."""
        self.points.append(tuple(point))
        self.data = {"points": self.points}
        self._update_line_visual()
        self._update_label_position()

    def update(self, points):
        """
        Update the freehand line with a list of points.

        Args:
            points: List of (x, y) tuples
        """
        self.points = [tuple(p) for p in points]
        self.data = {"points": self.points}
        self._update_line_visual()
        self._update_label_position()

        if self.selected:
            self._update_handles()

    def _update_line_visual(self):
        """Update the line visual from current points."""
        if len(self.points) < 2:
            # Need at least 2 points for a line
            pos = np.zeros((0, 3))
        else:
            pos = np.zeros((len(self.points), 3))
            for i, (x, y) in enumerate(self.points):
                pos[i, :2] = (x, y)
        self.line.set_data(pos=pos)

    def _update_label_position(self):
        if len(self.points) < 1:
            return
        # Position label at the midpoint of the path
        mid_idx = len(self.points) // 2
        mx, my = self.points[mid_idx]
        self.label_visual.pos = (mx, my - 5, 0)

    def _update_handles(self):
        """Show handles at each point when selected."""
        if len(self.points) < 1:
            return

        # Create handle for each point
        self.handle_points = {i: self.points[i] for i in range(len(self.points))}

        pts = list(self.handle_points.values())
        self.handle_visual.set_data(
            pos=np.array(pts), face_color="white", size=10
        )

    def hit_test(self, point):
        """
        Test if point hits the freehand line or its handles.

        Returns:
            - handle index if handle hit
            - 'center' if line body hit
            - None if no hit
        """
        # 1. Check handles first (if selected)
        hid = super().hit_test(point)
        if hid is not None:
            return hid

        # 2. Check proximity to any line segment
        if len(self.points) < 2:
            return None

        p = np.array(point)
        for i in range(len(self.points) - 1):
            p1 = np.array(self.points[i])
            p2 = np.array(self.points[i + 1])

            # Point-to-segment distance
            l2 = np.sum((p1 - p2) ** 2)
            if l2 == 0:
                continue
            t = np.dot(p - p1, p2 - p1) / l2
            t = max(0, min(1, t))
            projection = p1 + t * (p2 - p1)
            dist = np.linalg.norm(p - projection)

            if dist < 5:  # 5 pixel tolerance
                return "center"

        return None

    def move(self, delta):
        """Move the entire freehand line by delta (dx, dy)."""
        if len(self.points) < 1:
            return

        dx, dy = delta
        new_points = [(x + dx, y + dy) for x, y in self.points]
        self.update(new_points)

    def adjust(self, handle_id, new_pos):
        """Move a specific point to new_pos."""
        if isinstance(handle_id, int) and 0 <= handle_id < len(self.points):
            self.points[handle_id] = tuple(new_pos)
            self.data = {"points": self.points}
            self._update_line_visual()
            self._update_label_position()
            if self.selected:
                self._update_handles()

    def _update_visuals_from_data(self):
        """Rebuild visuals from serialized data."""
        if "points" in self.data:
            self.update(self.data["points"])
