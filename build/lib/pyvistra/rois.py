import numpy as np
from vispy import scene


class ROI:
    # Class-level flag to control label visibility for all ROIs
    show_labels = True

    def __init__(self, view, name="ROI"):
        self.view = view
        self.name = name
        self.group = "Default"
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
            "group": self.group,
            "data": self.data,
        }

    def from_dict(self, data, group=None):
        self.data = data
        if group is not None:
            self.group = group
        self._update_visuals_from_data()

    def _update_visuals_from_data(self):
        pass


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


class PointROI(RectangleROI):
    """Focused editable point proxy reusing RectangleROI handles/label behavior."""

    def __init__(
        self,
        view,
        name="Point",
        *,
        point_id: int | None = None,
        on_change=None,
    ):
        super().__init__(view, name=name)
        self.point_id = point_id
        self._on_change = on_change

    def set_from_point(self, x: float, y: float, size_data: float):
        half = max(float(size_data) * 0.5, 1e-6)
        p1 = (float(x) - half, float(y) - half)
        p2 = (float(x) + half, float(y) + half)
        self.update(p1, p2)

    def update(self, p1, p2):
        super().update(p1, p2)
        if self._on_change:
            self._on_change(self._current_state())

    def move(self, delta):
        super().move(delta)
        if self._on_change:
            self._on_change(self._current_state())

    def adjust(self, handle_id, new_pos):
        super().adjust(handle_id, new_pos)
        if self._on_change:
            self._on_change(self._current_state())

    def _current_state(self):
        p1 = self.data.get("p1", (0.0, 0.0))
        p2 = self.data.get("p2", (0.0, 0.0))
        cx = 0.5 * (float(p1[0]) + float(p2[0]))
        cy = 0.5 * (float(p1[1]) + float(p2[1]))
        size = max(abs(float(p2[0]) - float(p1[0])), abs(float(p2[1]) - float(p1[1])))
        return {"point_id": self.point_id, "x": cx, "y": cy, "box_size_data": size}
