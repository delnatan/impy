import matplotlib.pyplot as plt
import numpy as np
from magicgui import magicgui

from .rois import CircleROI, LineROI, RectangleROI

# ---- Lane Analysis Functions ----


def get_rect_bounds(roi):
    """Get normalized bounds (left, right, top, bottom) from a RectangleROI."""
    p1 = roi.data.get("p1", (0, 0))
    p2 = roi.data.get("p2", (0, 0))
    left = min(p1[0], p2[0])
    right = max(p1[0], p2[0])
    top = min(p1[1], p2[1])
    bottom = max(p1[1], p2[1])
    return left, right, top, bottom


def align_lanes(rois, reference="top"):
    """
    Align RectangleROIs to a common edge.

    For gel analysis, lanes should share a common origin (loading wells).
    This function aligns the top edges of all rectangles.

    Args:
        rois: List of RectangleROI objects
        reference: 'top' or 'bottom' - which edge to align

    Returns:
        int: Number of ROIs aligned
    """
    # Filter to only RectangleROIs
    rect_rois = [r for r in rois if isinstance(r, RectangleROI)]

    if len(rect_rois) < 2:
        print("Need at least 2 rectangle ROIs to align")
        return 0

    # Find reference y-coordinate
    if reference == "top":
        # Find minimum top (smallest y = topmost in image coords)
        target_y = min(get_rect_bounds(r)[2] for r in rect_rois)
    else:  # bottom
        # Find maximum bottom
        target_y = max(get_rect_bounds(r)[3] for r in rect_rois)

    aligned_count = 0
    for roi in rect_rois:
        left, right, top, bottom = get_rect_bounds(roi)

        if reference == "top":
            delta_y = target_y - top
        else:
            delta_y = target_y - bottom

        if abs(delta_y) > 0.1:  # Only move if needed
            roi.move((0, delta_y))
            aligned_count += 1

    print(
        f"Aligned {aligned_count} lanes to {reference} edge (y={target_y:.1f})"
    )
    return aligned_count


@magicgui(call_button="Crop Image")
def crop_image(image_data, roi: RectangleROI):
    """
    Crop the image to the ROI bounds and show in new window.
    """
    from .ui import imshow

    if not isinstance(roi, RectangleROI):
        print("Error: ROI must be a RectangleROI")
        return

    p1 = roi.data.get("p1")
    p2 = roi.data.get("p2")
    if not p1 or not p2:
        return

    x1, y1 = p1
    x2, y2 = p2
    l, r = int(min(x1, x2)), int(max(x1, x2))
    t, b = int(min(y1, y2)), int(max(y1, y2))

    # Handle 2D vs 5D
    # If 5D, crop Y and X dims
    if image_data.ndim == 5:
        # (T, Z, C, Y, X)
        cropped = image_data[:, :, :, t:b, l:r]
    elif image_data.ndim == 2:
        cropped = image_data[t:b, l:r]
    else:
        print(f"Unsupported dims: {image_data.ndim}")
        return

    imshow(cropped, title=f"Crop: {roi.name}")


@magicgui(call_button="Measure")
def measure_intensity(image_data, roi):
    """
    Measure mean/std intensity within ROI bounding box.
    """
    # Get bounds
    if isinstance(roi, RectangleROI):
        p1 = roi.data.get("p1")
        p2 = roi.data.get("p2")
        l, r = int(min(p1[0], p2[0])), int(max(p1[0], p2[0]))
        t, b = int(min(p1[1], p2[1])), int(max(p1[1], p2[1]))
    elif isinstance(roi, CircleROI):
        c = roi.data.get("center")
        r_val = roi.circle.radius
        l, r = int(c[0] - r_val), int(c[0] + r_val)
        t, b = int(c[1] - r_val), int(c[1] + r_val)
    else:
        print("Unsupported ROI for measure")
        return

    if image_data.ndim == 2:
        # Clip
        h, w = image_data.shape
        l = max(0, l)
        r = min(w, r)
        t = max(0, t)
        b = min(h, b)

        region = image_data[t:b, l:r]
        mn = np.mean(region)
        std = np.std(region)
        print(f"ROI {roi.name}: Mean={mn:.2f}, Std={std:.2f}")
    else:
        print("Measurement currently supports 2D slice only")
