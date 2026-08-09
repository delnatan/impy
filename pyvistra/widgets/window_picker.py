"""Enumerate open image windows for "sample this shape on another window"
pickers (region statistics, kymograph).

Deliberately not :func:`pyvistra.ui.manager.compatible_windows` — that
filters to windows whose YX pixel array shape matches exactly, which is
precisely the constraint cross-window physical-coordinate sampling (see
``data/calibration.py``) is meant to lift: two windows can have different
pixel dimensions and still be valid targets as long as they're both
calibrated (``meta["scale"]``) and the same kind of pixel space (real vs.
frequency) — checked at sample time, not at listing time, so this stays a
plain enumeration.
"""

from __future__ import annotations

from ..ui.manager import manager


def list_other_image_windows(exclude) -> list:
    """Open windows with image data, excluding *exclude*."""
    return [
        w
        for w in manager.get_all().values()
        if w is not exclude and getattr(w, "img_data", None) is not None
    ]
