"""Tests for the pure-numpy radial binning in widgets/radial_profile_dialog.py."""

import numpy as np

from pyvistra.widgets.radial_profile_dialog import radial_profile


def test_radial_profile_uniform_disk():
    size = 41
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2
    img = np.full((size, size), 3.0, dtype=np.float32)

    radii, profile = radial_profile(img, cx, cy, 10.0)
    assert radii[0] == 0.0
    assert radii[-1] == 10.0
    np.testing.assert_allclose(profile, 3.0)


def test_radial_profile_excludes_bounding_box_corners():
    """Regression test: the crop is a square bounding box, so its corners
    sit up to radius*sqrt(2) away from center -- outside the requested
    circle. The outermost bin must not be diluted by them.
    """
    size = 200
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2
    r_true = np.hypot(xx - cx, yy - cy)

    # Bright out to r=51 (comfortably covers the outermost bin's true
    # population, r in [50, 51)); dark only in the true bounding-box
    # corners (r > 51, up to radius*sqrt(2) ~= 70.7 for radius=50).
    img = np.where(r_true <= 51, 100.0, 0.0).astype(np.float32)

    radii, profile = radial_profile(img, cx, cy, 50.0)
    assert len(profile) == 51  # floor(50) + 1
    np.testing.assert_allclose(profile[48:51], 100.0)


def test_radial_profile_out_of_bounds_returns_none():
    img = np.zeros((50, 50), dtype=np.float32)
    radii, profile = radial_profile(img, -1000.0, -1000.0, 5.0)
    assert radii is None
    assert profile is None
