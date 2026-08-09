import numpy as np

from pyvistra.data.calibration import (
    points_phys_to_px,
    points_px_to_phys,
    radius_phys_to_px,
    radius_px_to_phys,
    sample_along_path,
    window_is_frequency_space,
    window_scale_yx,
)


class _FakeWindow:
    def __init__(self, scale=None, pixel_space="real"):
        self.meta = {"scale": scale} if scale is not None else {}
        self.pixel_space = pixel_space


def test_window_scale_yx_returns_sy_sx():
    win = _FakeWindow(scale=(1.0, 0.2, 0.1))
    assert window_scale_yx(win) == (0.2, 0.1)


def test_window_scale_yx_none_cases():
    assert window_scale_yx(None) is None
    assert window_scale_yx(_FakeWindow(scale=None)) is None
    assert window_scale_yx(_FakeWindow(scale=(1.0, 0.2))) is None  # too short


def test_window_is_frequency_space():
    assert window_is_frequency_space(_FakeWindow(pixel_space="frequency")) is True
    assert window_is_frequency_space(_FakeWindow(pixel_space="real")) is False
    assert window_is_frequency_space(None) is False


def test_points_px_to_phys_and_back_roundtrip():
    scale_yx = (0.2, 0.1)  # sy, sx
    points_xy = np.array([[0.0, 0.0], [10.0, 20.0], [5.0, 5.0]])

    phys = points_px_to_phys(points_xy, scale_yx)
    # x scaled by sx, y scaled by sy
    expected = points_xy * np.array([0.1, 0.2])
    assert np.allclose(phys, expected)

    back = points_phys_to_px(phys, scale_yx)
    assert np.allclose(back, points_xy)


def test_points_px_to_phys_independent_axes():
    # Anisotropic scale: an off-axis segment should not be approximated by
    # an averaged factor -- each axis scales independently.
    scale_yx = (1.0, 2.0)
    points_xy = np.array([[1.0, 1.0]])
    phys = points_px_to_phys(points_xy, scale_yx)
    assert np.allclose(phys, [[2.0, 1.0]])  # x*sx=1*2, y*sy=1*1


def test_radius_px_to_phys_and_back_roundtrip():
    scale_yx = (0.2, 0.1)
    radius_phys = radius_px_to_phys(10.0, scale_yx)
    assert radius_phys == 10.0 * 0.5 * (0.2 + 0.1)

    back = radius_phys_to_px(radius_phys, scale_yx)
    assert abs(back - 10.0) < 1e-9


def test_sample_along_path_matches_direct_map_coordinates():
    from scipy.ndimage import map_coordinates

    image = np.arange(25, dtype=float).reshape(5, 5)
    path_xy = np.array([[0.0, 0.0], [2.5, 1.5], [4.0, 4.0]])

    got = sample_along_path(image, path_xy)

    xs = path_xy[:, 0]
    ys = path_xy[:, 1]
    expected = map_coordinates(image, np.array([ys, xs]), order=1, mode="nearest")
    assert np.allclose(got, expected)
