"""Multi-window physical-unit correctness for Line/Radial Profile dialogs.

Comparing windows with different pixel sizes must sample each window in
its own pixel grid (converting through a shared physical coordinate),
not reuse one window's raw pixel coordinates against another's array.
"""

import numpy as np

from pyvistra import imshow
from pyvistra.data.shapes import CIRCLE, LINE, AddShape
from pyvistra.widgets import get_line_profile_dialog, get_radial_profile_dialog


def test_line_profile_aligns_windows_with_different_pixel_sizes():
    # Window A: 100x100, 0.1 um/px. Bright spot at physical (5, 5) um ->
    # pixel (50, 50).
    img_a = np.zeros((100, 100), dtype=np.float32)
    img_a[48:53, 48:53] = 100.0
    win_a = imshow(img_a[None, None, None, :, :], "A", scale=(1.0, 0.1, 0.1))

    # Window B: 50x50, 0.2 um/px -- same physical field of view (10um).
    # Same physical bright spot at (5, 5) um -> pixel (25, 25) in B.
    img_b = np.zeros((50, 50), dtype=np.float32)
    img_b[23:26, 23:26] = 200.0
    win_b = imshow(img_b[None, None, None, :, :], "B", scale=(1.0, 0.2, 0.2))

    # Line through the bright spot in A: physical (0,5)->(10,5) um.
    layer = win_a.add_shape_layer("Shapes")
    cmd = AddShape(LINE, [0, 50, 100, 50], t=0, z=0, label="line")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = get_line_profile_dialog()
    dlg._clear_series()
    dlg.set_shape_source(win_a, layer, sid)
    dlg._add_series_for_window(win_b)
    dlg._refresh_profiles()

    by_window = {s["window_id"]: s for s in dlg._computed_series}
    assert win_a.window_id in by_window
    assert win_b.window_id in by_window

    peak_a = float(by_window[win_a.window_id]["distances_um"][
        np.argmax(by_window[win_a.window_id]["values"])
    ])
    peak_b = float(by_window[win_b.window_id]["distances_um"][
        np.argmax(by_window[win_b.window_id]["values"])
    ])
    assert abs(peak_a - 5.0) < 0.5
    assert abs(peak_b - 5.0) < 0.5


def test_line_profile_skips_mismatched_pixel_space():
    img_a = np.zeros((20, 20), dtype=np.float32)
    win_a = imshow(img_a[None, None, None, :, :], "A", scale=(1.0, 0.1, 0.1))

    img_b = np.zeros((20, 20), dtype=np.float32)
    win_b = imshow(img_b[None, None, None, :, :], "B_freq", scale=(1.0, 0.2, 0.2))
    win_b.pixel_space = "frequency"

    layer = win_a.add_shape_layer("Shapes")
    cmd = AddShape(LINE, [0, 10, 19, 10], t=0, z=0, label="line")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = get_line_profile_dialog()
    dlg._clear_series()
    dlg.set_shape_source(win_a, layer, sid)
    dlg._add_series_for_window(win_b)
    dlg._refresh_profiles()

    assert len(dlg._computed_series) == 1
    assert dlg._computed_series[0]["window_id"] == win_a.window_id
    assert "skipped" in dlg.status_label.text()


def test_radial_profile_aligns_windows_with_different_pixel_sizes():
    size_a = 200
    yy, xx = np.mgrid[0:size_a, 0:size_a]
    c_a = size_a // 2
    r_a = np.hypot(xx - c_a, yy - c_a)
    img_a = np.where(r_a <= 51, 100.0, 0.0).astype(np.float32)
    win_a = imshow(img_a[None, None, None, :, :], "A", scale=(1.0, 0.1, 0.1))

    size_b = 100
    yy, xx = np.mgrid[0:size_b, 0:size_b]
    c_b = size_b // 2
    r_b = np.hypot(xx - c_b, yy - c_b)
    img_b = np.where(r_b <= 26, 200.0, 0.0).astype(np.float32)
    win_b = imshow(img_b[None, None, None, :, :], "B", scale=(1.0, 0.2, 0.2))

    layer = win_a.add_shape_layer("Shapes")
    cmd = AddShape(CIRCLE, [c_a, c_a, c_a + 50, c_a], t=0, z=0, label="c")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = get_radial_profile_dialog()
    dlg._clear_series()
    dlg.set_shape_source(win_a, layer, sid)
    dlg._add_series_for_window(win_b)
    dlg.mode_combo.setCurrentIndex(0)  # radial
    dlg._refresh_profiles()

    by_window = {s["window_id"]: s for s in dlg._computed_series}
    assert len(by_window[win_a.window_id]["values"]) == 51
    assert len(by_window[win_b.window_id]["values"]) == 26
    assert by_window[win_a.window_id]["values"][-1] > 90
    assert by_window[win_b.window_id]["values"][-1] > 180


def test_radial_profile_skips_mismatched_pixel_space():
    img_a = np.zeros((100, 100), dtype=np.float32)
    win_a = imshow(img_a[None, None, None, :, :], "A", scale=(1.0, 0.1, 0.1))

    img_b = np.zeros((50, 50), dtype=np.float32)
    win_b = imshow(img_b[None, None, None, :, :], "B_freq", scale=(1.0, 0.2, 0.2))
    win_b.pixel_space = "frequency"

    layer = win_a.add_shape_layer("Shapes")
    cmd = AddShape(CIRCLE, [50, 50, 80, 50], t=0, z=0, label="c")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = get_radial_profile_dialog()
    dlg._clear_series()
    dlg.set_shape_source(win_a, layer, sid)
    dlg._add_series_for_window(win_b)
    dlg._refresh_profiles()

    assert len(dlg._computed_series) == 1
    assert dlg._computed_series[0]["window_id"] == win_a.window_id
    assert "skipped" in dlg.status_label.text()
