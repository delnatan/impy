"""RegionStatisticsDialog: same-window stats plus cross-window physical-unit
correctness (mirrors tests/test_profile_dialogs.py's approach for line/
radial profile).
"""

import numpy as np

from pyvistra import imshow
from pyvistra.data.shapes import RECTANGLE, AddShape
from pyvistra.widgets.region_statistics_dialog import RegionStatisticsDialog


def _table_value(dlg, row, col_key):
    col = ["channel"] + ["mean", "std", "sum", "min", "max", "n_pixels"]
    j = col.index(col_key)
    item = dlg._table.item(row, j)
    return item.text() if item is not None else None


def test_same_window_stats_match_hand_computed():
    img = np.zeros((20, 20), dtype=np.float32)
    img[5:10, 5:10] = 4.0
    win = imshow(img[None, None, None, :, :], "A")

    layer = win.add_shape_layer("Shapes")
    cmd = AddShape(RECTANGLE, [5, 5, 10, 10], t=0, z=0, label="rect")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = RegionStatisticsDialog(win, layer, sid)
    assert dlg._target_window is win
    assert dlg._table.rowCount() == 1
    assert _table_value(dlg, 0, "mean") == "4.000"
    assert _table_value(dlg, 0, "n_pixels") == "25"
    dlg.close()


def test_cross_window_stats_match_physical_region():
    # Window A: 100x100px at 0.1um/px. Bright block at physical
    # (2,2)-(4,4)um -> pixel (20,20)-(40,40).
    img_a = np.zeros((100, 100), dtype=np.float32)
    win_a = imshow(img_a[None, None, None, :, :], "A", scale=(1.0, 0.1, 0.1))

    # Window B: 50x50px at 0.2um/px -- same physical FOV. Same physical
    # region -> pixel (10,10)-(20,20), filled with a different value so we
    # can tell we sampled B and not A.
    img_b = np.zeros((50, 50), dtype=np.float32)
    img_b[10:20, 10:20] = 7.0
    win_b = imshow(img_b[None, None, None, :, :], "B", scale=(1.0, 0.2, 0.2))

    layer = win_a.add_shape_layer("Shapes")
    # Rect in A's pixel space: physical (2,2)-(4,4)um.
    cmd = AddShape(RECTANGLE, [20, 20, 40, 40], t=0, z=0, label="rect")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = RegionStatisticsDialog(win_a, layer, sid)
    idx = dlg._target_combo.findData(win_b)
    assert idx >= 0
    dlg._target_combo.setCurrentIndex(idx)

    assert dlg._target_window is win_b
    assert _table_value(dlg, 0, "mean") == "7.000"
    assert _table_value(dlg, 0, "n_pixels") == "100"  # 10x10 px in B
    dlg.close()


def test_cross_window_refuses_missing_scale():
    img_a = np.zeros((20, 20), dtype=np.float32)
    win_a = imshow(img_a[None, None, None, :, :], "A", scale=(1.0, 0.1, 0.1))
    img_b = np.zeros((20, 20), dtype=np.float32)
    win_b = imshow(img_b[None, None, None, :, :], "B")  # no scale

    layer = win_a.add_shape_layer("Shapes")
    cmd = AddShape(RECTANGLE, [5, 5, 10, 10], t=0, z=0, label="rect")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = RegionStatisticsDialog(win_a, layer, sid)
    idx = dlg._target_combo.findData(win_b)
    dlg._target_combo.setCurrentIndex(idx)

    assert dlg._table.rowCount() == 0
    assert "calibration" in dlg._status.text().lower()
    dlg.close()


def test_cross_window_refuses_frequency_space_mismatch():
    img_a = np.zeros((20, 20), dtype=np.float32)
    win_a = imshow(img_a[None, None, None, :, :], "A", scale=(1.0, 0.1, 0.1))
    img_b = np.zeros((20, 20), dtype=np.float32)
    win_b = imshow(img_b[None, None, None, :, :], "B_freq", scale=(1.0, 0.2, 0.2))
    win_b.pixel_space = "frequency"

    layer = win_a.add_shape_layer("Shapes")
    cmd = AddShape(RECTANGLE, [5, 5, 10, 10], t=0, z=0, label="rect")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = RegionStatisticsDialog(win_a, layer, sid)
    idx = dlg._target_combo.findData(win_b)
    dlg._target_combo.setCurrentIndex(idx)

    assert dlg._table.rowCount() == 0
    assert "pixel space" in dlg._status.text().lower()
    dlg.close()


def test_target_window_close_falls_back_to_source():
    img_a = np.zeros((20, 20), dtype=np.float32)
    win_a = imshow(img_a[None, None, None, :, :], "A")
    img_b = np.zeros((20, 20), dtype=np.float32)
    win_b = imshow(img_b[None, None, None, :, :], "B")

    layer = win_a.add_shape_layer("Shapes")
    cmd = AddShape(RECTANGLE, [5, 5, 10, 10], t=0, z=0, label="rect")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = RegionStatisticsDialog(win_a, layer, sid)
    idx = dlg._target_combo.findData(win_b)
    dlg._target_combo.setCurrentIndex(idx)
    assert dlg._target_window is win_b

    win_b.close()

    assert dlg._target_window is win_a
    dlg.close()


def test_source_window_close_closes_dialog():
    img = np.zeros((20, 20), dtype=np.float32)
    win = imshow(img[None, None, None, :, :], "A")

    layer = win.add_shape_layer("Shapes")
    cmd = AddShape(RECTANGLE, [5, 5, 10, 10], t=0, z=0, label="rect")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = RegionStatisticsDialog(win, layer, sid)
    closed = []
    dlg.close = lambda: closed.append(1)

    win.close()

    assert closed
