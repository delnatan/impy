"""KymographDialog cross-window support: path conversion, refusal policies,
T/Z range refresh, target-close fallback, and end-to-end sampling
correctness (worker run directly, per this project's pattern for testing
threaded workers without a live QThread/event loop)."""

import numpy as np

from pyvistra import imshow
from pyvistra.data.shapes import LINE, AddShape
from pyvistra.widgets.kymograph_dialog import KymographDialog, KymographWorker


def _make_dialog(win, layer, sid):
    return KymographDialog(win, layer, sid)


def test_default_target_is_source_and_ranges_match():
    img = np.zeros((5, 1, 1, 20, 20), dtype=np.float32)
    win = imshow(img, "A")
    layer = win.add_shape_layer("Shapes")
    cmd = AddShape(LINE, [0, 10, 19, 10], t=0, z=0, label="line")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = _make_dialog(win, layer, sid)
    assert dlg._target_window is win
    assert dlg._t_end.maximum() == 4  # T=5 -> max index 4
    dlg.close()


def test_target_switch_updates_t_range():
    img_a = np.zeros((10, 1, 1, 20, 20), dtype=np.float32)
    win_a = imshow(img_a, "A")
    img_b = np.zeros((3, 1, 1, 20, 20), dtype=np.float32)
    win_b = imshow(img_b, "B")

    layer = win_a.add_shape_layer("Shapes")
    cmd = AddShape(LINE, [0, 10, 19, 10], t=0, z=0, label="line")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = _make_dialog(win_a, layer, sid)
    assert dlg._t_end.maximum() == 9  # T=10 in A

    idx = dlg._target_combo.findData(win_b)
    dlg._target_combo.setCurrentIndex(idx)
    assert dlg._target_window is win_b
    assert dlg._t_end.maximum() == 2  # T=3 in B -- range refreshed, not stale
    dlg.close()


def test_resample_coords_converts_through_physical_space():
    img_a = np.zeros((2, 1, 1, 100, 100), dtype=np.float32)
    win_a = imshow(img_a, "A", scale=(1.0, 0.1, 0.1))
    img_b = np.zeros((2, 1, 1, 50, 50), dtype=np.float32)
    win_b = imshow(img_b, "B", scale=(1.0, 0.2, 0.2))

    layer = win_a.add_shape_layer("Shapes")
    # Physical (0,5)um -> (10,5)um in A's pixel space.
    cmd = AddShape(LINE, [0, 50, 100, 50], t=0, z=0, label="line")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = _make_dialog(win_a, layer, sid)
    dlg._npoints.setValue(11)

    idx = dlg._target_combo.findData(win_b)
    dlg._target_combo.setCurrentIndex(idx)

    coords = dlg._resample_coords()
    assert coords is not None
    # Same physical endpoints, converted into B's pixel grid (0.2um/px):
    # (0,5)um -> (0,25)px, (10,5)um -> (50,25)px.
    np.testing.assert_allclose(coords[0], [0.0, 25.0], atol=1e-6)
    np.testing.assert_allclose(coords[-1], [50.0, 25.0], atol=1e-6)
    dlg.close()


def test_cross_window_refuses_missing_scale():
    img_a = np.zeros((2, 1, 1, 20, 20), dtype=np.float32)
    win_a = imshow(img_a, "A", scale=(1.0, 0.1, 0.1))
    img_b = np.zeros((2, 1, 1, 20, 20), dtype=np.float32)
    win_b = imshow(img_b, "B")  # no scale

    layer = win_a.add_shape_layer("Shapes")
    cmd = AddShape(LINE, [0, 10, 19, 10], t=0, z=0, label="line")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = _make_dialog(win_a, layer, sid)
    idx = dlg._target_combo.findData(win_b)
    dlg._target_combo.setCurrentIndex(idx)

    assert dlg._cross_window_error() is not None
    assert "calibration" in dlg._cross_window_error().lower()
    dlg.close()


def test_cross_window_refuses_frequency_space_mismatch():
    img_a = np.zeros((2, 1, 1, 20, 20), dtype=np.float32)
    win_a = imshow(img_a, "A", scale=(1.0, 0.1, 0.1))
    img_b = np.zeros((2, 1, 1, 20, 20), dtype=np.float32)
    win_b = imshow(img_b, "B_freq", scale=(1.0, 0.2, 0.2))
    win_b.pixel_space = "frequency"

    layer = win_a.add_shape_layer("Shapes")
    cmd = AddShape(LINE, [0, 10, 19, 10], t=0, z=0, label="line")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = _make_dialog(win_a, layer, sid)
    idx = dlg._target_combo.findData(win_b)
    dlg._target_combo.setCurrentIndex(idx)

    assert dlg._cross_window_error() is not None
    assert "pixel space" in dlg._cross_window_error().lower()
    dlg.close()


def test_target_window_close_falls_back_to_source():
    img_a = np.zeros((2, 1, 1, 20, 20), dtype=np.float32)
    win_a = imshow(img_a, "A")
    img_b = np.zeros((2, 1, 1, 20, 20), dtype=np.float32)
    win_b = imshow(img_b, "B")

    layer = win_a.add_shape_layer("Shapes")
    cmd = AddShape(LINE, [0, 10, 19, 10], t=0, z=0, label="line")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = _make_dialog(win_a, layer, sid)
    idx = dlg._target_combo.findData(win_b)
    dlg._target_combo.setCurrentIndex(idx)
    assert dlg._target_window is win_b

    win_b.close()

    assert dlg._target_window is win_a
    dlg.close()


def test_worker_samples_target_window_at_correct_physical_position():
    """End-to-end: run KymographWorker directly (no QThread/event loop,
    matching this project's pattern for testing threaded workers) against
    a target window with a different pixel size than the source, and
    confirm it samples the correct physical location."""
    from pyvistra.io import ImageBuffer

    img_a = np.zeros((3, 1, 1, 100, 100), dtype=np.float32)
    win_a = imshow(img_a, "A", scale=(1.0, 0.1, 0.1))

    img_b = np.zeros((3, 1, 1, 50, 50), dtype=np.float32)
    img_b[:, 0, 0, 25, :] = 42.0  # bright row at physical y=5um in B
    win_b = imshow(img_b, "B", scale=(1.0, 0.2, 0.2))

    layer = win_a.add_shape_layer("Shapes")
    # Horizontal line at physical y=5um in A (pixel y=50), spanning x.
    cmd = AddShape(LINE, [0, 50, 99, 50], t=0, z=0, label="line")
    layer.undo_stack.push(cmd, layer.data)
    sid = cmd.shape_id

    dlg = _make_dialog(win_a, layer, sid)
    dlg._npoints.setValue(20)
    idx = dlg._target_combo.findData(win_b)
    dlg._target_combo.setCurrentIndex(idx)
    assert dlg._cross_window_error() is None

    coords = dlg._resample_coords()
    buffer = ImageBuffer(shape=(1, 1, 1, 3, 20), dtype=np.float32)
    worker = KymographWorker(
        win_b.img_data, buffer,
        {"coords": coords, "z": 0, "t_start": 0, "t_end": 2},
    )
    worker.run()

    result = np.asarray(buffer[0, 0, 0, :, :])
    # Every sampled point lands on B's bright row (y=25px) for every frame.
    assert np.allclose(result, 42.0)
    dlg.close()
