"""Tests for readers/imaris_writer.py."""

import h5py
import numpy as np

from pyvistra.readers.imaris_writer import save_imaris


def _read_unit_attr(filepath):
    with h5py.File(filepath, "r") as f:
        raw = f["DataSetInfo/Image"].attrs["Unit"]
    return b"".join(raw).decode("utf-8")


def test_save_imaris_labels_frequency_space_scale_correctly(tmp_path):
    """FFT output tags itself space="frequency" (see fft_dialog.py) --
    saving it to file must not silently claim its cycles/um scale is a
    real-space "um" pixel size.
    """
    data = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
    out = tmp_path / "fft_result.ims"

    save_imaris(
        str(out),
        data,
        metadata={"scale": (1.0, 0.1, 0.1), "space": "frequency"},
        resolution_levels=False,
    )

    assert _read_unit_attr(str(out)) == "1/um"


def test_save_imaris_labels_real_space_scale_as_um(tmp_path):
    data = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
    out = tmp_path / "real_space.ims"

    save_imaris(
        str(out),
        data,
        metadata={"scale": (1.0, 0.1, 0.1)},
        resolution_levels=False,
    )

    assert _read_unit_attr(str(out)) == "um"
