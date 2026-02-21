from datetime import datetime, timedelta

import numpy as np
import pytest

from pyvistra.io import build_z_projection_metadata, project_z_max


def test_build_z_projection_metadata_normalizes_scale_and_timestamps():
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    source_meta = {
        "filename": "movie.ims",
        "scale": (0.5, 0.2, 0.2),
        "timestamps": [
            t0,
            t0 + timedelta(seconds=1),
            "2024-01-01T12:00:02",
            None,
        ],
        "timestamp_seconds": [0.0, 1.0, 2.0, 3.0],
    }

    out_meta = build_z_projection_metadata(
        metadata=source_meta,
        source_shape=(3, 5, 2, 8, 9),
        z_range=(1, 3),
        method="max",
    )

    assert out_meta["shape"] == (3, 1, 2, 8, 9)
    assert out_meta["scale"] == (1.5, 0.2, 0.2)
    assert out_meta["projection"] == {
        "axis": "z",
        "method": "max",
        "z_range": [1, 3],
    }
    assert len(out_meta["timestamps"]) == 3
    assert out_meta["timestamps"][0] == t0
    assert out_meta["timestamps"][1] == t0 + timedelta(seconds=1)
    assert out_meta["timestamps"][2] == t0 + timedelta(seconds=2)
    assert out_meta["timestamp_seconds"] == [0.0, 1.0, 2.0]


def test_project_z_max_projects_data_and_attaches_metadata():
    data = np.arange(2 * 4 * 1 * 3 * 3, dtype=np.uint16).reshape(
        (2, 4, 1, 3, 3)
    )
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    meta = {
        "filename": "sample",
        "scale": (2.0, 1.0, 1.0),
        "timestamps": [t0, t0 + timedelta(seconds=5)],
    }

    out = project_z_max(data, z_range=(1, 2), metadata=meta)
    try:
        expected = np.max(data[:, 1:3, :, :, :], axis=1, keepdims=True)
        np.testing.assert_array_equal(out[:], expected)
        assert out.shape == (2, 1, 1, 3, 3)
        assert out.metadata["scale"] == (4.0, 1.0, 1.0)
        assert out.metadata["timestamps"] == [t0, t0 + timedelta(seconds=5)]
    finally:
        out.release()


def test_project_z_max_rejects_invalid_z_range():
    data = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    with pytest.raises(ValueError):
        project_z_max(data, z_range=(0, 2))
