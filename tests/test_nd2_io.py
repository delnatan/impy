import numpy as np
import tifffile

from pyvistra.io import (
    _extract_nd2_channels,
    _normalize_nd2_to_5d,
    load_image,
    save_tiff,
)


def test_normalize_nd2_to_5d_reorders_tczyx():
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)  # T, C, Z, Y, X
    sizes = {"T": 2, "C": 3, "Z": 4, "Y": 5, "X": 6}

    out, collapsed = _normalize_nd2_to_5d(data, sizes)

    assert out.shape == (2, 4, 3, 5, 6)
    assert collapsed == []


def test_normalize_nd2_to_5d_collapses_extra_dims_into_t():
    data = np.zeros((2, 3, 4, 5, 6, 7), dtype=np.uint16)  # P, T, Z, C, Y, X
    sizes = {"P": 2, "T": 3, "Z": 4, "C": 5, "Y": 6, "X": 7}

    out, collapsed = _normalize_nd2_to_5d(data, sizes)

    assert out.shape == (6, 4, 5, 6, 7)
    assert collapsed == ["P"]


def test_extract_nd2_channels_maps_common_fields():
    metadata = {
        "channels": [
            {
                "channel": {
                    "name": "GFP",
                    "emissionLambdaNm": 510.0,
                    "excitationLambdaNm": 488.0,
                },
                "time": {"exposureTimeMs": 25.0},
            },
            {},
        ]
    }

    channels = _extract_nd2_channels(metadata, n_channels=2)

    assert channels[0]["name"] == "GFP"
    assert channels[0]["emission_wavelength"] == 510.0
    assert channels[0]["excitation_wavelength"] == 488.0
    assert channels[0]["exposure_time"] == 0.025
    assert channels[1]["name"] == "Channel 1"


def test_save_tiff_writes_imagej_frame_interval_from_timestamp_seconds(
    tmp_path,
):
    data = np.zeros((4, 1, 1, 8, 8), dtype=np.uint16)
    meta = {"timestamp_seconds": [0.0, 2.0, 4.0, 6.0]}
    out = tmp_path / "timelapse.tif"

    save_tiff(str(out), data, metadata=meta)

    with tifffile.TiffFile(str(out)) as tif:
        ij = tif.imagej_metadata or {}

    assert ij.get("finterval") == 2.0
    assert ij.get("tunit") == "sec"
    assert ij.get("fps") == 0.5


def test_load_tiff_reads_imagej_frame_interval_seconds(tmp_path):
    data = np.zeros((3, 1, 1, 8, 8), dtype=np.uint16)
    out = tmp_path / "timelapse_roundtrip.tif"

    save_tiff(
        str(out),
        data,
        metadata={"timestamp_seconds": [0.0, 1.5, 3.0]},
    )
    _, meta = load_image(str(out), use_memmap=False)

    assert meta["frame_interval_s"] == 1.5


def test_save_tiff_labels_frequency_space_scale_correctly(tmp_path):
    """FFT output tags itself space="frequency" (see fft_dialog.py) --
    saving it to file must not silently claim its cycles/um scale is a
    real-space "um" pixel size.
    """
    data = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
    out = tmp_path / "fft_result.tif"

    save_tiff(
        str(out),
        data,
        scale=(1.0, 0.1, 0.1),
        metadata={"space": "frequency"},
    )

    with tifffile.TiffFile(str(out)) as tif:
        ij = tif.imagej_metadata or {}

    assert ij.get("unit") == "1/um"


def test_save_tiff_labels_real_space_scale_as_um(tmp_path):
    data = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
    out = tmp_path / "real_space.tif"

    save_tiff(str(out), data, scale=(1.0, 0.1, 0.1), metadata={})

    with tifffile.TiffFile(str(out)) as tif:
        ij = tif.imagej_metadata or {}

    assert ij.get("unit") == "um"
