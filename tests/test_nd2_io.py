import numpy as np

from pyvistra.io import _extract_nd2_channels, _normalize_nd2_to_5d


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
