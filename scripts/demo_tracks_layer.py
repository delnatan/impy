"""Minimal track-layer demo on a blank image.

Run:
    python scripts/demo_tracks_layer.py
"""

import numpy as np

from pyvistra.tracks import TrackTable
from pyvistra.ui import imshow, run_app


def main():
    # Blank 2D timelapse: (T, Z, C, Y, X)
    data = np.zeros((40, 1, 1, 256, 256), dtype=np.uint16)

    viewer = imshow(data, title="Track Layer Demo")

    t = np.arange(40, dtype=np.int32)
    x = np.linspace(24, 232, 40, dtype=np.float32)
    y = (128 + 45 * np.sin(np.linspace(0, 3 * np.pi, 40))).astype(np.float32)

    tracks = TrackTable.from_arrays(
        track_id=np.ones_like(t),
        t=t,
        x=x,
        y=y,
    )

    viewer.add_track_layer("Tracks-1", tracks=tracks, trail_window=15)
    run_app()


if __name__ == "__main__":
    main()
