import numpy as np

from pyvistra.tracks import TrackTable


def test_track_table_sorts_by_track_and_time():
    tracks = TrackTable.from_arrays(
        track_id=[2, 1, 1, 2],
        t=[3, 5, 1, 2],
        x=[0, 1, 2, 3],
        y=[4, 5, 6, 7],
    )

    assert tracks.track_id.tolist() == [1, 1, 2, 2]
    assert tracks.t.tolist() == [1, 5, 2, 3]
    assert tracks.n_tracks == 2


def test_track_table_get_and_remove_track():
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 2, 2],
        t=[0, 1, 0, 1],
        x=[10, 11, 20, 21],
        y=[30, 31, 40, 41],
    )

    tr2 = tracks.get_track(2)
    assert tr2 is not None
    assert tr2["x"].tolist() == [20.0, 21.0]

    pruned = tracks.remove_track(1)
    assert pruned.n_tracks == 1
    assert np.all(pruned.track_id == 2)


def test_track_table_tracks_z_when_present():
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 1],
        t=[0, 1, 2],
        x=[0, 1, 2],
        y=[0, 1, 2],
        z=[4, 5, 6],
    )

    assert tracks.has_z
    tr = tracks.get_track(1)
    assert tr is not None
    assert tr["z"].tolist() == [4.0, 5.0, 6.0]
