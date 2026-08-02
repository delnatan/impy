import numpy as np

from pyvistra.data.tracks import TrackTable, load_tracks_csv, save_tracks_csv


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


def test_track_csv_roundtrip_without_z(tmp_path):
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 2],
        t=[0, 1, 0],
        x=[10.5, 11.5, 20.5],
        y=[30.5, 31.5, 40.5],
    )

    path = tmp_path / "tracks.csv"
    save_tracks_csv(str(path), tracks)
    loaded = load_tracks_csv(str(path))

    assert loaded.n_rows == tracks.n_rows
    assert loaded.track_id.tolist() == tracks.track_id.tolist()
    assert loaded.t.tolist() == tracks.t.tolist()
    assert np.allclose(loaded.x, tracks.x)
    assert np.allclose(loaded.y, tracks.y)
    assert not loaded.has_z


def test_track_csv_roundtrip_with_z(tmp_path):
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 2],
        t=[0, 1, 0],
        x=[10.5, 11.5, 20.5],
        y=[30.5, 31.5, 40.5],
        z=[1.0, 2.0, 3.0],
    )

    path = tmp_path / "tracks_z.csv"
    save_tracks_csv(str(path), tracks)
    loaded = load_tracks_csv(str(path))

    assert loaded.has_z
    assert np.allclose(loaded.z, tracks.z)


def test_load_tracks_csv_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("track_id,t,x\n1,0,1.0\n")

    try:
        load_tracks_csv(str(path))
    except ValueError as e:
        assert "Missing required columns" in str(e)
    else:
        raise AssertionError("Expected ValueError for missing columns")


def test_track_table_properties_sorted_and_validated():
    tracks = TrackTable.from_arrays(
        track_id=[2, 1],
        t=[0, 0],
        x=[10.0, 20.0],
        y=[30.0, 40.0],
        properties={"speed": [1.0, 2.0]},
    )

    # Sorted by (track_id, t): track 1 first, so its property follows.
    assert tracks.track_id.tolist() == [1, 2]
    assert tracks.properties["speed"].tolist() == [2.0, 1.0]

    try:
        TrackTable.from_arrays(
            track_id=[1, 2],
            t=[0, 0],
            x=[1.0, 2.0],
            y=[3.0, 4.0],
            properties={"speed": [1.0]},
        )
    except ValueError as e:
        assert "Property 'speed'" in str(e)
    else:
        raise AssertionError("Expected ValueError for mismatched property length")


def test_track_table_remove_track_slices_properties():
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 2, 2],
        t=[0, 1, 0, 1],
        x=[10, 11, 20, 21],
        y=[30, 31, 40, 41],
        properties={"quality": [0.1, 0.1, 0.9, 0.9]},
    )

    pruned = tracks.remove_track(1)
    assert pruned.properties["quality"].tolist() == [0.9, 0.9]


def test_track_csv_roundtrip_preserves_properties(tmp_path):
    tracks = TrackTable.from_arrays(
        track_id=[1, 1, 2],
        t=[0, 1, 0],
        x=[10.5, 11.5, 20.5],
        y=[30.5, 31.5, 40.5],
        properties={"speed": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]},
    )

    path = tmp_path / "tracks_props.csv"
    save_tracks_csv(str(path), tracks)
    loaded = load_tracks_csv(str(path))

    assert np.allclose(loaded.properties["speed"], [1.0, 2.0, 3.0])
    assert loaded.properties["label"].tolist() == ["a", "b", "c"]
