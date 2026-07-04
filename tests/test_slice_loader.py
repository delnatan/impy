"""Tests for data.slice_loader (async latest-wins slice loading)."""

import threading
import time

import numpy as np
import pytest

from pyvistra.data.slice_loader import SliceLoader


class SlowSource:
    """Readable5D that counts reads and can be slowed down."""

    def __init__(self, shape=(4, 3, 2, 8, 8), delay=0.0):
        self.data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(
            shape
        )
        self.shape = shape
        self.dtype = self.data.dtype
        self.ndim = 5
        self.delay = delay
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        if self.delay:
            time.sleep(self.delay)
        return self.data[key]


class Collector:
    def __init__(self):
        self.delivered = []
        self.event = threading.Event()

    def __call__(self, key, plane):
        self.delivered.append((key, plane))
        self.event.set()

    def wait(self, timeout=2.0):
        ok = self.event.wait(timeout)
        self.event.clear()
        return ok


@pytest.fixture
def loader_pair():
    src = SlowSource()
    col = Collector()
    loader = SliceLoader(src, col)
    yield src, col, loader
    loader.close()


def test_miss_delivers_then_hit_is_synchronous(loader_pair):
    src, col, loader = loader_pair

    assert loader.request(1, 2) is None  # cold: async
    assert col.wait()
    key, plane = col.delivered[0]
    assert key == (1, 2)
    np.testing.assert_array_equal(plane, src.data[1, 2])

    # warm: synchronous hit, no second delivery
    hit = loader.request(1, 2)
    np.testing.assert_array_equal(hit, src.data[1, 2])


def test_projection_key_loads_zmax(loader_pair):
    src, col, loader = loader_pair
    assert loader.request(0, slice(0, 3)) is None
    assert col.wait()
    key, plane = col.delivered[0]
    assert key == (0, ("proj", 0, 3))
    np.testing.assert_array_equal(plane, src.data[0, 0:3].max(axis=0))


def test_latest_wins_drops_stale_delivery():
    src = SlowSource(delay=0.05)
    col = Collector()
    loader = SliceLoader(src, col)
    try:
        # Burst of requests: only the last one must be delivered.
        for t in range(4):
            loader.request(t, 0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if any(k == (3, 0) for k, _ in col.delivered):
                break
            time.sleep(0.01)
        keys = [k for k, _ in col.delivered]
        assert (3, 0) in keys
        # intermediate frames may have been loading already when their
        # successor arrived, but none may be delivered *after* the final
        assert keys[-1] == (3, 0)
    finally:
        loader.close()


def test_prefetch_fills_neighbours(loader_pair):
    src, col, loader = loader_pair
    loader.request(1, 1)
    assert col.wait()
    # allow idle prefetch of (0,1), (2,1), (1,0), (1,2)
    deadline = time.monotonic() + 2.0
    want = {(2, 1), (0, 1), (1, 2), (1, 0)}
    while time.monotonic() < deadline:
        if all(loader.get_cached(t, z) is not None for t, z in want):
            break
        time.sleep(0.01)
    for t, z in want:
        np.testing.assert_array_equal(loader.get_cached(t, z), src.data[t, z])
    # prefetches are cached but never delivered
    assert len(col.delivered) == 1


def test_invalidate_clears_cache(loader_pair):
    src, col, loader = loader_pair
    loader.request(0, 0)
    assert col.wait()
    assert loader.get_cached(0, 0) is not None
    loader.invalidate()
    assert loader.get_cached(0, 0) is None


def test_eviction_respects_byte_budget():
    src = SlowSource()
    col = Collector()
    plane_bytes = src.data[0, 0].nbytes
    loader = SliceLoader(src, col, cache_bytes=2 * plane_bytes)
    try:
        for t in range(4):
            loader.request(t, 0)
            col.wait()
        cached = [loader.get_cached(t, 0) is not None for t in range(4)]
        assert sum(cached) <= 2
        assert cached[3]  # most recent always retained
    finally:
        loader.close()


def test_close_is_idempotent_and_stops_thread():
    src = SlowSource()
    loader = SliceLoader(src, lambda k, p: None)
    loader.close()
    loader.close()
    loader._thread.join(timeout=2.0)
    assert not loader._thread.is_alive()


def test_out_of_bounds_request_is_swallowed(loader_pair):
    src, col, loader = loader_pair
    loader.request(99, 0)  # invalid: must not crash the worker
    time.sleep(0.1)
    # worker still functional afterwards
    loader.request(0, 0)
    assert col.wait()
