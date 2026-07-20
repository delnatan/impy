"""Background slice loading for lazy 5D sources.

``SliceLoader`` moves per-slice disk reads (HDF5/CZI/zarr) and Z-max
projections off the GUI thread. It is *latest-wins*: a new ``request``
supersedes any not-yet-started one, so fast scrubbing only loads frames
that can actually be shown.

Loaded planes land in a byte-budgeted LRU cache, and after each request
the loader prefetches the (t±1, z±1) neighbours so adjacent scrubbing
becomes a synchronous cache hit.

Delivery contract: ``callback(key, plane)`` fires on the worker thread —
Qt consumers must marshal to the GUI thread (same rule as
``ImageBuffer.subscribe``). This module is pure data: no Qt, no vispy.
"""

import threading
from collections import OrderedDict

import numpy as np


class SliceLoader:
    """Async (t, z) plane loader over a ``Readable5D`` source.

    Parameters
    ----------
    source : Readable5D
        5D (T, Z, C, Y, X) data source. Sliced on the worker thread.
    callback : callable
        ``callback(key, plane)`` invoked on the worker thread for every
        *requested* load (never for prefetches). ``key`` is the value
        returned by :meth:`key_for`; ``plane`` is a (C, Y, X) array.
    cache_bytes : int
        LRU cache budget. The most recent plane is always retained.
    """

    def __init__(self, source, callback, cache_bytes=256 * 1024 * 1024):
        self._source = source
        self._callback = callback
        self._max_bytes = int(cache_bytes)
        self._cache = OrderedDict()  # key -> (C, Y, X) ndarray
        self._cache_nbytes = 0
        self._lock = threading.Condition()
        self._pending = None  # latest requested key not yet loading
        self._prefetch = []  # neighbour keys, loaded when idle
        self._current = None  # last requested key: eviction-protected
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="pyvistra-slice-loader", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    @staticmethod
    def key_for(t, z):
        """Cache/delivery key for a (t, z) view position.

        ``z`` is an int for a single plane or a ``slice`` for a Z-max
        projection over ``[start, stop)``.
        """
        if isinstance(z, slice):
            return (int(t), ("proj", int(z.start), int(z.stop)))
        return (int(t), int(z))

    # ------------------------------------------------------------------
    # Requests (caller thread)
    # ------------------------------------------------------------------

    def request(self, t, z):
        """Ask for the plane at (t, z). Returns it immediately on a cache
        hit; otherwise returns None and delivers via the callback later,
        superseding any queued request."""
        key = self.key_for(t, z)
        with self._lock:
            self._current = key
            plane = self._cache.get(key)
            if plane is not None:
                self._cache.move_to_end(key)
                self._pending = None
            else:
                self._pending = key
            self._prefetch = self._neighbour_keys(key)
            self._lock.notify()
        return plane

    def get_cached(self, t, z):
        """Cache lookup without scheduling a load."""
        with self._lock:
            return self._cache.get(self.key_for(t, z))

    def invalidate(self):
        """Drop every cached plane (call when source contents change)."""
        with self._lock:
            self._cache.clear()
            self._cache_nbytes = 0

    def close(self):
        """Stop the worker thread. Idempotent; pending work is dropped."""
        with self._lock:
            self._closed = True
            self._pending = None
            self._prefetch = []
            self._lock.notify()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run(self):
        while True:
            with self._lock:
                while (
                    not self._closed
                    and self._pending is None
                    and not self._prefetch
                ):
                    self._lock.wait()
                if self._closed:
                    return
                if self._pending is not None:
                    key, deliver = self._pending, True
                    self._pending = None
                else:
                    key, deliver = self._prefetch.pop(0), False
                    if key in self._cache:
                        continue

            plane = self._load(key)

            with self._lock:
                if self._closed:
                    return
                if plane is not None:
                    self._store(key, plane)
                # A newer request arrived while loading: don't deliver
                # the stale frame (it stays cached for a later hit).
                superseded = self._pending is not None

            if deliver and plane is not None and not superseded:
                self._callback(key, plane)

    def _load(self, key):
        t, zk = key
        try:
            if isinstance(zk, tuple):
                _, lo, hi = zk
                vol = self._source[t, lo:hi, :, :, :]
                return np.max(vol, axis=0)
            return self._source[t, zk, :, :, :]
        except Exception:
            # Source may have been closed mid-read (window closing) or
            # the key may be out of bounds after a data swap.
            return None

    def _neighbour_keys(self, key):
        t, zk = key
        T, Z = int(self._source.shape[0]), int(self._source.shape[1])
        out = []
        for dt in (1, -1):
            if 0 <= t + dt < T:
                out.append((t + dt, zk))
        if not isinstance(zk, tuple):
            for dz in (1, -1):
                if 0 <= zk + dz < Z:
                    out.append((t, zk + dz))
        return out

    def _store(self, key, plane):
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache_nbytes -= old.nbytes
        self._cache[key] = plane
        self._cache_nbytes += plane.nbytes
        while self._cache_nbytes > self._max_bytes and len(self._cache) > 1:
            candidate = next(iter(self._cache))
            if candidate == self._current:
                # Never evict the plane being displayed; a prefetch burst
                # must not push the visible frame out of the cache.
                self._cache.move_to_end(candidate)
                candidate = next(iter(self._cache))
                if candidate == self._current:
                    break
            evicted = self._cache.pop(candidate)
            self._cache_nbytes -= evicted.nbytes
