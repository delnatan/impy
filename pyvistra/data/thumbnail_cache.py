"""In-memory decode cache + CPU-side compositing for the fast thumbnail grid.

Mirrors ``data/slice_loader.py``'s byte-budgeted LRU + background-thread
request/callback pattern, but keyed by image path instead of ``(t, z)``,
and it also owns the CPU-side channel compositing math (a numpy port of
``CompositeImageVisual``'s GPU shader) since this render path has no GL
context at all. Pure numpy: no Qt, no vispy.

Delivery contract: ``callback(path)`` fires on the worker thread once
``path`` has been decoded into the cache -- Qt consumers must marshal to
the GUI thread themselves (same rule as ``ImageBuffer.subscribe`` /
``SliceLoader``).
"""

import threading
from collections import OrderedDict

import numpy as np

from .. import colormaps as _colormaps
from ..contrast import compute_percentile_clim
from ..io import load_image

_LUT_SIZE = 256
_lut_cache = {}


def _get_lut(colormap_name):
    """256x3 float LUT for colormap_name, cached.

    CPU-side: ``vispy.color.Colormap.map()`` works on plain numpy without
    a GL context, so this is safe to call from a background thread.
    """
    lut = _lut_cache.get(colormap_name)
    if lut is None:
        cmap, _ = _colormaps.get(colormap_name)
        lut = cmap.map(np.linspace(0.0, 1.0, _LUT_SIZE))[:, :3].astype(
            np.float32
        )
        _lut_cache[colormap_name] = lut
    return lut


def composite_to_rgb(plane, channel_params):
    """Composite a ``(C, H, W)`` plane into an ``(H, W, 3)`` uint8 image.

    CPU port of the per-layer GPU shader chain in ``visuals/image.py``
    (``apply_clim`` -> ``apply_gamma`` -> colormap LUT), then additive
    blend of visible channels clipped to ``[0, 1]`` -- matching vispy's
    ``blend_func=("one", "one")`` layers clamped by the framebuffer.

    Args:
        plane: ``(C, H, W)`` array.
        channel_params: sequence of ``(clim, gamma, colormap_name,
            visible)``, one per channel. Extra entries beyond
            ``plane.shape[0]`` are ignored.
    """
    C, H, W = plane.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(min(C, len(channel_params))):
        clim, gamma, colormap_name, visible = channel_params[c]
        if not visible:
            continue
        vmin, vmax = clim
        span = vmax - vmin
        if span <= 0:
            span = 1.0
        data = plane[c].astype(np.float32, copy=False)
        t = np.clip((data - vmin) / span, 0.0, 1.0)
        if gamma != 1.0:
            t = np.power(t, gamma, dtype=np.float32)
        lut = _get_lut(colormap_name)
        idx = np.clip((t * (_LUT_SIZE - 1)).astype(np.int32), 0, _LUT_SIZE - 1)
        rgb += lut[idx]
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return (rgb * 255.0).astype(np.uint8)


class DecodeCache:
    """Byte-budgeted LRU of decoded ``(C, H, W)`` planes, keyed by path.

    Each entry also carries a per-channel default clim -- one-shot
    auto-contrast computed at decode time, mirroring ``TileWidget``'s
    per-tile auto-contrast (intensity ranges differ per image, so a
    single global clim would wash most tiles out unless the user
    explicitly overrides a channel via the global panel).
    """

    def __init__(self, max_bytes=256 * 1024 * 1024):
        self._max_bytes = int(max_bytes)
        self._cache = OrderedDict()  # path -> (plane, default_clims, channels_meta)
        self._nbytes = 0
        self._lock = threading.Lock()

    def get(self, path):
        """Return the cached ``(plane, default_clims, channels_meta)``
        entry for ``path``, or ``None``."""
        with self._lock:
            entry = self._cache.get(path)
            if entry is not None:
                self._cache.move_to_end(path)
            return entry

    def decode(self, path, dims=None):
        """Decode ``path`` synchronously and store it. Safe to call from
        a background thread. Returns the cached entry."""
        proxy, meta = load_image(path, dims=dims)
        try:
            T, Z, C, H, W = proxy.shape
            z_mid = Z // 2
            # Force a real copy -- some proxies (e.g. memmap'd TIFF) hand
            # back a view still backed by the file we're about to close.
            plane = np.array(proxy[0, z_mid, :, :, :], copy=True)
        finally:
            proxy.release()
        default_clims = tuple(
            compute_percentile_clim(plane[c], 0.5, 99.5)
            for c in range(plane.shape[0])
        )
        entry = (plane, default_clims, meta.get("channels"))
        self._store(path, entry)
        return entry

    def _store(self, path, entry):
        plane = entry[0]
        nbytes = plane.nbytes
        with self._lock:
            old = self._cache.pop(path, None)
            if old is not None:
                self._nbytes -= old[0].nbytes
            self._cache[path] = entry
            self._nbytes += nbytes
            while self._nbytes > self._max_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._nbytes -= evicted[0].nbytes

    def __contains__(self, path):
        with self._lock:
            return path in self._cache

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._nbytes = 0


class ThumbnailDecodeWorker:
    """Background decode of a priority-ordered path list.

    ``set_priority`` is latest-wins: it replaces whatever the loop
    hasn't reached yet, so switching pages/sorting reprioritizes instead
    of queuing up stale work -- same idea as ``SliceLoader``'s
    request/prefetch contract, adapted to a whole ordered list instead
    of a single (t, z) key.
    """

    def __init__(self, cache, callback, dims=None):
        self._cache = cache
        self._callback = callback
        self._dims = dims
        self._lock = threading.Condition()
        self._queue = []
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="pyvistra-thumbnail-decode", daemon=True
        )
        self._thread.start()

    def set_priority(self, paths):
        """Replace the pending queue with ``paths`` (highest priority
        first), skipping ones already cached."""
        with self._lock:
            self._queue = [p for p in paths if p not in self._cache]
            self._lock.notify()

    def set_dims(self, dims):
        with self._lock:
            self._dims = dims

    def close(self):
        with self._lock:
            self._closed = True
            self._queue = []
            self._lock.notify()
        self._thread.join(timeout=1.0)

    def _run(self):
        while True:
            with self._lock:
                while not self._closed and not self._queue:
                    self._lock.wait()
                if self._closed:
                    return
                path = self._queue.pop(0)
                dims = self._dims
            if path in self._cache:
                continue
            try:
                self._cache.decode(path, dims=dims)
            except Exception:
                continue
            try:
                self._callback(path)
            except Exception:
                pass
