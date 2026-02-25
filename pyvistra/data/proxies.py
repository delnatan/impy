"""
5D image data proxies for pyvistra.

All proxies present a numpy-array-like interface over a (T, Z, C, Y, X)
data model.  They support slicing but only load the requested region into
memory, enabling lazy access to large files.
"""

import numpy as np
import zarr


# ---------------------------------------------------------------------------
# Reference-counting mixin
# ---------------------------------------------------------------------------

class RefCountMixin:
    """Reference counting for proxies that wrap file handles."""

    def _init_refcount(self):
        self._refcount = 1

    def acquire(self):
        """Increment reference count. Returns self."""
        self._refcount += 1
        return self

    def release(self):
        """Decrement ref count. Calls close() when it reaches 0."""
        self._refcount -= 1
        if self._refcount <= 0:
            self.close()


# ---------------------------------------------------------------------------
# Shared base for file-backed readers (Imaris, CZI, …)
# ---------------------------------------------------------------------------

class File5DProxy(RefCountMixin):
    """
    Base class for file-backed 5D proxies.

    Assumes the underlying reader exposes:
      - ``reader.shape`` as ``(T, C, Z, Y, X)``
      - ``reader.dtype``
      - ``reader.read(c, t, z)`` → 2-D or 3-D numpy array
      - ``reader.close()``
    """

    def __init__(self, reader):
        self.reader = reader
        # Readers store (T, C, Z, Y, X); we normalise to (T, Z, C, Y, X).
        t, c, z, y, x = reader.shape
        self.shape = (t, z, c, y, x)
        self.dtype = reader.dtype
        self.ndim = 5
        self._init_refcount()

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_key(key, ndim=5):
        """Expand a slicing key to a full *ndim*-tuple."""
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            idx = key.index(Ellipsis)
            n_expand = ndim - (len(key) - 1)
            key = key[:idx] + (slice(None),) * n_expand + key[idx + 1:]
        if len(key) < ndim:
            key = key + (slice(None),) * (ndim - len(key))
        return key

    def __getitem__(self, key):
        t_idx, z_idx, c_idx, y_idx, x_idx = self._normalise_key(key)

        if isinstance(t_idx, slice):
            start, stop, step = t_idx.indices(self.shape[0])
            t_indices = range(start, stop, step)
            if len(t_indices) == 0:
                return np.empty((0,) + self.shape[1:], dtype=self.dtype)
            data = np.array([self._read_timepoint(t, z_idx, c_idx)
                             for t in t_indices])
            return data[..., y_idx, x_idx]
        else:
            data = self._read_timepoint(t_idx, z_idx, c_idx)
            return data[..., y_idx, x_idx]

    def _read_timepoint(self, t, z_idx, c_idx):
        """Return (Z, C, Y, X) or a subset for a single timepoint."""
        if isinstance(c_idx, slice):
            start, stop, step = c_idx.indices(self.shape[2])
            planes = [self._read_z_slice(c, t, z_idx)
                      for c in range(start, stop, step)]
            stack = np.array(planes)          # (C, ...)
            if stack.ndim == 4:               # (C, Z, Y, X) → (Z, C, Y, X)
                stack = np.transpose(stack, (1, 0, 2, 3))
            return stack
        else:
            return self._read_z_slice(c_idx, t, z_idx)

    def _read_z_slice(self, c, t, z):
        """Return a single plane or a Z-stack for channel *c*, time *t*."""
        if isinstance(z, slice):
            start, stop, step = z.indices(self.shape[1])
            z_indices = range(start, stop, step)
            # Optimise: read the full stack in one call when possible
            if step == 1 and start == 0 and stop == self.shape[1]:
                return self.reader.read(c=c, t=t, z=None)
            if len(z_indices) == 0:
                return np.zeros((0, self.shape[3], self.shape[4]),
                                dtype=self.dtype)
            return np.array([self.reader.read(c=c, t=t, z=zi)
                             for zi in z_indices])
        else:
            return self.reader.read(c=c, t=t, z=z)


# ---------------------------------------------------------------------------
# Concrete file-backed proxies
# ---------------------------------------------------------------------------

class Imaris5DProxy(File5DProxy):
    """Lazy 5D proxy for Imaris .ims files (HDF5-backed)."""


class CZI5DProxy(File5DProxy):
    """Lazy 5D proxy for Zeiss .czi files."""


# ---------------------------------------------------------------------------
# In-memory proxy
# ---------------------------------------------------------------------------

class Numpy5DProxy:
    """Wraps a 5D numpy array so it has the same interface as the other proxies."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.ndim = 5

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            idx = key.index(Ellipsis)
            n_expand = 5 - (len(key) - 1)
            key = key[:idx] + (slice(None),) * n_expand + key[idx + 1:]
        if len(key) < 5:
            key = key + (slice(None),) * (5 - len(key))
        return self.array[key]


# ---------------------------------------------------------------------------
# Zarr-backed lazy proxy
# ---------------------------------------------------------------------------

class Zarr5DProxy(RefCountMixin):
    """
    Wraps a zarr array as a (T, Z, C, Y, X) 5D proxy.

    The underlying zarr array may have 2–5 dimensions; singleton dimensions
    are inserted as needed to present a consistent 5D interface.
    """

    def __init__(self, zarr_array, source_ndim):
        self._store = zarr_array
        self._source_ndim = source_ndim
        self.dtype = zarr_array.dtype
        self.ndim = 5

        src = zarr_array.shape
        if source_ndim == 2:
            self.shape = (1, 1, 1, src[0], src[1])
        elif source_ndim == 3:
            self.shape = (1, src[0], 1, src[1], src[2])
        elif source_ndim == 4:
            self.shape = (1, src[0], src[1], src[2], src[3])
        elif source_ndim == 5:
            self.shape = src
        else:
            raise ValueError(
                f"Unsupported zarr array dimensionality: {source_ndim}"
            )
        self._init_refcount()

    def _normalise_key(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            idx = key.index(Ellipsis)
            n_expand = 5 - (len(key) - 1)
            key = key[:idx] + (slice(None),) * n_expand + key[idx + 1:]
        if len(key) < 5:
            key = key + (slice(None),) * (5 - len(key))
        return key

    def _map_key(self, key_5d):
        t, z, c, y, x = key_5d
        if self._source_ndim == 2:
            return (y, x)
        elif self._source_ndim == 3:
            return (z, y, x)
        elif self._source_ndim == 4:
            return (z, c, y, x)
        return key_5d

    def __getitem__(self, key):
        key_5d = self._normalise_key(key)
        return np.asarray(self._store[self._map_key(key_5d)])

    def close(self):
        if hasattr(self._store, "store") and hasattr(self._store.store, "close"):
            self._store.store.close()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
