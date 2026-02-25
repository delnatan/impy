"""
ImageBuffer — zarr-backed 5D write buffer for streaming image operations.

The intended workflow is::

    buf = ImageBuffer(shape=(T, Z, C, Y, X), dtype=np.uint16, metadata=meta)

    # In a worker thread / loop:
    for t in range(T):
        result = heavy_computation(source[t, ...])
        buf[t, ...] = result

    # Meanwhile the UI can display buf live:
    win = imshow(buf, title="Processing…")

After processing, ``buf`` can be saved or kept in memory.  The underlying
zarr store is a temporary directory in ``~/.pyvistra/buffers/`` that is
deleted when ``buf.close()`` is called (or when the object is garbage-
collected and the refcount reaches zero).
"""

import shutil
import uuid
from pathlib import Path

import numpy as np
import zarr

from .proxies import RefCountMixin

BUFFER_DIR = Path.home() / ".pyvistra" / "buffers"


class ImageBuffer(RefCountMixin):
    """
    Zarr-backed 5D array with the same read interface as the proxies plus
    write support via ``__setitem__``.

    Shape must be 5D ``(T, Z, C, Y, X)``.
    """

    def __init__(self, shape, dtype, chunks=None, metadata=None):
        BUFFER_DIR.mkdir(parents=True, exist_ok=True)
        self._path = BUFFER_DIR / f"{uuid.uuid4()}.zarr"

        T, Z, C, Y, X = shape
        if chunks is None:
            chunks = (1, min(16, Z), C, min(512, Y), min(512, X))

        self._store = zarr.open(
            str(self._path),
            mode="w",
            shape=shape,
            dtype=dtype,
            chunks=chunks,
        )

        self.metadata = metadata or {}
        self.ndim = 5
        self._init_refcount()

    @property
    def shape(self):
        return self._store.shape

    @property
    def dtype(self):
        return self._store.dtype

    def __getitem__(self, key):
        return np.asarray(self._store[key])

    def __setitem__(self, key, value):
        self._store[key] = value

    def save_as(self, filepath):
        """Export buffer contents to OME-TIFF."""
        from .io import save_tiff
        scale = self.metadata.get("scale", (1.0, 1.0, 1.0))
        save_tiff(filepath, self._store[:], scale=scale, metadata=self.metadata)

    def close(self):
        if self._path.exists():
            shutil.rmtree(self._path)

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
