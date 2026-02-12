"""Imaris (.ims) HDF5 file writer.

Writes 5D image data in the Imaris HDF5 format (version 5.5.0),
compatible with the Bitplane Imaris viewer.
"""

from datetime import datetime

import h5py
import numpy as np


class ImarisWriter:
    """Streaming writer for Imaris .ims files.

    Pre-creates the HDF5 structure, then accepts one (Z, Y, X) volume at a
    time via write_channel(). Metadata and thumbnail are finalized on close().

    Usage::

        with ImarisWriter("out.ims", (T, Z, C, Y, X), np.uint16, meta) as w:
            for t in range(T):
                vol = source[t, :, :, :, :]   # (Z, C, Y, X)
                for c in range(C):
                    w.write_channel(t, c, vol[:, c, :, :])
    """

    def __init__(
        self,
        filepath,
        shape,
        dtype,
        metadata,
        resolution_levels=True,
        compression_level=2,
    ):
        """
        Args:
            filepath: Output path (.ims)
            shape: (T, Z, C, Y, X) — pyvistra convention
            dtype: numpy dtype for the data
            metadata: dict with 'scale', 'channels', etc.
            resolution_levels: Generate downsampled pyramid (default True)
            compression_level: gzip level 0-9 (default 2)
        """
        self._filepath = filepath
        self._T, self._Z, self._C, self._Y, self._X = shape
        self._dtype = np.dtype(dtype)
        self._metadata = metadata
        self._compression = compression_level

        # Pyramid
        if resolution_levels:
            self._levels = _compute_resolution_levels(
                self._Z, self._Y, self._X
            )
        else:
            self._levels = [(self._Z, self._Y, self._X)]

        # Per-channel stats accumulated during writes
        self._channel_mins = [np.inf] * self._C
        self._channel_maxs = [-np.inf] * self._C

        # Thumbnail: grab first mid-Z slice we see
        self._thumb_slice = None

        # Open file and create structure
        self._file = h5py.File(filepath, "w")
        self._setup_structure()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write_channel(self, t, c, volume):
        """Write a single (Z, Y, X) volume for timepoint t, channel c.

        Also writes downsampled pyramid levels, histogram, and accumulates
        channel min/max stats.
        """
        volume = np.asarray(volume)

        for level_idx, (lz, ly, lx) in enumerate(self._levels):
            if level_idx == 0:
                vol = volume
            else:
                vol = _downsample_3d(volume, (lz, ly, lx))

            hist, vmin, vmax = _compute_histogram(vol)

            # Track global channel stats (level 0 only)
            if level_idx == 0:
                self._channel_mins[c] = min(self._channel_mins[c], vmin)
                self._channel_maxs[c] = max(self._channel_maxs[c], vmax)

            # Write into pre-allocated dataset
            grp = self._file[
                f"DataSet/ResolutionLevel {level_idx}/"
                f"TimePoint {t}/Channel {c}"
            ]
            grp["Data"][:] = vol
            _set_attr(grp["Data"], "HistogramMin", f"{vmin:.6f}")
            _set_attr(grp["Data"], "HistogramMax", f"{vmax:.6f}")
            grp["Histogram"][:] = hist.astype(np.uint64)

        # Grab a thumbnail slice from the first timepoint
        if self._thumb_slice is None and t == 0:
            mid_z = volume.shape[0] // 2
            self._thumb_slice = volume[mid_z].copy()

    def close(self):
        """Write DataSetInfo metadata, thumbnail, and close the file."""
        if self._file is None:
            return

        f = self._file
        T, Z, C, Y, X = self._T, self._Z, self._C, self._Y, self._X
        meta = self._metadata
        scale = meta.get("scale", (1.0, 1.0, 1.0))
        vz, vy, vx = scale
        channels = meta.get("channels", [])
        if not channels:
            channels = [{"name": f"Channel {i}"} for i in range(C)]
        dataset_name = meta.get("name", meta.get("filename", "Image"))
        timestamps = meta.get("timestamps", [])

        # --- DataSetInfo ---
        info = f.create_group("DataSetInfo")

        img_grp = info.create_group("Image")
        _set_attr(img_grp, "X", str(X))
        _set_attr(img_grp, "Y", str(Y))
        _set_attr(img_grp, "Z", str(Z))
        _set_attr(img_grp, "Noc", str(C))
        _set_attr(img_grp, "ExtMin0", "0")
        _set_attr(img_grp, "ExtMin1", "0")
        _set_attr(img_grp, "ExtMin2", "0")
        _set_attr(img_grp, "ExtMax0", f"{X * vx:.6f}")
        _set_attr(img_grp, "ExtMax1", f"{Y * vy:.6f}")
        _set_attr(img_grp, "ExtMax2", f"{Z * vz:.6f}")
        _set_attr(img_grp, "Unit", "um")
        _set_attr(img_grp, "Name", str(dataset_name))
        _set_attr(
            img_grp,
            "RecordingDate",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        )
        levels = self._levels
        _set_attr(
            img_grp,
            "ResampleDimensionX",
            str(levels[-1][2]) if len(levels) > 1 else str(X),
        )
        _set_attr(
            img_grp,
            "ResampleDimensionY",
            str(levels[-1][1]) if len(levels) > 1 else str(Y),
        )
        _set_attr(
            img_grp,
            "ResampleDimensionZ",
            str(levels[-1][0]) if len(levels) > 1 else str(Z),
        )

        for c in range(C):
            ch_info = channels[c] if c < len(channels) else {}
            ch_grp = info.create_group(f"Channel {c}")
            _set_attr(ch_grp, "Name", ch_info.get("name", f"Channel {c}"))
            _set_attr(ch_grp, "ColorMode", "BaseColor")
            _set_attr(ch_grp, "ColorOpacity", "1")
            _set_attr(ch_grp, "Color", _default_channel_color(c, C))
            cmin = (
                self._channel_mins[c]
                if np.isfinite(self._channel_mins[c])
                else 0
            )
            cmax = (
                self._channel_maxs[c]
                if np.isfinite(self._channel_maxs[c])
                else 1
            )
            _set_attr(ch_grp, "ColorRange", f"{cmin:.6f} {cmax:.6f}")
            _set_attr(ch_grp, "Min", f"{cmin:.6f}")
            _set_attr(ch_grp, "Max", f"{cmax:.6f}")
            em = ch_info.get("emission_wavelength")
            if em is not None:
                _set_attr(ch_grp, "LSMEmissionWavelength", str(em))
            ex = ch_info.get("excitation_wavelength")
            if ex is not None:
                _set_attr(ch_grp, "LSMExcitationWavelength", str(ex))

        imaris_grp = info.create_group("Imaris")
        _set_attr(imaris_grp, "Version", "5.5")
        _set_attr(imaris_grp, "ThumbnailMode", "thumbnailMIP")

        ids_grp = info.create_group("ImarisDataSet")
        _set_attr(ids_grp, "Creator", "pyvistra")
        _set_attr(ids_grp, "Version", "5.5.0")

        ti_grp = info.create_group("TimeInfo")
        _set_attr(ti_grp, "DataSetTimePoints", str(T))
        _set_attr(ti_grp, "FileTimePoints", str(T))
        for t in range(T):
            if t < len(timestamps) and timestamps[t] is not None:
                ts_str = timestamps[t].strftime("%Y-%m-%d %H:%M:%S.%f")
            else:
                ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            _set_attr(ti_grp, f"TimePoint{t + 1}", ts_str)

        log_grp = info.create_group("Log")
        _set_attr(log_grp, "Entries", "0")

        # --- Thumbnail ---
        thumb = self._make_thumbnail()
        thumb_grp = f.create_group("Thumbnail")
        thumb_grp.create_dataset("Data", data=thumb)

        self._file.close()
        self._file = None

    def _setup_structure(self):
        """Pre-create all HDF5 groups and empty datasets."""
        f = self._file

        _set_attr(f, "DataSetDirectoryName", "DataSet")
        _set_attr(f, "DataSetInfoDirectoryName", "DataSetInfo")
        _set_attr(f, "ImarisDataSet", "ImarisDataSet")
        _set_attr(f, "ImarisVersion", "5.5.0")
        _set_attr(f, "NumberOfDataSets", "1")
        _set_attr(f, "ThumbnailDirectoryName", "Thumbnail")

        ds = f.create_group("DataSet")

        for level_idx, (lz, ly, lx) in enumerate(self._levels):
            res_grp = ds.create_group(f"ResolutionLevel {level_idx}")
            chunks = _compute_chunk_size(lz, ly, lx, self._dtype)
            valid_chunks = all(
                cs <= ds for cs, ds in zip(chunks, (lz, ly, lx))
            )

            for t in range(self._T):
                tp_grp = res_grp.create_group(f"TimePoint {t}")

                for c in range(self._C):
                    ch_grp = tp_grp.create_group(f"Channel {c}")

                    dset = ch_grp.create_dataset(
                        "Data",
                        shape=(lz, ly, lx),
                        dtype=self._dtype,
                        chunks=chunks if valid_chunks else None,
                        compression="gzip",
                        compression_opts=self._compression,
                    )
                    _set_attr(dset, "ImageSizeX", str(lx))
                    _set_attr(dset, "ImageSizeY", str(ly))
                    _set_attr(dset, "ImageSizeZ", str(lz))

                    ch_grp.create_dataset(
                        "Histogram",
                        shape=(256,),
                        dtype=np.uint64,
                    )

    def _make_thumbnail(self, size=256):
        """Generate RGBA thumbnail from the cached mid-Z slice."""
        if self._thumb_slice is None:
            return np.zeros((size, size, 4), dtype=np.uint8)

        img = self._thumb_slice.astype(np.float32)
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

        from scipy.ndimage import zoom

        f = min(size / img.shape[0], size / img.shape[1])
        img = zoom(img, f, order=1)

        thumb = np.zeros((size, size), dtype=np.uint8)
        oy = (size - img.shape[0]) // 2
        ox = (size - img.shape[1]) // 2
        thumb[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img[
            :size, :size
        ]

        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        rgba[:, :, 0] = thumb
        rgba[:, :, 1] = thumb
        rgba[:, :, 2] = thumb
        rgba[:, :, 3] = 255
        return rgba


def save_imaris(
    filepath,
    data,
    metadata,
    resolution_levels=True,
    compression_level=2,
    progress_cb=None,
):
    """
    Save 5D data (T, Z, C, Y, X) to Imaris .ims HDF5 format.

    Args:
        filepath: Output path (.ims)
        data: 5D array-like in pyvistra (T, Z, C, Y, X) format
        metadata: dict with keys:
            - 'scale': (vz, vy, vx) voxel sizes in microns
            - 'channels': list of dicts with 'name', optional wavelength info
            - 'filename' or 'name': dataset name
            - 'timestamps': optional list of datetime objects
        resolution_levels: If True, generate downsampled pyramid levels.
            If False, only write full resolution.
        compression_level: gzip compression level 0-9 (default 2)
        progress_cb: Optional callback(fraction) for progress reporting
    """
    # Normalize data to numpy
    if hasattr(data, "shape") and hasattr(data, "__getitem__"):
        T, Z, C, Y, X = data.shape
    else:
        data = np.asarray(data)
        T, Z, C, Y, X = data.shape

    scale = metadata.get("scale", (1.0, 1.0, 1.0))
    vz, vy, vx = scale

    # Compute extents (Imaris convention: 0=X, 1=Y, 2=Z)
    ext_max_x = X * vx
    ext_max_y = Y * vy
    ext_max_z = Z * vz

    # Compute resolution pyramid shapes
    level_shapes = [(Z, Y, X)]
    if resolution_levels:
        level_shapes = _compute_resolution_levels(Z, Y, X)

    # Channel metadata
    channels = metadata.get("channels", [])
    if not channels:
        channels = [{"name": f"Channel {i}"} for i in range(C)]

    dataset_name = metadata.get("name", metadata.get("filename", "Image"))
    timestamps = metadata.get("timestamps", [])

    # Track total work units for progress
    total_units = T * C * len(level_shapes)
    done_units = 0

    # Track per-channel global min/max across all timepoints
    channel_mins = [np.inf] * C
    channel_maxs = [-np.inf] * C

    with h5py.File(filepath, "w") as f:
        # Root attributes
        _set_attr(f, "DataSetDirectoryName", "DataSet")
        _set_attr(f, "DataSetInfoDirectoryName", "DataSetInfo")
        _set_attr(f, "ImarisDataSet", "ImarisDataSet")
        _set_attr(f, "ImarisVersion", "5.5.0")
        _set_attr(f, "NumberOfDataSets", "1")
        _set_attr(f, "ThumbnailDirectoryName", "Thumbnail")

        ds = f.create_group("DataSet")

        # Write each resolution level
        for level_idx, (lz, ly, lx) in enumerate(level_shapes):
            res_grp = ds.create_group(f"ResolutionLevel {level_idx}")
            chunks = _compute_chunk_size(lz, ly, lx, data.dtype)

            for t in range(T):
                tp_grp = res_grp.create_group(f"TimePoint {t}")

                for c in range(C):
                    ch_grp = tp_grp.create_group(f"Channel {c}")

                    # Get volume: (Z, Y, X)
                    volume = np.asarray(data[t, :, c, :, :])

                    # Downsample if needed
                    if level_idx > 0:
                        volume = _downsample_3d(volume, (lz, ly, lx))

                    # Compute histogram and stats
                    hist, vmin, vmax = _compute_histogram(volume)

                    # Track channel global min/max (level 0 only)
                    if level_idx == 0:
                        channel_mins[c] = min(channel_mins[c], vmin)
                        channel_maxs[c] = max(channel_maxs[c], vmax)

                    # Write data
                    dset = ch_grp.create_dataset(
                        "Data",
                        data=volume,
                        chunks=chunks
                        if all(
                            cs <= ds
                            for cs, ds in zip(chunks, volume.shape)
                        )
                        else None,
                        compression="gzip",
                        compression_opts=compression_level,
                    )
                    _set_attr(dset, "ImageSizeX", str(lx))
                    _set_attr(dset, "ImageSizeY", str(ly))
                    _set_attr(dset, "ImageSizeZ", str(lz))
                    _set_attr(dset, "HistogramMin", f"{vmin:.6f}")
                    _set_attr(dset, "HistogramMax", f"{vmax:.6f}")

                    # Write histogram
                    ch_grp.create_dataset(
                        "Histogram", data=hist.astype(np.uint64)
                    )

                    done_units += 1
                    if progress_cb:
                        progress_cb(done_units / total_units)

        # --- DataSetInfo ---
        info = f.create_group("DataSetInfo")

        # Image group
        img_grp = info.create_group("Image")
        _set_attr(img_grp, "X", str(X))
        _set_attr(img_grp, "Y", str(Y))
        _set_attr(img_grp, "Z", str(Z))
        _set_attr(img_grp, "Noc", str(C))
        _set_attr(img_grp, "ExtMin0", "0")
        _set_attr(img_grp, "ExtMin1", "0")
        _set_attr(img_grp, "ExtMin2", "0")
        _set_attr(img_grp, "ExtMax0", f"{ext_max_x:.6f}")
        _set_attr(img_grp, "ExtMax1", f"{ext_max_y:.6f}")
        _set_attr(img_grp, "ExtMax2", f"{ext_max_z:.6f}")
        _set_attr(img_grp, "Unit", "um")
        _set_attr(img_grp, "Name", str(dataset_name))
        _set_attr(
            img_grp,
            "RecordingDate",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        )
        _set_attr(
            img_grp,
            "ResampleDimensionX",
            str(level_shapes[-1][2]) if len(level_shapes) > 1 else str(X),
        )
        _set_attr(
            img_grp,
            "ResampleDimensionY",
            str(level_shapes[-1][1]) if len(level_shapes) > 1 else str(Y),
        )
        _set_attr(
            img_grp,
            "ResampleDimensionZ",
            str(level_shapes[-1][0]) if len(level_shapes) > 1 else str(Z),
        )

        # Channel groups
        for c in range(C):
            ch_info = channels[c] if c < len(channels) else {}
            ch_grp = info.create_group(f"Channel {c}")

            ch_name = ch_info.get("name", f"Channel {c}")
            _set_attr(ch_grp, "Name", ch_name)
            _set_attr(ch_grp, "ColorMode", "BaseColor")
            _set_attr(ch_grp, "ColorOpacity", "1")

            # Color: default to white for single channel, otherwise cycle
            color = _default_channel_color(c, C)
            _set_attr(ch_grp, "Color", color)

            # Data range
            cmin = channel_mins[c] if np.isfinite(channel_mins[c]) else 0
            cmax = channel_maxs[c] if np.isfinite(channel_maxs[c]) else 1
            _set_attr(ch_grp, "ColorRange", f"{cmin:.6f} {cmax:.6f}")
            _set_attr(ch_grp, "Min", f"{cmin:.6f}")
            _set_attr(ch_grp, "Max", f"{cmax:.6f}")

            # Wavelength info if available
            em = ch_info.get("emission_wavelength")
            if em is not None:
                _set_attr(ch_grp, "LSMEmissionWavelength", str(em))
            ex = ch_info.get("excitation_wavelength")
            if ex is not None:
                _set_attr(ch_grp, "LSMExcitationWavelength", str(ex))

        # Imaris group
        imaris_grp = info.create_group("Imaris")
        _set_attr(imaris_grp, "Version", "5.5")
        _set_attr(imaris_grp, "ThumbnailMode", "thumbnailMIP")

        # ImarisDataSet group
        ids_grp = info.create_group("ImarisDataSet")
        _set_attr(ids_grp, "Creator", "pyvistra")
        _set_attr(ids_grp, "Version", "5.5.0")

        # TimeInfo group
        ti_grp = info.create_group("TimeInfo")
        _set_attr(ti_grp, "DataSetTimePoints", str(T))
        _set_attr(ti_grp, "FileTimePoints", str(T))
        for t in range(T):
            if t < len(timestamps) and timestamps[t] is not None:
                ts_str = timestamps[t].strftime("%Y-%m-%d %H:%M:%S.%f")
            else:
                ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            _set_attr(ti_grp, f"TimePoint{t + 1}", ts_str)

        # Log group
        log_grp = info.create_group("Log")
        _set_attr(log_grp, "Entries", "0")

        # --- Thumbnail ---
        thumb = _generate_thumbnail(data)
        thumb_grp = f.create_group("Thumbnail")
        thumb_grp.create_dataset("Data", data=thumb)


# --- Internal helpers ---


def _encode_imaris_attribute(value):
    """Encode a string as Imaris byte-array attribute format.

    "600" -> np.array([b'6', b'0', b'0'], dtype='|S1')
    """
    s = str(value)
    return np.array([c.encode("utf-8") for c in s], dtype="|S1")


def _set_attr(obj, key, value):
    """Set an HDF5 attribute in Imaris character-array format."""
    obj.attrs[key] = _encode_imaris_attribute(value)


def _compute_extents(shape_zyx, voxel_size_zyx):
    """Compute Imaris extent values.

    Returns:
        dict with ExtMin0..2 and ExtMax0..2
    """
    z, y, x = shape_zyx
    vz, vy, vx = voxel_size_zyx
    return {
        "ExtMin0": 0.0,
        "ExtMin1": 0.0,
        "ExtMin2": 0.0,
        "ExtMax0": x * vx,
        "ExtMax1": y * vy,
        "ExtMax2": z * vz,
    }


def _compute_chunk_size(z, y, x, dtype):
    """Compute chunk size targeting ~1 MB per chunk."""
    bytes_per_voxel = np.dtype(dtype).itemsize
    target_bytes = 1 * 1024 * 1024  # 1 MB

    # Start with reasonable defaults
    cx = min(128, x)
    cy = min(128, y)
    cz = min(64, z)

    # Adjust if too large
    while cx * cy * cz * bytes_per_voxel > target_bytes and cz > 1:
        cz = max(1, cz // 2)
    while cx * cy * cz * bytes_per_voxel > target_bytes and cy > 16:
        cy = max(16, cy // 2)
    while cx * cy * cz * bytes_per_voxel > target_bytes and cx > 16:
        cx = max(16, cx // 2)

    return (cz, cy, cx)


def _compute_histogram(volume):
    """Compute 256-bin histogram of a 3D volume.

    Returns:
        (histogram_uint64, min_val, max_val)
    """
    vmin = float(volume.min())
    vmax = float(volume.max())

    if vmin == vmax:
        hist = np.zeros(256, dtype=np.uint64)
        hist[0] = volume.size
        return hist, vmin, vmax

    # For integer types, use integer range
    if np.issubdtype(volume.dtype, np.integer):
        info = np.iinfo(volume.dtype)
        hist, _ = np.histogram(
            volume.ravel(),
            bins=256,
            range=(info.min, info.max),
        )
    else:
        hist, _ = np.histogram(
            volume.ravel(),
            bins=256,
            range=(vmin, vmax),
        )

    return hist.astype(np.uint64), vmin, vmax


def _downsample_3d(volume, target_shape):
    """Downsample a 3D volume to target_shape using linear interpolation."""
    from scipy.ndimage import zoom

    factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    return zoom(volume, factors, order=1)


def _compute_resolution_levels(z, y, x):
    """Compute pyramid resolution level shapes.

    Downsamples 2x per level until total voxels < 1M.
    Includes the final level that drops below the threshold.

    Returns:
        List of (z, y, x) tuples, starting with full resolution.
    """
    levels = [(z, y, x)]
    cz, cy, cx = z, y, x

    while True:
        nz = max(1, cz // 2)
        ny = max(1, cy // 2)
        nx = max(1, cx // 2)

        # Stop if no actual downsampling occurred
        if (nz, ny, nx) == (cz, cy, cx):
            break

        cz, cy, cx = nz, ny, nx
        levels.append((cz, cy, cx))

        # Stop after adding a level that's small enough
        if cz * cy * cx < 1_000_000:
            break

    return levels


def _default_channel_color(index, total):
    """Return Imaris-format color string for a channel."""
    if total == 1:
        return "1.000 1.000 1.000"  # white for single channel

    # Standard microscopy palette: green, red, blue, magenta, cyan, yellow
    palette = [
        "0.000 1.000 0.000",  # green
        "1.000 0.000 0.000",  # red
        "0.000 0.000 1.000",  # blue
        "1.000 0.000 1.000",  # magenta
        "0.000 1.000 1.000",  # cyan
        "1.000 1.000 0.000",  # yellow
    ]
    return palette[index % len(palette)]


def _generate_thumbnail(data_5d, size=256):
    """Generate an RGBA thumbnail from 5D data.

    Takes the mid-Z slice of the first timepoint, max-projects channels,
    normalizes, and returns RGBA uint8 array.
    """
    T, Z, C, Y, X = data_5d.shape

    # Mid-Z of first timepoint
    mid_z = Z // 2
    try:
        slice_2d = np.asarray(data_5d[0, mid_z, :, :, :])  # (C, Y, X)
    except Exception:
        return np.zeros((size, size, 4), dtype=np.uint8)

    # Max project across channels
    if slice_2d.ndim == 3 and slice_2d.shape[0] > 1:
        img = np.max(slice_2d, axis=0)  # (Y, X)
    elif slice_2d.ndim == 3:
        img = slice_2d[0]
    else:
        img = slice_2d

    # Normalize to 0-255
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Resize to thumbnail size
    from scipy.ndimage import zoom

    fy = size / img.shape[0]
    fx = size / img.shape[1]
    f = min(fy, fx)
    img = zoom(img, f, order=1)

    # Pad to exact size if needed
    thumb = np.zeros((size, size), dtype=np.uint8)
    oy = (size - img.shape[0]) // 2
    ox = (size - img.shape[1]) // 2
    thumb[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img[
        :size, :size
    ]

    # Convert to RGBA
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:, :, 0] = thumb
    rgba[:, :, 1] = thumb
    rgba[:, :, 2] = thumb
    rgba[:, :, 3] = 255

    return rgba
