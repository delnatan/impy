import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import tifffile
import zarr

from .imaris_reader import ImarisReader

# Buffer directory for temporary Zarr files
BUFFER_DIR = Path.home() / '.pyvistra' / 'buffers'


def is_rgb_image(arr):
    """
    Detect if an array is likely an RGB/RGBA image.

    RGB images have shape (Y, X, 3) or (Y, X, 4) with the last dimension
    being small (3 or 4 for RGB/RGBA).

    Returns:
        bool: True if array appears to be RGB/RGBA
    """
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # Additional heuristic: Y and X should be larger than color channels
        if arr.shape[0] > 4 and arr.shape[1] > 4:
            return True
    return False


def load_standard_image(filepath):
    """
    Load standard image formats (PNG, JPEG, etc.) using matplotlib.

    Returns:
        tuple: (numpy array, is_rgb flag)
    """
    import matplotlib.image as mpimg

    img = mpimg.imread(filepath)

    # matplotlib returns floats [0,1] for PNG, uint8 for JPEG
    # Normalize to consistent format
    if img.dtype == np.float32 or img.dtype == np.float64:
        # Convert to uint8 for consistency
        img = (img * 255).astype(np.uint8)

    return img, is_rgb_image(img)


class Imaris5DProxy:
    """
    Wraps ImarisReader to behave like a 5D numpy array (Time, Z, Channel, Y, X).
    This allows Vispy to 'slice' it without loading the whole file.
    """

    def __init__(self, reader):
        self.reader = reader
        # ImarisReader shape is (T, C, Z, Y, X)
        # We want (T, Z, C, Y, X) to match our application standard
        t, c, z, y, x = reader.shape
        self.shape = (t, z, c, y, x)
        self.dtype = reader.dtype
        self.ndim = 5

    def close(self):
        """Close the underlying HDF5 file handle."""
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, key):
        """
        Intercepts slicing: data[t, z, c, y, x]
        """
        # Ensure key is a tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Expand Ellipsis to fill missing dimensions
        if Ellipsis in key:
            ellipsis_idx = key.index(Ellipsis)
            n_non_ellipsis = len(key) - 1
            n_expand = 5 - n_non_ellipsis
            key = (
                key[:ellipsis_idx]
                + (slice(None),) * n_expand
                + key[ellipsis_idx + 1 :]
            )

        # Fill missing dimensions with full slices
        if len(key) < 5:
            key = key + (slice(None),) * (5 - len(key))

        t_idx, z_idx, c_idx, y_idx, x_idx = key

        # --- Handle Time Slicing ---
        if isinstance(t_idx, slice):
            # Iterate over timepoints
            start, stop, step = t_idx.indices(self.shape[0])
            t_indices = range(start, stop, step)

            if len(t_indices) == 0:
                # Return empty array with correct dimensionality
                # We need to know the shape of the rest to return correct empty
                # Let's just return empty of 5D?
                # Shape: (0, Z', C', Y', X')
                # It's complex to calculate exact shape without reading.
                # Simplified: return empty array
                return np.empty((0,) + self.shape[1:], dtype=self.dtype)

            stack = []
            for t in t_indices:
                stack.append(self._read_timepoint(t, z_idx, c_idx))

            # Stack along Time (axis 0)
            # Result: (T, ...)
            data = np.array(stack)

            # Apply Y/X slicing
            # data is (T, Z, C, Y, X) or (T, C, Y, X) etc.
            # We need to apply y_idx, x_idx to the last two dimensions
            return data[..., y_idx, x_idx]

        else:
            # Single Timepoint
            data = self._read_timepoint(t_idx, z_idx, c_idx)
            return data[..., y_idx, x_idx]

    def _read_timepoint(self, t, z_idx, c_idx):
        """
        Reads a single timepoint with Z and C slicing.
        Returns data with shape (Z, C, Y, X) or subset.
        """
        # --- Handle Channel Slicing ---
        if isinstance(c_idx, slice):
            start, stop, step = c_idx.indices(self.shape[2])
            channels = range(start, stop, step)

            planes = []
            for c in channels:
                planes.append(self._read_z_slice(c, t, z_idx))

            # Stack into (C, ...)
            stack = np.array(planes)

            # If Z was also sliced (or is full stack), stack is (C, Z, Y, X).
            # We want (Z, C, Y, X).
            # If z_idx was int, stack is (C, Y, X) -> No transpose needed.
            if stack.ndim == 4:
                stack = np.transpose(stack, (1, 0, 2, 3))

            return stack

        else:
            # Single channel
            return self._read_z_slice(c_idx, t, z_idx)

    def _read_z_slice(self, c, t, z):
        """
        Helper to read Z-slice/stack for specific C and T.
        Optimized to use full-volume read if z is full slice.
        """
        if isinstance(z, slice):
            start, stop, step = z.indices(self.shape[1])
            z_indices = range(start, stop, step)

            # Optimization: If full Z-stack requested (step=1 and full range)
            if step == 1 and start == 0 and stop == self.shape[1]:
                return self.reader.read(c=c, t=t, z=None)

            if len(z_indices) == 0:
                return np.zeros(
                    (0, self.shape[3], self.shape[4]), dtype=self.dtype
                )

            # Read specific planes
            stack = []
            for z_i in z_indices:
                stack.append(self.reader.read(c=c, t=t, z=z_i))

            return np.array(stack)
        else:
            return self.reader.read(c=c, t=t, z=z)


class Numpy5DProxy:
    """
    Wraps a 5D numpy array (T, Z, C, Y, X) to support Z-projection slicing.
    """

    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.ndim = 5

    def __getitem__(self, key):
        # Ensure key is a tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Expand Ellipsis to fill missing dimensions
        if Ellipsis in key:
            ellipsis_idx = key.index(Ellipsis)
            n_non_ellipsis = len(key) - 1
            n_expand = 5 - n_non_ellipsis
            key = (
                key[:ellipsis_idx]
                + (slice(None),) * n_expand
                + key[ellipsis_idx + 1 :]
            )

        # Fill missing dimensions with full slices
        if len(key) < 5:
            key = key + (slice(None),) * (5 - len(key))

        # Standard slicing
        return self.array[key]


class Zarr5DProxy:
    """
    Wraps a zarr array to behave like a 5D numpy array (T, Z, C, Y, X).
    Data is loaded lazily - only requested slices are read from disk.
    """

    def __init__(self, zarr_array, source_ndim):
        """
        Args:
            zarr_array: The zarr.Array to wrap
            source_ndim: Original dimensionality of the data (for reshaping)
        """
        self._store = zarr_array
        self._source_ndim = source_ndim
        self.dtype = zarr_array.dtype
        self.ndim = 5

        # Calculate 5D shape based on source dimensions
        src_shape = zarr_array.shape
        if source_ndim == 2:  # (Y, X) -> (1, 1, 1, Y, X)
            self.shape = (1, 1, 1, src_shape[0], src_shape[1])
        elif source_ndim == 3:  # (Z, Y, X) -> (1, Z, 1, Y, X)
            self.shape = (1, src_shape[0], 1, src_shape[1], src_shape[2])
        elif source_ndim == 4:  # (Z, C, Y, X) -> (1, Z, C, Y, X)
            self.shape = (1, src_shape[0], src_shape[1], src_shape[2], src_shape[3])
        elif source_ndim == 5:  # Already 5D
            self.shape = src_shape
        else:
            raise ValueError(f"Unsupported zarr array dimensionality: {source_ndim}")

    def _normalize_key(self, key):
        """Normalize slicing key to 5D tuple."""
        if not isinstance(key, tuple):
            key = (key,)

        # Expand Ellipsis
        if Ellipsis in key:
            ellipsis_idx = key.index(Ellipsis)
            n_non_ellipsis = len(key) - 1
            n_expand = 5 - n_non_ellipsis
            key = (
                key[:ellipsis_idx]
                + (slice(None),) * n_expand
                + key[ellipsis_idx + 1 :]
            )

        # Fill missing dimensions
        if len(key) < 5:
            key = key + (slice(None),) * (5 - len(key))

        return key

    def _map_key_to_source(self, key_5d):
        """Map 5D key back to source array dimensions."""
        t, z, c, y, x = key_5d

        if self._source_ndim == 2:
            # Source is (Y, X), ignore T, Z, C
            return (y, x)
        elif self._source_ndim == 3:
            # Source is (Z, Y, X), ignore T, C
            return (z, y, x)
        elif self._source_ndim == 4:
            # Source is (Z, C, Y, X), ignore T
            return (z, c, y, x)
        else:
            # Source is 5D
            return key_5d

    def __getitem__(self, key):
        key_5d = self._normalize_key(key)
        source_key = self._map_key_to_source(key_5d)

        # Load data from zarr (lazy - only loads requested slice)
        data = np.asarray(self._store[source_key])

        # Reshape result to match expected 5D output shape
        # This handles singleton dimensions that were indexed with integers
        return data

    def close(self):
        """Close the zarr store if it has a close method."""
        if hasattr(self._store, 'store') and hasattr(self._store.store, 'close'):
            self._store.store.close()


class ImageBuffer:
    """
    Zarr-backed 5D array buffer for streaming image operations.

    Same interface as Numpy5DProxy for reading, plus write support.
    Temporary files are stored in ~/.pyvistra/buffers/ and cleaned up on close.
    """

    def __init__(self, shape, dtype, chunks=None, metadata=None):
        """
        Create a new buffer.

        Args:
            shape: 5D shape (T, Z, C, Y, X)
            dtype: numpy dtype
            chunks: Chunk shape, default (1, 16, C, 512, 512)
            metadata: Optional dict to preserve
        """
        BUFFER_DIR.mkdir(parents=True, exist_ok=True)
        self._path = BUFFER_DIR / f"{uuid.uuid4()}.zarr"

        T, Z, C, Y, X = shape
        if chunks is None:
            chunks = (1, min(16, Z), C, min(512, Y), min(512, X))

        self._store = zarr.open(
            str(self._path),
            mode='w',
            shape=shape,
            dtype=dtype,
            chunks=chunks,
        )

        self.metadata = metadata or {}
        self.ndim = 5

    @property
    def shape(self):
        return self._store.shape

    @property
    def dtype(self):
        return self._store.dtype

    def __getitem__(self, key):
        """Read slices - same interface as proxies."""
        return np.asarray(self._store[key])

    def __setitem__(self, key, value):
        """Write slices."""
        self._store[key] = value

    def save_as(self, filepath):
        """Export buffer to OME-TIFF."""
        scale = self.metadata.get('scale', (1.0, 1.0, 1.0))
        save_tiff(filepath, self._store[:], scale=scale)

    def close(self):
        """Close and delete the temporary buffer file."""
        if self._path.exists():
            shutil.rmtree(self._path)

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass


def apply_transform(source, rotation_deg, translate, metadata=None, progress_cb=None):
    """
    Apply 2D rotation and translation to create a new buffer.

    Matches vispy's transform convention:
    - Rotation is CCW for positive angles
    - Translation is applied after rotation (in output space)

    Args:
        source: Source proxy (any 5D array-like with shape attribute)
        rotation_deg: Rotation angle in degrees (positive = CCW)
        translate: (tx, ty) translation in pixels (applied after rotation)
        metadata: Optional metadata dict to attach to buffer
        progress_cb: Optional callback(progress_fraction)

    Returns:
        ImageBuffer with transformed data
    """
    from scipy.ndimage import affine_transform

    T, Z, C, Y, X = source.shape

    # Create output buffer
    buffer = ImageBuffer(
        shape=source.shape,
        dtype=source.dtype,
        metadata=metadata or getattr(source, 'metadata', {}),
    )

    # Build affine transform matrix (rotation around center + translation)
    # scipy uses inverse mapping: output[o] = input[matrix @ o + offset]
    # Negate angle because vispy's camera flips Y, inverting visual rotation direction
    cx, cy = X / 2, Y / 2
    theta = np.radians(-rotation_deg)  # Negate to match vispy's flipped-Y display
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    tx, ty = translate

    # 3D matrix: identity on batch dimension (Z*C), rotation on Y-X
    # This allows transforming all Z and C slices in one call
    matrix_3d = np.array([
        [1, 0, 0],
        [0, cos_t, sin_t],
        [0, -sin_t, cos_t]
    ])

    # Offset for rotation around center with translation applied after rotation
    # Translation in output space means we subtract it before inverse-rotating
    offset_3d = np.array([
        0,  # batch dimension unchanged
        cy * (1 - cos_t) - sin_t * cx - cos_t * ty - sin_t * tx,
        cx * (1 - cos_t) + sin_t * cy + sin_t * ty - cos_t * tx
    ])

    for t in range(T):
        # Load full volume for this timepoint: (Z, C, Y, X)
        volume = source[t, :, :, :, :]

        # Reshape to (Z*C, Y, X) for batch processing
        batch = volume.reshape(Z * C, Y, X)

        # Apply 3D affine transform (identity on batch dim, rotation on Y-X)
        transformed = affine_transform(batch, matrix_3d, offset_3d, order=1)

        # Reshape back to (Z, C, Y, X) and write
        buffer[t, :, :, :, :] = transformed.reshape(Z, C, Y, X)

        if progress_cb:
            progress_cb((t + 1) / T)

    return buffer


def normalize_to_5d(data, dims=None, rgb=None):
    """
    Normalizes a numpy array to (T, Z, C, Y, X) format.

    Args:
        data (np.ndarray): Input array.
        dims (str): Optional dimension string (e.g. 'tyx', 'zcyx', 'yxc' for RGB).
                    If None, heuristics are used.
        rgb (bool): If True, treat as RGB image. If None, auto-detect.

    Returns:
        Numpy5DProxy: Wrapped data.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a numpy array")

    final_img = data

    # Auto-detect RGB if not specified
    if rgb is None:
        rgb = is_rgb_image(data)

    if dims:
        dims = dims.lower()
        if len(dims) != data.ndim:
            raise ValueError(
                f"dims string length ({len(dims)}) must match data ndim ({data.ndim})"
            )

        # Target: t, z, c, y, x
        target_order = ["t", "z", "c", "y", "x"]

        present_dims = [d for d in target_order if d in dims]
        perm = [dims.index(d) for d in present_dims]

        final_img = np.transpose(data, perm)

        # Calculate target shape
        target_shape = []
        for char in target_order:
            if char in dims:
                target_shape.append(data.shape[dims.index(char)])
            else:
                target_shape.append(1)

        final_img = final_img.reshape(target_shape)

    else:
        # Heuristics
        ndim = data.ndim
        if ndim == 2:  # (Y, X) -> (1, 1, 1, Y, X)
            final_img = data[np.newaxis, np.newaxis, np.newaxis, :, :]
        elif ndim == 3:
            if rgb:
                # RGB image: (Y, X, C) -> (1, 1, C, Y, X)
                final_img = data.transpose(2, 0, 1)[np.newaxis, np.newaxis, :, :, :]
            else:
                # Z-stack: (Z, Y, X) -> (1, Z, 1, Y, X)
                final_img = data[np.newaxis, :, np.newaxis, :, :]
        elif ndim == 4:  # Assume (Z, C, Y, X) -> (1, Z, C, Y, X)
            final_img = data[np.newaxis, :, :, :, :]
        elif ndim == 5:  # Assume (T, Z, C, Y, X)
            final_img = data

    return Numpy5DProxy(final_img)


def load_zarr(filepath):
    """
    Load a zarr array from a .zarr directory with lazy loading.

    Supports:
        - Standard zarr arrays (any dimensionality, normalized to 5D)
        - OME-Zarr (multiscale, uses highest resolution from '0/')

    Args:
        filepath: Path to .zarr directory

    Returns:
        tuple: (Zarr5DProxy, metadata_dict)
    """
    filepath = str(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Zarr file not found: {filepath}")

    # Open zarr store
    store = zarr.open(filepath, mode='r')

    # Check if it's a Group or Array
    is_group = isinstance(store, zarr.Group)

    # Find the zarr array to wrap (don't load data yet)
    zarr_array = None
    attrs = {}

    if is_group and '0' in store:
        # OME-Zarr: use highest resolution level
        zarr_array = store['0']
        attrs = dict(store.attrs) if hasattr(store, 'attrs') else {}
    elif is_group:
        # Group without '0' - look for a data array
        for key in ['data', 'array']:
            if key in store and isinstance(store[key], zarr.Array):
                zarr_array = store[key]
                break
        if zarr_array is None:
            # Find first array in group
            for key in store.keys():
                if isinstance(store[key], zarr.Array):
                    zarr_array = store[key]
                    break
        if zarr_array is None:
            raise ValueError(f"No array found in zarr group: {filepath}")
        attrs = dict(store.attrs) if hasattr(store, 'attrs') else {}
    else:
        # Direct zarr array
        zarr_array = store
        attrs = dict(store.attrs) if hasattr(store, 'attrs') else {}

    # Wrap in lazy proxy
    source_ndim = len(zarr_array.shape)
    proxy = Zarr5DProxy(zarr_array, source_ndim)

    # Build metadata
    metadata = {
        'filename': os.path.basename(filepath),
        'shape': proxy.shape,
        'is_rgb': False,
    }

    # Extract scale from attrs if available
    if 'scale' in attrs:
        metadata['scale'] = tuple(attrs['scale'])
    elif 'spacing' in attrs:
        metadata['scale'] = tuple(attrs['spacing'])
    else:
        metadata['scale'] = (1.0, 1.0, 1.0)

    # Copy other attrs to metadata
    for key, value in attrs.items():
        if key not in metadata:
            metadata[key] = value

    return proxy, metadata


def load_image(filepath, use_memmap=True):
    """
    Loads an image and normalizes it to (T, Z, C, Y, X).
    Returns: (image_data_proxy, metadata_dict)

    Supported formats:
        - .ims (Imaris)
        - .tif, .tiff (TIFF)
        - .png, .jpg, .jpeg (standard images via matplotlib)
        - .zarr (Zarr arrays, including OME-Zarr)
        - .psf.zarr (PSF files)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Handle PSF zarr files (check before general .zarr)
    if filepath.endswith('.psf.zarr') or filepath.endswith('.psf'):
        return load_psf(filepath)

    # Handle general zarr directories
    if filepath.endswith('.zarr') and os.path.isdir(filepath):
        return load_zarr(filepath)

    ext = os.path.splitext(filepath)[1].lower()

    # --- IMARIS PATH ---
    if ext == ".ims":
        reader = ImarisReader(filepath)
        data = Imaris5DProxy(reader)

        meta = {
            "filename": os.path.basename(filepath),
            "shape": data.shape,
            "scale": reader.voxel_size,  # (Z, Y, X)
            "channels": reader.channels_info,
            "is_rgb": False,
        }
        return data, meta

    # --- STANDARD IMAGE PATH (PNG, JPEG) ---
    if ext in (".png", ".jpg", ".jpeg"):
        img, detected_rgb = load_standard_image(filepath)

        # Normalize to 5D
        final_img = normalize_to_5d(img, rgb=detected_rgb).array

        data_proxy = Numpy5DProxy(final_img)

        return data_proxy, {
            "filename": os.path.basename(filepath),
            "shape": final_img.shape,
            "scale": (1.0, 1.0, 1.0),  # No physical scale for standard images
            "is_rgb": detected_rgb,
        }

    # --- TIFF PATH ---
    scale = (1.0, 1.0, 1.0)

    if use_memmap:
        img = tifffile.memmap(filepath)
    else:
        img = tifffile.imread(filepath)

    # Detect RGB before any transformation
    detected_rgb = is_rgb_image(img)

    # Extract Metadata
    try:
        with tifffile.TiffFile(filepath) as tif:
            # Z-spacing (ImageJ metadata)
            ij_meta = tif.imagej_metadata
            sz = 1.0
            if ij_meta and "spacing" in ij_meta:
                sz = ij_meta["spacing"]

            # XY-spacing (Tags)
            # Resolution is usually (numerator, denominator) or float
            # TIFF resolution is pixels per unit.
            # We want unit per pixel (micron/pixel).
            page = tif.pages[0]
            sx, sy = 1.0, 1.0

            # Check Unit
            # 1: None, 2: Inch, 3: cm
            unit = page.tags.get("ResolutionUnit")
            unit_val = unit.value if unit else 0

            x_res = page.tags.get("XResolution")
            y_res = page.tags.get("YResolution")

            if x_res and y_res:
                rx = x_res.value
                ry = y_res.value

                # Handle tuple (num, den)
                if isinstance(rx, tuple):
                    rx = rx[0] / rx[1] if rx[1] != 0 else 0
                if isinstance(ry, tuple):
                    ry = ry[0] / ry[1] if ry[1] != 0 else 0

                if rx > 0:
                    sx = 1.0 / rx
                if ry > 0:
                    sy = 1.0 / ry

                # Convert to microns if needed
                if unit_val == 2:  # Inch
                    sx *= 25400.0
                    sy *= 25400.0
                elif unit_val == 3:  # cm
                    sx *= 10000.0
                    sy *= 10000.0

            scale = (sz, sy, sx)

    except Exception as e:
        print(f"Warning: Could not read TIFF metadata: {e}")

    # Use normalize_to_5d with RGB detection
    final_img = normalize_to_5d(img, rgb=detected_rgb).array

    # Wrap in Proxy
    data_proxy = Numpy5DProxy(final_img)

    return data_proxy, {
        "filename": os.path.basename(filepath),
        "shape": final_img.shape,
        "scale": scale,
        "is_rgb": detected_rgb,
    }


def save_tiff(filepath, data, scale=(1.0, 1.0, 1.0), axes="TZCYX", input_axes=None):
    """
    Saves a 5D array to a TIFF file with metadata.

    Args:
        filepath (str): Output path.
        data (array-like): Image data. If input_axes is None, expects 5D (T, Z, C, Y, X).
        scale (tuple): Voxel size (z, y, x).
        axes (str): Dimension order for output TIFF metadata.
        input_axes (str): Optional axes string describing input data order (e.g., "YX",
                          "ZYX", "CZYX"). When provided, data is normalized to 5D before
                          saving. Case-insensitive.

    Examples:
        # Save a 2D image
        save_tiff("out.tif", img_2d, input_axes="YX")

        # Save a 3D z-stack
        save_tiff("out.tif", zstack, input_axes="ZYX")

        # Save with channel dimension
        save_tiff("out.tif", multichannel, input_axes="CZYX")
    """
    # Ensure data is numpy array (loads into memory)
    # If it's a proxy, slicing [:] triggers reading.
    # We use np.asarray to avoid copying if it's already an array
    try:
        image = np.asarray(data[:])
    except TypeError:
        # Fallback if slicing not supported directly or data is list
        image = np.asarray(data)

    # Normalize to 5D if input_axes is specified
    if input_axes is not None:
        image = normalize_to_5d(image, dims=input_axes).array

    sz, sy, sx = scale

    # Resolution (pixels per unit)
    # If unit is 'um', then 1/sx.
    # Avoid division by zero
    rx = 1.0 / sx if sx > 0 else 1.0
    ry = 1.0 / sy if sy > 0 else 1.0

    metadata = {
        "axes": axes,
        "spacing": sz,
        "unit": "um",
    }

    tifffile.imwrite(
        filepath, image, imagej=True, resolution=(rx, ry), metadata=metadata
    )


def save_psf(filepath, psf_data, metadata):
    """
    Save PSF to .psf.zarr format.

    Args:
        filepath: Path ending in .psf.zarr
        psf_data: 3D array (Nz, Ny, Nx) or 5D ImageBuffer/array
        metadata: dict with PSF parameters
    """
    import json
    from datetime import datetime

    # Ensure .psf.zarr extension
    filepath = str(filepath)
    if not filepath.endswith('.psf.zarr'):
        filepath += '.psf.zarr'

    # Get data as numpy array
    if hasattr(psf_data, '__getitem__'):
        data = np.asarray(psf_data[:])
    else:
        data = np.asarray(psf_data)

    # Normalize to 5D if needed (3D -> 5D)
    if data.ndim == 3:
        # (Nz, Ny, Nx) -> (1, Nz, 1, Ny, Nx)
        data = data[np.newaxis, :, np.newaxis, :, :]
    elif data.ndim != 5:
        raise ValueError(f"PSF data must be 3D or 5D, got {data.ndim}D")

    # Remove existing directory if it exists
    filepath_path = Path(filepath)
    if filepath_path.exists():
        shutil.rmtree(filepath_path)

    # Create zarr array
    store = zarr.open(
        filepath,
        mode='w',
        shape=data.shape,
        dtype=data.dtype,
        chunks=(1, min(16, data.shape[1]), data.shape[2], min(512, data.shape[3]), min(512, data.shape[4])),
    )

    # Write data
    store[:] = data

    # Write metadata to .zattrs
    # Zarr stores attrs in a separate file
    store.attrs.update(metadata)


def load_psf(filepath):
    """
    Load PSF from .psf.zarr format.

    Args:
        filepath: Path to .psf.zarr directory

    Returns:
        tuple: (Numpy5DProxy, metadata_dict)
    """
    filepath = str(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PSF file not found: {filepath}")

    # Open zarr store
    store = zarr.open(filepath, mode='r')

    # Read data into memory and wrap in proxy
    data = np.asarray(store[:])
    proxy = Numpy5DProxy(data)

    # Read metadata from .zattrs
    metadata = dict(store.attrs)

    # Add filename to metadata
    metadata['filename'] = os.path.basename(filepath)
    metadata['shape'] = data.shape
    metadata['is_rgb'] = False

    # Extract scale from spacing if available
    if 'spacing' in metadata:
        spacing = metadata['spacing']
        metadata['scale'] = tuple(spacing)

    return proxy, metadata
