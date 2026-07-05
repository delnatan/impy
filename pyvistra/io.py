import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tifffile
import zarr

from .readers import ImarisReader
from .data.proxies import CZI5DProxy, Imaris5DProxy, Numpy5DProxy, Zarr5DProxy
from .data.buffer import ImageBuffer


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
    Load standard image formats (PNG, JPEG, etc.) using Pillow.

    Returns:
        tuple: (numpy array uint8, is_rgb flag)
    """
    from PIL import Image

    img = np.asarray(Image.open(filepath))
    return img, is_rgb_image(img)


def apply_transform(
    source, rotation_deg, translate, metadata=None, progress_cb=None
):
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
        metadata=metadata or getattr(source, "metadata", {}),
    )

    # Build affine transform matrix (rotation around center + translation)
    # scipy uses inverse mapping: output[o] = input[matrix @ o + offset]
    # Negate angle because vispy's camera flips Y, inverting visual rotation direction
    cx, cy = X / 2, Y / 2
    theta = np.radians(
        -rotation_deg
    )  # Negate to match vispy's flipped-Y display
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    tx, ty = translate

    # 3D matrix: identity on batch dimension (Z*C), rotation on Y-X
    # This allows transforming all Z and C slices in one call
    matrix_3d = np.array([[1, 0, 0], [0, cos_t, sin_t], [0, -sin_t, cos_t]])

    # Offset for rotation around center with translation applied after rotation
    # Translation in output space means we subtract it before inverse-rotating
    offset_3d = np.array(
        [
            0,  # batch dimension unchanged
            cy * (1 - cos_t) - sin_t * cx - cos_t * ty - sin_t * tx,
            cx * (1 - cos_t) + sin_t * cy + sin_t * ty - cos_t * tx,
        ]
    )

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


def _parse_timestamp_value(value):
    """Parse assorted timestamp representations into datetime or None."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, np.datetime64):
        try:
            iso = np.datetime_as_string(value, unit="us")
            return datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except Exception:
            return None

    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None

        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return datetime.strptime(txt, fmt)
            except ValueError:
                continue

        try:
            return datetime.fromisoformat(txt.replace("Z", "+00:00"))
        except ValueError:
            return None

    return None


def _normalize_timestamps_for_t(timestamps, t_size):
    """Return a list of length t_size with datetime/None entries."""
    out = [None] * max(0, int(t_size))
    if not isinstance(timestamps, (list, tuple, np.ndarray)):
        return out

    for i, ts in enumerate(list(timestamps)[:t_size]):
        out[i] = _parse_timestamp_value(ts)
    return out


def _normalize_time_seconds_for_t(timestamp_seconds, t_size):
    """Normalize timestamp_seconds to length t_size, preserving None gaps."""
    if not isinstance(timestamp_seconds, (list, tuple, np.ndarray)):
        return None

    out = []
    for item in list(timestamp_seconds)[:t_size]:
        if item is None:
            out.append(None)
            continue
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(None)

    if len(out) < t_size:
        out.extend([None] * (t_size - len(out)))
    return out


def coerce_scale_zyx(scale):
    """Return a numeric (vz, vy, vx) tuple with safe defaults."""
    if isinstance(scale, (list, tuple, np.ndarray)) and len(scale) >= 3:
        try:
            return (float(scale[0]), float(scale[1]), float(scale[2]))
        except (TypeError, ValueError):
            pass
    return (1.0, 1.0, 1.0)


def _normalize_z_range(z_range, z_size):
    """Validate and normalize a Z range to inclusive integer bounds."""
    if z_size <= 0:
        raise ValueError("Z size must be positive")

    if z_range is None:
        return (0, z_size - 1)

    if not isinstance(z_range, (list, tuple, np.ndarray)) or len(z_range) != 2:
        raise ValueError("z_range must be a 2-item tuple/list (z_start, z_end)")

    z0 = int(z_range[0])
    z1 = int(z_range[1])

    if z0 > z1:
        z0, z1 = z1, z0

    if z0 < 0 or z1 >= z_size:
        raise ValueError(
            f"z_range ({z0}, {z1}) is out of bounds for Z size {z_size}"
        )

    return (z0, z1)


def build_z_projection_metadata(
    metadata,
    source_shape,
    z_range=None,
    method="max",
):
    """
    Build output metadata for a Z projection while preserving time metadata.

    Args:
        metadata: Source metadata dict.
        source_shape: Source shape in (T, Z, C, Y, X).
        z_range: Inclusive (z_start, z_end). If None, uses full Z range.
        method: Projection method name, stored in metadata for provenance.
    """
    if len(source_shape) != 5:
        raise ValueError("source_shape must be 5D (T, Z, C, Y, X)")

    T, Z, C, Y, X = [int(v) for v in source_shape]
    z0, z1 = _normalize_z_range(z_range, Z)
    z_depth = z1 - z0 + 1

    out_meta = dict(metadata or {})
    vz, vy, vx = coerce_scale_zyx(out_meta.get("scale", (1.0, 1.0, 1.0)))

    # Keep XY spacing, encode the projected slab thickness in singleton Z scale.
    out_meta["scale"] = (vz * z_depth, vy, vx)
    out_meta["shape"] = (T, 1, C, Y, X)
    out_meta["timestamps"] = _normalize_timestamps_for_t(
        out_meta.get("timestamps"), T
    )

    timestamp_seconds = _normalize_time_seconds_for_t(
        out_meta.get("timestamp_seconds"), T
    )
    if timestamp_seconds is not None:
        out_meta["timestamp_seconds"] = timestamp_seconds

    out_meta["projection"] = {
        "axis": "z",
        "method": str(method).lower(),
        "z_range": [z0, z1],
    }

    base_name = str(out_meta.get("filename", out_meta.get("name", "Image")))
    suffix = f"_zproj_z{z0}-{z1}"
    out_meta["filename"] = f"{base_name}{suffix}"
    out_meta["name"] = f"{base_name}{suffix}"

    return out_meta


def project_z_max(source, z_range=None, metadata=None, progress_cb=None):
    """
    Max-project source data along Z into a singleton-Z ImageBuffer.

    Args:
        source: 5D array-like in (T, Z, C, Y, X) order.
        z_range: Optional inclusive (z_start, z_end). Defaults to full Z.
        metadata: Optional metadata override. Defaults to source.metadata if present.
        progress_cb: Optional callback(progress_fraction).

    Returns:
        ImageBuffer with shape (T, 1, C, Y, X) and projection-aware metadata.
    """
    if not hasattr(source, "shape") or len(source.shape) != 5:
        raise ValueError("source must be 5D (T, Z, C, Y, X)")

    T, Z, C, Y, X = [int(v) for v in source.shape]
    z0, z1 = _normalize_z_range(z_range, Z)

    source_meta = metadata
    if source_meta is None:
        source_meta = getattr(source, "metadata", {})

    out_meta = build_z_projection_metadata(
        metadata=source_meta,
        source_shape=(T, Z, C, Y, X),
        z_range=(z0, z1),
        method="max",
    )

    out_dtype = getattr(source, "dtype", None)
    if out_dtype is None:
        out_dtype = np.asarray(source[0, z0, 0, :, :]).dtype

    out = ImageBuffer(
        shape=(T, 1, C, Y, X),
        dtype=np.dtype(out_dtype),
        metadata=out_meta,
    )

    total = T * C
    done = 0
    if total == 0:
        return out

    for t in range(T):
        for c in range(C):
            stack = np.asarray(source[t, z0 : z1 + 1, c, :, :])
            if stack.ndim == 2:
                proj = stack
            else:
                proj = np.max(stack, axis=0)

            out[t, 0, c, :, :] = np.asarray(proj, dtype=out.dtype)
            done += 1
            if progress_cb is not None:
                progress_cb(done / total)

    return out


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
                final_img = data.transpose(2, 0, 1)[
                    np.newaxis, np.newaxis, :, :, :
                ]
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
    store = zarr.open(filepath, mode="r")

    # Check if it's a Group or Array
    is_group = isinstance(store, zarr.Group)

    # Find the zarr array to wrap (don't load data yet)
    zarr_array = None
    attrs = {}

    if is_group and "0" in store:
        # OME-Zarr: use highest resolution level
        zarr_array = store["0"]
        attrs = dict(store.attrs) if hasattr(store, "attrs") else {}
    elif is_group:
        # Group without '0' - look for a data array
        for key in ["data", "array"]:
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
        attrs = dict(store.attrs) if hasattr(store, "attrs") else {}
    else:
        # Direct zarr array
        zarr_array = store
        attrs = dict(store.attrs) if hasattr(store, "attrs") else {}

    # Wrap in lazy proxy
    source_ndim = len(zarr_array.shape)
    proxy = Zarr5DProxy(zarr_array, source_ndim)

    # Build metadata
    metadata = {
        "filename": os.path.basename(filepath),
        "shape": proxy.shape,
        "is_rgb": False,
    }

    # Extract scale from attrs if available
    if "scale" in attrs:
        metadata["scale"] = tuple(attrs["scale"])
    elif "spacing" in attrs:
        metadata["scale"] = tuple(attrs["spacing"])
    else:
        metadata["scale"] = (1.0, 1.0, 1.0)

    # Copy other attrs to metadata
    for key, value in attrs.items():
        if key not in metadata:
            metadata[key] = value

    return proxy, metadata


def _nd2_to_plain(value):
    """Convert ND2 metadata objects to plain Python containers."""
    if is_dataclass(value):
        return {k: _nd2_to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _nd2_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nd2_to_plain(v) for v in value]
    return value


def _nd2_attr(obj, key, default=None):
    """Read an attribute from dict-like or object-like values."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _nd2_first_value(obj, names):
    """Return first non-None value from candidate field names."""
    for name in names:
        value = _nd2_attr(obj, name)
        if value is not None:
            return value
    return None


def _extract_nd2_channels(metadata_obj, n_channels):
    """Extract channel metadata in pyvistra's channel schema."""
    channels = []
    nd2_channels = _nd2_attr(metadata_obj, "channels") or []

    for i in range(n_channels):
        info = {
            "id": i,
            "name": f"Channel {i}",
            "emission_wavelength": None,
            "excitation_wavelength": None,
            "exposure_time": None,
        }

        if i < len(nd2_channels):
            channel_obj = nd2_channels[i]
            channel_meta = _nd2_attr(channel_obj, "channel", channel_obj)

            name = _nd2_first_value(
                channel_meta,
                ("name", "channel_name", "label"),
            )
            if name:
                info["name"] = str(name)

            emission = _nd2_first_value(
                channel_meta,
                (
                    "emission_lambda_nm",
                    "emissionLambdaNm",
                    "emission_wavelength_nm",
                    "emissionWavelengthNm",
                ),
            )
            excitation = _nd2_first_value(
                channel_meta,
                (
                    "excitation_lambda_nm",
                    "excitationLambdaNm",
                    "excitation_wavelength_nm",
                    "excitationWavelengthNm",
                ),
            )

            info["emission_wavelength"] = emission
            info["excitation_wavelength"] = excitation

            time_meta = _nd2_attr(channel_obj, "time")
            exposure_ms = _nd2_first_value(
                time_meta,
                (
                    "exposureTimeMs",
                    "exposure_time_ms",
                    "exposureTime",
                    "exposure_time",
                ),
            )
            if exposure_ms is not None:
                # Store in seconds to match pyvistra metadata usage.
                try:
                    info["exposure_time"] = float(exposure_ms) / 1000.0
                except (TypeError, ValueError):
                    info["exposure_time"] = exposure_ms

        channels.append(info)

    return channels


def _normalize_nd2_to_5d(data, sizes):
    """
    Normalize ND2 array to (T, Z, C, Y, X).

    Any ND2 dimensions other than Z/C/Y/X are collapsed into T to preserve
    frame sequencing while keeping pyvistra's fixed 5D data model.
    """
    dims_keys = [str(k) for k in sizes.keys()]
    dims = [d.lower() for d in dims_keys]
    shape = tuple(data.shape)

    if len(dims) != len(shape):
        return normalize_to_5d(data).array, []

    y_axis = dims.index("y") if "y" in dims else None
    x_axis = dims.index("x") if "x" in dims else None
    if y_axis is None or x_axis is None:
        return normalize_to_5d(data).array, []

    z_axis = dims.index("z") if "z" in dims else None
    c_axis = dims.index("c") if "c" in dims else None

    time_axes = [
        i for i, d in enumerate(dims) if d not in {"x", "y", "z", "c"}
    ]
    collapsed_dims = [dims_keys[i] for i in time_axes if dims[i] != "t"]

    ordered_axes = list(time_axes)
    if z_axis is not None:
        ordered_axes.append(z_axis)
    if c_axis is not None:
        ordered_axes.append(c_axis)
    ordered_axes.extend([y_axis, x_axis])

    transposed = np.transpose(data, ordered_axes)

    t_size = (
        int(np.prod([shape[i] for i in time_axes], dtype=np.int64))
        if time_axes
        else 1
    )
    z_size = shape[z_axis] if z_axis is not None else 1
    c_size = shape[c_axis] if c_axis is not None else 1
    y_size = shape[y_axis]
    x_size = shape[x_axis]

    normalized = transposed.reshape((t_size, z_size, c_size, y_size, x_size))
    return normalized, collapsed_dims


def _extract_nd2_timestamp_seconds(nd2_file, t_size):
    """Extract per-timepoint relative acquisition times in seconds."""
    loop_indices = getattr(nd2_file, "loop_indices", None) or []
    if not loop_indices:
        return []

    # ND2 sequence indices are per-frame and may include Z/C/P loops.
    # We keep first observed time per T index.
    t_seconds = [None] * t_size
    for seq_idx, index_map in enumerate(loop_indices):
        t_idx = None
        if isinstance(index_map, dict):
            t_idx = index_map.get("T")
        else:
            t_idx = getattr(index_map, "get", lambda *_: None)("T")
        if t_idx is None or t_idx >= t_size or t_seconds[t_idx] is not None:
            continue

        try:
            frame_meta = nd2_file.frame_metadata(seq_idx)
        except Exception:
            continue

        channels = _nd2_attr(frame_meta, "channels") or []
        if not channels:
            continue
        time_meta = _nd2_attr(channels[0], "time")
        rel_ms = _nd2_first_value(
            time_meta, ("relativeTimeMs", "relative_time_ms")
        )
        if rel_ms is None:
            continue
        try:
            t_seconds[t_idx] = float(rel_ms) / 1000.0
        except (TypeError, ValueError):
            continue

    return t_seconds


def _build_nd2_timestamps(acquisition_start, rel_seconds):
    """Build datetime timestamps from acquisition start and relative seconds."""
    if acquisition_start is None:
        return []
    timestamps = []
    for dt_s in rel_seconds:
        if dt_s is None:
            timestamps.append(None)
        else:
            timestamps.append(acquisition_start + timedelta(seconds=dt_s))
    return timestamps


def _parse_nd2_acquisition_start(text_info):
    """Try to parse acquisition datetime from ND2 text info."""
    if not isinstance(text_info, dict):
        return None

    for key in (
        "date",
        "acquisition_date",
        "acquisitionDate",
        "capturing_date",
        "capturingDate",
        "time_start",
        "timeStart",
    ):
        raw = text_info.get(key)
        if not isinstance(raw, str):
            continue
        txt = raw.strip()
        if not txt:
            continue
        try:
            return datetime.fromisoformat(txt.replace("Z", "+00:00"))
        except ValueError:
            pass
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %I:%M:%S %p",
            "%m/%d/%Y %H:%M:%S",
        ):
            try:
                return datetime.strptime(txt, fmt)
            except ValueError:
                continue

    return None


def load_nd2(filepath):
    """
    Load ND2 image data and metadata.

    Returns:
        tuple: (Numpy5DProxy, metadata_dict)
    """
    try:
        import nd2
    except ImportError:
        raise ImportError(
            "ND2 support requires nd2. Install with: pip install pyvistra[nd2]"
        )

    with nd2.ND2File(filepath) as f:
        raw = f.asarray()
        sizes = dict(f.sizes)
        normalized, collapsed_dims = _normalize_nd2_to_5d(raw, sizes)

        voxel = f.voxel_size()
        sx = float(getattr(voxel, "x", 1.0) or 1.0)
        sy = float(getattr(voxel, "y", 1.0) or 1.0)
        sz = float(getattr(voxel, "z", 1.0) or 1.0)

        metadata_obj = _nd2_to_plain(getattr(f, "metadata", None))
        text_info = _nd2_to_plain(getattr(f, "text_info", None))
        attributes = _nd2_to_plain(getattr(f, "attributes", None))
        experiment = _nd2_to_plain(getattr(f, "experiment", None))

        n_channels = normalized.shape[2]
        channels = _extract_nd2_channels(metadata_obj, n_channels)

        rel_seconds = _extract_nd2_timestamp_seconds(
            f, normalized.shape[0]
        )
        acquisition_start = _parse_nd2_acquisition_start(text_info)
        timestamps = _build_nd2_timestamps(acquisition_start, rel_seconds)

        events = None
        try:
            events = f.events(orient="records")
        except TypeError:
            try:
                events = f.events()
            except Exception:
                events = None
        except Exception:
            events = None
        if hasattr(events, "to_dict"):
            events = events.to_dict(orient="records")

    data_proxy = Numpy5DProxy(normalized)

    meta = {
        "filename": os.path.basename(filepath),
        "shape": normalized.shape,
        "raw_shape": raw.shape,
        "scale": (sz, sy, sx),  # ND2 voxel size is in microns.
        "channels": channels,
        "is_rgb": False,
        "timestamps": timestamps,
        "timestamp_seconds": rel_seconds,
        "nd2_sizes": sizes,
        "nd2_collapsed_dims": collapsed_dims,
        "nd2_text_info": text_info,
        "nd2_attributes": attributes,
        "nd2_experiment": experiment,
    }
    if metadata_obj is not None:
        meta["nd2_metadata"] = metadata_obj
    if events is not None:
        meta["nd2_events"] = events

    return data_proxy, meta


def load_image(filepath, use_memmap=True, dims=None):
    """
    Loads an image and normalizes it to (T, Z, C, Y, X).
    Returns: (image_data_proxy, metadata_dict)

    Args:
        filepath: Path to the image file.
        use_memmap: If True, use memory-mapped loading for TIFFs.
        dims: Optional dimension string for TIFF files (e.g., 'tyx', 'zyx', 'tzyx').
              If None, heuristics are used. Only applies to TIFF/PNG/JPEG formats.

    Supported formats:
        - .ims (Imaris)
        - .nd2 (Nikon ND2)
        - .czi (Zeiss CZI)
        - .tif, .tiff (TIFF)
        - .png, .jpg, .jpeg (standard images via Pillow)
        - .zarr (Zarr arrays, including OME-Zarr)
        - .psf.h5, .pupil.h5 (deconlib PSF / pupil files)
    """
    data, meta = _load_image_raw(filepath, use_memmap=use_memmap, dims=dims)
    _normalize_channels_metadata(meta)
    return data, meta


def _load_image_raw(filepath, use_memmap=True, dims=None):
    """Format-dispatching body of :func:`load_image`, before metadata
    normalization. Every format branch below is free to hand back whatever
    each reader/vendor gives it (e.g. a wavelength as ``"600 nm"``) --
    ``load_image`` cleans that up in one place afterwards."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Handle deconlib PSF / pupil files (HDF5-backed).
    if filepath.endswith(".psf.h5"):
        return _load_psf_h5(filepath)
    if filepath.endswith(".pupil.h5"):
        return _load_pupil_h5(filepath)

    # Handle general zarr directories
    if filepath.endswith(".zarr/") and os.path.isdir(filepath):
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
            "timestamps": reader.timestamps,
        }
        return data, meta

    # --- CZI PATH ---
    if ext == ".czi":
        try:
            from .readers.czi import CZIReader
        except ImportError:
            raise ImportError(
                "CZI support requires aicspylibczi. "
                "Install with: pip install pyvistra[czi]"
            )

        reader = CZIReader(filepath)
        data = CZI5DProxy(reader)

        meta = {
            "filename": os.path.basename(filepath),
            "shape": data.shape,
            "scale": reader.voxel_size,  # (Z, Y, X)
            "channels": reader.channels_info,
            "is_rgb": False,
            "timestamps": reader.timestamps,
        }
        return data, meta

    # --- ND2 PATH ---
    if ext == ".nd2":
        return load_nd2(filepath)

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
        try:
            img = tifffile.memmap(filepath)
        except ValueError:
            img = tifffile.imread(filepath)
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
            frame_interval_s = None
            if ij_meta and "spacing" in ij_meta:
                sz = ij_meta["spacing"]
            if ij_meta:
                frame_interval_s = _parse_imagej_frame_interval_seconds(ij_meta)

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
        frame_interval_s = None

    # Use normalize_to_5d with RGB detection and optional dims override
    final_img = normalize_to_5d(img, dims=dims, rgb=detected_rgb).array

    # Wrap in Proxy
    data_proxy = Numpy5DProxy(final_img)

    return data_proxy, {
        "filename": os.path.basename(filepath),
        "shape": final_img.shape,
        "raw_shape": img.shape,
        "scale": scale,
        "is_rgb": detected_rgb,
        "frame_interval_s": frame_interval_s,
    }


def _coerce_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Matches the leading numeric portion of e.g. "600 nm", "600nm", "600.5" --
# some readers (e.g. Imaris, when the vendor-stored attribute has non-numeric
# formatting) can only give us a wavelength as a string with units attached.
_LEADING_NUMBER = re.compile(r"[-+]?\d*\.?\d+")


def _coerce_wavelength_nm(value):
    """Parse a wavelength as a plain float or unit-suffixed string
    (e.g. ``"600 nm"``) into a clean float, or ``None`` if unparseable."""
    if isinstance(value, str):
        match = _LEADING_NUMBER.search(value)
        value = match.group() if match else None
    return _coerce_float_or_none(value)


def _normalize_channels_metadata(meta):
    """Coerce every channel's wavelength fields to a clean float (or None).

    Readers pass through whatever the vendor metadata gives them verbatim
    -- sometimes a plain number, sometimes a string with units baked in.
    Normalizing once here, regardless of source format, means everything
    downstream (colormap auto-assignment, PSF wavelength lookups, the
    metadata dialog, ...) can assume these fields are already numeric.
    """
    channels = meta.get("channels")
    if not channels:
        return
    for ch in channels:
        for key in ("emission_wavelength", "excitation_wavelength"):
            if key in ch:
                ch[key] = _coerce_wavelength_nm(ch[key])


def _parse_imagej_frame_interval_seconds(imagej_metadata):
    """Parse ImageJ frame interval metadata and return seconds."""
    finterval = _coerce_float_or_none(imagej_metadata.get("finterval"))
    tunit = str(imagej_metadata.get("tunit", "sec")).lower()
    if finterval is not None and finterval > 0:
        if tunit in ("sec", "s", "second", "seconds"):
            return finterval
        if tunit in ("msec", "ms", "millisecond", "milliseconds"):
            return finterval / 1000.0
        if tunit in ("usec", "us", "microsecond", "microseconds"):
            return finterval / 1_000_000.0
        if tunit in ("min", "minute", "minutes"):
            return finterval * 60.0
        if tunit in ("hour", "hours", "h"):
            return finterval * 3600.0
        return finterval

    fps = _coerce_float_or_none(imagej_metadata.get("fps"))
    if fps is not None and fps > 0:
        return 1.0 / fps
    return None


def _extract_frame_interval_seconds(metadata, t_size):
    """Infer a single frame interval (seconds) from metadata."""
    if not metadata or t_size < 2:
        return None

    for key in ("frame_interval_s", "frame_interval_seconds"):
        interval = _coerce_float_or_none(metadata.get(key))
        if interval is not None and interval > 0:
            return interval

    timestamp_seconds = metadata.get("timestamp_seconds")
    if isinstance(timestamp_seconds, (list, tuple, np.ndarray)):
        seconds = [_coerce_float_or_none(v) for v in timestamp_seconds]
        diffs = [
            b - a
            for a, b in zip(seconds[:-1], seconds[1:])
            if a is not None and b is not None and b > a
        ]
        if diffs:
            return float(np.median(diffs))

    timestamps = metadata.get("timestamps")
    if isinstance(timestamps, (list, tuple)) and len(timestamps) >= 2:
        rel_seconds = []
        t0 = None
        for ts in timestamps:
            if ts is None:
                rel_seconds.append(None)
                continue
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    rel_seconds.append(None)
                    continue
            if not hasattr(ts, "timestamp"):
                rel_seconds.append(None)
                continue
            if t0 is None:
                t0 = ts
            rel_seconds.append((ts - t0).total_seconds())

        diffs = [
            b - a
            for a, b in zip(rel_seconds[:-1], rel_seconds[1:])
            if a is not None and b is not None and b > a
        ]
        if diffs:
            return float(np.median(diffs))

    return None


def save_tiff(
    filepath,
    data,
    scale=(1.0, 1.0, 1.0),
    axes="TZCYX",
    input_axes=None,
    metadata=None,
):
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
        metadata (dict): Optional acquisition metadata. If present, this can include
                         time information (for example ``timestamp_seconds``,
                         ``timestamps``, or ``frame_interval_s``) which will be written
                         using ImageJ's ``finterval``/``fps`` fields.

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

    # FFT output (see fft_dialog.py) tags itself "frequency" space and
    # replaces "scale" with the DFT's frequency spacing (cycles/unit per
    # pixel) -- same role in the resolution math above as a real-space
    # pixel size, but the calibration unit is 1/um, not um.
    is_frequency_space = (metadata or {}).get("space") == "frequency"
    ij_metadata = {
        "axes": axes,
        "spacing": sz,
        "unit": "1/um" if is_frequency_space else "um",
    }
    frame_interval_s = _extract_frame_interval_seconds(
        metadata or {}, image.shape[0]
    )
    if frame_interval_s is not None and frame_interval_s > 0:
        # ImageJ convention for constant frame interval in TIFF metadata.
        ij_metadata["finterval"] = frame_interval_s
        ij_metadata["tunit"] = "sec"
        ij_metadata["fps"] = 1.0 / frame_interval_s

    tifffile.imwrite(
        filepath, image, imagej=True, resolution=(rx, ry), metadata=ij_metadata
    )


def save_imaris(
    filepath, data, metadata=None, resolution_levels=True, progress_cb=None
):
    """
    Save 5D data to Imaris .ims format.

    Args:
        filepath: Output path (.ims)
        data: 5D array-like in pyvistra (T, Z, C, Y, X) format
        metadata: dict with 'scale', 'channels', etc. Defaults provided if None.
        resolution_levels: Generate downsampled pyramid (default True)
        progress_cb: Optional callback(fraction)
    """
    from .readers.imaris_writer import save_imaris as _save_imaris

    if metadata is None:
        metadata = {"scale": (1.0, 1.0, 1.0), "filename": "Image"}
    _save_imaris(
        filepath,
        data,
        metadata,
        resolution_levels=resolution_levels,
        progress_cb=progress_cb,
    )


def _build_psf_from_pyvistra_meta(data, metadata):
    """Translate the pyvistra 5D + metadata convention into a deconlib ``Psf``.

    pyvistra dialogs hand savers ``data`` as 5D ``(T, Z, C, Y, X)`` and a
    metadata dict shaped like ``psf_dialog._build_metadata`` produces
    (``parameters.{wavelength|wavelength_em, na, ni, ns}``, ``spacing``).
    Squeezes T/C to yield the 3D PSF deconlib expects.
    """
    from deconlib import Optics, Psf

    arr = np.asarray(data[:] if hasattr(data, "__getitem__") else data)
    if arr.ndim == 5:
        if arr.shape[0] != 1 or arr.shape[2] != 1:
            raise ValueError(
                f"PSF must have singleton T and C axes, got shape {arr.shape}"
            )
        arr = arr[0, :, 0, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"PSF data must be 3D or 5D, got {arr.ndim}D")

    params = (metadata or {}).get("parameters", {})
    wavelength = params.get("wavelength") or params.get("wavelength_em")
    if wavelength is None:
        raise ValueError("PSF metadata missing wavelength")

    optics = Optics(
        wavelength=float(wavelength),
        na=float(params["na"]),
        ni=float(params["ni"]),
        ns=float(params.get("ns", params["ni"])),
    )
    spacing = tuple(float(s) for s in metadata["spacing"])
    return Psf(
        psf=arr.astype(np.float32, copy=False),
        optics=optics,
        pixel_size=spacing,
        source=metadata.get("psf_source", "theoretical"),
    )


def _load_psf_h5(filepath):
    """Load a ``.psf.h5`` artifact as a (Numpy5DProxy, metadata) pair."""
    from deconlib import load_psf as _decon_load_psf

    psf_obj = _decon_load_psf(filepath)
    arr5d = psf_obj.psf[np.newaxis, :, np.newaxis, :, :] if psf_obj.psf.ndim == 3 \
        else psf_obj.psf[np.newaxis, np.newaxis, np.newaxis, :, :]
    proxy = Numpy5DProxy(arr5d)
    meta = {
        "filename": os.path.basename(filepath),
        "shape": arr5d.shape,
        "is_rgb": False,
        "scale": tuple(psf_obj.pixel_size),
        "spacing": list(psf_obj.pixel_size),
        "parameters": {
            "wavelength": psf_obj.optics.wavelength,
            "na": psf_obj.optics.na,
            "ni": psf_obj.optics.ni,
            "ns": psf_obj.optics.ns,
        },
        "psf_source": psf_obj.source,
    }
    if psf_obj.pupil_ref:
        meta["pupil_ref"] = psf_obj.pupil_ref
    return proxy, meta


def _load_pupil_h5(filepath):
    """Load a ``.pupil.h5`` artifact for image-viewer display.

    Returns the complex pupil amplitude as a single-channel 5D float32
    image plus a metadata dict that retains the Pupil dataclass under
    ``"pupil"`` for downstream consumers.
    """
    from deconlib import load_pupil as _decon_load_pupil

    pupil_obj = _decon_load_pupil(filepath)
    amp = np.abs(pupil_obj.pupil).astype(np.float32)
    arr5d = amp[np.newaxis, np.newaxis, np.newaxis, :, :]
    proxy = Numpy5DProxy(arr5d)
    meta = {
        "filename": os.path.basename(filepath),
        "shape": arr5d.shape,
        "is_rgb": False,
        "scale": (1.0, 1.0, 1.0),
        "pupil": pupil_obj,
        "pupil_source": pupil_obj.source,
        "parameters": {
            "wavelength": pupil_obj.optics.wavelength,
            "na": pupil_obj.optics.na,
            "ni": pupil_obj.optics.ni,
            "ns": pupil_obj.optics.ns,
        },
    }
    return proxy, meta


# ---- Sparse Labels I/O ----


def load_sparse_labels(path):
    """
    Load SparseLabels from file.

    Args:
        path: Path to .sparse.zarr or .sparse.npz file

    Returns:
        SparseLabels instance
    """
    from .labels import SparseLabels

    return SparseLabels.from_file(path)


def save_sparse_labels(path, labels):
    """
    Save SparseLabels to file.

    Args:
        path: Output path (.sparse.zarr or .sparse.npz)
        labels: SparseLabels instance
    """
    labels.save(path)


# ---------------------------------------------------------------------------
# Output format registry
# ---------------------------------------------------------------------------
#
# Maps extension → (display label, saver). Savers must accept the uniform
# signature ``(filepath: str, data, metadata: dict) -> None`` where *data*
# is 5D ``(T, Z, C, Y, X)`` array-like. Extra parameters (e.g. ``scale``)
# are pulled from *metadata* inside the saver wrapper.
#
# ``ImageOutputSelector`` looks up labels and savers here, so adding a new
# output format is one ``register_output_format(...)`` call from anywhere.

_OUTPUT_FORMATS: "dict[str, tuple[str, callable]]" = {}


def register_output_format(extension, label, saver):
    """Register an output format for ``ImageOutputSelector``.

    Args:
        extension: File extension including leading dot (e.g. ``".tif"``,
            ``".psf.h5"``).
        label: Human-readable name shown in the UI (e.g. ``"TIFF"``).
        saver: Callable ``(filepath, data, metadata) -> None``. *data* is
            5D ``(T, Z, C, Y, X)``; *metadata* is a plain dict.
    """
    _OUTPUT_FORMATS[extension] = (label, saver)


def get_output_format(extension):
    """Look up a registered format. Returns ``(label, saver)`` or ``None``."""
    return _OUTPUT_FORMATS.get(extension)


def available_output_formats():
    """All registered formats as ``[(label, extension), ...]``."""
    return [(label, ext) for ext, (label, _) in _OUTPUT_FORMATS.items()]


def _save_tiff_default(filepath, data, metadata):
    scale = (metadata or {}).get("scale", (1.0, 1.0, 1.0))
    save_tiff(filepath, data, scale=scale, metadata=metadata)


def _save_imaris_default(filepath, data, metadata):
    save_imaris(filepath, data, metadata=metadata)


def _save_psf_h5_default(filepath, data, metadata):
    """Saver for ``.psf.h5`` — wraps deconlib's :func:`save_psf`.

    Translates pyvistra's 5D-array + dict-metadata convention into a
    :class:`deconlib.Psf` and writes via deconlib's HDF5 I/O.
    """
    from deconlib import save_psf as _decon_save_psf

    psf = _build_psf_from_pyvistra_meta(data, metadata)
    _decon_save_psf(filepath, psf, metadata=metadata)


def _save_pupil_h5_default(filepath, data, metadata):
    """Saver for ``.pupil.h5`` — expects a :class:`deconlib.Pupil` in metadata.

    The streaming/buffer-routing convention can't construct a Pupil from a
    5D array alone (it's complex and carries optics). Dialogs that want to
    save a pupil pass the ``Pupil`` instance under ``metadata["pupil"]``.
    """
    from deconlib import save_pupil as _decon_save_pupil

    pupil = (metadata or {}).get("pupil")
    if pupil is None:
        raise ValueError(
            "save .pupil.h5: metadata must include a 'pupil' key holding a "
            "deconlib.Pupil instance"
        )
    _decon_save_pupil(filepath, pupil, metadata=metadata)


register_output_format(".tif", "TIFF", _save_tiff_default)
register_output_format(".ims", "Imaris", _save_imaris_default)
register_output_format(".psf.h5", "PSF (HDF5)", _save_psf_h5_default)
register_output_format(".pupil.h5", "Pupil (HDF5)", _save_pupil_h5_default)
