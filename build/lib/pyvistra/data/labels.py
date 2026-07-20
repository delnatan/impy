"""
SparseLabels - Sparse coordinate-based representation for image segmentation masks.

Stores segmentation masks as coordinate lists rather than dense arrays.
Each labeled object is represented by the set of pixel coordinates it occupies.
"""

from pathlib import Path
from typing import Iterator

import numpy as np


class SparseLabels:
    """
    Sparse coordinate-based representation for image segmentation masks.

    Stores coordinates in array order (row, col) = (y, x) for 2D,
    or (plane, row, col) = (z, y, x) for 3D.

    Example:
        >>> labels = SparseLabels(shape=(512, 512))
        >>> labels.add(1, (np.array([10, 11]), np.array([20, 21])))
        >>> image[labels.coords(1)]  # direct numpy indexing
    """

    def __init__(self, shape: tuple[int, ...]):
        """
        Create empty SparseLabels with given shape.

        Args:
            shape: Coordinate space dimensions (Y, X) or (Z, Y, X)
        """
        if len(shape) < 2 or len(shape) > 3:
            raise ValueError(f"shape must be 2D or 3D, got {len(shape)}D")

        self._shape = tuple(shape)
        self._objects: dict[int, np.ndarray] = {}  # label -> (ndim, n_pixels)
        self._metadata: dict[int, dict] = {}  # label -> arbitrary metadata

    @property
    def shape(self) -> tuple[int, ...]:
        """Coordinate space dimensions."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of dimensions (2 or 3)."""
        return len(self._shape)

    @property
    def labels(self) -> list[int]:
        """Sorted list of all labels."""
        return sorted(self._objects.keys())

    @property
    def n_objects(self) -> int:
        """Number of labeled objects."""
        return len(self._objects)

    # ---- Object Access ----

    def coords(self, label: int) -> tuple[np.ndarray, ...]:
        """
        Get coordinate arrays for indexing.

        Args:
            label: Object label

        Returns:
            Tuple of arrays (rows, cols) or (planes, rows, cols)
            for direct numpy indexing: image[labels.coords(label)]

        Raises:
            KeyError: If label not found
        """
        if label not in self._objects:
            raise KeyError(f"Label {label} not found")

        arr = self._objects[label]
        return tuple(arr[i] for i in range(arr.shape[0]))

    def mask(self, label: int, out: np.ndarray = None) -> np.ndarray:
        """
        Get dense boolean mask for single object.

        Args:
            label: Object label
            out: Optional output array to write into

        Returns:
            Boolean array of shape self.shape

        Raises:
            KeyError: If label not found
        """
        if label not in self._objects:
            raise KeyError(f"Label {label} not found")

        if out is None:
            out = np.zeros(self._shape, dtype=bool)
        else:
            out.fill(False)

        coords = self.coords(label)
        out[coords] = True
        return out

    def bounding_box(self, label: int) -> tuple[slice, ...]:
        """
        Get minimal bounding box as slice tuple.

        Args:
            label: Object label

        Returns:
            Tuple of slices for array indexing

        Raises:
            KeyError: If label not found
        """
        if label not in self._objects:
            raise KeyError(f"Label {label} not found")

        arr = self._objects[label]
        slices = []
        for i in range(arr.shape[0]):
            coords = arr[i]
            slices.append(slice(int(coords.min()), int(coords.max()) + 1))
        return tuple(slices)

    def area(self, label: int) -> int:
        """
        Get number of pixels for object.

        Args:
            label: Object label

        Returns:
            Pixel count

        Raises:
            KeyError: If label not found
        """
        if label not in self._objects:
            raise KeyError(f"Label {label} not found")
        return self._objects[label].shape[1]

    # ---- Mutation Operations ----

    def add(self, label: int, coords: tuple[np.ndarray, ...]) -> None:
        """
        Add new object with given coordinates.

        Args:
            label: Object label (must be positive integer)
            coords: Tuple of coordinate arrays (rows, cols) or (planes, rows, cols)

        Raises:
            ValueError: If label already exists or is <= 0
        """
        if label <= 0:
            raise ValueError(f"Label must be positive, got {label}")
        if label in self._objects:
            raise ValueError(f"Label {label} already exists")

        self._set_coords(label, coords)

    def remove(self, label: int) -> None:
        """
        Remove object entirely.

        Args:
            label: Object label

        Raises:
            KeyError: If label not found
        """
        if label not in self._objects:
            raise KeyError(f"Label {label} not found")
        del self._objects[label]
        self._metadata.pop(label, None)

    def update(self, label: int, coords: tuple[np.ndarray, ...]) -> None:
        """
        Replace coordinates for existing label.

        Args:
            label: Object label
            coords: New coordinate arrays

        Raises:
            KeyError: If label not found
        """
        if label not in self._objects:
            raise KeyError(f"Label {label} not found")
        self._set_coords(label, coords)

    def add_pixels(self, label: int, coords: tuple[np.ndarray, ...]) -> None:
        """
        Add pixels to existing object (union).

        Creates object if label doesn't exist.

        Args:
            label: Object label
            coords: Coordinate arrays to add
        """
        if label <= 0:
            raise ValueError(f"Label must be positive, got {label}")

        coords = self._validate_coords(coords)

        if label not in self._objects:
            self._objects[label] = coords
            return

        existing = self._objects[label]

        # Union: combine and remove duplicates
        combined = np.hstack([existing, coords])

        # Remove duplicates using structured array view
        if combined.shape[1] > 0:
            # Transpose to (n_pixels, ndim) for unique operation
            transposed = combined.T
            unique = np.unique(transposed, axis=0)
            self._objects[label] = unique.T.astype(np.int32)

    def remove_pixels(self, label: int, coords: tuple[np.ndarray, ...]) -> None:
        """
        Remove pixels from existing object (difference).

        Removes object entirely if no pixels remain.

        Args:
            label: Object label
            coords: Coordinate arrays to remove

        Raises:
            KeyError: If label not found
        """
        if label not in self._objects:
            raise KeyError(f"Label {label} not found")

        coords = self._validate_coords(coords)
        existing = self._objects[label]

        # Create set of coordinates to remove
        remove_set = set(zip(*[coords[i] for i in range(coords.shape[0])]))

        # Filter existing coordinates
        keep_mask = np.array(
            [
                tuple(existing[:, i]) not in remove_set
                for i in range(existing.shape[1])
            ]
        )

        if not np.any(keep_mask):
            # No pixels remain, remove object
            del self._objects[label]
            self._metadata.pop(label, None)
        else:
            self._objects[label] = existing[:, keep_mask]

    def relabel(self, old: int, new: int) -> None:
        """
        Change label identifier.

        Args:
            old: Current label
            new: New label (must be positive)

        Raises:
            KeyError: If old label not found
            ValueError: If new label already exists
        """
        if old not in self._objects:
            raise KeyError(f"Label {old} not found")
        if new <= 0:
            raise ValueError(f"New label must be positive, got {new}")
        if new in self._objects:
            raise ValueError(f"Label {new} already exists")

        self._objects[new] = self._objects.pop(old)
        if old in self._metadata:
            self._metadata[new] = self._metadata.pop(old)

    def merge(self, labels: list[int], new_label: int = None) -> int:
        """
        Combine multiple objects into one.

        Args:
            labels: List of labels to merge
            new_label: Optional label for merged object (default: min of inputs)

        Returns:
            Label of merged object

        Raises:
            KeyError: If any label not found
        """
        if not labels:
            raise ValueError("Must provide at least one label to merge")

        for label in labels:
            if label not in self._objects:
                raise KeyError(f"Label {label} not found")

        if new_label is None:
            new_label = min(labels)
        elif new_label <= 0:
            raise ValueError(f"New label must be positive, got {new_label}")

        # Collect all coordinates
        all_coords = [self._objects[label] for label in labels]
        combined = np.hstack(all_coords)

        # Remove duplicates
        if combined.shape[1] > 0:
            transposed = combined.T
            unique = np.unique(transposed, axis=0)
            combined = unique.T.astype(np.int32)

        # Remove old labels
        for label in labels:
            if label != new_label:
                del self._objects[label]
                self._metadata.pop(label, None)

        # Set merged coordinates
        self._objects[new_label] = combined

        return new_label

    def clear(self) -> None:
        """Remove all objects."""
        self._objects.clear()
        self._metadata.clear()

    # ---- Conversion ----

    def to_dense(self, dtype: type = np.uint16) -> np.ndarray:
        """
        Convert to dense label array.

        Args:
            dtype: Output dtype (default uint16)

        Returns:
            Dense label array of shape self.shape
        """
        out = np.zeros(self._shape, dtype=dtype)
        for label, arr in self._objects.items():
            coords = tuple(arr[i] for i in range(arr.shape[0]))
            out[coords] = label
        return out

    def to_dense_region(
        self, label: int, padding: int = 0
    ) -> tuple[np.ndarray, tuple[slice, ...]]:
        """
        Get dense mask for single object cropped to bounding box.

        Args:
            label: Object label
            padding: Pixels to add around bounding box

        Returns:
            (local_mask, slices) where slices index into full image
        """
        bbox = self.bounding_box(label)

        # Apply padding
        padded_slices = []
        for i, s in enumerate(bbox):
            start = max(0, s.start - padding)
            stop = min(self._shape[i], s.stop + padding)
            padded_slices.append(slice(start, stop))
        padded_slices = tuple(padded_slices)

        # Create local mask
        local_shape = tuple(s.stop - s.start for s in padded_slices)
        local_mask = np.zeros(local_shape, dtype=bool)

        # Map coordinates to local space
        coords = self.coords(label)
        local_coords = tuple(
            coords[i] - padded_slices[i].start for i in range(len(coords))
        )
        local_mask[local_coords] = True

        return local_mask, padded_slices

    def copy(self) -> "SparseLabels":
        """Create deep copy."""
        new = SparseLabels(self._shape)
        for label, arr in self._objects.items():
            new._objects[label] = arr.copy()
        for label, meta in self._metadata.items():
            new._metadata[label] = meta.copy()
        return new

    # ---- I/O ----

    def save(self, path: str | Path) -> None:
        """
        Save to file; format inferred from extension.

        Supported: .sparse.zarr (recommended), .sparse.npz

        Args:
            path: Output path
        """
        path = Path(path)
        path_str = str(path)

        if path_str.endswith(".sparse.zarr"):
            self._save_zarr(path)
        elif path_str.endswith(".sparse.npz"):
            self._save_npz(path)
        else:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                "Use .sparse.zarr or .sparse.npz"
            )

    def _save_zarr(self, path: Path) -> None:
        """Save to zarr format."""
        import shutil

        import zarr

        # Remove existing directory
        if path.exists():
            shutil.rmtree(path)

        # Create root group
        root = zarr.open_group(str(path), mode="w")

        # Root attributes
        root.attrs["sparse_labels_version"] = "1.0"
        root.attrs["shape"] = list(self._shape)
        root.attrs["ndim"] = self.ndim

        # Labels group
        labels_group = root.create_group("labels")
        labels_group.attrs["labels"] = self.labels

        # Save each label
        for label, arr in self._objects.items():
            label_group = labels_group.create_group(str(label))
            label_group.attrs["area"] = arr.shape[1]

            # Store coordinate array
            label_group.create_array(
                "coords",
                data=arr.astype(np.int32),
            )

            # Store metadata if present
            if label in self._metadata:
                label_group.attrs.update(self._metadata[label])

    def _save_npz(self, path: Path) -> None:
        """Save to npz format."""
        data = {"shape": np.array(self._shape)}

        for label, arr in self._objects.items():
            data[f"coords_{label}"] = arr

        np.savez_compressed(str(path), **data)

    @classmethod
    def from_file(cls, path: str | Path) -> "SparseLabels":
        """
        Load from file.

        Args:
            path: Path to .sparse.zarr or .sparse.npz file

        Returns:
            Loaded SparseLabels instance
        """
        path = Path(path)
        path_str = str(path)

        if path_str.endswith(".sparse.zarr"):
            return cls._load_zarr(path)
        elif path_str.endswith(".sparse.npz"):
            return cls._load_npz(path)
        else:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                "Use .sparse.zarr or .sparse.npz"
            )

    @classmethod
    def _load_zarr(cls, path: Path) -> "SparseLabels":
        """Load from zarr format."""
        import zarr

        root = zarr.open_group(str(path), mode="r")

        shape = tuple(root.attrs["shape"])
        instance = cls(shape)

        labels_group = root["labels"]
        label_list = labels_group.attrs.get("labels", [])

        for label in label_list:
            label_group = labels_group[str(label)]
            coords = np.asarray(label_group["coords"])
            instance._objects[label] = coords

            # Load metadata (excluding area which is computed)
            for key, value in label_group.attrs.items():
                if key != "area":
                    if label not in instance._metadata:
                        instance._metadata[label] = {}
                    instance._metadata[label][key] = value

        return instance

    @classmethod
    def _load_npz(cls, path: Path) -> "SparseLabels":
        """Load from npz format."""
        data = np.load(str(path))

        shape = tuple(data["shape"])
        instance = cls(shape)

        # Extract labels from key names
        for key in data.files:
            if key.startswith("coords_"):
                label = int(key[7:])  # Remove "coords_" prefix
                instance._objects[label] = data[key]

        return instance

    @classmethod
    def from_dense(cls, label_image: np.ndarray) -> "SparseLabels":
        """
        Convert dense label array to sparse representation.

        Args:
            label_image: Dense label array where 0 is background

        Returns:
            SparseLabels instance
        """
        instance = cls(label_image.shape)

        # Find unique labels (excluding 0)
        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels > 0]

        for label in unique_labels:
            coords = np.where(label_image == label)
            arr = np.vstack(coords).astype(np.int32)
            instance._objects[int(label)] = arr

        return instance

    # ---- Iteration ----

    def __iter__(self) -> Iterator[int]:
        """Iterate over labels."""
        return iter(sorted(self._objects.keys()))

    def __len__(self) -> int:
        """Number of objects (same as n_objects)."""
        return len(self._objects)

    def __contains__(self, label: int) -> bool:
        """Check if label exists."""
        return label in self._objects

    def items(self) -> Iterator[tuple[int, tuple[np.ndarray, ...]]]:
        """Iterate over (label, coords) pairs."""
        for label in sorted(self._objects.keys()):
            yield label, self.coords(label)

    # ---- Internal Helpers ----

    def _validate_coords(self, coords: tuple[np.ndarray, ...]) -> np.ndarray:
        """Validate and convert coordinates to internal format."""
        if isinstance(coords, np.ndarray) and coords.ndim == 2:
            # Already in internal format
            if coords.shape[0] != self.ndim:
                raise ValueError(
                    f"Expected {self.ndim} coordinate dimensions, "
                    f"got {coords.shape[0]}"
                )
            return coords.astype(np.int32)

        if len(coords) != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} coordinate arrays, got {len(coords)}"
            )

        # Stack into (ndim, n_pixels) format
        arr = np.vstack([np.asarray(c).ravel() for c in coords]).astype(np.int32)
        return arr

    def _set_coords(self, label: int, coords: tuple[np.ndarray, ...]) -> None:
        """Set coordinates for a label with validation."""
        arr = self._validate_coords(coords)

        # Bounds checking
        for i in range(arr.shape[0]):
            if arr.shape[1] > 0:
                if arr[i].min() < 0 or arr[i].max() >= self._shape[i]:
                    raise ValueError(
                        f"Coordinates out of bounds for dimension {i}"
                    )

        self._objects[label] = arr

    def __repr__(self) -> str:
        return (
            f"SparseLabels(shape={self._shape}, "
            f"n_objects={self.n_objects}, "
            f"labels={self.labels[:5]}{'...' if len(self.labels) > 5 else ''})"
        )

    # ---- Patch Extraction ----

    def extract_patches(
        self,
        size: int | tuple[int, ...],
        image: np.ndarray = None,
        labels: list[int] = None,
    ) -> Iterator[dict]:
        """
        Extract fixed-size patches centered on each object.

        Args:
            size: Output patch size (single int for square/cube, or tuple)
            image: Optional source image to crop (must match self.shape)
            labels: Subset of labels to extract (default: all)

        Yields:
            dict with keys:
                - 'label': int - object label
                - 'mask': np.ndarray - boolean mask (size x size)
                - 'image': np.ndarray - cropped image (only if image provided)
                - 'center': tuple - center coords in original image
                - 'valid': bool - True if object fits within patch size
        """
        # Normalize size to tuple
        if isinstance(size, int):
            patch_size = (size,) * self.ndim
        else:
            patch_size = tuple(size)
            if len(patch_size) != self.ndim:
                raise ValueError(
                    f"size must have {self.ndim} dimensions, got {len(patch_size)}"
                )

        # Validate image shape if provided
        if image is not None and image.shape != self._shape:
            raise ValueError(
                f"image shape {image.shape} must match labels shape {self._shape}"
            )

        # Determine which labels to process
        if labels is None:
            labels_to_process = self.labels
        else:
            labels_to_process = [l for l in labels if l in self._objects]

        for label in labels_to_process:
            yield self._extract_single_patch(label, patch_size, image)

    def _extract_single_patch(
        self,
        label: int,
        patch_size: tuple[int, ...],
        image: np.ndarray = None,
    ) -> dict:
        """Extract a single patch for one label."""
        coords = self.coords(label)
        arr = self._objects[label]

        # Compute centroid
        center = tuple(int(round(arr[i].mean())) for i in range(self.ndim))

        # Check if object fits within patch size
        bbox = self.bounding_box(label)
        obj_size = tuple(s.stop - s.start for s in bbox)
        valid = all(o <= p for o, p in zip(obj_size, patch_size))

        # Calculate ideal slice (centered on centroid)
        half_size = tuple(p // 2 for p in patch_size)

        # Source slice (what we read from original image)
        # Destination slice (where we write in output patch)
        src_slices = []
        dst_slices = []

        for dim in range(self.ndim):
            # Ideal start/end for centered patch
            ideal_start = center[dim] - half_size[dim]
            ideal_end = ideal_start + patch_size[dim]

            # Clip to image bounds
            src_start = max(0, ideal_start)
            src_end = min(self._shape[dim], ideal_end)

            # Compute offset in destination
            dst_start = src_start - ideal_start
            dst_end = dst_start + (src_end - src_start)

            src_slices.append(slice(src_start, src_end))
            dst_slices.append(slice(dst_start, dst_end))

        src_slices = tuple(src_slices)
        dst_slices = tuple(dst_slices)

        # Create output mask (zero-padded)
        out_mask = np.zeros(patch_size, dtype=bool)

        # Map coordinates to patch space
        patch_coords = tuple(
            coords[dim] - src_slices[dim].start + dst_slices[dim].start
            for dim in range(self.ndim)
        )

        # Filter coordinates that fall within patch bounds
        in_bounds = np.ones(arr.shape[1], dtype=bool)
        for dim in range(self.ndim):
            in_bounds &= (patch_coords[dim] >= 0) & (
                patch_coords[dim] < patch_size[dim]
            )

        valid_patch_coords = tuple(pc[in_bounds] for pc in patch_coords)
        out_mask[valid_patch_coords] = True

        result = {
            "label": label,
            "mask": out_mask,
            "center": center,
            "valid": valid,
        }

        # Extract image patch if provided
        if image is not None:
            out_image = np.zeros(patch_size, dtype=image.dtype)
            out_image[dst_slices] = image[src_slices]
            result["image"] = out_image

        return result
