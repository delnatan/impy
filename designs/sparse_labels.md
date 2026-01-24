# SparseLabels Design Document

A sparse coordinate-based representation for image segmentation masks, designed for cell biology and microscopy applications.

## Overview

SparseLabels stores segmentation masks as coordinate lists rather than dense arrays. Each labeled object is represented by the set of pixel coordinates it occupies, making storage efficient for typical microscopy data where objects occupy a small fraction of the image volume.

## Core Design Principles

1. **Dimension-agnostic**: Works identically for 2D and 3D data; coordinates are n-tuples
2. **Label-centric access**: Objects are accessed by integer label, not by spatial query
3. **Mutable**: Supports adding, removing, and editing objects
4. **Interoperable**: Converts to/from dense label arrays and standard file formats
5. **Visualization-ready**: Provides efficient iteration and coordinate access for rendering

## Data Structure

### In-Memory Representation

```
SparseLabels
├── shape: tuple[int, ...]        # (Y, X) or (Z, Y, X) - defines coordinate space
├── ndim: int                      # 2 or 3, derived from shape
├── objects: dict[int, ndarray]    # label -> coordinates array
│   │
│   └── coordinates array: ndarray of shape (ndim, n_pixels)
│       - Row 0: first axis coordinates (Y or Z)
│       - Row 1: second axis coordinates (X or Y)
│       - Row 2: third axis coordinates (X) [3D only]
│
└── metadata: dict[int, dict]      # label -> arbitrary metadata (optional)
```

### Coordinate Array Convention

Coordinates follow NumPy/scikit-image convention: `(row, col)` for 2D, `(plane, row, col)` for 3D. This matches the output of `np.where()` and allows direct use in array indexing:

```python
image[sparse_labels.coords(label)]  # works directly
```

### Label Semantics

- Labels are positive integers (1, 2, 3, ...)
- Label 0 is reserved for background (never stored)
- Labels need not be contiguous
- Labels are unique; one label maps to exactly one coordinate set

## API Specification

### Construction

```python
SparseLabels(shape: tuple[int, ...])
    # Create empty SparseLabels with given shape

SparseLabels.from_dense(label_image: ndarray) -> SparseLabels
    # Convert dense label array to sparse representation

SparseLabels.from_file(path: str | Path) -> SparseLabels
    # Load from file (.sparse.zarr or .sparse.npz)
```

### Properties

```python
.shape -> tuple[int, ...]    # Coordinate space dimensions
.ndim -> int                  # Number of dimensions (2 or 3)
.labels -> list[int]          # Sorted list of all labels
.n_objects -> int             # Number of labeled objects
```

### Object Access

```python
.coords(label: int) -> tuple[ndarray, ...]
    # Coordinate arrays for indexing: (rows, cols) or (planes, rows, cols)
    # Raises KeyError if label not found

.mask(label: int, out: ndarray = None) -> ndarray
    # Dense boolean mask for single object
    # If out provided, writes into it and returns it

.bounding_box(label: int) -> tuple[slice, ...]
    # Minimal bounding box as slice tuple for array indexing

.area(label: int) -> int
    # Number of pixels (shortcut for len(coords[0]))
```

### Mutation Operations

```python
.add(label: int, coords: tuple[ndarray, ...]) -> None
    # Add new object with given coordinates
    # Raises ValueError if label already exists

.remove(label: int) -> None
    # Remove object entirely
    # Raises KeyError if label not found

.update(label: int, coords: tuple[ndarray, ...]) -> None
    # Replace coordinates for existing label
    # Raises KeyError if label not found

.add_pixels(label: int, coords: tuple[ndarray, ...]) -> None
    # Add pixels to existing object (union)
    # Creates object if label doesn't exist

.remove_pixels(label: int, coords: tuple[ndarray, ...]) -> None
    # Remove pixels from existing object (difference)
    # Raises KeyError if label not found
    # Removes object entirely if no pixels remain

.relabel(old: int, new: int) -> None
    # Change label identifier
    # Raises KeyError if old not found, ValueError if new exists

.merge(labels: list[int], new_label: int = None) -> int
    # Combine multiple objects into one
    # Returns the label of merged object (min of inputs, or new_label if given)

.clear() -> None
    # Remove all objects
```

### Conversion

```python
.to_dense(dtype: type = np.uint16) -> ndarray
    # Convert to dense label array

.to_dense_region(label: int, padding: int = 0) -> tuple[ndarray, tuple[slice, ...]]
    # Dense mask for single object cropped to bounding box
    # Returns (local_mask, slices) where slices index into full image
```

### I/O

```python
.save(path: str | Path) -> None
    # Save to file; format inferred from extension
    # Supported: .sparse.zarr (recommended), .sparse.npz

.copy() -> SparseLabels
    # Deep copy
```

### Iteration

```python
.__iter__() -> Iterator[int]
    # Iterate over labels

.__len__() -> int
    # Number of objects (same as n_objects)

.__contains__(label: int) -> bool
    # Check if label exists

.items() -> Iterator[tuple[int, tuple[ndarray, ...]]]
    # Iterate over (label, coords) pairs
```

## File Format Specification

### Primary Format: Zarr Directory (`.sparse.zarr`)

```
example.sparse.zarr/
├── .zattrs                    # Root attributes (JSON)
│   {
│     "sparse_labels_version": "1.0",
│     "shape": [512, 512],
│     "ndim": 2
│   }
│
├── labels/                    # Group containing all objects
│   ├── .zattrs
│   │   {"labels": [1, 2, 5, 10]}   # List of all labels for fast enumeration
│   │
│   ├── 1/                     # Group for label 1
│   │   ├── .zattrs
│   │   │   {"area": 1523}     # Cached properties (optional)
│   │   └── coords             # Zarr array, shape (ndim, n_pixels), dtype int32
│   │
│   ├── 2/
│   │   └── coords
│   ...
│
└── metadata/                  # Optional user metadata group
    ├── 1/
    │   └── .zattrs            # Arbitrary JSON metadata per object
    ...
```

**Zarr array configuration:**
- `coords` arrays use Blosc compression (zstd, shuffle)
- Chunk size: single chunk per object (coordinates accessed together)
- Dtype: int32 (sufficient for images up to 2 billion pixels per axis)

### Alternative Format: NPZ (`.sparse.npz`)

For simpler use cases or environments without Zarr:

```python
{
    "shape": np.array([512, 512]),
    "coords_1": np.array([[y1, y2, ...], [x1, x2, ...]]),  # shape (ndim, n_pixels)
    "coords_2": ...,
    "coords_5": ...,
}
```

Labels extracted from key names via `coords_{label}` pattern.

### Format Selection Guidance

| Use Case | Recommended Format |
|----------|-------------------|
| Large datasets, many objects | `.sparse.zarr` |
| Cloud storage / remote access | `.sparse.zarr` |
| Simple scripts, small data | `.sparse.npz` |
| Archival / interchange | `.sparse.zarr` |

## Visualization Integration

### VisPy Rendering

SparseLabels provides coordinates suitable for direct use with VisPy visuals:

```python
# Point rendering (fastest for sparse objects)
for label in sparse_labels:
    coords = sparse_labels.coords(label)
    positions = np.column_stack([coords[1], coords[0]])  # (x, y) for VisPy
    # Add to Markers visual

# Polygon rendering (cleaner for dense objects)
# Convert to contours externally, SparseLabels provides the mask
mask, slices = sparse_labels.to_dense_region(label)
contours = skimage.measure.find_contours(mask, 0.5)
```

### Coordinate System Note

SparseLabels stores coordinates in array order (row, col) = (y, x). Display systems typically expect (x, y). The conversion is:

```python
# SparseLabels -> display coordinates
rows, cols = sparse_labels.coords(label)
display_xy = np.column_stack([cols, rows])  # swap order
```

## Usage Examples

### Basic workflow

```python
# Create from existing segmentation
labels = SparseLabels.from_dense(segmented_image)

# Access object pixels
values = image[labels.coords(10)]
mean_intensity = values.mean()

# Edit: remove small objects
for label in list(labels):
    if labels.area(label) < 100:
        labels.remove(label)

# Save
labels.save("cleaned_masks.sparse.zarr")
```

### Manual annotation workflow

```python
# Start empty
labels = SparseLabels(shape=image.shape)

# Add object from drawn coordinates
labels.add(1, (drawn_rows, drawn_cols))

# Refine: add pixels to existing object
labels.add_pixels(1, (new_rows, new_cols))

# Erase: remove pixels
labels.remove_pixels(1, (erased_rows, erased_cols))

# Merge two objects user marked as same cell
labels.merge([1, 2], new_label=1)
```

### Multi-channel intensity analysis

```python
labels = SparseLabels.from_file("segmentation.sparse.zarr")

results = []
for label in labels:
    coords = labels.coords(label)
    ch1 = channel1[coords]
    ch2 = channel2[coords]
    results.append({
        "label": label,
        "mean_ch1": ch1.mean(),
        "mean_ch2": ch2.mean(),
        "correlation": np.corrcoef(ch1, ch2)[0, 1]
    })
```

## Implementation Notes

### Coordinate Storage

Coordinates stored as `ndarray` of shape `(ndim, n_pixels)` rather than tuple of arrays for:
- Single allocation per object
- Efficient serialization
- Consistent memory layout

The `coords()` method returns a tuple view for NumPy indexing compatibility.

### Thread Safety

Not thread-safe. External synchronization required for concurrent mutation.

### Memory Considerations

Approximate memory per object: `ndim × n_pixels × 4 bytes` (int32)

For a 1000-pixel 2D object: ~8 KB
For a 10,000-voxel 3D object: ~120 KB

Dense equivalent for 1000 pixels in 2048×2048 uint16 image: 8 MB

## Version History

- **1.0**: Initial specification