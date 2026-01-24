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

## pyvistra Integration

### Architecture

SparseLabels integrates into pyvistra as a separate visualization layer, distinct from ROIs:

```
New Files:
  pyvistra/labels.py         # SparseLabels data structure (this spec)
  pyvistra/label_visual.py   # LabelOverlayVisual for GPU rendering
  pyvistra/label_manager.py  # LabelManager Qt widget

Modified Files:
  pyvistra/ui.py             # Painting tools, mouse handlers, toolbar
  pyvistra/io.py             # Sparse label I/O functions
```

**Why separate from ROIs?**
- ROIs: click-drag to define control points (corners, endpoints)
- Masks: paint strokes to accumulate pixels
- Different interaction models warrant dedicated systems

### LabelOverlayVisual

Renders SparseLabels as a semi-transparent RGBA overlay:

```python
class LabelOverlayVisual:
    def __init__(self, view, shape_yx: tuple[int, int])

    def set_labels(labels: SparseLabels)
    def update_slice(z_idx: int)  # For 3D: render current Z slice
    def refresh()

    def set_label_color(label: int, rgba: tuple)
    def set_label_visible(label: int, visible: bool)
    def set_opacity(alpha: float)  # Global overlay opacity
```

Rendering details:
- RGBA texture updated when labels change
- Alpha blending: `(src_alpha, one_minus_src_alpha)`
- Render order: above image channels, below ROIs
- Per-label color assignment with automatic palette

### Drawing Tools

#### Brush Tool

Standard brush for freehand painting:

```python
# State in ImageWindow
self._active_label: int = 1      # Label being painted
self._brush_size: int = 5        # Brush radius in pixels
self._preserve_labels: bool = True  # Protect existing labels (default: on)

# Mouse handling
on_mouse_press:  Start stroke, record start position
on_mouse_move:   Accumulate brush disk coordinates
on_mouse_release: Finalize stroke, check for closed contour
```

Brush disk generation:
```python
def _get_brush_coords(cx: int, cy: int, radius: int) -> tuple[ndarray, ndarray]:
    """Generate circular brush mask centered at (cx, cy)."""
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = xx**2 + yy**2 <= radius**2
    ys, xs = np.where(mask)
    return (ys + cy - radius, xs + cx - radius)
```

#### Eraser Tool

Removes pixels from the active label. Same interaction as brush, but calls `remove_pixels()` instead of `add_pixels()`.

#### Closed Contour Auto-Fill

**Problem**: Drawing many objects with brush is tedious—user must carefully fill interiors.

**Solution**: When a brush stroke forms a closed contour (end point near start point), automatically fill the enclosed region.

Detection and filling:
```python
def _finish_stroke(self):
    """Finalize stroke; auto-fill if contour is closed."""
    if len(self._stroke_points) < 10:
        return  # Too short to be a contour

    start = self._stroke_points[0]
    end = self._stroke_points[-1]
    distance = np.linalg.norm(np.array(end) - np.array(start))

    # Threshold: close if within 2x brush radius
    if distance < self._brush_size * 2:
        self._fill_closed_contour()

def _fill_closed_contour(self):
    """Fill interior of closed stroke path."""
    from skimage.draw import polygon_fill

    # Extract contour coordinates
    ys = [p[0] for p in self._stroke_points]
    xs = [p[1] for p in self._stroke_points]

    # Get all interior pixels
    rr, cc = polygon(ys, xs, shape=self.labels.shape[-2:])
    fill_coords = (rr, cc)  # 2D case

    if self._preserve_labels:
        fill_coords = self._filter_existing_labels(fill_coords)

    self.labels.add_pixels(self._active_label, fill_coords)
```

Visual feedback:
- Stroke path shown as thin line while drawing
- When stroke closes (end near start), path changes color to indicate fill will occur
- On release, filled region appears immediately

#### Label Preservation Mode

**Problem**: Accidentally overwriting existing labels when painting near boundaries.

**Solution**: By default, painting only affects background pixels (label 0). Existing labels are protected.

```python
def _filter_existing_labels(self, coords: tuple[ndarray, ...]) -> tuple[ndarray, ...]:
    """Remove coordinates that belong to any existing label."""
    # Create occupancy mask from all existing labels
    occupied = np.zeros(self.labels.shape[-2:], dtype=bool)
    for label in self.labels:
        label_coords = self.labels.coords(label)
        if self.labels.ndim == 3:
            # Filter to current Z slice
            z_mask = label_coords[0] == self.z_idx
            occupied[label_coords[1][z_mask], label_coords[2][z_mask]] = True
        else:
            occupied[label_coords[0], label_coords[1]] = True

    # Filter out occupied pixels
    if self.labels.ndim == 3:
        zs, ys, xs = coords
        free = ~occupied[ys, xs]
        return (zs[free], ys[free], xs[free])
    else:
        ys, xs = coords
        free = ~occupied[ys, xs]
        return (ys[free], xs[free])
```

Toggle in UI:
- Checkbox in Label Manager: "Preserve existing labels" (default: checked)
- Keyboard shortcut: `P` to toggle
- When disabled, painting can overwrite other labels (useful for corrections)

### LabelManager Widget

Qt widget for label management, following ROI Manager pattern:

```
+----------------------------------+
| Window: [dropdown v]             |
+----------------------------------+
| Labels:                          |
| [*] 1 Cell A         [color] [x] |
| [ ] 2 Cell B         [color] [x] |
| [*] 3 Nucleus        [color] [x] |
+----------------------------------+
| Active: [1 v]   Brush: [10] px   |
| [x] Preserve existing labels     |
+----------------------------------+
| [New] [Merge] [Delete] [Rename]  |
| [Save] [Load]                    |
+----------------------------------+
```

Features:
- Window selector (same pattern as ROI Manager)
- Label list with visibility checkboxes and color swatches
- Active label selector for painting
- Brush size spinner
- Label preservation toggle
- Action buttons: New, Merge selected, Delete, Rename
- Save/Load with file dialog

### Toolbar Integration

Add to existing toolbar (after ROI tools):

```
[Pointer] [Coord] [Rect] [Circle] [Line] | [Brush] [Eraser] | [ROI Mgr] [Label Mgr]
```

Keyboard shortcuts:
- `B`: Brush tool
- `E`: Eraser tool
- `1-9`: Select label 1-9
- `N`: New label (next available ID)
- `P`: Toggle preserve labels
- `[` / `]`: Decrease/increase brush size

### 3D Annotation Workflow

For 3D data (Z, Y, X):
- Paint on current Z slice only (consistent with Z slider)
- Label Manager shows combined label list across all Z
- Navigation: Z slider or scroll wheel to move between slices
- Each label can span multiple Z slices

```python
# Painting in 3D
def _paint_stroke_3d(self, x: int, y: int):
    brush_yx = self._get_brush_coords(x, y, self._brush_size)
    z_coords = np.full(len(brush_yx[0]), self.z_idx, dtype=np.int32)
    coords_3d = (z_coords, brush_yx[0], brush_yx[1])

    if self._preserve_labels:
        coords_3d = self._filter_existing_labels(coords_3d)

    self.labels.add_pixels(self._active_label, coords_3d)
```

### Scriptable API

```python
from pyvistra import imshow, load_image
from pyvistra.labels import SparseLabels

# Display image
data, meta = load_image("cells.tif")
viewer = imshow(data, meta)

# Create or load labels
labels = SparseLabels(shape=(512, 512))
# or: labels = SparseLabels.from_file("cells.sparse.zarr")
# or: labels = SparseLabels.from_dense(segmentation_result)

viewer.set_labels(labels)

# Programmatic analysis
for label in labels:
    coords = labels.coords(label)
    intensities = data[0, 0, 0][coords]  # T=0, Z=0, C=0
    print(f"Label {label}: area={labels.area(label)}, mean={intensities.mean():.1f}")

# Save progress
labels.save("annotated.sparse.zarr")
```

### Neural Network Integration

SparseLabels designed for easy integration with segmentation algorithms:

```python
# From any segmentation that produces dense label array
from cellpose import models

model = models.Cellpose(model_type='cyto2')
masks, flows, styles, diams = model.eval(image)

# Convert to SparseLabels
labels = SparseLabels.from_dense(masks)
viewer.set_labels(labels)

# User can now refine with brush/eraser
# Save when done
labels.save("refined_segmentation.sparse.zarr")
```

## Version History

- **1.0**: Initial specification
- **1.1**: Added pyvistra integration with drawing tools, closed contour auto-fill, and label preservation mode