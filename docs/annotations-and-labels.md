# Annotations and Labels

This section covers script-friendly annotation patterns. Task-specific workflows such as gel analysis are intentionally GUI-centric and described separately.

## Work with shape annotations

```python
from pyvistra.io import load_image
from pyvistra import imshow
from pyvistra.data.shapes import RECTANGLE, rectangle_bounds

data, meta = load_image("input.ims")
viewer = imshow(data, meta)

# After drawing shapes in the viewer, inspect them from Python.
shape_layers = viewer.layers.by_type("shapes")
print(sum(len(layer.data) for layer in shape_layers))
```

### Rectangle shape example

```python
cache = viewer.renderer.current_slice_cache  # (C, Y, X)
layer = viewer.layers.active("shapes")
sid = layer.data.shape_ids[0]
rec = layer.data.get(sid)

if rec.shape_type == RECTANGLE:
    x0, y0, x1, y1 = rectangle_bounds(rec, cache.shape[-2:])
    crop = cache[:, y0:y1, x0:x1]            # (C, h, w)
```

### Other shape types

```python
from pyvistra.data.shapes import CIRCLE, LINE, get_outline

for sid in layer.data.shape_ids:
    rec = layer.data.get(sid)
    if rec.shape_type in (CIRCLE, LINE):
        outline_xy = get_outline(rec)
        print(rec.label, outline_xy.shape)
```

## Sparse label files

Use these helpers for segmentation/annotation data persistence:

```python
from pyvistra.io import load_sparse_labels, save_sparse_labels

labels = load_sparse_labels("cells.sparse.zarr")
# ... modify labels ...
save_sparse_labels("cells_updated.sparse.zarr", labels)
```

Supported extensions:

- `.sparse.zarr`
- `.sparse.npz`

## Point annotations

Use `PointTable` to store per-point localization data (position + properties):

```python
from pyvistra.data.points import PointTable

points = PointTable.from_arrays(
    x=[32.4, 80.1],
    y=[19.2, 47.8],
    t=[0, 0],
    properties={
        "amplitude": [1250.0, 980.0],
        "background": [55.0, 42.0],
        "index": [1, 2],
    },
)

viewer.add_point_layer("Peaks", points=points)
```

Point tables can also be loaded/saved as CSV/JSON with
`load_points_csv`, `load_points_json`, `save_points_csv`, and
`save_points_json` from `pyvistra.data.points`.

### Point layer styling and metadata

- `size_mode="image"` makes boxes scale with zoom (data-space size).
- `size_mode="screen"` keeps boxes fixed in display pixels.
- Use the Annotation Manager inspector to change:
  - layer and selected-point colors
  - box size, edge width, cross visibility
  - canvas labels (`label_template`) vs tooltip-only mode
- Use **Promote to PointROI** for focused handle-based adjustment of a selected point.

## Practical reminder

When exporting image data that should stay spatially/temporally meaningful, reuse the original `meta` dict with `save_tiff(..., metadata=meta)`.
