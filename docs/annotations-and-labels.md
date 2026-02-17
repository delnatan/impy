# Annotations and Labels

This section covers script-friendly annotation patterns. Task-specific workflows such as gel analysis are intentionally GUI-centric and described separately.

## Open the annotation manager in interactive sessions

```python
from pyvistra.io import load_image
from pyvistra import imshow, show_annotation_manager

data, meta = load_image("input.ims")
viewer = imshow(data, meta)
annotation_mgr = show_annotation_manager(viewer)

# You can also open it from the window instance:
# annotation_mgr = viewer.show_annotation_manager()

# After drawing ROIs in the viewer, inspect them from Python
rois = viewer.rois
print(len(rois))
```

### Rectangle ROI example

```python
cache = viewer.renderer.current_slice_cache  # (C, Y, X)
rect = viewer.rois[0]
crop = rect.get_region(cache)                # (C, h, w)
```

### Circle ROI example

```python
circle = viewer.rois[1]
region, mask = circle.get_region(cache)
mean_per_channel = [region[c][mask].mean() for c in range(region.shape[0])]
```

### Line ROI example

```python
line = viewer.rois[2]
profile = line.get_profile(cache)            # (C, n_points)
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

## Practical reminder

When exporting image data that should stay spatially/temporally meaningful, reuse the original `meta` dict with `save_tiff(..., metadata=meta)`.
