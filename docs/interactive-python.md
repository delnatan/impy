# Interactive Python

This is the primary workflow when you want scripted control without relying on toolbar actions.

## IPython / Jupyter setup

Enable Qt event handling first:

```python
%gui qt
```

Then:

```python
from pyvistra.io import load_image
from pyvistra import imshow, show_annotation_manager

data, meta = load_image("movie.czi")
viewer = imshow(data, meta)
annotation_mgr = show_annotation_manager(viewer)
```

## Script setup

In plain Python scripts, start the Qt event loop manually:

```python
from pyvistra import imshow, run_app
from pyvistra.io import load_image

data, meta = load_image("movie.ims")
viewer = imshow(data, meta)
run_app()
```

## Accessing data consistently

`data` behaves like a 5D array in `(T, Z, C, Y, X)` order.

```python
# One 3D channel volume at t=0
vol = data[0, :, 0]     # (Z, Y, X)

# One XY plane at t=0, z=5, channel=1
plane = data[0, 5, 1]   # (Y, X)

# All channels in a plane
plane_all_c = data[0, 5, :]  # (C, Y, X)
```

## Showing numpy arrays with explicit dimensions

Use `dims` when starting from raw arrays:

```python
import numpy as np
from pyvistra import imshow

arr = np.random.rand(10, 256, 256)  # (Z, Y, X)
viewer = imshow(arr, title="Demo", dims="zyx")
```

## Colormaps

```python
viewer = imshow(data, meta, colormap={0: "Green", 1: "Magenta"})
```

## Useful pattern: keep metadata attached

When you process data and write output, keep `meta` around so scale/time metadata can be reused during export.
