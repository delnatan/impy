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
from pyvistra import imshow

data, meta = load_image("movie.czi")
viewer = imshow(data, meta)
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

## Workspace: tabs vs. floating windows

By default `imshow()` docks the viewer as a tab in a shared `Workspace`
window rather than opening an independent top-level window. The
workspace is a singleton created on first use, so calling `imshow()`
repeatedly in a notebook session — e.g. once per cell as you iterate —
adds a new tab to the same workspace instead of piling up separate
windows on screen:

```python
viewer1 = imshow(data1, meta1)   # creates the workspace, docks tab 1
viewer2 = imshow(data2, meta2)   # docks tab 2 in the same workspace
```

Either call still returns the underlying `ImageWindow`, so all the
usual scripting access (`viewer.layers`, `viewer.renderer`,
`viewer.img_data`, ...) works the same whether the window ended up
docked or floating.

Pass `floating=True` to opt out and get a standalone top-level window
instead (the pre-workspace behavior) — useful for e.g. comparing two
viewers on separate monitors:

```python
viewer = imshow(data, meta, floating=True)
```

Use `pyvistra.get_workspace()` to reach the shared workspace instance
directly (it's created lazily, so calling it before any `imshow()` call
still returns a valid, empty workspace).

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
