# pyvistra

`pyvistra` is an image visualization and analysis package for microscopy data. It supports both GUI workflows and interactive Python usage.

This documentation focuses on using `pyvistra` as a Python library to:

- load data from multiple formats into a consistent 5D view
- inspect and visualize data from IPython/Jupyter or scripts
- convert between formats while keeping useful metadata
- work with ROI outputs and sparse label files

Use this docs set as a quick reminder when you are working outside toolbar-driven workflows.

## Core idea

Most IO routes normalize image data to a common shape:

`(T, Z, C, Y, X)`

That makes format-to-format conversion and scripted processing predictable.

## Where to start

1. [Getting Started](getting-started.md)
2. [Interactive Python](interactive-python.md)
3. [Image IO and Conversion](io-and-conversion.md)
4. [Annotations and Labels](annotations-and-labels.md)

## Quick example

```python
from pyvistra.io import load_image, save_tiff
from pyvistra import imshow

# Load ND2/CZI/IMS/TIFF/...
data, meta = load_image("sample.nd2")

# Show interactively
viewer = imshow(data, meta)

# Convert and preserve spacing + timing metadata when possible
save_tiff("sample_export.tif", data, scale=meta.get("scale", (1, 1, 1)), metadata=meta)
```
