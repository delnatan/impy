# Getting Started

## Install

From your environment:

```bash
pip install -e .
```

Optional extras for specific formats:

```bash
pip install -e .[nd2,czi]
```

## Run the app

```bash
pyvistra
# or
python -m pyvistra
```

## Library import patterns

```python
import pyvistra as pv

# IO
from pyvistra.io import load_image, save_tiff, save_imaris

# Viewer helpers
from pyvistra import imshow, run_app
```

## Data model reminder

Most loaders produce data as `(T, Z, C, Y, X)` plus metadata:

```python
data, meta = load_image("input.ims")
print(data.shape)
print(meta.keys())
```

Common metadata fields include:

- `scale`: `(z, y, x)` voxel/pixel size
- `channels`: channel descriptors (when available)
- `timestamps`: per-timepoint timestamps (when available)
- `timestamp_seconds`: relative timepoints (ND2)

## Build docs locally

```bash
mkdocs serve
```

Then open `http://127.0.0.1:8000`.
