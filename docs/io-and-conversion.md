# Image IO and Conversion

## Supported input formats

`load_image()` supports:

- `.ims` (Imaris)
- `.czi` (Zeiss CZI)
- `.nd2` (Nikon ND2)
- `.tif`, `.tiff`
- `.png`, `.jpg`, `.jpeg`
- `.zarr`
- `.psf.zarr`

All are normalized to `(T, Z, C, Y, X)`.

## Basic conversion

```python
from pyvistra.io import load_image, save_tiff

data, meta = load_image("input.nd2")
save_tiff("output.tif", data, scale=meta.get("scale", (1, 1, 1)), metadata=meta)
```

## TIFF metadata behavior

`save_tiff()` writes ImageJ-compatible metadata:

- `spacing` for Z spacing
- XY resolution tags from `scale`
- time interval metadata when available in `metadata`

Time metadata written for ImageJ includes:

- `finterval`
- `tunit` (`sec`)
- `fps`

`save_tiff()` can infer frame interval from metadata keys such as:

- `frame_interval_s`
- `frame_interval_seconds`
- `timestamp_seconds`
- `timestamps`

If frame timing is irregular, a representative interval (median delta) is written because standard ImageJ TIFF metadata stores a single interval.

## Loading TIFF timing metadata

`load_image("*.tif")` parses ImageJ timing metadata and exposes:

- `meta["frame_interval_s"]`

## Save Imaris format

```python
from pyvistra.io import load_image, save_imaris

data, meta = load_image("input.czi")
save_imaris("output.ims", data, metadata=meta)
```

## Normalize arbitrary numpy input

```python
import numpy as np
from pyvistra.io import normalize_to_5d, save_tiff

arr = np.random.randint(0, 4096, size=(20, 512, 512), dtype=np.uint16)  # (Z,Y,X)
proxy = normalize_to_5d(arr, dims="zyx")
save_tiff("zstack.tif", proxy, scale=(1.0, 0.2, 0.2))
```

## Large intermediate results with `ImageBuffer`

```python
import numpy as np
from pyvistra.io import ImageBuffer

buf = ImageBuffer(shape=(4, 30, 2, 1024, 1024), dtype=np.uint16)

for t in range(4):
    for z in range(30):
        buf[t, z, :, :, :] = np.random.randint(0, 65535, (2, 1024, 1024), dtype=np.uint16)

buf.save_as("buffer_export.tif")
buf.close()
```
