# pyvistra

Image analysis and ROI management tool.

## Installation

To install the package for local development (editable mode), use the `-e` (editable) flag. This allows you to modify the code and see changes immediately without reinstalling.

### Using pip

```bash
pip install -e .
```

### Using uv

```bash
uv pip install -e .
```

## Usage

Run the application:

```bash
pyvistra
# or
python -m pyvistra
```

## Documentation (MkDocs)

Project documentation is in the `docs/` folder with configuration in `mkdocs.yml`.

```bash
mkdocs serve
```

Then open `http://127.0.0.1:8000`.

## Library Usage

You can use `pyvistra` as a library to read and process images programmatically.

### Reading Imaris Files

You can use the `ImarisReader` class to read `.ims` files directly:

```python
from pyvistra.imaris_reader import ImarisReader

# Open the file
reader = ImarisReader('path/to/file.ims')

# Print metadata
print(f"Shape: {reader.shape}")  # (Time, Channels, Z, Y, X)
print(f"Channels: {reader.channels_info}")
print(f"Voxel Size: {reader.voxel_size}")

# Read specific data
# Read timepoint 0, channel 1, z-slice 5
plane = reader.read(c=1, t=0, z=5)

# Read full volume for timepoint 0, channel 0
volume = reader.read(c=0, t=0, z=None)
```

### Loading a 3D Timelapse

To easily load a 3D timelapse, you can use the `load_image` function. The returned proxy object behaves like a 5D numpy array.
Supported inputs include `.ims`, `.czi`, `.nd2`, TIFF, and standard image formats.

```python
import numpy as np
from pyvistra.io import load_image

# Load image using the high-level loader
# Returns a proxy object that behaves like a numpy array
data, meta = load_image('path/to/file.ims')

# The proxy shape is standardized to (Time, Z, Channel, Y, X)
print(f"Data shape: {data.shape}")

# Get dimensions
n_timepoints = data.shape[0]
n_channels = data.shape[2]
height = data.shape[3]
width = data.shape[4]

# Read specific data
# Read timepoint 0, full Z-stack, channel 0
volume = data[0, :, 0]
print(f"Volume shape: {volume.shape}") # (Z, Y, X)
```

### Saving Images

You can save 5D arrays (or crops) to TIFF format using `save_tiff`. This preserves voxel size and dimension order metadata compatible with ImageJ.

```python
from pyvistra.io import save_tiff

# Save the volume we just read
# meta['scale'] contains the voxel size (z, y, x)
save_tiff("output.tif", volume, scale=meta['scale'])
```

### Interactive IPython Usage

`pyvistra` can be used interactively in IPython or Jupyter notebooks. First, enable Qt event loop integration:

```python
# In IPython, enable Qt event loop integration
%gui qt

# Now you can use imshow() interactively
import numpy as np
from pyvistra.ui import imshow

# Create some test data
data = np.random.rand(20, 256, 256)  # (Z, Y, X)

# Display image - returns immediately, window is interactive
viewer = imshow(data, title="Random Data", dims="zyx")

# You can continue working in IPython while the viewer is open
# Access the viewer's data
print(viewer.img_data.shape)  # (1, 20, 1, 256, 256) - normalized to 5D

# Load and display a file
from pyvistra.io import load_image
data, meta = load_image('path/to/file.ims')
viewer2 = imshow(data[0, :, 0], title="Channel 0", dims="zyx")
```

If you're running from a regular Python script (not IPython), you need to start the Qt event loop manually:

```python
from pyvistra.ui import imshow, run_app
import numpy as np

data = np.random.rand(10, 100, 100)
viewer = imshow(data, dims="zyx")

# Start the event loop (blocks until all windows are closed)
run_app()
```

### Working with ROIs

ROIs (Regions of Interest) can be drawn interactively and used to extract data from images.

#### Extracting Region Data

Each ROI type has methods to extract the corresponding region from image data:

```python
# Get the current displayed slice from the viewer
cache = viewer.renderer.current_slice_cache  # Shape: (C, Y, X)

# Rectangle ROI - extract rectangular region
rect_roi = viewer.rois[0]  # Assuming first ROI is a RectangleROI
cropped = rect_roi.get_region(cache)  # Shape: (C, height, width)

# Circle ROI - extract circular region with mask
circle_roi = viewer.rois[1]  # Assuming second ROI is a CircleROI
region, mask = circle_roi.get_region(cache)
# region: bounding box array (C, height, width)
# mask: boolean array (height, width) - True inside circle

# Get mean intensity inside circle for each channel
mean_per_channel = [region[c][mask].mean() for c in range(region.shape[0])]

# Line ROI - extract intensity profile along line
line_roi = viewer.rois[2]  # Assuming third ROI is a LineROI
profile = line_roi.get_profile(cache)  # Shape: (C, num_points)
```

### Image Transforms

You can apply rotation and translation to images. The Transform dialog (`Image > Transform...` or `Shift+T`) provides visual preview with GPU-accelerated rendering.

#### Applying Transforms Permanently

To bake the transform into the image data (WYSIWYG - what you see is what you get):

1. Open the Transform dialog
2. Adjust rotation and translation visually
3. Click "Apply Transform" to permanently apply the transform to the data

This creates a transformed copy in memory. After applying, ROI region extraction will match what you see on screen.

#### Programmatic Transform

```python
from pyvistra.io import apply_transform, load_image

# Load image
data, meta = load_image('input.ims')

# Apply 45-degree rotation and 10px translation
buffer = apply_transform(
    source=data,
    rotation_deg=45.0,      # Positive = counter-clockwise
    translate=(10.0, 5.0),  # (tx, ty) in pixels
    metadata=meta,
)

# The buffer behaves like a numpy array
print(buffer.shape)  # Same as input

# Save to TIFF
buffer.save_as('rotated_output.tif')

# Or display in a new viewer
from pyvistra.ui import imshow
viewer = imshow(buffer, title="Rotated")
```

### ImageBuffer for Large Images

For processing large images that don't fit in memory, use `ImageBuffer` which streams data to disk using Zarr:

```python
from pyvistra.io import ImageBuffer
import numpy as np

# Create a buffer for a large 5D dataset
buffer = ImageBuffer(
    shape=(10, 50, 3, 2048, 2048),  # (T, Z, C, Y, X)
    dtype=np.uint16,
    chunks=(1, 16, 3, 512, 512),   # Optional: custom chunk size
)

# Write data slice by slice (streaming)
for t in range(10):
    for z in range(50):
        # Process/acquire your data
        slice_data = np.random.randint(0, 65535, (3, 2048, 2048), dtype=np.uint16)
        buffer[t, z, :, :, :] = slice_data

# Read back (lazy loading)
volume = buffer[0, :, 0, :, :]  # Get Z-stack for t=0, c=0

# Export to TIFF when done
buffer.save_as('large_output.tif')

# Clean up (deletes temporary files)
buffer.close()
```

Buffers are stored in `~/.pyvistra/buffers/` and automatically cleaned up when closed or garbage collected.
