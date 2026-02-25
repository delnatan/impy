# CLAUDE.md - AI Assistant Guide for pyvistra

## Project Purpose

**pyvistra** provides a quick and painless way to view multi-dimensional microscopy images and work with ROIs (Regions of Interest). The library prioritizes simplicity and usability over feature bloat.

The key goal is to be able to quickly compose 'parts' — data proxies, computation buffers, viewers, and annotations — into analysis workflows. Heavy computation should benefit from live visual feedback via `ImageBuffer`.

## Design Principles

### Keep It Simple (KISS)
- Favor the simplest solution that works
- Avoid premature abstraction - three similar lines are better than a premature helper
- Don't add features beyond what was explicitly requested
- Delete unused code completely; no backwards-compatibility shims

### Separation of Concerns (Layered Architecture)
- **Data** (`data/`): Pure data models — proxies, buffers, shapes, points, tracks, labels, colors. No vispy, no Qt.
- **Layers** (`layers/`): Layer abstraction and command/undo system. Depends only on `data/`.
- **Visuals** (`visuals/`): Vispy renderers — image composite, shapes, points, tracks, labels, overlays. No Qt widgets.
- **UI** (`ui/`): Core viewer, toolbar, managers. Depends on everything above + Qt.
- **Viewers** (`viewers/`): Specialized multi-panel viewers (ortho, volume, tiled).
- **Widgets** (`widgets/`): Reusable Qt dialogs/widgets (no vispy).
- **Apps** (`apps/`): Standalone applications (gel analyzer, console) — not core.

Keep data structures separate from algorithms. A class should either hold data OR perform operations, rarely both.

### No matplotlib dependency
- Colormaps come from `colormaps.py` (vispy built-ins + user-registered custom colormaps)
- PNG/JPEG loading uses Pillow (`PIL`), not `matplotlib.image`
- Plotting widgets (line_profile, histogram) use pure Qt painting — follow this pattern

## Package Layout

```
pyvistra/
├── __init__.py              # Public API (thin re-exports)
├── colormaps.py             # Colormap registry (no matplotlib)
├── io.py                    # load_image, save_*, normalize_to_5d (thin dispatcher)
├── rois.py                  # ROI geometry + vispy visuals (legacy, being replaced by data/shapes.py)
├── theme.py                 # Qt theme
│
├── data/                    # Pure data models (no vispy, no Qt)
│   ├── proxies.py           # File5DProxy and all subclasses
│   ├── buffer.py            # ImageBuffer (zarr-backed async computation)
│   ├── shapes.py            # ShapeData + shape commands
│   ├── points.py            # PointTable
│   ├── tracks.py            # TrackTable
│   ├── labels.py            # SparseLabels
│   ├── colors.py            # Label color utilities (adjacency graph coloring)
│   ├── point_commands.py    # Mutable wrapper + commands for PointTable
│   └── track_commands.py    # Mutable wrapper + commands for TrackTable
│
├── layers/                  # Layer abstraction (glue between data and visuals)
│   ├── base.py              # Layer dataclass, LayerList container
│   └── commands.py          # Command protocol, UndoStack
│
├── visuals/                 # Vispy renderers (no Qt widgets)
│   ├── image.py             # CompositeImageVisual
│   ├── shapes.py            # ShapeLayerVisual
│   ├── points.py            # PointLayerVisual
│   ├── tracks.py            # TrackLayerVisual
│   ├── labels.py            # LabelOverlayVisual
│   └── overlays.py          # ScaleBar, Timestamp
│
├── readers/                 # File format readers
│   ├── imaris.py            # ImarisReader, ImarisWriter
│   └── czi.py               # CZIReader
│
├── ui/                      # Core viewer + interaction
│   ├── window.py            # ImageWindow (main viewer)
│   ├── toolbar.py           # Toolbar
│   ├── manager.py           # WindowManager singleton
│   ├── annotation_manager.py # AnnotationManager
│   ├── label_manager.py     # LabelManager
│   └── layer_manager.py     # Unified LayerManager (new)
│
├── viewers/                 # Specialized multi-panel viewers
│   ├── ortho.py             # OrthoViewer (3-panel orthogonal)
│   ├── volume.py            # VolumeViewer
│   └── tiled.py             # TiledViewer
│
├── widgets/                 # Reusable Qt dialogs/widgets
│   ├── line_profile.py      # Gold standard: pure Qt painting, no matplotlib
│   ├── histogram.py
│   ├── channel_panel.py
│   └── ...
│
└── apps/                    # Standalone applications (not core)
    ├── gel_analyzer.py
    └── console.py
```

### Dependency Rules
- `data/` → numpy, zarr only (no vispy, no Qt)
- `layers/` → `data/` only
- `visuals/` → `data/`, vispy only (no Qt widgets)
- `readers/` → external libs (h5py, aicspylibczi, tifffile)
- `ui/` → everything above + Qt
- `viewers/` → `data/`, `visuals/`, `ui/`, Qt
- `widgets/` → Qt only (no vispy)
- `apps/` → anything (but nothing imports from apps)

## Architecture

### Data Model: 5D Convention

All image data is normalized to `(T, Z, C, Y, X)`:
- **T**: Time points
- **Z**: Depth slices
- **C**: Channels
- **Y, X**: Spatial dimensions

This eliminates dimension confusion. Use `load_image()` which returns `(proxy, metadata)`.

### Lazy Loading with Proxies

Proxies behave like numpy arrays but load data on-demand. All live in `data/proxies.py` and share a common base class `File5DProxy`:
- `Imaris5DProxy` - HDF5-based .ims files
- `CZI5DProxy` - Zeiss .czi files
- `Numpy5DProxy` - In-memory arrays
- `Zarr5DProxy` - Zarr arrays (lazy)

```python
data, meta = load_image("large_file.ims")  # data is a proxy
slice_2d = data[0, 5, 1, :, :]  # loads only this slice from disk
```

### ImageBuffer — Async Computation Bridge

`ImageBuffer` (in `data/buffer.py`) is a zarr-backed 5D array that acts as a live write buffer for computation workers. Workers write slices in; `ImageWindow` reads and displays slices out. The canonical data flow:

```
Reader/Proxy → (compute in thread) → ImageBuffer → ImageWindow
```

`ImageBuffer` has the same read interface as proxies (`.shape`, `.__getitem__`) plus write support (`.__setitem__`). Pass an `ImageBuffer` to `imshow()` to watch computation in real time.

### Colormap System

`colormaps.py` provides a clean registry with no matplotlib dependency:

```python
import pyvistra.colormaps as cmap

# Built-in named colormaps (delegated to vispy):
cmap.get('viridis')   # returns vispy.color.Colormap
cmap.get('Green')     # simple black→#49FF49

# Register a custom colormap:
cmap.register('MyLUT', [(0,0,0,1), (0.5,0,1,1), (1,1,1,1)])
cmap.register('MyArray', my_256x4_lut_array)
```

### Layer System

The layer system (`layers/`) provides a unified way to manage all annotation types:

- **Layer** (`layers/base.py`): Dataclass bundling `name`, `layer_type`, `data`, `visual`, `undo_stack`
- **LayerList** (`layers/base.py`): Ordered named container with per-type active layer tracking
- **Command/UndoStack** (`layers/commands.py`): Command pattern for undoable operations. Each layer has its own undo stack.

Layer types: `"shapes"`, `"points"`, `"tracks"`, `"labels"`

Data models (in `data/`):
- **ShapeData** — rect/circle/line vectors with columnar `params` (N×8 float32)
- **PointTable** — point localizations (frozen dataclass, wrapped by `PointDataHolder` for mutability)
- **TrackTable** — trajectories (frozen dataclass, wrapped by `TrackDataHolder` for mutability)
- **SparseLabels** — pixel masks

Each `ImageWindow` owns a `LayerList` at `self.layers`. Methods like `add_shape_layer()`, `add_point_layer()`, etc. register in both the legacy dicts and the new LayerList during the migration period.

### Event System

**Key flows:**
1. **Mouse interaction**: Vispy canvas captures mouse → `ImageWindow` handlers → shape/ROI creation → signals
2. **Slider changes**: Qt widget signals → handler updates `CompositeImageVisual`
3. **Window lifecycle**: `WindowManager` emits signals → managers update their window lists
4. **Layer signals**: `layer_added`, `layer_removed`, `layer_data_changed`, `active_layer_changed`

### ROI System (Legacy)

The legacy ROI system in `rois.py` is being replaced by `data/shapes.py` + `visuals/shapes.py`. During migration, both coexist:
- Legacy: `rois.py` ROI subclasses with coupled geometry+visuals, `AnnotationManager`
- New: `ShapeData` (pure data) + `ShapeLayerVisual` (pure renderer) + `LayerManager`

## Development Guidelines

### Before Making Changes
1. Read the relevant module(s) first
2. Understand existing patterns before adding code
3. Check if the functionality already exists

### When Implementing
- Modify existing files; avoid creating new ones unless necessary
- Keep changes minimal and focused on the request
- Don't add error handling for impossible scenarios
- Don't add docstrings/comments to code you didn't change
- Trust internal code and framework guarantees
- **No matplotlib**: use Pillow for image loading, vispy/colormaps.py for colormaps, Qt painting for plots

### Widget/Plot Pattern (follow line_profile.py)
Widgets that draw plots should use pure Qt painter API (`QPainter`, `QPen`, `QPolygonF`) — not matplotlib. `line_profile.py` is the canonical example: multi-series overlay, axis labels, CSV export, all in ~900 lines with zero external plot dependencies.

### Coordinate Systems
- Vispy uses OpenGL coordinates (bottom-left origin)
- Images use numpy indexing (top-left origin)
- Use `_map_event_to_image()` for mouse-to-data coordinate conversion

### Layer Operations
Use the command pattern for undoable layer modifications:
```python
from pyvistra.data.shapes import AddShape, RECTANGLE
from pyvistra.layers.commands import UndoStack

layer = window.layers.active("shapes")
cmd = AddShape(RECTANGLE, params, t=0, z=0)
layer.undo_stack.push(cmd, layer.data)
# To undo: layer.undo_stack.undo(layer.data)
```

### ROI Manager Synchronization (Legacy)
After modifying `window.rois`, emit the appropriate signal:
```python
self.rois.append(new_roi)
self.roi_added.emit(new_roi)
```

## Common Patterns

### Adding a New Shape Type
1. Add the shape type constant in `data/shapes.py`
2. Implement `get_handles()` and `get_outline()` cases for the new type
3. Add draw/edit logic in `ui/window.py` mouse handlers
4. Register tool button in `ui/toolbar.py`

### Adding a New ROI Type (Legacy)
1. Create class in `rois.py` inheriting from `ROI`
2. Implement: `update()`, `hit_test()`, `move()`, `adjust()`, `to_dict()`, `_update_visuals_from_data()`
3. Register tool button in `ui/toolbar.py`

### Adding a Custom Colormap (programmatic)
```python
from pyvistra import colormaps
colormaps.register('MySingleColor', ['black', '#FF6600'])
colormaps.register('MyGradient', my_256x4_array)
```

### Adding File Format Support
1. Create reader class in `readers/` (follow `ImarisReader` pattern)
2. Add proxy class in `data/proxies.py` extending `File5DProxy` if lazy loading needed
3. Update `load_image()` in `io.py` with extension detection
4. Ensure output is 5D `(T, Z, C, Y, X)`

### Adding Analysis Functions
1. Create function in `analysis.py` with `@magicgui` decorator
2. Register in `AnnotationManager._setup_ui()`

### Adding a Standalone Application
Place in `apps/` — not in core. Apps can import from core freely; core must not import from apps.

## Critical Gotchas

| Issue | Solution |
|-------|----------|
| Wrong dimension order | Always use `(T, Z, C, Y, X)` after `load_image()` |
| Proxy slicing loads into RAM | Be aware when slicing large regions |
| Mouse coords don't match data | Use `_map_event_to_image()` |
| Annotation Manager out of sync | Emit signals after modifying `window.rois` (legacy) or use layer commands |
| Multiple Qt event loops | Only call `app.exec_()` once |
| HDF5 file handle leaks | Close reader when done |
| matplotlib import | Do not add matplotlib imports; use Pillow or Qt painting instead |

---

*Last Updated: 2026-02-24*
