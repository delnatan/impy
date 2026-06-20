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

`ImageBuffer` has the same read interface as proxies (`.shape`, `.__getitem__`) plus write support (`.__setitem__`).

**Change notifications.** `ImageBuffer.subscribe(callback)` registers a `callback(key)` invoked on every `__setitem__`, where `key` is the slicing tuple as passed to the writer. Callbacks fire on the writer's thread; Qt consumers must marshal to the GUI thread (the buffer module itself stays Qt-free per the `data/` dependency rule).

`ImageWindow` does this marshalling internally:

1. When given an `ImageBuffer`, it calls `subscribe()` and stores the unsubscribe handle.
2. The callback emits a queued internal Qt signal (`_buffer_dirty`).
3. The handler checks whether the write touches the currently displayed `(t, z)` and either calls `update_view()` or just `canvas.update()`.
4. On `set_data()` it re-subscribes; on `closeEvent()` it unsubscribes.

This means processing workers do **not** need per-slice signals like the old `plane_done` — they just write into the buffer:

```python
def run(self):
    for t in range(T):
        result = heavy_computation(self._source[t, ...])
        self._buffer[t, ...] = result   # window refreshes automatically
        self.progress.emit(t + 1, T)
    self.finished.emit()
```

Pass an `ImageBuffer` to `imshow()` to watch computation in real time.

### 5D Data Contracts (Protocols)

`data/protocols.py` defines structural-typing contracts that all 5D data sources satisfy:

- **`Readable5D`** — `.shape`, `.dtype`, `.ndim`, `__getitem__`. Implemented by every proxy and by `ImageBuffer`.
- **`Writable5D`** — adds `__setitem__`. Implemented by `ImageBuffer`.
- **`ObservableBuffer`** — adds `subscribe(callback) -> unsubscribe`. Implemented by `ImageBuffer`.

These are `typing.Protocol` classes (runtime-checkable, no inheritance required). They exist so external libraries (e.g. `deconlib`, `memsolve`) can type-annotate against pyvistra without importing concrete classes:

```python
def deconvolve(src: Readable5D, dst: Writable5D, psf, params): ...
```

Worker signatures and `BufferProcessingRunner.prepare_output()` use these annotations as the canonical contract.

### Refcounting (`acquire` / `release`)

Every 5D proxy and `ImageBuffer` extends `RefCountMixin` (`data/proxies.py`). The contract:

- `acquire()` increments the count and returns `self` (for chaining: `buf.acquire()`).
- `release()` decrements and calls `close()` at zero.
- `close()` does whatever cleanup is needed (close file handle, delete temp dir, no-op for in-memory arrays).

Consumers that hold a reference across thread boundaries or hand-offs (e.g. `BufferProcessingRunner` passing data to a worker and a viewer) call `acquire()` to bump the count and `release()` when done. There is no `hasattr(x, "acquire")` check anywhere — every 5D source has it.

### Colormap System

`colormaps.py` provides a clean registry with no matplotlib dependency:

```python
import pyvistra.colormaps as cmap

# Built-in named colormaps (delegated to vispy):
cmap.get('viridis')   # returns (vispy.color.Colormap, display_color_or_None)
cmap.get('Green')     # simple black→#49FF49

cmap.names()          # list of every registered name (built-in + custom)

# Register a custom colormap:
cmap.register('MyLUT', [(0,0,0,1), (0.5,0,1,1), (1,1,1,1)])
cmap.register('MyArray', my_256x4_lut_array)
```

### Per-Channel Display State

Every multi-channel renderer (`CompositeImageVisual`, `VolumeRendererProxy`, `TiledVisualProxy`, `OrthoVisualProxy` via its primary view) owns a `ChannelDisplayList` at `renderer.display` (defined in `data/channel_state.py`). Each entry is a `ChannelDisplayState` dataclass with `clim`, `gamma`, `colormap_name`, `visible` — plus a derived `display_color()` that consults the colormap registry.

- **Mutations**: `display.set_clim(c, lo, hi)`, `set_gamma`, `set_colormap_name`, `set_visible`.
- **Notifications**: `display.subscribe(callback)` registers `callback(channel_idx, field)` fired on every mutation. Returns an unsubscribe function.
- **Renderer mirror**: each renderer subscribes to its own `display` to push state changes into the underlying vispy layers (image/volume). UI widgets don't touch vispy.

This is the same pattern as `ImageBuffer.subscribe()` — pure-data list, callback-based notifications, no Qt in `data/`.

The renderer methods `set_clim`/`set_gamma`/`set_colormap`/`set_channel_visible` (and getters) are thin delegations to `self.display`. Code that needs the swatch color list uses the derived `renderer.channel_colors` property.

### Channels & Contrast Panel

One unified `ChannelPanel` (in `widgets/channel_panel.py`) handles all per-channel visual adjustment: colormap, contrast (min/max + compact histogram), gamma, visibility. Header buttons drive panel-wide actions ("Auto Contrast All", `+`/`-` percentile tighten/loosen — those use `pyvistra.contrast.compute_percentile_clim`).

The panel:
1. Writes user input through `renderer.display.set_*` (the renderer mirrors to vispy).
2. Subscribes to `renderer.display` to update rows when state changes from any source.
3. Subscribes to `viewer.view_changed` (when available) to refresh histograms on slice navigation.

There is no separate "ContrastDialog" — it was folded in. The panel is the single entry point for all visual adjustments and is wired to the "Channels && Contrast..." menu action (Shift+C) on every viewer.

The shared `ChannelRow` widget is the unit of UI per channel and is reused by `TiledChannelPanel` for the tiled viewer's global controls.

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
- Legacy: `rois.py` ROI subclasses with coupled geometry+visuals
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
5. For *write* support: define `save_myformat(filepath, data, metadata)` and register it once at module load:
   ```python
   from pyvistra.io import register_output_format
   register_output_format(".myext", "MyFormat", save_myformat)
   ```
   Every dialog that lists `".myext"` in its `ImageOutputSelector(formats=...)` picks it up automatically. No widget code edits required.

### I/O Routing Contract

All processor results — whether streamed, one-shot, or saved — flow through `ImageOutputSelector.send(data, metadata)`. The selector dispatches to one of three destinations:

- **Existing window**: `window.set_data(data, metadata)`.
- **New window**: constructs and shows an `ImageWindow`.
- **File**: looks up the saver in the format registry (`pyvistra.io.register_output_format`) and writes it.

This means dialogs do **not** decide where their output goes — the user does, via the selector combo. Adding new formats is a one-line `register_output_format(...)` call from anywhere (including downstream libraries like deconlib).

Two upstream paths reach the selector:

- **Streaming** (long-running, threaded): use `BufferProcessingRunner` — it owns the worker thread, source/buffer refcounts, and routes via `output_selector.send(buffer.acquire(), metadata)` automatically.
- **Synchronous** (one-shot, GUI-thread compute): construct an `ImageBuffer`, fill it, and call `output_selector.send(buffer, metadata)` directly. PSF computation (`widgets/psf_dialog.py`) is the canonical example.

### Adding a Streaming Image Processor
Follow the pattern in `widgets/z_projection_dialog.py` + `widgets/processing_helper.py`:

1. Subclass `QObject` for the worker. Required signals: `progress(done, total)`, `finished()`, `cancelled()`, `error(str)`. Implement `run(self)` and `cancel(self)`. **Do not emit per-slice signals** — write into the output `ImageBuffer` and the destination window refreshes automatically via `ImageBuffer.subscribe()`.
2. Type-annotate `run` as `(source: Readable5D, buffer: Writable5D, params)`.
3. In the dialog, instantiate `ImageOutputSelector` + `BufferProcessingRunner`; call `runner.prepare_output(...)` to get `(source, buffer)`, then `runner.start_worker(...)`.
4. The same dialog works for "reuse window / new window / save to file" — that's the `ImageOutputSelector` contract.

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

*Last Updated: 2026-06-04 (Stage V — visual adjustment refactor)*
