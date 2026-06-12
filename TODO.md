# pyvistra cleanup & integration plan

Stages are ordered for safe, incremental progress. Each stage leaves the
codebase shippable. Don't skip ahead — the cleanup stages assume the
foundational abstractions land first.

---

## Stage 0 — Prep (no code changes) ✅

- [x] Grep the wider workspace for external consumers of the legacy public API
      — `deconlib` only uses `pyvistra.io` (`load_image`, `save_tiff`) and
      `pyvistra.ui` (`imshow`, `run_app`); no legacy ROI/AnnotationManager
      usage. `memsolve` does not import pyvistra. Safe to refactor.
- [x] `tests/test_analysis.py` confirmed dead (imports `impy`).
      `tests/test_channel_panel.py` and `tests/test_dims_normalization.py`
      are also dead (same reason). `tests/test_points.py` imports
      `AnnotationManager` — flagged for migration in Stage 4d.

---

## Stage 1 — Foundational abstractions

Small, independent. Each unblocks downstream work and can be done in one
sitting.

### 1a. `ImageBuffer.subscribe()` + listener wiring ✅
**Files:** `data/buffer.py`, `ui/window.py`, `widgets/processing_helper.py`,
`widgets/z_projection_dialog.py`, `widgets/processing_helper.md`

- [x] Added callback-based `subscribe(callback)` on `ImageBuffer`. Fires
      from `__setitem__` on the writer's thread; returns an unsubscribe
      function. Kept Qt-free per the `data/` rule.
- [x] `ImageWindow` declares an internal `_buffer_dirty` Qt signal,
      connected with `Qt.QueuedConnection` to marshal worker-thread
      notifications onto the GUI thread.
- [x] Helper `_key_touches_current_slice(key)` checks whether the write
      affects `(self.t_idx, self.z_idx)`; calls `update_view()` if so,
      else `canvas.update()`.
- [x] Subscribe in `__init__`, re-subscribe on `set_data()`, unsubscribe
      in `closeEvent()`.
- [x] Removed the `plane_done` signal and `refresh_output_view()` poke
      from `BufferProcessingRunner` and `ZProjectionWorker`. Workers
      now just write.

### 1b. `Readable5D` / `Writable5D` Protocols ✅
**Files:** new `data/protocols.py`, `data/__init__.py`,
`widgets/processing_helper.py`

- [x] Defined `Readable5D`, `Writable5D`, and `ObservableBuffer` as
      `@runtime_checkable typing.Protocol`s in `data/protocols.py`.
- [x] Re-exported from `data/__init__.py`.
- [x] Annotated `BufferProcessingRunner.prepare_output` return type as
      `Tuple[Readable5D, Writable5D]`.
- [x] Verified at runtime: `isinstance(buf, ObservableBuffer)` →
      True; `isinstance(numpy_proxy, Writable5D)` → False.

### 1c. Normalize `acquire` / `release` ✅
**Files:** `data/proxies.py`, `widgets/processing_helper.py`

- [x] `Numpy5DProxy` now extends `RefCountMixin` with a no-op `close()`.
      Every 5D proxy and `ImageBuffer` is uniformly refcounted.
- [x] Removed `hasattr(source, "acquire")` and `hasattr(..., "release")`
      checks in `BufferProcessingRunner`.

> Broader `hasattr(self.img_data, "release")` branches in `ui/window.py`
> and `viewers/{ortho,volume,tiled}.py` are kept because those code paths
> can still receive raw `np.ndarray` (via `imshow` of an unwrapped array
> that the caller normalises themselves). Revisit if/when `imshow()`
> guarantees wrapping.

---

## Stage 2 — I/O routing as a first-class pattern ✅

### 2a. Format registry ✅
**Files:** `io.py`, `widgets/output_selector.py`,
`widgets/z_projection_dialog.py`, `widgets/psf_dialog.py`

- [x] Added `register_output_format(ext, label, saver)`,
      `get_output_format(ext)`, and `available_output_formats()` in
      `io.py`. Savers share a uniform signature
      `(filepath, data, metadata) -> None`. Built-ins registered at
      module load: `.tif`, `.ims`, `.psf.zarr`.
- [x] `ImageOutputSelector` now accepts `formats=[".tif", ".ims"]`
      (extension-only list); labels and savers come from the registry.
      Legacy `(label, ext)` tuples still accepted but labels are
      ignored — registry is authoritative.
- [x] Replaced the hardcoded if/elif dispatch in `_send_to_file` with a
      registry lookup. Adding a new format anywhere (incl. deconlib) is
      one `register_output_format(...)` call.
- [x] Updated `ZProjectionDialog` and `PSFComputeDialog` to pass
      extension-only `formats` lists.

### 2b. Single entry point for "send result" ✅
**Files:** `widgets/output_selector.py`, `widgets/processing_helper.py`

- [x] Audited every `ImageBuffer(...)` and `ImageWindow(...)` construction
      site. All result routing flows through
      `ImageOutputSelector.send(data, metadata)`. The four `ImageBuffer(...)`
      constructions are legitimate: two helper functions in `io.py`
      (return-only, no UI wiring); `BufferProcessingRunner.prepare_output`
      (streaming pattern); `PSFComputeDialog` (synchronous one-shot
      pattern — fills then sends).
- [x] Documented both patterns in CLAUDE.md ("I/O Routing Contract"
      section): streaming dialogs use `BufferProcessingRunner`, synchronous
      one-shot dialogs call `output_selector.send(...)` directly. PSF
      computation is the canonical synchronous example.
- [x] Hardened docstrings on `BufferProcessingRunner` and
      `ImageOutputSelector` to state the contract.

---

## Stage V — Visual adjustment refactor ✅

Audit found four parallel copies of channel-display state
(`CompositeImageVisual`, `VolumeRenderer`, `TiledViewer`, `OrthoViewer`),
duplicated auto-contrast logic in three places, a `channel_colors` list
that's derived but stored, a `COLORMAPS` dict-shim, and two dialogs
(`ContrastDialog`, `ChannelPanel`) that overlap heavily. Goal: one
shared state container, one unified panel, no API duplication.

### V1 — Low-effort cleanup ✅

- [x] **V1a.** Deleted `COLORMAPS` dict shim in `visuals/image.py` and
      `visuals/__init__.py`. Replaced 4 consumers (`contrast.py`,
      `channel_panel.py`, `tiled.py`, `volume.py`) with
      `pyvistra.colormaps.names()`.
- [x] **V1b.** Subsumed by V2 — `channel_colors` is now a `@property`
      on each renderer that derives the swatch list from the current
      `ChannelDisplayList` colormaps. The mutation in `set_colormap`
      is gone.
- [x] **V1c.** Added `pyvistra/contrast.py` with
      `compute_percentile_clim(plane, pct_low, pct_high)`. Replaced 5
      copies (3 in widgets + 2 in `tiled.py`).
- [x] **V1d.** Removed the auto-contrast block from
      `CompositeImageVisual.update_slice`. Added
      `CompositeImageVisual.auto_contrast()` — called once from
      `ImageWindow.__init__` and after data-rebuild paths in
      `set_data` / load. Scrubbing time no longer silently mutates
      clim.

### V2 — Extract `ChannelDisplayState` ✅

- [x] **V2a.** New `data/channel_state.py` with
      `ChannelDisplayState` (frozen-ish dataclass: `clim`, `gamma`,
      `colormap_name`, `visible`; derived `display_color()`) and
      `ChannelDisplayList` (ordered list + `subscribe(callback)` that
      fires `(channel_idx, field)` on every mutation). Exported from
      `pyvistra.data`.
- [x] **V2b.** `CompositeImageVisual` now owns
      `self.display: ChannelDisplayList`. Its `set_*` / `get_*`
      methods are thin delegations to the list; an internal
      `_on_display_changed` subscriber mirrors changes into vispy
      layers. `channel_colors` is a derived property.
- [x] **V2c.** Same pattern in `VolumeRendererProxy` and
      `TiledVisualProxy`. `OrthoVisualProxy` exposes the primary
      view's `display` (broadcast keeps the other two views in sync).
- [x] **V2d.** `ChannelPanel` subscribes to `renderer.display` for
      per-field state changes and to `viewer.view_changed` for
      histogram refresh on slice navigation. The manual
      `if self.channel_panel.isVisible(): refresh_ui()` pokes in
      `ui/window.py`, `viewers/ortho.py`, and the contrast-dialog
      sync calls are all deleted. (Volume still pokes since it has
      no `view_changed` signal yet — fine for now.)

### V3 — Unify panels, drop `ContrastDialog` ✅

- [x] **V3a.** Folded `ContrastDialog` features into `ChannelPanel`:
      `+`/`-` percentile-tighten/loosen buttons live in the action
      row beside "Auto Contrast All". The single-channel big-histogram
      view was intentionally dropped (per-row compact histograms
      cover the case).
- [x] **V3b.** Deleted `pyvistra/widgets/contrast.py`. Removed
      `ContrastDialog` from `widgets/__init__.py`.
- [x] **V3c.** Removed every `show_contrast_dialog` and
      `contrast_dialog` reference. Menu actions in `ui/window.py`,
      `viewers/ortho.py`, `viewers/volume.py`, `viewers/tiled.py`
      now all point at "Channels && Contrast..." → `show_channel_panel`
      (or the equivalent panel-spawning method).
- [x] **V3d.** Deleted `TiledChannelRow`. `TiledChannelPanel` now
      uses the shared `ChannelRow` from `widgets/channel_panel.py`
      and subscribes to `TiledVisualProxy.display` for state-driven
      refresh. Same wiring pattern as `ChannelPanel` modulo aggregate
      histogram data and per-tile canvas updates.

---

## Stage 3 — Processor base class

Do this **after** Stage 1 lands and after the next processor (deconvolve
or similar) is in flight — premature otherwise. The Z projection dialog
alone isn't enough signal to factor out cleanly.

### 3a. `BufferWorker` base
**Files:** new `widgets/buffer_worker.py` (or fold into `processing_helper.py`)

- [ ] `BufferWorker(QObject)` owns the standard signals (`progress`,
      `finished`, `cancelled`, `error`) and a `cancel()` flag. (No
      `plane_done` after Stage 1a.)
- [ ] Subclasses override `run(source, buffer, params)` only.
- [ ] Port `ZProjectionWorker` as the first user.

### 3b. `BufferProcessorDialog` base (optional, evaluate after Stage 3a)
**Files:** new `widgets/processor_dialog.py`

- [ ] Common scaffolding: form area, `ImageOutputSelector`, progress bar,
      status label, Start/Cancel/Close buttons, runner lifecycle.
- [ ] Subclass provides params widget + worker factory + a
      `validate_params()` hook.
- [ ] Port Z projection. Only generalize *after* a second dialog is written
      using the pattern.

---

## Stage 4 — Deletions

Now the abstractions are in place; cut the dead weight.

### 4a. Drop matplotlib + `analysis.py` ✅
- [x] Confirmed `analysis.py` was only consumed by `AnnotationManager`.
- [x] Deleted `pyvistra/analysis.py`.
- [x] Deleted `tests/test_analysis.py`.
- [x] Stripped the "Analysis" menu items (Crop Image, Measure Intensity)
      and "Lanes" menu (Align Lanes) from `AnnotationManager` — they
      were the only consumers. Line Profile + Gel Analyzer entries kept.
- [x] Removed optional matplotlib import in `apps/console.py`.
- [x] Fixed stale "via matplotlib" docstring in `io.py`.

### 4b. Unify shape/ROI tool paths in `window.py` ✅
- [x] Removed the `coordinate` tool (toolbar action, dispatch branch,
      F-key flip handler). `CoordinateROI` class lingers in `rois.py`
      until 4d.
- [x] Toolbar tools `rect/circle/line` now drive the `ShapeData` /
      `AddShape` path. Deleted the legacy `RectangleROI`/`CircleROI`/
      `LineROI` interactive-creation branches in `on_mouse_press`.
      Dropped the dead `shape_rect/shape_circle/shape_line` tool names.
- [x] Added shape-layer selection + drag/resize in the pointer tool.
      New helpers `_hit_test_shape_layers`, `_select_shape`,
      `_clear_shape_selection`. Body and handle drags mutate params
      live and finalise as a single undoable command via the new
      `SetShapeParams` command + `UndoStack.push_executed`.
- [x] `self.rois` is now programmatic-only. Pointer tool hit-tests
      both shape layers and `self.rois`, so gel_analyzer's
      `LaneROI` instances still receive clicks (no shim needed).
- [x] Escape clears both legacy ROI selection and shape-layer selection.

### 4c. Migrate `apps/gel_analyzer.py` — PARKED (see Stage G)

Decision (2026-06-05): the shape system is intentionally flat / not
class-based, so the original "swap `LaneROI` for `ShapeData` rows" bullet
doesn't fit. Bands aren't shapes — they move with, scale with, and have
no meaning detached from their parent lane. Treating them as sibling
shapes with a parent-link drags cascade-edit/undo coordination into the
command system; encoding them into the shape primitive leaks domain
vocabulary into core. Right move is a dedicated `Lane` primitive (peer
of points/tracks/labels). Tracked separately in Stage G; this stage is
paused until Stage G lands.

- [ ] (deferred) Replace all `window.rois` reads/writes with the
      `lanes` layer API once Stage G is in.
- [ ] (deferred) Delete `apps/gel_rois.py` (just a re-export).

### 4d. Delete `rois.py` and `AnnotationManager` — blocked by Stage G

Annotation manager audit (2026-06-05) also flagged: shape layers added
via `add_shape_layer()` emit `layer_added` but the AnnotationManager
only listens to per-type `*_layer_added` signals, so shape layers are
invisible in the group tree. Compact-UI pass also discussed. Both are
deferred — no point polishing a widget that's slated for deletion. Pick
them up only if Stage G slips far.

- [ ] Delete `pyvistra/rois.py`.
- [ ] Delete `pyvistra/ui/annotation_manager.py`.
- [ ] Remove their exports from `pyvistra/__init__.py`.
- [ ] Remove the `Toolbar` backward-compat re-export at the bottom of
      `ui/window.py`.
- [ ] Remove the `labels` / `label_overlay` backward-compat properties on
      `ImageWindow` (lines ~287-295) if grep confirms no consumers.

### 4e. Optional consolidations
- [ ] Inspect `data/point_commands.py`, `data/track_commands.py`,
      `layers/commands.py`, plus the inline commands in `data/shapes.py`.
      If they share structure, collapse into one `layers/commands.py`. If
      they meaningfully diverge, leave alone.
- [ ] Evaluate whether `window.add_layer(type, data, name)` can replace
      `add_shape_layer` / `add_point_layer` / `add_track_layer` /
      `add_label_layer`. Only unify if implementations are near-identical;
      keep typed methods if visuals/panel wiring diverge per type.

---

## Stage 5 — deconlib & memsolve integration

By this point the contract is small:

- deconlib depends on `numpy + zarr` (no pyvistra import). Its public
  surface is `deconvolve(src: Readable5D, dst: Writable5D, psf, params)`.
- pyvistra wires it via a `BufferWorker` subclass + a processor dialog
  built on Stage 3.
- Live preview is automatic via Stage 1a (`data_changed` → window refresh).

- [ ] Add a `widgets/deconvolution_dialog.py` (or under `apps/` if it pulls
      heavy UI). Use the processor pattern from Stage 3.
- [ ] Add a `widgets/psf_dialog.py` upgrade (already exists) to wire PSF
      calculation / distillation / pupil retrieval from deconlib.
- [ ] Slot memsolve in as the optimizer backend once the basic Richardson-
      Lucy or similar path works end-to-end.

---

## Stage G — `Lane` primitive + gel_analyzer rewrite (PARKED)

Independent of Stages 3 / 5. Park until the deconvolution path
(Stage 3 + Stage 5) is end-to-end. Picked up afterwards to unblock 4c
and 4d.

Why a new primitive (not a `ShapeData` extension): the shape system is
flat columnar (no subclassing), and a band is meaningless detached
from its parent lane — bands aren't shapes, they're sub-records of a
lane. A `Lane` is a peer of `points` / `tracks` / `labels`.

### G1. `data/lanes.py` — core data model
- [ ] `Band` dataclass: `y_local`, `label`, `color`.
- [ ] `Lane` dataclass: rectangle bounds (`x1, y1, x2, y2`), `locked`,
      `show_marker_labels`, `label_side`, `label_offset`,
      `label_font_size`, ordered `bands: list[Band]`, `t`, `z`.
- [ ] `LaneTable` container (dict-of-id → `Lane`), `from_list` /
      `to_list` serialization, `hit_test` returning either
      `("band", lane_id, band_idx)`, a handle name, `("body", lane_id)`,
      or `None`.
- [ ] Commands: `AddLane`, `RemoveLane`, `SetLaneBounds` (drag/resize
      snapshot, mirrors `SetShapeParams`), `AddBand`, `RemoveBand`,
      `SetBandY`, `SetBandLabel`. All in `data/lanes.py` for now (fold
      into `layers/commands.py` if Stage 4e ends up unifying).

### G2. `visuals/lanes.py` — renderer
- [ ] `LaneLayerVisual`: rectangle outline + per-band line + per-band
      Text label. Subscribes to `LaneTable` via a callback (same shape
      as `ShapeLayerVisual.update`). Honors `show_marker_labels`,
      `label_side`, `label_font_size`, `label_offset`, `locked`
      (suppresses body-drag cursor).
- [ ] Hit-handle priorities match LaneROI today: band > handle > body.

### G3. Wiring into core
- [ ] Register `"lanes"` as a fourth `layer_type` in `layers/base.py`.
- [ ] `ImageWindow.add_lane_layer(name, lanes=None)` following the
      `add_point_layer` / `add_track_layer` pattern. Emit
      `lane_layer_added(name)` and the generic `layer_added(Layer)`.
- [ ] Pointer-tool drag of a lane body / handle / band uses the same
      live-mutation + push-executed-snapshot pattern as shapes.
- [ ] Optional: a `lane` toolbar tool that seeds a new `Lane` via
      `AddLane`. Alternatively keep "drag a rect, then convert" as a
      gel-analyzer-only action — decide during G4.

### G4. Reimplement `apps/gel_analyzer.py`
- [ ] Rewrite against `window.layers.active("lanes")`. No more
      `window.rois`, no more `LaneROI`/`RectangleROI` imports.
- [ ] Band detection writes `Band` rows via `AddBand`; manual band
      drags fire `SetBandY` and the analyzer recomputes MW from the
      `LaneTable` snapshot (no callback indirection needed — just
      subscribe to the `LaneTable`).
- [ ] Delete `apps/gel_rois.py`.

### G5. Unblock 4c/4d
- [ ] Mark Stage 4c done.
- [ ] Run Stage 4d (delete `rois.py` + `AnnotationManager`).

---

## Notes / conventions

- Every stage must leave imports clean: no `from .rois import …` after 4d,
  no `import matplotlib` after 4a.
- The `data/` dir stays Qt-free and vispy-free. Notifications from
  `ImageBuffer` use plain callbacks, not Qt signals.
- No new docs files. Update docstrings in the touched modules only.
