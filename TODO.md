# pyvistra cleanup & integration plan

Stages are ordered for safe, incremental progress. Each stage leaves the
codebase shippable.

---

## Completed (historical, condensed)

Full rationale for these lives in git history / commit messages; this is
just a "what shipped" index so old design debates aren't re-litigated.

- **Stage 0 — Prep.** Confirmed `deconlib`/`memsolve` have no dependency on
  pyvistra's legacy ROI/AnnotationManager API, so refactoring it is safe.
  Identified `tests/test_analysis.py`, `tests/test_channel_panel.py`,
  `tests/test_dims_normalization.py` as dead (imported `impy`) — deleted.
- **Stage 1 — Foundational abstractions.** `ImageBuffer.subscribe()` +
  Qt-thread marshalling (replaced the old `plane_done` signal). `Readable5D`
  / `Writable5D` / `ObservableBuffer` protocols in `data/protocols.py`.
  Uniform `acquire()`/`release()` refcounting across every 5D proxy and
  `ImageBuffer` (`RefCountMixin`).
- **Stage 2 — I/O routing.** `register_output_format()` / format registry
  in `io.py` replacing hardcoded if/elif dispatch. Single entry point:
  every processor result flows through `ImageOutputSelector.send(...)`
  (see CLAUDE.md "I/O Routing Contract").
- **Stage V — Visual adjustment refactor.** Unified per-channel display
  state into `ChannelDisplayList`/`ChannelDisplayState`
  (`data/channel_state.py`), owned by every renderer
  (`CompositeImageVisual`, `VolumeRendererProxy`, and the shared
  `MultiViewChannelProxy` used by `OrthoViewer`/`ZMontageViewer`, plus
  `TiledVisualProxy`). Deleted the `COLORMAPS` shim and `ContrastDialog`;
  folded everything into one `ChannelPanel`.
- **Stage 4a — Drop matplotlib.** Deleted `analysis.py` (dead,
  `AnnotationManager`-only) and its test. Removed the matplotlib import
  from `apps/console.py` and, later, the leftover dependency declaration
  in `requirements.txt`/`pyproject.toml` (caught by a graphify audit —
  no import remained, just the unused declaration).
- **Stage 4b — Unify shape/ROI tool paths.** Toolbar `rect`/`circle`/`line`
  now drive `ShapeData`/`AddShape` exclusively; deleted the legacy
  interactive-creation branches in `on_mouse_press`. Added shape-layer
  selection + drag/resize commands (`SetShapeParams`,
  `UndoStack.push_executed`). Removed the `coordinate` toolbar tool and,
  later, its orphaned `coordinate.svg` icon + README entry.

---

## Stage 3 — Processor base class (superseded by convention)

Never built as a formal base class. Three worker classes now exist
(`ZProjectionWorker`, `NLCGDeconvolutionWorker`, `PSFDistillationWorker`)
and each independently follows the same shape — `progress`/`finished`/
`cancelled`/`error` signals, `run()`/`cancel()` — documented as a
convention in CLAUDE.md ("Adding a Streaming Image Processor") rather than
enforced by inheritance. That's held up fine across three processors; a
shared `BufferWorker` base is still on the table if a fourth processor
makes the duplication actually annoying, but nothing about pyvistra
currently needs it.

---

## Stage 4 (cont'd) — Deletions

### 4c. Migrate `apps/gel_analyzer.py` ✅

Landed differently than originally planned. The plan called for a
dedicated `Lane`/`Band` primitive (see the old Stage G, retired below);
what actually shipped is lighter: `gel_analyzer.py` defines `ShapeLane` /
`ShapeLaneRef`, a thin wrapper that tags existing `ShapeData` rectangles
as gel lanes (`mark_as_lane()`) instead of introducing a new data model.
`gel_analyzer.py` no longer imports `rois.py` or `LaneROI` at all.

### 4d. Delete `rois.py` and `AnnotationManager` — partially done, now unblocked

- [x] `AnnotationManager` / `pyvistra/ui/annotation_manager.py` — already
      deleted, zero references remain anywhere.
- [ ] `pyvistra/rois.py` is **not** fully dead, so this isn't a straight
      delete. Confirmed by grep (no constructor calls anywhere) that
      `CircleROI`, `LineROI`, `CoordinateROI`, `LaneROI` are dead code.
      `ROI`, `RectangleROI`, `PointROI` are still load-bearing:
      `ImageWindow._focused_point_roi` (window.py:2175) constructs a
      `PointROI` to reuse `RectangleROI`'s handle/drag machinery for the
      focused-point-edit feature.
- [ ] `window.self.rois` is dead: initialized to `[]` in `__init__` and
      never appended to anywhere in the codebase (gel_analyzer stopped
      populating it once it moved to `ShapeLane`). The `for roi in
      self.rois` loops in `window.py` and `widgets/line_profile.py` are
      therefore no-ops today.
- [ ] `apps/gel_rois.py` (the `LaneROI` re-export shim) is unused —
      nothing imports it anymore.
- [ ] Remaining cleanup, once someone wants to spend the time: delete the
      four dead ROI subclasses + `apps/gel_rois.py` + the dead
      `self.rois` plumbing (list, append/remove sites, the
      `line_profile.py` branch that iterates it) while keeping
      `ROI`/`RectangleROI`/`PointROI` for focused-point editing. Update
      `pyvistra/__init__.py` exports accordingly. Not done yet — flagged
      here so it isn't lost, not blocking anything.
- [ ] Remove the `Toolbar` backward-compat re-export at the bottom of
      `ui/window.py` if grep confirms no consumers.
- [ ] Remove the `labels` / `label_overlay` backward-compat properties on
      `ImageWindow` if grep confirms no consumers.

### 4e. Optional consolidations
- [ ] Inspect `data/point_commands.py`, `data/track_commands.py`,
      `layers/commands.py`, plus the inline commands in `data/shapes.py`.
      If they share structure, collapse into one `layers/commands.py`. If
      they meaningfully diverge, leave alone.
- [ ] Evaluate whether `window.add_layer(type, data, name)` can replace
      `add_shape_layer` / `add_point_layer` / `add_track_layer` /
      `add_label_layer`. Only unify if implementations are near-identical;
      keep typed methods if visuals/panel wiring diverge per type.
- [ ] **Finish the LayerList migration**: `window.py` still keeps 81
      references to the legacy `_mask_layers` / `_track_layers` /
      `_point_layers` dicts, mirrored into `self.layers` (LayerList).
      `Layer` already carries `visible` + `style`, so this is pure
      deduplication — but the references run through the uncovered
      mouse-interaction paths (brush/contour painting, point drag,
      track ops). Do it in a dedicated session with manual verification
      of those flows; all *external* consumers already read
      `window.layers`.

---

## Stage 5 — deconlib integration ✅

Done: `widgets/deconvolution_dialog.py` (NLCG deconvolution) and
`widgets/psf_distillation_dialog.py` + `psf_distillation_nlcg.py` +
`psf_distillation_worker.py` (PSF distillation) both wire directly into
`deconlib.deconvolution.nlcg_*`, follow the processor-convention pattern,
and are live in `ImageWindow`'s Image menu. `memsolve` was never slotted
in as a separate optimizer backend — deconlib's own NLCG solver has been
sufficient. Revisit only if a concrete need for it shows up.

---

## Stage 6 — Image math & FFT inspection ✅

Two small, independent Image-menu dialogs, both one-shot GUI-thread
compute (build an `ImageBuffer`, call `output_selector.send(...)` directly
— no streaming worker, per CLAUDE.md "I/O Routing Contract").

- **`widgets/image_math_dialog.py`** (`ImageMathDialog`): add/subtract/
  multiply/divide against a constant or a second open window (picked via
  a combo populated from `manager.get_all()`, excluding self — same
  pattern as `DeconvolutionDialog._refresh_psf_combo()`). Only Y/X must
  match between the two windows; T/Z/C broadcast via ordinary numpy rules.
- **`widgets/fft_dialog.py`** (`FFTDialog`): 2D (per-Z-slice) or 3D
  (full-volume) mode; real-input (`rfftn`) or full-complex (`fftn`)
  transform. Result is always `fftshift`ed so DC sits at the array center
  — the real-transform's compact last axis (no negative-frequency half)
  is correctly excluded from the shift. Displayed as log-magnitude,
  magnitude, or phase (vispy can't render complex data directly).

Both reuse existing conventions only — `numpy.fft`, no matplotlib,
existing window-picker and I/O-routing patterns.

---

## Stage 7 — Performance & workspace overhaul ✅ (2026-07-03)

Landed in one pass, in priority order from the graphify audit:

1. **GPU-side clim** — `texture_format="auto"` on `CompositeImageVisual`
   images + `Volume` visuals; clim/gamma are now shader uniforms
   (float32 scrub 6.7→2.1 ms/frame/channel; deleted the
   `MultiViewChannelProxy.set_clim` re-push workaround). Signed-int
   planes cast to float32 (`_GL_TEXTURE_DTYPES`).
2. **Histogram debounce** — `ChannelPanel` refreshes 150 ms after the
   last `view_changed`; zero work while hidden (stale-flag + showEvent).
3. **`ViewState`** (`data/view_state.py`) — observable t/z/projection
   nav state; sliders write, one subscription redraws; `t_idx`/`z_idx`
   are compatibility properties.
4. **`SliceLoader`** (`data/slice_loader.py`) — latest-wins worker-thread
   slice reads for lazy sources, byte-budget LRU + t±1/z±1 prefetch,
   `_slice_ready` queued-signal delivery. Z-projection also moved
   off-thread. Numpy sources stay synchronous.
5. **Channel docks** — `show_channel_dock()`; ChannelPanel &
   TiledChannelPanel are QWidgets docked in every QMainWindow viewer.
6. **Lazy package inits** — PEP 562 in root/ui/viewers/readers inits;
   `import pyvistra` ≈ 2 ms, zero import cycles (verified by standalone-
   importing every previously cyclic module).
7. **Workspace shell** (`ui/workspace.py`) — `imshow()` opens tabs in a
   single Workspace window (splitter of tab groups; split-right / float
   / close via tab context menu; `floating=True` opt-out). All viewer
   creation routes through `present_window()`. Embedded windows get
   `WidgetWithChildrenShortcut` scoping. LayerList migration split out
   (see 4e) — the legacy layer dicts remain for now.

## Notes / conventions

- The `data/` dir stays Qt-free and vispy-free. Notifications from
  `ImageBuffer` use plain callbacks, not Qt signals.
- No new docs files. Update docstrings in the touched modules only.
