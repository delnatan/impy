# pyvistra design

This is the project-specific companion to `CLAUDE.md`. `CLAUDE.md` holds
Ousterhout's principles (timeless, generic); this file holds the current
architecture and the running scorecard against those principles, so the two
don't get re-tangled. Update this file as the design evolves — it is meant to
be scrutinized and argued with, not treated as settled.

Regenerate the numbers below with `/graphify --update` on this repo; the
current snapshot is from the god-node analysis in `graphify-out/GRAPH_REPORT.md`
(2026-08-01, 2860 nodes / 4865 edges).

## Module map

```
pyvistra/
  data/       Qt-free, vispy-free data models. Plain callbacks, not Qt
              signals (see rationale in TODO.md). subscribe(callback)/notify
              is the shared idiom across ChannelDisplayList, ViewState,
              OverlayState, ShapeData, SparseLabels.
  layers/     Layer/LayerList abstraction: bundles a data model + visual
              renderer + undo stack. commands.py is the Command pattern
              (reversible mutations) behind every undo stack.
  readers/    Format-specific IO adapters (czi, imaris, imaris_writer).
  visuals/    vispy rendering primitives (image, points, shapes, labels,
              tracks, overlays) — the GPU-facing layer.
  viewers/    Viewer *windows* that compose visuals + layers for a specific
              display mode (ortho, volume, tiled/zmontage grid).
  ui/         Qt chrome: main window (ImageWindow), workspace/menus, layer
              and label management panels, playback controller.
  widgets/    Qt dialogs — one file per dialog, mostly thin and independent.
  plugins.py  Extension points: discover_plugins(), add_menu_item(),
              register_shape_context_action(). External packages (gelkit,
              psfkit, resolvde) register into pyvistra without pyvistra
              importing them back.
```

Layering rule of thumb: `data/` never imports Qt or vispy; `visuals/` never
imports Qt; `ui/` is the only layer allowed to know about both Qt chrome and
the data/visual layers underneath it. `readers/` is a leaf — format-specific,
nothing depends on one reader knowing about another.

## Core abstractions

- **5D array model** `(T, Z, C, Y, X)`. `Readable5D`/`Writable5D`/
  `ObservableBuffer` (`data/protocols.py`) are structural-typing contracts so
  a plugin can type-check against pyvistra without importing its concrete
  proxy classes.
- **Command pattern** (`layers/commands.py`) — every reversible mutation
  (shape edit, label paint, track edit) goes through an `UndoStack` owned by
  the `Layer`, not through ad-hoc undo flags.
- **subscribe/notify state objects** — `ChannelDisplayList`, `ViewState`,
  `OverlayState` are the "one owner, many subscribers" pattern for state that
  used to be duplicated across renderers (composite image, volume, tiled,
  ortho). New shared UI state should follow this shape rather than inventing
  a new one.
- **MENU_SPEC + keyPressEvent dual binding** — menu items declare their own
  keyboard shortcut once in a spec dict; both the menu and the keyPressEvent
  dispatcher read from it. Modal-dialog actions are deliberately excluded
  from the keyPressEvent binding (see memory: `feedback_tiled_menu_shortcut_pattern`).

## God-node scorecard

The refactor initiative (started 2026-07-31, see project memory
`project_ousterhout_refactor`) tracks pyvistra's most-connected nodes and
whether their fan-out is a smell or healthy reuse.

| Node | Degree (2026-08-01) | Degree (pre-refactor) | Verdict |
|---|---|---|---|
| `ImageWindow` (`ui/window.py`) | 161 | 179 | Was a 7-responsibility god class; actively being carved down (see backlog). Still the largest single file (4346 lines). |
| `ShapeData` (`data/shapes.py`) | 88 | 90 | Qt-free data model, high fan-in is legitimate reuse across viewers — not a refactor target. |
| `TiledViewer` (`viewers/tiled.py`) | 69 | — | Fast/normal-mode duplication mostly consolidated (`_for_each_tile`, commit `a21f143`). 2438 lines, largest remaining single class after ImageWindow. |
| `SparseLabels` (`data/labels.py`) | 41 | 42 | Same as ShapeData — data model, healthy fan-in. |
| `LineProfileDialog` | 41 | — | Not yet reviewed. |
| `LabelManager` (`ui/label_manager.py`) | 39 | — | 788 lines, real single-purpose tree-view widget — not a pass-through. Backlog item, lowest priority. |
| `LayerManager` (`ui/layer_manager.py`) | 39 | — | 611 lines, same verdict as LabelManager. |
| `Workspace` (`ui/workspace.py`) | — | — | Not a fan-out target (low degree), but flagged for a design-driven rewrite: split/float bookkeeping (637 lines, ~26 methods) exists to support generic N-way tab splitting nothing actually uses that way — see "Proposed redesign" below. |

`ImageWindow` extraction backlog (chronological, each phase independently
verified via the `verify` skill — no automated GUI test suite exists):

1. ✅ `PlaybackController` → `ui/playback.py`
2. ⏸️ `ShapeEditController` — **paused, not attempted.** The mouse dispatcher
   (`on_mouse_press`/`_move`/`_release`, ~730 lines) is a single tangled
   switchboard across every tool (pointer/shape/brush/point/gel-marker).
   Not cleanly separable without a much bigger rewrite; highest-risk,
   least-covered code path in the app. Don't attempt without re-confirming
   scope first.
3. ✅ `GelAnalyzerWidget` → external `gelkit` plugin (same pattern as
   `resolvde`/`psfkit`). Gel-marker mouse-drag primitives deliberately left
   in `window.py` — moving them means touching the item-2 dispatcher.
4. ✅ Six identical `show_*_dialog` singleton launchers collapsed into
   `_show_singleton_dialog(attr_name, factory)`. The other 8 `show_*`
   methods are structurally different and were left alone.
5. ✅ `_mask_layers`/`_track_layers`/`_point_layers` add/remove/visible/active
   skeleton shared via `_register_layer_entry`/`_unregister_layer_entry`/
   `_set_layer_visible`/`_set_active_layer`. Data-replacement, point
   style/selection, and mouse-handler code left untouched (reaches into the
   item-2 dispatcher).
6. ✅ `TiledViewer`'s fast/normal-mode branches consolidated where truly
   identical (4 of 10 candidate sites); the rest differ in scope or are
   already cleanly split.
7. ⬜ `LabelManager`, `LayerManager` review — lowest priority, not started.

## Known accepted risk

- The shape/point/brush/gel-marker mouse dispatcher in `window.py` (item 2
  above) is intentionally left alone. It's tangled, but every attempt to
  separate it risks the app's most delicate interaction code with no
  automated coverage. Any future work here should be scoped and confirmed
  before starting, not treated as a normal refactor pass.

## Proposed redesign: splits as an explicit Comparison concept (not yet built)

`ui/workspace.py` currently implements "split" as generic N-way tab-group
layout: a `QSplitter` of `_TabGroup` (`QTabWidget`) instances, with tabs
freely movable between groups via a "Split Right"/"Float Window" tab-bar
context menu. There is no persisted relationship between the panes a split
creates — it's indistinguishable from any other multi-tab-group layout.

In practice, though, the only reason to split is to compare two images
side by side (denoise/deconvolution before-vs-after, etc.), and this
conflation is exactly backwards: layout (which should stay simple and
general-purpose, Ousterhout #8) is the *only* thing currently suggesting
"these two windows are related," and that conflation is what's driving
several hacks:

- **Active-pane tracking** is split across two mechanisms that must be
  kept in sync — `Workspace._active_tab_widget` (menu-mirror dispatch) and
  `manager.active_window` (focus-driven, for cross-window dialogs).
  `_TabGroup._activate_current` has to double-wire `currentChanged` *and*
  `tabBarClicked` because Qt's `currentChanged` never fires when clicking a
  tab that's already current in its own group — exactly the state of two
  freshly-split single-tab groups.
- **Un-split** is implicit (last tab in a group closes → group collapses via
  `_collapse_empty_groups`) and needs its own recovery path,
  `_reactivate_surviving_tab`, purely because closing a group can silently
  orphan the workspace's active-tab pointer.
- **Float** (`float_window`) requires a 147-line GL-context rebuild
  (`ImageWindow._rebuild_canvas_for_float`) because reparenting a live vispy
  `QOpenGLWidget` across a top-level-window boundary corrupts its GL context
  on this platform, plus a deferred-resize nudge and macOS native-menu-bar
  juggling.

Meanwhile, the actual cross-window comparison tooling already exists and is
completely decoupled from split/tab adjacency: `LineProfileDialog` keeps a
`series_config` keyed by window id, lets the user pick *any* open window via
"Add Window..." (built from the `manager` registry), does per-window
physical-unit-aware sampling, and reflects the source line onto compared
windows as a read-only overlay via `ImageWindow.show_line_overlay`/
`hide_line_overlay`. `RadialProfileDialog` follows the same pattern. None of
it cares whether the two windows are in split panes, plain tabs, or
floating — so splits should be borrowing this mechanism, not the reverse.

**Recommended direction** (scope confirmed with the user: a comparison is
strictly two windows, not N-way; the plain single-tab experience is
untouched):

- **Layout** stays exactly what it is today for the plain tab-strip case —
  dumb, general-purpose tab management with no comparison awareness.
- **Comparison** becomes a new, special-purpose `ComparisonPair`: references
  exactly two windows, created by an explicit "Compare With..." action
  (mirroring `LineProfileDialog`'s existing window-picker UX and
  `manager.compatible_windows()`), not by incidental tab-group adjacency.
  It's shown in a dedicated two-pane container — each side hosts exactly one
  window persistently, no tab bar per side, no further re-splitting. This
  deliberate loss of N-way generality is what removes the active-tracking
  double-wiring and the implicit-unsplit recovery path: a pair has an
  explicit lifecycle (created by "Compare With...", ended by "Stop
  Comparing" or either member closing) instead of being inferred from
  tab-group emptiness (Ousterhout #11 — design the orphaned-pointer error out
  of existence rather than recovering from it). Floating stays defined only
  for plain tabs; floating a comparison member means ending the comparison
  first.
- What a pair unlocks, mostly by reuse rather than new plumbing:
  - **Line/radial profile**: auto-add the other pair member as a comparison
    series when a line/circle shape is drawn — a thin convenience over
    existing `series_config`/overlay plumbing, not a new sync mechanism.
  - **Kymograph** (currently single-window only, `KymographDialog(viewer,
    layer, shape_id)`): extend to sample both pair members and lay the two
    outputs out together, reusing `BufferProcessingRunner`/
    `ImageOutputSelector` twice.
  - **Linked pan/zoom and linked T/Z/C**: genuinely new (no independent
    `ImageWindow`s currently share camera/playback state — only
    `TiledViewer`'s intra-widget tile sync and `OrthoViewer`'s intra-volume
    camera sync exist as precedent). Model it with a `ViewState`-style
    subscribe/notify object owned by the pair, following the existing
    "one owner, many subscribers" idiom rather than inventing a new one.
  - Shape ownership is unchanged — shapes stay one-per-window in each
    member's own `LayerList`/`ShapeData`; no shared/synced shape model
    needed.

**Backlog for building this** (chronological, same style as the `ImageWindow`
extraction backlog above):

1. ✅ `ComparisonPair` model + "Compare With..." picker + two-pane container
   widget, reusing `manager`/`compatible_windows` (`ui/comparison.py`, new
   `Workspace.compare_windows`/`_end_comparison`/`compare_with_dialog`,
   `ImageWindow.compare_with_dialog`). `add_window`'s per-window docking
   block was factored into reusable `_dock_menu_bar`/`_undock_menu_bar`
   helpers so both a single tab and a pair's two members get the same
   treatment. `Workspace._splitter`/`split_right`/`_collapse_empty_groups`
   needed no changes — a `ComparisonView` docks as one ordinary tab, same
   as `OrthoViewer`/`VolumeViewer`.
   - Bug fixed after initial landing: a pair's two members come from a
     `_TabGroup` (`QTabWidget`), which explicitly hides every tab page
     that isn't current — and leaves the *outgoing* current page hidden
     too once its tab is removed. That explicit hidden state doesn't
     clear on reparenting into a new, visible parent; `QSplitter` then
     allocates a hidden child exactly zero space. Both members stayed
     hidden this way, so a freshly created comparison showed as a blank
     tab with no visible canvas at all. Fixed by explicitly `show()`-ing
     both members in `ComparisonView.__init__` right after they're added
     to the splitter.
2. ✅ Auto-wire `LineProfileDialog`/`RadialProfileDialog` to a pair's other
   member: new `ui/comparison.py:paired_window(window)` helper (shared
   lookup for "the other side of the comparison," reusable by a future
   kymograph-for-pairs pass since it needs the same lookup), called from
   both dialogs' `set_shape_source` right after the source window's own
   series is added — no change to `series_config`/overlay plumbing, purely
   a convenience over the existing manual "Add Window..." step.
3. ✅ Linked pan/zoom + linked T/Z/C, both opt-in per pair (`ComparisonView`
   checkboxes, off by default) and independent of each other:
   - Pan/zoom: `ComparisonPair` connects each member's
     `canvas.events.mouse_wheel`/`mouse_release` (the same coarse
     "camera probably changed" hook `TiledViewer`'s own tile sync already
     uses) and copies `view.camera.rect` to the other member when enabled,
     guarded by a `_syncing_view` flag against feedback loops.
   - T/Z: reuses `ViewState.subscribe()` as-is (already the "one owner,
     many subscribers" pattern) — no new plumbing, just a pair-level
     listener that forwards `t`/`z` changes to the other member's own
     `ViewState`, clamped to that member's own T/Z extent (panes being
     compared don't have to share dimensions).
   - Channel: `ViewState` has no channel field, so `ImageWindow` gained a
     small new `channel_changed = Signal(int)` (emitted from
     `on_channel_change`, alongside the existing coarser `view_changed`)
     — the same fine-grained-signal convention already used for
     `roi_added`/`layer_added`/etc., rather than diffing the coarser
     signal.
   - All three propagation paths share one reentrancy guard
     (`_syncing_tzc` for T/Z/channel, `_syncing_view` for the camera) and
     one generic `(emitter, slot)` connection list torn down in `end()` —
     Qt signals and vispy `EventEmitter`s both expose `connect`/`disconnect`,
     so one list covers both without separate bookkeeping per mechanism.
   - Fixed in passing: `Workspace.add_window`'s GL-canvas-rebuild gate
     was keying off `window.isVisible()` alone, which incorrectly also
     matched a `ComparisonView` member returning to an ordinary tab via
     `_end_comparison` (visible, but never top-level) — every "Stop
     Comparing" was needlessly forcing both members through the
     147-line `_rebuild_canvas_for_float` rebuild. Now gated on
     `isVisible() and isWindow()`, which only the genuine
     float/re-adopt case satisfies.
4. ⬜ Extend `KymographDialog` to run against both pair members.
5. ⬜ Retire `Workspace.split_right`/the generic multi-group splitter path,
   keeping `Workspace` to plain-tab layout only.

Item 5 remains not started; item 4 (kymograph) is deliberately deferred —
it's a rarer tool and would reuse `paired_window` from item 2 directly once
picked back up.

## Health checks worth re-running

- `graphify` god-node degree for `ImageWindow`/`TiledViewer` — should keep
  trending down, not up, as new features land in `window.py`.
- File line counts for `ui/window.py` and `viewers/tiled.py` — both should
  shrink or stay flat; a jump signals new logic landing in the god class
  instead of its own module.
- Grep for a new `show_*_dialog` singleton launcher that duplicates the
  `_show_singleton_dialog` pattern instead of reusing it.
