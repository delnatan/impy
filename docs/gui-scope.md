# GUI Scope

`pyvistra` supports both interactive scripting and GUI workflows, but not all features are equally script-first.

## Best for interactive Python

- multi-format loading with `load_image()`
- direct array/proxy slicing in `(T, Z, C, Y, X)`
- quick visualization with `imshow()`
- conversions via `save_tiff()` and `save_imaris()`
- sparse label IO with `load_sparse_labels()` / `save_sparse_labels()`

## Primarily GUI-driven features

These are available in the application and toolbar/dialog workflow:

- gel analysis workflow
- line profile dialog workflow
- PSF/transform/alignment dialogs
- annotation manager panel interactions

These can still be combined with scripting by loading data in Python first and then opening viewers, but they are designed primarily around GUI interactions.

## Recommended split

1. Use Python scripts/notebooks for IO, conversion, reproducible preprocessing, and batch work.
2. Use the GUI for task-specific interactive analysis sessions.
