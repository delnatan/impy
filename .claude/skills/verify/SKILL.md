---
name: verify
description: Manually run pyvistra's Qt/vispy GUI to observe a change end-to-end (no automated GUI test suite exists yet).
---

# Verifying pyvistra GUI changes

pyvistra is a PyQt6 (via `qtpy`) + vispy desktop app. There is no
pytest-qt harness — `tests/` covers pure-data modules only. Verification
of UI/viewer changes means actually constructing and driving the real
widgets, following the pattern already used in `scripts/verify_*.py`.

## Recipe

```python
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no display in CI/agent sessions

from qtpy.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

# construct real widgets/viewers, e.g.:
from pyvistra.viewers.tiled import TiledViewer
viewer = TiledViewer(image_paths)
viewer.show()

# pump the event loop after anything async (background threads,
# queued signals, deferred deletes):
app.processEvents()
```

Run with the project venv: `.venv/bin/python <script>`.

- `QT_QPA_PLATFORM=offscreen` avoids needing a real display; vispy/GL
  logs a stream of harmless `QOpenGLWidget`/`createPlatformOpenGLContext`
  warnings under offscreen — ignore them, they don't affect assertions.
- For real mouse/keyboard interaction (not just calling methods
  directly), use `qtpy.QtTest.QTest.mouseClick(widget, Qt.LeftButton,
  Qt.NoModifier, QPoint(...))` / `QTest.keyClick(...)` so the actual
  event path (`mousePressEvent`, signal chains) executes, not just the
  handler in isolation.
- Background-thread work (e.g. `data/slice_loader.py`,
  `data/thumbnail_cache.py`) delivers via queued Qt signals — poll with
  a short `time.sleep` + `app.processEvents()` loop and a deadline
  rather than a single `processEvents()` call.
- Generate throwaway test images with Pillow/numpy directly in the
  script (`tempfile.mkdtemp()`), don't rely on fixture files.
- Clean up temp dirs (`shutil.rmtree`) and call `viewer.close()` when
  done — some viewers own background threads (`ThumbnailDecodeWorker`,
  `SliceLoader`) that should be stopped via `closeEvent`.

## Worked example

`TiledViewer` fast-mode thumbnail grid (`pyvistra/viewers/tiled.py`,
`pyvistra/widgets/thumbnail_grid.py`, `pyvistra/data/thumbnail_cache.py`)
was verified this way: generated a folder of small PNGs, constructed
`TiledViewer(paths)`, asserted `viewer._fast_mode` and the concrete
widget type, waited for the background decode worker via
processEvents-in-a-loop, drove a real `QTest.mouseClick` to select a
tile, and paged/sorted/annotated through the real methods, checking
timing on a page revisit to confirm the decode cache actually avoided
re-decoding.
