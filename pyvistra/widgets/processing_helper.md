# Processing Helper Pattern

`BufferProcessingRunner` is the recommended pattern for long-running image
processing widgets in `pyvistra`.

It handles:

- acquiring/releasing the source proxy
- creating an `ImageBuffer`
- routing output via `ImageOutputSelector`
- wiring a worker into `QThread`
- safe cleanup after finish/cancel/error

## Minimal Usage

```python
from qtpy.QtCore import QObject, Signal
from pyvistra.widgets.output_selector import ImageOutputSelector
from pyvistra.widgets.processing_helper import BufferProcessingRunner


class MyWorker(QObject):
    progress = Signal(int, int)
    plane_done = Signal(int, int, int)
    finished = Signal()
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, source, out_buffer, params):
        super().__init__()
        self._source = source
        self._buffer = out_buffer
        self._params = params
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        # Do processing, write into self._buffer, emit signals
        ...


# In your dialog __init__:
self.output_selector = ImageOutputSelector(
    default_title="Result",
    formats=[("TIFF", ".tif"), ("Imaris", ".ims")],
)
self.runner = BufferProcessingRunner(self.viewer, self.output_selector)


# In your start handler:
source, out_buffer = self.runner.prepare_output(
    output_shape=(T, Z, C, Y, X),
    output_dtype=np.float32,
    output_meta=output_meta,
)

worker = MyWorker(source, out_buffer, params)
self.runner.start_worker(
    worker=worker,
    on_progress=self._on_progress,
    on_plane_done=self._on_plane_done,
    on_finished=self._on_finished,
    on_cancelled=self._on_cancelled,
    on_error=self._on_error,
    on_thread_finished=self._cleanup_thread,
)


# In callbacks:
def _on_plane_done(self, t, z, c):
    self.runner.refresh_output_view(t, z)


def _on_finished(self):
    # For "Save to File", finalize_output() triggers the actual write.
    self.runner.finalize_output()


def _cleanup_thread(self):
    self.runner.cleanup()
```

## Notes

- For `new`/`existing` window output, the viewer can read from the buffer while
  the worker writes, enabling live updates.
- For `file` output, processing runs first and the save happens at finish.
- Your dialog should ignore close while worker is active and call `runner.cancel()`.
