from typing import Tuple

from qtpy.QtCore import QThread

from pyvistra.data import Readable5D, Writable5D


class BufferProcessingRunner:
    """
    Reusable helper for long-running image processing dialogs.

    Responsibilities:
    - acquire/release source data proxy
    - create ImageBuffer output
    - route output via ImageOutputSelector
    - manage QThread + worker wiring and cleanup

    Workers receive a :class:`Readable5D` source and a :class:`Writable5D`
    destination (an :class:`ImageBuffer`). Live preview happens
    automatically when the destination is a window — the viewer subscribes
    to buffer changes and refreshes the displayed slice on overlap.

    A processor dialog needs three pieces:

      1. A params form (its own widgets).
      2. A worker (``QObject`` with ``progress`` / ``finished`` /
         ``cancelled`` / ``error`` signals and a ``run`` method).
      3. An :class:`ImageOutputSelector`.

    Everything else — thread lifetime, refcounting, output routing — is
    this runner's job. Synchronous one-shot computations (no worker
    thread) should skip the runner and call
    ``output_selector.send(buffer, metadata)`` directly.
    """

    def __init__(self, viewer, output_selector):
        self.viewer = viewer
        self.output_selector = output_selector

        self.source_data = None
        self.output_buffer = None
        self.output_viewer = None
        self.output_meta = None
        self.output_type = None
        self.thread = None
        self.worker = None

    def is_running(self):
        return self.thread is not None

    def prepare_output(
        self, output_shape, output_dtype, output_meta
    ) -> Tuple[Readable5D, Writable5D]:
        from pyvistra.io import ImageBuffer

        # Every 5D proxy/buffer is refcounted via RefCountMixin, so
        # acquire() is always available.
        self.source_data = self.viewer.img_data.acquire()
        self.output_meta = output_meta
        self.output_type = self.output_selector.get_selection_type()
        self.output_buffer = ImageBuffer(
            shape=output_shape,
            dtype=output_dtype,
            metadata=output_meta,
        )
        self.output_viewer = None

        # Stream live only for window outputs.
        if self.output_type in ("new", "existing"):
            result = self.output_selector.send(
                self.output_buffer.acquire(),
                output_meta.copy(),
            )
            if hasattr(result, "update_view"):
                self.output_viewer = result

        return self.source_data, self.output_buffer

    def start_worker(
        self,
        worker,
        on_progress,
        on_finished,
        on_cancelled,
        on_error,
        on_thread_finished,
    ):
        self.worker = worker
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(on_progress)
        self.worker.finished.connect(on_finished)
        self.worker.cancelled.connect(on_cancelled)
        self.worker.error.connect(on_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(on_thread_finished)

        self.thread.start()

    def finalize_output(self):
        if self.output_type == "file" and self.output_buffer is not None:
            return self.output_selector.send(
                self.output_buffer, self.output_meta.copy()
            )
        return self.output_viewer

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()

    def cleanup(self):
        if self.source_data is not None:
            self.source_data.release()
        if self.output_buffer is not None:
            self.output_buffer.release()

        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()

        self.source_data = None
        self.output_buffer = None
        self.output_viewer = None
        self.output_meta = None
        self.output_type = None
        self.worker = None
        self.thread = None
