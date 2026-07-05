from typing import Tuple

import numpy as np
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
        self._batch = None

    def is_running(self):
        return self.thread is not None or self._batch is not None

    def prepare_output(
        self,
        output_shape,
        output_dtype,
        output_meta,
        *,
        reuse_existing_buffer: bool = False,
    ) -> Tuple[Readable5D, Writable5D]:
        from pyvistra.io import ImageBuffer

        # Every 5D proxy/buffer is refcounted via RefCountMixin, so
        # acquire() is always available.
        self.source_data = self.viewer.img_data.acquire()
        self.output_meta = output_meta
        self.output_type = self.output_selector.get_selection_type()
        self.output_viewer = None

        if reuse_existing_buffer and self.output_type == "existing":
            window = self.output_selector.selected_existing_window()
            data = None if window is None else getattr(window, "img_data", None)
            if (
                window is not self.viewer
                and data is not None
                and hasattr(data, "acquire")
                and hasattr(data, "__setitem__")
                and tuple(getattr(data, "shape", ())) == tuple(output_shape)
                and np.dtype(getattr(data, "dtype", object))
                == np.dtype(output_dtype)
            ):
                self.output_buffer = data.acquire()
                self.output_viewer = window
                if hasattr(data, "metadata"):
                    data.metadata.update(output_meta)
                window.meta.update(output_meta)
                return self.source_data, self.output_buffer

        self.output_buffer = ImageBuffer(
            shape=output_shape,
            dtype=output_dtype,
            metadata=output_meta,
        )

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
        self._batch = None

    # ---------------------------------------------------------------- #
    # Sequential multi-frame batches (e.g. a T range)

    def run_frames(
        self,
        *,
        frame_ts,
        output_shape,
        output_dtype,
        output_meta,
        prepare_for_t,
        make_worker,
        on_progress,
        on_status,
        on_all_finished,
        on_cancelled,
        on_error,
        reuse_existing_buffer: bool = False,
    ):
        """Run one worker per entry in `frame_ts`, writing every result into
        a single output buffer sized for the whole batch.

        `prepare_for_t(t)` returns per-frame data for `make_worker`, or
        `None` after reporting its own error (same contract as a dialog's
        `_prepare()`). `make_worker(frame_data, output_frame_idx)` builds
        the engine-specific worker for that frame (constructed with
        `output_frame=output_frame_idx` so it writes into the right slot).

        `on_progress`/`on_cancelled`/`on_error` are the same per-worker
        callbacks `start_worker` takes. `on_status` is wrapped with a
        "Frame i/N: " prefix automatically when there's more than one
        frame. `on_all_finished` (no args) fires once, after the *last*
        frame's worker finishes -- source/output refcounts are held for
        the whole batch and released only then (or on cancel/error), not
        per frame.
        """
        self.prepare_output(
            output_shape=output_shape,
            output_dtype=output_dtype,
            output_meta=output_meta,
            reuse_existing_buffer=reuse_existing_buffer,
        )
        self._batch = {
            "frame_ts": list(frame_ts),
            "idx": 0,
            "prepare_for_t": prepare_for_t,
            "make_worker": make_worker,
            "on_progress": on_progress,
            "on_status": on_status,
            "on_all_finished": on_all_finished,
            "on_cancelled": on_cancelled,
            "on_error": on_error,
            "cancel_requested": False,
            "errored": False,
            "error_message": "",
        }
        self._run_next_frame()

    def _run_next_frame(self):
        b = self._batch
        t = b["frame_ts"][b["idx"]]
        frame_data = b["prepare_for_t"](t)
        if frame_data is None:
            b["errored"] = True
            b["error_message"] = "Frame preparation failed."
            b["on_error"](b["error_message"])
            self.cleanup()
            return

        n = len(b["frame_ts"])
        worker = b["make_worker"](frame_data, b["idx"])
        self.worker = worker
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(b["on_progress"])
        if b["on_status"] is not None:
            idx = b["idx"]
            if n > 1:
                self.worker.status.connect(
                    lambda msg, i=idx: b["on_status"](f"Frame {i + 1}/{n}: {msg}")
                )
            else:
                self.worker.status.connect(b["on_status"])
        self.worker.finished.connect(self._on_frame_finished)
        self.worker.cancelled.connect(self._on_frame_cancelled)
        self.worker.error.connect(self._on_frame_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self._on_frame_thread_finished)

        self.thread.start()

    def _on_frame_finished(self, result=None):
        self._batch["idx"] += 1

    def _on_frame_cancelled(self):
        self._batch["cancel_requested"] = True

    def _on_frame_error(self, message):
        self._batch["errored"] = True
        self._batch["error_message"] = message

    def _on_frame_thread_finished(self):
        b = self._batch
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None

        if b["errored"]:
            b["on_error"](b["error_message"])
            self.cleanup()
            return
        if b["cancel_requested"]:
            b["on_cancelled"]()
            self.cleanup()
            return
        if b["idx"] >= len(b["frame_ts"]):
            b["on_all_finished"]()
            self.cleanup()
            return
        self._run_next_frame()
