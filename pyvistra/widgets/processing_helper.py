from qtpy.QtCore import QThread


class BufferProcessingRunner:
    """
    Reusable helper for long-running image processing dialogs.

    Responsibilities:
    - acquire/release source data proxy
    - create ImageBuffer output
    - route output via ImageOutputSelector
    - manage QThread + worker wiring and cleanup
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

    def prepare_output(self, output_shape, output_dtype, output_meta):
        from pyvistra.io import ImageBuffer

        source = self.viewer.img_data
        if hasattr(source, "acquire"):
            source = source.acquire()

        self.source_data = source
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
        on_plane_done,
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
        self.worker.plane_done.connect(on_plane_done)
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

    def refresh_output_view(self, t, z):
        if self.output_viewer is None:
            return
        if self.output_viewer.t_idx == t and self.output_viewer.z_idx == z:
            self.output_viewer.update_view()
        else:
            self.output_viewer.canvas.update()

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()

    def cleanup(self):
        if self.source_data is not None and hasattr(self.source_data, "release"):
            self.source_data.release()
        if self.output_buffer is not None and hasattr(
            self.output_buffer, "release"
        ):
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
