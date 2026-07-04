"""Background worker for Richardson-Lucy deconvolution.

A thin Qt shim around deconlib's RL engine
(`deconlib.deconvolution.richardson_lucy_with_operator` /
`richardson_lucy_solver` / `process_tiles`). Structurally mirrors
`NLCGDeconvolutionWorker` (`deconvolution_worker.py`) -- same
prepared-inputs contract, same live-preview-via-buffer callback pattern --
minus the regularizer (RL has none in deconlib) and NLCG-specific solver
knobs.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QObject, Signal

from .decon_common import (
    mean_poisson_i_divergence,
    raise_if_nonfinite,
    write_to_buffer,
)
from .decon_rl import PreparedInputs, RLDialogState, log


class RLDeconvolutionWorker(QObject):
    """Run Richardson-Lucy (single-volume or tiled) off the GUI thread."""

    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)        # RLResult (single-volume) or None (tiled)
    cancelled = Signal()
    error = Signal(str)

    def __init__(
        self,
        *,
        prepared: PreparedInputs,
        state: RLDialogState,
        buffer,                      # ImageBuffer for live preview / final write
        output_channel: int = 0,
    ):
        super().__init__()
        self._p = prepared
        self._state = state
        self._buffer = buffer
        self._output_channel = max(0, int(output_channel))
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            if self._state.tiled:
                self._run_tiled()
            else:
                self._run_single()
        except Exception as exc:
            log.exception("Richardson-Lucy worker failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._cancel:
            self.cancelled.emit()

    # ------------------------------------------------------------------ #
    # Single-volume path

    def _run_single(self) -> None:
        from deconlib.deconvolution import make_forward_model, richardson_lucy_with_operator

        p, s = self._p, self._state
        log.info(
            "RL single-volume run: y=%s psf=%s zoom=%s num_iter=%d background=%g",
            p.y.shape, p.psf.shape, p.zoom, s.num_iter, s.background,
        )
        self.status.emit("Building forward model…")
        model = make_forward_model(p.psf, p.y.shape, p.zoom)

        def callback(k: int, x) -> bool:
            if k % max(1, s.eval_interval) == 0:
                restored = np.asarray(x, dtype=np.float32)
                raise_if_nonfinite(restored, "RL estimate")
                if s.crop_to_visible:
                    restored = restored[model.valid_slices]
                write_to_buffer(
                    self._buffer, restored, channel=self._output_channel,
                    log_stats=False,
                )
                self.progress.emit(k + 1, s.num_iter)
                if s.verbose:
                    pred = np.asarray(model.op.forward(x), dtype=np.float32) + s.background
                    loss = mean_poisson_i_divergence(p.y, pred)
                    self.status.emit(
                        f"RL: iter {k + 1}/{s.num_iter}  mean I-div={loss:.4g}"
                    )
                else:
                    self.status.emit(f"RL: iter {k + 1}/{s.num_iter}")
            return bool(self._cancel)

        self.status.emit("Running Richardson-Lucy…")
        self.progress.emit(0, s.num_iter)

        result = richardson_lucy_with_operator(
            observed=p.y,
            blur_op=model.op,
            num_iter=s.num_iter,
            background=s.background,
            callback=callback,
            eval_interval=s.eval_interval,
            verbose=s.verbose,
        )

        if self._cancel:
            return

        restored = np.asarray(result.restored, dtype=np.float32)
        raise_if_nonfinite(restored, "RL result")
        if s.crop_to_visible:
            restored = restored[model.valid_slices]
        write_to_buffer(self._buffer, restored, channel=self._output_channel)
        self.progress.emit(result.iterations, s.num_iter)
        self.status.emit(f"Done after {result.iterations} iterations")
        log.info(
            "RL done: iterations=%d final_loss=%s restored=%s",
            result.iterations,
            f"{result.loss_history[-1]:.6g}" if result.loss_history else "n/a",
            tuple(restored.shape),
        )
        self.finished.emit(result)

    # ------------------------------------------------------------------ #
    # Tiled path

    def _run_tiled(self) -> None:
        from deconlib.deconvolution import plan_tiles, process_tiles, richardson_lucy_solver

        p, s = self._p, self._state
        guard = s.guard_px if s.guard_px > 0 else None
        plan = plan_tiles(
            p.y.shape, p.zoom,
            guard=(guard if guard is not None else max(p.psf.shape[-2:]) // 2),
            tile_size=s.tile_size,
            min_z_slices=s.min_z_slices,
        )
        n_tiles = len(plan.tiles)
        log.info(
            "RL tiled run: y=%s psf=%s zoom=%s tiles=%d tile_shape=%s "
            "guard=%s num_iter=%d",
            p.y.shape, p.psf.shape, p.zoom, n_tiles, plan.tile_shape,
            guard, s.num_iter,
        )

        solve = richardson_lucy_solver(
            num_iter=s.num_iter,
            background=s.background,
            init_value=float(np.mean(p.y)) / float(np.prod(p.zoom)),
            eval_interval=s.eval_interval,
            verbose=s.verbose,
        )

        done = {"n": 0}

        def on_tile_done(spec, output_so_far) -> bool:
            done["n"] += 1
            restored = np.asarray(output_so_far, dtype=np.float32)
            raise_if_nonfinite(restored, "RL tile result")
            write_to_buffer(
                self._buffer, restored, channel=self._output_channel,
                log_stats=False,
            )
            self.progress.emit(done["n"], n_tiles)
            self.status.emit(f"RL: tile {done['n']}/{n_tiles}")
            return bool(self._cancel)

        self.status.emit(f"Running tiled Richardson-Lucy ({n_tiles} tiles)…")
        self.progress.emit(0, n_tiles)

        output = process_tiles(
            p.y, p.psf, p.zoom, solve,
            guard=guard,
            tile_size=s.tile_size,
            min_z_slices=s.min_z_slices,
            on_tile_done=on_tile_done,
        )

        if self._cancel:
            return

        raise_if_nonfinite(output, "RL tiled result")
        write_to_buffer(self._buffer, output, channel=self._output_channel)
        self.progress.emit(n_tiles, n_tiles)
        self.status.emit(f"Done — {n_tiles} tiles")
        log.info("RL tiled done: output=%s", tuple(output.shape))
        self.finished.emit(None)
