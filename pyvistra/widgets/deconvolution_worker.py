"""Background worker for NLCG deconvolution.

A thin Qt shim around deconlib's NLCG engine
(`deconlib.deconvolution.nlcg_with_operator` / `nlcg_solver` /
`process_tiles`). The dialog hands it a fully-prepared
:class:`~.decon_nlcg.PreparedInputs` plus the dialog state and an optional
regularizer; the worker runs the solve in a QThread and emits
progress / status / finished / error signals, writing live previews into
the destination `ImageBuffer` as it goes.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QObject, Signal

from .decon_nlcg import NLCGDialogState, PreparedInputs, log


class NLCGDeconvolutionWorker(QObject):
    """Run NLCG (single-volume or tiled) off the GUI thread."""

    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)        # NLCGResult (single-volume) or None (tiled)
    cancelled = Signal()
    error = Signal(str)

    def __init__(
        self,
        *,
        prepared: PreparedInputs,
        state: NLCGDialogState,
        regularizer,
        buffer,                      # ImageBuffer for live preview / final write
        output_channel: int = 0,
    ):
        super().__init__()
        self._p = prepared
        self._state = state
        self._regularizer = regularizer
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
            log.exception("NLCG worker failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._cancel:
            self.cancelled.emit()

    # ------------------------------------------------------------------ #
    # Single-volume path

    def _run_single(self) -> None:
        import mlx.core as mx
        from deconlib.deconvolution import make_forward_model, nlcg_with_operator

        p, s = self._p, self._state
        log.info(
            "NLCG single-volume run: y=%s psf=%s zoom=%s num_iter=%d "
            "background=%g regularizer=%s(weight=%g)",
            p.y.shape, p.psf.shape, p.zoom, s.num_iter, s.background,
            s.regularizer_kind, s.reg_weight,
        )
        self.status.emit("Building forward model…")
        model = make_forward_model(p.psf, p.y.shape, p.zoom)

        data_mean = float(np.mean(p.y))
        init_value = data_mean / float(np.prod(p.zoom))
        init = mx.full(model.padded_shape, init_value, dtype=mx.float32)

        def callback(k: int, f) -> bool:
            if k % max(1, s.eval_interval) == 0:
                restored = np.asarray(f, dtype=np.float32)
                _raise_if_nonfinite(restored, "NLCG estimate")
                if s.crop_to_visible:
                    restored = restored[model.valid_slices]
                _write_to_buffer(
                    self._buffer, restored, channel=self._output_channel,
                    log_stats=False,
                )
                self.progress.emit(k + 1, s.num_iter)
                if s.verbose:
                    pred = np.asarray(model.op.forward(f), dtype=np.float32) + s.background
                    loss = _mean_poisson_i_divergence(p.y, pred)
                    self.status.emit(
                        f"NLCG: iter {k + 1}/{s.num_iter}  mean I-div={loss:.4g}"
                    )
                else:
                    self.status.emit(f"NLCG: iter {k + 1}/{s.num_iter}")
            return bool(self._cancel)

        self.status.emit("Running NLCG…")
        self.progress.emit(0, s.num_iter)
        restart_interval = s.restart_interval if s.restart_interval > 0 else None

        result = nlcg_with_operator(
            observed=p.y,
            blur_op=model.op,
            num_iter=s.num_iter,
            background=s.background,
            regularizer=self._regularizer,
            reg_weight=s.reg_weight,
            init=init,
            callback=callback,
            eval_interval=s.eval_interval,
            slack=s.slack,
            tol=s.tol,
            min_iter=s.min_iter,
            restart_interval=restart_interval,
            newton_iters=s.newton_iters,
            verbose=s.verbose,
        )

        if self._cancel:
            return

        restored = np.asarray(result.restored, dtype=np.float32)
        _raise_if_nonfinite(restored, "NLCG result")
        if s.crop_to_visible:
            restored = restored[model.valid_slices]
        _write_to_buffer(self._buffer, restored, channel=self._output_channel)
        self.progress.emit(result.iterations, s.num_iter)
        self.status.emit(f"Done after {result.iterations} iterations")
        log.info(
            "NLCG done: iterations=%d converged=%s final_loss=%s restored=%s",
            result.iterations, result.converged,
            f"{result.loss_history[-1]:.6g}" if result.loss_history else "n/a",
            tuple(restored.shape),
        )
        self.finished.emit(result)

    # ------------------------------------------------------------------ #
    # Tiled path

    def _run_tiled(self) -> None:
        from deconlib.deconvolution import nlcg_solver, plan_tiles, process_tiles

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
            "NLCG tiled run: y=%s psf=%s zoom=%s tiles=%d tile_shape=%s "
            "guard=%s num_iter=%d",
            p.y.shape, p.psf.shape, p.zoom, n_tiles, plan.tile_shape,
            guard, s.num_iter,
        )

        data_mean = float(np.mean(p.y))
        init_value = data_mean / float(np.prod(p.zoom))
        restart_interval = s.restart_interval if s.restart_interval > 0 else None

        solve = nlcg_solver(
            num_iter=s.num_iter,
            background=s.background,
            regularizer=self._regularizer,
            reg_weight=s.reg_weight,
            init_value=init_value,
            eval_interval=s.eval_interval,
            slack=s.slack,
            tol=s.tol,
            min_iter=s.min_iter,
            restart_interval=restart_interval,
            newton_iters=s.newton_iters,
            verbose=s.verbose,
        )

        done = {"n": 0}

        def on_tile_done(spec, output_so_far) -> bool:
            done["n"] += 1
            restored = np.asarray(output_so_far, dtype=np.float32)
            _raise_if_nonfinite(restored, "NLCG tile result")
            _write_to_buffer(
                self._buffer, restored, channel=self._output_channel,
                log_stats=False,
            )
            self.progress.emit(done["n"], n_tiles)
            self.status.emit(f"NLCG: tile {done['n']}/{n_tiles}")
            return bool(self._cancel)

        self.status.emit(f"Running tiled NLCG ({n_tiles} tiles)…")
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

        _raise_if_nonfinite(output, "NLCG tiled result")
        _write_to_buffer(self._buffer, output, channel=self._output_channel)
        self.progress.emit(n_tiles, n_tiles)
        self.status.emit(f"Done — {n_tiles} tiles")
        log.info("NLCG tiled done: output=%s", tuple(output.shape))
        self.finished.emit(None)


# --------------------------------------------------------------------------- #
# Helpers

def _mean_poisson_i_divergence(
    observed: np.ndarray, model: np.ndarray, eps: float = 1e-6
) -> float:
    """Mean Poisson I-divergence I(data || model), for verbose status text.

    Mirrors ``deconlib.deconvolution.rl_mlx.poisson_i_divergence`` (a small,
    stable numpy formula, kept local rather than importing an internal
    module path). Only computed when the "Verbose" option is on — it costs
    one extra forward convolution per logged iteration.
    """
    obs = np.asarray(observed, dtype=np.float64)
    mod = np.maximum(np.asarray(model, dtype=np.float64), eps)
    obs_safe = np.maximum(obs, eps)
    return float(np.mean(obs * np.log(obs_safe / mod) - obs + mod))


def _raise_if_nonfinite(arr: np.ndarray, label: str) -> None:
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count or inf_count:
        raise ValueError(
            f"{label} contains {nan_count} NaN and {inf_count} Inf values; "
            "the solver diverged before producing a usable result"
        )


def _write_to_buffer(
    buffer, arr: np.ndarray, *, channel: int = 0, log_stats: bool = True
) -> None:
    channel = max(0, int(channel))
    if channel >= buffer.shape[2]:
        raise ValueError(
            f"output channel {channel} is outside buffer channel count {buffer.shape[2]}"
        )

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count or inf_count:
        log.error(
            "result contains %d NaN and %d Inf values out of %d — "
            "the solver diverged. Common causes: PSF / data spacing "
            "mismatch, negative pixel values in the data, or a PSF that "
            "is not at the same fine-grid sampling the model expects.",
            nan_count, inf_count, arr.size,
        )
    elif log_stats:
        log.info("result stats: min=%.6g max=%.6g mean=%.6g sum=%.6g",
                 float(arr.min()), float(arr.max()),
                 float(arr.mean()), float(arr.sum()))

    if arr.ndim == 2:
        target = buffer.shape[-2:]
        if arr.shape != target:
            log.error("buffer/result shape mismatch: result=%s buffer=%s",
                      arr.shape, buffer.shape)
        buffer[0, 0, channel, :, :] = arr
    elif arr.ndim == 3:
        target = (buffer.shape[1], buffer.shape[3], buffer.shape[4])
        if arr.shape != target:
            log.error("buffer/result shape mismatch: result=%s buffer=%s",
                      arr.shape, buffer.shape)
        buffer[0, :, channel, :, :] = arr
    else:
        raise ValueError(f"unexpected result ndim {arr.ndim}")
