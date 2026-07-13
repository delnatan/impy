"""Background worker for ER-Decon (edge-preserving Hessian-log) deconvolution.

A thin Qt shim around deconlib's ER-Decon engine
(`deconlib.deconvolution.erdecon_with_operator` / `erdecon_solver` /
`process_tiles`). The dialog hands it a fully-prepared
:class:`~.decon_erdecon.PreparedInputs` plus the dialog state and the
spacing-weighted Hessian regularizer; the worker runs the solve in a
QThread and emits progress / status / finished / error signals, writing
live previews into the destination `ImageBuffer` as it goes. Mirrors
`deconvolution_worker.NLCGDeconvolutionWorker` structurally.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QObject, Signal

from .decon_common import (
    raise_if_nonfinite as _raise_if_nonfinite,
    write_to_buffer as _write_to_buffer,
)
from .decon_erdecon import ERDeconDialogState, PreparedInputs, log


class ERDeconWorker(QObject):
    """Run ER-Decon (single-volume or tiled) off the GUI thread."""

    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)        # ERDeconResult (single-volume) or None (tiled)
    cancelled = Signal()
    error = Signal(str)

    def __init__(
        self,
        *,
        prepared: PreparedInputs,
        state: ERDeconDialogState,
        hessian,
        buffer,                      # ImageBuffer for live preview / final write
        combine_channels: bool = True,
        output_channel: int = 0,
        output_frame: int = 0,
    ):
        super().__init__()
        self._p = prepared
        self._state = state
        self._hessian = hessian
        self._combine_channels = bool(combine_channels)
        self._buffer = buffer
        self._output_channel = max(0, int(output_channel))
        self._output_frame = max(0, int(output_frame))
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
            log.exception("ER-Decon worker failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._cancel:
            self.cancelled.emit()

    # ------------------------------------------------------------------ #
    # Single-volume path

    def _run_single(self) -> None:
        from deconlib.deconvolution import erdecon_with_operator, make_forward_model

        p, s = self._p, self._state
        log.info(
            "ER-Decon single-volume run: y=%s psf=%s zoom=%s num_iter=%d "
            "background=%g data_term=%s reg_weight=%g eps_reg=%g floor_frac=%g",
            p.y.shape, p.psf.shape, p.zoom, s.num_iter, s.background,
            s.data_term, s.reg_weight, s.eps_reg, s.floor_frac,
        )
        self.status.emit("Building forward model…")
        model = make_forward_model(p.psf, p.y.shape, p.zoom)

        def callback(k: int, g) -> bool:
            if k % max(1, s.eval_interval) == 0:
                restored = np.asarray(g, dtype=np.float32)
                _raise_if_nonfinite(restored, "ER-Decon estimate")
                if s.crop_to_visible:
                    restored = restored[model.valid_slices]
                _write_to_buffer(
                    self._buffer, restored, channel=self._output_channel,
                    t=self._output_frame, log_stats=False,
                )
                self.progress.emit(k + 1, s.num_iter)
                self.status.emit(f"ER-Decon: iter {k + 1}/{s.num_iter}")
            return bool(self._cancel)

        self.status.emit("Running ER-Decon…")
        self.progress.emit(0, s.num_iter)

        result = erdecon_with_operator(
            observed=p.y,
            blur_op=model.op,
            hessian=self._hessian,
            reg_weight=s.reg_weight,
            eps_reg=s.eps_reg,
            data_term=s.data_term,
            combine_channels=self._combine_channels,
            floor_frac=s.floor_frac,
            num_iter=s.num_iter,
            background=s.background,
            normalize=s.normalize,
            callback=callback,
            eval_interval=s.eval_interval,
            newton_tol=s.newton_tol,
            tol=s.tol,
            min_iter=s.min_iter,
            cg_max_steps=s.cg_max_steps,
            cg_tol=s.cg_tol,
            ls_max_backtracks=s.ls_max_backtracks,
            ls_c1=s.ls_c1,
            verbose=s.verbose,
        )

        if self._cancel:
            return

        restored = np.asarray(result.restored, dtype=np.float32)
        _raise_if_nonfinite(restored, "ER-Decon result")
        if s.crop_to_visible:
            restored = restored[model.valid_slices]
        _write_to_buffer(
            self._buffer, restored, channel=self._output_channel,
            t=self._output_frame,
        )
        self.progress.emit(result.iterations, s.num_iter)
        self.status.emit(f"Done after {result.iterations} iterations")
        log.info(
            "ER-Decon done: iterations=%d converged=%s final_loss=%s restored=%s",
            result.iterations, result.converged,
            f"{result.loss_history[-1]:.6g}" if result.loss_history else "n/a",
            tuple(restored.shape),
        )
        self.finished.emit(result)

    # ------------------------------------------------------------------ #
    # Tiled path

    def _run_tiled(self) -> None:
        from deconlib.deconvolution import erdecon_solver, plan_tiles, process_tiles

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
            "ER-Decon tiled run: y=%s psf=%s zoom=%s tiles=%d tile_shape=%s "
            "guard=%s num_iter=%d",
            p.y.shape, p.psf.shape, p.zoom, n_tiles, plan.tile_shape,
            guard, s.num_iter,
        )

        solve = erdecon_solver(
            reg_weight=s.reg_weight,
            eps_reg=s.eps_reg,
            num_iter=s.num_iter,
            background=s.background,
            hessian=self._hessian,
            newton_tol=s.newton_tol,
            data_term=s.data_term,
            combine_channels=self._combine_channels,
            floor_frac=s.floor_frac,
            normalize=s.normalize,
            tol=s.tol,
            min_iter=s.min_iter,
            cg_max_steps=s.cg_max_steps,
            cg_tol=s.cg_tol,
            ls_max_backtracks=s.ls_max_backtracks,
            ls_c1=s.ls_c1,
            verbose=s.verbose,
        )

        done = {"n": 0}

        def on_tile_done(spec, output_so_far) -> bool:
            done["n"] += 1
            restored = np.asarray(output_so_far, dtype=np.float32)
            _raise_if_nonfinite(restored, "ER-Decon tile result")
            _write_to_buffer(
                self._buffer, restored, channel=self._output_channel,
                t=self._output_frame, log_stats=False,
            )
            self.progress.emit(done["n"], n_tiles)
            self.status.emit(f"ER-Decon: tile {done['n']}/{n_tiles}")
            return bool(self._cancel)

        self.status.emit(f"Running tiled ER-Decon ({n_tiles} tiles)…")
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

        _raise_if_nonfinite(output, "ER-Decon tiled result")
        _write_to_buffer(
            self._buffer, output, channel=self._output_channel,
            t=self._output_frame,
        )
        self.progress.emit(n_tiles, n_tiles)
        self.status.emit(f"Done — {n_tiles} tiles")
        log.info("ER-Decon tiled done: output=%s", tuple(output.shape))
        self.finished.emit(None)
