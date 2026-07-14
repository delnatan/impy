"""Background worker for jetnewton (non-dimensional log-penalty) deconvolution.

A thin Qt shim around deconlib's jetnewton engine
(`deconlib.deconvolution.jetnewton_with_operator`), the successor to and
replacement for the earlier ER-Decon dialog. The dialog hands it a
fully-prepared :class:`~.decon_jetnewton.PreparedInputs` plus the dialog
state, the auto-calibrated Hessian regularizer, and its `s0`/`eta`; the
worker runs the solve in a QThread and emits progress / status / finished /
error signals, writing live previews into the destination `ImageBuffer` as
it goes.

Tiled path: deconlib does not ship a `jetnewton_solver` tile-adapter
(jetnewton is new enough that only the single-volume entry point has been
validated) -- `_run_tiled` below builds the `solve(data_tile, model) ->
visible array` closure `process_tiles` needs locally. `hessian`/`s0`/`eta`
are calibrated once (by the dialog, on the full requested crop) and reused
for every tile: the Hessian is a local finite-difference stencil, so the
per-voxel noise-floor statistics `eta` calibrates against don't depend on
the domain size being probed.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QObject, Signal

from .decon_common import (
    raise_if_nonfinite as _raise_if_nonfinite,
    write_to_buffer as _write_to_buffer,
)
from .decon_jetnewton import JetNewtonDialogState, PreparedInputs, log


class JetNewtonWorker(QObject):
    """Run jetnewton (single-volume or tiled) off the GUI thread."""

    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)        # JetNewtonResult (single-volume) or None (tiled)
    cancelled = Signal()
    error = Signal(str)

    def __init__(
        self,
        *,
        prepared: PreparedInputs,
        state: JetNewtonDialogState,
        hessian,
        s0: float,
        eta: float,
        buffer,                      # ImageBuffer for live preview / final write
        output_channel: int = 0,
        output_frame: int = 0,
    ):
        super().__init__()
        self._p = prepared
        self._state = state
        self._hessian = hessian
        self._s0 = float(s0)
        self._eta = float(eta)
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
            log.exception("jetnewton worker failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._cancel:
            self.cancelled.emit()

    # ------------------------------------------------------------------ #
    # Single-volume path

    def _run_single(self) -> None:
        from deconlib.deconvolution import jetnewton_with_operator, make_forward_model

        p, s = self._p, self._state
        log.info(
            "jetnewton single-volume run: y=%s psf=%s zoom=%s num_iter=%d "
            "background=%g data_term=%s beta=%g eta=%g s0=%g",
            p.y.shape, p.psf.shape, p.zoom, s.num_iter, s.background,
            s.data_term, s.beta, self._eta, self._s0,
        )
        self.status.emit("Building forward model…")
        model = make_forward_model(p.psf, p.y.shape, p.zoom)

        def callback(k: int, x) -> bool:
            if k % max(1, s.eval_interval) == 0:
                restored = np.asarray(x, dtype=np.float32)
                _raise_if_nonfinite(restored, "jetnewton estimate")
                if s.crop_to_visible:
                    restored = restored[model.valid_slices]
                _write_to_buffer(
                    self._buffer, restored, channel=self._output_channel,
                    t=self._output_frame, log_stats=False,
                )
                self.progress.emit(k + 1, s.num_iter)
                self.status.emit(f"jetnewton: iter {k + 1}/{s.num_iter}")
            return bool(self._cancel)

        self.status.emit("Running jetnewton…")
        self.progress.emit(0, s.num_iter)

        result = jetnewton_with_operator(
            observed=p.y,
            blur_op=model.op,
            hessian=self._hessian,
            s0=self._s0,
            background=s.background,
            beta=s.beta,
            eta=self._eta,
            data_term=s.data_term,
            num_iter=s.num_iter,
            cg_max_steps=s.cg_max_steps,
            eps_bar=s.eps_bar,
            freeze_tau=s.freeze_tau,
            freeze_delta=s.freeze_delta,
            newton_tol=s.newton_tol,
            tol=s.tol,
            min_iter=s.min_iter,
            ls_sigma=s.ls_sigma,
            ls_max_backtracks=s.ls_max_backtracks,
            callback=callback,
            eval_interval=s.eval_interval,
            verbose=s.verbose,
        )

        if self._cancel:
            return

        restored = np.asarray(result.restored, dtype=np.float32)
        _raise_if_nonfinite(restored, "jetnewton result")
        if s.crop_to_visible:
            restored = restored[model.valid_slices]
        _write_to_buffer(
            self._buffer, restored, channel=self._output_channel,
            t=self._output_frame,
        )
        self.progress.emit(result.iterations, s.num_iter)
        self.status.emit(f"Done after {result.iterations} iterations")
        log.info(
            "jetnewton done: iterations=%d converged=%s final_idiv=%s restored=%s",
            result.iterations, result.converged,
            f"{result.idiv_history[-1]:.4g}" if result.idiv_history else "n/a",
            tuple(restored.shape),
        )
        self.finished.emit(result)

    # ------------------------------------------------------------------ #
    # Tiled path

    def _run_tiled(self) -> None:
        from deconlib.deconvolution import jetnewton_with_operator, process_tiles

        p, s = self._p, self._state
        guard = s.guard_px if s.guard_px > 0 else None
        hessian, s0, eta, background = self._hessian, self._s0, self._eta, s.background

        def solve(data_tile: np.ndarray, model) -> np.ndarray:
            result = jetnewton_with_operator(
                observed=data_tile,
                blur_op=model.op,
                hessian=hessian,
                s0=s0,
                background=background,
                beta=s.beta,
                eta=eta,
                data_term=s.data_term,
                num_iter=s.num_iter,
                cg_max_steps=s.cg_max_steps,
                eps_bar=s.eps_bar,
                freeze_tau=s.freeze_tau,
                freeze_delta=s.freeze_delta,
                newton_tol=s.newton_tol,
                tol=s.tol,
                min_iter=s.min_iter,
                ls_sigma=s.ls_sigma,
                ls_max_backtracks=s.ls_max_backtracks,
                verbose=s.verbose,
            )
            return np.asarray(result.restored[model.valid_slices])

        done = {"n": 0}
        n_tiles = {"total": None}

        def on_tile_done(spec, output_so_far) -> bool:
            done["n"] += 1
            restored = np.asarray(output_so_far, dtype=np.float32)
            _raise_if_nonfinite(restored, "jetnewton tile result")
            _write_to_buffer(
                self._buffer, restored, channel=self._output_channel,
                t=self._output_frame, log_stats=False,
            )
            total = n_tiles["total"] or done["n"]
            self.progress.emit(done["n"], total)
            self.status.emit(f"jetnewton: tile {done['n']}/{total}")
            return bool(self._cancel)

        from deconlib.deconvolution import plan_tiles

        plan = plan_tiles(
            p.y.shape, p.zoom,
            guard=(guard if guard is not None else max(p.psf.shape[-2:]) // 2),
            tile_size=s.tile_size,
            min_z_slices=s.min_z_slices,
        )
        n_tiles["total"] = len(plan.tiles)
        log.info(
            "jetnewton tiled run: y=%s psf=%s zoom=%s tiles=%d tile_shape=%s "
            "guard=%s num_iter=%d",
            p.y.shape, p.psf.shape, p.zoom, n_tiles["total"], plan.tile_shape,
            guard, s.num_iter,
        )

        self.status.emit(f"Running tiled jetnewton ({n_tiles['total']} tiles)…")
        self.progress.emit(0, n_tiles["total"])

        output = process_tiles(
            p.y, p.psf, p.zoom, solve,
            guard=guard,
            tile_size=s.tile_size,
            min_z_slices=s.min_z_slices,
            on_tile_done=on_tile_done,
        )

        if self._cancel:
            return

        _raise_if_nonfinite(output, "jetnewton tiled result")
        _write_to_buffer(
            self._buffer, output, channel=self._output_channel,
            t=self._output_frame,
        )
        self.progress.emit(n_tiles["total"], n_tiles["total"])
        self.status.emit(f"Done — {n_tiles['total']} tiles")
        log.info("jetnewton tiled done: output=%s", tuple(output.shape))
        self.finished.emit(None)
