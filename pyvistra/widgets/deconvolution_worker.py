"""Background workers for deconvolution.

Both workers are thin Qt shims around deconlib's workflow drivers
(`run_deconvolution_workflow` and `run_richardson_lucy`). The dialog
hands them a fully-prepared :class:`PreparedInputs` plus the
algorithm-specific config; the workers run the driver in a QThread and
emit progress / status / finished / error signals.

MEM can forward structured workflow progress from deconlib; RL drives a
callback that reports each iteration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy.QtCore import QObject, Signal

from .decon_recipe import PreparedInputs, log


# --------------------------------------------------------------------------- #
# MEM worker

class MemDeconvolutionWorker(QObject):
    """Run `run_deconvolution_workflow` off-thread.

    The driver forwards deconlib's structured per-iteration workflow
    progress into Qt signals for the dialog.
    """

    progress = Signal(int, int)
    status   = Signal(str)
    finished = Signal(object)        # WorkflowResult
    cancelled = Signal()
    error    = Signal(str)

    def __init__(
        self,
        *,
        prepared: PreparedInputs,
        map_config,
        icf_sweep,
        posterior,
        buffer,                     # ImageBuffer for live preview
        likelihood: str = "poisson",
        preview_every_outer: int = 1,
    ):
        super().__init__()
        self._p = prepared
        self._map_config = map_config
        self._icf_sweep = icf_sweep
        self._posterior = posterior
        self._buffer = buffer
        self._likelihood = likelihood
        self._preview_every_outer = max(1, int(preview_every_outer))
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            from deconlib import (
                WorkflowCancelled,
                WorkflowProgress,
                run_deconvolution_workflow,
            )

            log.info(
                "MEM run: y=%s hidden=%s psf=%s likelihood=%s "
                "icf_sweep=%s posterior=%s",
                self._p.y.shape, self._p.geometry.hidden_shape,
                self._p.psf.psf.shape, self._likelihood,
                None if self._icf_sweep is None
                else f"sigmas={self._icf_sweep.sigmas_um} refine={self._icf_sweep.refine}",
                None if self._posterior is None
                else f"n_samples={self._posterior.n_samples}",
            )
            def progress_cb(update: WorkflowProgress) -> bool:
                self.progress.emit(
                    int(update.total_iteration),
                    int(update.total_max_iterations),
                )
                if update.preview is not None:
                    if update.preview_space is None:
                        raise ValueError("MEM preview is missing preview_space metadata")
                    self._write_preview(
                        np.asarray(update.preview, dtype=np.float32),
                        preview_space=update.preview_space,
                    )
                self.status.emit(self._format_progress_status(update))
                return bool(self._cancel)

            self.status.emit("Running MEM workflow…")
            self.progress.emit(0, 1)
            result = run_deconvolution_workflow(
                self._p.y,
                self._p.prior,
                base_recipe=self._p.recipe,
                psf=self._p.psf,
                optics=self._p.optics,
                geometry=self._p.geometry,
                sigma=self._p.sigma,
                likelihood=self._likelihood,
                icf_sweep=self._icf_sweep,
                map_config=self._map_config,
                posterior=self._posterior,
                progress=progress_cb,
                preview_every_outer=self._preview_every_outer,
            )
            self._write_final(result)
            log.info(
                "MEM done: chosen_recipe.icf=%s scan_rows=%d refined=%s",
                result.chosen_recipe.icf, len(result.scan), result.refined,
            )
        except WorkflowCancelled:
            log.info("MEM worker cancelled via workflow progress callback")
            self.cancelled.emit()
            return
        except Exception as exc:
            log.exception("MEM worker failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._cancel:
            self.cancelled.emit()
        else:
            self.finished.emit(result)

    def _format_progress_status(self, update) -> str:
        stage = {
            "single": "MEM",
            "baseline": "Baseline",
            "scan": "ICF sweep",
            "refine": "Refining sigma",
            "final": "Final run",
        }.get(update.stage, str(update.stage))

        details = []
        if update.stage == "scan" and update.sweep_index is not None and update.sweep_total is not None:
            details.append(f"{update.sweep_index}/{update.sweep_total}")
        if update.sigma_um is not None:
            details.append(f"σ={update.sigma_um:.4g} µm")
        details.append(f"outer iter {int(update.stage_iteration)}")
        if update.chi2 is not None:
            details.append(f"chi2={float(update.chi2):.4g}")
        if update.omega is not None:
            details.append(f"omega={float(update.omega):.4g}")
        if update.converged:
            details.append("converged")
        return f"{stage}: " + " | ".join(details)

    def _write_final(self, result) -> None:
        """Push the final MAP into the live buffer.

        Prefer the visible-space MAP ``f = C(h)`` when available so the
        final write matches the live MEM previews. When the dialog
        requested ``return_region == "valid"``, crop to the detector
        field on the fine grid (matching deconlib's RL "valid" crop).
        """
        arr = np.asarray(result.final.map.f, dtype=np.float32)
        if self._p.output_slices is not None:
            log.info("MEM crop: %s → %s via %s",
                     arr.shape,
                     tuple(s.stop - s.start for s in self._p.output_slices),
                     self._p.output_slices)
            arr = arr[self._p.output_slices]
        _write_to_buffer(self._buffer, arr)

    def _coerce_preview_array(
        self,
        arr: np.ndarray,
        *,
        preview_space: str,
    ) -> np.ndarray:
        if preview_space == "hidden":
            expected_shape = tuple(self._p.geometry.hidden_shape)
        elif preview_space == "visible":
            expected_shape = tuple(self._p.geometry.visible_shape)
        else:
            raise ValueError(f"unknown MEM preview_space {preview_space!r}")
        if arr.shape == expected_shape:
            return arr
        raise ValueError(
            "MEM preview shape mismatch: "
            f"got {arr.shape}, expected {expected_shape} "
            f"for preview_space={preview_space!r}"
        )

    def _write_preview(
        self,
        arr: np.ndarray,
        *,
        preview_space: str,
    ) -> None:
        arr = self._coerce_preview_array(arr, preview_space=preview_space)
        if self._p.output_slices is not None:
            arr = arr[self._p.output_slices]
        _write_to_buffer(self._buffer, arr, log_stats=False)


# --------------------------------------------------------------------------- #
# Richardson-Lucy worker

class RLDeconvolutionWorker(QObject):
    """Run `run_richardson_lucy` off-thread.

    RL has cheap per-iteration updates — we stream those iterations into
    the live buffer through the driver's callback hook.
    """

    progress = Signal(int, int)
    status   = Signal(str)
    finished = Signal(object)        # RichardsonLucyResult
    cancelled = Signal()
    error    = Signal(str)

    def __init__(self, *, prepared: PreparedInputs, rl_config, buffer):
        super().__init__()
        self._p = prepared
        self._rl_config = rl_config
        self._buffer = buffer
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            import mlx.core as mx
            from deconlib.deconvolution import richardson_lucy_with_operator
            from deconlib.workflow import (
                RichardsonLucyResult,
                _blur_op_from_recipe,
            )

            log.info(
                "RL run: y=%s hidden=%s psf=%s num_iter=%d "
                "background=%g eval_interval=%d return_region=%s",
                self._p.y.shape, self._p.geometry.hidden_shape,
                self._p.psf.psf.shape,
                self._rl_config.num_iter, self._rl_config.background,
                self._rl_config.eval_interval, self._rl_config.return_region,
            )
            self.status.emit("Running Richardson–Lucy…")
            self.progress.emit(0, self._rl_config.num_iter)

            blur_op = _blur_op_from_recipe(
                self._p.recipe,
                psf=self._p.psf,
                optics=self._p.optics,
                geometry=self._p.geometry,
                likelihood="poisson",
            )

            def callback(iteration: int, estimate) -> bool:
                restored = np.asarray(estimate, dtype=np.float32)
                if self._p.output_slices is not None:
                    restored = restored[self._p.output_slices]
                _write_to_buffer(self._buffer, restored, log_stats=False)
                done = int(iteration) + 1
                self.progress.emit(done, self._rl_config.num_iter)
                self.status.emit(
                    f"Richardson–Lucy: iter {done}/{self._rl_config.num_iter}"
                )
                return bool(self._cancel)

            rl_result = richardson_lucy_with_operator(
                observed=np.asarray(self._p.y, dtype=np.float32),
                blur_op=blur_op,
                num_iter=self._rl_config.num_iter,
                background=self._rl_config.background,
                eval_interval=self._rl_config.eval_interval,
                return_region=self._rl_config.return_region,
                callback=callback,
            )

            restored_np = np.asarray(rl_result.restored, dtype=np.float32)
            full_for_pred_mx = mx.array(np.asarray(rl_result.restored, dtype=np.float32))
            if (
                self._rl_config.return_region == "valid"
                and rl_result.valid_slices is not None
            ):
                full = np.zeros(rl_result.full_shape, dtype=np.float32)
                full[rl_result.valid_slices] = restored_np
                full_for_pred_mx = mx.array(full)
            pred_mx = blur_op.forward(full_for_pred_mx)
            mx.eval(pred_mx)
            result = RichardsonLucyResult(
                restored=restored_np,
                pred=np.asarray(pred_mx, dtype=np.float32),
                iterations=int(rl_result.iterations),
                loss_history=tuple(float(v) for v in rl_result.loss_history),
                background=float(self._rl_config.background),
                return_region=str(self._rl_config.return_region),
                full_shape=tuple(int(s) for s in rl_result.full_shape),
                valid_slices=rl_result.valid_slices,
                recipe=self._p.recipe,
            )

            _write_to_buffer(self._buffer, restored_np)
            self.progress.emit(result.iterations, self._rl_config.num_iter)
            self.status.emit(
                f"Done after {result.iterations} iterations"
            )
            log.info(
                "RL done: iterations=%d final_loss=%s restored=%s",
                result.iterations,
                f"{result.loss_history[-1]:.6g}" if result.loss_history else "n/a",
                tuple(result.restored.shape),
            )
        except Exception as exc:
            log.exception("RL worker failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._cancel:
            self.cancelled.emit()
        else:
            self.finished.emit(result)


# --------------------------------------------------------------------------- #
# Helpers

def _write_to_buffer(buffer, arr: np.ndarray, *, log_stats: bool = True) -> None:
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count or inf_count:
        log.error(
            "result contains %d NaN and %d Inf values out of %d — "
            "the solver diverged. Common causes: PSF / data spacing "
            "mismatch, negative pixel values in the data, or a PSF that "
            "is not at the same fine-grid sampling as the recipe expects.",
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
        buffer[0, 0, 0, :, :] = arr
    elif arr.ndim == 3:
        target = (buffer.shape[1], buffer.shape[3], buffer.shape[4])
        if arr.shape != target:
            log.error("buffer/result shape mismatch: result=%s buffer=%s",
                      arr.shape, buffer.shape)
        buffer[0, :, 0, :, :] = arr
    else:
        raise ValueError(f"unexpected hidden-space ndim {arr.ndim}")
