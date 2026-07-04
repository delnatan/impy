"""Background worker for MaxEnt (memsolve) deconvolution.

A thin Qt shim around the `mem` package's `LinearInverseProblem` /
`validate_problem` / `run_inference` (see `decon_mem.build_mem_problem` for
how the operator chain is assembled -- it mirrors
`deconlib/scripts/memsolve_gaussian_icf.py`).

`mem.run_inference` drives its MAP loop internally with no per-iteration
callback hook (unlike deconlib's NLCG/RL, which accept one) -- reimplementing
that loop here would mean depending on `mem.maxent`'s private step/state
API. Instead, `MaxEntConfig(print_outer=True)`'s one-line-per-iteration
diagnostic print is captured live via `contextlib.redirect_stdout` into a
small file-like `_StreamToSignal`, whose `.write()` emits each completed
line as a Qt `status` signal -- giving live per-iteration text feedback
without touching memsolve internals. There's no way to get an incremental
*image* preview this way, so the caller should treat the progress bar as
indeterminate and expect a single buffer write at the end.
"""

from __future__ import annotations

import contextlib

import numpy as np
from qtpy.QtCore import QObject, Signal

from .decon_common import log, raise_if_nonfinite, write_to_buffer
from .decon_mem import MemProblemInputs, MemsolveDialogState


class _StreamToSignal:
    """File-like object that emits each completed line as a Qt signal."""

    def __init__(self, signal):
        self._signal = signal
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._signal.emit(line)
        return len(text)

    def flush(self) -> None:
        pass


class MemsolveDeconvolutionWorker(QObject):
    """Run a MaxEnt (memsolve) MAP solve off the GUI thread."""

    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)        # mem.InferenceResult
    cancelled = Signal()
    error = Signal(str)

    def __init__(
        self,
        *,
        problem_inputs: MemProblemInputs,
        state: MemsolveDialogState,
        buffer,                      # ImageBuffer for the final write
        output_channel: int = 0,
    ):
        super().__init__()
        self._pi = problem_inputs
        self._state = state
        self._buffer = buffer
        self._output_channel = max(0, int(output_channel))
        self._cancel = False

    def cancel(self) -> None:
        # MEM's MAP solve is one blocking call with no cooperative
        # cancellation point -- honored only up to the point the solve
        # actually starts (matches the "Cancelling…" label the dialog
        # shows; the run still completes on the worker thread).
        self._cancel = True

    def run(self) -> None:
        try:
            self._run_single()
        except Exception as exc:
            log.exception("Memsolve worker failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._cancel:
            self.cancelled.emit()

    def _run_single(self) -> None:
        import mem

        pi, s = self._pi, self._state
        if self._cancel:
            return

        log.info(
            "MEM run: y=%s prior=%s zoom=%s icf_gamma=%g max_iter=%d",
            pi.y.shape, pi.prior.shape, pi.zoom, s.icf_gamma_um, s.max_iter,
        )
        self.status.emit("Validating problem…")

        problem = mem.LinearInverseProblem(
            y=pi.y,
            prior=pi.prior,
            likelihood="poisson",
            R=pi.R,
            Rt=pi.Rt,
            C=pi.C,
            Ct=pi.Ct,
            RC=pi.RC,
            RCt=pi.RCt,
            name="memsolve_gaussian_icf",
        )
        # Relaxed from the 1e-5 default: MLX's Metal/GPU matmul kernel
        # (FractionalAreaDownsample/Upsample) trades fp32 accumulation
        # precision for speed, so the adjoint check is precision-limited by
        # the backend, not by the operator's adjoint math (see
        # memsolve_gaussian_icf.py).
        validation = mem.validate_problem(problem, adjoint_rtol=1e-3)
        log.info(
            "MEM problem validated: n_hidden=%d n_visible=%d n_data=%d "
            "adjoint_rel_error=%.2e",
            validation.n_hidden, validation.n_visible, validation.n_data,
            validation.adjoint_rel_error,
        )

        config = mem.InferenceConfig(
            map_space="data",
            map_config=mem.MaxEntConfig(
                max_iter=s.max_iter,
                tol_omega=s.tol_omega,
                aim=s.aim,
                rate=s.rate,
                omega_mode=s.omega_mode,
                cg_epsilon=s.cg_epsilon,
                cg_max_steps=s.cg_max_steps,
                n_probe_g=s.n_probe_g,
                print_outer=True,
                print_inner_cg=False,
                seed=s.seed,
            ),
            posterior=None,
        )

        self.status.emit("Running MEM solver…")
        self.progress.emit(0, 0)  # indeterminate: no per-iteration callback

        with contextlib.redirect_stdout(_StreamToSignal(self.status)):
            result = mem.run_inference(problem, config)

        if self._cancel:
            return

        f_map = np.asarray(result.map.f, dtype=np.float32)
        raise_if_nonfinite(f_map, "MEM result")
        if s.crop_to_visible:
            f_map = f_map[pi.valid_slices]
        write_to_buffer(self._buffer, f_map, channel=self._output_channel)

        mem_result = result.map.result
        self.status.emit(
            f"Done after {mem_result.iterations} iterations "
            f"(converged={mem_result.converged})"
        )
        log.info(
            "MEM done: iterations=%d converged=%s chi2=%.4g entropy=%.4g "
            "log_evidence=%.4g restored=%s",
            mem_result.iterations, mem_result.converged, mem_result.chi2,
            mem_result.entropy, mem_result.log_evidence, tuple(f_map.shape),
        )
        self.finished.emit(result)
