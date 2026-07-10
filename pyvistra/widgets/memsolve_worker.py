"""Background worker for MaxEnt (memsolve) deconvolution.

A thin Qt shim around the `mem` package's MAP solver (see
`decon_mem.build_mem_problem` for how the operator chain is assembled -- it
mirrors `deconlib/scripts/memsolve_gaussian_icf.py`).

`mem.run_inference`/`mem.run_map` drive their MAP loop internally with no
per-iteration callback hook (unlike deconlib's NLCG/RL, which accept one).
For a live preview, this worker instead drives the loop itself with `mem`'s
public `MaxEntProblem` / `init_state` / `step` / `finalize` trio --
`mem.maxent`'s module docstring names exactly this trio as the supported
entry point for callers that need to interleave custom per-iteration work
with a MAP solve, so this stays on public API. `step()` only returns
reduced-space coordinates plus scalar diagnostics (`MaxEntStepRecord`), not
the hidden/visible iterate itself, so recovering a visible-space image at a
given iterate means calling `finalize()` on that state -- which redoes the
curvature/evidence diagnostics unless the state is already converged. That
makes each preview roughly as expensive as one extra outer iteration, so
previews are throttled by `MemsolveDialogState.eval_interval` (mirroring
`RLDialogState.eval_interval`) rather than taken every iteration.

Tiled (large-field) runs use a different, simpler path (`_run_tiled`):
mirroring `deconlib/scripts/tiled_memsolve_demo.py`, the forward
model/ICF/prior are built once for the shared tile shape, and each tile is
solved with a single `mem.run_map` call (no per-iteration preview inside a
tile -- the live buffer is updated once per tile, matching
`RLDeconvolutionWorker._run_tiled`'s per-tile-not-per-iteration
granularity). Cancellation there is only checked between tiles.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QObject, Signal

from .decon_common import PreparedInputs, log, raise_if_nonfinite, write_to_buffer
from .decon_mem import MemProblemInputs, MemsolveDialogState

# Padding-region prior floor for tiled runs' shared flat prior -- see
# `_run_tiled`'s docstring and `decon_mem.py`'s module docstring for why
# tiled runs use a flat prior rather than the single-volume RCt(y) one.
_TILED_PADDING_PRIOR_VALUE = 1e-3


class MemsolveDeconvolutionWorker(QObject):
    """Run a MaxEnt (memsolve) MAP solve off the GUI thread."""

    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)        # mem.MaxEntResult (single) or None (tiled)
    cancelled = Signal()
    error = Signal(str)

    def __init__(
        self,
        *,
        state: MemsolveDialogState,
        buffer,                      # ImageBuffer for live preview / final write
        output_channel: int = 0,
        output_frame: int = 0,
        problem_inputs: MemProblemInputs | None = None,   # single-volume runs
        prepared: PreparedInputs | None = None,            # tiled runs
    ):
        super().__init__()
        self._pi = problem_inputs
        self._prepared = prepared
        self._state = state
        self._buffer = buffer
        self._output_channel = max(0, int(output_channel))
        self._output_frame = max(0, int(output_frame))
        self._cancel = False

    def cancel(self) -> None:
        # Checked at each outer-iteration boundary (single-volume) or tile
        # boundary (tiled), so this stops promptly rather than running the
        # solve to completion.
        self._cancel = True

    def run(self) -> None:
        try:
            if self._state.tiled:
                self._run_tiled()
            else:
                self._run_single()
        except Exception as exc:
            log.exception("Memsolve worker failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._cancel:
            self.cancelled.emit()

    def _write_preview(self, h) -> None:
        pi, s = self._pi, self._state
        preview = np.asarray(pi.C(h), dtype=np.float32)
        raise_if_nonfinite(preview, "MEM iterate")
        if s.crop_to_visible:
            preview = preview[pi.valid_slices]
        write_to_buffer(
            self._buffer, preview, channel=self._output_channel,
            t=self._output_frame, log_stats=False,
        )

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
            likelihood=s.likelihood,
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

        if self._cancel:
            return

        config = mem.MaxEntConfig(
            max_iter=s.max_iter,
            tol_omega=s.tol_omega,
            aim=s.aim,
            rate=s.rate,
            omega_mode=s.omega_mode,
            cg_epsilon=s.cg_epsilon,
            cg_max_steps=s.cg_max_steps,
            n_probe_g=s.n_probe_g,
            seed=s.seed,
        )

        # Combined RC/RCt as R/Rt with C/Ct left at their identity default --
        # mirrors mem.inference's own combined-operator path, so this matches
        # what mem.run_inference would have built for the same problem.
        mem_problem = mem.MaxEntProblem.data(
            pi.y, pi.prior, pi.RC, pi.RCt, likelihood=s.likelihood,
        )

        self.status.emit("Running MEM solver…")
        self.progress.emit(0, config.max_iter)

        eval_interval = max(1, int(s.eval_interval))
        state = mem.init_state(mem_problem, config)
        result = None

        for it in range(config.max_iter):
            if self._cancel:
                break
            state = mem.step(mem_problem, state, config)

            if state.converged or it % eval_interval == 0 or it == config.max_iter - 1:
                row = state.last_row
                result = mem.finalize(mem_problem, state, config)
                self._write_preview(result.h)
                self.progress.emit(it + 1, config.max_iter)
                if s.verbose:
                    self.status.emit(
                        f"MEM: iter {it + 1}/{config.max_iter}  "
                        f"chi2={row.chi2:.4g}  omega={row.omega:.4g}  "
                        f"alpha={row.alpha_curr:.4g}"
                    )
                else:
                    self.status.emit(f"MEM: iter {it + 1}/{config.max_iter}")

            if state.converged:
                break

        if self._cancel:
            return

        # config.max_iter == 0 is the only way the loop above never ran.
        if result is None:
            result = mem.finalize(mem_problem, state, config)

        f_map = np.asarray(pi.C(result.h), dtype=np.float32)
        raise_if_nonfinite(f_map, "MEM result")
        if s.crop_to_visible:
            f_map = f_map[pi.valid_slices]
        write_to_buffer(
            self._buffer, f_map, channel=self._output_channel,
            t=self._output_frame,
        )

        # result.iterations reflects the caller-supplied trace (empty here,
        # since we don't accumulate one), not the outer-iteration count --
        # use state.iteration for that instead.
        self.status.emit(
            f"Done after {state.iteration} iterations "
            f"(converged={result.converged}, omega={result.omega:.4g})"
        )
        log.info(
            "MEM done: iterations=%d converged=%s omega=%.4g chi2=%.4g "
            "entropy=%.4g log_evidence=%.4g restored=%s",
            state.iteration, result.converged, result.omega, result.chi2,
            result.entropy, result.log_evidence, tuple(f_map.shape),
        )
        self.finished.emit(result)

    # ------------------------------------------------------------------ #
    # Tiled path

    def _run_tiled(self) -> None:
        import mem
        from deconlib.deconvolution import (
            GaussianICF,
            as_numpy_op,
            make_forward_model,
            plan_tiles,
        )

        p, s = self._prepared, self._state
        if self._cancel:
            return

        guard = s.guard_px if s.guard_px > 0 else max(p.psf.shape[-2:]) // 2
        plan = plan_tiles(
            p.y.shape, p.zoom, guard=guard,
            tile_size=s.tile_size, min_z_slices=s.min_z_slices,
        )
        n_tiles = len(plan.tiles)
        log.info(
            "MEM tiled run: y=%s psf=%s zoom=%s tiles=%d tile_shape=%s "
            "guard=%d icf_gamma=%g max_iter=%d",
            p.y.shape, p.psf.shape, p.zoom, n_tiles, plan.tile_shape,
            guard, s.icf_gamma_um, s.max_iter,
        )

        # Built once -- every tile reads a window of the same shape, so the
        # forward model, ICF, and combined operators are shared.
        model = make_forward_model(p.psf, plan.tile_shape, p.zoom)
        R, Rt = as_numpy_op(model.op)
        icf = GaussianICF(
            shape=model.padded_shape,
            sigmas=(float(s.icf_gamma_um),) * len(plan.tile_shape),
            spacings=p.voxel_spacing,
            normalize=True,
        )
        C, Ct = as_numpy_op(icf)  # C == Ct (self-adjoint)

        def RC(h):
            return R(C(h))

        def RCt(u):
            return Ct(Rt(u))

        # Global flat prior, valid region only -- shared across every tile
        # rather than RCt(y_tile) (see decon_mem.py's module docstring: a
        # per-tile adjoint prior would give each tile a different absolute
        # baseline and show up as seams once stitched).
        prior_value = float(np.mean(p.y)) / float(np.prod(p.zoom))
        tile_prior = np.full(model.padded_shape, _TILED_PADDING_PRIOR_VALUE, dtype=np.float32)
        tile_prior[model.valid_slices] = prior_value
        log.info(
            "MEM tiled prior: padded_shape=%s prior_value=%.6g",
            model.padded_shape, prior_value,
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
                seed=s.seed,
            ),
            posterior=None,
        )

        output = np.zeros(plan.visible_shape, dtype=np.float32)

        self.status.emit(f"Running tiled MEM solver ({n_tiles} tiles)…")
        self.progress.emit(0, n_tiles)

        for i, spec in enumerate(plan.tiles):
            if self._cancel:
                break
            data_tile = np.asarray(p.y[spec.read], dtype=np.float32)

            problem = mem.LinearInverseProblem(
                y=data_tile,
                prior=tile_prior,
                likelihood=s.likelihood,
                R=R, Rt=Rt, C=C, Ct=Ct, RC=RC, RCt=RCt,
                name=f"tile_{spec.index}",
            )
            if i == 0:
                # One-time adjoint sanity check on the shared operator.
                validation = mem.validate_problem(problem, adjoint_rtol=1e-3)
                log.info(
                    "MEM tile problem validated: adjoint_rel_error=%.2e",
                    validation.adjoint_rel_error,
                )

            estimate = mem.run_map(problem, config)
            mem_result = estimate.result

            visible_result = np.asarray(estimate.f, dtype=np.float32)[model.valid_slices]
            crop = tuple(
                slice(sl.start, min(sl.stop, visible_result.shape[j]))
                for j, sl in enumerate(spec.crop)
            )
            core = visible_result[crop]
            write = tuple(
                slice(sl.start, sl.start + core.shape[j])
                for j, sl in enumerate(spec.write)
            )
            output[write] = core

            write_to_buffer(
                self._buffer, output, channel=self._output_channel,
                t=self._output_frame, log_stats=False,
            )
            self.progress.emit(i + 1, n_tiles)
            self.status.emit(
                f"MEM: tile {i + 1}/{n_tiles}  iters {mem_result.iterations}  "
                f"converged {mem_result.converged}  omega {mem_result.omega:.4g}"
            )

        if self._cancel:
            return

        raise_if_nonfinite(output, "MEM tiled result")
        write_to_buffer(
            self._buffer, output, channel=self._output_channel,
            t=self._output_frame,
        )
        self.status.emit(f"Done — {n_tiles} tiles")
        log.info("MEM tiled done: tiles=%d output=%s", n_tiles, tuple(output.shape))
        self.finished.emit(None)
