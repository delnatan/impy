"""Background worker for NLCG-based PSF distillation.

A thin Qt shim around deconlib's NLCG engine
(`deconlib.deconvolution.nlcg_with_operator`), applied to the swapped
PSF-distillation operator from
``deconlib/scripts/psf_distillation_nlcg_demo.py``: the fixed bead comb
(``PreparedDistillInputs.object_comb_eff``) is the convolution kernel, and
the unknown being reconstructed is the PSF. The dialog hands this worker a
fully-prepared :class:`~.psf_distillation_nlcg.PreparedDistillInputs` plus
the dialog state and an optional regularizer; the worker runs the solve in
a QThread and emits progress / status / finished / error signals, writing
live PSF previews into the destination `ImageBuffer` as it goes.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QObject, Signal

from .psf_distillation_nlcg import PreparedDistillInputs, PSFDistillDialogState, log


def _corner_axis_chunks(small_n: int, padded_n: int):
    """Per-axis (small_slice, padded_slice) pairs for a corner-origin embed.

    Mirrors ``deconlib.utils.padding.pad_corner_origin_kernel``'s own split:
    positive offsets ``0..n_positive-1`` map to the low end of the padded
    axis; negative offsets (the remainder) map to its high end. Precomputed
    once per axis so :func:`_corner_embed`/:func:`_corner_extract` share an
    identical (and therefore exactly adjoint) chunk layout.
    """
    n_positive = (small_n + 1) // 2
    chunks = [(slice(0, n_positive), slice(0, n_positive))]
    if n_positive < small_n:
        n_negative = small_n - n_positive
        chunks.append(
            (slice(n_positive, small_n), slice(padded_n - n_negative, padded_n))
        )
    return chunks


def _apply_corner_chunks(x, out_shape, chunks, small_to_padded: bool):
    """Scatter (``small_to_padded``) or gather (else) through corner chunks.

    Gather is the exact mathematical adjoint of scatter here -- both walk
    the identical set of disjoint (small_slice, padded_slice) index pairs,
    just swapping which side is read and which is written.
    """
    import mlx.core as mx

    out = mx.zeros(out_shape, dtype=x.dtype)
    ndim = len(out_shape)

    def recurse(axis, idx_small, idx_padded):
        if axis == ndim:
            if small_to_padded:
                out[tuple(idx_padded)] = x[tuple(idx_small)]
            else:
                out[tuple(idx_small)] = x[tuple(idx_padded)]
            return
        for small_slice, padded_slice in chunks[axis]:
            recurse(axis + 1, idx_small + [small_slice], idx_padded + [padded_slice])

    recurse(0, [], [])
    return out


def _corner_embed(h_small, padded_shape, chunks):
    return _apply_corner_chunks(h_small, padded_shape, chunks, small_to_padded=True)


def _corner_extract(h_padded, small_shape, chunks):
    return _apply_corner_chunks(h_padded, small_shape, chunks, small_to_padded=False)


class _PsfInverseOperator:
    """Forward operator for the PSF sub-problem.

    The bead comb is the fixed kernel; the PSF is the unknown reconstruction.
    NLCG optimizes directly over the *compact* ``psf_shape`` domain (corner-
    origin, matching the initial PSF's own convention) -- the PSF has known,
    bounded support by construction, so there is nothing to gain (and
    accuracy to lose, see below) from letting the solver explore a padded
    margin the way deconvolution's object domain does.

    The padded FFT canvas (``padded_shape``, sized for wrap-free linear
    convolution against the bead comb) exists purely as an internal
    implementation detail of ``forward``/``adjoint``: each call embeds the
    compact PSF into it via :func:`_corner_embed` and extracts back via its
    exact adjoint :func:`_corner_extract`. Domain: ``psf_shape``. Range: the
    bead image (``out_shape``, no zero-padding) -- ``embed.adjoint``
    corner-crops the padded canvas back to the real detector footprint,
    keeping every ``nlcg_with_operator`` data pixel a genuine observation.
    """

    def __init__(self, conv, embed, psf_shape, padded_shape, out_shape):
        self._conv = conv
        self._embed = embed
        self._padded_shape = tuple(padded_shape)
        self.operator_norm_sq = conv.operator_norm_sq
        self.in_shape = tuple(psf_shape)
        self.out_shape = tuple(out_shape)
        self._chunks = [
            _corner_axis_chunks(n_small, n_padded)
            for n_small, n_padded in zip(self.in_shape, self._padded_shape)
        ]

    def forward(self, h_small):
        h_padded = _corner_embed(h_small, self._padded_shape, self._chunks)
        return self._embed.adjoint(self._conv.forward(h_padded))

    def adjoint(self, residual):
        r_padded = self._embed.forward(residual)
        g_padded = self._conv.adjoint(r_padded)
        return _corner_extract(g_padded, self.in_shape, self._chunks)

    def __call__(self, h_small):
        return self.forward(h_small)


class PSFDistillationWorker(QObject):
    """Run NLCG PSF distillation off the GUI thread."""

    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(object)        # NLCGResult
    cancelled = Signal()
    error = Signal(str)

    def __init__(
        self,
        *,
        prepared: PreparedDistillInputs,
        state: PSFDistillDialogState,
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
            self._run()
        except Exception as exc:
            log.exception("PSF distillation worker failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._cancel:
            self.cancelled.emit()

    def _run(self) -> None:
        import mlx.core as mx
        from deconlib.deconvolution import FFTConvolver, Pad, fast_padded_shape, nlcg_with_operator
        from deconlib.psf.distillation import project_psf

        p, s = self._p, self._state
        data_shape = p.y.shape
        psf_shape = p.init_psf.shape

        log.info(
            "PSF distillation run: y=%s init_psf=%s beads=%d num_iter=%d "
            "background=%g regularizer=%s(weight=%g)",
            data_shape, psf_shape, len(p.positions), s.num_iter, s.background,
            s.regularizer_kind, s.reg_weight,
        )

        # padded_shape is only the internal FFT canvas for wrap-free linear
        # convolution against the (large) bead comb -- NLCG itself optimizes
        # over the compact psf_shape domain (see _PsfInverseOperator), so no
        # padding margin ever becomes a free/modeled quantity.
        self.status.emit("Building swapped forward model…")
        padded_shape = fast_padded_shape(data_shape, psf_shape)
        embed = Pad(tuple((0, pp - n) for pp, n in zip(padded_shape, data_shape)))
        object_padded = embed.forward(mx.array(p.object_comb_eff))
        object_conv = FFTConvolver(object_padded, normalize=False)
        operator = _PsfInverseOperator(object_conv, embed, psf_shape, padded_shape, data_shape)

        # Compact, corner-origin, nonnegative -- no embedding needed, this is
        # already exactly the domain nlcg_with_operator solves over.
        init = mx.array(p.init_psf.astype(np.float32))

        def callback(k: int, f) -> bool:
            if k % max(1, s.eval_interval) == 0:
                raw = np.asarray(f, dtype=np.float32)
                _raise_if_nonfinite(raw, "NLCG estimate")
                preview = project_psf(raw)
                if s.center_output:
                    preview = np.fft.ifftshift(preview)
                _write_to_buffer(
                    self._buffer, preview, channel=self._output_channel,
                    log_stats=False,
                )
                self.progress.emit(k + 1, s.num_iter)
                if s.verbose:
                    pred = (
                        np.asarray(operator.forward(f), dtype=np.float32) + s.background
                    )
                    loss = _mean_poisson_i_divergence(p.y, pred)
                    self.status.emit(
                        f"NLCG: iter {k + 1}/{s.num_iter}  mean I-div={loss:.4g}"
                    )
                else:
                    self.status.emit(f"NLCG: iter {k + 1}/{s.num_iter}")
            return bool(self._cancel)

        self.status.emit(f"Running NLCG ({len(p.positions)} beads)…")
        self.progress.emit(0, s.num_iter)
        restart_interval = s.restart_interval if s.restart_interval > 0 else None

        result = nlcg_with_operator(
            observed=mx.array(p.y),
            blur_op=operator,
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

        raw = np.asarray(result.restored, dtype=np.float32)
        _raise_if_nonfinite(raw, "NLCG result")
        psf = project_psf(raw)
        if s.center_output:
            psf = np.fft.ifftshift(psf)
        _write_to_buffer(self._buffer, psf, channel=self._output_channel)
        self.progress.emit(result.iterations, s.num_iter)
        self.status.emit(
            f"Done after {result.iterations} iterations ({len(p.positions)} beads)"
        )
        log.info(
            "PSF distillation done: iterations=%d converged=%s final_loss=%s psf=%s",
            result.iterations, result.converged,
            f"{result.loss_history[-1]:.6g}" if result.loss_history else "n/a",
            tuple(psf.shape),
        )
        self.finished.emit(result)


# --------------------------------------------------------------------------- #
# Helpers (mirrors deconvolution_worker.py)

def _mean_poisson_i_divergence(
    observed: np.ndarray, model: np.ndarray, eps: float = 1e-6
) -> float:
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
            "result contains %d NaN and %d Inf values out of %d — the "
            "solver diverged.", nan_count, inf_count, arr.size,
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
