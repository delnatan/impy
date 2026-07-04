"""Dialog-state -> deconlib Richardson-Lucy inputs.

`RLDialogState` is the flat dataclass the Richardson-Lucy dialog reads its
widgets into. Shape math and `prepare_inputs` are generic across every
deconvolution engine and live in `decon_common` -- re-exported here so
`richardson_lucy_dialog.py`/`richardson_lucy_worker.py` have a single
import site, mirroring `decon_nlcg.py`'s own re-export of the same names.
"""

from __future__ import annotations

from dataclasses import dataclass

from .decon_common import (  # noqa: F401 (re-exported for `drl.*` call sites)
    PreparedInputs,
    compact_psf_shape_for_data,
    estimate_tile_plan,
    log,
    output_5d_shape,
    output_shape,
    padded_shape,
    per_axis_zoom,
    prepare_inputs,
    valid_slices,
    visible_shape,
)


@dataclass(frozen=True)
class RLDialogState:
    # Forward model — independent lateral (XY) / axial (Z) zoom factors.
    # >= 1.0; fractional values use deconlib's FractionalAreaDownsample.
    zoom_xy: float = 1.0
    zoom_z: float = 1.0

    # Richardson-Lucy solver knobs (see
    # deconlib.deconvolution.richardson_lucy_with_operator). RL has no
    # regularizer in deconlib -- early stopping (num_iter) is the only
    # control over noise amplification.
    num_iter: int = 100
    background: float = 0.0
    eval_interval: int = 5
    verbose: bool = False

    # Output region (single-volume only; tiled output is always cropped).
    crop_to_visible: bool = True

    # Tiling (large fields; see deconlib.deconvolution.process_tiles).
    tiled: bool = False
    tile_size: int = 256
    guard_px: int = 0                       # 0 -> deconlib default (half PSF width)
    min_z_slices: int = 48
