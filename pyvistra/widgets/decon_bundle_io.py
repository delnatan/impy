"""Thin wrappers around deconlib bundle I/O.

`save_bundle` dispatches on result type (`WorkflowResult` →
`save_memsolve_bundle`, `RichardsonLucyResult` → `save_richardson_lucy_bundle`).
`open_bundle` dispatches on disk via `peek_bundle_algorithm`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def save_bundle(
    path,
    result,
    *,
    psf,
    optics,
    geometry,
    y: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
    name: str = "",
) -> None:
    """Save either a `WorkflowResult` or a `RichardsonLucyResult` bundle."""
    from deconlib import (
        RichardsonLucyResult,
        WorkflowResult,
        save_memsolve_bundle,
        save_richardson_lucy_bundle,
    )

    if isinstance(result, WorkflowResult):
        save_memsolve_bundle(
            path,
            result.final,
            optics=optics,
            geometry=geometry,
            recipe=result.chosen_recipe,
            psf=psf,
            metadata=metadata,
            name=name,
        )
    elif isinstance(result, RichardsonLucyResult):
        save_richardson_lucy_bundle(
            path,
            result,
            y=y,
            optics=optics,
            geometry=geometry,
            recipe=result.recipe,
            psf=psf,
            prior=prior,
            sigma=sigma,
            metadata=metadata,
            name=name,
        )
    else:
        raise TypeError(f"unsupported result type {type(result).__name__}")


def open_bundle(path):
    """Open a `.decon.h5` and return the algorithm-specific bundle object."""
    from deconlib import (
        load_memsolve_bundle,
        load_richardson_lucy_bundle,
        peek_bundle_algorithm,
    )

    algorithm = peek_bundle_algorithm(path)
    if algorithm == "memsolve_mem":
        return load_memsolve_bundle(path)
    if algorithm == "richardson_lucy":
        return load_richardson_lucy_bundle(path)
    raise ValueError(f"unknown bundle algorithm {algorithm!r}")
