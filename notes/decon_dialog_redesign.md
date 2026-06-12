# Deconvolution dialog redesign — recipe-driven via deconlib

Status: design sketch. Target: replace the hand-wired `DeconvolutionParams`
+ worker plumbing with a thin layer over deconlib's recipe + workflow API.

## Deconlib surface to consume

All of these are exported from `deconlib` top-level (verified by
`from deconlib import …`):

| Concept                                          | Source                                             |
| ------------------------------------------------ | -------------------------------------------------- |
| `ForwardRecipe`                                  | `deconlib/memsolve_io.py` (`class ForwardRecipe`)  |
| `BundleGeometry`                                 | `deconlib/memsolve_io.py`                          |
| Recipe registry (`RECIPE_REGISTRY`, builders)    | `deconlib/memsolve_io.py`                          |
| MEM workflow (`run_deconvolution_workflow`, `IcfSweep`) | `deconlib/workflow.py`                       |
| RL workflow (`run_richardson_lucy`, `RichardsonLucyConfig`) | `deconlib/workflow.py`                    |
| MEM bundle I/O (`save_memsolve_bundle`, `load_memsolve_bundle`) | `deconlib/memsolve_io.py`              |
| RL bundle I/O (`save_richardson_lucy_bundle`, `load_richardson_lucy_bundle`) | `deconlib/workflow.py`     |
| Algorithm dispatch (`peek_bundle_algorithm`)     | `deconlib/memsolve_io.py`                          |
| On-disk format spec                              | `deconlib/notes/memsolve_hdf5_spec.md`             |

The two built-in recipe kinds (`"fft_conv"`, `"super_res_idc"`) cover the
typical workflow you currently support. Adding a new kind is one
`@register_recipe("name")` away.

## What the dialog needs to produce

Three values per Start:

1. **`ForwardRecipe`** — describes the forward model.
2. **algorithm** — `"memsolve_mem"` or `"richardson_lucy"`.
3. **solver knobs** — `mem.MaxEntConfig` + optional `IcfSweep` + optional
   `mem.PosteriorConfig`, OR `RichardsonLucyConfig`.

Everything else (PSF, optics, y, prior, geometry) comes from the viewer
state and the chosen ROI.

## Mapping current form fields → new recipe / config

```
Current DeconvolutionParams (deconvolution_worker.py)         Where it goes
─────────────────────────────────────────────────────────────────────────
t, c, z_slice, y_slice, x_slice                               viewer/ROI (unchanged)
psf                                                           Psf object → save_*_bundle(psf=...)
likelihood, sigma                                             run_deconvolution_workflow(likelihood=…, sigma=…)
icf_sigma  (None ⇒ off; >0 ⇒ on)                              ForwardRecipe.icf = None  OR
                                                              {"kind":"gaussian","sigmas_um":(σ,σ[,σ])}
                                                              OR pass to IcfSweep.sigmas_um for a scan
upsampling (1 ⇒ standard, >1 ⇒ super-res)                     ForwardRecipe.kind = "fft_conv" if 1,
                                                              else "super_res_idc"
                                                              with super_res_factor=(u, u[, u])
show_visible                                                  display-side toggle (h vs f); no recipe impact
use_combined_rc                                               drop — built-in builders return both
                                                              R/Rt and C/Ct; mem decides.
max_iter, tol_omega, rate, omega_mode, cg_*, n_probe_g        → mem.MaxEntConfig(...)
map_space                                                     → mem.InferenceConfig(map_space=…)
preview_every                                                 callback cadence (unchanged)
```

`use_combined_rc` goes away — memsolve picks the right path from the
operator dict the builder returns. `show_visible` stays as a viewer-side
toggle (read `bundle.map.f` vs `bundle.map.h` post-run).

## Proposed widget structure

Two tabs survive (`Source & PSF`, `Solver`) with the following
reorganization:

### Tab 1 — Source & PSF (unchanged structure, sharper semantics)

* Time/channel/ROI selector — as today.
* PSF source — three radios:
  1. Theoretical (from active optics + ROI shape).
  2. From file (`.psf.h5`).
  3. Distilled (existing PSF dialog flow).
* Optics block (auto-filled when PSF carries an `Optics`; editable).

### Tab 2 — Solver (the part this redesign actually changes)

* **Algorithm** radio: MEM | Richardson-Lucy.
* **Forward model** group (drives `ForwardRecipe`):
  * Super-resolution factor spin (1 ⇒ `fft_conv`, ≥2 ⇒ `super_res_idc`).
  * Detector padding (only enabled when super-res ≥ 2).
* **ICF** group (only enabled when algorithm == MEM):
  * Mode: Off | Fixed σ | Sweep.
  * Fixed σ: single QDoubleSpinBox (μm).
  * Sweep: QLineEdit for comma-separated σ list + "Refine" checkbox.
* **MEM solver knobs** (only visible when algorithm == MEM):
  * `max_iter`, `tol_omega`, `rate`, `omega_mode`, `cg_*`, `n_probe_g`,
    `map_space` (same widgets as today).
* **RL knobs** (only visible when algorithm == RL):
  * `num_iter`, `background`, `eval_interval`, `return_region`.

A small helper turns the form into deconlib types:

```python
# pyvistra/widgets/decon_recipe.py (new, ~80 lines)

@dataclass(frozen=True)
class DecondialogState:
    algorithm: str                       # "memsolve_mem" | "richardson_lucy"
    super_res_factor: int                # 1 = no super-res
    detector_padding: int                # symmetric, per-axis; 0 disables
    icf_mode: str                        # "off" | "fixed" | "sweep"
    icf_sigma_um: float | None
    icf_sweep_sigmas_um: tuple[float, ...]
    icf_refine: bool
    # MEM
    max_iter: int = 60
    tol_omega: float = 0.05
    rate: float = 0.3
    omega_mode: str = "auto"
    cg_epsilon: float = 1e-2
    cg_max_steps: int = 120
    n_probe_g: int = 1
    map_space: str = "hidden"
    # RL
    rl_num_iter: int = 50
    rl_background: float = 0.0
    rl_eval_interval: int = 10
    rl_return_region: str = "full"
    # Posterior (MEM only)
    posterior_n_samples: int = 0
    posterior_seed: int = 0

def build_recipe(state, ndim: int) -> ForwardRecipe:
    factor = max(int(state.super_res_factor), 1)
    if factor == 1:
        kind = "fft_conv"
        srf = ()
    else:
        kind = "super_res_idc"
        srf = (factor,) * ndim
    pad = (state.detector_padding,) * ndim if state.detector_padding else ()
    icf = None
    if state.algorithm == "memsolve_mem" and state.icf_mode == "fixed" \
            and state.icf_sigma_um is not None:
        icf = {"kind": "gaussian", "sigmas_um": (state.icf_sigma_um,) * ndim}
    return ForwardRecipe(
        kind=kind, super_res_factor=srf, detector_padding=pad,
        psf_source="embedded", icf=icf,
    )

def build_mem_config(state) -> mem.MaxEntConfig: ...
def build_rl_config(state) -> RichardsonLucyConfig: ...
def build_icf_sweep(state) -> IcfSweep | None: ...
def build_posterior(state) -> mem.PosteriorConfig | None: ...
```

The dialog's `_start` becomes:

```python
def _start(self):
    state = self._read_dialog_state()
    recipe = build_recipe(state, ndim=self._ndim())
    if state.algorithm == "memsolve_mem":
        self._launch_mem(recipe, state)
    else:
        self._launch_rl(recipe, state)
```

## Worker simplification

`deconvolution_worker.py` becomes a thin Qt shim around the deconlib
driver. The whole "build R/Rt/C/Ct from params" body collapses into one
call:

```python
class MemDeconvolutionWorker(QObject):
    progress = Signal(int, int)
    status   = Signal(str)
    finished = Signal(object)   # WorkflowResult
    error    = Signal(str)

    def __init__(self, *, y, prior, sigma, psf, optics, geometry,
                 recipe, map_config, icf_sweep, posterior):
        ...

    def run(self):
        try:
            result = run_deconvolution_workflow(
                self.y, self.prior,
                base_recipe=self.recipe,
                psf=self.psf, optics=self.optics, geometry=self.geometry,
                sigma=self.sigma, likelihood="gaussian",
                icf_sweep=self.icf_sweep,
                map_config=self.map_config,
                posterior=self.posterior,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

class RLDeconvolutionWorker(QObject):
    ...
    def run(self):
        try:
            result = run_richardson_lucy(
                self.y,
                base_recipe=self.recipe,
                psf=self.psf, optics=self.optics, geometry=self.geometry,
                config=self.rl_config,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
```

Progress callback: memsolve currently doesn't emit per-iteration progress
through the workflow driver. Two options:
1. Drive a `mem.MaxEntState` step loop in the worker (more code, but
   per-iteration progress for the progress bar).
2. Emit "stage" progress instead: baseline / scan candidate i of N /
   refinement / final. Coarser but matches what `run_deconvolution_workflow`
   exposes today. **Recommended** for v1.

For RL, `richardson_lucy_with_operator` already takes a per-iteration
`callback`; the existing wiring works as-is.

## Save / open wiring

A small `BundleIO` helper next to `BufferProcessingRunner`:

```python
def save_bundle(path, workflow_or_rl_result, *, recipe, psf, optics,
                geometry, y, prior=None, sigma=None, **meta):
    if isinstance(workflow_or_rl_result, WorkflowResult):
        save_memsolve_bundle(
            path, workflow_or_rl_result.final,
            optics=optics, geometry=geometry,
            recipe=workflow_or_rl_result.chosen_recipe,
            psf=psf, **meta,
        )
    elif isinstance(workflow_or_rl_result, RichardsonLucyResult):
        save_richardson_lucy_bundle(
            path, workflow_or_rl_result,
            y=y, prior=prior, sigma=sigma,
            optics=optics, geometry=geometry,
            recipe=workflow_or_rl_result.recipe,
            psf=psf, **meta,
        )

def open_bundle(path):
    algorithm = peek_bundle_algorithm(path)
    if algorithm == "memsolve_mem":
        return load_memsolve_bundle(path)
    return load_richardson_lucy_bundle(path)
```

`open_bundle` returns a polymorphic object; pyvistra dispatches on type
to populate the dialog (re-rendering the recipe form) or render
read-only viewers (MAP, samples summaries, masks, ICF scan plot).

## Files to touch

| File                                                      | Action                                       |
| --------------------------------------------------------- | -------------------------------------------- |
| `pyvistra/widgets/deconvolution_dialog.py`                 | reorganize Solver tab; algorithm radio; per-mode visibility; replace `_start` payload |
| `pyvistra/widgets/deconvolution_worker.py`                 | split into `MemDeconvolutionWorker` + `RLDeconvolutionWorker` (each ~60 lines) |
| `pyvistra/widgets/decon_recipe.py` (new)                  | `DecondialogState`, `build_recipe`, `build_*_config`, `build_icf_sweep` |
| `pyvistra/widgets/decon_bundle_io.py` (new)               | `save_bundle`, `open_bundle` wrappers around deconlib I/O |
| `pyvistra/widgets/convergence_plot.py`                    | minor — trace columns are the same; add an "ICF scan" plot if you want to surface `WorkflowResult.scan` |
| `pyvistra/widgets/psf_dialog.py`                          | unchanged                                    |

## What this buys you

* The dialog form maps 1:1 to a `ForwardRecipe`. Adding a new forward
  model (e.g. 4Pi, structured illumination) means adding one recipe
  `kind` in deconlib + one row to the Super-res radio in the dialog.
* Bundles are self-resuming: opening any `.decon.h5` re-renders the same
  form (no per-app glue).
* RL gets the same form scaffolding for free — only the algorithm-specific
  knobs differ.
* The worker shrinks by ~80%; almost all the existing operator-wiring
  body lives in deconlib now.

## Open questions for pyvistra side

1. **Per-iteration progress for MEM.** Stage progress is the cheap path;
   per-iteration needs driving the `mem.step()` loop in the worker.
   Probably fine for v1 to keep coarse-grained progress.
2. **Showing posterior summaries.** `bundle.samples.hidden_std` is the
   natural overlay — render as a heatmap toggled in the layer panel.
3. **ICF scan plot.** `WorkflowResult.scan` rows have `sigma_um` and
   `log_evidence`. A small Qt plot with the chosen σ highlighted would
   make the "why did it pick this σ" question answerable at a glance.
4. **Bundle re-open into the dialog vs. viewer-only.** Two flows: "open
   to inspect" (no dialog) and "open and edit recipe to re-run" (dialog
   pre-populated from `bundle.recipe`). v1 can ship inspect-only.
