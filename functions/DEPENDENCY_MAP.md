# `functions/` Dependency Map

This file records the actual module relationships in `functions/` so the package can be cleaned up without breaking the experiment workflows.

## Runtime Requirements

- Python: `numpy`, `pandas`, `scipy`, `numba`, `joblib`, `matplotlib`, `seaborn`
- R bridge: `rpy2`
- Required R packages: `latentcor`, `fastfrechet`

Important behavior:
Importing `functions.regression` immediately calls `setup_r_environment()` and imports the R packages. Anything that imports `functions.regression` therefore has an R runtime dependency at import time, not just at function-call time.
Because `functions.__init__` re-exports regression helpers, importing `functions` also inherits that import-time R dependency today.

## Internal Dependency Graph

```text
dist.py
corr_barycenter.py -> dist.py
r_utils.py
regression.py -> corr_barycenter.py, dist.py, r_utils.py
plot_helper.py -> no internal imports
__init__.py -> dist.py, corr_barycenter.py, regression.py
```

## Module Inventory

| Module | Primary responsibility | Internal imports | Main downstream users |
|---|---|---|---|
| `dist.py` | Bures-Wasserstein distance and PSD matrix operations | None | `corr_barycenter.py`, `regression.py`, `experiments/simulation/gaussian_frechet.py` |
| `corr_barycenter.py` | BW projection, barycenter objective, Riemannian correlation barycenter | `dist.py` | `regression.py`, `experiments/simulation/gaussian_frechet.py`, `functions.__init__` |
| `r_utils.py` | R discovery and `rpy2` package import helpers | None | `regression.py` |
| `regression.py` | NPT Fr\'echet regression core, marginal/correlation regression, high-level wrapper | `corr_barycenter.py`, `dist.py`, `r_utils.py` | `experiments/simulation/Simulation.py`, `experiments/real/ai_permutation.py`, notebooks, `functions.__init__` |
| `plot_helper.py` | General plotting helpers for fitted distributions and latent correlation trends | None | AI-READI notebooks |
| `__init__.py` | Re-exported public API for the package | `dist.py`, `corr_barycenter.py`, `regression.py` | `experiments/simulation/simul_generation.py`, `experiments/simulation/gaussian_frechet.py` |

## Public Surface Used Outside `functions/`

### Imported directly from modules

- `functions.regression.npt_frechetreg`
- `functions.regression.multivariate_frechet_regression`
- `functions.plot_helper.plot_multivar_distributions`
- `functions.plot_helper.plot_correlation_trends`

### Imported through `functions.__init__`

- `functions.Bures_Wasserstein`
- `functions.BW_projection`
- `functions.riemannian_corr_barycenter`
- `functions.corr_frechet`
- `functions.global_frechet_weights`

## R-Dependency Boundary Inside `regression.py`

These functions currently need the R bridge at runtime:

- `marginal_frechet()`
- `get_latent_cor()`
- `npt_frechetreg()` through the two functions above
- `multivariate_frechet_regression()` through the two functions above

These functions are structurally pure Python and are the main candidates to become import-safe after lazy R initialization:

- `corr_frechet()`
- `global_frechet_weights()`
- `_validate_frechet_inputs()`
- `_process_bounds_dict()`

## Reverse Dependency View

### Real-data workflow

- `ai_readi_window_processing.ipynb`
  - uses `data.window_processing` for window generation and Fr'echet-preparation helpers
- `ai_readi_regresseion.ipynb`
  - uses `data.window_processing` plus `multivariate_frechet_regression` and `plot_helper.py`
- `ai_readi_permutation.ipynb`
  - uses `data.window_processing` plus `multivariate_frechet_regression` and `plot_helper.py`
- `ai_permutation.py`
  - uses `prepare_frechet_data` and `multivariate_frechet_regression`
- `window_processing.py`
  - imports `get_latent_cor`, so its current import path also inherits the eager R setup in `functions.regression`

### Simulation workflow

- `Simulation.py`
  - uses `npt_frechetreg`
- `simul_generation.py`
  - uses package export `BW_projection`
- `gaussian_frechet.py`
  - uses package exports `riemannian_corr_barycenter`, `BW_projection`, `Bures_Wasserstein`, `global_frechet_weights`, `corr_frechet`

## Packaging Notes

- Stage-1 dead-code cleanup is complete; the public surface listed above is the remaining live surface scanned in this repository.
- The AI-READI window-processing layer lives under `data/window_processing.py`; it is intentionally outside the `functions/` package.
- That helper module still mixes three concerns:
  - window feature extraction
  - plotting utilities
  - Fr\'echet-regression data preparation
- `plot_helper.py` remains a general plotting helper module; its current live users in this repository are notebook workflows rather than script entry points.
- `plot_multivar_distributions()` still supports auto-sampling from fitted marginals, but it now keeps that sampling local instead of mutating the caller's DataFrame.
- The package export surface is intentionally narrow: `__init__.py` does not re-export `window_processing.py` or `plot_helper.py`.
