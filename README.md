# Nonparanormal Frechet regression

Implementations for the paper:

*Fr\'echet regression of multivariate distributions with nonparanormal transport* (2026+)  
by Junyoung Park and Irina Gaynanova

If you have any questions, please feel free to reach out to: junyoup@umich.edu

## Repository Layout

- `data/`
  - AI-READI dataset preprocessing scripts and notebook
- `experiments/real/`
  - AI-READI CGM data experiments
- `experiments/simulation/`
  - Simulations experiments
- `functions/`
  - method implementations
- `results/`
  - figures, tables, and simulation summaries

## Requirements

- Python 3.10+ with:
  - `numpy>=1.26`, `pandas>=2.2`, `scipy>=1.14`, `matplotlib>=3.9`, `seaborn>=0.13`, `joblib>=1.4`, `numba>=0.63`, `rpy2>=3.5`, `pot>=0.9`
- R with `latentcor` and `fastfrechet`
- for the real-data workflow, an external AI-READI export placed in `../dataset/`

If R is installed in a non-standard location, set `R_HOME` before running code that imports `functions.regression`.

Minimal installation guide:

```powershell
pip install "numpy>=1.26" "pandas>=2.2" "scipy>=1.14" "matplotlib>=3.9" "seaborn>=0.13" "joblib>=1.4" "numba>=0.63" "rpy2>=3.5" "pot>=0.9"
```

```r
install.packages("latentcor")
install.packages("devtools")
devtools::install_github("https://github.com/alexandercoulter/fastfrechet", build_vignettes = TRUE)
```

Reference links:

- `rpy2`: [documentation](https://rpy2.github.io/doc/latest/html/overview.html), [PyPI package](https://pypi.org/project/rpy2/)
- `latentcor`: [package page](https://cran.r-project.org/package=latentcor)
- `fastfrechet`: [repository](https://github.com/alexandercoulter/fastfrechet)

## Minimal reproducibility guide

CGM (AI-READI) experiments:
Data is available from the study website upon request. See [data/Processing_guide.md](./data/Processing_guide.md) for details.
1. Put the downloaded AI-READI data in `../dataset/`.
2. Run `python data/processing_ai_readi.py`.
3. Run `data/ai_readi_window_processing.ipynb`.
4. Run `experiments/real/ai_readi_regresseion.ipynb` (regression with HbA1c as predictor)
5. Run `experiments/real/ai_readi_permutation.ipynb` or `python experiments/real/ai_permutation.py` (permutation test; recommended to use HPC clusters for parallel programming)

Simulations (recommended to use HPC clusters for parallel programming):

1. Run `python experiments/simulation/Simulation.py` (main simulation)
2. Run `python experiments/simulation/simul_wasserstein_eval.py` (supplementary; very slow due to the Wasserstein distance computation)
3. Run `experiments/simulation/Simulation_plots.ipynb` (results figure generation)

## Main Outputs
- real-data figures:
  - `results/real/figs/`
- real-data permutation results:
  - `results/real/R2/`
  - `results/tables/R2_adjusted_pvalues.tex`
- simulation summaries and figures:
  - `results/simulations/simulation_results.csv`
  - `results/simulations/simulation_results_wasserstein_*.csv`
  - `results/simulations/*.pdf`

## Detailed Guides

- AI-READI dataset preprocessing: [data/Processing_guide.md](./data/Processing_guide.md)
- CGM-experiments reproducibility: [experiments/real/Reproducibility_guide.md](./experiments/real/Reproducibility_guide.md)
- simulation reproducibility: [experiments/simulation/Reproducibility_guide.md](./experiments/simulation/Reproducibility_guide.md)
