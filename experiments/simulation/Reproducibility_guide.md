# Simulation Reproducibility Guide

This directory contains the simulation workflows used for the paper.

## Steps

1. Run the main simulation benchmark.

   ```powershell
   python experiments/simulation/Simulation.py
   ```

   Main outputs:
   - `results/simulations/simulation_results.csv`
   - `results/simulations/boxplot_combined.pdf`
   - `results/simulations/fitted_objects/`

2. Run the Wasserstein evaluation.

   ```powershell
   python experiments/simulation/simul_wasserstein_eval.py
   ```

   Main outputs:
   - `results/simulations/simulation_results_wasserstein_*.csv`

   Optional environment variables:
   - `N_VALUES`
   - `D_VALUES`
   - `N0_VALUES`
   - `TYPES`

3. Generate the simulation figures.

   Run:
   - `experiments/simulation/Simulation_plots.ipynb`

   Main outputs:
   - `results/simulations/correlation_visualization_total.pdf`
   - `results/simulations/boxplot_Wasserstein.pdf`

## Notes

- `Simulation.py` is the main simulation entry point.
- `simul_wasserstein_eval.py` consumes fitted objects written by `Simulation.py`.
- `Simulation_plots.ipynb` is the figure notebook used for the paper.
- `results/simulations/fitted_objects/` is a regeneration artifact and is not part of the publication-facing committed outputs.
