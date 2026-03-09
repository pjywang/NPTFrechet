# Simulation Reproducibility Guide

This directory contains the simulation workflows used for the paper.

## Steps

1. Run the main simulation script.

   ```powershell
   python experiments/simulation/Simulation.py
   ```
   Recommended to run on HPC clusters due to the high-volume of Monte Carlo experiments and settings.

   Main outputs:
   - `results/simulations/simulation_results.csv`
   - `results/simulations/boxplot_combined.pdf`
   - `results/simulations/fitted_objects/`
   
   $\quad$

2. Run the Wasserstein evaluation (supplementary; very slow due to Wasserstein evaluation).

   ```powershell
   python experiments/simulation/simul_wasserstein_eval.py
   ```

   Main outputs:
   - `results/simulations/simulation_results_wasserstein_*.csv`  

   $\quad$

3. Generate the simulation figures.
   Run:
   - `experiments/simulation/Simulation_plots.ipynb`

   Main outputs:
   - `results/simulations/correlation_visualization_total.pdf`
   - `results/simulations/boxplot_Wasserstein.pdf`

## Notes

- `Simulation.py` is the simulation experiments conducted in the main body of the paper.
- `simul_wasserstein_eval.py` consumes fitted objects written by `Simulation.py`.
- `Simulation_plots.ipynb` is the figure notebook used for the paper.
- `results/simulations/fitted_objects/` will be generated once the scripts run.
