# Real-data Reproducibility Guide

This directory contains the real-data figure and permutation workflows.

## Inputs

Required files:

- `data/ai_readi.csv`
- `data/ai_readi_metadata.csv`
- `data/ai_readi_metadata_cleaned.csv`
- `data/aireadi_window120.csv`

Generate these files by following [data/Processing_Guide.md](../../data/Processing_Guide.md).

## Steps

1. Generate the real-data figures.

   Run:
   - `experiments/real/ai_readi_regresseion.ipynb`

   Main outputs:
   - `results/real/figs/ai_readi_hba1c_multivar_mad.pdf`
   - `results/real/figs/ai_readi_hba1c_latentcor.pdf`

   $\quad$

2. Generate permutation result objects.

   Run (recommended on HPC cluster):

   ```powershell
   python experiments/real/ai_permutation.py
   ```

   or `experiments/real/ai_readi_permutation.ipynb` also has sample script for obtaining permutation-based null distributions

   $\quad$

3. Adjust the p-values and export the `R^2` table.

   Run:
   - `experiments/real/ai_readi_permutation.ipynb`

   Main output:
   - `results/tables/R2_adjusted_pvalues.tex`

## Notes

- `ai_readi_regresseion.ipynb` is the notebook used to generate figures for the paper.
   - It also includes additional figures not presented in the paper for further reference.
- `ai_readi_permutation.ipynb` is the notebook used for Westfall--Young adjusted `R^2` inference and table export.
- `ai_permutation.py` runs the permutation test jobs, generating only R^2 values.