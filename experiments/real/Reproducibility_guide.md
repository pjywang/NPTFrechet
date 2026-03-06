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

2. Generate permutation result objects.

   Run either:

   ```powershell
   python experiments/real/ai_permutation.py
   ```

   or

   - `experiments/real/ai_readi_permutation.ipynb`

3. Export the adjusted `R^2` table.

   Run:
   - `experiments/real/ai_readi_permutation.ipynb`

   Main output:
   - `results/tables/R2_adjusted_pvalues.tex`

## Notes

- `ai_readi_regresseion.ipynb` is the figure notebook used for the paper.
- `ai_readi_permutation.ipynb` is the notebook used for adjusted `R^2` inference and table export.
- `ai_permutation.py` runs the saved-model permutation jobs from the command line.
- The filename `ai_readi_regresseion.ipynb` is historical.
