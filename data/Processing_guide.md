# AI-READI Processing Guide

Use this guide to generate the AI-READI CGM dataset files used by our paper.

## External Input

The data is available upon request to the AI-READI study:
- dataset documentation: [docs.aireadi.org](https://docs.aireadi.org/)
- project website: [aireadi.org](https://aireadi.org/)
- historical v2.0.0 documentation: [AI-READI docs v2.0.0](https://docs.aireadi.org/docs/2/about)

Note that we use the previous version v2.0.0, while v3.0.0 is available as of 2026.

After downloading CGM measurements and clinical variables, place them in the sibling directory:

```text
../dataset/
```

Expected files:

- `participants.tsv`
- `clinical_data/measurement.csv`
- `wearable_blood_glucose/manifest.tsv`


## Processing Steps

1. Build the CGM and metadata CSV files.

   ```powershell
   python data/processing_ai_readi.py
   ```

   Outputs:
   - `data/ai_readi.csv`
   - `data/ai_readi_metadata.csv`
   - `data/ai_readi_metadata_cleaned.csv`
   - `data/interpolation_summary.csv`

2. Build the window-feature CSV file for CGM trivariate distributional representation.

   Run:
   - `data/ai_readi_window_processing.ipynb`

   Output:
   - `data/aireadi_window120.csv`

3. Continue with the real-data workflows in `experiments/real/`.

## Notes

- `processing_ai_readi.py` reads the external export and writes repository-local CSV files.
- `window_processing.py` contains reusable preprocessing helpers.
- `ai_readi_window_processing.ipynb` is the canonical path for generating `data/aireadi_window120.csv`.