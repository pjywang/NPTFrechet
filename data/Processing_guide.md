# AI-READI Processing Guide

A guide to generate the AI-READI CGM dataset files used by our paper.

## External Input

The data is available from the AI-READI study upon request:
- dataset documentation: [docs.aireadi.org](https://docs.aireadi.org/)
- project website: [aireadi.org](https://aireadi.org/)
- historical v2.0.0 documentation: [AI-READI docs v2.0.0](https://docs.aireadi.org/docs/2/about)

Note that we use the previous version v2.0.0, while v3.0.0 is available as of 2026.

After downloading CGM measurements and clinical variables, place them in the sibling directory:

```text
../dataset/
```

Expected files downloaded from the study (in the directory `../dataset/`):

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
   - `data/interpolation_summary.csv` (complementary reference indicating how many points are interpolated within the CGM trajectory)

2. Build the window-feature CSV file for CGM trivariate distributional representation.

   Run:
   - `data/ai_readi_window_processing.ipynb`

   Output:
   - `data/aireadi_window120.csv`

3. Continue with the real-data workflows in `experiments/real/`.

## Notes

- `processing_ai_readi.py` reads the external export of AI-READI data and writes processed CSV files into the `data/` directory.
- `window_processing.py` contains preprocessing helpers for multivariate distributional representation of wearable measurements.
- `ai_readi_window_processing.ipynb` generates `data/aireadi_window120.csv`.