"""Batch permutation workflow for the AI-READI adjusted R^2 analysis."""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.window_processing import prepare_frechet_data
from functions.regression import multivariate_frechet_regression

SEED = 20251225
DEFAULT_NJOBS = int(os.environ.get("N_JOBS", "-1"))
FEATURE_NAMES = ["Mean", "CV", "MAD"]
DEFAULT_BOUNDS = {"Mean": (40.0, 400), "CV": (0.0, None), "MAD": (0.0, None)}
CANONICAL_PREDICTOR_RUNS = [
    ["HbA1c"],
    ["log(TG)"],
    ["HDL-C"],
    ["Total-C"],
    ["HbA1c", "log(TG)", "HDL-C", "Total-C"],
]

_AIREADI_PROCESSED = None
_METADATA = None


def _load_real_data_inputs():
    """Load and cache the committed processed window data and cleaned metadata."""

    global _AIREADI_PROCESSED, _METADATA

    if _AIREADI_PROCESSED is None:
        features_aireadi = pd.read_csv(REPO_ROOT / "data" / "aireadi_window120.csv")
        _AIREADI_PROCESSED = prepare_frechet_data(
            features_aireadi,
            feature_names=FEATURE_NAMES,
        )

    if _METADATA is None:
        _METADATA = pd.read_csv(REPO_ROOT / "data" / "ai_readi_metadata_cleaned.csv")

    return _AIREADI_PROCESSED, _METADATA


def load_real_data_inputs():
    """Public wrapper for loading the committed processed inputs used in permutation runs."""

    return _load_real_data_inputs()


def predictor_key(predictor_list):
    """Convert a predictor list into the saved filename stem."""

    if isinstance(predictor_list, str):
        predictor_list = [predictor_list]
    return "_".join(predictor_list)


def get_results_dir():
    """Return the canonical directory for saved permutation result objects."""

    return REPO_ROOT / "results" / "real" / "R2"


def permutation_R2(
    predictor_list,
    n_permutations=2000,
    seed=SEED,
    njobs=DEFAULT_NJOBS,
    aireadi_processed=None,
    metadata=None,
):
    """
    Permutation test for Nonparanormal Frechet R^2.
    1. Get the original R^2 for given predictor(s).
    2. Permute the predictor(s) n_permutations times, get R^2 for each permutation.
    3. Get dataframes of R^2 vectors: one original and n_permuted R^2s.
    4. Save results to results/real/R2 in a pickle file.
    5. Return p-values with original and permuted R^2 values.
    """
    rs = np.random.RandomState(seed)

    if isinstance(predictor_list, str):
        predictor_list = [predictor_list]

    if aireadi_processed is None or metadata is None:
        aireadi_processed, metadata = _load_real_data_inputs()

    # Get original R^2
    joined_df = aireadi_processed["data"].merge(metadata, on="id", how="left")
    result = multivariate_frechet_regression(
        predictor_list,
        joined_df,
        r_squared=True,
        feature_names=aireadi_processed["feature_names"],
        bounds=DEFAULT_BOUNDS,
        space_interval=None,
        verbose=False,
    )
    original_R2 = result["R_squares"]

    # Permutation test
    n_samples = joined_df.shape[0]
    perm_indices = [rs.permutation(n_samples) for _ in range(n_permutations)]

    def permute_and_compute_R2(perm_idx):
        joined_df_permuted = joined_df.copy()
        for predictor in predictor_list:
            joined_df_permuted[predictor] = joined_df_permuted[predictor].values[perm_idx]

        result_perm = multivariate_frechet_regression(
            predictor_list,
            joined_df_permuted,
            r_squared=True,
            space_interval=2,
            feature_names=aireadi_processed["feature_names"],
            bounds=DEFAULT_BOUNDS,
            verbose=False,
        )
        return result_perm["R_squares"]

    permuted_R2s = Parallel(n_jobs=njobs, verbose=10)(
        delayed(permute_and_compute_R2)(idx) for idx in perm_indices
    )
    permuted_R2s_df = pd.concat(permuted_R2s, axis=0, ignore_index=True)

    # Compute p-values
    p_values = pd.DataFrame(columns=original_R2.columns)
    for feature in original_R2.columns:
        orig_value = original_R2[feature].values[0]
        perm_values = permuted_R2s_df[feature].values
        p_values[feature] = np.sum(perm_values >= orig_value) / n_permutations

    all_results = {
        "original_R2": original_R2,
        "permuted_R2s": permuted_R2s_df,
        "p_values": p_values,
    }

    # Save results
    results_dir = get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    save_path = results_dir / f"permutation_R2_{predictor_key(predictor_list)}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)

    return all_results


def main():
    """Run the canonical paper-relevant permutation models and save raw result objects."""

    aireadi_processed, metadata = _load_real_data_inputs()
    for predictor_list in CANONICAL_PREDICTOR_RUNS:
        permutation_R2(
            predictor_list,
            n_permutations=2000,
            seed=SEED,
            njobs=DEFAULT_NJOBS,
            aireadi_processed=aireadi_processed,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
