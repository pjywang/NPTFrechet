"""Real-data window-processing and Frechet-preparation utilities."""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import linregress


DEFAULT_WINDOW_SIZE = 120
DEFAULT_OVERLAP = 0.75
DEFAULT_N_JOBS = 10


def window_process(
    data,
    measurement_name="gl",
    parallel=True,
    n_jobs=DEFAULT_N_JOBS,
    window_size=120,
    overlap=0.75,
    do_slope=False,
    do_tir=False,
):
    """Process all patients' data with optional joblib parallelism."""

    if isinstance(data[measurement_name].iloc[0], list):
        gl_data = data.copy()
    else:
        gl_data = data.groupby("id").agg({measurement_name: list, "time": list}).reset_index()

    patient_data_list = [
        (row["id"], row[measurement_name], row["time"]) for _, row in gl_data.iterrows()
    ]

    start_time = time.time()
    if parallel and len(patient_data_list) > 1:
        print(f"Processing {len(patient_data_list)} patients in parallel...")
        all_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_patient)(patient_data, window_size, overlap, do_slope, do_tir)
            for patient_data in patient_data_list
        )
        all_window_features = [
            feature for patient_features in all_results for feature in patient_features
        ]
    else:
        print(f"Processing {len(patient_data_list)} patients sequentially...")
        all_window_features = []
        for i, patient_data in enumerate(patient_data_list):
            patient_features = process_patient(
                patient_data,
                window_size=window_size,
                overlap=overlap,
                do_slope=do_slope,
                do_tir=do_tir,
            )
            all_window_features.extend(patient_features)
            if (i + 1) % 10 == 0 or i + 1 == len(patient_data_list):
                print(f"Processed {i + 1}/{len(patient_data_list)} patients")

    features_df = pd.DataFrame(all_window_features)

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Total windows extracted: {len(features_df)}")

    if len(features_df) > 0:
        print(f"Mean number of windows per patient: {features_df.groupby('id').size().mean():.2f}")

    return features_df


def process_patient(patient_data, window_size=120, overlap=0.75, do_slope=False, do_tir=True):
    """Process one patient's glucose profile into window-level features."""

    patient_id, gl_list, time_list = patient_data
    gl_list = [x if x < 400 else np.nan for x in gl_list]

    patient_features = extract_window_features(
        gl_list,
        time_list,
        window_size,
        overlap,
        do_slope=do_slope,
        do_tir=do_tir,
    )

    for feature_dict in patient_features:
        feature_dict["id"] = patient_id

    return patient_features


def extract_window_features(
    glucose_values,
    times,
    window_size=120,
    overlap=0.75,
    do_slope=False,
    do_tir=True,
    measure_interval=5,
):
    """Extract sliding-window glucose features from one profile."""

    glucose = np.array(glucose_values, dtype=np.float64)

    try:
        if isinstance(times[0], str):
            times = np.array([pd.to_datetime(t) for t in times])
        else:
            times = np.array(times)
    except Exception as exc:
        print(f"Error converting time values: {exc}")
        return []

    sort_idx = np.argsort(times)
    glucose = glucose[sort_idx]
    times = times[sort_idx]

    try:
        time_diffs_raw = np.array(
            [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]
        )
    except Exception as exc:
        print(f"Error calculating time differences: {exc}")
        print(f"Time values type: {type(times[0])}")
        return []

    time_diffs = time_diffs_raw / 60.0
    expected_num = int(window_size / measure_interval)

    cumulative_time = np.zeros(len(times))
    cumulative_time[1:] = np.cumsum(time_diffs)

    overlap_min = overlap * window_size
    step = window_size - overlap_min

    start_indices = []
    end_indices = []

    start_idx = 0
    while start_idx < len(glucose) - 1:
        end_idx = np.searchsorted(
            cumulative_time,
            cumulative_time[start_idx] + window_size - 3 / 5 * measure_interval,
            side="right",
        )
        end_idx -= 1

        if end_idx < len(glucose) - 1 and len(glucose[start_idx : end_idx + 1]) >= expected_num * 0.75:
            start_indices.append(start_idx)
            end_indices.append(end_idx)

        new_start_idx = start_idx
        target_time = cumulative_time[start_idx] + step
        while new_start_idx < len(glucose) - 1 and cumulative_time[new_start_idx] < target_time:
            new_start_idx += 1

        start_idx = new_start_idx if new_start_idx > start_idx else start_idx + 1

    window_features = []

    for i in range(len(start_indices)):
        start_idx = start_indices[i]
        end_idx = end_indices[i]

        window_glucose = glucose[start_idx : end_idx + 1]
        window_times = times[start_idx : end_idx + 1]
        window_cum_times = cumulative_time[start_idx : end_idx + 1] - cumulative_time[start_idx]

        selected_indices = [0]
        for target_time in np.arange(
            measure_interval,
            window_cum_times[-1] + measure_interval * 0.4,
            measure_interval,
        ):
            time_diffs_to_target = np.abs(window_cum_times - target_time)
            closest_idx = np.argmin(time_diffs_to_target)
            if closest_idx not in selected_indices:
                selected_indices.append(closest_idx)

        window_glucose = window_glucose[selected_indices]
        window_times = window_times[selected_indices]
        window_cum_times = window_cum_times[selected_indices]

        valid_mask = ~np.isnan(window_glucose)
        window_glucose = window_glucose[valid_mask]
        window_times = window_times[valid_mask]
        window_cum_times = window_cum_times[valid_mask]

        if len(window_glucose) < expected_num * 0.75:
            continue

        mean_glucose = np.mean(window_glucose)
        cv_glucose = np.std(window_glucose) / mean_glucose if mean_glucose > 0 else np.nan
        mad_consecutive = np.mean(np.abs(np.diff(window_glucose)))

        features = {
            "start_time": window_times[0],
            "end_time": window_times[-1],
            "mean_glucose": mean_glucose,
            "cv_glucose": cv_glucose,
            "mad_consecutive": mad_consecutive,
            "n_points": len(window_glucose),
        }

        if do_slope:
            slope, _, _, _, _ = linregress(window_cum_times, window_glucose)
            features["slope"] = slope

        if do_tir:
            tbr = np.mean(window_glucose < 70)
            tir = np.mean((window_glucose >= 70) & (window_glucose <= 180))
            tar = np.mean(window_glucose > 180)
            features["tbr<70"] = tbr
            features["tir70-180"] = tir
            features["tar>180"] = tar

        window_features.append(features)

    return window_features


def plot_feature_distributions(features_df):
    """Plot marginal distributions of extracted window features."""

    features_df = features_df.copy()
    features_df = features_df.drop(columns=["id", "start_time", "end_time", "n_points"], errors="ignore")
    col_numbers = len(features_df.columns)

    fig, axes = plt.subplots(1, col_numbers, figsize=(4 * col_numbers - 1, 4))
    axes = axes.flatten()

    for i, col in enumerate(features_df.columns):
        axes[i].hist(features_df[col], bins=30, alpha=0.7, density=True)
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")

    plt.tight_layout()
    plt.show()


def plot_feature_correlations(
    features_df,
    correlation_features=["mean_glucose", "cv_glucose", "slope"],
    show=True,
):
    """Plot the feature correlation heatmap."""

    correlation_matrix = features_df[correlation_features].corr()

    plt.figure(figsize=(6, 4))
    plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(
        np.arange(len(correlation_features)),
        correlation_features,
        rotation=45,
        ha="right",
    )
    plt.yticks(np.arange(len(correlation_features)), correlation_features)

    for i in range(len(correlation_features)):
        for j in range(len(correlation_features)):
            plt.text(
                j,
                i,
                f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black",
            )

    plt.title("Correlation between Glucose Features")

    if show:
        plt.tight_layout()
        plt.show()


def prepare_latent_corr_data(
    df,
    features=["mean_glucose", "cv_glucose", "tbr<70", "tir70-180"],
):
    """Prepare patient-level feature arrays for latent correlation estimation."""

    patient_data = []
    patient_ids = []

    for patient_id, group in df.groupby("id"):
        patient_features = group[features].values
        if len(patient_features) < 5:
            continue

        patient_data.append(patient_features)
        patient_ids.append(patient_id)

    return patient_data, patient_ids


def prepare_frechet_data(window_df, id_col="id", feature_names=["Mean", "CV", "MAD"]):
    """Prepare patient-level multivariate distributions for Frechet regression."""

    from functions.regression import get_latent_cor

    multivar_data = []

    for patient_id, group in window_df.groupby(id_col):
        multivariate_values = group[feature_names].values
        if len(multivariate_values) < 5:
            continue

        patient_record = {
            id_col: patient_id,
            "multivar_distribution": multivariate_values,
            "n_windows": len(multivariate_values),
        }
        multivar_data.append(patient_record)

    multivar_df = pd.DataFrame(multivar_data)

    print(f"Processed {len(multivar_df)} patients with sufficient data")
    print(f"Average windows per patient: {multivar_df['n_windows'].mean():.2f}")

    multivar_list = multivar_df["multivar_distribution"].tolist()

    print("Calculating latent correlations...", end=" ")
    latent_cors = get_latent_cor(multivar_list)
    print("Done!")

    multivar_df["latentcor"] = list(latent_cors)
    multivar_df = multivar_df[[id_col, "multivar_distribution", "n_windows", "latentcor"]]

    return {
        "data": multivar_df,
        "feature_names": feature_names,
    }


def select_features(processed_df, feature_names, sub_names):
    """Select a subset of features from processed multivariate outputs."""

    if not set(sub_names).issubset(set(feature_names)):
        raise ValueError("Some sub_names are not present in processed_data['feature_names']")

    indices = [feature_names.index(name) for name in sub_names]
    selected_df = processed_df.copy()

    if "multivar_distribution" in selected_df.columns:
        selected_df["multivar_distribution"] = selected_df["multivar_distribution"].apply(
            lambda arr: arr[:, indices] if hasattr(arr, "shape") and arr.shape[1] >= len(indices) else arr
        )
    if "marginal_fits" in selected_df.columns:
        selected_df["marginal_fits"] = selected_df["marginal_fits"].apply(
            lambda arr: arr[indices, :] if hasattr(arr, "shape") and arr.shape[1] >= len(indices) else arr
        )

    if "latentcor" in selected_df.columns:
        selected_df["latentcor"] = selected_df["latentcor"].apply(
            lambda arr: arr[:, indices][indices, :]
            if hasattr(arr, "shape") and arr.shape[0] >= len(indices)
            else arr
        )
    if "latentcor_fits" in selected_df.columns:
        selected_df["latentcor_fits"] = selected_df["latentcor_fits"].apply(
            lambda arr: arr[:, indices][indices, :]
            if hasattr(arr, "shape") and arr.shape[0] >= len(indices)
            else arr
        )

    return selected_df


def generate_window_features(
    data,
    metadata,
    n_jobs=DEFAULT_N_JOBS,
    window_size=DEFAULT_WINDOW_SIZE,
    overlap=DEFAULT_OVERLAP,
):
    """Replicate the notebook's window-feature generation path."""

    gl_data = data.groupby("id").agg({"gl": list, "time": list}).reset_index()
    gl_data = gl_data[gl_data["id"].isin(metadata["id"])].reset_index(drop=True)

    features = window_process(
        gl_data,
        n_jobs=n_jobs,
        do_slope=True,
        do_tir=True,
        window_size=window_size,
        overlap=overlap,
    )
    features = features.rename(
        columns={
            "mean_glucose": "Mean",
            "cv_glucose": "CV",
            "mad_consecutive": "MAD",
        }
    )
    return features


__all__ = [
    "extract_window_features",
    "generate_window_features",
    "plot_feature_correlations",
    "plot_feature_distributions",
    "prepare_frechet_data",
    "prepare_latent_corr_data",
    "process_patient",
    "select_features",
    "window_process",
]
