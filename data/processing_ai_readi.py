"""AI-READI raw ingest and metadata-cleaning script."""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


METADATA_RENAME_MAP = {
    "participant_id": "id",
    "HbA1c (%)": "HbA1c",
    "Triglycerides (mg/dL)": "TG",
    "HDL Cholesterol (mg/dL)": "HDL-C",
    "Total Cholesterol (mg/dL)": "Total-C",
}

METADATA_COLUMNS = ["id", "HbA1c", "HDL-C", "Total-C", "TG"]
METADATA_REQUIRED_PREDICTORS = ["HbA1c", "HDL-C", "Total-C", "TG"]
MIN_HBA1C = 2.5
DEFAULT_N_JOBS = 10

def flatten_json(y):
    out = {}

    def flatten(x, name=""):
        # print(f'type(x) is {type(x)}')
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + "_")
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + "_")
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def convert_time_string_to_datetime(t_str):
    """Converts time string to datetime format. Does not convert to local time.
    Args:
        t_str (str): UTC time string such as 2023-08-01T20:39:33Z
    Returns: datetime object
    """
    datetime_object = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%SZ")  # 4 digit Year
    return datetime_object


def parse_measurement(df, identifier):
    return df[df["measurement_source_value"] == identifier]


def build_raw_metadata(data_dir):
    """Build the unfiltered metadata table from the AI-READI clinical files."""
    participants_df = pd.read_csv(data_dir / "participants.tsv", sep="\t")
    measurement_df = pd.read_csv(data_dir / "clinical_data" / "measurement.csv")

    identifiers = {
        "HbA1c (%)",
        "HDL Cholesterol (mg/dL)",
        "Total Cholesterol (mg/dL)",
        "Triglycerides (mg/dL)",
    }

    metadata = pd.DataFrame(columns=["participant_id"])
    metadata["participant_id"] = participants_df["participant_id"]

    for key in identifiers:
        temp_df = parse_measurement(measurement_df, key)
        temp_df = temp_df.rename(columns={"value_as_number": key, "person_id": "participant_id"})
        temp_df = temp_df[["participant_id", key]]
        metadata = pd.merge(metadata, temp_df, on="participant_id", how="left")

    return metadata


def clean_metadata(metadata, valid_ids):
    """Apply the notebook's metadata renaming and filtering in script form."""
    metadata = metadata.rename(columns=METADATA_RENAME_MAP).copy()
    metadata = metadata[METADATA_COLUMNS]

    for source_col, log_col in (("HbA1c", "log(HbA1c)"), ("TG", "log(TG)")):
        metadata[log_col] = np.nan
        valid_mask = metadata[source_col] > 0
        metadata.loc[valid_mask, log_col] = np.log(metadata.loc[valid_mask, source_col])

    valid_ids = pd.Index(valid_ids).dropna().unique()
    metadata = metadata[metadata["id"].isin(valid_ids)]
    metadata = metadata.dropna(subset=METADATA_REQUIRED_PREDICTORS)
    metadata = metadata[metadata["HbA1c"] >= MIN_HBA1C]
    metadata = metadata.reset_index(drop=True)

    return metadata


def process_participant_data(participant_id, glucose_filepath, data_dir, low_value=40, high_value=400):
    """
    Process CGM data for a single participant.
    
    Args:
        participant_id: The participant ID
        glucose_filepath: Path to the participant's glucose data file (relative to data_dir)
        data_dir: Base data directory
        low_value: Value to replace "Low" readings with
        high_value: Value to replace "High" readings with
    
    Returns:
        pandas.DataFrame: Processed CGM data with participant_id column added
    """
    try:
        # Construct full path to CGM data file
        cgm_path = os.path.join(data_dir, glucose_filepath)
        
        # Read the mHealth formatted data as json
        with open(cgm_path, "r") as f:
            data = json.load(f)
        
        # CGM observations are in a list of nested dicts; flatten these
        list_of_body_dicts = list()
        for observation in data["body"]["cgm"]:
            flat_obs = flatten_json(observation)
            list_of_body_dicts.append(flat_obs)
        
        # Convert to pandas dataframe
        df_participant = pd.DataFrame.from_records(list_of_body_dicts)
        
        # Rename columns to more readable names
        df_participant.rename(
            columns={
                "effective_time_frame_time_interval_start_date_time": "start_time",
                "effective_time_frame_time_interval_end_date_time": "end_time",
            },
            inplace=True,
        )
        
        # Exclusion criterion: must have records for at least 70% of 1 week
        if df_participant.shape[0] < 0.7*(7*24*60/5):
            print(f"    Participant {participant_id} excluded due to insufficient record counts (<4.9 day): {df_participant.shape[0]}")
            return None

        # Handle non-numeric blood glucose values (Low/High)
        def replace_alt(val, low_value, high_value):
            if val == "Low":
                return low_value
            elif val == "High":
                return high_value
            else:
                return val

        df_participant["blood_glucose_value"] = df_participant.apply(
            lambda x: replace_alt(x["blood_glucose_value"], low_value, high_value), axis=1
        )

        # Exclusion for variability regression: skip participants with too frequent 400+ readings (>5% of readings)
        if (df_participant['blood_glucose_value'] == 400).sum() > 0.05 * df_participant.shape[0]:
            print(f"    Participant {participant_id} excluded due to excessive high glucose readings (>=400)")
            return None

        # Convert start_time to datetime
        df_participant["start_dtime"] = df_participant.apply(
            lambda row: convert_time_string_to_datetime(row["start_time"]), axis=1
        )

        # Sort by timestamp
        df_participant = df_participant.sort_values(by="start_dtime").reset_index(drop=True)
        
        # Interpolate missing timestamps for gaps <= 30 minutes
        rows_to_add = []
        for i in range(len(df_participant) - 1):
            current_time = df_participant.loc[i, "start_dtime"]
            next_time = df_participant.loc[i + 1, "start_dtime"]
            time_gap = (next_time - current_time).total_seconds() / 60  # gap in minutes

            # If gap is > 9 minutes and <= 32 minutes, add interpolated rows
            if 9 < time_gap <= 32:
                num_intervals = int(round(time_gap / 5))  # number of 5-minute intervals
                
                # Get blood glucose values for interpolation
                current_bg = df_participant.loc[i, "blood_glucose_value"]
                next_bg = df_participant.loc[i + 1, "blood_glucose_value"]
                
                # Add intermediate timestamps with interpolated values
                for j in range(1, num_intervals):
                    new_time = current_time + pd.Timedelta(minutes=5 * j)
                    # Linear interpolation
                    interpolated_bg = current_bg + (next_bg - current_bg) * (j / num_intervals)
                    
                    new_row = {
                        'start_dtime': new_time,
                        'blood_glucose_value': interpolated_bg,
                        'start_time': None,
                        'end_time': None
                    }
                    rows_to_add.append(new_row)
        
        # Add the interpolated rows to the dataframe
        n_original = len(df_participant)
        n_interpolated = len(rows_to_add)
        if rows_to_add:
            df_interpolated = pd.DataFrame(rows_to_add)
            df_participant = pd.concat([df_participant, df_interpolated], ignore_index=True)
            df_participant = df_participant.sort_values(by="start_dtime").reset_index(drop=True)
        
        # Add participant ID column
        df_participant['participant_id'] = participant_id
        df_participant['_n_original'] = n_original
        df_participant['_n_interpolated'] = n_interpolated
        
        return df_participant
        
    except Exception as e:
        print(f"Error processing participant {participant_id}: {str(e)}")
        return None

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root.parent / "dataset"

    # Manifest file for the cgm data
    cgm_manifest_path = data_dir / "wearable_blood_glucose" / "manifest.tsv"
    dfm = pd.read_csv(cgm_manifest_path, sep='\t')

    # Load all participants' data
    print("Starting to load data for all participants...")
    print(f"Total participants to process: {len(dfm)}")

    # Initialize list to store all participant dataframes
    all_participants_data = []
    failed_participants = []
    print(f"Using {DEFAULT_N_JOBS} parallel workers")

    participant_specs = dfm[["participant_id", "glucose_filepath"]].to_dict("records")
    participant_results = Parallel(n_jobs=DEFAULT_N_JOBS, verbose=10)(
        delayed(process_participant_data)(
            participant_id=spec["participant_id"],
            glucose_filepath=spec["glucose_filepath"],
            data_dir=data_dir,
            low_value=40,
            high_value=400,
        )
        for spec in participant_specs
    )

    # Process each participant result
    for spec, df_participant in zip(participant_specs, participant_results):
        participant_id = spec["participant_id"]
        if df_participant is not None:
            all_participants_data.append(df_participant)
        else:
            failed_participants.append(participant_id)

    print(f"\nData loading completed!")
    print(f"Successfully processed: {len(all_participants_data)} participants")
    print(f"Skipped participants: {len(failed_participants)} participants")

    if failed_participants:
        print(f"Skipped participant IDs: {failed_participants[:10]}...")  # Show first 10


    # Combine all participant data into a single DataFrame
    if all_participants_data:
        print("Combining all participant data into a single DataFrame...")
        
        # Concatenate all dataframes
        combined_df = pd.concat(all_participants_data, ignore_index=True)

        # --- Interpolation summary ---
        interp_summary = (
            combined_df.groupby('participant_id')
            .agg(
                n_original=('_n_original', 'first'),
                n_interpolated=('_n_interpolated', 'first'),
            )
            .reset_index()
        )
        n_profiles_with_interp = (interp_summary['n_interpolated'] > 0).sum()
        total_original = interp_summary['n_original'].sum()
        total_interpolated = interp_summary['n_interpolated'].sum()
        pct_interpolated = 100 * total_interpolated / (total_original + total_interpolated)

        print(f"\n--- Interpolation Summary ---")
        print(f"Profiles with >=1 interpolated reading : {n_profiles_with_interp} / {len(interp_summary)}")
        print(f"Total original readings                : {total_original}")
        print(f"Total interpolated readings            : {total_interpolated}")
        print(f"Interpolated share                     : {pct_interpolated:.2f}%")

        interp_summary_path = Path(__file__).parent / "interpolation_summary.csv"
        interp_summary.to_csv(interp_summary_path, index=False)
        print(f"Interpolation summary saved to: {interp_summary_path}")
        # --- end summary ---

        # Drop helper columns before further processing
        combined_df = combined_df.drop(columns=['_n_original', '_n_interpolated'])
        
        print(f"Combined DataFrame shape: {combined_df.shape}")
        print(f"Columns: {list(combined_df.columns)}")
        print(f"Unique participants in combined data: {combined_df['participant_id'].nunique()}")
        print(f"Date range: {combined_df['start_dtime'].min()} to {combined_df['start_dtime'].max()}")
        
        # Display basic statistics
        print("\nBasic statistics for blood glucose values:")
        print(combined_df['blood_glucose_value'].describe())
        
    else:
        raise RuntimeError("No participant data was successfully loaded.")


    # Select only columns 'blood_glucose_value', 'start_dtime', and 'participant_id'
    selected_columns = ['blood_glucose_value', 'start_dtime', 'participant_id']

    combined_df = combined_df[selected_columns]
    print(f"Final DataFrame shape after selection: {combined_df.shape}")

    # Rename columns for clarity
    combined_df.rename(columns={
        'blood_glucose_value': 'gl',
        'start_dtime': 'time',
        'participant_id': 'id'
    }, inplace=True)


    # Save the final dataframe to CSV
    filename = repo_root / "data" / "ai_readi.csv"
    combined_df.to_csv(filename, index=False)

    # Metadata processing
    raw_metadata = build_raw_metadata(data_dir)
    raw_metadata_path = repo_root / "data" / "ai_readi_metadata.csv"
    raw_metadata.to_csv(raw_metadata_path, index=False)
    print(f"Saved raw metadata to: {raw_metadata_path}")

    cleaned_metadata = clean_metadata(raw_metadata, valid_ids=combined_df["id"])
    cleaned_metadata_path = repo_root / "data" / "ai_readi_metadata_cleaned.csv"
    cleaned_metadata.to_csv(cleaned_metadata_path, index=False)
    print(f"Saved cleaned metadata to: {cleaned_metadata_path}")
    print(f"Cleaned metadata shape: {cleaned_metadata.shape}")
