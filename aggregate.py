import os
import glob
import pandas as pd

# Configuration (Matching your existing setup)
SWEEPS_DIR = "pybamm_param_sweeps_sep_microstructure"
CSV_GLOB = os.path.join(SWEEPS_DIR, "params_sep_micro_*_run_*.csv")
OUTPUT_MASTER_CSV = "master_parameters.csv"


def aggregate_csvs():
    csv_files = sorted(glob.glob(CSV_GLOB))

    if not csv_files:
        print(f"No files found in {SWEEPS_DIR} matching the pattern.")
        return

    print(f"Found {len(csv_files)} files. Starting aggregation...")

    all_runs = []

    for file_path in csv_files:
        # 1. Read the individual CSV (key, value)
        df = pd.read_csv(file_path)

        # 2. Convert to a dictionary: {key: value}
        # We use strip() to ensure no whitespace issues in keys
        run_data = dict(zip(df["key"].str.strip(), df["value"]))

        # 3. Add the filename/run_id so we can track which row belongs to which file
        run_data["source_file"] = os.path.basename(file_path)
        run_data["run_name"] = os.path.splitext(os.path.basename(file_path))[0]

        all_runs.append(run_data)

    # 4. Create a master DataFrame
    master_df = pd.DataFrame(all_runs)

    # 5. Reorder columns to put 'run_name' and 'source_file' at the front for readability
    cols = ["run_name", "source_file"] + [
        c for c in master_df.columns if c not in ["run_name", "source_file"]
    ]
    master_df = master_df[cols]

    # 6. Save to CSV
    master_df.to_csv(OUTPUT_MASTER_CSV, index=False)
    print(f"Success! Master CSV saved to: {OUTPUT_MASTER_CSV}")
    print(f"Shape: {master_df.shape} (Rows = Runs, Columns = Parameters + Metadata)")


if __name__ == "__main__":
    aggregate_csvs()
