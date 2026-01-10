import pandas as pd
import json
import numpy as np
import os

# -----------------------------
# Configuration
# -----------------------------
INPUT_FILE = "consolidated_results_dfn.parquet"
OUTPUT_FILE = "verification_sample.json"
SAMPLE_SIZE = 10


class NumpyEncoder(json.JSONEncoder):
    """
    Helper to ensure numpy numbers and arrays are converted
    to standard Python types for JSON saving.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} does not exist.")
        return

    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except Exception as e:
        print(f"Failed to read parquet file. Reason: {e}")
        return

    total_rows = len(df)
    print(f"Total records found: {total_rows}")
    print(f"Columns: {list(df.columns)}")

    # Pick a sample
    n = min(SAMPLE_SIZE, total_rows)
    # Use random sample if we have enough data, else take head
    if total_rows > n:
        df_sample = df.sample(n)
    else:
        df_sample = df.head(n)

    # Convert to list of dictionaries
    records = df_sample.to_dict(orient="records")

    print(f"Exporting {n} records to {OUTPUT_FILE}...")

    # Save to human-readable JSON
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(records, f, indent=4, cls=NumpyEncoder)

        print("Done.")
        print(f"Open '{OUTPUT_FILE}' in your text editor to verify the data.")

        # Optional: Print a quick summary to console
        print("\n--- Quick Preview (Scalar values only) ---")
        # Drop columns that look like arrays (dictionaries) for console print
        scalar_cols = [
            c
            for c in df.columns
            if not isinstance(df_sample.iloc[0][c], (dict, list, np.ndarray))
        ]
        print(df_sample[scalar_cols].to_string(index=False))

    except Exception as e:
        print(f"Error writing JSON: {e}")


if __name__ == "__main__":
    main()
