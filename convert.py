import pandas as pd
import os

# --- Configuration ---
input_filename = "master_parameters_constrained.csv"
output_filename = "master_parameters.csv"

# --- Main Script Logic ---

try:
    # 1. Load the input CSV file.
    print(f"Reading data from '{input_filename}'...")
    df = pd.read_csv(input_filename)

    # 2. Create a new DataFrame that excludes the first two columns.
    #    df.iloc[:, 2:] selects all rows (:) and all columns from the 3rd column (index 2) onwards.
    df_transformed = df.iloc[:, 2:].copy()

    # 3. Insert the new 'param_id' column at the beginning (position 0).
    #    The values will be a simple zero-based index (0, 1, 2, ...).
    df_transformed.insert(0, "param_id", range(len(df_transformed)))

    # 4. Save the resulting DataFrame to the new CSV file.
    #    index=False is essential to prevent pandas from adding another index column.
    df_transformed.to_csv(output_filename, index=False)

    print(
        f"Success! The file has been correctly transformed and saved as '{output_filename}'."
    )
    print(f"Location: {os.path.abspath(output_filename)}")

except FileNotFoundError:
    print(f"Error: The input file '{input_filename}' was not found.")
    print("Please make sure the script and the CSV file are in the same directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
