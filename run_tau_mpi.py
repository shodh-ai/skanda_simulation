#!/usr/bin/env python
"""
MPI Taufactor Post-Processing Script.

Usage:
    mpirun -n <cores> python run_tau_mpi.py --input_dir <path_to_generated_files>
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import tifffile
import taufactor as tau
from mpi4py import MPI

# ==========================================
# 1. TAUFACTOR WRAPPER
# ==========================================


def calculate_properties(file_path, axis=0):
    """
    Calculates Porosity, Tortuosity, and Diffusivity.

    Args:
        file_path (str): Path to the tiff file.
        axis (int): The axis along which to calculate tortuosity (0=Z, 1=Y, 2=X).
                    Your generation script applies anisotropy in Z (axis 0).
    """
    try:
        # Load data
        vol_data = tifffile.imread(file_path)

        # Ensure binary (1=Pore, 0=Solid)
        # Your generation script returns (~final_solid), so 1 is pore.
        vol_data = (vol_data > 0).astype(np.uint8)

        # 1. Porosity
        phi = float(vol_data.mean())

        # Check for empty or full volumes to prevent solver crash
        if phi <= 0.0 or phi >= 1.0:
            return {
                "porosity_measured": phi,
                "tau_factor": np.nan,
                "D_eff": 0.0 if phi <= 0 else 1.0,
                "error": "Volume is fully solid or fully void",
            }

        # 2. Run Solver
        # Note: Solver(vol_data) assumes 1 is the conducting phase.
        solver = tau.Solver(vol_data)

        # Determine iteration limit based on size to avoid hanging
        solver.solve(verbose=False, iter_limit=2000)

        # 3. Extract Results
        results = {
            "porosity_measured": phi,
            "tau_factor": float(solver.tau),
            "D_eff": float(solver.D_eff),
            "error": None,
        }
        return results

    except Exception as e:
        return {
            "porosity_measured": np.nan,
            "tau_factor": np.nan,
            "D_eff": np.nan,
            "error": str(e),
        }


# ==========================================
# 2. MPI LOGIC
# ==========================================


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # -----------------------------
    # Rank 0: Setup and Task List
    # -----------------------------
    args = None
    tasks = []
    output_dir = ""

    if rank == 0:
        parser = argparse.ArgumentParser(description="MPI Taufactor Analysis")
        parser.add_argument(
            "--input_dir",
            required=True,
            help="Directory containing TIFFs and final_results.csv",
        )
        parser.add_argument(
            "--output_file", default="taufactor_results.csv", help="Name of output CSV"
        )
        args = parser.parse_args()

        output_dir = args.input_dir
        input_csv = os.path.join(args.input_dir, "final_results.csv")

        if not os.path.exists(input_csv):
            print(f"Error: {input_csv} not found. Cannot map IDs to files.")
            sys.exit(1)

        print(f"--- Starting Taufactor Analysis on {size} ranks ---")

        # Read the manifest from the generation step
        df = pd.read_csv(input_csv)

        # Create a list of dictionaries for tasks
        # We need 'id' and 'filename'
        tasks = df[["id", "filename"]].to_dict("records")
        print(f"Found {len(tasks)} samples to process.")

    # -----------------------------
    # Broadcast Tasks to All Ranks
    # -----------------------------
    # Broadcast the input directory and the task list
    output_dir = comm.bcast(output_dir, root=0)
    tasks = comm.bcast(tasks, root=0)

    # -----------------------------
    # Parallel Processing (Round Robin)
    # -----------------------------
    local_results = []

    # Each rank processes indices: rank, rank+size, rank+2*size...
    for i in range(rank, len(tasks), size):
        task = tasks[i]
        sample_id = task["id"]
        filename = task["filename"]
        full_path = os.path.join(output_dir, filename)

        if not os.path.exists(full_path):
            res = {
                "id": sample_id,
                "filename": filename,
                "porosity_measured": np.nan,
                "tau_factor": np.nan,
                "D_eff": np.nan,
                "error": "File not found",
            }
        else:
            t0 = time.time()
            calc = calculate_properties(full_path)
            duration = time.time() - t0

            res = {
                "id": sample_id,
                "filename": filename,
                "porosity_measured": calc["porosity_measured"],
                "tau_factor": calc["tau_factor"],
                "D_eff": calc["D_eff"],
                "error": calc["error"],
            }

            # Optional: Print progress from specific ranks to avoid clutter
            if calc["error"]:
                print(f"[Rank {rank}] ID {sample_id} FAILED: {calc['error']}")
            else:
                print(
                    f"[Rank {rank}] ID {sample_id}: Tau={calc['tau_factor']:.4f}, Time={duration:.2f}s"
                )

        local_results.append(res)

    # -----------------------------
    # Gather Results
    # -----------------------------
    # gathered_results will be a list of lists: [[rank0_res], [rank1_res], ...]
    gathered_results = comm.gather(local_results, root=0)

    # -----------------------------
    # Rank 0: Aggregation and Save
    # -----------------------------
    if rank == 0:
        # Flatten list of lists
        flat_results = [item for sublist in gathered_results for item in sublist]

        # Create DataFrame
        df_out = pd.DataFrame(flat_results)

        # Sort by ID
        df_out = df_out.sort_values(by="id")

        # Reorder columns as requested
        cols = ["id", "filename", "porosity_measured", "tau_factor", "D_eff", "error"]
        df_out = df_out[cols]

        save_path = os.path.join(output_dir, args.output_file)
        df_out.to_csv(save_path, index=False)

        print(f"--- Processing Complete ---")
        print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
