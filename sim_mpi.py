import os
import time
import traceback
import pandas as pd
import numpy as np
import pybamm
from mpi4py import MPI

# -----------------------------
# MPI Initialization
# -----------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -----------------------------
# Configuration
# -----------------------------
MASTER_CSV = "master_parameters.csv"
RESULTS_DIR = "checkpoints"  # Directory for intermediate chunks
FINAL_OUTPUT_FILE = "consolidated_results.parquet"  # Final single file

N_CYCLES_MAX = 500
TERMINATION = "80% capacity"
SAVE_FREQUENCY = 10  # Save to disk every 10 runs to prevent data loss on crash

# Cycles to extract arrays from
CYCLES_TO_SAVE = ["first", "middle", "last"]

EXPERIMENT = pybamm.Experiment(
    [
        (
            "Discharge at 1C until 3V",
            "Rest for 1 hour",
            "Charge at 1C until 4.2V",
            "Hold at 4.2V until C/50",
        )
    ]
    * N_CYCLES_MAX,
    termination=TERMINATION,
)

pybamm.set_logging_level("ERROR")


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def extract_cycle_data(cycle_sol):
    """
    Extracts Time, Voltage, and Current arrays from a specific cycle.
    Returns a dict compatible with Parquet storage.
    """
    try:
        return {
            "Time_h": cycle_sol["Time [h]"].data.tolist(),
            "Voltage_V": cycle_sol["Voltage [V]"].data.tolist(),
            "Current_A": cycle_sol["Current [A]"].data.tolist(),
        }
    except Exception:
        return None


def try_extract_summary(sol) -> dict:
    """Extracts scalar summary variables from the end of the simulation."""
    summary = {}
    candidates = [
        "Capacity [A.h]",
        "Discharge capacity [A.h]",
        "Loss of lithium inventory [%]",
        "Loss of active material in negative electrode [%]",
    ]
    for name in candidates:
        try:
            # We take the last value of the summary variable
            v = sol.summary_variables[name]
            arr = getattr(v, "data", None) or getattr(v, "entries", None)
            if arr is not None and len(arr) > 0:
                summary[name] = float(arr[-1])
        except Exception:
            pass
    return summary


def save_chunk(data_buffer, filepath):
    """Appends a list of dicts to a parquet file."""
    if not data_buffer:
        return

    df_chunk = pd.DataFrame(data_buffer)

    # If file exists, append; otherwise create
    if os.path.exists(filepath):
        # Read existing to append (Pandas < 1.4 doesn't support direct append well,
        # but for safety/robustness we read-concat-write or use fastparquet/pyarrow tables.
        # For simplicity in this script, we read-concat-write.
        # For 1M runs, using pyarrow.Table is better, but this is 'minimal change' friendly).
        df_existing = pd.read_parquet(filepath)
        df_combined = pd.concat([df_existing, df_chunk], ignore_index=True)
        df_combined.to_parquet(filepath, index=False)
    else:
        df_chunk.to_parquet(filepath, index=False)


def get_completed_runs(filepath):
    """Reads the checkpoint file to find which run_names are already finished."""
    if not os.path.exists(filepath):
        return set()
    try:
        # Only read the 'run_name' column to save memory
        df = pd.read_parquet(filepath, columns=["run_name"])
        return set(df["run_name"].unique())
    except Exception:
        return set()


# -----------------------------
# Main MPI Runner
# -----------------------------
def run_all_mpi():
    ensure_dir(RESULTS_DIR)

    # 1. Rank 0 loads the master list
    if rank == 0:
        if not os.path.exists(MASTER_CSV):
            print(f"[Error] {MASTER_CSV} not found!")
            comm.Abort()
        df_master = pd.read_csv(MASTER_CSV)
        all_tasks = df_master.to_dict("records")
        print(f"[Master] Found {len(all_tasks)} simulations. Distributing...")
    else:
        all_tasks = None

    # Broadcast tasks to all ranks
    all_tasks = comm.bcast(all_tasks, root=0)

    # 2. Split tasks among ranks
    my_tasks = np.array_split(all_tasks, size)[rank]

    # -----------------------------
    # Checkpointing / Resume Logic
    # -----------------------------
    my_checkpoint_file = os.path.join(RESULTS_DIR, f"checkpoint_rank_{rank}.parquet")
    completed_runs = get_completed_runs(my_checkpoint_file)

    # Filter out tasks that are already done
    tasks_to_do = [t for t in my_tasks if t.get("run_name") not in completed_runs]

    if len(tasks_to_do) < len(my_tasks):
        print(
            f"[Rank {rank}] Resuming: Skipping {len(my_tasks) - len(tasks_to_do)} already finished tasks."
        )

    local_buffer = []

    # 3. Processing Loop
    for i, task in enumerate(tasks_to_do):
        run_name = task.get("run_name", "unknown")

        # Prepare the result row with input parameters
        row = task.copy()

        start_time = time.time()

        # Setup Model
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        params_to_update = {
            k: v for k, v in task.items() if k not in ["run_name", "source_file"]
        }
        parameter_values.update(params_to_update, check_already_exists=False)

        model = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})

        try:
            parameter_values.set_initial_state(1)
        except:
            pass

        sim = pybamm.Simulation(
            model, experiment=EXPERIMENT, parameter_values=parameter_values
        )

        run_ok = True
        error_message = ""
        sol = None

        try:
            sol = sim.solve()
        except Exception as e:
            run_ok = False
            error_message = str(e)  # Simplified error string

        runtime_s = time.time() - start_time

        # 4. Populate Output Data
        row["status"] = "success" if run_ok else "failed"
        row["runtime_s"] = round(runtime_s, 2)
        row["cycles_completed"] = len(sol.cycles) if (run_ok and sol) else 0
        row["error_message"] = error_message

        if run_ok and sol:
            # Summary stats
            summary_data = try_extract_summary(sol)
            row.update(summary_data)

            # Cycle Arrays
            num_cycles = len(sol.cycles)
            # Map logical names to indices
            indices_map = {
                "first": 0,
                "middle": num_cycles // 2,
                "last": num_cycles - 1,
            }

            for label in CYCLES_TO_SAVE:
                idx = indices_map.get(label)
                if idx is not None and 0 <= idx < num_cycles:
                    row[f"cycle_{label}"] = extract_cycle_data(sol.cycles[idx])
                else:
                    row[f"cycle_{label}"] = None
        else:
            # Fill None for missing data
            for label in CYCLES_TO_SAVE:
                row[f"cycle_{label}"] = None

        local_buffer.append(row)

        # 5. Checkpoint to Disk
        if len(local_buffer) >= SAVE_FREQUENCY or i == len(tasks_to_do) - 1:
            save_chunk(local_buffer, my_checkpoint_file)
            local_buffer = []  # Clear memory
            print(
                f"[Rank {rank}] Progress: {i+1}/{len(tasks_to_do)} saved to {my_checkpoint_file}"
            )

    # Wait for all ranks to finish writing their checkpoints
    comm.Barrier()

    # 6. Rank 0 merges everything into one Single File
    if rank == 0:
        print("[Master] merging checkpoint files into single Parquet file...")
        all_dfs = []
        for r in range(size):
            cp_file = os.path.join(RESULTS_DIR, f"checkpoint_rank_{r}.parquet")
            if os.path.exists(cp_file):
                try:
                    df_r = pd.read_parquet(cp_file)
                    all_dfs.append(df_r)
                except Exception as e:
                    print(f"[Warning] Could not read checkpoint {cp_file}: {e}")

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df.to_parquet(FINAL_OUTPUT_FILE, index=False)
            print(
                f"[Done] Successfully saved {len(final_df)} runs to {FINAL_OUTPUT_FILE}"
            )
        else:
            print("[Error] No results found to merge.")


if __name__ == "__main__":
    run_all_mpi()
