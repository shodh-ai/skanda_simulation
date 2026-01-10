import os
import time
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
RESULTS_DIR = "checkpoints_dfn"  # Changed folder name to separate from SPM runs
FINAL_OUTPUT_FILE = "consolidated_results_dfn.parquet"

N_CYCLES_MAX = 500
TERMINATION = "80% capacity"
SAVE_FREQUENCY = 5  # Reduced frequency slightly because DFN is slower
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
# Utilities: Math & Integration
# -----------------------------
def _first_positive_segment(t: np.ndarray, I: np.ndarray):
    """Return (t_seg, I_seg) for the first contiguous discharge region."""
    if t is None or I is None or len(t) == 0:
        return None
    # Threshold to ignore noise
    thr = max(1e-6, 0.01 * np.nanmax(np.abs(I)))
    mask = I > thr
    if not np.any(mask):
        return None
    start = np.argmax(mask)
    end = start
    n = len(mask)
    while end + 1 < n and mask[end + 1]:
        end += 1
    return t[start : end + 1], I[start : end + 1]


def capacity_from_cycle(cycle):
    """Calculates discharge capacity (Ah) via integration or PyBaMM variable."""
    # 1. Try built-in variable
    try:
        v = cycle["Discharge capacity [A.h]"]
        data = np.asarray(v.data).astype(float)
        if data.size > 0:
            cap = float(np.nanmax(data))
            if np.isfinite(cap) and cap > 0:
                return cap
    except Exception:
        pass

    # 2. Fallback: integrate I over first positive segment
    try:
        t = np.asarray(cycle["Time [h]"].data).astype(float)
        I = np.asarray(cycle["Current [A]"].data).astype(float)
    except Exception:
        return None

    seg = _first_positive_segment(t, I)
    if seg is None:
        return None
    t_seg, I_seg = seg
    # Trapz integration in hours gives A*h
    cap_ah = float(np.trapz(I_seg, t_seg))
    if np.isfinite(cap_ah) and cap_ah > 0:
        return cap_ah
    return None


def estimate_eol_linear(
    cycles: np.ndarray, caps: np.ndarray, cap_eol: float, window: int = 10
):
    """
    Linear least-squares estimate of EoL cycle (where capacity reaches 'cap_eol').
    Uses the last 'window' points. Returns None if slope >= 0 (no fade).
    """
    n = len(cycles)
    if n < 2:
        return None
    w = min(window, n)
    x = cycles[-w:].astype(float)
    y = caps[-w:].astype(float)
    # Fit y = a + b*x
    A = np.vstack([np.ones_like(x), x]).T
    try:
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        return None
    if b >= 0:  # no degradation trend
        return None
    # Solve a + b*x* = cap_eol
    x_star = (cap_eol - a) / b
    if np.isfinite(x_star) and x_star > 0:
        return float(x_star)
    return None


# -----------------------------
# Utilities: File I/O & Extraction
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def extract_cycle_data(cycle_sol):
    """Extracts raw Time, Voltage, Current arrays."""
    try:
        return {
            "Time_h": cycle_sol["Time [h]"].data.tolist(),
            "Voltage_V": cycle_sol["Voltage [V]"].data.tolist(),
            "Current_A": cycle_sol["Current [A]"].data.tolist(),
        }
    except Exception:
        return None


def analyze_run_results(sol, parameter_values, eol_fraction=0.8):
    """
    Computes SOH, Capacity history, and RUL prediction.
    Returns a dict to update the main row.
    """
    results = {}

    # Get Nominal Capacity
    try:
        cap_nom = float(parameter_values["Nominal cell capacity [A.h]"])
    except:
        cap_nom = np.nan

    # Extract capacity per cycle
    cycles_idx = []
    caps_ah = []

    for k, cycle in enumerate(sol.cycles):
        cap = capacity_from_cycle(cycle)
        if cap is not None:
            cycles_idx.append(k + 1)
            caps_ah.append(cap)

    # If we still don't have nominal capacity, guess from first cycle
    if (np.isnan(cap_nom) or cap_nom <= 0) and len(caps_ah) > 0:
        cap_nom = caps_ah[0]

    cycles_arr = np.array(cycles_idx)
    caps_arr = np.array(caps_ah)

    # Save the arrays (lists) into Parquet for later plotting
    results["capacity_trend_cycles"] = cycles_idx
    results["capacity_trend_ah"] = caps_ah

    if len(caps_ah) == 0:
        return results

    # Calculate EoL things
    cap_eol = eol_fraction * cap_nom

    # Measured EoL (Did we actually hit it?)
    eol_measured = None
    below = np.where(caps_arr <= cap_eol)[0]
    if len(below) > 0:
        eol_measured = int(cycles_arr[below[0]])

    # Predicted EoL (Extrapolation)
    eol_pred = estimate_eol_linear(cycles_arr, caps_arr, cap_eol)

    results["nominal_capacity_Ah"] = cap_nom
    results["eol_cycle_measured"] = eol_measured
    results["eol_cycle_predicted"] = eol_pred

    # Calculate RUL at the end of simulation
    current_cycle = cycles_arr[-1]
    if eol_measured:
        results["final_RUL"] = 0
    elif eol_pred:
        results["final_RUL"] = max(0, eol_pred - current_cycle)
    else:
        results["final_RUL"] = None

    return results


def save_chunk(data_buffer, filepath):
    if not data_buffer:
        return
    df_chunk = pd.DataFrame(data_buffer)
    if os.path.exists(filepath):
        try:
            df_existing = pd.read_parquet(filepath)
            df_combined = pd.concat([df_existing, df_chunk], ignore_index=True)
            df_combined.to_parquet(filepath, index=False)
        except:
            df_chunk.to_parquet(filepath, index=False)
    else:
        df_chunk.to_parquet(filepath, index=False)


def get_completed_runs(filepath):
    if not os.path.exists(filepath):
        return set()
    try:
        df = pd.read_parquet(filepath, columns=["run_name"])
        return set(df["run_name"].unique())
    except Exception:
        return set()


# -----------------------------
# Main MPI Runner
# -----------------------------
def run_all_mpi():
    ensure_dir(RESULTS_DIR)

    if rank == 0:
        if not os.path.exists(MASTER_CSV):
            print(f"[Error] {MASTER_CSV} not found!")
            comm.Abort()
        df_master = pd.read_csv(MASTER_CSV)
        all_tasks = df_master.to_dict("records")
        print(f"[Master] Found {len(all_tasks)} simulations. Distributing...")
    else:
        all_tasks = None

    all_tasks = comm.bcast(all_tasks, root=0)
    my_tasks = np.array_split(all_tasks, size)[rank]

    my_checkpoint_file = os.path.join(RESULTS_DIR, f"checkpoint_rank_{rank}.parquet")
    completed_runs = get_completed_runs(my_checkpoint_file)

    tasks_to_do = [t for t in my_tasks if t.get("run_name") not in completed_runs]

    if len(tasks_to_do) < len(my_tasks):
        print(
            f"[Rank {rank}] Resuming: Skipping {len(my_tasks) - len(tasks_to_do)} finished tasks."
        )

    local_buffer = []

    for i, task in enumerate(tasks_to_do):
        row = task.copy()
        start_time = time.time()

        # Build Parameter Set
        parameter_values = pybamm.ParameterValues("Mohtat2020")

        # Coworker's requested tweak:
        parameter_values.update(
            {"SEI kinetic rate constant [m.s-1]": 1e-14}, check_already_exists=False
        )

        params_to_update = {
            k: v for k, v in task.items() if k not in ["run_name", "source_file"]
        }
        # Handle Bruggeman keys safely (as per coworker's logic) if needed,
        # or assume CSV keys are clean. Here we just update directly for simplicity.
        parameter_values.update(params_to_update, check_already_exists=False)

        # ---------------------------------------------------------
        # CHANGE: Use DFN Model (Much slower, but higher fidelity)
        # ---------------------------------------------------------
        model = pybamm.lithium_ion.DFN({"SEI": "ec reaction limited"})

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
            error_message = str(e)

        runtime_s = time.time() - start_time

        row["status"] = "success" if run_ok else "failed"
        row["runtime_s"] = round(runtime_s, 2)
        row["cycles_completed"] = len(sol.cycles) if (run_ok and sol) else 0
        row["error_message"] = error_message

        if run_ok and sol:
            # -----------------------------------------------------
            # CHANGE: Run new Analytics (SOH, RUL, Capacity Arrays)
            # -----------------------------------------------------
            try:
                analysis_data = analyze_run_results(sol, parameter_values)
                row.update(analysis_data)
            except Exception as e:
                row["error_message"] += f" | Analysis failed: {e}"

            # Extract raw cycle data for specific cycles
            num_cycles = len(sol.cycles)
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
            for label in CYCLES_TO_SAVE:
                row[f"cycle_{label}"] = None
            row["capacity_trend_ah"] = []  # Ensure column exists even on fail

        local_buffer.append(row)

        if len(local_buffer) >= SAVE_FREQUENCY or i == len(tasks_to_do) - 1:
            save_chunk(local_buffer, my_checkpoint_file)
            local_buffer = []
            print(f"[Rank {rank}] {i+1}/{len(tasks_to_do)} done.")

    comm.Barrier()

    if rank == 0:
        print("[Master] merging checkpoint files...")
        all_dfs = []
        for r in range(size):
            cp_file = os.path.join(RESULTS_DIR, f"checkpoint_rank_{r}.parquet")
            if os.path.exists(cp_file):
                try:
                    df_r = pd.read_parquet(cp_file)
                    all_dfs.append(df_r)
                except Exception as e:
                    print(f"[Warning] Could not read {cp_file}: {e}")

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df.to_parquet(FINAL_OUTPUT_FILE, index=False)
            print(f"[Done] Saved {len(final_df)} runs to {FINAL_OUTPUT_FILE}")
        else:
            print("[Error] No results found.")


if __name__ == "__main__":
    run_all_mpi()
