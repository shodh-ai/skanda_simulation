import os
import json
import time
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
RESULTS_ROOT = "results"
N_CYCLES_MAX = 500
TERMINATION = "80% capacity"

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

EXPORT_PLOT_CYCLES = ["first", "middle", "last"]
pybamm.set_logging_level("ERROR")  # Higher level to avoid console flooding


# -----------------------------
# Re-usable Utilities from original script
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def select_cycle_indices(sol, labels):
    num = len(sol.cycles)
    indices = []
    for lbl in labels:
        if isinstance(lbl, int):
            if 0 <= lbl < num:
                indices.append(lbl)
        elif isinstance(lbl, str):
            s = lbl.lower()
            if s == "first" and num > 0:
                indices.append(0)
            elif s == "middle" and num > 0:
                indices.append(num // 2)
            elif s == "last" and num > 0:
                indices.append(num - 1)
    return sorted(set([i for i in indices if i is not None]))


def export_cycle_csv_and_plot(cycle, out_dir: str, tag: str):
    try:
        t = cycle["Time [h]"].data
        I = cycle["Current [A]"].data
        V = cycle["Voltage [V]"].data
        df = pd.DataFrame({"Time [h]": t, "Current [A]": I, "Voltage [V]": V})
        df.to_csv(os.path.join(out_dir, f"cycle_{tag}.csv"), index=False)

        plt.figure(figsize=(8, 4.5))
        plt.plot(t, V, lw=1.5)
        plt.xlabel("Time [h]")
        plt.ylabel("Voltage [V]")
        plt.title(f"Cycle {tag}: Voltage vs Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cycle_{tag}_voltage.png"), dpi=200)
        plt.close()
    except Exception as e:
        with open(os.path.join(out_dir, f"cycle_{tag}_export_error.txt"), "w") as f:
            f.write(str(e))


def try_extract_summary(sol) -> dict:
    summary = {"cycles_completed": len(sol.cycles)}
    candidates = [
        "Capacity [A.h]",
        "Discharge capacity [A.h]",
        "Total SEI thickness [m]",
        "Loss of lithium inventory [%]",
        "Loss of active material in negative electrode [%]",
    ]
    found = {}
    for name in candidates:
        try:
            v = sol.summary_variables[name]
            arr = getattr(v, "data", None) or getattr(v, "entries", None)
            if arr is not None:
                found[name] = float(arr[-1])
        except:
            pass
    summary["summary_vars_last"] = found
    return summary


# -----------------------------
# Main MPI Runner
# -----------------------------
def run_all_mpi():
    # 1. Rank 0 loads the tasks
    if rank == 0:
        ensure_dir(RESULTS_ROOT)
        if not os.path.exists(MASTER_CSV):
            print(f"[Error] {MASTER_CSV} not found! Run the aggregation script first.")
            comm.Abort()

        df_master = pd.read_csv(MASTER_CSV)
        all_tasks = df_master.to_dict("records")
        print(f"[MPI Master] {len(all_tasks)} simulations found. Using {size} ranks.")
    else:
        all_tasks = None

    # Broadcast tasks to all ranks
    all_tasks = comm.bcast(all_tasks, root=0)

    # 2. Split tasks among ranks
    my_tasks = np.array_split(all_tasks, size)[rank]
    local_summary_rows = []

    for task in my_tasks:
        run_name = task["run_name"]
        run_dir = os.path.join(RESULTS_ROOT, run_name)
        ensure_dir(run_dir)

        print(f"[Rank {rank}] Running: {run_name}")
        start_time = time.time()

        # Save a copy of parameters in the run folder (as requested)
        pd.DataFrame([task]).to_csv(
            os.path.join(run_dir, "parameters_used.csv"), index=False
        )

        # Build Parameter Set
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        # Direct dictionary update from the master CSV row
        params_to_update = {
            k: v for k, v in task.items() if k not in ["run_name", "source_file"]
        }
        parameter_values.update(params_to_update, check_already_exists=False)

        # Model Setup
        model = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})
        try:
            parameter_values.set_initial_stoichiometries(1)
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
            error_message = "".join(traceback.format_exception_only(type(e), e))
            with open(os.path.join(run_dir, "error.txt"), "w") as f:
                f.write(error_message)

        runtime_s = time.time() - start_time

        # Create output row for global summary
        row = {
            "run": run_name,
            "ok": run_ok,
            "runtime_s": f"{runtime_s:.2f}",
            "cycles_completed": len(sol.cycles) if run_ok and sol else 0,
        }

        # Export Files (JSON, CSVs, Plots) if successful
        if run_ok and sol is not None:
            summary_data = try_extract_summary(sol)
            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                json.dump(summary_data, f, indent=2)

            idxs = select_cycle_indices(sol, EXPORT_PLOT_CYCLES)
            for cidx in idxs:
                export_cycle_csv_and_plot(sol.cycles[cidx], run_dir, f"{cidx:04d}")

            # Simple file list for consistency with your old script
            with open(os.path.join(run_dir, "files.txt"), "w") as f:
                f.write("\n".join(sorted(os.listdir(run_dir))))

        local_summary_rows.append(row)

    # 3. Gather all summaries back to Rank 0
    all_gathered = comm.gather(local_summary_rows, root=0)

    if rank == 0:
        flat_results = [item for sublist in all_gathered for item in sublist]
        summary_path = os.path.join(RESULTS_ROOT, "summary_all_runs.csv")
        pd.DataFrame(flat_results).to_csv(summary_path, index=False)
        print(f"\n[Done] All simulations finished. Global summary at: {summary_path}")


if __name__ == "__main__":
    run_all_mpi()
