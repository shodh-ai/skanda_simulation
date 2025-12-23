import os
import glob
import json
import time
import traceback
import shutil
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import pybamm

# -----------------------------
# User configuration
# -----------------------------
# Folder where your CSV sweeps live (the one that contains the generated files)
SWEEPS_DIR = "pybamm_param_sweeps_sep_microstructure"

# IMPORTANT: pattern that matches the earlier generated filenames
CSV_GLOB = os.path.join(SWEEPS_DIR, "params_sep_micro_*_run_*.csv")

# Root output directory for all results
RESULTS_ROOT = "results"

# Experiment definition (long CCCV ageing with EoL)
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

# Cycles to export/plot (labels or integer indices)
EXPORT_PLOT_CYCLES = ["first", "middle", "last"]

# Logging detail (so you see PyBaMM progress messages)
pybamm.set_logging_level("NOTICE")


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_name_from_csv(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    name, _ = os.path.splitext(base)
    return name


def copy_into(src: str, dst_dir: str, new_name: str = None):
    ensure_dir(dst_dir)
    if new_name is None:
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
    else:
        shutil.copy2(src, os.path.join(dst_dir, new_name))


def apply_params_csv(
    csv_path: str, parameter_values: "pybamm.ParameterValues"
) -> "pybamm.ParameterValues":
    """
    Read 'key,value' rows from csv_path and parameter_values.update() where possible.
    Handles different separator Bruggeman key variants across PyBaMM versions.
    """
    df = pd.read_csv(csv_path)
    brugg_keys = [
        "Separator Bruggeman coefficient (electrolyte)",
        "Separator Bruggeman coefficient",
    ]
    for _, row in df.iterrows():
        key = str(row["key"]).strip()
        val_raw = row["value"]
        # robust float conversion
        try:
            val = float(val_raw)
        except Exception:
            val = val_raw  # leave strings as-is (rare)

        if key in brugg_keys:
            updated = False
            for bk in brugg_keys:
                try:
                    parameter_values.update({bk: val}, check_already_exists=False)
                    updated = True
                    break
                except Exception:
                    pass
            if not updated:
                print(f"[warn] Could not set any separator Bruggeman key to {val}")
            continue

        # Try direct update; if not applicable in this PyBaMM version, skip silently
        try:
            parameter_values.update({key: val}, check_already_exists=False)
        except Exception:
            # print(f"[skip] Key not found or not applicable: {key}")
            pass
    return parameter_values


def select_cycle_indices(sol, labels: List[str]) -> List[int]:
    """Convert labels ['first','middle','last'] to actual indices available in sol.cycles."""
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
    # De-duplicate and sort
    indices = sorted(set([i for i in indices if i is not None]))
    return indices


def export_cycle_csv_and_plot(cycle, out_dir: str, tag: str):
    """
    Save a cycle's time/current/voltage to CSV and a Voltage-vs-Time PNG.
    """
    try:
        t = cycle["Time [h]"].data
        I = cycle["Current [A]"].data
        V = cycle["Voltage [V]"].data
    except Exception as e:
        # If any variable is missing in this model/version, skip exporting for this cycle
        with open(os.path.join(out_dir, f"cycle_{tag}_export_error.txt"), "w") as f:
            f.write("Failed to export cycle variables:\n")
            f.write("".join(traceback.format_exception_only(type(e), e)))
        return

    df = pd.DataFrame({"Time [h]": t, "Current [A]": I, "Voltage [V]": V})
    df.to_csv(os.path.join(out_dir, f"cycle_{tag}.csv"), index=False)

    # Plot Voltage vs Time (save to PNG; no interactive display needed)
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, V, lw=1.5)
    plt.xlabel("Time [h]")
    plt.ylabel("Voltage [V]")
    plt.title(f"Cycle {tag}: Voltage vs Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cycle_{tag}_voltage.png"), dpi=200)
    plt.close()


def try_extract_summary(sol) -> Dict:
    """
    Build a small summary dict, tolerant to PyBaMM version changes.
    """
    summary = {}
    try:
        summary["cycles_completed"] = len(sol.cycles)
    except Exception:
        summary["cycles_completed"] = None

    # Optional: grab a few summary variables if they exist
    candidates = [
        "Capacity [A.h]",
        "Discharge capacity [A.h]",
        "Charge capacity [A.h]",
        "Total SEI thickness [m]",
        "Loss of active material in negative electrode [%]",
        "Loss of active material in positive electrode [%]",
        "Loss of lithium inventory [mol]",
        "Loss of lithium inventory [%]",
        "X-averaged negative particle surface concentration [mol.m-3]",
        "X-averaged positive particle surface concentration [mol.m-3]",
    ]
    found = {}
    for name in candidates:
        try:
            v = sol.summary_variables[name]
            arr = getattr(v, "data", None) or getattr(v, "entries", None)
            if arr is not None:
                found[name] = float(arr[-1])  # last value
        except Exception:
            pass
    summary["summary_vars_last"] = found
    return summary


# -----------------------------
# Main runner
# -----------------------------
def run_all(csv_glob: str = CSV_GLOB, results_root: str = RESULTS_ROOT):
    print(f"[info] Using sweeps directory: {os.path.abspath(SWEEPS_DIR)}")
    print(f"[info] Glob pattern: {csv_glob}")
    ensure_dir(results_root)

    csv_files = sorted(glob.glob(csv_glob))
    print(f"[info] Found {len(csv_files)} CSV files to run.")
    if len(csv_files) > 0:
        print("[info] First 5 files:")
        for f in csv_files[:5]:
            print("   ", f)

    if not csv_files:
        print(
            "[error] No CSV files matched the pattern. Check SWEEPS_DIR and CSV_GLOB."
        )
        print(
            "[tip] Example files earlier were named like 'params_sep_micro_01_run_001.csv'."
        )
        print(
            "[tip] Set CSV_GLOB = os.path.join(SWEEPS_DIR, 'params_sep_micro_*_run_*.csv')"
        )
        return  # don't raise; just return so you see the prints in notebooks

    all_rows = []  # for global CSV summary

    for idx, csv_path in enumerate(csv_files, start=1):
        run_name = run_name_from_csv(csv_path)
        run_dir = os.path.join(results_root, run_name)
        ensure_dir(run_dir)

        print(f"\n=== [{idx}/{len(csv_files)}] Running: {run_name} ===")
        print(f"[info] Parameters file: {csv_path}")
        start_time = time.time()

        # Save a copy of the parameters used
        copy_into(csv_path, run_dir, "parameters_used.csv")

        # Build parameter set
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        # Optional: keep your SEI tweak unless overridden by CSV
        parameter_values.update(
            {"SEI kinetic rate constant [m.s-1]": 1e-14}, check_already_exists=False
        )
        # Apply CSV-specified parameters
        parameter_values = apply_params_csv(csv_path, parameter_values)

        # Build model and set initial SOC
        model = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})
        try:
            parameter_values.set_initial_stoichiometries(1)
        except Exception:
            pass

        # Run experiment
        sim = pybamm.Simulation(
            model, experiment=EXPERIMENT, parameter_values=parameter_values
        )
        run_ok = True
        error_message = ""
        sol = None
        try:
            print("[info] Solving experiment...")
            sol = sim.solve()
            print("[info] Solve complete.")
        except Exception as e:
            run_ok = False
            error_message = "".join(traceback.format_exception_only(type(e), e))
            print("[error] Simulation failed:", error_message)
            with open(os.path.join(run_dir, "error.txt"), "w") as f:
                f.write("Simulation failed:\n")
                f.write(error_message)

        runtime_s = time.time() - start_time
        print(f"[info] Runtime: {runtime_s:.2f} s")

        # Extract and persist outputs
        per_run_row = {
            "run": run_name,
            "csv": os.path.basename(csv_path),
            "ok": run_ok,
            "runtime_s": f"{runtime_s:.2f}",
            "termination": TERMINATION,
        }

        if run_ok and sol is not None:
            # Summary
            summary = try_extract_summary(sol)
            per_run_row.update(
                {
                    "cycles_completed": summary.get("cycles_completed", None),
                }
            )
            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[info] Cycles completed: {per_run_row['cycles_completed']}")

            # Export selected cycles (first/middle/last or explicit indices)
            idxs = select_cycle_indices(sol, EXPORT_PLOT_CYCLES)
            print(f"[info] Exporting cycles: {idxs}")
            for cidx in idxs:
                tag = f"{cidx:04d}"
                cycle = sol.cycles[cidx]
                export_cycle_csv_and_plot(cycle, run_dir, tag)

            # Quick list of saved files:
            with open(os.path.join(run_dir, "files.txt"), "w") as f:
                f.write("\n".join(sorted(os.listdir(run_dir))))
        else:
            per_run_row.update({"cycles_completed": None, "error": error_message})

        all_rows.append(per_run_row)

    # Write global summary
    df_all = pd.DataFrame(all_rows)
    summary_path = os.path.join(results_root, "summary_all_runs.csv")
    df_all.to_csv(summary_path, index=False)
    print("\nAll runs completed.")
    print(f"[info] Global summary: {summary_path}")


# -----------------------------
# Execute
# -----------------------------

if __name__ == "__main__":
    run_all()
