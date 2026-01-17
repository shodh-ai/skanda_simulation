import os
import sys
import time
import argparse
import logging
import pandas as pd
import numpy as np
import pybamm
from mpi4py import MPI

# ==========================================
# 1. CONFIGURATION
# ==========================================

N_CYCLES_MAX = 500
TERMINATION = "80% capacity"
SAVE_FREQUENCY = 10  # Save to disk every N simulations
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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | [Rank %(rank)s] | %(message)s"
)

# ==========================================
# 2. UTILITIES (From your Code)
# ==========================================


def _first_positive_segment(t: np.ndarray, I: np.ndarray):
    """Return (t_seg, I_seg) for the first contiguous discharge region."""
    if t is None or I is None or len(t) == 0:
        return None
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
    try:
        v = cycle["Discharge capacity [A.h]"]
        data = np.asarray(v.data).astype(float)
        if data.size > 0:
            cap = float(np.nanmax(data))
            if np.isfinite(cap) and cap > 0:
                return cap
    except Exception:
        pass

    try:
        t = np.asarray(cycle["Time [h]"].data).astype(float)
        I = np.asarray(cycle["Current [A]"].data).astype(float)
    except Exception:
        return None

    seg = _first_positive_segment(t, I)
    if seg is None:
        return None
    t_seg, I_seg = seg
    cap_ah = float(np.trapz(I_seg, t_seg))
    if np.isfinite(cap_ah) and cap_ah > 0:
        return cap_ah
    return None


def estimate_eol_linear(
    cycles: np.ndarray, caps: np.ndarray, cap_eol: float, window: int = 10
):
    n = len(cycles)
    if n < 2:
        return None
    w = min(window, n)
    x = cycles[-w:].astype(float)
    y = caps[-w:].astype(float)
    A = np.vstack([np.ones_like(x), x]).T
    try:
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        return None
    if b >= 0:
        return None
    x_star = (cap_eol - a) / b
    if np.isfinite(x_star) and x_star > 0:
        return float(x_star)
    return None


def extract_cycle_data(cycle_sol):
    try:
        return {
            "Time_h": cycle_sol["Time [h]"].data.tolist(),
            "Voltage_V": cycle_sol["Voltage [V]"].data.tolist(),
            "Current_A": cycle_sol["Current [A]"].data.tolist(),
        }
    except Exception:
        return None


def analyze_run_results(sol, parameter_values, eol_fraction=0.8):
    results = {}
    try:
        cap_nom = float(parameter_values["Nominal cell capacity [A.h]"])
    except:
        cap_nom = np.nan

    cycles_idx = []
    caps_ah = []

    for k, cycle in enumerate(sol.cycles):
        cap = capacity_from_cycle(cycle)
        if cap is not None:
            cycles_idx.append(k + 1)
            caps_ah.append(cap)

    if (np.isnan(cap_nom) or cap_nom <= 0) and len(caps_ah) > 0:
        cap_nom = caps_ah[0]

    cycles_arr = np.array(cycles_idx)
    caps_arr = np.array(caps_ah)

    results["capacity_trend_cycles"] = cycles_idx
    results["capacity_trend_ah"] = caps_ah

    if len(caps_ah) == 0:
        return results

    cap_eol = eol_fraction * cap_nom
    eol_measured = None
    below = np.where(caps_arr <= cap_eol)[0]
    if len(below) > 0:
        eol_measured = int(cycles_arr[below[0]])

    eol_pred = estimate_eol_linear(cycles_arr, caps_arr, cap_eol)

    results["nominal_capacity_Ah"] = cap_nom
    results["eol_cycle_measured"] = eol_measured
    results["eol_cycle_predicted"] = eol_pred

    current_cycle = cycles_arr[-1]
    if eol_measured:
        results["final_RUL"] = 0
    elif eol_pred:
        results["final_RUL"] = max(0, eol_pred - current_cycle)
    else:
        results["final_RUL"] = None

    return results


def calculate_bruggeman(porosity, tau):
    """
    Derives Bruggeman exponent b from Tau and Porosity.
    Tau = Porosity / Porosity^b  =>  Tau = Porosity^(1-b)
    log(Tau) = (1-b)*log(Porosity) => b = 1 - log(Tau)/log(Porosity)
    """
    if porosity <= 0 or porosity >= 1:
        return 1.5
    if tau < 1.0:
        tau = 1.0

    try:
        b = 1 - (np.log(tau) / np.log(porosity))
        # Clamp to avoid numerical instability in PyBaMM
        return max(0.5, min(b, 10.0))
    except:
        return 1.5


def save_chunk(data_buffer, filepath):
    if not data_buffer:
        return
    df_chunk = pd.DataFrame(data_buffer)
    # Ensure complex objects (lists) are handled if parquet engine complains,
    # but pyarrow usually handles lists fine.

    if os.path.exists(filepath):
        try:
            df_existing = pd.read_parquet(filepath)
            df_combined = pd.concat([df_existing, df_chunk], ignore_index=True)
            df_combined.to_parquet(filepath, index=False)
        except:
            df_chunk.to_parquet(filepath, index=False)
    else:
        df_chunk.to_parquet(filepath, index=False)


def get_completed_tasks(filepath):
    """Returns set of (sample_id, param_id) tuples that are already done."""
    if not os.path.exists(filepath):
        return set()
    try:
        df = pd.read_parquet(filepath, columns=["sample_id", "param_id"])
        # Create tuple set
        return set(zip(df["sample_id"], df["param_id"]))
    except Exception:
        return set()


# ==========================================
# 3. MPI WORKER
# ==========================================


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Argument Parsing
    input_tau_csv = "taufactor_results.csv"
    input_params_csv = "master_parameters.csv"
    output_dir = "final_output"

    # Allow overriding via args
    if rank == 0:
        parser = argparse.ArgumentParser()
        parser.add_argument("--tau_csv", default=input_tau_csv)
        parser.add_argument("--params_csv", default=input_params_csv)
        parser.add_argument("--output_dir", default=output_dir)
        args, _ = parser.parse_known_args()

        input_tau_csv = args.tau_csv
        input_params_csv = args.params_csv
        output_dir = args.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Broadcast config
    input_tau_csv = comm.bcast(input_tau_csv, root=0)
    input_params_csv = comm.bcast(input_params_csv, root=0)
    output_dir = comm.bcast(output_dir, root=0)

    # 1. READ DATA
    my_structures = []
    all_params = []

    if rank == 0:
        # Load all structures
        if os.path.exists(input_tau_csv):
            df_tau = pd.read_csv(input_tau_csv)
            # Ensure we have consistent ID naming
            if "id" in df_tau.columns:
                df_tau.rename(columns={"id": "sample_id"}, inplace=True)
            structures_list = df_tau.to_dict("records")
            print(f"[Master] Loaded {len(structures_list)} structures.")
        else:
            print(f"[Error] {input_tau_csv} not found.")
            sys.exit(1)

        # Load all parameters
        if os.path.exists(input_params_csv):
            df_params = pd.read_csv(input_params_csv)
            if "param_id" not in df_params.columns:
                df_params["param_id"] = np.arange(len(df_params))
            all_params = df_params.to_dict("records")
            print(f"[Master] Loaded {len(all_params)} parameter sets.")
        else:
            print(f"[Error] {input_params_csv} not found.")
            sys.exit(1)

        # Split Structures among ranks
        # We only split structures, every rank gets ALL parameters.
        # This creates the cross-join effect distributedly.
        structures_split = np.array_split(structures_list, size)
    else:
        structures_split = None
        all_params = None

    # Scatter Structures
    my_structures = comm.scatter(structures_split, root=0)
    # Broadcast Parameters
    all_params = comm.bcast(all_params, root=0)

    # 2. CHECK RESUME STATUS
    my_checkpoint_file = os.path.join(output_dir, f"results_rank_{rank}.parquet")
    completed_set = get_completed_tasks(my_checkpoint_file)

    # 3. WORK LOOP
    local_buffer = []
    total_tasks = len(my_structures) * len(all_params)
    processed_count = 0

    # Iterate Structures
    for struct_row in my_structures:
        s_id = struct_row.get("sample_id")

        # Extract Structural properties for PyBaMM
        # Assuming Positive Electrode based on context
        porosity = struct_row.get("porosity_measured", 0.3)
        tau = struct_row.get("tau_factor", 1.5)
        # Calculate Bruggeman b
        b_exp = calculate_bruggeman(porosity, tau)

        # Iterate Parameters
        for param_row in all_params:
            p_id = param_row.get("param_id")

            # Skip if done
            if (s_id, p_id) in completed_set:
                processed_count += 1
                continue

            t0 = time.time()

            # Prepare Output Row
            result_row = {
                "sample_id": s_id,
                "param_id": p_id,
                "filename": struct_row.get("filename"),
                "bruggeman_derived": b_exp,
            }
            # Copy input metadata into result
            result_row.update({f"input_{k}": v for k, v in param_row.items()})

            # Setup PyBaMM
            parameter_values = pybamm.ParameterValues("Mohtat2020")

            # Update 1: From master_parameters.csv
            # Filter out non-PyBaMM keys (ids, etc)
            p_dict = {
                k: v for k, v in param_row.items() if k not in ["param_id", "key"]
            }
            parameter_values.update(p_dict, check_already_exists=False)

            # Update 2: From Taufactor (Structure)
            # We enforce the structural properties on the Positive Electrode
            struct_updates = {
                "Positive electrode porosity": porosity,
                "Positive electrode Bruggeman coefficient (electrolyte)": b_exp,
                "Positive electrode active material volume fraction": (1.0 - porosity)
                * 0.95,
            }
            parameter_values.update(struct_updates, check_already_exists=False)

            # Build Model & Sim
            model = pybamm.lithium_ion.DFN({"SEI": "ec reaction limited"})
            try:
                parameter_values.set_initial_state(1)
            except:
                pass

            sim = pybamm.Simulation(
                model, experiment=EXPERIMENT, parameter_values=parameter_values
            )

            # Solve
            run_ok = True
            error_message = ""
            sol = None

            try:
                sol = sim.solve()
            except Exception as e:
                run_ok = False
                error_message = str(e)

            # Analyze
            runtime = time.time() - t0
            result_row["status"] = "success" if run_ok else "failed"
            result_row["runtime_s"] = round(runtime, 2)
            result_row["error_message"] = error_message

            if run_ok and sol:
                try:
                    analysis = analyze_run_results(sol, parameter_values)
                    result_row.update(analysis)
                except Exception as e:
                    result_row["error_message"] += f" | Analysis err: {e}"

                # Extract Cycles
                num_cycles = len(sol.cycles)
                indices_map = {
                    "first": 0,
                    "middle": num_cycles // 2,
                    "last": num_cycles - 1,
                }
                for label in CYCLES_TO_SAVE:
                    idx = indices_map.get(label)
                    if idx is not None and 0 <= idx < num_cycles:
                        # Serialize safely (might be large)
                        # We convert to list to store in Parquet/JSON
                        c_data = extract_cycle_data(sol.cycles[idx])
                        # Flattening for Parquet: Store as string or keep as list?
                        # Parquet handles lists.
                        result_row[f"cycle_{label}"] = c_data

            # Add to buffer
            local_buffer.append(result_row)
            processed_count += 1

            # Periodic Save
            if len(local_buffer) >= SAVE_FREQUENCY:
                save_chunk(local_buffer, my_checkpoint_file)
                local_buffer = []
                print(f"[Rank {rank}] Progress: {processed_count}/{total_tasks}")

    # Final Save
    save_chunk(local_buffer, my_checkpoint_file)
    print(f"[Rank {rank}] Completed.")

    comm.Barrier()
    if rank == 0:
        print("[Master] All ranks finished. Data is in:", output_dir)


if __name__ == "__main__":
    main()
