import os
import sys
import time
import argparse
import logging
import pandas as pd
import numpy as np
import pybamm
from mpi4py import MPI
import fcntl
from threading import Lock

file_lock = Lock()

# ==========================================
# 1. CONFIGURATION
# ==========================================

N_CYCLES_MAX = 1000
TERMINATION = "80% capacity"
CYCLES_TO_SAVE = ["first", "middle", "last"]
EOL_FRACTION = 0.8
RUL_FIT_WINDOW = 10

SOLVER_MODE = "safe"
SOLVER_RTOL = 1e-6
SOLVER_ATOL = 1e-6

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

pybamm.set_logging_level("NOTICE")


class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


# Configure after MPI initialization
def setup_logging(rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | [Rank %(rank)s] | %(message)s")
    )
    handler.addFilter(RankFilter(rank))

    logger.addHandler(handler)


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
    """Extract cycle data with additional degradation metrics."""
    try:
        data = {
            "Time_h": cycle_sol["Time [h]"].data.tolist(),
            "Voltage_V": cycle_sol["Voltage [V]"].data.tolist(),
            "Current_A": cycle_sol["Current [A]"].data.tolist(),
        }
        try:
            data["SEI_thickness_m"] = cycle_sol["Total SEI thickness [m]"].data.tolist()
        except:
            pass

        try:
            data["LAM_neg_pct"] = cycle_sol[
                "Loss of active material in negative electrode [%]"
            ].data.tolist()
        except:
            pass

        try:
            data["LAM_pos_pct"] = cycle_sol[
                "Loss of active material in positive electrode [%]"
            ].data.tolist()
        except:
            pass

        return data
    except Exception:
        return None


def analyze_run_results(sol, parameter_values, eol_fraction=0.8, rul_window=10):
    """Enhanced analysis with SOH and RUL tracking per cycle."""
    results = {}

    # Get nominal capacity
    try:
        cap_nom = float(parameter_values["Nominal cell capacity [A.h]"])
    except:
        cap_nom = np.nan

    # Extract capacities per cycle
    cycles_idx = []
    caps_ah = []

    for k, cycle in enumerate(sol.cycles):
        cap = capacity_from_cycle(cycle)
        if cap is not None:
            cycles_idx.append(k + 1)
            caps_ah.append(cap)
        else:
            cycles_idx.append(k + 1)
            caps_ah.append(np.nan)

    cycles_arr = np.array(cycles_idx, dtype=float)
    caps_arr = np.array(caps_ah, dtype=float)

    # Use first valid capacity if nominal is unknown
    if (np.isnan(cap_nom) or cap_nom <= 0) and np.any(~np.isnan(caps_arr)):
        first_valid = caps_arr[~np.isnan(caps_arr)]
        if len(first_valid) > 0:
            cap_nom = first_valid[0]

    # Calculate SOH per cycle
    soh_pct = (
        (caps_arr / cap_nom) * 100.0 if cap_nom > 0 else np.full_like(caps_arr, np.nan)
    )

    # Determine EoL
    cap_eol = eol_fraction * cap_nom if cap_nom > 0 else np.nan
    eol_measured = None
    eol_pred = None

    finite_mask = np.isfinite(caps_arr)
    if np.any(finite_mask) and np.isfinite(cap_eol):
        finite_cycles = cycles_arr[finite_mask]
        finite_caps = caps_arr[finite_mask]
        below = np.where(finite_caps <= cap_eol)[0]
        if len(below) > 0:
            eol_measured = int(finite_cycles[below[0]])
        else:
            eol_pred = estimate_eol_linear(
                finite_cycles, finite_caps, cap_eol, window=rul_window
            )

    # Calculate RUL per cycle
    rul_cycles = np.full_like(cycles_arr, np.nan, dtype=float)
    if eol_measured is not None:
        rul_cycles = np.maximum(eol_measured - cycles_arr, 0.0)
    elif eol_pred is not None and np.isfinite(eol_pred):
        rul_cycles = np.maximum(eol_pred - cycles_arr, 0.0)

    # Store results
    results["nominal_capacity_Ah"] = float(cap_nom) if np.isfinite(cap_nom) else None
    results["eol_capacity_Ah"] = float(cap_eol) if np.isfinite(cap_eol) else None
    results["eol_fraction"] = float(eol_fraction)
    results["eol_cycle_measured"] = (
        int(eol_measured) if eol_measured is not None else None
    )
    results["eol_cycle_predicted"] = (
        float(eol_pred) if eol_pred is not None and np.isfinite(eol_pred) else None
    )
    results["n_valid_cycles"] = int(np.sum(finite_mask))

    # Store per-cycle arrays
    results["capacity_trend_cycles"] = cycles_idx
    results["capacity_trend_ah"] = caps_ah
    results["soh_trend_pct"] = soh_pct.tolist()
    results["rul_trend_cycles"] = rul_cycles.tolist()

    # Final RUL
    if eol_measured is not None:
        results["final_RUL"] = 0
    elif eol_pred is not None and np.isfinite(eol_pred):
        current_cycle = cycles_arr[-1] if len(cycles_arr) > 0 else 0
        results["final_RUL"] = max(0, float(eol_pred - current_cycle))
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
        return max(0.5, min(b, 10.0))
    except:
        return 1.5


def save_sample_result(sample_id, data_rows, output_dir):
    """
    Save all parameter results for a single sample to one parquet file.
    Thread-safe and MPI-safe using file locking.
    """
    if not data_rows:
        return

    filepath = os.path.join(output_dir, f"sample_{sample_id}.parquet")
    df_new = pd.DataFrame(data_rows)

    with file_lock:
        try:
            if os.path.exists(filepath):
                with open(filepath, "r+b") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        df_existing = pd.read_parquet(filepath)
                        df_combined = pd.concat(
                            [df_existing, df_new], ignore_index=True
                        )
                        df_combined.to_parquet(filepath, index=False)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            else:
                df_new.to_parquet(filepath, index=False)
        except Exception as e:
            logging.error(f"Error saving to {filepath}: {e}")
            df_new.to_parquet(filepath, index=False)


def get_completed_params_for_sample(sample_id, output_dir):
    """Returns set of param_ids already completed for this sample."""
    filepath = os.path.join(output_dir, f"sample_{sample_id}.parquet")
    if not os.path.exists(filepath):
        return set()
    try:
        df = pd.read_parquet(filepath, columns=["param_id"])
        return set(df["param_id"].values)
    except Exception:
        return set()


# ==========================================
# 3. MPI WORKER
# ==========================================


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Setup logging for this rank
    setup_logging(rank)

    # Argument Parsing (only rank 0)
    input_tau_csv = "taufactor_results.csv"
    input_params_csv = "master_parameters.csv"
    output_dir = "final_output"

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

        logging.info(f"[Master] Starting MPI run with {size} ranks")
        logging.info(f"[Master] Input tau CSV: {input_tau_csv}")
        logging.info(f"[Master] Input params CSV: {input_params_csv}")
        logging.info(f"[Master] Output directory: {output_dir}")

    # Broadcast config to all ranks
    input_tau_csv = comm.bcast(input_tau_csv, root=0)
    input_params_csv = comm.bcast(input_params_csv, root=0)
    output_dir = comm.bcast(output_dir, root=0)

    # 1. READ DATA (rank 0 loads, then distributes)
    my_structures = []
    all_params = []

    if rank == 0:
        # Load all structures
        if os.path.exists(input_tau_csv):
            df_tau = pd.read_csv(input_tau_csv)
            # Ensure consistent ID naming
            if "id" in df_tau.columns:
                df_tau.rename(columns={"id": "sample_id"}, inplace=True)
            structures_list = df_tau.to_dict("records")
            logging.info(
                f"[Master] Loaded {len(structures_list)} structures from {input_tau_csv}"
            )
        else:
            logging.error(f"[Master] {input_tau_csv} not found!")
            sys.exit(1)

        # Load all parameters
        if os.path.exists(input_params_csv):
            df_params = pd.read_csv(input_params_csv)
            if "param_id" not in df_params.columns:
                df_params["param_id"] = np.arange(len(df_params))
            all_params = df_params.to_dict("records")
            logging.info(
                f"[Master] Loaded {len(all_params)} parameter sets from {input_params_csv}"
            )
        else:
            logging.error(f"[Master] {input_params_csv} not found!")
            sys.exit(1)

        # Split structures among ranks (each rank gets different samples)
        # All ranks get ALL parameters (to create cross-join distributedly)
        structures_split = np.array_split(structures_list, size)
        logging.info(f"[Master] Distributing samples across {size} ranks")
    else:
        structures_split = None
        all_params = None

    # Scatter structures (each rank gets subset of samples)
    my_structures = comm.scatter(structures_split, root=0)
    # Broadcast ALL parameters to all ranks
    all_params = comm.bcast(all_params, root=0)

    if rank == 0:
        logging.info(f"[Master] Data distribution complete")

    logging.info(f"[Rank {rank}] Received {len(my_structures)} samples to process")
    logging.info(f"[Rank {rank}] Will run {len(all_params)} parameter sets per sample")
    logging.info(f"[Rank {rank}] Total tasks: {len(my_structures) * len(all_params)}")

    # 2. WORK LOOP - Process by SAMPLE (each sample = one parquet file)
    total_samples = len(my_structures)
    overall_start = time.time()

    for sample_idx, struct_row in enumerate(my_structures):
        s_id = struct_row.get("sample_id")
        sample_start = time.time()

        logging.info(f"[Rank {rank}] ========================================")
        logging.info(
            f"[Rank {rank}] Processing sample {s_id} ({sample_idx+1}/{total_samples})"
        )
        logging.info(f"[Rank {rank}] ========================================")

        # Check what's already completed for THIS sample
        completed_params = get_completed_params_for_sample(s_id, output_dir)
        remaining_params = len(all_params) - len(completed_params)

        if len(completed_params) > 0:
            logging.info(
                f"[Rank {rank}] Sample {s_id}: Found {len(completed_params)} completed params, {remaining_params} remaining"
            )

        # Extract structural properties for PyBaMM
        porosity = struct_row.get("porosity_measured", 0.3)
        tau = struct_row.get("tau_factor", 1.5)
        b_exp = calculate_bruggeman(porosity, tau)

        logging.info(
            f"[Rank {rank}] Sample {s_id}: porosity={porosity:.4f}, tau={tau:.4f}, bruggeman={b_exp:.4f}"
        )

        # Buffer for this sample's results
        sample_results = []
        params_processed = 0
        params_skipped = 0
        params_failed = 0

        # Iterate all parameters for this sample
        for param_idx, param_row in enumerate(all_params):
            p_id = param_row.get("param_id")

            # Skip if already done
            if p_id in completed_params:
                params_skipped += 1
                continue

            t0 = time.time()

            # Prepare result row
            result_row = {
                "sample_id": s_id,
                "param_id": p_id,
                "filename": struct_row.get("filename"),
                "porosity": porosity,
                "tau_factor": tau,
                "bruggeman_derived": b_exp,
            }
            # Copy input metadata
            result_row.update({f"input_{k}": v for k, v in param_row.items()})

            # Setup PyBaMM
            parameter_values = pybamm.ParameterValues("Mohtat2020")

            # Update 1: From master_parameters.csv
            p_dict = {
                k: v for k, v in param_row.items() if k not in ["param_id", "key"]
            }
            parameter_values.update(p_dict, check_already_exists=False)

            # Update 2: From structure (tau factor results)
            struct_updates = {
                "Positive electrode porosity": porosity,
                "Positive electrode Bruggeman coefficient (electrolyte)": b_exp,
                "Positive electrode active material volume fraction": (1.0 - porosity)
                * 0.95,
            }
            parameter_values.update(struct_updates, check_already_exists=False)

            # Build model
            model = pybamm.lithium_ion.DFN({"SEI": "ec reaction limited"})
            try:
                parameter_values.set_initial_stoichiometries(1)
            except:
                pass

            # Create simulation with enhanced solver options for stability
            try:
                sim = pybamm.Simulation(
                    model,
                    experiment=EXPERIMENT,
                    parameter_values=parameter_values,
                    solver=pybamm.CasadiSolver(
                        mode=SOLVER_MODE, rtol=SOLVER_RTOL, atol=SOLVER_ATOL
                    ),
                )
            except Exception as e:
                # Fallback to default solver if CasadiSolver fails
                logging.warning(
                    f"[Rank {rank}] CasadiSolver failed for sample {s_id}, param {p_id}, using default"
                )
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
                error_message = str(e)[:500]  # Limit error message length
                params_failed += 1
                logging.error(
                    f"[Rank {rank}] Sample {s_id}, Param {p_id} FAILED: {error_message[:100]}..."
                )

            # Record results
            runtime = time.time() - t0
            result_row["status"] = "success" if run_ok else "failed"
            result_row["runtime_s"] = round(runtime, 2)
            result_row["error_message"] = error_message

            # Analyze if successful
            if run_ok and sol:
                try:
                    analysis = analyze_run_results(
                        sol,
                        parameter_values,
                        eol_fraction=EOL_FRACTION,
                        rul_window=RUL_FIT_WINDOW,
                    )
                    result_row.update(analysis)

                    if (param_idx + 1) % 20 == 0 or param_idx == 0:
                        cycles_done = analysis.get("n_valid_cycles", "?")
                        eol_meas = analysis.get("eol_cycle_measured", "N/A")
                        eol_pred = analysis.get("eol_cycle_predicted", "N/A")
                        logging.info(
                            f"[Rank {rank}] Sample {s_id}, Param {p_id}: {cycles_done} cycles, EoL={eol_meas}/{eol_pred}, {runtime:.1f}s"
                        )

                except Exception as e:
                    logging.warning(
                        f"[Rank {rank}] Analysis failed for sample {s_id}, param {p_id}: {e}"
                    )
                    result_row["error_message"] += f" | Analysis err: {str(e)[:200]}"

                # Extract selected cycles (first, middle, last)
                num_cycles = len(sol.cycles)
                indices_map = {
                    "first": 0,
                    "middle": num_cycles // 2,
                    "last": num_cycles - 1,
                }
                for label in CYCLES_TO_SAVE:
                    idx = indices_map.get(label)
                    if idx is not None and 0 <= idx < num_cycles:
                        c_data = extract_cycle_data(sol.cycles[idx])
                        if c_data is not None:
                            result_row[f"cycle_{label}"] = c_data

            sample_results.append(result_row)
            params_processed += 1

            # Periodic progress log (every 10 params)
            if params_processed % 10 == 0:
                elapsed = time.time() - sample_start
                avg_time = elapsed / params_processed
                remaining = remaining_params - params_processed
                eta = avg_time * remaining
                logging.info(
                    f"[Rank {rank}] Sample {s_id} progress: {params_processed}/{remaining_params} params ({params_failed} failed), ETA: {eta/60:.1f} min"
                )

        # Save all results for this sample to ONE parquet file
        if sample_results:
            save_sample_result(s_id, sample_results, output_dir)
            sample_elapsed = time.time() - sample_start
            logging.info(
                f"[Rank {rank}] ✓ Sample {s_id} COMPLETE: {len(sample_results)} new results saved ({params_failed} failed, {params_skipped} skipped) in {sample_elapsed/60:.1f} min"
            )
        else:
            logging.info(
                f"[Rank {rank}] Sample {s_id}: No new results (all {len(completed_params)} params already done)"
            )

    # All samples done for this rank
    overall_elapsed = time.time() - overall_start
    logging.info(f"[Rank {rank}] ========================================")
    logging.info(f"[Rank {rank}] ALL SAMPLES COMPLETE")
    logging.info(
        f"[Rank {rank}] Processed {total_samples} samples in {overall_elapsed/60:.1f} minutes"
    )
    logging.info(f"[Rank {rank}] ========================================")

    # Wait for all ranks to finish
    comm.Barrier()

    if rank == 0:
        logging.info("[Master] ========================================")
        logging.info("[Master] ALL RANKS FINISHED!")
        logging.info(f"[Master] Results directory: {os.path.abspath(output_dir)}")
        logging.info(
            f"[Master] Output format: sample_<ID>.parquet (one file per sample)"
        )

        # Count output files
        try:
            parquet_files = [
                f for f in os.listdir(output_dir) if f.endswith(".parquet")
            ]
            logging.info(f"[Master] Total parquet files created: {len(parquet_files)}")
        except:
            pass

        logging.info("[Master] ========================================")


if __name__ == "__main__":
    main()
