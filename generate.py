"""
generate.py
===========
Distributed dataset generation for 100k battery microstructure samples.
Uses MPI for HPC parallelism via mpi4py. Falls back to single-process
if mpi4py is not available.

Usage
-----
    # Single process:
    python generate.py --n 1000

    # MPI (HPC):
    mpirun -n 64 python generate.py --n 100000

    # With srun (SLURM):
    srun -n 64 python generate.py --n 100000

Output
------
    images/                  ← per-sample .tiff volumes
    res.parquet              ← full results table (rank 0 merges)
    res.csv                  ← CSV fallback if pyarrow unavailable
    generate.log             ← per-rank log lines
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from mpi4py import MPI

from structure.schema import GenConfig, SimConfig

from structure.sampling import LHSSampler, N_GEN_DIMS
from structure import (
    load_materials_db,
    resolve_generation,
    resolve_simulation,
    run_generation,
    run_simulation,
    PipelineResult,
    SimulationResult,
)

_comm = MPI.COMM_WORLD
RANK = _comm.Get_rank()
SIZE = _comm.Get_size()
_MPI = True
# ── Hardcoded run parameters ───────────────────────────────────────────────────
SEED: int = 42
START_ID: int = 0
DB_PATH: str = "materials_db.yml"
SIM_CFG_PATH: str = "simulation_config.yml"  # unused — sim is sampled
OUT_DIR: Path = Path("output")
IMAGES_DIR: Path = OUT_DIR / "images"
MAX_RETRIES: int = 10

# C-rates used in rate capability (must match _sim_map.py fixed c_rates list)
_RC_CRATES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


# ── Logging ───────────────────────────────────────────────────────────────────


def _setup_logging() -> logging.Logger:
    fmt = f"[rank {RANK:>4}] %(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUT_DIR / f"generate_rank{RANK:04d}.log", mode="w"),
        ],
    )
    return logging.getLogger("generate")


# ── Result extraction ─────────────────────────────────────────────────────────


def _flatten_sim_result(sim_result: SimulationResult) -> dict:
    """Extract all scalar outputs from SimulationResult into a flat dict."""
    row: dict = {}
    nan = float("nan")

    # TauFactor
    if sim_result.taufactor:
        tf = sim_result.taufactor
        row.update(
            {
                "res_tau_ionic": tf.tau_ionic,
                "res_epsilon_ionic": tf.epsilon_ionic,
                "res_Deff_ionic_m2s": tf.D_eff_ionic_m2_s,
                "res_bruggeman": tf.bruggeman_exponent,
                "res_tau_electronic": (
                    tf.tau_electronic if tf.tau_electronic is not None else nan
                ),
                "res_tau_converged": tf.converged,
            }
        )
    else:
        row.update(
            {
                k: nan
                for k in [
                    "res_tau_ionic",
                    "res_epsilon_ionic",
                    "res_Deff_ionic_m2s",
                    "res_bruggeman",
                    "res_tau_electronic",
                    "res_tau_converged",
                ]
            }
        )

    # Rate capability — one column per c-rate
    if sim_result.rate_capability:
        rc = sim_result.rate_capability
        rc_map = dict(zip(rc.c_rates, rc.capacities_mAh_cm2))
        en_map = dict(zip(rc.c_rates, rc.energy_densities_mWh_cm2))
        for cr in _RC_CRATES:
            tag = f"{cr:.1f}".replace(".", "p")
            row[f"res_rc_cap_mAhcm2_C{tag}"] = rc_map.get(cr, nan)
            row[f"res_rc_energy_mWhcm2_C{tag}"] = en_map.get(cr, nan)
        row["res_nominal_cap_mAhcm2"] = rc.nominal_capacity_mAh_cm2
        row["res_nominal_energy_mWhcm2"] = rc.energy_density_mWh_cm2
    else:
        for cr in _RC_CRATES:
            tag = f"{cr:.1f}".replace(".", "p")
            row[f"res_rc_cap_mAhcm2_C{tag}"] = nan
            row[f"res_rc_energy_mWhcm2_C{tag}"] = nan
        row["res_nominal_cap_mAhcm2"] = nan
        row["res_nominal_energy_mWhcm2"] = nan

    # DCIR
    if sim_result.dcir:
        dc = sim_result.dcir
        row.update(
            {
                "res_dcir_mOhm_cm2": dc.dcir_mOhm_cm2,
                "res_deltaV_mV": dc.delta_V_mV,
            }
        )
    else:
        row.update({"res_dcir_mOhm_cm2": nan, "res_deltaV_mV": nan})

    # Cycle life
    if sim_result.cycle_life:
        cl = sim_result.cycle_life
        row.update(
            {
                "res_projected_cycle_life": (
                    cl.projected_cycle_life if cl.projected_cycle_life else nan
                ),
                "res_capacity_fade_pct_per_cycle": cl.capacity_fade_rate_pct_per_cycle,
                "res_initial_cap_mAhcm2": cl.initial_capacity_mAh_cm2,
                "res_final_cap_mAhcm2": cl.final_capacity_mAh_cm2,
                "res_retention_at_final": cl.retention_at_final_cycle,
            }
        )
    else:
        row.update(
            {
                k: nan
                for k in [
                    "res_projected_cycle_life",
                    "res_capacity_fade_pct_per_cycle",
                    "res_initial_cap_mAhcm2",
                    "res_final_cap_mAhcm2",
                    "res_retention_at_final",
                ]
            }
        )

    return row


# ── Per-sample processing ─────────────────────────────────────────────────────


def _process_one(
    idx: int,
    gen_cfg: GenConfig,
    sim_cfg: SimConfig,
    db,
    log: logging.Logger,
) -> dict | None:
    """
    Run generation + simulation for one sample.
    Returns a CSV-ready dict, or None if generation failed all retries.
    """
    nan = float("nan")

    # ── Generation ────────────────────────────────────────────────────────────
    try:
        resolved_gen = resolve_generation(gen_cfg, db)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline: PipelineResult = run_generation(
                resolved_gen, max_retries=MAX_RETRIES, verbose=False
            )
    except RuntimeError as exc:
        log.warning(f"[{idx}] Generation failed after {MAX_RETRIES} retries: {exc}")
        return None
    except Exception as exc:
        log.warning(f"[{idx}] Generation unexpected error: {type(exc).__name__}: {exc}")
        return None

    vol = pipeline.volume

    # ── Save volume ───────────────────────────────────────────────────────────
    filename = f"{idx:08d}.tiff"
    tiff_path = IMAGES_DIR / filename
    vol.save_tiff(str(tiff_path))

    # ── Simulation ────────────────────────────────────────────────────────────
    sim_result = None
    sim_elapsed = nan
    try:
        resolved_sim = resolve_simulation(sim_cfg, db)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim_result: SimulationResult = run_simulation(
                vol=vol, sim=resolved_sim, verbose=False
            )
        sim_elapsed = sim_result.elapsed_s
    except Exception as exc:
        log.warning(f"[{idx}] Simulation failed: {exc}")

    # ── Assemble row ──────────────────────────────────────────────────────────
    row: dict = {
        "id": idx,
        "filename": filename,
        "rank": RANK,
        "seed_used": pipeline.seed_used,
        "attempts": pipeline.attempts,
        "gen_elapsed_s": round(pipeline.elapsed_s, 3),
        "sim_elapsed_s": round(sim_elapsed, 3) if not math.isnan(sim_elapsed) else nan,
        "sim_ok": sim_result is not None,
        **gen_cfg.to_flat_dict(),
        **sim_cfg.to_flat_dict(),
        **(
            _flatten_sim_result(sim_result)
            if sim_result
            else {k: nan for k in _flatten_sim_result.__code__.co_varnames}
        ),  # NaN placeholders
    }

    # Ensure sim result NaN placeholders are always present
    if sim_result is None:
        nan_row = _flatten_sim_result(
            type(
                "_R",
                (),
                {
                    "taufactor": None,
                    "rate_capability": None,
                    "dcir": None,
                    "cycle_life": None,
                },
            )()
        )
        row.update(nan_row)

    log.info(
        f"[{idx}] done  gen={pipeline.elapsed_s:.1f}s  "
        f"attempts={pipeline.attempts}  sim_ok={sim_result is not None}"
    )
    return row


# ── Save helpers ──────────────────────────────────────────────────────────────


def _save_parquet(rows: list[dict], path: Path) -> None:
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        # Also write CSV as human-readable backup
        df.to_csv(path.with_suffix(".csv"), index=False)
    except ImportError:
        # pandas not available — fall back to plain CSV
        if not rows:
            return
        csv_path = path.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0].keys()), extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate battery microstructure dataset."
    )
    parser.add_argument("--n", type=int, required=True, help="Total number of samples.")
    args = parser.parse_args()
    n = args.n

    # ── Setup dirs (rank 0 only) ───────────────────────────────────────────────
    if RANK == 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if _MPI:
        _comm.Barrier()  # all ranks wait until dirs exist

    log = _setup_logging()

    if RANK == 0:
        log.info(f"Starting dataset generation: n={n}  seed={SEED}  ranks={SIZE}")
        log.info(f"Output → {OUT_DIR.resolve()}")

    # ── Load DB (every rank loads independently — read-only) ──────────────────
    db = load_materials_db(DB_PATH)

    # ── Build LHS (deterministic — every rank builds the same matrix) ─────────
    # This avoids any MPI broadcast of a potentially large array.
    sampler = LHSSampler(n=n, seed=SEED, start_id=START_ID)
    unit = sampler._get_unit()  # shape (n, N_GEN_DIMS + N_SIM_DIMS)

    # ── Distribute work: round-robin across ranks ──────────────────────────────
    my_indices = list(range(RANK, n, SIZE))
    log.info(
        f"Assigned {len(my_indices)} samples "
        f"(indices {my_indices[0]}…{my_indices[-1]})"
    )

    # ── Process ───────────────────────────────────────────────────────────────
    from structure.sampling._gen_map import map_gen_config
    from structure.sampling._sim_map import map_sim_config

    my_rows: list[dict] = []
    for idx in my_indices:
        run_id = START_ID + idx
        gen_kwargs = map_gen_config(unit[idx, :N_GEN_DIMS], run_id=run_id, seed=run_id)
        sim_kwargs = map_sim_config(unit[idx, N_GEN_DIMS:])

        gen_cfg = GenConfig.model_validate(gen_kwargs)
        sim_cfg = SimConfig.model_validate(sim_kwargs)

        row = _process_one(idx, gen_cfg, sim_cfg, db, log)
        if row is not None:
            my_rows.append(row)

    log.info(
        f"Finished local work: {len(my_rows)}/{len(my_indices)} samples succeeded."
    )

    # ── Gather all rows to rank 0 ─────────────────────────────────────────────
    if _MPI:
        all_rows_nested = _comm.gather(my_rows, root=0)
    else:
        all_rows_nested = [my_rows]

    if RANK == 0:
        all_rows = sorted(
            [r for sublist in all_rows_nested for r in sublist],
            key=lambda r: r["id"],
        )
        log.info(f"Total rows collected: {len(all_rows)} / {n}")

        result_path = OUT_DIR / "res"
        _save_parquet(all_rows, result_path)
        log.info(f"Saved → {result_path.with_suffix('.parquet')}")
        log.info(f"Saved → {result_path.with_suffix('.csv')}")

    if _MPI:
        _comm.Barrier()

    log.info("Done.")


if __name__ == "__main__":
    main()
