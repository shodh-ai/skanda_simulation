#!/usr/bin/env python
"""
Multi-GPU Microstructure Parameter Sweep.

- Runs on a single node (detects all GPUs).
- STRICTLY requires FiPy + PyAMGX.
- Automatically generates diverse parameters (Porosity, PSD, Drying) for each sample.
"""

import os
import sys
import time
import yaml
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
import tifffile
import multiprocessing as mp
from scipy.fft import ifftn

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. PARAMETER GENERATION LOGIC
# ==========================================


def generate_parameter_manifest(num_samples, seed):
    """
    Generates a DataFrame of randomized parameters for the sweep.
    Expanded to cover all requested physical and temporal parameters.
    """
    rng = np.random.default_rng(seed)

    # 1. Define all ranges
    RANGES = {
        "target_porosity": (0.20, 0.80),
        "voxel_size_um": (0.08, 0.15),
        "psd_power": (0.8, 2.2),
        "anisotropy_x": (0.7, 1.3),
        "anisotropy_y": (0.7, 1.3),
        "anisotropy_z": (0.7, 1.3),
        "drying_intensity": (0.1, 0.7),
        "velocity_scale": (0.2, 1.0),
        "time_steps": (20, 80),
        "dt": (0.03, 0.10),
        "diff_pore": (0.5, 1.5),
        "diff_solid": (0.05, 0.30),
        "evap_strength": (4.0, 8.0),
        "redeposition_factor": (0.03, 0.12),
        "top_mask_zmin": (0.70, 0.85),
    }

    # 2. Generate LHS Matrix
    keys = list(RANGES.keys())
    n_dims = len(keys)
    H = lhs(num_samples, n_dims, rng)

    # 3. Map [0,1] to actual ranges
    data = {"id": np.arange(num_samples)}
    for i, key in enumerate(keys):
        vmin, vmax = RANGES[key]
        data[key] = vmin + H[:, i] * (vmax - vmin)

    # 4. Enforce Integer Types where necessary
    data["time_steps"] = data["time_steps"].astype(int)

    df = pd.DataFrame(data)

    # 5. Assign Methods (Physics vs GRF)
    methods = ["physics"] * num_samples
    num_grf = max(1, int(num_samples * 0.2))
    for i in range(num_grf):
        methods[i] = "GRF"

    rng.shuffle(methods)
    df["method"] = methods

    return df


# ==========================================
# 2. GPU SOLVER & PHYSICS ENGINE
# ==========================================


def lhs(n_samples, n_dims, rng):
    """Simple Latin Hypercube Sampling (NumPy only)."""
    H = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        cut = np.linspace(0, 1, n_samples + 1)
        u = rng.uniform(low=cut[:-1], high=cut[1:], size=n_samples)
        P = rng.permutation(n_samples)
        H[:, j] = u[P]
    return H


def get_gpu_solver_class():
    """Lazy loader for GPU libraries to prevent context init in parent process."""
    try:
        import pyamgx
        from fipy.solvers.solver import Solver as _FipySolver
        from fipy.matrices.scipyMatrix import _ScipyMeshMatrix
        from fipy.tools import numerix
        from scipy.sparse import linalg
    except ImportError as e:
        logger.critical(f"MISSING GPU DEPENDENCY: {e}")
        sys.exit(1)

    class GPUAMGXSolver(_FipySolver):
        """PyAMGX Solver Wrapper for FiPy."""

        CONFIG = {
            "config_version": 2,
            "determinism_flag": 1,
            "solver": {
                "algorithm": "AGGREGATION",
                "solver": "AMG",
                "selector": "SIZE_2",
                "monitor_residual": 1,
                "max_levels": 1000,
                "cycle": "V",
                "smoother": "BLOCK_JACOBI",
                "presweeps": 2,
                "postsweeps": 2,
            },
        }

        def __init__(self, tolerance=1e-6, iterations=100):
            super().__init__(
                tolerance=tolerance, criterion="default", iterations=iterations
            )
            cfg_dict = self.CONFIG.copy()
            cfg_dict["solver"]["max_iters"] = iterations
            cfg_dict["solver"]["tolerance"] = tolerance
            cfg_dict["solver"]["convergence"] = "RELATIVE_INI_CORE"

            self.cfg = pyamgx.Config().create_from_dict(cfg_dict)
            self.resources = pyamgx.Resources().create_simple(self.cfg)
            self.x_gpu = pyamgx.Vector().create(self.resources)
            self.b_gpu = pyamgx.Vector().create(self.resources)
            self.A_gpu = pyamgx.Matrix().create(self.resources)

        @property
        def _matrixClass(self):
            return _ScipyMeshMatrix

        def _solve_(self, L, x, b):
            self.x_gpu.upload(x)
            self.b_gpu.upload(b)
            self.A_gpu.upload_CSR(L)
            solver = pyamgx.Solver().create(self.resources, self.cfg)
            solver.setup(self.A_gpu)
            solver.solve(self.b_gpu, self.x_gpu)
            self.x_gpu.download(x)
            solver.destroy()
            return x

        def __del__(self):
            for attr in ["A_gpu", "b_gpu", "x_gpu", "cfg", "resources"]:
                if hasattr(self, attr):
                    try:
                        getattr(self, attr).destroy()
                    except:
                        pass

    return GPUAMGXSolver


def gaussian_random_field(shape, psd_power, anisotropy, seed):
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    kz = np.fft.fftfreq(nz)[:, None, None]
    ky = np.fft.fftfreq(ny)[None, :, None]
    kx = np.fft.fftfreq(nx)[None, None, :]

    # Apply anisotropy (z-stretch)
    k2 = (
        (kz * anisotropy[0]) ** 2
        + (ky * anisotropy[1]) ** 2
        + (kx * anisotropy[2]) ** 2
    )
    k2[0, 0, 0] = 1e-12

    amplitude = 1.0 / (k2 ** (psd_power / 2.0))
    phase = rng.random(shape) * 2.0 * np.pi
    F = (np.cos(phase) + 1j * np.sin(phase)) * amplitude
    field = ifftn(F).real
    return (field - field.mean()) / (field.std() + 1e-12)


def run_simulation(shape, global_cfg, row_params, seed, solver_cls):
    """
    Core Physics Logic.
    Receives specific parameters for this sample from row_params.
    """
    from fipy import (
        CellVariable,
        FaceVariable,
        Grid3D,
        TransientTerm,
        DiffusionTerm,
        ConvectionTerm,
        ImplicitSourceTerm,
    )

    nz, ny, nx = shape
    rng = np.random.default_rng(seed)

    # 1. Unpack Parameters
    porosity = row_params["target_porosity"]
    psd = row_params["psd_power"]
    ani_z = row_params["anisotropy_z"]
    ani_y = row_params["anisotropy_y"]
    ani_x = row_params["anisotropy_x"]
    drying = row_params["drying_intensity"]
    v_scale = row_params["velocity_scale"]
    evap_strength = row_params["evap_strength"]
    redep_factor = row_params["redeposition_factor"]
    z_mask_thresh = row_params["top_mask_zmin"]
    diff_pore = row_params["diff_pore"]
    diff_solid = row_params["diff_solid"]
    dt = row_params["dt"]
    steps = int(row_params["time_steps"])

    # 2. Initial Structure
    field = gaussian_random_field(shape, psd, (ani_z, ani_y, ani_x), seed)
    solid = field > np.quantile(field, porosity)
    solid_flat = solid.transpose(2, 1, 0).ravel()

    if row_params["method"] == "GRF":
        neighbors = np.zeros_like(solid, dtype=int)
        for axis in range(3):
            neighbors += np.roll(solid, 1, axis) + np.roll(solid, -1, axis)
        solid = solid & (neighbors >= 2)
        return (~solid).astype(np.uint8)

    # 3. FiPy Setup
    mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0)
    solver = solver_cls(tolerance=1e-6, iterations=100)

    # 4. Physics Variables
    # Binder initially distributed in solid phase
    binder = CellVariable(mesh=mesh, value=0.1)
    binder.setValue(binder.value * (1.0 + 0.5 * solid_flat))

    D = CellVariable(mesh=mesh, value=diff_pore)
    D.setValue(diff_solid, where=solid_flat)

    # Convection (Upwards)
    velocity = FaceVariable(mesh=mesh, rank=1, value=(0.0, 0.0, v_scale * drying))

    # Evaporation (Exponential decay from top)
    z = np.array(mesh.cellCenters[2])
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
    evap = CellVariable(mesh=mesh, value=np.exp(-evap_strength * (1.0 - z_norm)))

    # PDE
    eq = TransientTerm() == DiffusionTerm(D) - ConvectionTerm(
        velocity
    ) - ImplicitSourceTerm(0.2 * drying * evap)
    top_mask = z_norm > z_mask_thresh

    for _ in range(steps):
        prev_mass = float(np.sum(binder.value))
        eq.solve(var=binder, dt=dt, solver=solver)

        # Redeposition Logic (Mass Conservation)
        curr_mass = float(np.sum(binder.value))
        lost = max(0.0, prev_mass - curr_mass)
        if lost > 0:
            vals = binder.value.copy()
            vals[top_mask] += redep_factor * drying * lost / (np.sum(top_mask) + 1e-9)
            binder.setValue(vals)

    # 6. Finalize Binary Structure
    binder_vol = np.array(binder.value).reshape((nx, ny, nz)).transpose(2, 1, 0)
    binder_norm = (binder_vol - binder_vol.min()) / (
        binder_vol.max() - binder_vol.min() + 1e-9
    )

    # Stochastic thresholding based on binder concentration
    dens_mask = rng.random(shape) < (drying * binder_norm)
    final_solid = solid | dens_mask

    # Prune isolated pixels (simple neighbor check)
    neighbors = np.zeros_like(final_solid, dtype=int)
    for axis in range(3):
        neighbors += np.roll(final_solid, 1, axis) + np.roll(final_solid, -1, axis)
    final_solid = final_solid & (neighbors >= 2)

    return (~final_solid).astype(np.uint8)


# ==========================================
# 3. WORKER PROCESS
# ==========================================


def worker_main(gpu_id, sample_rows, config, out_dir, base_seed):
    """
    Processes a list of sample rows (dicts) on a specific GPU.
    """
    # 1. GPU ASSIGNMENT
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        import pyamgx

        pyamgx.initialize()
    except Exception as e:
        return [{"id": r["id"], "status": "fatal", "msg": str(e)} for r in sample_rows]

    GPUAMGXSolver = get_gpu_solver_class()
    results = []

    logger.info(f"[GPU {gpu_id}] Started. Batch size: {len(sample_rows)}")

    for row in sample_rows:
        run_id = int(row["id"])
        t0 = time.perf_counter()

        try:
            seed = base_seed + run_id
            shape = tuple(config["system"]["volume_shape"])

            # Run
            vol = run_simulation(shape, config, row, seed, GPUAMGXSolver)

            # Save
            fname = f"sample_{run_id:04d}.tif"
            tifffile.imwrite(os.path.join(out_dir, fname), vol * 255)

            t_el = time.perf_counter() - t0

            # Record Result
            res = row.copy()  # Includes the physics params
            res.update(
                {
                    "status": "success",
                    "filename": fname,
                    "gpu_id": gpu_id,
                    "duration_s": round(t_el, 2),
                }
            )
            results.append(res)
            logger.info(
                f"[GPU {gpu_id}] Sample {run_id} Done. (P={row['target_porosity']:.2f}, D={row['drying_intensity']:.2f})"
            )

        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Sample {run_id} Failed: {e}")
            res = {"id": run_id, "status": "error", "error_msg": str(e)}
            results.append(res)

    try:
        pyamgx.finalize()
    except:
        pass

    return results


# ==========================================
# 4. ORCHESTRATION
# ==========================================


def get_gpu_ids():
    """Returns list of GPU IDs."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            encoding="utf-8",
        )
        return [int(x) for x in out.strip().split("\n") if x.strip()]
    except:
        logger.warning("Could not detect GPUs via nvidia-smi. Defaulting to GPU 0.")
        return [0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit("Config not found.")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = config["run"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 1. Setup Resources
    gpu_ids = get_gpu_ids()
    num_gpus = len(gpu_ids)
    num_samples = config["run"]["num_samples"]
    logger.info(f"Running Sweep on GPUs: {gpu_ids} | Total Samples: {num_samples}")

    # 2. Generate Parameter Sweep Manifest
    df_params = generate_parameter_manifest(num_samples, config["run"]["base_seed"])

    # Save the manifest immediately so we know what we intended to run
    df_params.to_csv(os.path.join(out_dir, "planned_parameters.csv"), index=False)

    # 3. Distribute to Workers
    # Convert dataframe to list of dicts for distribution
    all_rows = df_params.to_dict("records")

    worker_payloads = []
    # Round-robin distribution
    chunks = [[] for _ in range(num_gpus)]
    for i, row in enumerate(all_rows):
        chunks[i % num_gpus].append(row)

    for i, gpu_id in enumerate(gpu_ids):
        if chunks[i]:
            worker_payloads.append(
                (gpu_id, chunks[i], config, out_dir, config["run"]["base_seed"])
            )

    # 4. Execute (Spawn Context)
    ctx = mp.get_context("spawn")
    t_start = time.time()

    with ctx.Pool(processes=len(worker_payloads)) as pool:
        all_results_lists = pool.starmap(worker_main, worker_payloads)

    # 5. Collate Results
    flat_results = [item for sublist in all_results_lists for item in sublist]
    df_results = pd.DataFrame(flat_results)

    # Reorder columns to put status first
    cols = ["id", "status", "filename", "duration_s", "gpu_id"] + [
        c
        for c in df_results.columns
        if c not in ["id", "status", "filename", "duration_s", "gpu_id", "error_msg"]
    ]
    if "error_msg" in df_results.columns:
        cols.append("error_msg")

    final_csv = os.path.join(out_dir, "final_results.csv")
    df_results[cols].to_csv(final_csv, index=False)

    logger.info(f"Sweep Complete. {time.time()-t_start:.2f}s total.")
    logger.info(f"Results saved to {final_csv}")


if __name__ == "__main__":
    main()
