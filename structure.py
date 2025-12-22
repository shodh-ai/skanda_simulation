
# generate_microstructures.py
# ------------------------------------------------------------
# Generates 10 microstructures (20% GRF, 80% FiPy/physics) with extra knobs.
# Saves 3D TIFF stacks and an index CSV/JSON. Physics falls back to GRF if FiPy
# is unavailable locally. Ensures top_mask and z_norm are NumPy arrays to avoid
# MeshVariable.sum(out=...) errors.
# ------------------------------------------------------------

import os
import json
import csv
import numpy as np
import tifffile

# Try importing FiPy; if unavailable, the script will fallback to GRF for physics mode
try:
    from fipy import (
        CellVariable,
        FaceVariable,
        Grid3D,
        TransientTerm,
        DiffusionTerm,
        ConvectionTerm,
        ImplicitSourceTerm,
    )
    FIPY_AVAILABLE = True
except Exception:
    FIPY_AVAILABLE = False


# ---------------------------
# Utility & generation helpers
# ---------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_porosity(binary_vol: np.ndarray) -> float:
    """
    Porosity as fraction of pore voxels. Here, binary_vol == 1 is pore, 0 is solid.
    """
    return float(np.mean(binary_vol))


def prune_isolated_voxels(solid_grid: np.ndarray, min_neighbors: int = 2) -> np.ndarray:
    """
    NumPy-only 6-neighbor pruning of isolated solids.
    Keeps voxels with >= min_neighbors solid neighbors.
    solid_grid: bool or uint8 array where True/1 means solid
    """
    solid_grid = solid_grid.astype(bool, copy=False)
    neigh = np.zeros_like(solid_grid, dtype=int)

    # Z neighbors
    neigh[1:, :, :] += solid_grid[:-1, :, :]
    neigh[:-1, :, :] += solid_grid[1:, :, :]
    # Y neighbors
    neigh[:, 1:, :] += solid_grid[:, :-1, :]
    neigh[:, :-1, :] += solid_grid[:, 1:, :]
    # X neighbors
    neigh[:, :, 1:] += solid_grid[:, :, :-1]
    neigh[:, :, :-1] += solid_grid[:, :, 1:]

    return np.logical_and(solid_grid, neigh >= min_neighbors)


def gaussian_random_field(
    shape: tuple,
    psd_power: float = 1.5,
    anisotropy: tuple = (1.0, 1.0, 1.0),
    seed: int = 42,
) -> np.ndarray:
    """
    Generates a 3D Gaussian Random Field using NumPy FFTs.
    psd_power controls spectral decay; anisotropy scales frequencies per axis.
    Returns: float array (nz, ny, nx)
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape

    kz = np.fft.fftfreq(nz)[:, None, None]
    ky = np.fft.fftfreq(ny)[None, :, None]
    kx = np.fft.fftfreq(nx)[None, None, :]

    k2 = (kz * anisotropy[0]) ** 2 + (ky * anisotropy[1]) ** 2 + (kx * anisotropy[2]) ** 2
    k2[0, 0, 0] = 1e-12  # avoid singularity

    amplitude = 1.0 / (k2 ** (psd_power / 2.0))
    phase = rng.random(shape) * 2.0 * np.pi
    F = (np.cos(phase) + 1j * np.sin(phase)) * amplitude

    field = np.fft.ifftn(F).real
    # Normalize to zero mean and unit std
    field = (field - field.mean()) / (field.std() + 1e-12)
    return field


def threshold_to_porosity(field: np.ndarray, target_porosity: float) -> np.ndarray:
    """
    Threshold the field to obtain desired porosity (fraction of pore voxels = 1's).
    Here we define pore as "field <= threshold".
    Returns: uint8 array (0=solid, 1=pore)
    """
    thr = np.quantile(field, target_porosity)
    binary_vol = (field <= thr).astype(np.uint8)
    return binary_vol


def generate_grf_structure(
    shape: tuple,
    target_porosity: float,
    psd_power: float,
    anisotropy: tuple,
    seed: int,
) -> np.ndarray:
    """
    GRF-based microstructure generation.
    Returns binary_vol with 1=pore, 0=solid.
    """
    field = gaussian_random_field(shape=shape, psd_power=psd_power, anisotropy=anisotropy, seed=seed)
    binary_vol = threshold_to_porosity(field, target_porosity)
    # Optional: prune isolated solids (visual cleanup)
    solid = binary_vol == 0
    solid = prune_isolated_voxels(solid, min_neighbors=2)
    binary_vol = (~solid).astype(np.uint8)
    return binary_vol


def generate_physics_structure(shape: tuple, params: dict, seed: int) -> np.ndarray:
    """
    Physics-based microstructure using FiPy convection-diffusion-evaporation.
    Extra knobs:
      - evap_strength (float, default 5.0)
      - redeposition_factor (float, default 0.08)
      - top_mask_zmin (float in [0,1], default 0.80)

    Returns binary_vol with 1=pore, 0=solid.
    If FiPy isn't available, falls back to GRF with same target porosity.
    """
    if not FIPY_AVAILABLE:
        return generate_grf_structure(
            shape=shape,
            target_porosity=params.get("target_porosity", 0.4),
            psd_power=1.5,
            anisotropy=(1.0, 1.0, 1.0),
            seed=seed,
        )

    nz, ny, nx = shape
    rng = np.random.default_rng(seed)

    # 1) Initial packing via GRF (light anisotropy for base solids)
    initial_field = gaussian_random_field(shape, psd_power=1.5, anisotropy=(0.9, 1.0, 1.1), seed=seed)
    thr = np.quantile(initial_field, params["target_porosity"])
    solid = initial_field > thr  # True = solid (obstacle)

    # 2) FiPy Grid (FiPy order is nx, ny, nz; NumPy is nz, ny, nx)
    mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0)

    # Map numpy solid mask to FiPy ordering (z,y,x) -> (x,y,z)
    solid_mesh = solid.transpose(2, 1, 0)  # (nz, ny, nx) -> (nx, ny, nz)
    solid_flat = solid_mesh.ravel()

    # 3) Physics parameters
    drying_intensity = float(params.get("drying_intensity", 0.5))
    nsteps = int(params.get("time_steps", 40))
    dt = float(params.get("dt", 0.05))
    v_scale = float(params.get("velocity_scale", 0.6))
    diff_pore = float(params.get("diff_pore", 1.0))
    diff_solid = float(params.get("diff_solid", 0.1))
    evap_strength = float(params.get("evap_strength", 5.0))         # NEW
    redeposition_factor = float(params.get("redeposition_factor", 0.08))  # NEW
    top_mask_zmin = float(params.get("top_mask_zmin", 0.80))        # NEW

    # 4) Variables
    binder = CellVariable(name="binder", mesh=mesh, value=0.1)
    binder.setValue(binder.value * (1.0 + 0.5 * solid_flat))  # more binder near solids

    D = CellVariable(mesh=mesh, value=diff_pore)
    D.setValue(diff_solid, where=solid_flat)

    # Upward convection along z
    velocity = FaceVariable(mesh=mesh, rank=1, value=(0.0, 0.0, v_scale * drying_intensity))

    # Evaporation profile stronger near top
    # IMPORTANT: use NumPy arrays for z_centers / z_norm / top_mask to avoid FiPy Variable reductions
    z_centers = mesh.cellCenters[2].value  # NumPy array of cell z-coordinates
    z_min, z_max = z_centers.min(), z_centers.max()
    z_norm = (z_centers - z_min) / (z_max - z_min + 1e-12)  # NumPy array
    evap_profile = np.exp(-evap_strength * (1.0 - z_norm))  # NumPy array
    k_evap = 0.2 * drying_intensity
    evap = CellVariable(mesh=mesh, value=evap_profile)       # assign NumPy array

    eq = TransientTerm() == DiffusionTerm(coeff=D) - ConvectionTerm(coeff=velocity) - ImplicitSourceTerm(k_evap * evap)

    # Top accumulation/crust formation (NumPy boolean array)
    top_mask = z_norm > top_mask_zmin  # NumPy boolean array

    for step in range(nsteps):
        # Conservation check to re-deposit evaporated material
        prev_total = float(binder.sum())   # FiPy API, returns scalar
        eq.solve(var=binder, dt=dt)
        after_total = float(binder.sum())  # FiPy API, returns scalar

        lost = max(0.0, prev_total - after_total)
        if lost > 0:
            deposit = redeposition_factor * drying_intensity * lost
            current_vals = binder.value.copy()         # NumPy array of cell values
            top_count = int(np.count_nonzero(top_mask))  # safe NumPy reduction
            if top_count > 0:
                current_vals[top_mask] += deposit / (top_count + 1e-12)
                binder.setValue(current_vals)

    # 6) Convert binder field back to (nz, ny, nx)
    binder_field = np.array(binder.value).reshape((nx, ny, nz)).transpose(2, 1, 0)

    # Normalize binder field
    b_min, b_max = binder_field.min(), binder_field.max()
    binder_norm = (binder_field - b_min) / (b_max - b_min + 1e-12)

    # Probabilistic densification based on binder-rich regions
    dens_prob = drying_intensity * binder_norm
    rng = np.random.default_rng(seed)
    dens_mask = rng.random((nz, ny, nx)) < dens_prob

    final_solid = np.logical_or(solid, dens_mask)
    final_solid = prune_isolated_voxels(final_solid, min_neighbors=2)

    # Return pore=1, solid=0
    binary_vol = (~final_solid).astype(np.uint8)
    return binary_vol


def save_tiff(path: str, binary_vol: np.ndarray):
    """
    Save 3D TIFF with 0=solid, 255=pore for visualization convenience.
    """
    tifffile.imwrite(path, (binary_vol * 255).astype(np.uint8))


# ---------------------------
# Simple Latin Hypercube Sampling (NumPy only)
# ---------------------------

def lhs(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """
    Latin Hypercube in [0,1] -> (n_samples x n_dims).
    """
    H = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        cut = np.linspace(0, 1, n_samples + 1)
        u = rng.uniform(low=cut[:-1], high=cut[1:], size=n_samples)
        P = rng.permutation(n_samples)
        H[:, j] = u[P]
    return H


# ---------------------------
# Main orchestration
# ---------------------------

def main():
    # Output folder
    out_dir = "./out_micro"
    ensure_dir(out_dir)

    # Choose total samples (demo: 10). Later, set to 10000 for GPU.
    N_TOTAL = 10

    # Mode mix: 20% GRF, 80% physics (rounded)
    N_GRF = max(1, int(round(N_TOTAL * 0.20)))
    N_PHY = N_TOTAL - N_GRF

    # Base shape (small for local run). Later: (128, 128, 128).
    base_shape = (8, 8, 8)

    rng = np.random.default_rng(12345)

    # Parameter ranges
    # Common
    POR_MIN, POR_MAX = 0.20, 0.80
    VOXEL_MIN, VOXEL_MAX = 0.08, 0.15  # micron per voxel (metadata only)

    # GRF ranges
    GRF_RANGES = {
        "psd_power": (0.8, 2.2),
        "anis_x": (0.7, 1.3),
        "anis_y": (0.7, 1.3),
        "anis_z": (0.7, 1.3),
    }

    # Physics ranges (for FiPy)
    PHY_RANGES = {
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

    # LHS for GRF and Physics (+2 dims each for porosity and voxel_size_um)
    U_grf = lhs(N_GRF, len(GRF_RANGES) + 2, rng) if N_GRF > 0 else np.zeros((0, len(GRF_RANGES) + 2))
    U_phy = lhs(N_PHY, len(PHY_RANGES) + 2, rng) if N_PHY > 0 else np.zeros((0, len(PHY_RANGES) + 2))

    records = []
    run_id = 0

    # ---- GRF samples ----
    for i in range(N_GRF):
        # Map LHS to physical ranges
        psd_power = GRF_RANGES["psd_power"][0] + U_grf[i, 0] * (GRF_RANGES["psd_power"][1] - GRF_RANGES["psd_power"][0])
        anis_x = GRF_RANGES["anis_x"][0] + U_grf[i, 1] * (GRF_RANGES["anis_x"][1] - GRF_RANGES["anis_x"][0])
        anis_y = GRF_RANGES["anis_y"][0] + U_grf[i, 2] * (GRF_RANGES["anis_y"][1] - GRF_RANGES["anis_y"][0])
        anis_z = GRF_RANGES["anis_z"][0] + U_grf[i, 3] * (GRF_RANGES["anis_z"][1] - GRF_RANGES["anis_z"][0])

        # Normalize anisotropy to keep mean ~1.0
        m = (anis_x + anis_y + anis_z) / 3.0
        anis = (anis_x / m, anis_y / m, anis_z / m)

        target_por = POR_MIN + U_grf[i, 4] * (POR_MAX - POR_MIN)
        voxel_size_um = VOXEL_MIN + U_grf[i, 5] * (VOXEL_MAX - VOXEL_MIN)

        seed = 12345 + run_id

        # Generate
        binary_vol = generate_grf_structure(
            shape=base_shape,
            target_porosity=target_por,
            psd_power=psd_power,
            anisotropy=anis,
            seed=seed,
        )
        por = compute_porosity(binary_vol)

        # Save
        out_path = os.path.join(out_dir, f"grf_{run_id:04d}.tif")
        save_tiff(out_path, binary_vol)

        # Record
        rec = {
            "run_id": run_id,
            "method": "GRF",
            "file": out_path,
            "seed": seed,
            "target_porosity": round(float(target_por), 4),
            "achieved_porosity": round(por, 4),
            "shape": list(base_shape),
            "voxel_size_um": round(float(voxel_size_um), 4),
            "psd_power": round(float(psd_power), 4),
            "anisotropy": [round(float(anis[0]), 4), round(float(anis[1]), 4), round(float(anis[2]), 4)],
        }
        records.append(rec)
        print(f"[GRF {run_id}] target={rec['target_porosity']} achieved={rec['achieved_porosity']} -> {out_path}")
        run_id += 1

    # ---- Physics samples ----
    for i in range(N_PHY):
        drying_intensity = PHY_RANGES["drying_intensity"][0] + U_phy[i, 0] * (PHY_RANGES["drying_intensity"][1] - PHY_RANGES["drying_intensity"][0])
        velocity_scale = PHY_RANGES["velocity_scale"][0] + U_phy[i, 1] * (PHY_RANGES["velocity_scale"][1] - PHY_RANGES["velocity_scale"][0])
        time_steps = int(round(PHY_RANGES["time_steps"][0] + U_phy[i, 2] * (PHY_RANGES["time_steps"][1] - PHY_RANGES["time_steps"][0])))
        dt = PHY_RANGES["dt"][0] + U_phy[i, 3] * (PHY_RANGES["dt"][1] - PHY_RANGES["dt"][0])
        diff_pore = PHY_RANGES["diff_pore"][0] + U_phy[i, 4] * (PHY_RANGES["diff_pore"][1] - PHY_RANGES["diff_pore"][0])
        diff_solid = PHY_RANGES["diff_solid"][0] + U_phy[i, 5] * (PHY_RANGES["diff_solid"][1] - PHY_RANGES["diff_solid"][0])
        evap_strength = PHY_RANGES["evap_strength"][0] + U_phy[i, 6] * (PHY_RANGES["evap_strength"][1] - PHY_RANGES["evap_strength"][0])
        redeposition_factor = PHY_RANGES["redeposition_factor"][0] + U_phy[i, 7] * (PHY_RANGES["redeposition_factor"][1] - PHY_RANGES["redeposition_factor"][0])
        top_mask_zmin = PHY_RANGES["top_mask_zmin"][0] + U_phy[i, 8] * (PHY_RANGES["top_mask_zmin"][1] - PHY_RANGES["top_mask_zmin"][0])

        target_por = POR_MIN + U_phy[i, 9] * (POR_MAX - POR_MIN)
        voxel_size_um = VOXEL_MIN + U_phy[i, 10] * (VOXEL_MAX - VOXEL_MIN)

        seed = 12345 + run_id

        params = {
            "target_porosity": float(target_por),
            "drying_intensity": float(drying_intensity),
            "time_steps": int(time_steps),
            "dt": float(dt),
            "velocity_scale": float(velocity_scale),
            "diff_pore": float(diff_pore),
            "diff_solid": float(diff_solid),
            "evap_strength": float(evap_strength),
            "redeposition_factor": float(redeposition_factor),
            "top_mask_zmin": float(top_mask_zmin),
        }

        binary_vol = generate_physics_structure(base_shape, params, seed)
        por = compute_porosity(binary_vol)

        out_path = os.path.join(out_dir, f"physics_{run_id:04d}.tif")
        save_tiff(out_path, binary_vol)

        rec = {
            "run_id": run_id,
            "method": "physics" if FIPY_AVAILABLE else "GRF_fallback",
            "file": out_path,
            "seed": seed,
            "target_porosity": round(float(target_por), 4),
            "achieved_porosity": round(por, 4),
            "shape": list(base_shape),
            "voxel_size_um": round(float(voxel_size_um), 4),

            # Store parameters used for traceability
            "drying_intensity": round(float(drying_intensity), 4),
            "time_steps": int(time_steps),
            "dt": round(float(dt), 4),
            "velocity_scale": round(float(velocity_scale), 4),
            "diff_pore": round(float(diff_pore), 4),
            "diff_solid": round(float(diff_solid), 4),
            "evap_strength": round(float(evap_strength), 4),
            "redeposition_factor": round(float(redeposition_factor), 4),
            "top_mask_zmin": round(float(top_mask_zmin), 4),
        }
        records.append(rec)
        print(f"[PHY {run_id}] target={rec['target_porosity']} achieved={rec['achieved_porosity']} -> {out_path}")
        run_id += 1

    # Save index.json and index.csv
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(records, f, indent=2)

    cols = sorted({k for r in records for k in r.keys()})
    with open(os.path.join(out_dir, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow(r)

    print(f"\nDone. Saved {len(records)} samples in {out_dir}/, with index.json and index.csv.")


if __name__ == "__main__":
    main()
