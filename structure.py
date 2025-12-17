import os
import numpy as np
import tifffile
from scipy.fft import fftn, ifftn

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
except ImportError:
    FIPY_AVAILABLE = False


def prune_isolated_voxels(solid_grid, min_neighbors=2):
    """
    Remove isolated solid voxels using 6-neighbor count (NumPy-only).
    Keeps voxels with >= min_neighbors solid neighbors.
    """
    neigh = np.zeros_like(solid_grid, dtype=int)

    # Calculate neighbors using shifting (vectorized)
    # Z neighbors
    neigh[1:, :, :] += solid_grid[:-1, :, :]
    neigh[:-1, :, :] += solid_grid[1:, :, :]
    # Y neighbors
    neigh[:, 1:, :] += solid_grid[:, :-1, :]
    neigh[:, :-1, :] += solid_grid[:, 1:, :]
    # X neighbors
    neigh[:, :, 1:] += solid_grid[:, :, :-1]
    neigh[:, :, :-1] += solid_grid[:, :, 1:]

    return solid_grid & (neigh >= min_neighbors)


def gaussian_random_field(shape, psd_power=1.0, anisotropy=(1.0, 1.0, 1.0), seed=42):
    """
    Generates a 3D Gaussian Random Field.
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape

    kz = np.fft.fftfreq(nz)[:, None, None]
    ky = np.fft.fftfreq(ny)[None, :, None]
    kx = np.fft.fftfreq(nx)[None, None, :]

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


def generate_physics_structure(shape, params, seed):
    """
    Generates microstructure using FiPy convection-diffusion-evaporation.
    Mimics binder migration and drying.
    """
    if not FIPY_AVAILABLE:
        raise ImportError("FiPy is required for physics-based generation.")

    nz, ny, nx = shape
    rng = np.random.default_rng(seed)

    # 1. Initial Packing via GRF
    # Use slightly anisotropic GRF for base particles
    initial_field = gaussian_random_field(
        shape, psd_power=1.5, anisotropy=(0.8, 1.0, 1.0), seed=seed
    )
    # Target slightly higher porosity initially, densification happens later
    thr = np.quantile(initial_field, params["target_porosity"])
    solid = initial_field > thr  # True = solid (obstacle)

    # 2. Setup FiPy Grid
    # FiPy uses (nx, ny, nz) ordering, numpy uses (nz, ny, nx)
    mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0)

    # Map numpy solid mask to FiPy ordering
    solid_mesh = solid.transpose(2, 1, 0)  # (z,y,x) -> (x,y,z)
    solid_flat = solid_mesh.ravel()

    # 3. Physics Parameters from Config
    drying_intensity = params.get("drying_intensity", 0.3)
    nsteps = params.get("time_steps", 40)
    dt = params.get("dt", 0.05)
    v_scale = params.get("velocity_scale", 0.6)

    # 4. Define Variables
    # Binder concentration
    binder = CellVariable(name="binder", mesh=mesh, value=0.1)
    # Higher initial binder on solid particles (adhesion)
    binder.setValue(binder.value * (1.0 + 0.5 * solid_flat))

    # Diffusion Coeff (D) - lower in solid
    D = CellVariable(mesh=mesh, value=params.get("diff_pore", 1.0))
    D.setValue(params.get("diff_solid", 0.1), where=solid_flat)

    # Convection (Velocity) - Upward (Z axis is last in FiPy 3D tuples?)
    # Grid3D faces are ordered. For simple vertical flow in Z:
    v_mag = v_scale * drying_intensity
    velocity = FaceVariable(mesh=mesh, rank=1, value=(0.0, 0.0, v_mag))

    # Evaporation Sink (stronger at top)
    z_centers = mesh.cellCenters[2]
    z_min, z_max = z_centers.min(), z_centers.max()
    z_norm = (z_centers - z_min) / (z_max - z_min + 1e-12)

    evap_profile = np.exp(-5.0 * (1.0 - z_norm))  # Exponential increase near top
    k_evap = 0.2 * drying_intensity
    evap = CellVariable(mesh=mesh, value=evap_profile)

    # 5. Solve PDE: dC/dt = Div(D grad C) - Div(v C) - Sink
    eq = TransientTerm() == DiffusionTerm(coeff=D) - ConvectionTerm(
        coeff=velocity
    ) - ImplicitSourceTerm(k_evap * evap)

    # Top accumulation helper
    top_mask = z_norm > 0.8  # Top 20%

    for step in range(nsteps):
        # Conservation check to redeposit evaporated material
        prev_total = float(binder.sum())
        eq.solve(var=binder, dt=dt)
        after_total = float(binder.sum())

        # Artificial redeposition (crust formation)
        lost = max(0.0, prev_total - after_total)
        if lost > 0:
            deposit = 0.08 * drying_intensity * lost
            # Add to top cells
            current_vals = binder.value.copy()
            # Simple uniform addition to top region for robustness
            current_vals[top_mask] += deposit / (np.sum(top_mask) + 1e-9)
            binder.setValue(current_vals)

    # 6. Convert Binder Field back to Microstructure
    binder_field = np.array(binder.value).reshape((nx, ny, nz)).transpose(2, 1, 0)

    # Normalize binder
    b_min, b_max = binder_field.min(), binder_field.max()
    binder_norm = (binder_field - b_min) / (b_max - b_min + 1e-12)

    # Probabilistic densification based on binder rich regions
    dens_prob = drying_intensity * binder_norm
    dens_mask = rng.random(shape) < dens_prob

    final_solid = np.logical_or(solid, dens_mask)

    # 7. Post-process (Pruning)
    final_solid = prune_isolated_voxels(final_solid, min_neighbors=2)

    # Invert for porosity (1 = pore, 0 = solid)
    # The pipeline expects binary_vol where 1 is pore.
    binary_vol = (~final_solid).astype(np.uint8)

    return binary_vol


def generate_structure(run_id, config, output_path):
    """
    Generates a structure, thresholds it, and saves as a 3D TIFF.
    Returns: The file path and the calculated porosity.
    """
    params = config["structure"]
    shape = tuple(map(int, params["volume_shape"]))
    target_porosity = params["target_porosity"]

    method = params.get("method", "GRF").lower()
    run_seed = config["general"]["base_seed"] + run_id

    if method == "physics":
        binary_vol = generate_physics_structure(shape, params, run_seed)
    else:
        field = gaussian_random_field(
            shape=shape,
            psd_power=params["psd_power"],
            anisotropy=tuple(params["anisotropy"]),
            seed=run_seed,
        )
        thr = np.quantile(field, params["target_porosity"])
        binary_vol = (field <= thr).astype(np.uint8)

    filename = f"sample_{run_id:04d}.tif"
    file_path = os.path.join(output_path, filename)

    if config["general"]["save_images"]:
        tifffile.imwrite(file_path, binary_vol * 255)

    return file_path, binary_vol
