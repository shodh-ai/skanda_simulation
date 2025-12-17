import os
import numpy as np
import tifffile
from scipy import ndimage
from scipy.fft import fftn, ifftn


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


class SimplifiedPhysicsGenerator:
    """
    Explicit advection-diffusion-evaporation transport:
    - Upwind advection in z (upward)
    - Face-averaged diffusion with spatially varying D(solid vs pore)
    - Neumann (zero-gradient) BCs via edge padding
    - Mass accounting + controlled top deposition (crust)
    """

    def __init__(self, shape):
        # shape is (nz, ny, nx)
        self.nz, self.ny, self.nx = shape

    def generate(
        self,
        seed=42,
        target_porosity=0.40,
        drying_intensity=0.3,
        nsteps=60,
        dt=0.01,
        D_pore=1.0,
        D_solid=0.5,
        v_scale=1.0,
        k_evap_scale=0.2,
        deposit_fraction=0.10,
        top_layers=3,
        psd_power_init=1.5,
    ):

        rng = np.random.default_rng(seed)

        # --- Step 1: Initial packing via GRF ---
        # Use slightly anisotropic GRF for base particles
        initial_field = gaussian_random_field(
            (self.nz, self.ny, self.nx),
            psd_power=psd_power_init,
            anisotropy=(0.8, 1.0, 1.0),
            seed=seed,
        )
        threshold = np.quantile(initial_field, target_porosity)
        solid = initial_field > threshold  # Boolean: True = solid

        # --- Step 2: Explicit transport (NumPy) ---

        # Binder concentration (retain more in solid initially)
        C = np.ones((self.nz, self.ny, self.nx), dtype=np.float64) * 0.1
        C[solid] *= 1.5

        # Spatially varying diffusivity
        D = np.where(solid, D_solid, D_pore).astype(np.float64)

        # Upward advection velocity (z-direction)
        v = float(v_scale * drying_intensity)

        # Evaporation sink profile (stronger near top)
        z_coords = np.linspace(0.0, 1.0, self.nz)
        evap_profile = np.exp(-5.0 * (1.0 - z_coords))
        k_ev = (k_evap_scale * drying_intensity) * evap_profile.reshape(self.nz, 1, 1)

        # Neumann BC padding helpers
        def pad_z(arr):
            return np.pad(arr, ((1, 1), (0, 0), (0, 0)), mode="edge")

        def pad_y(arr):
            return np.pad(arr, ((0, 0), (1, 1), (0, 0)), mode="edge")

        def pad_x(arr):
            return np.pad(arr, ((0, 0), (0, 0), (1, 1)), mode="edge")

        for step in range(nsteps):
            C_prev = C.copy()

            # --- Diffusion: Face-averaged D and zero-flux BCs ---

            # Z-direction
            C_pad = pad_z(C_prev)
            D_pad = pad_z(D)
            C_zm = C_pad[:-2, :, :]
            C_zp = C_pad[2:, :, :]
            D_zm = D_pad[:-2, :, :]
            D_zp = D_pad[2:, :, :]
            Dm_z = 0.5 * (D + D_zm)
            Dp_z = 0.5 * (D + D_zp)
            diff_z = Dp_z * (C_zp - C_prev) - Dm_z * (C_prev - C_zm)

            # Y-direction
            C_pad = pad_y(C_prev)
            D_pad = pad_y(D)
            C_ym = C_pad[:, :-2, :]
            C_yp = C_pad[:, 2:, :]
            D_ym = D_pad[:, :-2, :]
            D_yp = D_pad[:, 2:, :]
            Dm_y = 0.5 * (D + D_ym)
            Dp_y = 0.5 * (D + D_yp)
            diff_y = Dp_y * (C_yp - C_prev) - Dm_y * (C_prev - C_ym)

            # X-direction
            C_pad = pad_x(C_prev)
            D_pad = pad_x(D)
            C_xm = C_pad[:, :, :-2]
            C_xp = C_pad[:, :, 2:]
            D_xm = D_pad[:, :, :-2]
            D_xp = D_pad[:, :, 2:]
            Dm_x = 0.5 * (D + D_xm)
            Dp_x = 0.5 * (D + D_xp)
            diff_x = Dp_x * (C_xp - C_prev) - Dm_x * (C_prev - C_xm)

            diffusion = diff_x + diff_y + diff_z  # Assuming dx=dy=dz=1

            # --- Advection: Upwind & Neumann BCs ---
            C_pad = pad_z(C_prev)
            C_zm = C_pad[:-2, :, :]
            C_zp = C_pad[2:, :, :]
            grad_upwind = (C_prev - C_zm) if v >= 0.0 else (C_zp - C_prev)
            advection = -v * grad_upwind

            # --- Evaporation sink ---
            sink_term = k_ev * C_prev
            sink_loss = dt * sink_term.sum()

            # --- Update concentration ---
            C = C_prev + dt * (diffusion + advection - sink_term)
            C = np.maximum(C, 0.0)

            # --- Deposit fraction of evaporated mass (Crust formation) ---
            layers = max(1, min(top_layers, self.nz))
            top_slice = slice(self.nz - layers, self.nz)
            n_top_cells = layers * self.ny * self.nx
            deposit_mass = deposit_fraction * sink_loss

            if n_top_cells > 0 and deposit_fraction > 0.0:
                C[top_slice, :, :] += deposit_mass / n_top_cells

        binder = C

        # --- Step 3: Densification ~ binder ---
        bmin, bmax = binder.min(), binder.max()
        binder_normalized = (binder - bmin) / (bmax - bmin + 1e-12)
        densification_prob = drying_intensity * binder_normalized
        densification_mask = (
            rng.random((self.nz, self.ny, self.nx)) < densification_prob
        )

        final_solid = np.logical_or(solid, densification_mask)

        # --- Step 4: Top crust artifact ---
        layers = max(1, min(top_layers, self.nz))
        top_densification = (
            rng.random((layers, self.ny, self.nx)) < 0.3 * drying_intensity
        )
        final_solid[-layers:] = np.logical_or(final_solid[-layers:], top_densification)

        # --- Step 5: Smoothing (SciPy) ---
        # Removes small floating artifacts
        final_solid = ndimage.binary_opening(final_solid, structure=np.ones((2, 2, 2)))

        # Convert to pore phase (1=pore, 0=solid)
        pores = (~final_solid).astype(np.uint8)

        return pores


def generate_structure(run_id, config, output_path):
    """
    Generates a structure, thresholds it, and saves as a 3D TIFF.
    Returns: The file path and the calculated porosity.
    """
    params = config["structure"]
    shape = tuple(map(int, params["volume_shape"]))
    target_porosity = params["target_porosity"]

    run_seed = config["general"]["base_seed"] + run_id
    method = params.get("method", "GRF").lower()

    if method == "physics":
        gen = SimplifiedPhysicsGenerator(shape)
        binary_vol = gen.generate(
            seed=run_seed,
            target_porosity=target_porosity,
            drying_intensity=params.get("drying_intensity", 0.3),
            nsteps=params.get("time_steps", 60),
            dt=params.get("dt", 0.01),
            D_pore=params.get("diff_pore", 1.0),
            D_solid=params.get("diff_solid", 0.5),
            v_scale=params.get("velocity_scale", 1.0),
            k_evap_scale=params.get("k_evap_scale", 0.2),
            deposit_fraction=params.get("deposit_fraction", 0.10),
            top_layers=params.get("top_layers", 3),
            psd_power_init=params.get("psd_power", 1.5),
        )

    else:
        field = gaussian_random_field(
            shape=shape,
            psd_power=params.get("psd_power", 1.0),
            anisotropy=tuple(params.get("anisotropy", (1.0, 1.0, 1.0))),
            seed=run_seed,
        )
        thr = np.quantile(field, target_porosity)
        binary_vol = (field <= thr).astype(np.uint8)

    filename = f"sample_{run_id:04d}.tif"
    file_path = os.path.join(output_path, filename)

    if config["general"]["save_images"]:
        tifffile.imwrite(file_path, binary_vol * 255)

    return file_path, binary_vol
