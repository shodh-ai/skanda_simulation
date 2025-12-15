import os
import numpy as np
import tifffile
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


def generate_structure(run_id, config, output_path):
    """
    Generates a structure, thresholds it, and saves as a 3D TIFF.
    Returns: The file path and the calculated porosity.
    """
    params = config["structure"]
    shape = tuple(map(int, params["volume_shape"]))
    target_porosity = params["target_porosity"]

    field = gaussian_random_field(
        shape=shape,
        psd_power=params["psd_power"],
        anisotropy=tuple(params["anisotropy"]),
        seed=config["general"]["base_seed"] + run_id,
    )

    thr = np.quantile(field, target_porosity)
    binary_vol = (field <= thr).astype(np.uint8)

    filename = f"sample_{run_id:04d}.tif"
    file_path = os.path.join(output_path, filename)

    if config["general"]["save_images"]:
        tifffile.imwrite(file_path, binary_vol * 255)

    return file_path, binary_vol
