"""
Utility functions for graphite microstructure generation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_random_field(
    shape: tuple,
    alpha: float = 3.0,
    anisotropy: tuple = (1.0, 1.0, 1.0),
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a 3D Gaussian Random Field using FFT.

    Args:
        shape: (nz, ny, nx) shape of output
        alpha: Power spectrum exponent (higher = smoother)
        anisotropy: Scaling factors for (z, y, x) directions
        seed: Random seed

    Returns:
        Normalized field with zero mean and unit variance
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape

    # Frequency grids
    kz = np.fft.fftfreq(nz)[:, None, None]
    ky = np.fft.fftfreq(ny)[None, :, None]
    kx = np.fft.fftfreq(nx)[None, None, :]

    # Anisotropic frequency magnitude
    k2 = (
        (kz * anisotropy[0]) ** 2 +
        (ky * anisotropy[1]) ** 2 +
        (kx * anisotropy[2]) ** 2
    )
    k2[0, 0, 0] = 1e-10  # Avoid singularity

    # Power spectrum: E(k) ~ k^(-alpha)
    amplitude = k2 ** (-alpha / 4.0)

    # Random phase
    phase = rng.random(shape) * 2.0 * np.pi

    # Complex field in Fourier space
    F = amplitude * np.exp(1j * phase)

    # Inverse FFT to get real field
    field = np.fft.ifftn(F).real

    # Normalize to zero mean, unit variance
    field = (field - field.mean()) / (field.std() + 1e-10)

    return field


def threshold_to_porosity(
    field: np.ndarray,
    target_porosity: float,
) -> np.ndarray:
    """
    Threshold field to achieve target porosity.

    Args:
        field: Input scalar field
        target_porosity: Desired porosity (fraction of pores)

    Returns:
        Boolean array where True = pore, False = solid
    """
    threshold = np.quantile(field, target_porosity)
    pore_mask = field <= threshold
    return pore_mask


def fix_shape(
    array: np.ndarray,
    target_shape: tuple,
) -> np.ndarray:
    """
    Crop or pad array to match target shape.

    Args:
        array: Input array
        target_shape: Desired (nz, ny, nx) shape

    Returns:
        Array with target shape
    """
    current_shape = array.shape
    result = array.copy()

    for axis in range(3):
        current = current_shape[axis]
        target = target_shape[axis]

        if current > target:
            # Crop
            slices = [slice(None)] * 3
            slices[axis] = slice(0, target)
            result = result[tuple(slices)]
        elif current < target:
            # Pad
            padding = [(0, 0)] * 3
            padding[axis] = (0, target - current)
            result = np.pad(result, padding, mode='constant', constant_values=False)

    return result
