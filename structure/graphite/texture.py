"""
Primary particle texture generation.
"""

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, label


def add_primary_particle_texture(
    solid: np.ndarray,
    primary_size_voxels: float,
    seed: int,
) -> np.ndarray:
    """
    Add surface texture at primary particle scale.

    Safety: Preserves bulk material, only modifies surfaces
    """
    if primary_size_voxels < 2:
        print("  Primary particles too small to resolve, skipping texture")
        return solid

    rng = np.random.default_rng(seed)
    print(
        f"   Adding primary particle texture (scale: {primary_size_voxels:.1f} voxels)"
    )

    textured = _add_surface_roughness(solid, primary_size_voxels, rng)

    volume_before = np.sum(solid)
    volume_after = np.sum(textured)
    volume_change = (volume_before - volume_after) / volume_before

    if volume_change > 0.10:  # More than 10% loss is suspicious
        print(f"   ⚠️ WARNING: Excessive volume loss ({volume_change:.1%}), reverting")
        return solid

    # Check connectivity wasn't broken
    n_particles_before = label(solid)[1]
    n_particles_after = label(textured)[1]

    if (
        n_particles_after > n_particles_before * 1.2
    ):  # More than 20% increase = fragmentation
        print(f"   ⚠️ WARNING: Particle fragmentation detected, reverting")
        return solid

    print(f"   Applied texture, volume change: {volume_change:.1%}")

    return textured


def _add_surface_roughness(
    solid: np.ndarray, feature_size: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Add surface texture through alternating erosion/dilation.

    This creates roughness WITHOUT net material loss.
    """
    # Generate spatially correlated noise pattern
    noise = rng.random(solid.shape)
    noise = gaussian_filter(noise, sigma=feature_size / 2.0)

    # Find surface region (2 voxel depth)
    eroded_1 = binary_erosion(solid, iterations=1)
    eroded_2 = binary_erosion(solid, iterations=2)
    surface_shell = np.logical_and(solid, ~eroded_2)

    # Create random patches on surface
    threshold_erode = np.quantile(noise[surface_shell], 0.7)  # Erode top 30%
    threshold_dilate = np.quantile(noise[surface_shell], 0.3)  # Dilate bottom 30%

    erode_mask = np.logical_and(surface_shell, noise > threshold_erode)
    dilate_mask = np.logical_and(~solid, noise < threshold_dilate)

    # Apply texture
    textured = solid.copy()
    textured[erode_mask] = False
    textured[dilate_mask] = True

    return textured
