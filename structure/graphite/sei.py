"""
SEI (Solid Electrolyte Interphase) layer generation - FIXED.

Physics:
- SEI forms on ALL exposed anode surfaces in contact with electrolyte
- Thickness controlled by formation conditions (10-50 nm typical)
- Coverage ~100% of exposed surfaces (with optional non-uniformity)
- Very thin layer extending into pore space from particle surfaces
"""

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
)


def generate_sei_locations(
    solid_mask: np.ndarray,
    pore_mask: np.ndarray,
    thickness_nm: float,
    uniformity: float,
    voxel_size_nm: float,
    seed: int,
) -> np.ndarray:
    """
    Generate SEI layer on exposed anode surfaces.

    Physics:
    - SEI forms on ALL surfaces exposed to electrolyte
    - Thickness: typically 10-50 nm
    - Coverage: ~100% (uniformity controls variations)
    - Passivation layer preventing further decomposition

    Args:
        solid_mask: Boolean mask of all solid phases (graphite, etc.)
        pore_mask: Boolean mask of pore space (electrolyte-filled)
        thickness_nm: SEI layer thickness in nanometers
        uniformity: 0.0 (patchy) to 1.0 (perfectly uniform)
        voxel_size_nm: Voxel size for thickness calculation
        seed: Random seed

    Returns:
        Boolean mask of SEI layer locations
    """
    if thickness_nm < 1.0:
        print("   SEI thickness too small, skipping")
        return np.zeros_like(solid_mask, dtype=bool)

    if voxel_size_nm > thickness_nm * 2:
        print(f"   ⚠️ Voxels too coarse for SEI ({voxel_size_nm:.1f} nm > 2× thickness)")
        print(f"   Using sub-sampling approach instead of full coating")

        # Sample only a fraction of surfaces
        fraction = thickness_nm / voxel_size_nm * 0.5  # More conservative
        return _generate_sei_subsampled(solid_mask, pore_mask, fraction, seed)

    rng = np.random.default_rng(seed)

    print(f"   SEI thickness: {thickness_nm:.1f} nm")
    print(f"   SEI uniformity: {uniformity:.2f}")

    # 🟢 Step 1: Find ALL exposed surfaces (100% coverage by default)
    exposed_surfaces = _find_exposed_surfaces(solid_mask, pore_mask)

    n_exposed = np.sum(exposed_surfaces)
    if n_exposed == 0:
        print("   No exposed surfaces found")
        return np.zeros_like(solid_mask, dtype=bool)

    print(f"   Exposed surface voxels: {n_exposed}")

    # 🟢 Step 2: Calculate SEI thickness in voxels
    thickness_voxels = max(1, int(np.round(thickness_nm / voxel_size_nm)))
    print(
        f"   SEI thickness: {thickness_voxels} voxels ({thickness_voxels * voxel_size_nm:.1f} nm)"
    )

    # 🟢 Step 3: Grow SEI layer into pore space
    sei_mask = _grow_sei_layer(
        exposed_surfaces,
        pore_mask,
        thickness_voxels,
        uniformity,
        rng,
    )

    n_sei = np.sum(sei_mask)
    sei_thickness_actual = n_sei / n_exposed if n_exposed > 0 else 0

    print(f"   SEI voxels: {n_sei}")
    print(f"   Average layers per surface: {sei_thickness_actual:.2f}")

    return sei_mask


def _find_exposed_surfaces(
    solid_mask: np.ndarray,
    pore_mask: np.ndarray,
) -> np.ndarray:
    """
    Find all solid surfaces exposed to pores (electrolyte).

    These are the surfaces where SEI forms.
    """
    # Find solid boundaries
    eroded = binary_erosion(solid_mask, iterations=1)
    surface = np.logical_and(solid_mask, ~eroded)

    # Only surfaces adjacent to pores (electrolyte-accessible)
    dilated = binary_dilation(surface, iterations=1)
    exposed = np.logical_and(dilated, pore_mask)

    return exposed


def _grow_sei_layer(
    exposed_surfaces: np.ndarray,
    pore_mask: np.ndarray,
    thickness_voxels: int,
    uniformity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Grow SEI layer from exposed surfaces into pore space.

    Uniformity controls thickness variation:
    - 1.0: Perfectly uniform thickness
    - 0.5: Moderate variation
    - 0.0: Highly non-uniform (patchy)
    """
    # Start with exposed surfaces
    sei_mask = exposed_surfaces.copy()

    if thickness_voxels <= 1:
        # Very thin SEI - just the surface layer
        return np.logical_and(sei_mask, pore_mask)

    # 🟢 Create thickness variation field based on uniformity
    if uniformity < 0.99:
        # Generate non-uniformity pattern
        variation = _generate_thickness_variation(
            exposed_surfaces.shape,
            uniformity,
            rng,
        )
    else:
        # Perfectly uniform
        variation = np.ones(exposed_surfaces.shape, dtype=float)

    # 🟢 Grow layer by layer with thickness variation
    current_layer = exposed_surfaces.copy()

    for layer in range(1, thickness_voxels):
        # Dilate current layer
        next_layer = binary_dilation(current_layer, iterations=1)

        # Keep only in pore space
        next_layer = np.logical_and(next_layer, pore_mask)

        # Apply thickness variation (some regions stop growing earlier)
        if uniformity < 0.99:
            # Normalized layer number (0 to 1)
            layer_fraction = layer / thickness_voxels

            # Regions with low variation value stop growing earlier
            # As layer increases, fewer regions continue growing
            threshold = 1.0 - layer_fraction * (1.0 - uniformity)
            keep_mask = variation >= threshold

            next_layer = np.logical_and(next_layer, keep_mask)

        # Add to SEI mask (excluding already covered)
        new_voxels = np.logical_and(next_layer, ~sei_mask)
        sei_mask = np.logical_or(sei_mask, new_voxels)

        # Update current layer for next iteration
        current_layer = new_voxels.copy()

        # Stop if no new voxels added
        if not np.any(new_voxels):
            break

    # Final cleanup: ensure only in pore space
    sei_mask = np.logical_and(sei_mask, pore_mask)

    return sei_mask


def _generate_thickness_variation(
    shape: tuple,
    uniformity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate spatially correlated thickness variation field.

    Higher uniformity = smoother, less variation
    Lower uniformity = patchy, more variation
    """
    # Generate random field
    noise = rng.random(shape)

    # Smooth based on uniformity
    # High uniformity → large sigma → very smooth
    # Low uniformity → small sigma → patchy
    sigma = 2.0 + uniformity * 8.0  # 2-10 voxels

    variation = gaussian_filter(noise, sigma=sigma)

    # Normalize to [0, 1]
    variation = (variation - variation.min()) / (
        variation.max() - variation.min() + 1e-10
    )

    # Adjust contrast based on uniformity
    # High uniformity → all values near 1.0
    # Low uniformity → wide range of values
    variation = variation ** (1.0 / (uniformity + 0.1))

    return variation


def _generate_sei_subsampled(
    solid_mask: np.ndarray,
    pore_mask: np.ndarray,
    coverage_fraction: float,
    seed: int,
) -> np.ndarray:
    """
    Generate SEI using sub-sampling when voxels are too coarse.

    Instead of coating every surface voxel (which would be too much),
    randomly sample surface voxels to achieve target volume fraction.
    """
    rng = np.random.default_rng(seed)

    # Find exposed surfaces
    eroded = binary_erosion(solid_mask, iterations=1)
    surface = np.logical_and(solid_mask, ~eroded)
    dilated = binary_dilation(surface, iterations=1)
    exposed = np.logical_and(dilated, pore_mask)

    n_exposed = np.sum(exposed)
    if n_exposed == 0:
        return np.zeros_like(solid_mask, dtype=bool)

    # Randomly sample to achieve target fraction
    n_target = int(n_exposed * coverage_fraction)
    n_target = max(1, min(n_target, n_exposed))

    exposed_indices = np.argwhere(exposed)
    selected = rng.choice(len(exposed_indices), n_target, replace=False)

    sei_mask = np.zeros_like(solid_mask, dtype=bool)
    for idx in exposed_indices[selected]:
        sei_mask[tuple(idx)] = True

    print(
        f"   SEI subsampled: {n_target}/{n_exposed} surface voxels ({coverage_fraction*100:.1f}%)"
    )

    return sei_mask


__all__ = ["generate_sei_locations"]
