"""
Calendering compression simulation.
"""

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    generate_binary_structure,
    zoom,
)


def apply_calendering(
    solid: np.ndarray,
    target_shape: tuple,
    seed: int,
    compression_ratio: float,
    particle_deformation: float,
) -> np.ndarray:
    """
    Apply calendering compression in z-direction.

    Physics:
    - Material densifies (porosity decreases)
    - Particles deform and flatten
    - NO new material added

    Args:
        solid: Boolean solid array
        compression_ratio: Final_thickness / initial_thickness (0.6-0.8 typical)
        particle_deformation: Degree of particle flattening (0-1)

    Returns:
        Compressed solid array (same shape, higher density)
    """
    if compression_ratio >= 0.99:
        print("  No calendering compression applied")
        return _ensure_exact_shape(solid, target_shape)

    rng = np.random.default_rng(seed)
    current_shape = solid.shape
    target_nz, target_ny, target_nx = target_shape
    print(f"   Compressing: {current_shape} → {target_shape}")

    if particle_deformation > 0.01:
        deformed_solid = _deform_particles(solid, particle_deformation, rng)
    else:
        deformed_solid = solid

    zoom_factors = (
        target_nz / current_shape[0],
        target_ny / current_shape[1],
        target_nx / current_shape[2],
    )
    compressed = zoom(deformed_solid.astype(np.uint8), zoom_factors, order=0).astype(
        bool
    )

    compressed = _ensure_exact_shape(compressed, target_shape)

    assert (
        compressed.shape == target_shape
    ), f"Calendering failed: {compressed.shape} != {target_shape}"

    print(f"   ✅ Calendered to exact shape: {compressed.shape}")

    return compressed


def _deform_particles(
    solid: np.ndarray, deformation: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Deform particles to make them flatter in z-direction.

    Method: Erode in z, dilate in x,y to conserve volume
    """
    if deformation < 0.01:
        return solid

    # Calculate morphological operations based on deformation
    # Higher deformation = more flattening
    iterations = int(np.ceil(deformation * 3))  # 0-3 iterations

    # Create anisotropic structuring elements
    # Z-direction: smaller (compress)
    struct_z = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ],
        dtype=bool,
    )

    # XY-direction: larger (expand)
    struct_xy = generate_binary_structure(3, 1)
    struct_xy[0, :, :] = False  # No z-expansion
    struct_xy[2, :, :] = False

    deformed = solid.copy()

    for i in range(iterations):
        # Slight erosion in z-direction (flattening)
        eroded = binary_erosion(deformed, structure=struct_z)

        # Compensate with dilation in xy (spreading)
        dilated = binary_dilation(eroded, structure=struct_xy, iterations=2)

        # Blend with original to control deformation amount
        blend = deformation / iterations
        random_mask = rng.random(deformed.shape) > blend * 0.3
        deformed = np.logical_or(
            np.logical_and(deformed, random_mask),
            dilated,
        )

    print(f"   Deformed particles: {iterations} iterations")

    return deformed


def _ensure_exact_shape(array: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Ensure array is EXACTLY target_shape.
    Handles minor rounding from scipy.zoom (typically ±1 voxel).
    """
    result = array
    for axis in range(3):
        current = result.shape[axis]
        target = target_shape[axis]

        if current > target:
            # Crop excess
            slices = [slice(None)] * 3
            slices[axis] = slice(0, target)
            result = result[tuple(slices)]
        elif current < target:
            # Pad deficit (only 1-2 voxels from rounding)
            padding = [(0, 0)] * 3
            padding[axis] = (0, target - current)
            result = np.pad(result, padding, mode="edge")

    return result
