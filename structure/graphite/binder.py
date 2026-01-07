"""
Binder distribution generation with proper mode support.
"""

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
)


def generate_binder_locations(
    graphite_mask: np.ndarray,
    pore_mask: np.ndarray,
    distribution_mode: str,
    film_thickness_voxels: int,
    weight_fraction: float,
    seed: int,
) -> np.ndarray:
    """
    Generate binder distribution based on mode.

    Args:
        graphite_mask: Boolean mask of graphite particles
        pore_mask: Boolean mask of pore space
        distribution_mode: "necks", "uniform", or "patchy"
        film_thickness_voxels: Thickness of binder coating in voxels
        weight_fraction: Target weight fraction of binder
        seed: Random seed

    Returns:
        Boolean mask of binder locations
    """
    if weight_fraction < 0.001:
        return np.zeros_like(graphite_mask, dtype=bool)

    rng = np.random.default_rng(seed)

    # Find particle surfaces
    eroded = binary_erosion(graphite_mask, iterations=1)
    surface = np.logical_and(graphite_mask, ~eroded)

    # Generate candidate regions based on distribution mode
    if distribution_mode == "necks":
        candidate_region = _generate_neck_candidates(
            graphite_mask, pore_mask, film_thickness_voxels
        )
    elif distribution_mode == "patchy":
        candidate_region = _generate_patchy_candidates(
            surface, pore_mask, film_thickness_voxels, rng
        )
    else:  # uniform
        candidate_region = _generate_uniform_candidates(
            surface, pore_mask, film_thickness_voxels
        )

    # Calculate how many voxels to add based on weight fraction
    n_candidates = np.sum(candidate_region)
    if n_candidates == 0:
        return np.zeros_like(graphite_mask, dtype=bool)

    # Target amount: weight_fraction controls fraction of candidates to fill
    # Reduce by 95% to prevent porosity collapse (only 5% of available space)
    fill_fraction = weight_fraction * 0.05
    n_to_add = int(n_candidates * fill_fraction)
    n_to_add = min(n_to_add, n_candidates)

    if n_to_add == 0:
        return np.zeros_like(graphite_mask, dtype=bool)

    # Randomly select locations from candidates
    candidate_indices = np.argwhere(candidate_region)
    selected_indices = rng.choice(len(candidate_indices), n_to_add, replace=False)

    # Create binder mask
    binder_mask = np.zeros_like(graphite_mask, dtype=bool)
    for idx in candidate_indices[selected_indices]:
        binder_mask[tuple(idx)] = True

    # Apply film thickness by dilating selected points
    if film_thickness_voxels > 1:
        struct = generate_binary_structure(3, 1)
        binder_mask = binary_dilation(
            binder_mask, struct, iterations=film_thickness_voxels - 1
        )
        # Keep only in pore space
        binder_mask = np.logical_and(binder_mask, pore_mask)

    return binder_mask


def _generate_neck_candidates(
    graphite_mask: np.ndarray,
    pore_mask: np.ndarray,
    film_thickness: int,
) -> np.ndarray:
    """
    Generate candidates at particle necks/contacts.

    Finds narrow gaps between particles where binder bridges form.
    """
    # Distance from solid (identifies narrow gaps)
    dist = distance_transform_edt(~graphite_mask)

    # Narrow gaps: close to particles but not inside them
    # Threshold based on film thickness
    max_gap = max(3.0, film_thickness * 1.5)
    narrow_gaps = np.logical_and(dist > 0, dist <= max_gap)

    # Must be in pore space
    candidates = np.logical_and(narrow_gaps, pore_mask)

    return candidates


def _generate_uniform_candidates(
    surface: np.ndarray,
    pore_mask: np.ndarray,
    film_thickness: int,
) -> np.ndarray:
    """
    Generate uniform coating candidates around all particle surfaces.

    Creates a shell of specified thickness around particles.
    """
    # Dilate surface into pore space by film thickness
    struct = generate_binary_structure(3, 1)
    coating_region = binary_dilation(surface, struct, iterations=film_thickness)

    # Keep only in pore space
    candidates = np.logical_and(coating_region, pore_mask)

    return candidates


def _generate_patchy_candidates(
    surface: np.ndarray,
    pore_mask: np.ndarray,
    film_thickness: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate patchy/clustered candidates.

    Creates random patches of binder on particle surfaces.
    """
    # Start with uniform coating region
    struct = generate_binary_structure(3, 1)
    coating_region = binary_dilation(surface, struct, iterations=film_thickness)
    coating_region = np.logical_and(coating_region, pore_mask)

    # Create random patches using noise field
    noise = rng.random(surface.shape)
    noise = gaussian_filter(noise, sigma=3.0)  # Large sigma = large patches

    # Threshold noise to create patches (keep ~30% of coating region)
    threshold = np.quantile(noise[coating_region], 0.7)
    patch_mask = noise > threshold

    # Combine with coating region
    candidates = np.logical_and(coating_region, patch_mask)

    return candidates


__all__ = ["generate_binder_locations"]
