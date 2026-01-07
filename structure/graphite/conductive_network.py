"""
Conductive additive network generation - FIXED.

Implements physically accurate carbon black/CNT/graphene networks with:
- Proper density-based volume fraction conversion
- Sparse percolating networks at particle contacts
- Distribution mode support (aggregate/dispersed/network)
- Particle size scaling
"""

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
)


def generate_conductive_locations(
    graphite_mask: np.ndarray,
    pore_mask: np.ndarray,
    additive_type: str,  # ConductiveAdditiveType enum value
    weight_fraction: float,
    particle_size_nm: float,
    distribution_mode: str,  # DistributionMode enum value
    voxel_size_nm: float,
    seed: int,
) -> np.ndarray:
    """
    Generate conductive additive network with physical accuracy.

    Physics:
    - Converts wt% to vol% using material densities
    - Accounts for aggregate porosity (sparse networks)
    - Preferentially locates at particle necks for percolation
    - Scales morphology by particle size

    Args:
        graphite_mask: Boolean mask of graphite particles
        pore_mask: Boolean mask of pore space
        additive_type: Type of conductive additive
        weight_fraction: Target weight fraction (0.02-0.03 typical)
        particle_size_nm: Particle/feature size of additive
        distribution_mode: Distribution pattern
        voxel_size_nm: Voxel size for scaling
        seed: Random seed

    Returns:
        Boolean mask of conductive additive locations
    """
    if weight_fraction < 0.001:
        return np.zeros_like(graphite_mask, dtype=bool)

    rng = np.random.default_rng(seed)

    print(f"   Conductive additive: {additive_type}")
    print(f"   Target wt%: {weight_fraction*100:.2f}%")
    print(f"   Particle size: {particle_size_nm:.0f} nm")
    print(f"   Distribution: {distribution_mode}")

    # 🟢 Step 1: Convert weight to volume fraction
    effective_volume_fraction = _calculate_volume_fraction(
        additive_type, weight_fraction
    )

    print(f"   Effective vol%: {effective_volume_fraction*100:.3f}%")

    # 🟢 Step 2: Find candidate locations based on distribution mode
    candidates = _get_candidate_locations(
        graphite_mask,
        pore_mask,
        distribution_mode,
        particle_size_nm,
        voxel_size_nm,
    )

    n_candidates = np.sum(candidates)
    if n_candidates == 0:
        print("   ⚠️ No candidate locations found")
        return np.zeros_like(graphite_mask, dtype=bool)

    # 🟢 Step 3: Calculate target number of voxels
    n_pores = np.sum(pore_mask)
    n_target = int(n_pores * effective_volume_fraction)
    n_target = min(n_target, n_candidates)

    if n_target == 0:
        print("   ⚠️ Target voxels = 0")
        return np.zeros_like(graphite_mask, dtype=bool)

    print(f"   Candidate voxels: {n_candidates}")
    print(f"   Target voxels: {n_target}")

    # 🟢 Step 4: Select locations
    candidate_indices = np.argwhere(candidates)
    selected_indices = rng.choice(len(candidate_indices), n_target, replace=False)

    conductive_mask = np.zeros_like(graphite_mask, dtype=bool)
    for idx in candidate_indices[selected_indices]:
        conductive_mask[tuple(idx)] = True

    # 🟢 Step 5: Create morphology based on additive type
    conductive_mask = _apply_morphology(
        conductive_mask,
        pore_mask,
        additive_type,
        particle_size_nm,
        voxel_size_nm,
    )

    n_final = np.sum(conductive_mask)
    actual_fraction = n_final / n_pores if n_pores > 0 else 0

    print(f"   Final voxels: {n_final} ({actual_fraction*100:.3f}% of pores)")

    return conductive_mask


def _calculate_volume_fraction(additive_type: str, weight_fraction: float) -> float:
    """
    Convert weight fraction to effective volume fraction.

    Accounts for:
    1. Density differences
    2. Aggregate porosity (carbon black forms porous aggregates)
    """
    # Material densities (g/cm³)
    rho_graphite = 2.26

    densities = {
        "carbon_black": 1.8,  # Aggregate density
        "super_p": 1.9,  # Similar to carbon black
        "cnt": 1.3,  # CNT mat density (not individual tube)
        "graphene": 1.5,  # Graphene flake stack density
    }

    rho_additive = densities.get(additive_type, 1.8)

    # Volume fraction relative to solid
    volume_fraction = weight_fraction * (rho_graphite / rho_additive)

    # Aggregate porosity factor: Carbon black forms porous networks
    # Only ~20-30% of the "volume" is actual solid carbon
    porosity_factors = {
        "carbon_black": 0.25,  # Very porous aggregates
        "super_p": 0.30,  # Slightly denser
        "cnt": 0.20,  # Very sparse mats
        "graphene": 0.35,  # Flakes stack with gaps
    }

    porosity_factor = porosity_factors.get(additive_type, 0.25)

    effective_volume_fraction = volume_fraction * porosity_factor
    effective_volume_fraction *= 0.2

    return effective_volume_fraction


def _get_candidate_locations(
    graphite_mask: np.ndarray,
    pore_mask: np.ndarray,
    distribution_mode: str,
    particle_size_nm: float,
    voxel_size_nm: float,
) -> np.ndarray:
    """
    Find candidate locations based on distribution mode.

    Modes:
    - NETWORK: Percolating paths at particle contacts (realistic)
    - AGGREGATE: Clustered aggregates at contacts
    - DISPERSED: Random distribution in pores
    """
    if distribution_mode == "network":
        # Percolating network at particle necks
        return _find_network_locations(graphite_mask, pore_mask, voxel_size_nm)

    elif distribution_mode == "aggregate":
        # Clustered aggregates
        return _find_aggregate_locations(
            graphite_mask, pore_mask, particle_size_nm, voxel_size_nm
        )

    else:  # "dispersed"
        # Uniform dispersion near surfaces
        return _find_dispersed_locations(graphite_mask, pore_mask)


def _find_network_locations(
    graphite_mask: np.ndarray,
    pore_mask: np.ndarray,
    voxel_size_nm: float,
) -> np.ndarray:
    """
    Find locations for percolating networks.

    Preferentially at particle contacts/necks where electrical
    connection is critical.
    """
    # Distance from solid
    dist = distance_transform_edt(~graphite_mask)

    # Narrow gaps between particles (necks)
    max_gap_nm = 150.0  # Typical carbon black aggregate size
    max_gap_voxels = max_gap_nm / voxel_size_nm

    narrow_gaps = np.logical_and(dist > 0, dist <= max(3.0, max_gap_voxels))

    # Must be in pore space
    candidates = np.logical_and(narrow_gaps, pore_mask)

    # Find local minima in distance (centers of necks)
    dist_smooth = gaussian_filter(dist, sigma=1.5)

    # Create patches at neck centers
    for _ in range(3):
        dist_smooth = gaussian_filter(dist_smooth, sigma=1.0)

    # Threshold to find neck centers
    threshold = (
        np.quantile(dist_smooth[candidates], 0.4) if np.sum(candidates) > 0 else 0
    )
    neck_centers = np.logical_and(candidates, dist_smooth <= threshold)

    return neck_centers


def _find_aggregate_locations(
    graphite_mask: np.ndarray,
    pore_mask: np.ndarray,
    particle_size_nm: float,
    voxel_size_nm: float,
) -> np.ndarray:
    """
    Find locations for aggregate clusters.

    Creates discrete clusters near particle surfaces.
    """
    # Find particle surfaces
    eroded = binary_erosion(graphite_mask, iterations=1)
    surface = np.logical_and(graphite_mask, ~eroded)

    # Expand into pores
    dilated = binary_dilation(surface, iterations=2)
    near_surface = np.logical_and(dilated, pore_mask)

    # Create random cluster seeds
    rng = np.random.default_rng(42)
    noise = rng.random(graphite_mask.shape)

    # Cluster size based on particle size
    cluster_size_voxels = max(2.0, particle_size_nm / voxel_size_nm / 10)
    noise = gaussian_filter(noise, sigma=cluster_size_voxels)

    # Select cluster regions (sparse)
    threshold = (
        np.quantile(noise[near_surface], 0.85) if np.sum(near_surface) > 0 else 0.5
    )
    clusters = np.logical_and(near_surface, noise > threshold)

    return clusters


def _find_dispersed_locations(
    graphite_mask: np.ndarray,
    pore_mask: np.ndarray,
) -> np.ndarray:
    """
    Find locations for dispersed (uniform) distribution.

    All pore regions near particle surfaces are candidates.
    """
    # Find particle surfaces
    eroded = binary_erosion(graphite_mask, iterations=1)
    surface = np.logical_and(graphite_mask, ~eroded)

    # Expand slightly into pores
    dilated = binary_dilation(surface, iterations=1)
    candidates = np.logical_and(dilated, pore_mask)

    return candidates


def _apply_morphology(
    seed_mask: np.ndarray,
    pore_mask: np.ndarray,
    additive_type: str,
    particle_size_nm: float,
    voxel_size_nm: float,
) -> np.ndarray:
    """
    Apply characteristic morphology based on additive type.

    - Carbon black: Small spherical aggregates
    - CNT: Thin filaments
    - Graphene: Sheet-like structures
    """
    feature_size_voxels = max(1, int(particle_size_nm / voxel_size_nm))

    if additive_type in ["carbon_black", "super_p"]:
        # Compact aggregates - minimal dilation
        result = binary_dilation(seed_mask, iterations=1)

    elif additive_type == "cnt":
        # Thin filamentary structures
        # Use anisotropic dilation to create fiber-like features
        struct = np.zeros((3, 3, 3), dtype=bool)
        struct[1, :, :] = True  # Extend in y-x plane
        result = binary_dilation(seed_mask, structure=struct, iterations=1)

    elif additive_type == "graphene":
        # Sheet-like structures
        struct = np.zeros((3, 3, 3), dtype=bool)
        struct[:, 1, :] = True  # Sheet in z-x plane
        result = binary_dilation(seed_mask, structure=struct, iterations=1)

    else:
        # Default: minimal dilation
        result = binary_dilation(seed_mask, iterations=1)

    # Keep only in pore space
    result = np.logical_and(result, pore_mask)

    return result


__all__ = ["generate_conductive_locations"]
