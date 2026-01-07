"""
Graphite anode microstructure generator - clean multi-phase output.
"""

import numpy as np
from ..schema import (
    Geometry,
    GraphiteParams,
    GenerationParams,
    DefectParams,
)

from .particle_packing import generate_particle_packing
from .texture import add_primary_particle_texture
from .calendering import apply_calendering
from .conductive_network import generate_conductive_locations
from .binder import generate_binder_locations
from .sei import generate_sei_locations


# Phase labels - distinct grayscale values
PHASE_PORE = 0  # Black - empty space
PHASE_CONDUCTIVE = 100  # Very dark gray - carbon black (hard to see)
PHASE_GRAPHITE = 145  # Medium gray - MAIN PHASE (bulk material)
PHASE_SEI = 175  # Medium-light gray - surface layer
PHASE_BINDER = 200  # Light gray - PVDF (brightest, most visible)


def _generate_graphite(
    run_id: int,
    seed: int,
    geometry: Geometry,
    params: GraphiteParams,
    generation: GenerationParams,
    defects: DefectParams,
) -> np.ndarray:
    """Generate multi-phase graphite microstructure."""

    print(f"[Graphite] Type: {params.graphite_type.value}")
    print(f"[Graphite] Target porosity: {params.target_porosity:.3f}")

    target_shape = geometry.shape
    voxel_size_um = geometry.voxel_size_nm / 1000.0

    print(f"[Graphite] Final output shape (guaranteed): {target_shape}")

    phase_volumes = _estimate_phase_volumes(params, generation, voxel_size_um * 1000.0)

    print(f"\n[Graphite] Phase Volume Estimates:")
    print(f"   Conductive: {phase_volumes['conductive']*100:.3f}%")
    print(f"   Binder: {phase_volumes['binder']*100:.3f}%")
    print(f"   SEI: {phase_volumes['sei']*100:.3f}%")
    print(f"   Total secondary phases: {phase_volumes['total']*100:.3f}%")

    adjusted_target_porosity = params.target_porosity + phase_volumes["total"]

    print(f"[Graphite] Adjusted initial porosity: {adjusted_target_porosity:.3f}")
    print(
        f"   (accounts for phases consuming {phase_volumes['total']*100:.2f}% of volume)"
    )

    compression_ratio = generation.calendering.compression_ratio
    if compression_ratio < 0.99:
        # Calculate Poisson expansion (optional - can set to 1.0 for z-only)
        poisson_ratio = 0.25
        axial_strain = compression_ratio - 1.0
        lateral_strain = -poisson_ratio * axial_strain
        lateral_expansion = 1.0 + lateral_strain

        # Pre-compression shape (thicker in z, narrower in x,y)
        nz, ny, nx = target_shape
        initial_nz = int(np.ceil(nz / compression_ratio))
        initial_ny = int(np.ceil(ny / lateral_expansion))
        initial_nx = int(np.ceil(nx / lateral_expansion))
        working_shape = (initial_nz, initial_ny, initial_nx)

        print(f"[Graphite] Pre-calendering shape: {working_shape}")
        print(f"[Graphite] Target shape: {target_shape}")
    else:
        working_shape = target_shape
        print(f"[Graphite] No calendering - generating at target shape")

    # Initialize all as pore
    phases = np.zeros(working_shape, dtype=np.uint8)

    # Step 1: Generate graphite particles
    print("[Graphite] Step 1/6: Generating particles...")
    graphite_mask = generate_particle_packing(
        shape=working_shape,
        target_porosity=adjusted_target_porosity,
        aspect_ratio=params.aspect_ratio,
        orientation_degree=params.orientation_degree,
        seed=seed,
    )
    phases[graphite_mask] = PHASE_GRAPHITE

    p1 = np.sum(phases == PHASE_PORE) / phases.size
    print(f"  Porosity: {p1:.3f}")

    # Step 2: Texture (skip if too fine)
    print("[Graphite] Step 2/6: Adding texture...")
    primary_size = params.primary_particle_size_um / voxel_size_um
    if primary_size >= 3:
        graphite_mask = add_primary_particle_texture(
            graphite_mask, primary_size, seed + 1
        )
        phases[:] = PHASE_PORE
        phases[graphite_mask] = PHASE_GRAPHITE
        print(f"  Applied texture")
    else:
        print(f"  Skipping texture (too fine)")

    p2 = np.sum(phases == PHASE_PORE) / phases.size
    print(f"  Porosity: {p2:.3f}")

    # Step 3: Calendering
    print("[Graphite] Step 3/6: Applying calendering...")
    if compression_ratio < 0.99:
        graphite_mask = apply_calendering(
            graphite_mask,
            target_shape,
            seed + 2,
            generation.calendering.compression_ratio,
            generation.calendering.particle_deformation,
        )
        phases = np.zeros(target_shape, dtype=np.uint8)
        phases[graphite_mask] = PHASE_GRAPHITE
    assert (
        phases.shape == target_shape
    ), f"Shape mismatch: {phases.shape} != {target_shape}"

    p3 = np.sum(phases == PHASE_PORE) / phases.size
    print(f"  Porosity: {p3:.3f}")

    # Step 4: Conductive network (minimal)
    print("[Graphite] Step 4/6: Adding conductive network...")
    cond_mask = generate_conductive_locations(
        graphite_mask,
        phases == PHASE_PORE,
        params.conductive_additive_type.value,
        params.conductive_additive_wt_frac,
        params.conductive_additive_particle_size_nm,
        params.conductive_additive_distribution.value,
        voxel_size_um * 1000.0,
        seed + 3,
    )
    phases[np.logical_and(cond_mask, phases == PHASE_PORE)] = PHASE_CONDUCTIVE

    n_cond = np.sum(phases == PHASE_CONDUCTIVE)
    p4 = np.sum(phases == PHASE_PORE) / phases.size
    print(f"  Added {n_cond} conductive voxels")
    print(f"  Porosity: {p4:.3f}")

    # Step 5: Binder (very minimal - only at contacts)
    print("[Graphite] Step 5/6: Adding binder...")
    binder_mask = generate_binder_locations(
        graphite_mask,
        phases == PHASE_PORE,
        params.binder_distribution.value,
        1,
        params.binder_wt_frac * 0.05,  # Only 5%
        seed + 4,
    )
    phases[np.logical_and(binder_mask, phases == PHASE_PORE)] = PHASE_BINDER

    n_bind = np.sum(phases == PHASE_BINDER)
    p5 = np.sum(phases == PHASE_PORE) / phases.size
    print(f"  Added {n_bind} binder voxels")
    print(f"  Porosity: {p5:.3f}")

    # Step 6: SEI layer (very sparse)
    if generation.sei_layer.enabled:
        print("[Graphite] Step 6/6: Adding SEI layer...")
        graphite_only_mask = phases == PHASE_GRAPHITE
        sei_mask = generate_sei_locations(
            graphite_only_mask,
            phases == PHASE_PORE,
            generation.sei_layer.thickness_nm,
            generation.sei_layer.uniformity,
            voxel_size_um * 1000.0,
            seed + 5,
        )
        phases[np.logical_and(sei_mask, phases == PHASE_PORE)] = PHASE_SEI
        n_sei = np.sum(phases == PHASE_SEI)
        print(f"  Added {n_sei} SEI voxels")
    else:
        print("[Graphite] Step 6/6: Skipping SEI layer")

    # Final statistics
    total = phases.size
    print(f"\n[Graphite] Phase Distribution:")
    print(
        f"  Pore (0):         {np.sum(phases==PHASE_PORE):7d} ({np.sum(phases==PHASE_PORE)/total:.3f})"
    )
    print(
        f"  Graphite (120):   {np.sum(phases==PHASE_GRAPHITE):7d} ({np.sum(phases==PHASE_GRAPHITE)/total:.3f})"
    )
    print(
        f"  Conductive (160): {np.sum(phases==PHASE_CONDUCTIVE):7d} ({np.sum(phases==PHASE_CONDUCTIVE)/total:.3f})"
    )
    print(
        f"  Binder (200):     {np.sum(phases==PHASE_BINDER):7d} ({np.sum(phases==PHASE_BINDER)/total:.3f})"
    )
    print(
        f"  SEI (240):        {np.sum(phases==PHASE_SEI):7d} ({np.sum(phases==PHASE_SEI)/total:.3f})"
    )

    final_porosity = np.sum(phases == PHASE_PORE) / total
    print(f"\n[Graphite] Final porosity: {final_porosity:.3f}")
    print(f"[Graphite] Target: {params.target_porosity:.3f}")
    print(f"[Graphite] Error: {abs(final_porosity - params.target_porosity):.3f}")

    return phases


def _estimate_phase_volumes(
    params: GraphiteParams,
    generation: GenerationParams,
    voxel_size_nm: float,
) -> dict:
    """
    Estimate volume fractions of secondary phases.

    These phases will consume pore space, so we need to account
    for them when setting initial porosity target.

    Returns:
        Dictionary with volume fractions for each phase
    """
    volumes = {}

    # 🟢 Conductive additive
    # Use the proper conversion from conductive_network.py
    rho_graphite = 2.26
    rho_carbon_black = 1.8
    conductive_vol = params.conductive_additive_wt_frac * (
        rho_graphite / rho_carbon_black
    )
    conductive_vol *= 0.25  # Aggregate porosity factor
    volumes["conductive"] = conductive_vol

    # 🟢 Binder
    # Binder is minimal and already heavily reduced
    rho_binder = 1.1  # PVDF/CMC density
    binder_vol = params.binder_wt_frac * (rho_graphite / rho_binder)
    binder_vol *= 0.05  # Current reduction factor from binder.py
    volumes["binder"] = binder_vol

    if generation.sei_layer.enabled:
        thickness_nm = generation.sei_layer.thickness_nm
        thickness_voxels = max(1, thickness_nm / voxel_size_nm)

        # Surface area factor (fraction of volume that is near-surface)
        # For packed particles, this is roughly 0.2-0.3
        surface_area_factor = 0.25

        # SEI volume = surface_area_factor × thickness_voxels × voxel_volume
        # As fraction of total volume
        sei_vol = surface_area_factor * thickness_voxels / 10.0  # Empirical scaling
        sei_vol = min(sei_vol, 0.01)  # Cap at 1% of total volume

        volumes["sei"] = sei_vol
    else:
        volumes["sei"] = 0.0

    # Total volume consumed by secondary phases
    volumes["total"] = volumes["conductive"] + volumes["binder"] + volumes["sei"]

    return volumes


__all__ = ["_generate_graphite"]
