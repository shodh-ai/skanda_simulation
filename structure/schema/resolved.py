"""
Combines a validated RunConfig + MaterialsDB into a single resolved object.
This is the ONLY object the generator ever reads — it never touches raw YAML.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .config import RunConfig
from .materials import (
    MaterialsDB,
    GraphiteMaterial,
    SiBaseMaterial,
    ConductiveAdditiveMaterial,
    BinderMaterial,
    CoatingMaterial,
    SEIMaterial,
)


@dataclass
class ResolvedSilicon:
    d50_nm: float
    size_cv: float
    morphology: str
    internal_porosity: float
    distribution: str
    void_enabled: bool
    void_fraction: float
    coating_enabled: bool
    coating_type: str
    coating_thickness_nm: float
    coating_material: Optional[CoatingMaterial]
    # Size-dependent (computed from si_base correlations)
    BET_m2_g: float
    li_diffusivity_m2_s: float
    electrical_conductivity_S_m: float
    # Intrinsic from DB
    density_g_cm3: float
    young_modulus_GPa: float
    poisson_ratio: float
    volume_expansion_factor: float
    theoretical_capacity_mAh_g: float
    molar_mass_g_mol: float


@dataclass
class ResolvedCarbon:
    material: GraphiteMaterial
    d50_nm: float
    size_cv: float


@dataclass
class ResolvedComposition:
    active_material_wt_frac: float
    silicon_wt_frac: float
    carbon_matrix_wt_frac: float
    conductive_additive_wt_frac: float
    binder_wt_frac: float
    target_porosity: float


@dataclass
class ResolvedSimulation:
    """Single resolved object handed to the generator."""

    run_id: int
    seed: int
    voxel_resolution: int
    voxel_size_nm: float

    composition: ResolvedComposition
    silicon: ResolvedSilicon
    carbon: ResolvedCarbon
    additive: ConductiveAdditiveMaterial
    additive_distribution: str
    binder: BinderMaterial
    binder_distribution: str

    calendering_compression_ratio: float
    calendering_particle_deformation: float
    calendering_orientation_enhancement: float

    sei_enabled: bool
    sei_material: SEIMaterial
    sei_thickness_nm: float
    sei_uniformity_cv: float

    percolation_enforce: bool
    percolation_min_threshold: float
    contacts_coordination_number: float
    contacts_contact_area_fraction: float

    defects_particle_cracks_enabled: bool
    defects_particle_cracks_probability: float
    defects_particle_cracks_width_nm: float
    defects_binder_agglomeration_enabled: bool
    defects_binder_agglomeration_probability: float
    defects_delamination_enabled: bool
    defects_delamination_fraction: float
    defects_pore_clustering_enabled: bool
    defects_pore_clustering_degree: float


def resolve(cfg: RunConfig, db: MaterialsDB) -> ResolvedSimulation:
    """
    Merge RunConfig + MaterialsDB into one flat ResolvedSimulation.
    This is the entry point used by the generator.
    """
    si_db: SiBaseMaterial = db.si_base

    coating_mat: Optional[CoatingMaterial] = (
        db.get_coating(cfg.si_coating_type.value) if cfg.si_coating_enabled else None
    )

    silicon = ResolvedSilicon(
        d50_nm=cfg.si_particle_d50_nm,
        size_cv=cfg.si_particle_size_cv,
        morphology=cfg.si_morphology.value,
        internal_porosity=cfg.si_internal_porosity,
        distribution=cfg.si_distribution.value,
        void_enabled=cfg.si_void_enabled,
        void_fraction=cfg.si_void_fraction,
        coating_enabled=cfg.si_coating_enabled,
        coating_type=cfg.si_coating_type.value,
        coating_thickness_nm=cfg.si_coating_thickness_nm,
        coating_material=coating_mat,
        # Computed
        BET_m2_g=si_db.compute_BET(cfg.si_particle_d50_nm),
        li_diffusivity_m2_s=si_db.compute_li_diffusivity(cfg.si_particle_d50_nm),
        electrical_conductivity_S_m=si_db.compute_electrical_conductivity(
            cfg.si_particle_d50_nm
        ),
        # Intrinsic
        density_g_cm3=si_db.density_g_cm3,
        young_modulus_GPa=si_db.young_modulus_GPa,
        poisson_ratio=si_db.poisson_ratio,
        volume_expansion_factor=si_db.volume_expansion_factor,
        theoretical_capacity_mAh_g=si_db.theoretical_capacity_mAh_g,
        molar_mass_g_mol=si_db.molar_mass_g_mol,
    )

    carbon = ResolvedCarbon(
        material=db.get_graphite(cfg.carbon_type.value),
        d50_nm=cfg.carbon_particle_d50_nm,
        size_cv=cfg.carbon_particle_size_cv,
    )

    composition = ResolvedComposition(
        active_material_wt_frac=cfg.active_material_wt_frac,
        silicon_wt_frac=cfg.silicon_wt_frac,
        carbon_matrix_wt_frac=cfg.carbon_matrix_wt_frac,
        conductive_additive_wt_frac=cfg.conductive_additive_wt_frac,
        binder_wt_frac=cfg.binder_wt_frac,
        target_porosity=cfg.target_porosity,
    )

    return ResolvedSimulation(
        run_id=cfg.run_id,
        seed=cfg.seed,
        voxel_resolution=cfg.voxel_resolution,
        voxel_size_nm=cfg.voxel_size_nm,
        composition=composition,
        silicon=silicon,
        carbon=carbon,
        additive=db.get_conductive_additive(cfg.conductive_additive_type.value),
        additive_distribution=cfg.conductive_additive_distribution.value,
        binder=db.get_binder(cfg.binder_type.value),
        binder_distribution=cfg.binder_distribution.value,
        calendering_compression_ratio=cfg.calendering.compression_ratio,
        calendering_particle_deformation=cfg.calendering.particle_deformation,
        calendering_orientation_enhancement=cfg.calendering.orientation_enhancement,
        sei_enabled=cfg.sei.enabled,
        sei_material=db.sei_generic,
        sei_thickness_nm=cfg.sei.thickness_nm,
        sei_uniformity_cv=cfg.sei.uniformity_cv,
        percolation_enforce=cfg.percolation.enforce,
        percolation_min_threshold=cfg.percolation.min_threshold,
        contacts_coordination_number=cfg.contacts.coordination_number,
        contacts_contact_area_fraction=cfg.contacts.contact_area_fraction,
        defects_particle_cracks_enabled=cfg.defects.particle_cracks.enabled,
        defects_particle_cracks_probability=cfg.defects.particle_cracks.crack_probability,
        defects_particle_cracks_width_nm=cfg.defects.particle_cracks.crack_width_nm,
        defects_binder_agglomeration_enabled=cfg.defects.binder_agglomeration.enabled,
        defects_binder_agglomeration_probability=cfg.defects.binder_agglomeration.agglomeration_probability,
        defects_delamination_enabled=cfg.defects.delamination.enabled,
        defects_delamination_fraction=cfg.defects.delamination.delamination_fraction,
        defects_pore_clustering_enabled=cfg.defects.pore_clustering.enabled,
        defects_pore_clustering_degree=cfg.defects.pore_clustering.clustering_degree,
    )
