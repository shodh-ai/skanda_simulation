"""
Maps a unit-hypercube vector (length N_GEN_DIMS) to a GenConfig-compatible
kwargs dict.  All cross-field Pydantic hard constraints are enforced here
before the model is instantiated.

Dimension index table  (order is fixed — never reorder):
  0  coatingthicknessum
  1  voxelresolution           categorical {64, 128}
  2  conductiveadditivewtfrac
  3  binderwtfrac              constrained by dim-2
  4  siwtfracinam
  5  targetporosity
  6  siparticled50nm           log-scale
  7  siparticlesizecv
  8  simorphology              categorical
  9  siinternalporosity        (only active when morphology==porous)
 10  sidistribution            categorical
 11  sivoidenabled             bool
 12  sivoidfraction            (only active when sivoidenabled)
 13  siembeddinguniformitycv   (only active when distribution==embedded)
 14  sicoreshellcarbonthicknessnm  (only active when distribution==coreshell)
 15  sicoatingenabled          bool
 16  sicoatingtype             categorical
 17  sicoatingthicknessnm      constrained by dim-6 and dim-16
 18  carbonparticled50nm       log-scale
 19  carbonparticlesizecv
 20  carbontype                categorical
 21  carbonorientationdegree
 22  conductiveadditivetype    categorical
 23  conductiveadditivedistribution  categorical
 24  bindertype                categorical
 25  binderdistribution        categorical
 26  calendering.compressionratio
 27  calendering.particledeformation
 28  calendering.orientationenhancement
 29  sei.enabled               bool
 30  sei.thicknessnm
 31  sei.uniformitycv
 32  sei.seicorrelationlengthnm  log-scale, constrained >= 1 voxel
 33  percolation.enforce       bool  (p_true=0.90)
 34  percolation.minthreshold
 35  contacts.coordinationnumber
 36  contacts.contactareafraction
 37  defects.particlecracks.enabled   bool (p_true=0.30)
 38  defects.particlecracks.crackprobability
 39  defects.particlecracks.crackwidthnm   log-scale
 40  defects.binderagglomeration.enabled   bool (p_true=0.30)
 41  defects.binderagglomeration.agglomerationprobability
 42  defects.delamination.enabled          bool (p_true=0.20)
 43  defects.delamination.delaminationfraction
 44  defects.poreclustering.enabled        bool (p_true=0.30)
 45  defects.poreclustering.clusteringdegree
"""

from __future__ import annotations
import numpy as np
from ._primitives import _cont, _cat, _bool, _int

N_GEN_DIMS: int = 46

_VOXEL_CHOICES = [128]
_MORPHOLOGY = ["spherical", "irregular", "porous"]
_DISTRIBUTION = ["embedded", "surface_anchored", "core_shell"]
_COATING_TYPE = ["carbon_coating", "siox_coating"]
_CARBON_TYPE = ["graphite_artificial", "graphite_natural", "graphite_mcmb"]
_CA_TYPE = ["cb_superp", "cb_c65", "cb_ketjenblack"]
_CA_DIST = ["aggregate", "dispersed", "network"]
_BINDER_TYPE = ["pvdf_kynar", "cmc_sbr"]
_BINDER_DIST = ["uniform", "patchy", "necks"]


def map_gen_config(u: np.ndarray, run_id: int, seed: int) -> dict:
    """
    Map a unit vector u of length N_GEN_DIMS to a dict accepted by
    GenConfig.model_validate().

    Args:
        u:       1-D array of length N_GEN_DIMS with values in [0, 1].
        run_id:  Unique integer ID for this sample.
        seed:    RNG seed stored inside GenConfig.

    Returns:
        dict ready for ``GenConfig.model_validate(d)``.
    """
    if len(u) != N_GEN_DIMS:
        raise ValueError(f"Expected u of length {N_GEN_DIMS}, got {len(u)}")

    i = iter(u)

    def take() -> float:
        return float(next(i))

    # ── Domain ────────────────────────────────────────────────────────────────
    coating_thickness_um = _cont(take(), 20.0, 150.0)
    voxel_resolution = _cat(take(), _VOXEL_CHOICES)

    # ── Composition ───────────────────────────────────────────────────────────
    conductive_additive_wt_frac = _cont(take(), 0.01, 0.10)
    binder_max = min(0.10, 0.19 - conductive_additive_wt_frac)  # hard: sum <= 0.20
    binder_max = max(0.02, binder_max)  # keep >= min
    binder_wt_frac = _cont(take(), 0.02, binder_max)
    si_wt_frac_in_am = _cont(take(), 0.03, 0.60)
    target_porosity = _cont(take(), 0.20, 0.50)

    # ── Silicon ───────────────────────────────────────────────────────────────
    si_particle_d50_nm = _cont(take(), 100.0, 1500.0, log=True)
    si_particle_size_cv = _cont(take(), 0.10, 0.60)
    si_morphology = _cat(take(), _MORPHOLOGY)
    si_internal_porosity = _cont(take(), 0.30, 0.70)
    si_distribution = _cat(take(), _DISTRIBUTION)
    si_void_enabled = _bool(take())
    si_void_fraction = _cont(take(), 0.20, 0.50)
    si_embedding_uniformity_cv = _cont(take(), 0.0, 0.60)
    si_core_shell_carbon_thickness_nm = _cont(take(), 0.0, 100.0)
    si_coating_enabled = _bool(take())
    si_coating_type = _cat(take(), _COATING_TYPE)

    # coating thickness: constrained by type AND particle size (<= 50% of radius)
    max_by_particle = 0.50 * (si_particle_d50_nm / 2.0)
    if si_coating_type == "siox_coating":
        coat_lo, coat_hi = 2.0, min(20.0, max_by_particle)
    else:  # carboncoating
        coat_lo, coat_hi = 5.0, min(50.0, max_by_particle)
    coat_hi = max(coat_hi, coat_lo)  # safety clamp
    # Ensure the ranges are strictly enforced
    coat_hi = (
        min(coat_hi, 20.0) if si_coating_type == "siox_coating" else min(coat_hi, 50.0)
    )
    si_coating_thickness_nm = _cont(take(), coat_lo, coat_hi)

    # ── Carbon matrix ─────────────────────────────────────────────────────────
    carbon_particle_d50_nm = _cont(take(), 8_000.0, 20_000.0, log=True)
    carbon_particle_size_cv = _cont(take(), 0.15, 0.45)
    carbon_type = _cat(take(), _CARBON_TYPE)
    carbon_orientation_degree = _cont(take(), 0.0, 0.90)

    # ── Conductive additive & binder ──────────────────────────────────────────
    conductive_additive_type = _cat(take(), _CA_TYPE)
    conductive_additive_distribution = _cat(take(), _CA_DIST)
    binder_type = _cat(take(), _BINDER_TYPE)
    binder_distribution = _cat(take(), _BINDER_DIST)

    # ── Calendering ───────────────────────────────────────────────────────────
    compression_ratio = _cont(take(), 0.50, 0.90)
    particle_deformation = _cont(take(), 0.00, 1.00)
    orientation_enhancement = _cont(take(), 0.00, 0.50)

    # ── SEI ───────────────────────────────────────────────────────────────────
    sei_enabled = _bool(take())
    sei_thickness = _cont(take(), 5.0, 80.0)
    sei_uniformity = _cont(take(), 0.00, 0.80)
    # correlation length must be >= 1 voxel (soft warning, not hard error, but
    # we respect it anyway to keep samples clean)
    voxelsizenm_est = coating_thickness_um * 1000.0 / voxel_resolution
    corr_lo = max(5.0, voxelsizenm_est)
    corr_hi = max(corr_lo * 1.5, min(10_000.0, coating_thickness_um * 500.0))
    sei_corr = _cont(take(), corr_lo, corr_hi, log=True)

    # ── Percolation ───────────────────────────────────────────────────────────
    perc_enforce = _bool(take(), 0.90)
    perc_threshold = _cont(take(), 0.80, 0.99)

    # ── Contacts ──────────────────────────────────────────────────────────────
    coord_number = _cont(take(), 4.0, 8.0)
    contact_area = _cont(take(), 0.05, 0.20)

    # ── Defects ───────────────────────────────────────────────────────────────
    cracks_enabled = _bool(take(), 0.30)
    cracks_prob = _cont(take(), 0.01, 0.20)
    cracks_width = _cont(take(), 10.0, 200.0, log=True)

    bagglom_enabled = _bool(take(), 0.30)
    bagglom_prob = _cont(take(), 0.05, 0.30)

    delam_enabled = _bool(take(), 0.20)
    delam_frac = _cont(take(), 0.01, 0.20)

    pore_enabled = _bool(take(), 0.30)
    pore_degree = _cont(take(), 0.00, 1.00)

    return dict(
        run_id=run_id,
        seed=seed,
        coating_thickness_um=coating_thickness_um,
        voxel_resolution=voxel_resolution,
        # composition
        conductive_additive_wt_frac=conductive_additive_wt_frac,
        binder_wt_frac=binder_wt_frac,
        si_wt_frac_in_am=si_wt_frac_in_am,
        target_porosity=target_porosity,
        # silicon
        si_particle_d50_nm=si_particle_d50_nm,
        si_particle_size_cv=si_particle_size_cv,
        si_morphology=si_morphology,
        si_internal_porosity=si_internal_porosity,
        si_distribution=si_distribution,
        si_void_enabled=si_void_enabled,
        si_void_fraction=si_void_fraction,
        si_embedding_uniformity_cv=si_embedding_uniformity_cv,
        si_core_shell_carbon_thickness_nm=si_core_shell_carbon_thickness_nm,
        si_coating_enabled=si_coating_enabled,
        si_coating_type=si_coating_type,
        si_coating_thickness_nm=si_coating_thickness_nm,
        # carbon
        carbon_particle_d50_nm=carbon_particle_d50_nm,
        carbon_particle_size_cv=carbon_particle_size_cv,
        carbon_type=carbon_type,
        carbon_orientation_degree=carbon_orientation_degree,
        # additive & binder
        conductive_additive_type=conductive_additive_type,
        conductive_additive_distribution=conductive_additive_distribution,
        binder_type=binder_type,
        binder_distribution=binder_distribution,
        # manufacturing
        calendering=dict(
            compression_ratio=compression_ratio,
            particle_deformation=particle_deformation,
            orientation_enhancement=orientation_enhancement,
        ),
        sei=dict(
            enabled=sei_enabled,
            sei_material="sei_generic",
            thickness_nm=sei_thickness,
            uniformity_cv=sei_uniformity,
            sei_correlation_length_nm=sei_corr,
        ),
        percolation=dict(
            enforce=perc_enforce,
            min_threshold=perc_threshold,
        ),
        contacts=dict(
            coordination_number=coord_number,
            contact_area_fraction=contact_area,
        ),
        defects=dict(
            particle_cracks=dict(
                enabled=cracks_enabled,
                crack_probability=cracks_prob,
                crack_width_nm=cracks_width,
            ),
            binder_agglomeration=dict(
                enabled=bagglom_enabled,
                agglomeration_probability=bagglom_prob,
            ),
            delamination=dict(
                enabled=delam_enabled,
                delamination_fraction=delam_frac,
            ),
            pore_clustering=dict(
                enabled=pore_enabled,
                clustering_degree=pore_degree,
            ),
        ),
    )
