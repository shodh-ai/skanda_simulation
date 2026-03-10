"""
Step 0 — Composition Calculator

Converts weight fractions → volume fractions → particle counts → moles → capacity.
This runs once before any geometry is touched.
All downstream generation steps read CompositionState; none re-derive from wt fracs.

Units throughout:
    volumes   → nm³
    lengths   → nm
    mass      → g
    moles     → mol
    capacity  → mAh
"""

from __future__ import annotations
import numpy as np
from structure.data import CompositionState
from structure.schema import ResolvedGeneration
from structure.constants import _NM3_TO_CM3
from structure.utils.composition import (
    _lognormal_d3_correction,
    _oblate_spheroid_volume,
    _sphere_volume,
    _assert_wf_sum,
    _validate,
)


def compute_composition(sim: ResolvedGeneration) -> CompositionState:
    """
    Compute the full CompositionState from a ResolvedGeneration.

    This is the only public symbol from this module.
    Call once at the start of the generation pipeline.

    Args:
        sim: Fully resolved simulation (from schema.resolve()).

    Returns:
        CompositionState with all derived quantities.

    Raises:
        ValueError: If weight fractions don't sum to 1.0 (within 1e-4).
    """
    c = sim.composition
    si = sim.silicon
    ca = sim.additive
    bi = sim.binder
    cb = sim.carbon

    # ── 1. Weight fractions ─────────────────────────────────────────────────
    wf_si = c.silicon_wt_frac
    wf_carbon = c.carbon_matrix_wt_frac
    wf_additive = c.conductive_additive_wt_frac
    wf_binder = c.binder_wt_frac

    _assert_wf_sum(wf_si, wf_carbon, wf_additive, wf_binder)

    # ── 2. Volume fractions of solid phase ──────────────────────────────────
    #
    # For a multi-phase solid, the volume fraction of phase i is:
    #   vf_i = (w_i / ρ_i) / Σ(w_j / ρ_j)
    #
    # This is exact for immiscible, non-porous solid phases.

    spec_vols = np.array(
        [
            wf_si / si.density_g_cm3,
            wf_carbon / cb.material.density_g_cm3,
            wf_additive / ca.density_g_cm3,
            wf_binder / bi.density_g_cm3,
        ]
    )
    total_spec_vol = spec_vols.sum()
    vf_si_val, vf_carbon_val, vf_additive_val, vf_binder_val = (
        spec_vols / total_spec_vol
    )

    # ── 3. Domain geometry ───────────────────────────────────────────────────
    L_nm = sim.voxel_size_nm * sim.voxel_resolution  # coating_thickness in nm
    V_domain = L_nm**3
    V_solid = V_domain * (1.0 - c.target_porosity)

    V_si_nm3 = vf_si_val * V_solid
    V_carbon_nm3 = vf_carbon_val * V_solid
    V_additive_nm3 = vf_additive_val * V_solid
    V_binder_nm3 = vf_binder_val * V_solid

    # ── 4. Carbon particle geometry ──────────────────────────────────────────
    #
    # Carbon flakes modelled as oblate spheroids:
    #   basal semi-axis  a = d50 / 2
    #   thickness c-axis c = a / aspect_ratio
    #   volume V = (4/3)π a² c
    #
    # PSD correction: the mean particle volume (which controls how many
    # particles fill a given V_carbon) is larger than the median particle
    # volume by (1 + cv²)^(9/2) for a log-normal distribution.

    carbon_d50 = cb.d50_nm
    carbon_a = carbon_d50 / 2.0
    carbon_ar = cb.material.aspect_ratio_mean
    carbon_c = carbon_a / carbon_ar
    carbon_size_cv = cb.size_cv

    V_carbon_med = _oblate_spheroid_volume(carbon_a, carbon_ar)
    V_carbon_mean = V_carbon_med * _lognormal_d3_correction(cb.size_cv)
    N_carbon = max(1, round(V_carbon_nm3 / V_carbon_mean))

    # ── 5. Si particle geometry ──────────────────────────────────────────────
    si_d50 = si.d50_nm
    si_r = si_d50 / 2.0

    V_si_med = _sphere_volume(si_r)
    V_si_mean = V_si_med * _lognormal_d3_correction(si.size_cv)
    N_si = max(1, round(V_si_nm3 / V_si_mean))

    # ── 6. Mass per phase ────────────────────────────────────────────────────
    mass_si_g = V_si_nm3 * _NM3_TO_CM3 * si.density_g_cm3
    mass_carbon_g = V_carbon_nm3 * _NM3_TO_CM3 * cb.material.density_g_cm3
    mass_additive_g = V_additive_nm3 * _NM3_TO_CM3 * ca.density_g_cm3
    mass_binder_g = V_binder_nm3 * _NM3_TO_CM3 * bi.density_g_cm3

    # ── 7. Moles ─────────────────────────────────────────────────────────────
    mol_si = mass_si_g / si.molar_mass_g_mol
    mol_carbon = mass_carbon_g / cb.material.molar_mass_g_mol
    mol_additive = mass_additive_g / ca.molar_mass_g_mol
    mol_binder = mass_binder_g / bi.repeat_unit_mass_g_mol

    # ── 8. Theoretical capacity ──────────────────────────────────────────────
    cap_si = mass_si_g * si.theoretical_capacity_mAh_g
    cap_carbon = mass_carbon_g * cb.material.theoretical_capacity_mAh_g
    cap_total = cap_si + cap_carbon
    cap_si_frac = cap_si / cap_total if cap_total > 0.0 else 0.0
    vol_cap = cap_total / (V_domain * _NM3_TO_CM3)

    # ── 9. Pre-calendering geometry ──────────────────────────────────────────
    #
    # Before calendering the electrode is thicker in Z.
    # All RSA packing targets this expanded domain.
    # After packing, Z is compressed by compression_ratio.
    #
    # phi_solid_pre = V_solid / V_pre
    #              = (V_domain × (1 - porosity)) / (L² × L/cr)
    #              = (1 - porosity) × cr

    cr = sim.calendering_compression_ratio
    L_z_pre = L_nm / cr
    phi_pre = V_solid / (L_nm**2 * L_z_pre)  # = (1 - porosity) × cr
    phi_carbon_pre = phi_pre * vf_carbon_val  # carbon's share of pre-calender solid

    # ── 10. Validation ───────────────────────────────────────────────────────
    warns = _validate(
        N_carbon=N_carbon,
        N_si=N_si,
        phi_pre=phi_pre,
        vf_si=vf_si_val,
        si_d50_nm=si_d50,
        voxel_size_nm=sim.voxel_size_nm,
        voxel_resolution=sim.voxel_resolution,
        porosity=c.target_porosity,
        compression_ratio=cr,
    )

    return CompositionState(
        wf_si=wf_si,
        wf_carbon=wf_carbon,
        wf_additive=wf_additive,
        wf_binder=wf_binder,
        vf_si=vf_si_val,
        vf_carbon=vf_carbon_val,
        vf_additive=vf_additive_val,
        vf_binder=vf_binder_val,
        domain_L_nm=L_nm,
        voxel_resolution=sim.voxel_resolution,
        V_domain_nm3=V_domain,
        V_solid_nm3=V_solid,
        porosity=c.target_porosity,
        V_si_nm3=V_si_nm3,
        V_carbon_nm3=V_carbon_nm3,
        V_additive_nm3=V_additive_nm3,
        V_binder_nm3=V_binder_nm3,
        carbon_d50_nm=carbon_d50,
        carbon_a_nm=carbon_a,
        carbon_c_nm=carbon_c,
        carbon_aspect_ratio=carbon_ar,
        carbon_V_median_nm3=V_carbon_med,
        carbon_V_mean_nm3=V_carbon_mean,
        carbon_size_cv=carbon_size_cv,
        si_d50_nm=si_d50,
        si_r_nm=si_r,
        si_V_median_nm3=V_si_med,
        si_V_mean_nm3=V_si_mean,
        N_carbon=N_carbon,
        N_si=N_si,
        mass_si_g=mass_si_g,
        mass_carbon_g=mass_carbon_g,
        mass_additive_g=mass_additive_g,
        mass_binder_g=mass_binder_g,
        mol_si=mol_si,
        mol_carbon=mol_carbon,
        mol_additive=mol_additive,
        mol_binder=mol_binder,
        capacity_si_mah=cap_si,
        capacity_carbon_mah=cap_carbon,
        capacity_total_mah=cap_total,
        capacity_si_fraction=cap_si_frac,
        volumetric_capacity_mah_cm3=vol_cap,
        compression_ratio=cr,
        L_z_pre_nm=L_z_pre,
        phi_solid_pre=phi_pre,
        phi_carbon_pre=phi_carbon_pre,
        validation_warnings=warns,
    )
