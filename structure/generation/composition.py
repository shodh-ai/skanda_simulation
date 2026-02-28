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

import math
from dataclasses import dataclass, field

import numpy as np

from structure.schema.resolved import ResolvedSimulation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RSA jamming limit for oblate spheroids at aspect_ratio ≈ 5
# From simulation literature: ~35–42% depending on orientation freedom.
# Using 0.45 as a conservative upper bound.
_RSA_JAMMING_LIMIT: float = 0.45

# nm³ → cm³
_NM3_TO_CM3: float = 1e-21


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lognormal_d3_correction(size_cv: float) -> float:
    """
    Ratio of mean(d³) to median(d)³ for a log-normal PSD.

    For X ~ LogNormal with median = d50 and coefficient of variation cv = σ/μ:
        mean(X³) = d50³ × (1 + cv²)^(9/2)

    Used to correct particle count: we want N such that N × mean(V_particle) = V_total.
    Using median volume alone overestimates N by this factor.

    Example: cv=0.25 → correction=1.31, cv=0.30 → correction=1.47
    """
    return (1.0 + size_cv**2) ** (9.0 / 2.0)


def _oblate_spheroid_volume(a_nm: float, aspect_ratio: float) -> float:
    """V = (4/3)π a² c, where c = a / aspect_ratio (a = basal semi-axis)."""
    c_nm = a_nm / aspect_ratio
    return (4.0 / 3.0) * math.pi * (a_nm**2) * c_nm


def _sphere_volume(r_nm: float) -> float:
    return (4.0 / 3.0) * math.pi * (r_nm**3)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class CompositionState:
    """
    Fully resolved composition for one simulation run.
    Produced by compute_composition(); consumed by every generation step.
    Never modified after construction.
    """

    # ── Weight fractions (stored for reference / logging) ──────────────────
    wf_si: float
    wf_carbon: float
    wf_additive: float
    wf_binder: float

    # ── Solid volume fractions (sum = 1.0 exactly) ─────────────────────────
    vf_si: float
    vf_carbon: float
    vf_additive: float
    vf_binder: float

    # ── Domain geometry ────────────────────────────────────────────────────
    domain_L_nm: float  # cube edge in nm (= coating_thickness_um × 1000)
    voxel_resolution: int  # 64 | 128 | 256
    V_domain_nm3: float
    V_solid_nm3: float
    porosity: float

    # ── Per-phase volumes in domain (nm³) ──────────────────────────────────
    V_si_nm3: float
    V_carbon_nm3: float
    V_additive_nm3: float
    V_binder_nm3: float

    # ── Carbon particle geometry ────────────────────────────────────────────
    carbon_d50_nm: float
    carbon_a_nm: float  # basal semi-axis = d50/2
    carbon_c_nm: float  # thickness semi-axis = a / aspect_ratio
    carbon_aspect_ratio: float
    carbon_V_median_nm3: float  # volume of the median particle (d50)
    carbon_V_mean_nm3: float  # PSD-corrected mean volume

    # ── Si particle geometry ───────────────────────────────────────────────
    si_d50_nm: float
    si_r_nm: float  # = d50/2
    si_V_median_nm3: float
    si_V_mean_nm3: float  # PSD-corrected mean volume

    # ── Particle counts ────────────────────────────────────────────────────
    N_carbon: int  # target for explicit RSA packing
    N_si: int  # informational only — Si is placed statistically, not by RSA

    # ── Mass per phase in domain (g) ───────────────────────────────────────
    mass_si_g: float
    mass_carbon_g: float
    mass_additive_g: float
    mass_binder_g: float

    # ── Moles per phase in domain ──────────────────────────────────────────
    mol_si: float
    mol_carbon: float
    mol_additive: float
    mol_binder: float

    # ── Theoretical capacity ───────────────────────────────────────────────
    capacity_si_mah: float
    capacity_carbon_mah: float
    capacity_total_mah: float
    capacity_si_fraction: float  # Si share of total theoretical capacity
    volumetric_capacity_mah_cm3: float  # mAh / cm³ of electrode volume

    # ── Pre-calendering geometry ────────────────────────────────────────────
    compression_ratio: float
    L_z_pre_nm: float  # expanded Z domain before calendering
    phi_solid_pre: float  # solid vol fraction before calendering

    # ── Validation warnings (populated by _validate) ───────────────────────
    # Default must be last since it has a default value.
    validation_warnings: list[str] = field(default_factory=list)

    # ── Derived properties ─────────────────────────────────────────────────

    @property
    def voxel_size_nm(self) -> float:
        """Physical size of one output voxel in nm."""
        return self.domain_L_nm / self.voxel_resolution

    @property
    def V_domain_cm3(self) -> float:
        return self.V_domain_nm3 * _NM3_TO_CM3

    @property
    def si_in_voxels(self) -> float:
        """Si d50 expressed in voxel units — <1.0 means sub-voxel (statistical fill)."""
        return self.si_d50_nm / self.voxel_size_nm

    @property
    def carbon_in_voxels(self) -> float:
        """Carbon d50 expressed in voxel units."""
        return self.carbon_d50_nm / self.voxel_size_nm

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        w = self.validation_warnings
        lines = [
            "=" * 62,
            "  COMPOSITION STATE",
            "=" * 62,
            f"  Domain          : {self.domain_L_nm/1000:.1f} µm cube"
            f"  ({self.V_domain_cm3:.3e} cm³)",
            f"  Voxel size      : {self.voxel_size_nm:.1f} nm/voxel"
            f"  ({self.voxel_resolution}³)",
            "",
            "  Weight fractions (of total dry electrode mass):",
            f"    Si            : {self.wf_si:.4f}",
            f"    C-matrix      : {self.wf_carbon:.4f}",
            f"    Additive (CB) : {self.wf_additive:.4f}",
            f"    Binder        : {self.wf_binder:.4f}",
            f"    ─────────────   {self.wf_si+self.wf_carbon+self.wf_additive+self.wf_binder:.4f}",
            "",
            "  Solid volume fractions (of total solid):",
            f"    Si            : {self.vf_si:.4f}  ({self.vf_si*100:.2f}%)",
            f"    C-matrix      : {self.vf_carbon:.4f}  ({self.vf_carbon*100:.2f}%)",
            f"    Additive (CB) : {self.vf_additive:.4f}  ({self.vf_additive*100:.2f}%)",
            f"    Binder        : {self.vf_binder:.4f}  ({self.vf_binder*100:.2f}%)",
            f"    ─────────────   {self.vf_si+self.vf_carbon+self.vf_additive+self.vf_binder:.4f}",
            "",
            "  Particle counts:",
            f"    N_carbon      : {self.N_carbon}"
            f"  (explicit RSA,"
            f" d50={self.carbon_d50_nm/1000:.1f}µm,"
            f" AR={self.carbon_aspect_ratio:.1f},"
            f" {self.carbon_in_voxels:.0f} vx)",
            f"    N_Si          : {self.N_si:.2e}"
            f"  (statistical fill,"
            f" d50={self.si_d50_nm:.0f}nm,"
            f" {self.si_in_voxels:.2f} vx ← sub-voxel)",
            "",
            "  Moles in domain:",
            f"    Si            : {self.mol_si:.3e} mol",
            f"    C (graphite)  : {self.mol_carbon:.3e} mol",
            f"    CB (additive) : {self.mol_additive:.3e} mol",
            f"    Binder        : {self.mol_binder:.3e} mol",
            "",
            "  Theoretical capacity:",
            f"    Si            : {self.capacity_si_mah:.4e} mAh"
            f"  ({self.capacity_si_fraction*100:.1f}% of total)",
            f"    C-matrix      : {self.capacity_carbon_mah:.4e} mAh",
            f"    Total         : {self.capacity_total_mah:.4e} mAh",
            f"    Volumetric    : {self.volumetric_capacity_mah_cm3:.2f} mAh/cm³",
            "",
            "  Pre-calendering domain:",
            f"    compression   : {self.compression_ratio:.2f}",
            f"    L_z_pre       : {self.L_z_pre_nm/1000:.1f} µm"
            f"  (final: {self.domain_L_nm/1000:.1f} µm)",
            f"    φ_solid_pre   : {self.phi_solid_pre:.3f}"
            f"  ({self.phi_solid_pre*100:.1f}%)"
            f"  [RSA limit ≈ {_RSA_JAMMING_LIMIT*100:.0f}%]",
        ]
        if w:
            lines += ["", f"  ⚠  {len(w)} WARNING(s):"]
            lines += [f"     [{i+1}] {msg}" for i, msg in enumerate(w)]
        lines.append("=" * 62)
        return "\n".join(lines)

    def raise_if_critical(self) -> None:
        """
        Re-raise validation warnings as errors for any critical issues.
        Call this if you want strict mode (e.g., in batch generation).
        """
        critical = [w for w in self.validation_warnings if w.startswith("[CRITICAL]")]
        if critical:
            raise ValueError(
                f"CompositionState has {len(critical)} critical issue(s):\n"
                + "\n".join(critical)
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_composition(sim: ResolvedSimulation) -> CompositionState:
    """
    Compute the full CompositionState from a ResolvedSimulation.

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

    # ── 10. Validation ───────────────────────────────────────────────────────
    warns = _validate(
        N_carbon=N_carbon,
        phi_pre=phi_pre,
        vf_si=vf_si_val,
        si_d50_nm=si_d50,
        voxel_size_nm=sim.voxel_size_nm,
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
        validation_warnings=warns,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_wf_sum(wf_si, wf_carbon, wf_additive, wf_binder) -> None:
    total = wf_si + wf_carbon + wf_additive + wf_binder
    if not math.isclose(total, 1.0, abs_tol=1e-4):
        raise ValueError(
            f"Weight fractions must sum to 1.0 (got {total:.6f}). "
            f"  si={wf_si:.4f}, carbon={wf_carbon:.4f}, "
            f"additive={wf_additive:.4f}, binder={wf_binder:.4f}\n"
            f"  Hint: active_material_wt_frac is derived as "
            f"(1 - conductive_additive - binder)."
        )


def _validate(
    N_carbon: int,
    phi_pre: float,
    vf_si: float,
    si_d50_nm: float,
    voxel_size_nm: float,
) -> list[str]:
    """
    Returns a list of warning strings.
    Prefixed with [CRITICAL] for issues that will cause generation to fail.
    Prefixed with [INFO] for expected sub-voxel conditions.
    """
    warns: list[str] = []

    # Carbon count too low → not statistically representative
    if N_carbon < 20:
        warns.append(
            f"[CRITICAL] N_carbon={N_carbon} is too low for a representative "
            f"microstructure (minimum ~20). "
            f"Increase coating_thickness_um or decrease carbon_particle_d50_nm."
        )

    # Carbon count too high → RSA will be extremely slow
    if N_carbon > 5000:
        warns.append(
            f"N_carbon={N_carbon} is high — RSA packing will be slow. "
            f"Consider reducing coating_thickness_um."
        )

    # Pre-calendering solid fraction exceeds RSA jamming limit
    if phi_pre > _RSA_JAMMING_LIMIT:
        warns.append(
            f"[CRITICAL] phi_solid_pre={phi_pre:.3f} exceeds RSA jamming limit "
            f"(~{_RSA_JAMMING_LIMIT}) for oblate spheroids. "
            f"RSA will not converge. Options:\n"
            f"    (a) Decrease target_porosity (currently {1 - phi_pre/phi_pre:.2f})\n"
            f"    (b) Increase compression_ratio (currently {phi_pre:.2f})"
        )

    # Si sub-voxel — expected and handled, but flagged for transparency
    si_vox = si_d50_nm / voxel_size_nm
    if si_vox < 1.0:
        warns.append(
            f"[INFO] Si d50={si_d50_nm:.0f}nm is sub-voxel "
            f"({si_vox:.2f} voxels at {voxel_size_nm:.1f}nm/voxel). "
            f"Si will be placed as a volume-fraction map, not individual particles."
        )

    # Si volume fraction unreasonably high
    if vf_si > 0.25:
        warns.append(
            f"vf_Si={vf_si:.3f} (>{25}%) is unusually high. "
            f"Verify si_wt_frac_in_am={vf_si:.3f} is intentional."
        )

    return warns
