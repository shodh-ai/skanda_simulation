import math
from structure.constants import _RSA_JAMMING_LIMIT


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
    N_si: int,
    phi_pre: float,
    vf_si: float,
    si_d50_nm: float,
    voxel_size_nm: float,
    voxel_resolution: int,
    porosity: float,
    compression_ratio: float,
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
            f" (a) Decrease target_porosity (currently {porosity:.2f})\n"
            f" (b) Increase compression_ratio (currently {compression_ratio:.2f})"
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
            f"vf_Si={vf_si:.3f} (>25%) is unusually high. Verify si_wt_frac_in_am={vf_si:.3f} is intentional."
        )

    N_voxels_total = voxel_resolution**3
    particles_per_voxel = N_si / N_voxels_total if N_voxels_total > 0 else 0.0

    if N_si < 100:
        warns.append(
            f"[CRITICAL] N_si={N_si:,} is extremely low. "
            f"The Si VF field will have very high statistical variance "
            f"({particles_per_voxel:.2f} particles/voxel on average). "
            f"Si distribution mode results will be unreliable. "
            f"Increase coating_thickness_um or si_wt_frac_in_am."
        )
    elif N_si < 1_000:
        warns.append(
            f"N_si={N_si:,} is low ({particles_per_voxel:.2f} particles/voxel). "
            f"Statistical quality of the Si VF field may be marginal. "
            f"Consider increasing domain size or Si weight fraction."
        )
    elif N_si > 50_000_000:
        warns.append(
            f"[INFO] N_si={N_si:,.0f} ({particles_per_voxel:.0f} particles/voxel). "
            f"Very high Si particle count — VF field statistical quality is excellent. "
            f"This is normal for nano Si (d50={si_d50_nm:.0f}nm) at this domain size."
        )
    else:
        warns.append(
            f"[INFO] N_si={N_si:,.0f} ({particles_per_voxel:.1f} particles/voxel). "
            f"Si VF field statistical representation: "
            f"{'good' if particles_per_voxel >= 10 else 'adequate'}."
        )

    return warns
