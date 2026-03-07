import numpy as np
from structure.data import (
    CBDBinderResult,
    CompositionState,
    DomainGeometry,
    MicrostructureVolume,
    PackingResult,
    PercolationResult,
    SEIResult,
    SiMapResult,
    VolumeMetadata,
)
from structure.schema import ResolvedSimulation
from structure.phases import PHASE_GRAPHITE


def _validate_volume(
    carbon_vf: np.ndarray,
    si_vf: np.ndarray,
    coating_vf: np.ndarray,
    cbd_vf: np.ndarray,
    binder_vf: np.ndarray,
    sei_vf: np.ndarray,
    pore_vf: np.ndarray,
    meta: VolumeMetadata,
) -> list[str]:
    """
    Three physical self-consistency checks run at assembly time.
    All use float64 accumulation to avoid float32 precision drift.
    """
    warns: list[str] = []
    N = carbon_vf.size

    # ── 1. Phase overlap ─────────────────────────────────────────────────
    # Solid phases can legitimately share sub-voxel space (e.g. SEI on
    # carbon surface), so some overlap is expected. Flag if excessive.
    solid_sum = (
        carbon_vf.astype(np.float64)
        + si_vf.astype(np.float64)
        + coating_vf.astype(np.float64)
        + cbd_vf.astype(np.float64)
        + binder_vf.astype(np.float64)
        + sei_vf.astype(np.float64)
    )
    overlap_frac = float((solid_sum > 1.01).sum()) / N

    if overlap_frac > 0.05:
        warns.append(
            f"[CRITICAL] {overlap_frac*100:.2f}% of voxels have solid_sum > 1.01. "
            f"Phase fields are significantly overlapping — check Si / CBD / "
            f"coating VF normalization steps."
        )
    elif overlap_frac > 0.001:
        warns.append(
            f"solid_sum > 1.01 in {overlap_frac*100:.3f}% of voxels. "
            f"Minor sub-voxel overlap at phase boundaries — expected for "
            f"surface-anchored phases (SEI, coating, binder necks)."
        )

    carbon_coating_overlap = float(((carbon_vf > 0.5) & (coating_vf > 0.01)).sum()) / N
    if carbon_coating_overlap > 0.001:
        warns.append(
            f"carbon+coating co-occupancy in {carbon_coating_overlap*100:.3f}% "
            f"of voxels (carbon_vf>0.5 AND coating_vf>0.01). "
            f"Coating was not correctly suppressed inside carbon particles — "
            f"check _build_coating_vf in si_mapper.py."
        )

    # ── 2. Porosity consistency ───────────────────────────────────────────
    measured_por = float(pore_vf.astype(np.float64).mean())
    por_delta = abs(measured_por - meta.target_porosity)
    if por_delta > 0.05:
        warns.append(
            f"Measured porosity {measured_por:.4f} deviates from target "
            f"{meta.target_porosity:.4f} by {por_delta*100:.2f} pp. "
            f"Check composition normalization and calendering transform."
        )

    # ── 3. Coating-Si volume ratio ────────────────────────────────────────
    # For a thin spherical shell: V_shell / V_sphere ≈ 3t/r
    # → sum(coating_vf) / sum(si_vf) ≈ 3t/r
    if meta.si_coating_enabled and meta.si_coating_thickness_nm > 0:
        r_nm = meta.si_d50_nm / 2.0
        expected_ratio = 3.0 * meta.si_coating_thickness_nm / r_nm
        V_si = float(si_vf.astype(np.float64).sum())
        V_coating = float(coating_vf.astype(np.float64).sum())
        if V_si > 1e-6:
            actual_ratio = V_coating / V_si
            ratio_err = abs(actual_ratio - expected_ratio) / (expected_ratio + 1e-12)
            if ratio_err > 0.10:
                warns.append(
                    f"Coating/Si volume ratio: actual={actual_ratio:.4f}, "
                    f"expected≈{expected_ratio:.4f} "
                    f"(3t/r = 3×{meta.si_coating_thickness_nm}nm / {r_nm:.1f}nm). "
                    f"Deviation={ratio_err*100:.1f}% — coating VF may be "
                    f"inconsistent with Si loading."
                )

    return warns


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def assemble_volume(
    comp: CompositionState,
    domain: DomainGeometry,
    sim: ResolvedSimulation,
    packing: PackingResult,
    carbon_label: np.ndarray,
    si_result: SiMapResult,
    cbd_result: CBDBinderResult,
    sei_result: SEIResult,
    perc_result: PercolationResult,
) -> MicrostructureVolume:
    """
    Assemble all generation outputs into a MicrostructureVolume.

    Called immediately after Step 7 (percolation) succeeds.
    This is the canonical pipeline terminal — no further steps needed.

    Carbon VF is cast from the binary uint8 carbon_label rather than
    re-rasterized: at 390nm/voxel, graphite particles span ~30 voxels
    and are fully resolved, so binary 0/1 is analytically accurate.

    Args:
        comp         : CompositionState (Step 0)
        domain       : DomainGeometry   (Step 1)
        sim          : ResolvedSimulation
        packing      : PackingResult    (Step 2)
        carbon_label : uint8 intermediate label map (from test.py rasterizer)
        si_result    : SiMapResult post-calendering  (Steps 3/5)
        cbd_result   : CBDBinderResult post-calendering (Steps 4/5)
        sei_result   : SEIResult  (Step 6)
        perc_result  : PercolationResult (Step 7)

    Returns:
        MicrostructureVolume with all phase VF fields + validated metadata.
    """
    # Carbon: binary label → float32
    carbon_vf = (carbon_label == PHASE_GRAPHITE).astype(np.float32)

    si_vf = si_result.si_vf.astype(np.float32)
    coating_vf = si_result.coating_vf.astype(np.float32)
    cbd_vf = cbd_result.cbd_vf.astype(np.float32)
    binder_vf = cbd_result.binder_vf.astype(np.float32)
    sei_vf = sei_result.sei_vf.astype(np.float32)

    # Derive pore_vf (float64 accumulation then cast)
    solid_sum = (
        carbon_vf.astype(np.float64)
        + si_vf.astype(np.float64)
        + coating_vf.astype(np.float64)
        + cbd_vf.astype(np.float64)
        + binder_vf.astype(np.float64)
        + sei_vf.astype(np.float64)
    )
    pore_vf = np.clip(1.0 - solid_sum, 0.0, 1.0).astype(np.float32)
    measured_porosity = float(pore_vf.astype(np.float64).mean())

    meta = VolumeMetadata(
        run_id=sim.run_id,
        seed=sim.seed,
        voxel_size_nm=domain.voxel_size_nm,
        voxel_resolution=sim.voxel_resolution,
        electrode_thickness_um=domain.Lx_nm / 1000.0,
        wf_si=comp.wf_si,
        wf_carbon=comp.wf_carbon,
        wf_additive=comp.wf_additive,
        wf_binder=comp.wf_binder,
        vf_si=comp.vf_si,
        vf_carbon=comp.vf_carbon,
        vf_additive=comp.vf_additive,
        vf_binder=comp.vf_binder,
        target_porosity=comp.porosity,
        measured_porosity=measured_porosity,
        capacity_total_mah=comp.capacity_total_mah,
        capacity_si_fraction=comp.capacity_si_fraction,
        volumetric_capacity_mah_cm3=comp.volumetric_capacity_mah_cm3,
        electronic_fraction=perc_result.electronic_fraction,
        ionic_fraction=perc_result.ionic_fraction,
        electronic_percolating=perc_result.electronic_percolating,
        ionic_percolating=perc_result.ionic_percolating,
        n_carbon_particles=packing.N_placed,
        carbon_d50_nm=comp.carbon_d50_nm,
        si_d50_nm=comp.si_d50_nm,
        si_coating_enabled=sim.silicon.coating_enabled,
        si_coating_thickness_nm=sim.silicon.coating_thickness_nm,
    )

    warns = _validate_volume(
        carbon_vf,
        si_vf,
        coating_vf,
        cbd_vf,
        binder_vf,
        sei_vf,
        pore_vf,
        meta,
    )

    return MicrostructureVolume(
        carbon_vf=carbon_vf,
        si_vf=si_vf,
        coating_vf=coating_vf,
        cbd_vf=cbd_vf,
        binder_vf=binder_vf,
        sei_vf=sei_vf,
        pore_vf=pore_vf,
        metadata=meta,
        warnings=warns,
    )
