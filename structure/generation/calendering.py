"""
Step 5 — Calendering

Applies mechanical compression of the electrode to placed particles
and sub-voxel volume-fraction fields.

MUTATION SEMANTICS — two intentionally different patterns:

calender_particles(particles, ...)
    Mutates OblateSpheroid objects IN-PLACE.
    Rationale: particles are large mutable objects already owned by
    PackingResult. The pre-calender particle list is never needed after
    Step 5. Deep-copying numpy arrays (center, R, A_inv) per particle
    would add cost with no benefit.
    ⚠ Callers must not retain references to the original particle
    geometry and expect it to remain unchanged.

calender_fields(si_result, cbd_result, ...)
    Returns NEW SiMapResult and CBDBinderResult objects.
    The input objects are NOT modified.
    Rationale: SiMapResult and CBDBinderResult are dataclass value-objects
    carrying numpy array fields. Other pipeline steps may hold references
    to the pre-calender results for diagnostics or logging. Mutating the
    arrays in-place would corrupt those references silently.
    ✓ Callers may safely retain the original si_result / cbd_result
    alongside the returned compressed versions.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import zoom
from typing import Iterable, Tuple

from structure.schema import ResolvedGeneration
from structure.data import (
    DomainGeometry,
    CompositionState,
    OblateSpheroid,
    SiMapResult,
    CBDBinderResult,
)


# ---------------------------------------------------------------------------
# Particle transform
# ---------------------------------------------------------------------------


def calender_particles(
    particles: list[OblateSpheroid],
    comp: CompositionState,
    domain: DomainGeometry,
    particle_deformation: float = 1.0,
    orientation_enhancement: float = 0.0,
) -> tuple[list[OblateSpheroid], list[str]]:
    """
    Apply calendering transform to all placed particles in-place.

    For each particle:
      1. Compress Z centre: center_z *= cr
      2. Deform axes (volume-conserving):
           a_new = a × deform_factor^0.5
           c_new = c / deform_factor
         where deform_factor = 1 + particle_deformation × (1/cr - 1)
      3. SLERP c-axis toward ẑ by orientation_enhancement weight
      4. Recompute A_inv with updated (a_new, c_new, R_new)

    Post-transform, validate that every particle centre lies within
    [c_new, Lz_final - c_new] in Z and within [0, Lx] × [0, Ly] in XY
    (XY is periodic so only check positivity and max).

    Args:
        particles             : list of OblateSpheroid (mutated in-place)
        comp                  : CompositionState (provides compression_ratio)
        domain                : DomainGeometry (provides final domain bounds)
        particle_deformation   : 0.0 = rigid translation only, 1.0 = full deformation
        orientation_enhancement: 0.0 = no SLERP, 1.0 = full alignment to ẑ
    """
    cr = comp.compression_ratio
    Lx = domain.Lx_nm
    Ly = domain.Ly_nm
    Lz = domain.Lz_final_nm
    warns: list[str] = []

    n_clipped_z = 0
    n_outside_xy = 0

    for p in particles:
        # ── 1. Compress Z centre ─────────────────────────────────────────
        p.center[2] *= cr

        # ── 2. Volume-conserving axis deformation ────────────────────────
        p.invalidate_shape_matrix()
        deform_factor = 1.0 + particle_deformation * (1.0 / cr - 1.0)
        p.a = p.a * math.sqrt(deform_factor)
        p.c = p.c / deform_factor

        # ── 3. SLERP c-axis toward ẑ ─────────────────────────────────────
        if orientation_enhancement > 0.0:
            c_axis = p.R[:, 2]
            z_hat = np.array([0.0, 0.0, 1.0])
            dot = float(np.clip(np.dot(c_axis, z_hat), -1.0, 1.0))
            angle = math.acos(abs(dot)) * (1.0 - orientation_enhancement)
            axis = np.cross(c_axis, z_hat)
            ax_norm = np.linalg.norm(axis)
            if ax_norm > 1e-9:
                axis /= ax_norm
                K = np.array(
                    [
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0],
                    ],
                    dtype=np.float64,
                )
                Rot = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * K @ K
                p.R = Rot @ p.R

        # ── 4. Recompute A_inv ───────────────────────────────────────────
        p.recompute_shape_matrix()

        # ── 5. Post-transform domain bounds validation ───────────────────
        # Z: particle must fit within [0, Lz_final].
        # The RSA enforced center_z ∈ [c_pre, Lz_pre - c_pre] but after
        # axis deformation c_new ≠ c_pre — re-check with updated c.
        z_lo = p.center[2] - p.c
        z_hi = p.center[2] + p.c
        if z_lo < 0.0 or z_hi > Lz:
            n_clipped_z += 1
            p.center[2] = float(np.clip(p.center[2], p.c, Lz - p.c))

        # XY: periodic domain — centers must be in [0, Lx) × [0, Ly).
        # Wrap rather than clip (consistent with packer convention).
        if not (0.0 <= p.center[0] < Lx):
            p.center[0] = p.center[0] % Lx
            n_outside_xy += 1
        if not (0.0 <= p.center[1] < Ly):
            p.center[1] = p.center[1] % Ly
            n_outside_xy += 1

    # ── Warnings ─────────────────────────────────────────────────────────
    if n_clipped_z > 0:
        warns.append(
            f"calender_particles: {n_clipped_z}/{len(particles)} particles "
            f"had Z extent outside [0, {Lz:.1f}nm] after calendering — "
            f"centres clipped to [c_new, Lz - c_new]. "
            f"Consider reducing particle_deformation or compression_ratio."
        )
    if n_outside_xy > 0:
        warns.append(
            f"calender_particles: {n_outside_xy} particle centre coordinates "
            f"were outside [0, Lx/Ly) after calendering — wrapped periodically. "
            f"This should not happen; check packer XY bounds."
        )

    return particles, warns


# ---------------------------------------------------------------------------
# Field transforms (vf maps)
# ---------------------------------------------------------------------------


def calender_fields(
    si_result: SiMapResult,
    cbd_result: CBDBinderResult,
    comp: CompositionState,
    domain: DomainGeometry,
) -> Tuple[SiMapResult, CBDBinderResult]:
    """
    Resample Si / coating / void / CBD / binder vf maps along Z to account
    for calendering compression.

    zoom(field, [1, 1, cr], order=1) compresses the Z axis by cr.
    The zoomed output has shape (nx, ny, round(nz * cr)) which is then
    zero-padded back to (nx, ny, nz) — the extra Z voxels at the top
    correctly become pore after compression.

    Renormalization after zoom conserves the integrated phase volume:
        sum(out) * V_voxel == sum(original) * V_voxel
    """
    cr = comp.compression_ratio
    nz = domain.nz

    def _compress_vf(
        field: np.ndarray,
        field_name: str,
    ) -> tuple[np.ndarray, str | None]:
        """
        Compress a float vf field in Z, renormalize, clip to [0,1].

        Returns (compressed_field, warning_or_None).

        Volume loss path:
        zoom → renormalize → clip(0,1)
        Renormalization scales values to conserve V_before. But if zoomed
        values exceed 1.0 near high-VF boundaries, the subsequent clip
        silently removes that excess. A post-clip check catches this and
        emits a warning when deviation exceeds CLIP_VOLUME_LOSS_THRESHOLD.
        """
        CLIP_VOLUME_LOSS_THRESHOLD = 0.005  # 0.5%

        V_before = float(field.sum())
        if V_before < 1e-10:
            return np.zeros_like(field, dtype=np.float32), None

        zoomed = zoom(field.astype(np.float32), [1.0, 1.0, cr], order=1)
        nz_new = min(zoomed.shape[2], nz)
        out = np.zeros((field.shape[0], field.shape[1], nz), dtype=np.float32)
        out[:, :, :nz_new] = zoomed[:, :, :nz_new]

        # Renormalize before clip
        V_after_zoom = float(out.sum())
        if V_after_zoom > 1e-10:
            out *= V_before / V_after_zoom

        # Clip — may silently remove volume if renormalized values exceed 1.0
        V_pre_clip = float(out.sum())
        out = np.clip(out, 0.0, 1.0)
        V_post_clip = float(out.sum())

        warn = None
        if V_pre_clip > 1e-10:
            loss_frac = (V_pre_clip - V_post_clip) / V_pre_clip
            if loss_frac > CLIP_VOLUME_LOSS_THRESHOLD:
                warn = (
                    f"calender_fields: '{field_name}' lost "
                    f"{loss_frac * 100:.2f}% of integrated volume after "
                    f"zoom→renormalize→clip (V_pre_clip={V_pre_clip:.4e}, "
                    f"V_post_clip={V_post_clip:.4e}). "
                    f"High local VF near Z boundaries is being truncated. "
                    f"Consider reducing si_void_fraction or increasing "
                    f"voxel_resolution to reduce boundary gradients."
                )

        return out, warn

    def _compress_mask(mask: np.ndarray) -> np.ndarray:
        """Compress a boolean mask in Z using nearest-neighbour interpolation."""
        zoomed = zoom(mask.astype(np.float32), [1.0, 1.0, cr], order=0)
        nz_new = min(zoomed.shape[2], nz)
        out = np.zeros((mask.shape[0], mask.shape[1], nz), dtype=bool)
        out[:, :, :nz_new] = zoomed[:, :, :nz_new] > 0.5
        return out

    si_vf_c, w_si_vf = _compress_vf(si_result.si_vf, "si_vf")
    coating_vf_c, w_coating_vf = _compress_vf(si_result.coating_vf, "coating_vf")
    cbd_vf_c, w_cbd_vf = _compress_vf(cbd_result.cbd_vf, "cbd_vf")
    binder_vf_c, w_binder_vf = _compress_vf(cbd_result.binder_vf, "binder_vf")

    # Collect non-None warnings
    compress_warns = [
        w for w in [w_si_vf, w_coating_vf, w_cbd_vf, w_binder_vf] if w is not None
    ]

    new_si_result = SiMapResult(
        si_vf=si_vf_c,
        coating_vf=coating_vf_c,
        void_mask=_compress_mask(si_result.void_mask),
        void_enabled=si_result.void_enabled,
        V_si_actual_nm3=si_result.V_si_actual_nm3,
        V_si_target_nm3=si_result.V_si_target_nm3,
        mass_error_pct=si_result.mass_error_pct,
        distribution=si_result.distribution,
        warnings=si_result.warnings + compress_warns,
    )

    new_cbd_result = CBDBinderResult(
        cbd_vf=cbd_vf_c,
        binder_vf=binder_vf_c,
        V_cbd_nm3=cbd_result.V_cbd_nm3,
        V_binder_nm3=cbd_result.V_binder_nm3,
        V_cbd_target_nm3=cbd_result.V_cbd_target_nm3,
        V_binder_target_nm3=cbd_result.V_binder_target_nm3,
        cbd_mass_error_pct=cbd_result.cbd_mass_error_pct,
        binder_mass_error_pct=cbd_result.binder_mass_error_pct,
        cbd_percolating=cbd_result.cbd_percolating,
        warnings=cbd_result.warnings + compress_warns,
    )

    return new_si_result, new_cbd_result


# ---------------------------------------------------------------------------
# Public convenience
# ---------------------------------------------------------------------------


def apply_calendering(
    particles: Iterable[OblateSpheroid],
    comp: CompositionState,
    domain: DomainGeometry,
    si_result: SiMapResult,
    cbd_result: CBDBinderResult,
    sim: ResolvedGeneration,
) -> Tuple[SiMapResult, CBDBinderResult]:
    """
    Canonical entry point for Step 5.

    Args:
      particles   : carbon particles (Step 2)
      comp        : CompositionState
      domain      : DomainGeometry
      si_result   : SiMapResult (Step 3)
      cbd_result  : CBDBinderResult (Step 4)
      sim         : ResolvedGeneration (for calendering params)

    Returns:
      (si_result_out, cbd_result_out)  — currently identical to inputs
      after geometry has been calendered.
    """
    calender_particles(
        particles,
        comp,
        domain,
        particle_deformation=sim.calendering_particle_deformation,
        orientation_enhancement=sim.calendering_orientation_enhancement,
    )
    return calender_fields(si_result, cbd_result, comp, domain)
