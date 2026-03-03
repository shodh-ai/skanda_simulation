"""
Step 5 — Calendering Transform

Applies mechanical calendering to the *geometry*:

  - Compresses particle centers in Z by the global compression_ratio
  - Flattens particles in Z and widens them in XY (volume-conserving)
  - Increases c-axis alignment toward Z (orientation_enhancement)
  - Resamples Si / CBD / binder volume-fraction fields in Z

DomainGeometry itself stays fixed (final, post-calender cube).
Only particle coordinates/shapes and vf fields are transformed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy.ndimage import zoom

from .domain import DomainGeometry
from .composition import CompositionState
from .carbon_packer import OblateSpheroid
from .si_mapper import SiMapResult
from .cbd_binder import CBDBinderResult


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _blend_toward_z_aligned(R: np.ndarray, strength: float) -> np.ndarray:
    """
    Blend the particle's current orientation R toward Z-aligned.

    strength ∈ [0, 1]:
      0 → no change
      1 → fully Z-aligned (c-axis || ẑ)

    Implementation:
      - Extract current c-axis = R[:, 2]
      - Slerp between c-axis and ẑ with weight=strength
      - Build new orthonormal frame (e1, e2, c_new)

    This keeps the particle's c-axis alignment consistent with the
    macroscopic calendering direction while preserving randomness
    around that axis.
    """
    if strength <= 0.0:
        return R

    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    c_old = R[:, 2] / np.linalg.norm(R[:, 2])

    # Compute blended c-axis
    dot = float(np.clip(np.dot(c_old, z_hat), -1.0, 1.0))
    if dot > 0.999:
        c_new = c_old
    elif dot < -0.999:
        c_new = -c_old
    else:
        angle = np.arccos(dot)
        t = np.clip(strength, 0.0, 1.0)
        # Slerp on the unit sphere
        s1 = np.sin((1 - t) * angle) / np.sin(angle)
        s2 = np.sin(t * angle) / np.sin(angle)
        c_new = s1 * c_old + s2 * z_hat
        c_new /= np.linalg.norm(c_new)

    # Build orthonormal frame with c_new as third axis
    # Start with old e1, project out c_new component, renormalize
    e1_old = R[:, 0]
    e1 = e1_old - np.dot(e1_old, c_new) * c_new
    n_e1 = np.linalg.norm(e1)
    if n_e1 < 1e-8:
        # e1_old was ~parallel to c_new: pick any transverse vector
        e1 = np.array([1.0, 0.0, 0.0], dtype=float)
        e1 -= np.dot(e1, c_new) * c_new
        e1 /= np.linalg.norm(e1)
    else:
        e1 /= n_e1
    e2 = np.cross(c_new, e1)

    return np.column_stack([e1, e2, c_new])


# ---------------------------------------------------------------------------
# Particle transform
# ---------------------------------------------------------------------------


def calender_particles(
    particles: Iterable[OblateSpheroid],
    comp: CompositionState,
    domain: DomainGeometry,
    particle_deformation: float,
    orientation_enhancement: float,
) -> None:
    """
    In-place calendering transform on all carbon particles.

    Args:
      particles             : list of OblateSpheroid (pre-calender coordinates)
      comp                  : CompositionState (for compression_ratio)
      domain                : DomainGeometry
      particle_deformation  : 0 → rigid; 1 → maximum flattening
      orientation_enhancement: 0 → no extra alignment; 1 → strong Z alignment

    Effects per particle:
      center_z *= compression_ratio
      a *= sqrt(deform_factor)
      c *= 1 / deform_factor
      R  = blend_toward_z_aligned(R, orientation_enhancement)

    deform_factor = 1 + particle_deformation * (1/compression_ratio - 1)
      particle_deformation=0 → deform_factor=1 (no shape change)
      particle_deformation=1 → a widened, c scaled by 1/compression_ratio

    Volume preservation:
      V ∝ a²c; scaling gives a'² c' = (f a)² (c/f²) = a² c.
    """
    cr = comp.compression_ratio
    if cr <= 0.0 or cr > 1.0:
        raise ValueError(f"Invalid compression_ratio={cr:.3f}")

    pd = float(np.clip(particle_deformation, 0.0, 1.0))
    deform_factor = 1.0 + pd * (1.0 / cr - 1.0)  # ≥ 1.0
    xy_scale = deform_factor**0.5
    z_scale = 1.0 / deform_factor

    for p in particles:
        # Compress center in Z to final domain
        p.center[2] *= cr

        # Deform particle shape (volume-conserving)
        p.a *= xy_scale
        p.c *= z_scale

        # Update orientation toward Z alignment
        p.R = _blend_toward_z_aligned(p.R, orientation_enhancement)

        # NOTE: if you cached A_inv in the spheroid, you MUST recompute here.
        # Either:
        #   - store only (a, c, R) and let overlap tests recompute A_inv, or
        #   - recompute A_inv after calendering.
        # Since RSA is finished at this point, we do NOT need A_inv anymore
        # for overlap, only for voxelization (which uses analytical (x/a,y/a,z/c)
        # membership). So we don't need to update A_inv.


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
    Resample Si / coating / CBD / binder vf maps along Z to account for
    calendering (thickness compression).

    Strategy:
      - All vf maps currently live in the FINAL cubic domain grid already
        (128³). Steps 2–4 used domain.voxel_size_nm & compression_ratio
        consistently, so no further Z resampling is strictly required.
      - However, if we later support pre-calender grids, this function
        is the correct place to zoom in Z.

    For now, calender_fields is a no-op that simply returns the inputs.

    If we ever adopt pre-calender grids (e.g., nz_pre ≠ nz_final), we will:
      - Use zoom(field, [1, 1, compression_ratio], order=1)
      - Re-normalize to preserve integrated volume for each phase.
    """
    # Placeholder for future: currently all fields are in final domain coordinates.
    return si_result, cbd_result


# ---------------------------------------------------------------------------
# Public convenience
# ---------------------------------------------------------------------------


def apply_calendering(
    particles: Iterable[OblateSpheroid],
    comp: CompositionState,
    domain: DomainGeometry,
    si_result: SiMapResult,
    cbd_result: CBDBinderResult,
    sim,
) -> Tuple[SiMapResult, CBDBinderResult]:
    """
    Canonical entry point for Step 5.

    Args:
      particles   : carbon particles (Step 2)
      comp        : CompositionState
      domain      : DomainGeometry
      si_result   : SiMapResult (Step 3)
      cbd_result  : CBDBinderResult (Step 4)
      sim         : ResolvedSimulation (for calendering params)

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
