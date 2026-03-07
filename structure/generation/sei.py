"""
Step 6 — SEI Shell Adder

Generates sei_vf[nx, ny, nz]: a float32 field where each surface voxel
value represents the local SEI volume fraction.

SEI is sub-voxel (15nm << 390nm/voxel → 0.038 VF per surface face).
It cannot be resolved as a geometric shell, so it's stored as a
volume-fraction field on solid-surface voxels.

Physics model:
  Each surface voxel has Nf faces exposed to pore space.
  The SEI volume contribution from that voxel is:
      V_sei = thickness_nm × Nf × vs²          [nm³]
      sei_vf = V_sei / vs³ = thickness_nm × Nf / vs

  Spatial variation in thickness is modeled as a GRF modulation:
      thickness_local = thickness_nm × max(0, 1 + uniformity_cv × Z)
  where Z ~ N(0,1) is a spatially correlated Gaussian field.

SEI forms on:
  - Carbon (graphite) surfaces
  - Si surfaces (where si_vf > threshold)

SEI does NOT form inside void zones — the expansion buffer is
a gap, not an electrolyte-filled region that would form SEI.
"""

from __future__ import annotations

import numpy as np
from typing import List
from scipy.ndimage import gaussian_filter, convolve

from structure.schema import ResolvedSimulation
from structure.data import CompositionState, DomainGeometry, SEIResult, SiMapResult
from structure.phases import PHASE_GRAPHITE


# ---------------------------------------------------------------------------
# SEIShellAdder
# ---------------------------------------------------------------------------
class SEIShellAdder:
    def __init__(
        self,
        comp: CompositionState,
        domain: DomainGeometry,
        sim: ResolvedSimulation,
    ) -> None:
        self.comp = comp
        self.domain = domain
        self.sim = sim

    # ── Public ───────────────────────────────────────────────────────────

    def add(
        self,
        carbon_label: np.ndarray,  # uint8 (nx,ny,nz)
        si_result: SiMapResult,
        rng: np.random.Generator,
    ) -> SEIResult:

        sim = self.sim
        domain = self.domain
        warns: List[str] = []

        if not sim.sei_enabled:
            return SEIResult(
                sei_vf=np.zeros(carbon_label.shape, dtype=np.float32),
                V_sei_nm3=0.0,
                surface_area_nm2=0.0,
                mean_thickness_nm=0.0,
                warnings=["SEI disabled in config."],
            )

        vs = domain.voxel_size_nm
        t_mean = sim.sei_thickness_nm  # mean SEI thickness (nm)
        unif_cv = sim.sei_uniformity_cv  # spatial variation coefficient

        # ------------------------------------------------------------------
        # 1. Build solid mask for SEI target surfaces
        # ------------------------------------------------------------------
        carbon_mask = carbon_label == PHASE_GRAPHITE

        # Si surface: anywhere si_vf is meaningful
        # Threshold at ~0.5× the expected mean interior value
        si_interior_threshold = 0.02
        si_mask = si_result.si_vf > si_interior_threshold

        # Solid = carbon ∪ Si  (excludes void buffer — no SEI in void gap)
        solid_mask = carbon_mask | si_mask
        pore_mask = ~solid_mask

        _FACE_KERNEL_XY = self._extracted_from_add_46(0, 1, 2, 1)
        _FACE_KERNEL_XY[0, 1, 1] = 1  # -X
        _FACE_KERNEL_XY[2, 1, 1] = 1  # +X

        _FACE_KERNEL_Z = self._extracted_from_add_46(1, 0, 1, 2)
        pore_int = pore_mask.astype(np.int16)
        exposed_faces_xy = convolve(
            pore_int, _FACE_KERNEL_XY.astype(np.int16), mode="wrap"
        )
        exposed_faces_z = convolve(
            pore_int, _FACE_KERNEL_Z.astype(np.int16), mode="constant", cval=1
        )
        exposed_faces = (exposed_faces_xy + exposed_faces_z).astype(np.int8)

        # Surface voxels: solid voxels with ≥1 exposed face
        surface_mask = solid_mask & (exposed_faces >= 1)

        if not surface_mask.any():
            warns.append("[CRITICAL] No surface voxels found — no SEI placed.")
            return SEIResult(
                sei_vf=np.zeros(carbon_label.shape, dtype=np.float32),
                V_sei_nm3=0.0,
                surface_area_nm2=0.0,
                mean_thickness_nm=0.0,
                warnings=warns,
            )

        # ------------------------------------------------------------------
        # 3. Base sei_vf from face count
        #    sei_vf_base = t_mean × Nf / vs
        # ------------------------------------------------------------------
        sei_vf = np.zeros(carbon_label.shape, dtype=np.float64)
        sei_vf[surface_mask] = (
            t_mean * exposed_faces[surface_mask].astype(np.float64) / vs
        )

        # ------------------------------------------------------------------
        # 4. Spatial thickness variation via GRF
        #
        #    t_local = t_mean × (1 + unif_cv × Z),  Z ~ corr. Gaussian
        #    We multiply sei_vf by the local thickness ratio (t_local/t_mean)
        #    and clip the Gaussian multiplier to [0, 3] so extreme outliers
        #    don't push sei_vf > 1.
        # ------------------------------------------------------------------
        if unif_cv > 1e-4:
            sigma_vox = max(
                0.5,
                sim.sei_correlation_length_nm / domain.voxel_size_nm,
            )
            # Correlated Gaussian field with sigma = 2 voxels (≈ 781nm, sub-particle)
            Z_raw = rng.normal(size=carbon_label.shape).astype(np.float32)
            Z_corr = gaussian_filter(Z_raw, sigma=sigma_vox)
            # Standardize to unit variance
            Z_corr /= Z_corr.std() + 1e-9
            # Thickness multiplier: 1 + cv × Z, clipped to [0, 3]
            t_mult = np.clip(1.0 + unif_cv * Z_corr, 0.0, 3.0)
            sei_vf[surface_mask] *= t_mult[surface_mask]

        # ------------------------------------------------------------------
        # 5. Clip to [0, 1] — sei_vf cannot exceed 1 by definition
        # ------------------------------------------------------------------
        sei_vf = np.clip(sei_vf, 0.0, 1.0)

        # ------------------------------------------------------------------
        # 6. Diagnostics
        # ------------------------------------------------------------------
        V_voxel = vs**3
        V_sei = float(sei_vf.sum() * V_voxel)

        # Surface area from face count:
        # Each exposed face contributes vs² of interface area
        surface_area = float(exposed_faces[surface_mask].sum()) * (vs**2)

        # Effective thickness
        t_effective = V_sei / surface_area if surface_area > 0.0 else 0.0

        if t_effective < 0.5 * t_mean:
            warns.append(
                f"Effective SEI thickness ({t_effective:.2f}nm) is less than "
                f"half the configured mean ({t_mean:.2f}nm). "
                f"Surface area may be underestimated due to sub-voxel particles."
            )

        return SEIResult(
            sei_vf=sei_vf.astype(np.float32),
            V_sei_nm3=V_sei,
            surface_area_nm2=surface_area,
            mean_thickness_nm=t_effective,
            warnings=warns,
        )

    def _extracted_from_add_46(self, arg0, arg1, arg2, arg3):
        # ------------------------------------------------------------------
        # 2. Count exposed faces per solid voxel
        # ------------------------------------------------------------------
        # X and Y are periodic — pad with wrap, convolve only Z-faces with constant.
        # Split the 6-face kernel into XY faces and Z faces separately so we can
        # apply the correct boundary condition per axis.

        result = np.zeros((3, 3, 3), dtype=np.int8)
        result[1, arg0, arg1] = 1
        result[1, arg2, arg3] = 1
        return result


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def add_sei_shell(
    comp: CompositionState,
    domain: DomainGeometry,
    sim: ResolvedSimulation,
    carbon_label: np.ndarray,
    si_result: SiMapResult,
    rng: np.random.Generator,
) -> SEIResult:
    """
    Canonical pipeline entry for Step 6.

    Args:
      comp         : CompositionState (Step 0)
      domain       : DomainGeometry (Step 1)
      sim          : ResolvedSimulation
      carbon_label : uint8 label map post-calendering (Step 2)
      si_result    : SiMapResult post-calendering (Step 3)
      rng          : seeded Generator

    Returns:
      SEIResult with sei_vf field and diagnostics.
    """
    return SEIShellAdder(comp, domain, sim).add(carbon_label, si_result, rng)
