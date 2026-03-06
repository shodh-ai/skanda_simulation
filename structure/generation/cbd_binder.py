"""
Step 4 — CBD + Binder Fill (Gaussian Random Field)

Produces:
  cbd_vf[nx,ny,nz]    : conductive additive volume fraction (float32)
  binder_vf[nx,ny,nz] : binder volume fraction (float32)

Both are sub-voxel fields. CBD is generated as a Gaussian Random Field
(GRF) in the interstitial space and normalized to the target volume.
Binder is concentrated at carbon contact "necks" and smeared spatially.

Inputs:
  - CompositionState (V_CA, V_B targets, etc.)
  - DomainGeometry
  - ResolvedSimulation (distribution modes)
  - carbon_label (0=pore, 1=graphite)
  - si_result (SiMapResult: si_vf, void_mask)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt,
    generate_binary_structure,
    convolve,
)

from structure.schema.resolved import ResolvedSimulation


from .composition import CompositionState
from .domain import DomainGeometry
from .si_mapper import SiMapResult
from ..phases import PHASE_PORE, PHASE_GRAPHITE
from ._percolation_utils import check_percolates_z


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class CBDBinderResult:
    cbd_vf: np.ndarray  # float32 (nx,ny,nz)
    binder_vf: np.ndarray  # float32 (nx,ny,nz)
    V_cbd_nm3: float
    V_binder_nm3: float
    V_cbd_target_nm3: float
    V_binder_target_nm3: float
    cbd_mass_error_pct: float
    binder_mass_error_pct: float
    cbd_percolating: bool
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  CBD + BINDER FILL",
            "=" * 62,
            f"  CBD volume   : {self.V_cbd_nm3:.4e} nm³  "
            f"(target {self.V_cbd_target_nm3:.4e} nm³, "
            f"err={self.cbd_mass_error_pct:.3f}%)",
            f"  Binder volume: {self.V_binder_nm3:.4e} nm³  "
            f"(target {self.V_binder_target_nm3:.4e} nm³, "
            f"err={self.binder_mass_error_pct:.3f}%)",
            f"  CBD percolates in 3D: {self.cbd_percolating}",
            f"  CBD voxels>0       : {(self.cbd_vf>0).sum():,}",
            f"  Binder voxels>0    : {(self.binder_vf>0).sum():,}",
        ]
        if self.warnings:
            lines += ["", f"  ⚠  {len(self.warnings)} WARNING(s):"]
            lines += [f"     [{i+1}] {w}" for i, w in enumerate(self.warnings)]
        lines.append("=" * 62)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main mapper
# ---------------------------------------------------------------------------


class CBDBinderMapper:
    """
    Step 4 implementation. Uses GRF + distance weighting for CBD,
    and contact-based localization for binder.
    """

    MAX_CBD_RETRIES: int = 5

    def __init__(
        self, comp: CompositionState, domain: DomainGeometry, sim: ResolvedSimulation
    ) -> None:
        self.comp = comp
        self.domain = domain
        self.sim = sim

    # ── Public entry point ────────────────────────────────────────────────

    def fill(
        self,
        carbon_label: np.ndarray,
        si_result: SiMapResult,
        rng: np.random.Generator,
    ) -> CBDBinderResult:
        comp = self.comp
        domain = self.domain
        sim = self.sim

        # ------------------------------------------------------------------
        # CBD / conductive additive
        # ------------------------------------------------------------------
        cbd_vf, cbd_warnings, cbd_perc = self._make_cbd_vf(
            carbon_label=carbon_label,
            si_result=si_result,
            rng=rng,
        )

        # ------------------------------------------------------------------
        # Binder
        # ------------------------------------------------------------------
        binder_vf, binder_warnings = self._make_binder_vf(
            carbon_label=carbon_label,
            rng=rng,
        )

        # ------------------------------------------------------------------
        # Volumes + errors
        # ------------------------------------------------------------------
        V_voxel = domain.voxel_size_nm**3
        V_cbd = float(cbd_vf.sum() * V_voxel)
        V_binder = float(binder_vf.sum() * V_voxel)
        V_cbd_t = comp.V_additive_nm3
        V_binder_t = comp.V_binder_nm3

        err_cbd = abs(V_cbd - V_cbd_t) / V_cbd_t * 100.0 if V_cbd_t > 0 else 0.0
        err_binder = (
            abs(V_binder - V_binder_t) / V_binder_t * 100.0 if V_binder_t > 0 else 0.0
        )

        warns = cbd_warnings + binder_warnings

        return CBDBinderResult(
            cbd_vf=cbd_vf.astype(np.float32),
            binder_vf=binder_vf.astype(np.float32),
            V_cbd_nm3=V_cbd,
            V_binder_nm3=V_binder,
            V_cbd_target_nm3=V_cbd_t,
            V_binder_target_nm3=V_binder_t,
            cbd_mass_error_pct=err_cbd,
            binder_mass_error_pct=err_binder,
            cbd_percolating=cbd_perc,
            warnings=warns,
        )

    # ----------------------------------------------------------------------
    # CBD — GRF + percolation
    # ----------------------------------------------------------------------

    def _make_cbd_vf(
        self,
        carbon_label: np.ndarray,
        si_result: SiMapResult,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[str], bool]:
        comp = self.comp
        domain = self.domain
        sim = self.sim

        nx, ny, nz = carbon_label.shape
        V_voxel = domain.voxel_size_nm**3

        V_target = comp.V_additive_nm3
        if V_target <= 0.0:
            return np.zeros((nx, ny, nz), dtype=np.float32), [], False

        # Effective "solid" mask: anything that isn't pure pore
        # CBD lives in the interstitial region between carbon and other solids.
        pore_mask = carbon_label == PHASE_PORE
        carbon_mask = carbon_label == PHASE_GRAPHITE
        si_mask = si_result.si_vf > 1e-3
        void_mask = si_result.void_mask

        # CBD possible region: not carbon, not void-only; prefer near carbon/Si
        base_region = pore_mask | si_mask | void_mask

        if sim.additive_distribution == "aggregate":
            # more clumpy: restrict to pore-only region
            base_region = pore_mask

        if not base_region.any():
            return (
                np.zeros((nx, ny, nz), dtype=np.float32),
                ["No available region for CBD placement (base_region empty)."],
                False,
            )

        # Correlation length in voxels from aggregate size
        agg_nm = getattr(sim.additive, "primary_particle_nm", 40.0)
        corr_len = max(0.5, agg_nm / domain.voxel_size_nm)  # ≥ 0.5 voxels

        warnings = []
        percolates = False
        cbd_vf = np.zeros((nx, ny, nz), dtype=np.float32)

        for attempt in range(self.MAX_CBD_RETRIES):
            # 1. GRF on full domain
            noise = rng.normal(size=(nx, ny, nz)).astype(np.float32)
            field = gaussian_filter(noise, sigma=corr_len)

            # 2. Bias toward carbon surfaces: closer to carbon → higher weight
            dist_to_carbon = distance_transform_edt(~carbon_mask)  # 0 at carbon
            # Scale to [0, 1] within base_region
            d = dist_to_carbon[base_region]
            if d.size > 0:
                d_norm = d / (d.max() + 1e-9)
                bias = np.exp(-d_norm * 2.0)  # exp decay away from carbon
                field[base_region] *= 0.5 + bias  # 0.5–1.5 multiplier

            # 3. Clip to base_region, zero elsewhere, positive only
            field[~base_region] = -np.inf  # exclude impossible voxels
            field = np.maximum(field, -10.0)

            # 4. Convert field to vf by ranking: highest values get CBD
            flat = field.ravel()
            idx = flat.argsort()[::-1]  # descending
            n_target_vox = int(np.ceil(V_target / V_voxel))

            vf = np.zeros_like(flat, dtype=np.float32)
            n_target_vox = min(n_target_vox, flat.size)
            vf[idx[:n_target_vox]] = 1.0  # initial binary allocation

            cbd_vf_candidate = vf.reshape(nx, ny, nz)
            cbd_vf_candidate[~base_region] = 0.0

            # Smooth slightly to remove pixel noise and create aggregates
            cbd_vf_candidate = gaussian_filter(cbd_vf_candidate, sigma=corr_len / 2.0)
            cbd_vf_candidate = np.clip(cbd_vf_candidate, 0.0, 1.0)

            # Normalize to exact volume target
            V_current = float(cbd_vf_candidate.sum() * V_voxel)
            if V_current <= 0.0:
                warnings.append(
                    f"CBD attempt {attempt+1}: zero volume after smoothing."
                )
                continue

            scale = V_target / V_current
            cbd_vf_candidate *= scale
            cbd_vf_candidate = np.clip(cbd_vf_candidate, 0.0, 1.0)

            # Percolation check: CBD + carbon combined must percolate
            percolates = check_percolates_z((cbd_vf_candidate > 0.05) | carbon_mask)
            if percolates:
                cbd_vf = cbd_vf_candidate
                break

            warnings.append(
                f"CBD attempt {attempt+1}: network not percolating — retrying."
            )

        if not percolates:
            warnings.append(
                f"CBD GRF: failed to produce percolating network after "
                f"{self.MAX_CBD_RETRIES} attempts. Using last realization."
            )
            cbd_vf = cbd_vf_candidate

        return cbd_vf, warnings, percolates

    # ----------------------------------------------------------------------
    # Binder — necks at carbon contacts
    # ----------------------------------------------------------------------

    def _make_binder_vf(
        self,
        carbon_label: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[str]]:
        comp = self.comp
        domain = self.domain
        sim = self.sim

        nx, ny, nz = carbon_label.shape
        V_voxel = domain.voxel_size_nm**3
        V_target = comp.V_binder_nm3

        if V_target <= 0.0:
            return np.zeros((nx, ny, nz), dtype=np.float32), []

        binder_vf = np.zeros((nx, ny, nz), dtype=np.float32)

        carbon_mask = carbon_label == PHASE_GRAPHITE
        pore_mask = carbon_label == PHASE_PORE

        # Contact voxels = carbon voxels with ≥2 carbon neighbors
        struct = generate_binary_structure(3, 1)  # 6-connectivity
        carbon_neighbors = convolve(
            carbon_mask.astype(np.int16),
            struct.astype(np.int16),
            mode="constant",
            cval=0,
        )
        # subtract self-count to get true neighbors
        carbon_neighbors = carbon_neighbors - carbon_mask.astype(np.int16)
        contact_mask = carbon_mask & (carbon_neighbors >= 2)

        if not contact_mask.any():
            return np.zeros_like(binder_vf), [
                "No carbon contacts detected — binder 'necks' cannot be formed."
            ]

        # Distribution mode
        dist = sim.binder_distribution

        if dist == "uniform":
            # Simple: binder follows carbon volume
            binder_vf[carbon_mask] = 1.0
        elif dist == "patchy":
            # Binder blobs — GRF restricted to carbon+near-pore
            base = carbon_mask | (distance_transform_edt(~carbon_mask) <= 1.5)
            field = gaussian_filter(
                rng.normal(size=(nx, ny, nz)).astype(np.float32),
                sigma=1.0,
            )
            field[~base] = -np.inf
            flat = field.ravel()
            idx = flat.argsort()[::-1]
            n_target = int(np.ceil(V_target / V_voxel))
            n_target = min(n_target, flat.size)
            vf = np.zeros_like(flat, dtype=np.float32)
            vf[idx[:n_target]] = 1.0
            binder_vf = vf.reshape(nx, ny, nz)
        else:  # "necks"
            # 1. Start at contact voxels
            binder_vf[contact_mask] = 1.0

            # 2. Spread into neighboring pore region with Gaussian kernel
            # scale in voxels from binder film thickness
            t_nm = sim.binder.film_thickness_max_nm
            sigma_vox = max(0.5, t_nm / domain.voxel_size_nm)  # ~0.04 → clamp 0.5
            binder_vf = gaussian_filter(binder_vf, sigma=sigma_vox)

            # Allow binder only in carbon + immediate pore vicinity
            mask_near_carbon = distance_transform_edt(~carbon_mask) <= 1.5
            binder_vf[~mask_near_carbon] = 0.0

        # Normalize to volume target
        V_current = float(binder_vf.sum() * V_voxel)
        warnings: List[str] = []
        if V_current <= 0.0:
            warnings.append(
                "Binder vf is zero after initial placement — using uniform distribution on carbon."
            )
            binder_vf[carbon_mask] = 1.0
            V_current = float(binder_vf.sum() * V_voxel)

        scale = V_target / V_current
        binder_vf *= scale
        binder_vf = np.clip(binder_vf, 0.0, 1.0)

        return binder_vf, warnings


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def fill_cbd_binder(
    comp: CompositionState,
    domain: DomainGeometry,
    sim: ResolvedSimulation,
    carbon_label: np.ndarray,
    si_result: SiMapResult,
    rng: np.random.Generator,
) -> CBDBinderResult:
    """
    Canonical pipeline entry for Step 4.

    Args:
      comp         : CompositionState (Step 0)
      domain       : DomainGeometry (Step 1)
      sim          : ResolvedSimulation
      carbon_label : uint8 label map (Step 2)
      si_result    : SiMapResult (Step 3)
      rng          : seeded Generator

    Returns:
      CBDBinderResult with vf maps + diagnostics.
    """
    mapper = CBDBinderMapper(comp, domain, sim)
    return mapper.fill(carbon_label=carbon_label, si_result=si_result, rng=rng)
