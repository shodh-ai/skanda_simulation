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
  - ResolvedGeneration (distribution modes)
  - carbon_label (0=pore, 1=graphite)
  - si_result (SiMapResult: si_vf, void_mask)
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt,
    convolve,
)

from structure.schema import ResolvedGeneration
from structure.data import (
    CBDBinderResult,
    CompositionState,
    DomainGeometry,
    SiMapResult,
)
from structure.phases import PHASE_PORE, PHASE_GRAPHITE
from structure.utils.percolation import check_percolates_z


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
        self, comp: CompositionState, domain: DomainGeometry, sim: ResolvedGeneration
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
        # Only treat void zones as available CBD space if void was actually computed.
        # If void_enabled=False, si_result.void_mask is an empty sentinel — using it
        # as a region mask is correct (empty contributes nothing) but the intent
        # should be explicit.
        void_mask = si_result.void_mask if si_result.void_enabled else None

        # CBD possible region: not carbon, not void-only; prefer near carbon/Si
        base_region = pore_mask | si_mask
        if void_mask is not None:
            base_region = base_region | void_mask

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
        agg_nm = sim.additive.primary_particle_nm
        corr_len = max(0.5, agg_nm / domain.voxel_size_nm)  # ≥ 0.5 voxels

        warnings = []
        percolates = False
        cbd_vf = np.zeros((nx, ny, nz), dtype=np.float32)
        cbd_vf_candidate = np.zeros((nx, ny, nz), dtype=np.float32)

        # Precompute distance transform once — carbon_mask is fixed across all retries.
        # distance_transform_edt is O(nx×ny×nz); moving it outside saves
        # (MAX_CBD_RETRIES - 1) redundant calls.
        dist_to_carbon = distance_transform_edt(~carbon_mask)  # 0 at carbon surface
        # Precompute the bias weights for base_region too — base_region is also
        # fixed across retries (depends only on carbon_mask, si_mask, void_mask).
        _d = dist_to_carbon[base_region]
        if _d.size > 0:
            _d_norm = _d / (_d.max() + 1e-9)
            _bias = np.exp(-_d_norm * 2.0)  # shape: (n_base_region_voxels,)
        else:
            _bias = None

        for attempt in range(self.MAX_CBD_RETRIES):
            # 1. GRF on full domain
            noise = rng.normal(size=(nx, ny, nz)).astype(np.float32)
            field = gaussian_filter(noise, sigma=corr_len)

            # 2. Bias toward carbon surfaces: closer to carbon → higher weight
            if _bias is not None:
                field[base_region] *= 0.5 + _bias

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
            if cbd_vf_candidate.any():
                warnings.append(
                    f"CBD GRF: failed to produce percolating network after "
                    f"{self.MAX_CBD_RETRIES} attempts. Using last realization."
                )
            else:
                warnings.append(
                    f"CBD GRF: all {self.MAX_CBD_RETRIES} attempts produced zero "
                    f"volume (V_current <= 0.0 after smoothing). "
                    f"CBD field is empty. Check base_region and GRF sigma."
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

        dist_from_carbon = distance_transform_edt(~carbon_mask)  # 0 at carbon surface
        near_carbon_pore = pore_mask & (dist_from_carbon <= 1.5)

        if not near_carbon_pore.any():
            return np.zeros((nx, ny, nz), dtype=np.float32), [
                "No near-carbon pore region found — binder cannot be placed. "
                "Check carbon packing and porosity target."
            ]

        dist = getattr(sim, "binder_distribution", "necks")

        if dist == "patchy":
            # Binder blobs restricted to near-carbon pore space only.
            # Previously used carbon_mask | (dist <= 1.5) which included carbon interior.
            field = gaussian_filter(
                rng.normal(size=(nx, ny, nz)).astype(np.float32),
                sigma=1.0,
            )
            field[~near_carbon_pore] = -np.inf
            flat = field.ravel()
            idx = flat.argsort()[::-1]
            n_target = min(
                int(np.ceil(V_target / V_voxel)), int(near_carbon_pore.sum())
            )
            vf = np.zeros_like(flat, dtype=np.float32)
            vf[idx[:n_target]] = 1.0
            binder_vf = vf.reshape(nx, ny, nz)

        elif dist == "uniform":
            # Binder distributed uniformly over all near-carbon pore voxels.
            # Previously incorrectly set binder_vf[carbon_mask] = 1.0 (inside carbon).
            binder_vf[near_carbon_pore] = 1.0

        else:
            _KERNEL_XY = self._extracted_from__make_binder_vf_57(0, 1, 2, 1)
            _KERNEL_XY[1, 0, 1] = 1  # -Y
            _KERNEL_XY[1, 2, 1] = 1  # +Y

            _KERNEL_Z = self._extracted_from__make_binder_vf_57(1, 0, 1, 2)
            carbon_int = carbon_mask.astype(np.int16)
            carbon_neighbors_xy = convolve(carbon_int, _KERNEL_XY, mode="wrap")
            carbon_neighbors_z = convolve(
                carbon_int, _KERNEL_Z, mode="constant", cval=0
            )
            # Total carbon neighbors excluding self
            carbon_neighbors = (
                carbon_neighbors_xy + carbon_neighbors_z - carbon_mask.astype(np.int16)
            )
            contact_mask = carbon_mask & (carbon_neighbors >= 2)

            if not contact_mask.any():
                return np.zeros_like(binder_vf), [
                    "No carbon contacts detected — binder 'necks' cannot be formed."
                ]

            # Seed at contact voxels, spread into pore with Gaussian kernel
            binder_vf[contact_mask] = 1.0
            t_nm = getattr(sim, "binder_film_thickness_nm", 15.0)
            sigma_vox = max(0.5, t_nm / domain.voxel_size_nm)
            binder_vf = gaussian_filter(binder_vf, sigma=sigma_vox)

            # Clip to near-carbon pore only — removes any spread back into carbon
            binder_vf[~near_carbon_pore] = 0.0

        # Normalize to volume target
        warnings: list[str] = []
        V_current = float(binder_vf.sum() * V_voxel)
        if V_current <= 0.0:
            warnings.append(
                "Binder vf is zero after placement — falling back to uniform "
                "distribution over near-carbon pore region."
            )
            binder_vf[near_carbon_pore] = 1.0
            V_current = float(binder_vf.sum() * V_voxel)

        scale = V_target / V_current
        binder_vf *= scale
        binder_vf = np.clip(binder_vf, 0.0, 1.0)

        return binder_vf, warnings

    def _extracted_from__make_binder_vf_57(self, arg0, arg1, arg2, arg3):
        # Contact voxels = carbon voxels with ≥2 carbon neighbors
        result = np.zeros((3, 3, 3), dtype=np.int16)
        result[arg0, 1, arg1] = 1
        result[arg2, 1, arg3] = 1
        return result


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def fill_cbd_binder(
    comp: CompositionState,
    domain: DomainGeometry,
    sim: ResolvedGeneration,
    carbon_label: np.ndarray,
    si_result: SiMapResult,
    rng: np.random.Generator,
) -> CBDBinderResult:
    """
    Canonical pipeline entry for Step 4.

    Args:
      comp         : CompositionState (Step 0)
      domain       : DomainGeometry (Step 1)
      sim          : ResolvedGeneration
      carbon_label : uint8 label map (Step 2)
      si_result    : SiMapResult (Step 3)
      rng          : seeded Generator

    Returns:
      CBDBinderResult with vf maps + diagnostics.
    """
    mapper = CBDBinderMapper(comp, domain, sim)
    return mapper.fill(carbon_label=carbon_label, si_result=si_result, rng=rng)
