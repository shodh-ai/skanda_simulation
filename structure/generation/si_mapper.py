"""
Step 3 — Si Volume-Fraction Mapper

Generates si_vf[nx, ny, nz]: a float32 field where each voxel value
represents the local Si volume fraction (0.0 to 1.0).

Si particles are sub-voxel at standard resolution (d50=100nm << 390nm/voxel),
so individual Si spheres cannot be resolved. Instead, Si is represented as a
continuous volume-fraction field — physically correct because the generator
downstream treats this as a probability density for lithium storage capacity.

Three distribution modes (from config):
  embedded        : Si embedded homogeneously within carbon particle volume
  surface_anchored: Si concentrated in a band at carbon particle surfaces
  core_shell      : Si cores with carbon shell grown around them

Void space around Si (expansion buffer) and coating shells are both
stored as sub-voxel fractional modifiers — not as hard geometry — because
both are thinner than one voxel at standard resolution.

Normalization:
  After distribution, si_vf is globally scaled so that:
    sum(si_vf) × V_voxel_nm3 = V_Si_target_nm3
  This guarantees mass/mole conservation regardless of distribution mode.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
    gaussian_filter,
)

from structure.schema.resolved import ResolvedSimulation

from .composition import CompositionState
from .domain import DomainGeometry
from .carbon_packer import OblateSpheroid, PackingResult
from ..phases import PHASE_GRAPHITE


# ---------------------------------------------------------------------------
# Distribution mode
# ---------------------------------------------------------------------------


class SiDistribution(str, Enum):
    EMBEDDED = "embedded"
    SURFACE_ANCHORED = "surface_anchored"
    CORE_SHELL = "core_shell"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class SiMapResult:
    """
    Output of SiVfMapper.map().

    si_vf      : float32 (nx, ny, nz) — local Si volume fraction per voxel
    coating_vf : float32 (nx, ny, nz) — local coating volume fraction per voxel
                 (zero if si_coating_enabled=False)
    void_mask  : bool    (nx, ny, nz) — True where void space is reserved
                 around Si particles (expansion buffer)
    V_si_actual: float   — actual Si volume placed (nm³), should match V_Si_target
    V_si_target: float   — target Si volume (nm³) from CompositionState
    mass_error_pct: float — |actual - target| / target × 100 (should be < 0.1%)
    distribution:  str   — which mode was used
    warnings:      list[str]
    """

    si_vf: np.ndarray
    coating_vf: np.ndarray
    void_mask: np.ndarray
    V_si_actual_nm3: float
    V_si_target_nm3: float
    mass_error_pct: float
    distribution: str
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  SI VF MAP",
            "=" * 62,
            f"  Distribution      : {self.distribution}",
            f"  V_Si target       : {self.V_si_target_nm3:.4e} nm³",
            f"  V_Si actual       : {self.V_si_actual_nm3:.4e} nm³",
            f"  Mass error        : {self.mass_error_pct:.4f}%",
            f"  si_vf  max        : {self.si_vf.max():.4f}",
            (
                f"  si_vf  mean(>0)   : {self.si_vf[self.si_vf>0].mean():.4f}"
                if self.si_vf.any()
                else "  si_vf  mean(>0)   : N/A"
            ),
            f"  Voxels with Si>0  : {(self.si_vf > 0).sum():,}",
            f"  Coating voxels>0  : {(self.coating_vf > 0).sum():,}",
            f"  Void voxels       : {self.void_mask.sum():,}",
        ]
        if self.warnings:
            lines += ["", f"  ⚠  {len(self.warnings)} WARNING(s):"]
            lines += [f"     [{i+1}] {w}" for i, w in enumerate(self.warnings)]
        lines.append("=" * 62)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SiVfMapper
# ---------------------------------------------------------------------------


class SiVfMapper:
    """
    Generates the Si volume-fraction map from placed carbon particles.

    Usage:
        mapper = SiVfMapper(comp, domain, sim)
        result = mapper.map(carbon_label_map, packing_result, rng)
    """

    def __init__(
        self,
        comp: CompositionState,
        domain: DomainGeometry,
        sim: ResolvedSimulation,
    ) -> None:
        self.comp = comp
        self.domain = domain
        self.sim = sim

    # ── Public entry point ────────────────────────────────────────────────

    def map(
        self,
        carbon_label: np.ndarray,  # uint8 (nx, ny, nz) from Step 2
        packing: PackingResult,  # particle list from Step 2
        rng: np.random.Generator,
    ) -> SiMapResult:

        sim = self.sim
        comp = self.comp
        dist = SiDistribution(sim.silicon.distribution)

        if dist == SiDistribution.EMBEDDED:
            si_vf = self._embedded(carbon_label, packing, rng)
        elif dist == SiDistribution.SURFACE_ANCHORED:
            si_vf = self._surface_anchored(carbon_label, rng)
        elif dist == SiDistribution.CORE_SHELL:
            si_vf = self._core_shell(carbon_label, packing, rng)
        else:
            raise ValueError(f"Unknown si_distribution: {dist}")

        # ── Void space (expansion buffer) ─────────────────────────────────
        void_mask = self._build_void_mask(si_vf, carbon_label)
        if sim.silicon.void_enabled:
            # Suppress Si vf in void zones — void volume fraction reduces
            # local Si loading to make room for expansion during lithiation
            si_vf[void_mask] *= 1.0 - sim.silicon.void_fraction

        # ── Coating shell ─────────────────────────────────────────────────
        coating_vf = self._build_coating_vf(si_vf, carbon_label)

        # ── Normalize si_vf to conserve Si mass ───────────────────────────
        si_vf, V_actual, V_target, err_pct, warns = self._normalize(si_vf)

        return SiMapResult(
            si_vf=si_vf.astype(np.float32),
            coating_vf=coating_vf.astype(np.float32),
            void_mask=void_mask,
            V_si_actual_nm3=V_actual,
            V_si_target_nm3=V_target,
            mass_error_pct=err_pct,
            distribution=dist.value,
            warnings=warns,
        )

    # ── Distribution modes ────────────────────────────────────────────────

    def _embedded(
        self,
        carbon_label: np.ndarray,
        packing: PackingResult,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Si embedded homogeneously within each carbon particle's volume.

        For each carbon particle:
          1. Identify interior voxels via the analytical spheroid test
          2. Assign si_vf_local = target_inside × N(1, uniformity_cv²)
             (spatial noise models inhomogeneous Si mixing within the flake)

        target_inside = vf_Si / vf_C
          = fraction of each carbon voxel volume occupied by Si
        """
        comp = self.comp
        domain = self.domain
        sim = self.sim
        nx, ny, nz = domain.nx, domain.ny, domain.nz
        vs = domain.voxel_size_nm
        cr = comp.compression_ratio

        target_inside = comp.vf_si / comp.vf_carbon
        spatial_cv = sim.silicon.embedding_uniformity_cv

        si_vf = np.zeros((nx, ny, nz), dtype=np.float64)

        # Voxel center coordinates (final / post-calender domain)
        xs = (np.arange(nx) + 0.5) * vs
        ys = (np.arange(ny) + 0.5) * vs
        zs = (np.arange(nz) + 0.5) * vs
        Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")

        for p in packing.particles:
            inside = _voxels_inside_spheroid(
                Xg, Yg, Zg, p, domain.Lx_nm, domain.Ly_nm, cr
            )
            if not inside.any():
                continue

            # Spatially varying Si loading within this particle
            noise = rng.normal(1.0, spatial_cv, size=inside.sum())
            noise = np.clip(noise, 0.0, None)
            si_vf[inside] += target_inside * noise

        # Smooth slightly to avoid hard per-particle edges
        si_vf = gaussian_filter(si_vf, sigma=0.5)
        return si_vf

    def _surface_anchored(
        self,
        carbon_label: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Si concentrated at carbon particle surfaces.

        Method:
          1. Build binary carbon mask from label map
          2. Compute distance transform from carbon surface
             (distance = 0 at surface, increases inward and outward)
          3. Si vf = Gaussian weight centered at surface (sigma ≈ 1 voxel)
          4. Zero out Si that is deep inside carbon (> 2 voxels from surface)
             and far outside (> 2 voxels from surface externally)

        Physically: Si nanoparticles anchored to graphite flake surfaces —
        common in graphite-Si composite electrode architectures.
        """
        comp = self.comp
        domain = self.domain
        nx, ny, nz = domain.nx, domain.ny, domain.nz

        carbon_mask = carbon_label == PHASE_GRAPHITE

        # Distance from the nearest carbon surface voxel (in voxels)
        # distance_transform_edt gives distance from nearest True pixel
        # We want distance from SURFACE, not from interior/exterior separately
        dist_from_interior = distance_transform_edt(carbon_mask)  # 0 outside C
        dist_from_exterior = distance_transform_edt(~carbon_mask)  # 0 inside C

        # Surface distance: min of (dist_into_C, dist_out_of_C)
        surface_dist = np.minimum(dist_from_interior, dist_from_exterior)

        sigma_vox = max(1.0, self.sim.silicon.r_nm / domain.voxel_size_nm)

        si_vf = np.exp(-0.5 * (surface_dist / sigma_vox) ** 2)

        # Only allow Si in a band: [-2 vox outside C, +1 vox inside C]
        band = (dist_from_exterior <= 2.0) | (dist_from_interior <= 1.0)
        si_vf[~band] = 0.0

        # Must not overlap with pure carbon interior (keep thin shell only)
        si_vf[dist_from_interior > 1.5] = 0.0

        # Add small spatial noise for realism
        noise = rng.normal(1.0, 0.08, size=si_vf.shape)
        si_vf *= np.clip(noise, 0.0, None)

        return si_vf

    def _core_shell(
        self,
        carbon_label: np.ndarray,
        packing: PackingResult,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Si cores with carbon shell around them.

        In this architecture the RSA placed carbon flakes already define the
        outer shell geometry. We identify the inner region of each carbon
        particle (distance from surface > shell_thickness_vox) and assign
        that volume as the Si core.

        shell_thickness_vox is derived from the config coating thickness
        scaled to voxels. If sub-voxel, at least 1 voxel is reserved.

        Physical interpretation: pre-formed Si@C core-shell particles
        where carbon encapsulates a Si core to buffer expansion.
        """
        comp = self.comp
        domain = self.domain
        sim = self.sim
        nx, ny, nz = domain.nx, domain.ny, domain.nz

        carbon_mask = carbon_label == PHASE_GRAPHITE

        # Distance from carbon surface (inward)
        dist_inward = distance_transform_edt(carbon_mask)  # voxels into C interior

        # Shell thickness: use coating thickness if enabled, else 20% of c-axis
        if sim.silicon.coating_enabled:
            shell_t_nm = sim.silicon.coating_thickness_nm
        else:
            shell_t_nm = comp.carbon_c_nm * 0.20  # 20% of flake half-thickness

        shell_t_vox = max(1.0, shell_t_nm / domain.voxel_size_nm)

        # Si core = carbon interior deeper than shell_thickness
        si_core_mask = dist_inward > shell_t_vox

        si_vf = si_core_mask.astype(np.float64)

        # Smooth core boundary slightly
        si_vf = gaussian_filter(si_vf, sigma=0.3)

        # Add spatial noise
        noise = rng.normal(1.0, 0.05, size=si_vf.shape)
        si_vf *= np.clip(noise, 0.0, None)
        si_vf[~carbon_mask] = 0.0  # Si only inside carbon

        return si_vf

    # ── Void mask ─────────────────────────────────────────────────────────

    def _build_void_mask(
        self,
        si_vf: np.ndarray,
        carbon_label: np.ndarray,
    ) -> np.ndarray:
        """
        Build a boolean mask of void zones around Si particles.

        Void = expansion buffer space intentionally left around Si to
        accommodate ~280% volume expansion during full lithiation.
        In a real electrode this appears as a gap between Si and the
        carbon matrix.

        At sub-voxel resolution, void is modeled as a suppression of
        the Si vf at the periphery of Si-containing voxels. The void
        mask marks which voxels have their Si loading reduced.

        Method: dilate the Si support region by 1 voxel, then subtract
        the original support to get a 1-voxel-wide periphery band.
        """
        si_support = si_vf > 0.01  # voxels with meaningful Si presence
        struct = generate_binary_structure(3, 1)  # 6-connectivity
        dilated = binary_dilation(si_support, structure=struct, iterations=1)
        void_band = dilated & ~si_support
        return void_band

    # ── Coating vf ────────────────────────────────────────────────────────

    def _build_coating_vf(
        self,
        si_vf: np.ndarray,
        carbon_label: np.ndarray,
    ) -> np.ndarray:
        """
        Build a sub-voxel coating volume-fraction map.

        The coating (carbon or SiOx) wraps each Si particle.
        At sub-voxel resolution, we cannot draw an explicit shell.
        Instead, each Si-containing voxel gets a coating_vf proportional
        to the Si surface area in that voxel:

            coating_vf = si_vf × (coating_thickness_nm / r_Si) × 3
                       = si_vf × (3t/r)

        This is the thin-shell volume ratio for a sphere:
            V_shell / V_sphere ≈ 3t/r  for t << r

        Returns zero array if si_coating_enabled=False.
        """
        sim = self.sim
        if not sim.silicon.coating_enabled:
            return np.zeros_like(si_vf)

        t = sim.silicon.coating_thickness_nm
        r = self.comp.si_r_nm

        thin_shell_ratio = 3.0 * t / r  # V_shell / V_sphere
        coating_vf = si_vf * thin_shell_ratio

        # Coating cannot exceed 1.0 (physical cap)
        coating_vf = np.clip(coating_vf, 0.0, 1.0)
        return coating_vf

    # ── Normalization ─────────────────────────────────────────────────────

    def _normalize(
        self,
        si_vf: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float, list[str]]:
        """
        Scale si_vf globally so that:
            sum(si_vf) × V_voxel_nm3 == V_Si_target_nm3

        Returns (scaled_si_vf, V_actual, V_target, error_pct, warnings).
        """
        comp = self.comp
        domain = self.domain
        V_voxel = domain.voxel_size_nm**3
        V_target = comp.V_si_nm3

        V_current = si_vf.sum() * V_voxel
        warns = []

        if V_current < 1e-10:
            warns.append(
                "[CRITICAL] si_vf is all zero before normalization. "
                "No Si volume was placed. Check distribution mode and carbon packing."
            )
            return si_vf, 0.0, V_target, 100.0, warns

        scale = V_target / V_current
        si_vf_out = np.clip(si_vf * scale, 0.0, 1.0)

        # Check if clipping distorted the normalization
        V_actual = si_vf_out.sum() * V_voxel
        err_pct = abs(V_actual - V_target) / V_target * 100.0

        if err_pct > 1.0:
            warns.append(
                f"Si mass error after normalization = {err_pct:.2f}% (>1%). "
                f"Some si_vf values were clipped at 1.0. "
                f"Si loading is too concentrated. Consider increasing voxel_resolution."
            )
        elif err_pct > 0.1:
            warns.append(f"Si mass error = {err_pct:.3f}% (minor, within tolerance).")

        return si_vf_out, V_actual, V_target, err_pct, warns


# ---------------------------------------------------------------------------
# Geometry helper (shared with voxelizer)
# ---------------------------------------------------------------------------


def _voxels_inside_spheroid(
    Xg: np.ndarray,
    Yg: np.ndarray,
    Zg: np.ndarray,
    p: OblateSpheroid,
    Lx: float,
    Ly: float,
    compression_ratio: float,
) -> np.ndarray:
    """
    Return boolean mask of voxels whose centers fall inside oblate spheroid p.
    Z is compressed: z_final = z_pre × compression_ratio.
    Periodic boundary applied in X, Y.
    """
    cx = p.center[0]
    cy = p.center[1]
    cz = p.center[2] * compression_ratio

    dx = Xg - cx
    dy = Yg - cy
    dz = Zg - cz

    # Minimum image convention for X, Y
    dx = dx - Lx * np.round(dx / Lx)
    dy = dy - Ly * np.round(dy / Ly)

    # Rotate into body frame
    RT = p.R.T
    dx_b = RT[0, 0] * dx + RT[0, 1] * dy + RT[0, 2] * dz
    dy_b = RT[1, 0] * dx + RT[1, 1] * dy + RT[1, 2] * dz
    dz_b = RT[2, 0] * dx + RT[2, 1] * dy + RT[2, 2] * dz

    return (dx_b / p.a) ** 2 + (dy_b / p.a) ** 2 + (dz_b / p.c) ** 2 <= 1.0


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def map_si_distribution(
    comp: CompositionState,
    domain: DomainGeometry,
    sim: ResolvedSimulation,
    carbon_label: np.ndarray,
    packing: PackingResult,
    rng: np.random.Generator,
) -> SiMapResult:
    """
    Canonical pipeline entry point for Step 3.

    Args:
        comp         : CompositionState from Step 0
        domain       : DomainGeometry from Step 1
        sim          : ResolvedSimulation
        carbon_label : uint8 label volume from Step 2 voxelizer
        packing      : PackingResult from Step 2 (particle list)
        rng          : seeded Generator (same generator, continues from Step 2)

    Returns:
        SiMapResult with si_vf, coating_vf, void_mask, diagnostics
    """
    mapper = SiVfMapper(comp=comp, domain=domain, sim=sim)
    return mapper.map(carbon_label=carbon_label, packing=packing, rng=rng)
