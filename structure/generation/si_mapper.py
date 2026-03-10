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

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
    gaussian_filter,
)

from structure.schema import ResolvedGeneration
from structure.schema.gen_config import SiDistribution
from structure.data import SiMapResult, CompositionState, DomainGeometry, PackingResult
from structure.phases import PHASE_GRAPHITE


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
        sim: ResolvedGeneration,
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
        if sim.silicon.void_enabled:
            void_mask = self._build_void_mask(si_vf, carbon_label)
            si_vf[void_mask] *= 1.0 - sim.silicon.void_fraction
        else:
            void_mask = np.zeros(si_vf.shape, dtype=bool)  # sentinel: not computed

        # ── Coating shell ─────────────────────────────────────────────────
        coating_vf = self._build_coating_vf(si_vf, carbon_label)

        # ── Normalize si_vf to conserve Si mass ───────────────────────────
        si_vf, V_actual, V_target, err_pct, warns = self._normalize(si_vf)

        return SiMapResult(
            si_vf=si_vf.astype(np.float32),
            coating_vf=coating_vf.astype(np.float32),
            void_mask=void_mask,
            void_enabled=sim.silicon.void_enabled,
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
        1. Compute AABB in final-domain voxel indices (periodic X/Y, clamped Z)
        2. Build sub-meshgrid of voxel centres for that AABB only
        3. Run analytical spheroid membership test on the sub-grid
        4. Assign si_vf_local = target_inside × N(1, embedding_uniformity_cv²)

        target_inside = vf_Si / vf_C
            = fraction of each carbon voxel volume occupied by Si

        AABB half-extent = ceil(p.a / vs) + 1  (p.a is the max semi-axis)
        Cost: O(N_carbon × AABB³) vs O(N_carbon × nx × ny × nz) before.
        Typical speedup: ~60× at 128³ for 12µm graphite (AABB ≈ 32³).
        """
        comp = self.comp
        domain = self.domain
        sim = self.sim
        nx, ny, nz = domain.nx, domain.ny, domain.nz
        vs = domain.voxel_size_nm
        cr = comp.compression_ratio
        Lx = domain.Lx_nm
        Ly = domain.Ly_nm

        target_inside = comp.vf_si / comp.vf_carbon
        spatial_cv = sim.silicon.embedding_uniformity_cv

        si_vf = np.zeros((nx, ny, nz), dtype=np.float64)

        for p in packing.particles:
            # ── Final-domain centre ──────────────────────────────────────────
            cx = p.center[0]
            cy = p.center[1]
            cz = p.center[2] * cr  # compressed Z in final domain

            # ── AABB half-extent in voxels ───────────────────────────────────
            # p.a is the largest semi-axis (a >= c for oblate spheroid).
            # Conservative bound: any point on the spheroid satisfies
            # |Δx|, |Δy|, |Δz| ≤ p.a in the lab frame.
            half = int(np.ceil(p.a / vs)) + 1

            # ── Integer index ranges ─────────────────────────────────────────
            ix_c = int(cx / vs)
            iy_c = int(cy / vs)
            iz_c = int(cz / vs)

            ix_arr = np.arange(ix_c - half, ix_c + half + 1)
            iy_arr = np.arange(iy_c - half, iy_c + half + 1)
            iz_arr = np.arange(iz_c - half, iz_c + half + 1)

            # Z — hard wall: clamp to valid range
            iz_arr = iz_arr[(iz_arr >= 0) & (iz_arr < nz)]
            if iz_arr.size == 0:
                continue

            # X, Y — periodic: wrap with modulo for storage indices
            ix_w = ix_arr % nx
            iy_w = iy_arr % ny

            # ── Sub-grid voxel centres (actual nm coordinates, may exceed Lx/Ly)
            # Using unmodded ix_arr/iy_arr here so minimum-image displacement
            # is computed correctly before applying modulo for storage.
            xs_sub = (ix_arr + 0.5) * vs
            ys_sub = (iy_arr + 0.5) * vs
            zs_sub = (iz_arr + 0.5) * vs

            Xs, Ys, Zs = np.meshgrid(xs_sub, ys_sub, zs_sub, indexing="ij")

            # ── Displacement from particle centre (minimum image for X, Y) ──
            dx = Xs - cx
            dx -= Lx * np.round(dx / Lx)  # minimum image — periodic X
            dy = Ys - cy
            dy -= Ly * np.round(dy / Ly)  # minimum image — periodic Y
            dz = Zs - cz  # hard wall Z — no wrapping

            # ── Particle body-frame coordinates ─────────────────────────────
            RT = p.R.T
            dx_b = RT[0, 0] * dx + RT[0, 1] * dy + RT[0, 2] * dz
            dy_b = RT[1, 0] * dx + RT[1, 1] * dy + RT[1, 2] * dz
            dz_b = RT[2, 0] * dx + RT[2, 1] * dy + RT[2, 2] * dz

            # ── Spheroid membership test ─────────────────────────────────────
            inside = (dx_b / p.a) ** 2 + (dy_b / p.a) ** 2 + (dz_b / p.c) ** 2 <= 1.0

            if not inside.any():
                continue

            # ── Spatially varying Si loading within this particle ────────────
            noise = rng.normal(1.0, spatial_cv, size=inside.sum())
            noise = np.clip(noise, 0.0, None)

            # ── Write back to global grid using wrapped storage indices ──────
            # Broadcasting to (len(ix_arr), len(iy_arr), len(iz_arr)) shape.
            ixg = ix_w[:, None, None] * np.ones(
                (1, len(iy_arr), len(iz_arr)), dtype=np.intp
            )
            iyg = iy_w[None, :, None] * np.ones(
                (len(ix_arr), 1, len(iz_arr)), dtype=np.intp
            )
            izg = iz_arr[None, None, :] * np.ones(
                (len(ix_arr), len(iy_arr), 1), dtype=np.intp
            )

            si_vf[ixg[inside], iyg[inside], izg[inside]] += target_inside * noise

        # Smooth slightly to remove hard per-particle edges
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

        # Shell thickness
        shell_t_nm = sim.silicon.core_shell_carbon_thickness_nm
        if shell_t_nm <= 0.0:
            # Fallback: 20% of carbon particle c-axis half-thickness
            shell_t_nm = comp.carbon_c_nm * 0.20
            self.domain  # (silence linter — comp is in scope)

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
        return dilated & ~si_support

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

            coating_vf = si_vf × (3t/r)

        This is the thin-shell volume ratio for a sphere:
            V_shell / V_sphere ≈ 3t/r  for t << r

        Coating is physically suppressed wherever the voxel is already
        dominated by carbon matrix (carbon_vf = 1). In embedded and
        core_shell modes Si lives inside carbon particles — the Si-C
        interface is already solid, so coating cannot occupy additional
        volume there without pushing solid_sum > 1.

        Specifically: coating_vf is zeroed wherever carbon_label == PHASE_GRAPHITE.
        In surface_anchored mode, Si sits at the carbon boundary facing pore
        space — this mask leaves most coating_vf intact there.

        Returns zero array if si_coating_enabled=False.
        """
        sim = self.sim
        if not sim.silicon.coating_enabled:
            return np.zeros_like(si_vf)

        t = sim.silicon.coating_thickness_nm
        r = self.comp.si_r_nm

        thin_shell_ratio = 3.0 * t / r  # V_shell / V_sphere
        coating_vf = si_vf * thin_shell_ratio

        # Suppress coating inside carbon voxels — Si-C interface is already
        # solid; coating here would cause carbon_vf + coating_vf > 1.
        carbon_mask = carbon_label == PHASE_GRAPHITE
        coating_vf[carbon_mask] = 0.0

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
# Public factory
# ---------------------------------------------------------------------------
def map_si_distribution(
    comp: CompositionState,
    domain: DomainGeometry,
    sim: ResolvedGeneration,
    carbon_label: np.ndarray,
    packing: PackingResult,
    rng: np.random.Generator,
) -> SiMapResult:
    """
    Canonical pipeline entry point for Step 3.

    Args:
        comp         : CompositionState from Step 0
        domain       : DomainGeometry from Step 1
        sim          : ResolvedGeneration
        carbon_label : uint8 label volume from Step 2 voxelizer
        packing      : PackingResult from Step 2 (particle list)
        rng          : seeded Generator (same generator, continues from Step 2)

    Returns:
        SiMapResult with si_vf, coating_vf, void_mask, diagnostics
    """
    mapper = SiVfMapper(comp=comp, domain=domain, sim=sim)
    return mapper.map(carbon_label=carbon_label, packing=packing, rng=rng)
