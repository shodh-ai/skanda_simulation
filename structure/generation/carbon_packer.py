"""
Step 2 — Carbon Scaffold Packer

Places N_carbon oblate spheroids in the pre-calendering domain using
Random Sequential Addition (RSA) with:

  - Perram-Wertheim exact overlap test
  - Uniform 3D spatial grid for O(N) neighbor lookup
  - Periodic boundaries in X, Y
  - Hard wall in Z (particles clipped to Z ∈ [c, Lz_pre - c])
  - Log-normal PSD per particle
  - von Mises-Fisher orientation toward Z (c-axis alignment)
  - Jamming escape: if rejected > MAX_REJECT times, inflate placed
    particles uniformly in XY to hit target volume fraction,
    then terminate gracefully

Physical model:
  Particles are oblate spheroids (flattened along Z for graphite flakes).
  Basal semi-axis a = d50 / 2, thickness semi-axis c = a / aspect_ratio.
  Rotation matrix R encodes the full 3D orientation.
  Shape matrix: M = R @ diag(1/a², 1/a², 1/c²) @ R.T
  Overlap test: Perram & Wertheim (1985) — maximize F(s) over s ∈ [0,1];
                overlap iff F_max < 1.
"""

from __future__ import annotations

import math
import numpy as np
from structure.schema import ResolvedSimulation
from structure.data import (
    CompositionState,
    DomainGeometry,
    OblateSpheroid,
    PackingResult,
)
from structure.utils.carbon_packer import (
    _make_spheroid,
    _sample_size,
    _sample_rotation,
    _od_to_kappa,
    _spheroids_overlap,
)


# ---------------------------------------------------------------------------
# CarbonScaffoldPacker
# ---------------------------------------------------------------------------
class CarbonScaffoldPacker:
    """
    Places N_carbon oblate spheroids in the pre-calendering domain via RSA.

    Usage:
        packer = CarbonScaffoldPacker(comp, domain, sim)
        result = packer.pack(rng)

    The result.particles list is in pre-calendering (expanded Z) coordinates.
    CalenderingTransform (Step 5) compresses them into the final domain.
    """

    # RSA control
    MAX_REJECT_PER_PARTICLE: int = 2_000  # attempts before declaring jamming
    MAX_TOTAL_ATTEMPTS: int = 500_000  # hard stop for the entire run
    MAX_INFLATION_STEPS: int = 50  # iterations in the inflation loop
    INFLATION_STEP_SIZE: float = 0.002  # per-step XY scale increment

    def __init__(
        self,
        comp: CompositionState,
        domain: DomainGeometry,
        orientation_degree: float,
    ) -> None:
        self.comp = comp
        self.domain = domain
        self.orientation_degree = orientation_degree

        # Spatial grid cell size = 2 × max possible basal radius
        # max(a) ≈ d50 * (1 + 3*cv) as a conservative 3σ upper bound
        max_a = (comp.carbon_d50_nm / 2.0) * (1.0 + 3.0 * comp.carbon_size_cv)
        self._cell_size = 2.0 * max_a
        self._grid: dict[tuple[int, int, int], list[int]] = {}
        self._particles: list[OblateSpheroid] = []

    # ── Public entry point ────────────────────────────────────────────────

    def pack(self, rng: np.random.Generator) -> PackingResult:
        """Run RSA packing. Returns PackingResult."""
        comp = self.comp
        domain = self.domain
        N = comp.N_carbon
        kappa = _od_to_kappa(self.orientation_degree)

        total_attempts = 0
        warns: list[str] = []
        inflated = False
        inflation_factor = 1.0

        for i in range(N):
            placed = False
            rejects = 0

            while not placed:
                if total_attempts >= self.MAX_TOTAL_ATTEMPTS:
                    warns.append(
                        f"[CRITICAL] Hit MAX_TOTAL_ATTEMPTS={self.MAX_TOTAL_ATTEMPTS:,} "
                        f"after placing {i}/{N} particles."
                    )
                    break

                # ── Propose particle ──────────────────────────────────────
                center = domain.random_point_pre(rng)
                a, c = _sample_size(
                    comp.carbon_d50_nm,
                    comp.carbon_size_cv,
                    rng,
                    comp.carbon_aspect_ratio,
                )
                R = _sample_rotation(kappa, rng)

                # Enforce hard wall in Z: particle must fit fully inside [0, Lz_pre]
                center[2] = np.clip(center[2], c, domain.Lz_pre_nm - c)

                p = _make_spheroid(center, a, c, R)
                total_attempts += 1

                # ── Check overlap ────────────────────────────────────────
                if not self._overlaps_any(p):
                    self._place(p)
                    placed = True
                else:
                    rejects += 1

                # ── Jamming escape ───────────────────────────────────────
                if rejects >= self.MAX_REJECT_PER_PARTICLE:
                    warns.append(
                        f"Jamming at particle {i+1}/{N} after "
                        f"{rejects} rejects — triggering inflation escape."
                    )
                    inflation_factor = self._inflate_to_target(
                        target_phi=comp.phi_carbon_pre,
                        domain=domain,
                    )
                    inflated = True

                    n_overlap, n_checked = self._count_overlapping_pairs()
                    if n_overlap > 0:
                        overlap_pct = 100.0 * n_overlap / max(n_checked, 1)
                        warns.append(
                            f"Post-inflation overlap (jamming escape at particle {i+1}/{N}): "
                            f"{n_overlap}/{n_checked} particle pairs intersect "
                            f"({overlap_pct:.1f}%). "
                            f"Packing is unphysical — microstructure has forced overlaps. "
                            f"To avoid: (a) reduce target_porosity, "
                            f"(b) reduce carbon_particle_d50_nm, "
                            f"(c) increase coating_thickness_um."
                        )
                    break  # stop trying to place this particle

            if total_attempts >= self.MAX_TOTAL_ATTEMPTS:
                break

        # ── Final diagnostics ────────────────────────────────────────────
        N_placed = len(self._particles)
        phi_achieved = self._current_phi(domain)

        # If we still missed target volume without inflation, inflate now
        if not inflated and abs(phi_achieved - comp.phi_carbon_pre) > 0.01:
            inflation_factor = self._inflate_to_target(
                target_phi=comp.phi_carbon_pre,
                domain=domain,
            )
            inflated = True
            phi_achieved = self._current_phi(domain)
            warns.append(
                f"Post-RSA inflation applied to close "
                f"φ gap (inflation_factor={inflation_factor:.4f})"
            )

            n_overlap, n_checked = self._count_overlapping_pairs()
            if n_overlap > 0:
                overlap_pct = 100.0 * n_overlap / max(n_checked, 1)
                warns.append(
                    f"Post-inflation overlap (post-RSA gap close): "
                    f"{n_overlap}/{n_checked} particle pairs intersect "
                    f"({overlap_pct:.1f}%). "
                    f"Packing is unphysical — microstructure has forced overlaps. "
                    f"To avoid: (a) reduce target_porosity, "
                    f"(b) reduce carbon_particle_d50_nm, "
                    f"(c) increase coating_thickness_um."
                )

        final_n_overlap, final_n_checked = (
            self._count_overlapping_pairs() if inflated else (0, 0)
        )

        return PackingResult(
            particles=list(self._particles),
            N_placed=N_placed,
            N_target=N,
            phi_achieved=phi_achieved,
            phi_target=comp.phi_carbon_pre,
            inflated=inflated,
            inflation_factor=inflation_factor,
            total_attempts=total_attempts,
            n_overlapping_pairs=final_n_overlap,
            n_pairs_checked=final_n_checked,
            warnings=warns,
        )

    # ── Spatial grid ─────────────────────────────────────────────────────

    def _cell(self, center: np.ndarray) -> tuple[int, int, int]:
        cs = self._cell_size
        return (
            int(center[0] // cs),
            int(center[1] // cs),
            int(center[2] // cs),
        )

    def _neighbors(self, p: OblateSpheroid) -> list[int]:
        """
        Return indices of all particles in the 3×3×3 neighborhood of p's cell.
        Handles periodic X/Y wrapping of cell keys.
        """
        cx, cy, cz = self._cell(p.center)
        nx_cells = max(1, int(self.domain.Lx_nm // self._cell_size) + 1)
        ny_cells = max(1, int(self.domain.Ly_nm // self._cell_size) + 1)

        indices = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (
                        (cx + dx) % nx_cells,
                        (cy + dy) % ny_cells,
                        cz + dz,  # no wrapping in Z
                    )
                    indices.extend(self._grid.get(key, []))
        return indices

    def _place(self, p: OblateSpheroid) -> None:
        idx = len(self._particles)
        self._particles.append(p)
        key = self._cell(p.center)
        self._grid.setdefault(key, []).append(idx)

    # ── Overlap detection ─────────────────────────────────────────────────

    def _overlaps_any(self, p: OblateSpheroid) -> bool:
        for idx in self._neighbors(p):
            q = self._particles[idx]
            if _spheroids_overlap(p, q, self.domain):
                return True
        return False

    # ── Volume fraction ───────────────────────────────────────────────────

    def _current_phi(self, domain: DomainGeometry) -> float:
        V_particles = sum(p.volume_nm3 for p in self._particles)
        return V_particles / domain.V_pre_nm3

    # ── Jamming escape: inflate XY ────────────────────────────────────────

    def _inflate_to_target(
        self,
        target_phi: float,
        domain: DomainGeometry,
    ) -> float:
        """
        Scale all placed particles' basal semi-axis `a` (and recompute A_inv)
        in small steps until either:
          (a) target φ is reached, or
          (b) MAX_INFLATION_STEPS is exhausted.

        Physics rationale:
          After RSA jams, we have N particles with total volume V_placed < V_target.
          The volume deficit is split equally among all placed particles by scaling
          their XY basal axes. The c (thickness) axis is NOT scaled — this preserves
          the aspect ratio constraint from the config and keeps the Z extent physical.

        Volume scales as a² × c, so to reach target volume:
          sum(a_new² × c) = V_target / (4π/3)
          a_new = a_old × sqrt(V_target_per_particle / (a_old² × c_old))
          → uniform scale factor: f = (target_V_total / current_V_total)^(1/2)
            (exponent 1/2 because V ∝ a²)
        """
        if not self._particles:
            return 1.0

        V_target = target_phi * domain.V_pre_nm3
        V_current = sum(p.volume_nm3 for p in self._particles)

        if V_current <= 0.0:
            return 1.0

        # Single-shot analytical scale factor on a only
        # V_new = f² × a² × c × (4π/3)  →  f = sqrt(V_target / V_current)
        f = math.sqrt(V_target / V_current)
        # Safety cap: allow up to 2× basal axis growth (4× volume) to cover
        # any realistic RSA-to-target gap while preventing runaway.
        f = min(f, 2.0)

        for p in self._particles:
            p.invalidate_shape_matrix()
            p.a = p.a * f
            p.recompute_shape_matrix()

        return f

    def _count_overlapping_pairs(self) -> tuple[int, int]:
        """
        Count intersecting particle pairs after inflation using the spatial grid.

        Uses the same grid-accelerated neighbor lookup as the RSA loop so
        complexity is O(N) average rather than O(N²). Pairs are counted once
        (by enforcing i < j) to avoid double-counting.

        Returns:
            (n_overlapping_pairs, n_total_pairs_checked)
        """
        particles = self._particles
        n_overlapping = 0
        checked: set[tuple[int, int]] = set()

        for i, p in enumerate(particles):
            for j in self._neighbors(p):
                if j == i:
                    continue
                key = (min(i, j), max(i, j))
                if key in checked:
                    continue
                checked.add(key)
                if _spheroids_overlap(p, particles[j], self.domain):
                    n_overlapping += 1

        return n_overlapping, len(checked)


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------
def pack_carbon_scaffold(
    comp: CompositionState,
    domain: DomainGeometry,
    sim: ResolvedSimulation,
    rng: np.random.Generator,
) -> PackingResult:
    """
    Convenience wrapper. The canonical entry point used by the pipeline.

    Args:
        comp   : CompositionState from Step 0
        domain : DomainGeometry from Step 1
        sim    : ResolvedSimulation (for orientation params)
        rng    : seeded Generator (seed comes from run_config)

    Returns:
        PackingResult with placed particles + diagnostics
    """
    packer = CarbonScaffoldPacker(
        comp=comp,
        domain=domain,
        orientation_degree=sim.carbon.orientation_degree,
    )
    return packer.pack(rng)
