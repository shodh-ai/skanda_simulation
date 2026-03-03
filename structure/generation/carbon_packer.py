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
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from structure.schema.resolved import ResolvedSimulation

from .composition import CompositionState
from .domain import DomainGeometry
from ..phases import PHASE_GRAPHITE


# ---------------------------------------------------------------------------
# OblateSpheroid dataclass
# ---------------------------------------------------------------------------


@dataclass
class OblateSpheroid:
    """
    One graphite flake particle in nm coordinates.

    Attributes:
        center  : [x, y, z] position in nm (pre-calendering coordinates)
        a       : basal semi-axis (nm) — the two equal long axes
        c       : thickness semi-axis (nm) = a / aspect_ratio
        R       : 3×3 rotation matrix; columns are the body-frame axes
                  in the lab frame. The c-axis is R[:, 2].
        A_inv   : cached inverse shape matrix = inv(R @ diag(a⁻²,a⁻²,c⁻²) @ R.T)
                  pre-computed once to avoid repeated inversion during overlap checks
        phase_id: always PHASE_GRAPHITE = 1
    """

    center: np.ndarray
    a: float
    c: float
    R: np.ndarray
    A_inv: np.ndarray = field(repr=False)
    phase_id: int = PHASE_GRAPHITE

    @property
    def volume_nm3(self) -> float:
        return (4.0 / 3.0) * math.pi * self.a**2 * self.c

    @property
    def aspect_ratio(self) -> float:
        return self.a / self.c

    @property
    def c_axis(self) -> np.ndarray:
        """Unit vector along the c-axis (thickness direction) in lab frame."""
        return self.R[:, 2]

    @property
    def bounding_radius(self) -> float:
        """Radius of the smallest enclosing sphere. Used for fast rejection."""
        return self.a  # a ≥ c always for oblate


# ---------------------------------------------------------------------------
# PackingResult
# ---------------------------------------------------------------------------


@dataclass
class PackingResult:
    """
    Output of CarbonScaffoldPacker.pack().
    Contains the placed particles and packing diagnostics.
    """

    particles: list[OblateSpheroid]
    N_placed: int
    N_target: int
    phi_achieved: float  # actual solid volume fraction in pre-calender box
    phi_target: float  # target from CompositionState
    inflated: bool  # True if jamming escape was triggered
    inflation_factor: float  # XY scale applied during escape (1.0 if no inflation)
    total_attempts: int
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "INFLATED (jamming escape)" if self.inflated else "RSA converged"
        lines = [
            "=" * 62,
            "  CARBON SCAFFOLD PACKING",
            "=" * 62,
            f"  Status            : {status}",
            f"  Particles placed  : {self.N_placed} / {self.N_target}",
            f"  φ_solid achieved  : {self.phi_achieved:.4f}"
            f"  (target {self.phi_target:.4f},"
            f"  Δ = {abs(self.phi_achieved - self.phi_target):.4f})",
            f"  Total RSA attempts: {self.total_attempts:,}",
        ]
        if self.inflated:
            lines.append(
                f"  Inflation factor  : {self.inflation_factor:.4f}"
                f"  (XY basal axes scaled by this factor)"
            )
        if self.warnings:
            lines += ["", f"  ⚠  {len(self.warnings)} WARNING(s):"]
            lines += [f"     [{i+1}] {w}" for i, w in enumerate(self.warnings)]
        lines.append("=" * 62)
        return "\n".join(lines)


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
        orientation_enhancement: float = 0.0,
    ) -> None:
        self.comp = comp
        self.domain = domain
        self.orientation_degree = orientation_degree
        self.orientation_enhancement = orientation_enhancement

        # Spatial grid cell size = 2 × max possible basal radius
        # max(a) ≈ d50 * (1 + 3*cv) as a conservative 3σ upper bound
        max_a = (comp.carbon_d50_nm / 2.0) * (1.0 + 3.0 * 0.25)
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
                        target_phi=comp.phi_solid_pre,
                        domain=domain,
                    )
                    inflated = True
                    break  # stop trying to place this particle

            if total_attempts >= self.MAX_TOTAL_ATTEMPTS:
                break

        # ── Final diagnostics ────────────────────────────────────────────
        N_placed = len(self._particles)
        phi_achieved = self._current_phi(domain)

        # If we still missed target volume without inflation, inflate now
        if not inflated and abs(phi_achieved - comp.phi_solid_pre) > 0.01:
            inflation_factor = self._inflate_to_target(
                target_phi=comp.phi_solid_pre,
                domain=domain,
            )
            inflated = True
            phi_achieved = self._current_phi(domain)
            warns.append(
                f"Post-RSA inflation applied to close "
                f"φ gap (inflation_factor={inflation_factor:.4f})"
            )

        return PackingResult(
            particles=list(self._particles),
            N_placed=N_placed,
            N_target=N,
            phi_achieved=phi_achieved,
            phi_target=comp.phi_solid_pre,
            inflated=inflated,
            inflation_factor=inflation_factor,
            total_attempts=total_attempts,
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
        f = min(f, 1.0 + self.MAX_INFLATION_STEPS * self.INFLATION_STEP_SIZE)

        for p in self._particles:
            p.a = p.a * f
            # Recompute A_inv with new a
            D = np.diag([1.0 / p.a**2, 1.0 / p.a**2, 1.0 / p.c**2])
            p.A_inv = np.linalg.inv(p.R @ D @ p.R.T)

        return f


# ---------------------------------------------------------------------------
# Geometry helpers (module-level, no class state)
# ---------------------------------------------------------------------------


def _make_spheroid(
    center: np.ndarray,
    a: float,
    c: float,
    R: np.ndarray,
) -> OblateSpheroid:
    """Construct an OblateSpheroid with pre-computed A_inv."""
    D = np.diag([1.0 / a**2, 1.0 / a**2, 1.0 / c**2])
    A_inv = np.linalg.inv(R @ D @ R.T)
    return OblateSpheroid(center=center, a=a, c=c, R=R, A_inv=A_inv)


def _spheroids_overlap(
    p1: OblateSpheroid,
    p2: OblateSpheroid,
    domain: DomainGeometry,
) -> bool:
    """
    Test overlap between two oblate spheroids using the
    Perram-Wertheim (1985) separating hyperplane criterion.

    Two spheroids do NOT overlap iff there exists s ∈ [0,1] such that:
        F(s) = s(1-s) d·M(s)⁻¹·d > 1
    where M(s) = (1-s)A⁻¹ + s B⁻¹  and  d = center_2 - center_1.

    Step 1 (fast reject): bounding sphere check.
        If |d| > a1 + a2, cannot possibly overlap.
    Step 2 (exact test): maximize F(s) over [0,1].
        If max F < 1 → overlap. Else → no overlap.

    Periodic boundary in X, Y handled via minimum-image displacement.
    """
    # Fast bounding sphere rejection (uses MIC displacement for periodicity)
    d = domain.min_image_vector(p1.center, p2.center)
    dist = float(np.linalg.norm(d))
    if dist > (p1.bounding_radius + p2.bounding_radius):
        return False

    # Exact Perram-Wertheim test
    A_inv = p1.A_inv
    B_inv = p2.A_inv

    def neg_F(s: float) -> float:
        M_inv = (1.0 - s) * A_inv + s * B_inv
        M = np.linalg.inv(M_inv)
        return -(s * (1.0 - s) * float(d @ M @ d))

    result = minimize_scalar(
        neg_F, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6}
    )
    F_max = -result.fun
    return F_max < 1.0


def _sample_size(
    d50_nm: float,
    size_cv: float,
    rng: np.random.Generator,
    aspect_ratio: float = 5.0,
) -> tuple[float, float]:
    """
    Draw one (a, c) pair from the log-normal PSD.

    The aspect ratio is fixed per the config — only the overall size
    (d50) varies between particles, not their flatness.

        d  ~ LogNormal(median=d50, cv=size_cv)
        a  = d / 2
        c  = a / aspect_ratio
    """
    sigma_ln = math.sqrt(math.log(1.0 + size_cv**2))
    mu_ln = math.log(d50_nm)
    d = float(rng.lognormal(mu_ln, sigma_ln))
    a = d / 2.0
    c = a / aspect_ratio
    return a, c


def _sample_rotation(kappa: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a 3×3 rotation matrix R such that the c-axis (R[:, 2])
    is drawn from vMF(μ=[0,0,1], κ).

    κ = 0     → isotropic (random orientation)
    κ = large → c-axis closely aligned with Z

    The two basal axes (R[:, 0], R[:, 1]) are chosen to form a
    right-handed orthonormal frame with the sampled c-axis.
    """
    c_axis = _sample_vmf_z(kappa, rng)

    # Build orthonormal frame: Gram-Schmidt with a random transverse vector
    v = rng.normal(size=3)
    v -= v.dot(c_axis) * c_axis
    norm_v = float(np.linalg.norm(v))
    if norm_v < 1e-10:
        # c_axis nearly parallel to v — use a fixed fallback
        v = np.array([1.0, 0.0, 0.0])
        v -= v.dot(c_axis) * c_axis
        norm_v = float(np.linalg.norm(v))
    e1 = v / norm_v
    e2 = np.cross(c_axis, e1)

    R = np.column_stack([e1, e2, c_axis])
    return R


def _sample_vmf_z(kappa: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample one unit vector from vMF(μ=[0,0,1], κ) using Wood (1994).

    Returns:
        Unit vector np.ndarray of shape (3,).
    """
    if kappa < 1e-6:
        # Uniform on sphere — Muller method
        u = rng.normal(size=3)
        return u / np.linalg.norm(u)

    # Wood (1994) algorithm
    dim = 3
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa**2 + (dim - 1) ** 2)) / (dim - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (dim - 1) * math.log(1.0 - x0**2)

    while True:
        z = float(rng.beta((dim - 1) / 2.0, (dim - 1) / 2.0))
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = float(rng.uniform())
        if kappa * w + (dim - 1) * math.log(max(1.0 - x0 * w, 1e-300)) - c >= math.log(
            u
        ):
            break

    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    sin_theta = math.sqrt(max(0.0, 1.0 - w**2))
    return np.array([sin_theta * math.cos(phi), sin_theta * math.sin(phi), w])


def _od_to_kappa(orientation_degree: float) -> float:
    """
    Map orientation_degree ∈ [0, 1] to vMF concentration κ ≥ 0.

    Mapping: κ = -10 × ln(1 - orientation_degree)
      od=0.0  → κ=0     (isotropic)
      od=0.6  → κ≈9.2   (strong Z preference)
      od=1.0  → κ=∞     (clamped to 1000)

    Calibrated so that mean|cos θ| matches the qualitative
    orientation_degree label (0=random, 1=perfect).
    """
    if orientation_degree <= 0.0:
        return 0.0
    if orientation_degree >= 1.0:
        return 1000.0
    return -10.0 * math.log(1.0 - orientation_degree)


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
        orientation_enhancement=sim.calendering_orientation_enhancement,
    )
    return packer.pack(rng)
