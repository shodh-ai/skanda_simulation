"""
Step 1 — Domain Geometry

Defines the physical simulation box in nm coordinates.

Two boxes exist:
  Pre-calendering : Lx × Ly × Lz_pre   — all RSA packing happens here
  Final (post)    : Lx × Ly × Lz_final — voxelization target (cubic)

Boundary conditions:
  X, Y : periodic  (electrode is laterally infinite)
  Z     : hard wall (Z=0 = current collector, Z=Lz = electrolyte interface)

Coordinate origin: bottom-left corner of the pre-calendering box.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .composition import CompositionState


# ---------------------------------------------------------------------------
# Domain dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainGeometry:
    """
    Immutable physical simulation box.

    All downstream generation steps use this as the single geometric
    reference. Never recomputed, never mutated.
    """

    # ── Physical extents (nm) ───────────────────────────────────────────────
    Lx_nm: float  # lateral X = coating_thickness_nm
    Ly_nm: float  # lateral Y = coating_thickness_nm (square cross-section)
    Lz_pre_nm: float  # Z before calendering (expanded)
    Lz_final_nm: float  # Z after calendering (= Lx = Ly → cubic output)

    # ── Voxel output grid ───────────────────────────────────────────────────
    nx: int  # = voxel_resolution
    ny: int  # = voxel_resolution
    nz: int  # = voxel_resolution
    voxel_size_nm: float  # = Lx_nm / nx  (isotropic — same in all directions)

    # ── Calendering ─────────────────────────────────────────────────────────
    compression_ratio: float  # Lz_final / Lz_pre

    # ── Derived volumes ─────────────────────────────────────────────────────

    @property
    def V_pre_nm3(self) -> float:
        """Volume of the pre-calendering box (nm³)."""
        return self.Lx_nm * self.Ly_nm * self.Lz_pre_nm

    @property
    def V_final_nm3(self) -> float:
        """Volume of the final post-calendering box (nm³)."""
        return self.Lx_nm * self.Ly_nm * self.Lz_final_nm

    @property
    def voxel_shape(self) -> tuple[int, int, int]:
        """Output voxel grid shape as (nx, ny, nz)."""
        return (self.nx, self.ny, self.nz)

    # ── Point / boundary helpers ────────────────────────────────────────────

    def is_inside_pre(self, point: np.ndarray) -> bool:
        """True if point (nm) is within the pre-calendering box."""
        x, y, z = point
        return (
            0.0 <= x <= self.Lx_nm
            and 0.0 <= y <= self.Ly_nm
            and 0.0 <= z <= self.Lz_pre_nm
        )

    def wrap_xy(self, point: np.ndarray) -> np.ndarray:
        """
        Apply periodic boundary in X and Y only.
        Z is hard-walled and is never wrapped.
        """
        p = point.copy()
        p[0] = p[0] % self.Lx_nm
        p[1] = p[1] % self.Ly_nm
        return p

    def min_image_vector(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Displacement vector p1 → p2 under minimum-image convention.

        Applies periodic wrapping in X and Y so the returned vector
        always takes the shortest path across X/Y boundaries.
        Z displacement is returned as-is (hard wall, no wrapping).

        Used by CarbonScaffoldPacker for overlap detection near X/Y edges.

        Example:
            p1 = [49900, 0, 5000], p2 = [100, 0, 5000], Lx = 50000
            Naive d_x = -49800  (wrong — crosses the periodic boundary)
            MIC   d_x =    200  (correct — short path wraps around)
        """
        d = p2 - p1
        d[0] -= self.Lx_nm * round(d[0] / self.Lx_nm)
        d[1] -= self.Ly_nm * round(d[1] / self.Ly_nm)
        return d

    def random_point_pre(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a uniformly random point inside the pre-calendering box."""
        return np.array(
            [
                rng.uniform(0.0, self.Lx_nm),
                rng.uniform(0.0, self.Ly_nm),
                rng.uniform(0.0, self.Lz_pre_nm),
            ]
        )

    # ── Voxel helpers ────────────────────────────────────────────────────────

    def to_voxel_index(self, point_nm: np.ndarray) -> tuple[int, int, int]:
        """
        Map a physical coordinate (nm) in the FINAL box to (ix, iy, iz).
        Indices are clamped to [0, n-1] — points exactly on the far wall
        map to the last voxel rather than going out of bounds.

        Note: use FINAL box coordinates (post-calendering), not pre-calendering.
        """
        vs = self.voxel_size_nm
        ix = int(np.clip(point_nm[0] / vs, 0, self.nx - 1))
        iy = int(np.clip(point_nm[1] / vs, 0, self.ny - 1))
        iz = int(np.clip(point_nm[2] / vs, 0, self.nz - 1))
        return ix, iy, iz

    def voxel_centers(self) -> np.ndarray:
        """
        Physical center coordinates of every voxel in the FINAL box.

        Returns array of shape (nx, ny, nz, 3) in nm.
        Consumed by the Voxelizer for analytical phase assignment.

        Memory: 128³ × 3 × float32 ≈ 24 MB — computed once, reused.
        """
        vs = self.voxel_size_nm
        xs = (np.arange(self.nx, dtype=np.float32) + 0.5) * vs
        ys = (np.arange(self.ny, dtype=np.float32) + 0.5) * vs
        zs = (np.arange(self.nz, dtype=np.float32) + 0.5) * vs
        xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([xg, yg, zg], axis=-1)

    def voxel_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Voxel face positions along each axis (nm).
        Returns (x_edges, y_edges, z_edges), each of length n+1.
        Useful for histogram-style queries (e.g., counting particles per layer).
        """
        vs = self.voxel_size_nm
        x_edges = np.linspace(0.0, self.Lx_nm, self.nx + 1)
        y_edges = np.linspace(0.0, self.Ly_nm, self.ny + 1)
        z_edges = np.linspace(0.0, self.Lz_final_nm, self.nz + 1)
        return x_edges, y_edges, z_edges

    # ── Z-layer helpers (for through-thickness analysis) ────────────────────

    def z_layer_thickness_nm(self) -> float:
        """Thickness of one voxel layer in Z (nm). Same as voxel_size_nm."""
        return self.voxel_size_nm

    def z_layer_index(self, z_nm: float) -> int:
        """Which Z-layer (0-indexed) does physical z coordinate fall in?"""
        return int(np.clip(z_nm / self.voxel_size_nm, 0, self.nz - 1))

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  DOMAIN GEOMETRY",
            "=" * 62,
            f"  Pre-calender box  : {self.Lx_nm/1000:.1f} × "
            f"{self.Ly_nm/1000:.1f} × {self.Lz_pre_nm/1000:.1f} µm",
            f"  Final box         : {self.Lx_nm/1000:.1f} × "
            f"{self.Ly_nm/1000:.1f} × {self.Lz_final_nm/1000:.1f} µm  (cubic)",
            f"  Voxel grid        : {self.nx} × {self.ny} × {self.nz}",
            f"  Voxel size        : {self.voxel_size_nm:.2f} nm  (isotropic)",
            f"  Boundary X, Y     : periodic",
            f"  Boundary Z        : hard wall",
            f"    Z = 0           : current collector",
            f"    Z = {self.Lz_final_nm/1000:.1f} µm       : electrolyte interface",
            f"  Compression       : {self.compression_ratio:.2f}  "
            f"({self.Lz_pre_nm/1000:.1f} µm → {self.Lz_final_nm/1000:.1f} µm)",
            f"  V_pre             : {self.V_pre_nm3:.3e} nm³",
            f"  V_final           : {self.V_final_nm3:.3e} nm³",
            "=" * 62,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_domain(comp: CompositionState) -> DomainGeometry:
    """
    Build a DomainGeometry from a CompositionState.

    All geometric quantities are derived from comp — no new information
    is required. The domain is fully determined by the composition.

    Args:
        comp: Fully computed CompositionState (from compute_composition).

    Returns:
        Frozen DomainGeometry ready for use by all generation steps.
    """
    L = comp.domain_L_nm
    vs = comp.voxel_size_nm
    res = comp.voxel_resolution

    return DomainGeometry(
        Lx_nm=L,
        Ly_nm=L,
        Lz_pre_nm=comp.L_z_pre_nm,
        Lz_final_nm=L,
        nx=res,
        ny=res,
        nz=res,
        voxel_size_nm=vs,
        compression_ratio=comp.compression_ratio,
    )
