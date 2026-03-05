"""
Step 8 — Voxelizer

Converts particle geometry + sub-voxel vf fields into:
  label_map  uint8   (nx,ny,nz) — one discrete phase per voxel
  si_vf      float16 (nx,ny,nz)
  cbd_vf     float16 (nx,ny,nz)
  binder_vf  float16 (nx,ny,nz)
  sei_vf     float16 (nx,ny,nz)

Colors for all visualization are sourced from materials_db via
phases.build_phase_colors(sim) — never hardcoded here.

Speed: bounding-box rasterization (~49× faster than full-grid).
Priority: SEI(6) > Graphite(5) > Coating(4) > Si(3) > CBD(2) > Binder(1) > Pore(0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .composition import CompositionState
from .domain import DomainGeometry
from .carbon_packer import OblateSpheroid
from .si_mapper import SiMapResult
from .cbd_binder import CBDBinderResult
from .sei import SEIResult
from ..phases import (
    PHASE_PORE,
    PHASE_GRAPHITE,
    PHASE_SI,
    PHASE_CBD,
    PHASE_BINDER,
    PHASE_SEI,
    PHASE_NAMES,
    LABEL_PRIORITY,
    build_phase_colors,
    build_phase_colors_hex,
)


# vf thresholds for converting continuous → discrete label
_THRESHOLDS: dict[int, float] = {
    PHASE_SI: 0.02,
    PHASE_CBD: 0.10,
    PHASE_BINDER: 0.15,
    PHASE_SEI: 0.005,
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class VoxelGrid:
    """
    Final voxelized microstructure.

    All colors are stored as resolved float RGB tuples sourced from
    materials_db — not hardcoded defaults.
    """

    label_map: np.ndarray  # uint8  (nx,ny,nz)
    si_vf: np.ndarray  # float16 (nx,ny,nz)
    cbd_vf: np.ndarray  # float16 (nx,ny,nz)
    binder_vf: np.ndarray  # float16 (nx,ny,nz)
    sei_vf: np.ndarray  # float16 (nx,ny,nz)
    voxel_size_nm: float
    phase_fractions: dict[str, float]
    phase_colors: dict[int, tuple[float, float, float]]  # from materials_db
    phase_colors_hex: dict[int, str]  # from materials_db
    warnings: List[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.label_map.shape  # type: ignore[return-value]

    def to_rgb(self) -> np.ndarray:
        """
        Convert label_map to a float32 RGB volume (nx, ny, nz, 3).
        Colors sourced from self.phase_colors (materials_db values).
        Used for slicing and 3D rendering.
        """
        nx, ny, nz = self.shape
        rgb = np.zeros((nx, ny, nz, 3), dtype=np.float32)
        for phase_id, color in self.phase_colors.items():
            mask = self.label_map == phase_id
            rgb[mask] = color
        return rgb

    def slice_rgb(self, axis: int, index: int) -> np.ndarray:
        """
        Return a 2D RGB image (H, W, 3) for a slice along `axis` at `index`.
        axis: 0=X, 1=Y, 2=Z
        """
        if axis == 0:
            sl = self.label_map[index, :, :]
        elif axis == 1:
            sl = self.label_map[:, index, :]
        else:
            sl = self.label_map[:, :, index]

        H, W = sl.shape
        img = np.zeros((H, W, 3), dtype=np.float32)
        for phase_id, color in self.phase_colors.items():
            img[sl == phase_id] = color
        return img

    def summary(self) -> str:
        nx, ny, nz = self.shape
        N = self.label_map.size

        # ── Memory (fix: binder_vf was missing) ───────────────────────────
        mb = (
            self.label_map.nbytes
            + self.si_vf.nbytes
            + self.cbd_vf.nbytes
            + self.binder_vf.nbytes          # was missing before
            + self.sei_vf.nbytes
        ) / 1e6

        # ── Actual VF: sum float16 arrays directly ─────────────────────────
        # Cast to float32 first to avoid float16 accumulation error on large grids.
        actual_fracs: dict[str, float] = {}
        for pid in sorted(PHASE_NAMES.keys()):
            name = PHASE_NAMES[pid]
            if pid == PHASE_SI:
                actual_fracs[name] = float(self.si_vf.astype(np.float32).sum()) / N
            elif pid == PHASE_CBD:
                actual_fracs[name] = float(self.cbd_vf.astype(np.float32).sum()) / N
            elif pid == PHASE_BINDER:
                actual_fracs[name] = float(self.binder_vf.astype(np.float32).sum()) / N
            elif pid == PHASE_SEI:
                actual_fracs[name] = float(self.sei_vf.astype(np.float32).sum()) / N
            else:
                # Graphite and Pore have no continuous VF field — use label count
                actual_fracs[name] = float((self.label_map == pid).sum()) / N

        # ── Label VF: count uint8 label_map entries ────────────────────────
        label_fracs: dict[str, float] = {
            PHASE_NAMES[pid]: float((self.label_map == pid).sum()) / N
            for pid in sorted(PHASE_NAMES.keys())
        }

        # ── Format ─────────────────────────────────────────────────────────
        col_w = 70
        lines = [
            "=" * col_w,
            "  VOXEL GRID",
            "=" * col_w,
            f"  Shape         : {nx}×{ny}×{nz} = {N:,} voxels",
            f"  Voxel size    : {self.voxel_size_nm:.2f} nm",
            f"  Memory        : {mb:.1f} MB",
            "",
            "  Phase volume fractions",
            f"  {'Phase':<12}  {'Actual VF':>10}  {'Label VF':>10}  Color     Bar (Actual)",
            f"  {'-'*12}  {'-'*10}  {'-'*10}  --------  -----------",
        ]

        for pid in sorted(PHASE_NAMES.keys()):
            name   = PHASE_NAMES[pid]
            af     = actual_fracs.get(name, 0.0)
            lf     = label_fracs.get(name, 0.0)
            hx     = self.phase_colors_hex.get(pid, "#888888")
            bar    = "█" * int(af * 30)
            # Mark divergence between actual and label VF with a flag
            diff   = abs(af - lf)
            flag   = "  ◄ Δ{:.3f}".format(diff) if diff > 0.01 else ""
            lines.append(
                f"  {name:<12}  {af:>10.4f}  {lf:>10.4f}  {hx}  {bar}{flag}"
            )

        lines += [
            "",
            "  Notes:",
            "  • Actual VF  — summed from float16 sub-voxel fields (pre-threshold truth).",
            "  • Label VF   — counted from uint8 label_map  (what solvers/renderers see).",
            "  • ◄ Δ flag   — phases where Actual vs Label diverges by > 1 pp.",
            "  • Graphite / Pore have no float field; both columns use label_map counts.",
        ]

        if self.warnings:
            lines += ["", f"  ⚠  {len(self.warnings)} WARNING(s):"]
            lines += [f"     [{i+1}] {w}" for i, w in enumerate(self.warnings)]

        lines.append("=" * col_w)
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save all arrays + voxel_size to .npz."""
        np.savez_compressed(
            path,
            label_map=self.label_map,
            si_vf=self.si_vf,
            cbd_vf=self.cbd_vf,
            binder_vf=self.binder_vf,
            sei_vf=self.sei_vf,
            voxel_size_nm=np.float32(self.voxel_size_nm),
        )

    @staticmethod
    def load(path: str, sim=None) -> "VoxelGrid":
        """
        Load a saved VoxelGrid.
        Pass sim to restore material colors; otherwise fallback colors used.
        """
        from ..phases import (
            _FALLBACK_COLORS_RGB,
            build_phase_colors,
            build_phase_colors_hex,
        )

        d = np.load(path)
        lm = d["label_map"]
        vs = float(d["voxel_size_nm"])
        pf = _compute_phase_fractions(lm)

        if sim is not None:
            pc = build_phase_colors(sim)
            pc_hex = build_phase_colors_hex(sim)
        else:
            pc = _FALLBACK_COLORS_RGB
            pc_hex = {
                pid: "#{:02X}{:02X}{:02X}".format(
                    int(r * 255), int(g * 255), int(b * 255)
                )
                for pid, (r, g, b) in _FALLBACK_COLORS_RGB.items()
            }

        return VoxelGrid(
            label_map=lm,
            si_vf=d["si_vf"],
            cbd_vf=d["cbd_vf"],
            binder_vf=d["binder_vf"],
            sei_vf=d["sei_vf"],
            voxel_size_nm=vs,
            phase_fractions=pf,
            phase_colors=pc,
            phase_colors_hex=pc_hex,
        )


# ---------------------------------------------------------------------------
# Bounding-box rasterizer (core inner loop)
# ---------------------------------------------------------------------------


def _rasterize_spheroid(
    volume: np.ndarray,  # uint8 (nx,ny,nz) — in-place
    priority_map: np.ndarray,  # int8  (nx,ny,nz) — in-place
    p: OblateSpheroid,
    phase_id: int,
    priority: int,
    vs: float,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    nz: int,
) -> int:
    """
    Rasterize one oblate spheroid using a bounding-box sub-grid.

    ~49× faster than evaluating the inside-test on the full 128³ grid
    because we only operate on the (2·half+1)³ box around each particle.

    Returns number of voxels written.
    """
    half = int(np.ceil(p.a / vs)) + 1

    ix_c = int(p.center[0] / vs)
    iy_c = int(p.center[1] / vs)
    iz_c = int(p.center[2] / vs)

    lx = np.arange(ix_c - half, ix_c + half + 1)
    ly = np.arange(iy_c - half, iy_c + half + 1)
    lz = np.arange(iz_c - half, iz_c + half + 1)

    # Hard wall in Z, periodic in X and Y
    lz = lz[(lz >= 0) & (lz < nz)]
    if lz.size == 0:
        return 0

    lx_w = lx % nx
    ly_w = ly % ny

    # Physical centers of bounding-box voxels (use unwrapped for coords)
    xs = (lx + 0.5) * vs
    ys = (ly + 0.5) * vs
    zs = (lz + 0.5) * vs

    Xs, Ys, Zs = np.meshgrid(xs, ys, zs, indexing="ij")

    dx = Xs - p.center[0]
    dx -= Lx * np.round(dx / Lx)
    dy = Ys - p.center[1]
    dy -= Ly * np.round(dy / Ly)
    dz = Zs - p.center[2]

    RT = p.R.T
    dx_b = RT[0, 0] * dx + RT[0, 1] * dy + RT[0, 2] * dz
    dy_b = RT[1, 0] * dx + RT[1, 1] * dy + RT[1, 2] * dz
    dz_b = RT[2, 0] * dx + RT[2, 1] * dy + RT[2, 2] * dz

    inside = (dx_b / p.a) ** 2 + (dy_b / p.a) ** 2 + (dz_b / p.c) ** 2 <= 1.0

    if not inside.any():
        return 0

    # Index grids for scatter (broadcast, no memory copy)
    ixg = lx_w[:, None, None] * np.ones((1, len(ly), len(lz)), dtype=np.intp)
    iyg = ly_w[None, :, None] * np.ones((len(lx), 1, len(lz)), dtype=np.intp)
    izg = lz[None, None, :] * np.ones((len(lx), len(ly), 1), dtype=np.intp)

    overwrite = inside & (priority_map[ixg, iyg, izg] < priority)
    if not overwrite.any():
        return 0

    ix_s = ixg[overwrite]
    iy_s = iyg[overwrite]
    iz_s = izg[overwrite]

    volume[ix_s, iy_s, iz_s] = phase_id
    priority_map[ix_s, iy_s, iz_s] = priority

    return int(overwrite.sum())


# ---------------------------------------------------------------------------
# Voxelizer
# ---------------------------------------------------------------------------


class Voxelizer:

    def __init__(
        self,
        comp: CompositionState,
        domain: DomainGeometry,
        sim,  # ResolvedSimulation — for color lookup
    ) -> None:
        self.comp = comp
        self.domain = domain
        self.sim = sim

    def voxelize(
        self,
        particles: List[OblateSpheroid],
        si_result: SiMapResult,
        cbd_result: CBDBinderResult,
        sei_result: SEIResult,
    ) -> VoxelGrid:

        domain = self.domain
        comp = self.comp
        nx, ny, nz = domain.nx, domain.ny, domain.nz
        vs, Lx, Ly = domain.voxel_size_nm, domain.Lx_nm, domain.Ly_nm

        label_map = np.zeros((nx, ny, nz), dtype=np.uint8)
        priority_map = np.full((nx, ny, nz), -1, dtype=np.int8)

        warns: List[str] = []

        # ------------------------------------------------------------------
        # Pass 1 — sub-voxel fields (lowest → highest priority before carbon)
        # ------------------------------------------------------------------
        for phase_id, vf_field in [
            (PHASE_BINDER, cbd_result.binder_vf),
            (PHASE_CBD, cbd_result.cbd_vf),
            (PHASE_SI, si_result.si_vf),
        ]:
            pri = LABEL_PRIORITY[phase_id]
            thr = _THRESHOLDS[phase_id]
            mask = (vf_field > thr) & (priority_map < pri)
            label_map[mask] = phase_id
            priority_map[mask] = pri

        # ------------------------------------------------------------------
        # Pass 2 — carbon particles (bounding-box rasterizer, priority 5)
        # ------------------------------------------------------------------
        n_carbon_voxels = 0
        for p in particles:
            n_carbon_voxels += _rasterize_spheroid(
                volume=label_map,
                priority_map=priority_map,
                p=p,
                phase_id=PHASE_GRAPHITE,
                priority=LABEL_PRIORITY[PHASE_GRAPHITE],
                vs=vs,
                Lx=Lx,
                Ly=Ly,
                nx=nx,
                ny=ny,
                nz=nz,
            )

        if n_carbon_voxels == 0:
            warns.append("[CRITICAL] Zero carbon voxels filled.")

        # ------------------------------------------------------------------
        # Pass 3 — SEI (highest priority, surface only)
        # ------------------------------------------------------------------
        pri_sei = LABEL_PRIORITY[PHASE_SEI]
        thr_sei = _THRESHOLDS[PHASE_SEI]
        solid = label_map > PHASE_PORE
        sei_mask = (sei_result.sei_vf > thr_sei) & solid & (priority_map < pri_sei)
        label_map[sei_mask] = PHASE_SEI
        priority_map[sei_mask] = pri_sei

        # ------------------------------------------------------------------
        # Pass 4 — Pore Dominance
        #
        # Sub-voxel phases (Si, CBD, Binder, SEI) represent partial
        # occupancy: a voxel labeled "Silicon" may only be 15% Si and
        # 85% void.  If the discrete label map says "3 = Silicon",
        # downstream solvers (TauFactor, PyBaMM) will treat it as a
        # solid block, closing off the pore network.
        #
        # Rule: for non-carbon voxels, if the total sub-voxel solid
        # fraction is < 50%, the voxel MUST stay Pore.  Carbon
        # (Graphite) voxels are analytically rasterized — they are
        # definitively inside a particle body and always 100% solid.
        # ------------------------------------------------------------------
        _PORE_DOMINANCE_THRESHOLD = 0.50

        total_solid_vf = (
            si_result.si_vf
            + cbd_result.cbd_vf
            + cbd_result.binder_vf
            + sei_result.sei_vf
        )

        # Only reset voxels that were filled by sub-voxel fields, not
        # by the carbon rasterizer (Graphite voxels are fully solid).
        pore_dominant = (
            (total_solid_vf < _PORE_DOMINANCE_THRESHOLD)
            & (label_map != PHASE_GRAPHITE)
            & (label_map != PHASE_PORE)     # already pore — skip
        )
        label_map[pore_dominant] = PHASE_PORE
        priority_map[pore_dominant] = LABEL_PRIORITY[PHASE_PORE]

        # ------------------------------------------------------------------
        # Colors from materials_db
        # ------------------------------------------------------------------
        phase_colors = build_phase_colors(self.sim)
        phase_colors_hex = build_phase_colors_hex(self.sim)

        # ------------------------------------------------------------------
        # Phase fractions + sanity warnings
        # ------------------------------------------------------------------
        phase_fractions = _compute_phase_fractions(
            label_map,
            cbd_vf=cbd_result.cbd_vf,
            binder_vf=cbd_result.binder_vf,
        )

        c_frac = phase_fractions.get("Graphite", 0.0)
        expected_c = comp.vf_carbon * (1.0 - comp.porosity)
        if c_frac < 0.5 * expected_c:
            warns.append(
                f"Graphite fraction {c_frac:.3f} is well below expected "
                f"{expected_c:.3f}. Check particle coordinates / compression_ratio."
            )
        if phase_fractions.get("Pore", 1.0) < 0.05:
            warns.append("Pore fraction < 5% — electrode may be over-packed.")

        return VoxelGrid(
            label_map=label_map,
            si_vf=si_result.si_vf.astype(np.float16),
            cbd_vf=cbd_result.cbd_vf.astype(np.float16),
            binder_vf=cbd_result.binder_vf.astype(np.float16),
            sei_vf=sei_result.sei_vf.astype(np.float16),
            voxel_size_nm=vs,
            phase_fractions=phase_fractions,
            phase_colors=phase_colors,
            phase_colors_hex=phase_colors_hex,
            warnings=warns,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_phase_fractions(
    label_map: np.ndarray,
    *,
    cbd_vf: np.ndarray | None = None,
    binder_vf: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute per-phase volume fractions.

    For CBD and Binder the *float* VF arrays are summed so that the
    reported fractions reflect the true continuous values even when
    the label-map thresholds are set high enough to hide them from
    the 3D viewer.
    """
    N = label_map.size
    fracs: dict[str, float] = {}
    for pid in sorted(PHASE_NAMES.keys()):
        name = PHASE_NAMES[pid]
        if pid == PHASE_CBD and cbd_vf is not None:
            fracs[name] = float(cbd_vf.sum()) / N
        elif pid == PHASE_BINDER and binder_vf is not None:
            fracs[name] = float(binder_vf.sum()) / N
        else:
            fracs[name] = float((label_map == pid).sum()) / N
    return fracs


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def voxelize_microstructure(
    comp: CompositionState,
    domain: DomainGeometry,
    sim,
    particles: List[OblateSpheroid],
    si_result: SiMapResult,
    cbd_result: CBDBinderResult,
    sei_result: SEIResult,
) -> VoxelGrid:
    """
    Canonical pipeline entry for Step 8.
    sim is required here (unlike previous steps) to resolve material colors.
    """
    return Voxelizer(comp, domain, sim).voxelize(
        particles=particles,
        si_result=si_result,
        cbd_result=cbd_result,
        sei_result=sei_result,
    )
