"""
Carbon Scaffold Rasterizer — Step 2b

Converts placed OblateSpheroid particles (post-calendering, final-domain
coordinates) into a uint8 voxel label array:

    carbon_label[x, y, z] = PHASE_GRAPHITE (1) if voxel centre is inside
                             any spheroid, else PHASE_PORE (0).

This is the only array that requires hard discrete geometry — all other
phases (Si, CBD, binder, SEI) are sub-voxel volume-fraction fields.

Algorithm:
    Per-particle AABB slicing + analytical spheroid membership test.
    Complexity: O(N_carbon × AABB³) vs O(N_carbon × nx × ny × nz) brute force.
    Typical speedup: ~60× at 128³ for 12µm graphite (AABB ≈ 32³ voxels).

Boundary conditions:
    X, Y — periodic (minimum-image displacement, modulo storage index)
    Z     — hard wall (clamp to [0, nz-1], skip out-of-range slices)

Input coordinates:
    Particles must be in FINAL (post-calendering) domain coordinates.
    Call calender_particles() before rasterize_carbon().
"""

from __future__ import annotations

import numpy as np

from structure.data import RasterResult, PackingResult, DomainGeometry
from structure.phases import PHASE_GRAPHITE


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Core rasterizer
# ---------------------------------------------------------------------------


def rasterize_carbon(
    packing: PackingResult,
    domain: DomainGeometry,
) -> RasterResult:
    """
    Rasterize placed carbon particles into a uint8 label array.

    Args:
        packing : PackingResult from Step 2 — particles must already be in
                  final-domain (post-calendering) coordinates.
        domain  : DomainGeometry (final domain).

    Returns:
        RasterResult with carbon_label uint8 array and diagnostics.
    """
    nx, ny, nz = domain.nx, domain.ny, domain.nz
    vs = domain.voxel_size_nm
    Lx = domain.Lx_nm
    Ly = domain.Ly_nm

    carbon_label = np.zeros((nx, ny, nz), dtype=np.uint8)
    warns: list[str] = []
    n_empty = 0

    for p in packing.particles:
        cx, cy, cz = p.center[0], p.center[1], p.center[2]

        # ── AABB half-extent ─────────────────────────────────────────────
        # p.a is the largest semi-axis (oblate: a >= c).
        # Conservative: all spheroid points satisfy |Δx|,|Δy|,|Δz| ≤ p.a.
        half = int(np.ceil(p.a / vs)) + 1

        # ── Integer index ranges ─────────────────────────────────────────
        ix_c = int(cx / vs)
        iy_c = int(cy / vs)
        iz_c = int(cz / vs)

        ix_arr = np.arange(ix_c - half, ix_c + half + 1)
        iy_arr = np.arange(iy_c - half, iy_c + half + 1)
        iz_arr = np.arange(iz_c - half, iz_c + half + 1)

        # Z — hard wall: clamp strictly to valid voxel range
        iz_arr = iz_arr[(iz_arr >= 0) & (iz_arr < nz)]
        if iz_arr.size == 0:
            n_empty += 1
            continue

        # X, Y — periodic: wrapped storage indices
        ix_w = ix_arr % nx
        iy_w = iy_arr % ny

        # ── Sub-grid voxel centres (unmodded for correct displacement) ───
        xs_sub = (ix_arr + 0.5) * vs
        ys_sub = (iy_arr + 0.5) * vs
        zs_sub = (iz_arr + 0.5) * vs

        Xs, Ys, Zs = np.meshgrid(xs_sub, ys_sub, zs_sub, indexing="ij")

        # ── Minimum-image displacement from particle centre ──────────────
        dx = Xs - cx
        dx -= Lx * np.round(dx / Lx)  # periodic X
        dy = Ys - cy
        dy -= Ly * np.round(dy / Ly)  # periodic Y
        dz = Zs - cz  # hard wall Z — no wrapping

        # ── Body-frame coordinates via rotation matrix ───────────────────
        RT = p.R.T
        dx_b = RT[0, 0] * dx + RT[0, 1] * dy + RT[0, 2] * dz
        dy_b = RT[1, 0] * dx + RT[1, 1] * dy + RT[1, 2] * dz
        dz_b = RT[2, 0] * dx + RT[2, 1] * dy + RT[2, 2] * dz

        # ── Spheroid membership: (x/a)² + (y/a)² + (z/c)² ≤ 1 ──────────
        inside = ((dx_b / p.a) ** 2 + (dy_b / p.a) ** 2 + (dz_b / p.c) ** 2) <= 1.0

        if not inside.any():
            n_empty += 1
            continue

        # ── Write to label array via wrapped storage indices ─────────────
        ixg = ix_w[:, None, None] * np.ones(
            (1, len(iy_arr), len(iz_arr)), dtype=np.intp
        )
        iyg = iy_w[None, :, None] * np.ones(
            (len(ix_arr), 1, len(iz_arr)), dtype=np.intp
        )
        izg = iz_arr[None, None, :] * np.ones(
            (len(ix_arr), len(iy_arr), 1), dtype=np.intp
        )

        carbon_label[ixg[inside], iyg[inside], izg[inside]] = PHASE_GRAPHITE

    # ── Diagnostics ───────────────────────────────────────────────────────
    if n_empty > 0:
        warns.append(
            f"{n_empty}/{len(packing.particles)} particles produced no "
            f"voxels inside the domain after AABB clipping. "
            f"Check that calender_particles() was called before rasterize_carbon() "
            f"and that particle centres are in final-domain coordinates."
        )

    vf_carbon = float(carbon_label.sum()) / carbon_label.size

    # Sanity check: measured VF vs target
    # (packing.phi_target is pre-calender; after compression vf ≈ phi_target)
    if packing.phi_target > 0:
        vf_delta = abs(vf_carbon - packing.phi_target)
        if vf_delta > 0.05:
            warns.append(
                f"Rasterized carbon VF={vf_carbon:.4f} deviates from "
                f"packing target phi={packing.phi_target:.4f} by "
                f"{vf_delta*100:.1f} pp. "
                f"Expected for small N_carbon or near-boundary particles. "
                f"Large deviations may indicate coordinate mismatch between "
                f"pre- and post-calender domains."
            )

    return RasterResult(
        carbon_label=carbon_label,
        vf_carbon=vf_carbon,
        n_particles=len(packing.particles),
        warnings=warns,
    )
