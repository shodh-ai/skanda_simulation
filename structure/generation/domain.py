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
from structure.data import CompositionState, DomainGeometry


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
