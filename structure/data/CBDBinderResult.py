from dataclasses import dataclass, field
import numpy as np
from typing import List


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
