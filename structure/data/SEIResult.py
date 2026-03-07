from dataclasses import dataclass, field
import numpy as np
from typing import List


@dataclass
class SEIResult:
    sei_vf: np.ndarray  # float32 (nx,ny,nz) — local SEI vf
    V_sei_nm3: float  # total SEI volume placed
    surface_area_nm2: float  # estimated solid–pore interface area
    mean_thickness_nm: float  # effective mean thickness (V / surface_area)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  SEI SHELL",
            "=" * 62,
            f"  SEI voxels > 0    : {(self.sei_vf > 0).sum():,}",
            f"  V_SEI             : {self.V_sei_nm3:.4e} nm³",
            f"  Surface area      : {self.surface_area_nm2:.4e} nm²",
            f"  Effective thickness: {self.mean_thickness_nm:.2f} nm",
            f"  sei_vf max        : {self.sei_vf.max():.5f}",
            (
                f"  sei_vf mean (>0)  : {self.sei_vf[self.sei_vf>0].mean():.5f}"
                if self.sei_vf.any()
                else "  sei_vf mean (>0)  : N/A"
            ),
        ]
        if self.warnings:
            lines += ["", f"  ⚠  {len(self.warnings)} WARNING(s):"]
            lines += [f"     [{i+1}] {w}" for i, w in enumerate(self.warnings)]
        lines.append("=" * 62)
        return "\n".join(lines)
