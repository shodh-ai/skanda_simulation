import numpy as np
from dataclasses import dataclass, field


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
    void_enabled: bool
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
            f" Void voxels : "
            f"{self.void_mask.sum():,}"
            f"{'' if self.void_enabled else ' (void disabled — mask not computed)'}",
        ]
        if self.warnings:
            lines += ["", f"  ⚠  {len(self.warnings)} WARNING(s):"]
            lines += [f"     [{i+1}] {w}" for i, w in enumerate(self.warnings)]
        lines.append("=" * 62)
        return "\n".join(lines)
