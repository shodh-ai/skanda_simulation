from dataclasses import dataclass, field
import numpy as np


@dataclass
class RasterResult:
    """
    Output of rasterize_carbon().

    carbon_label : uint8 (nx, ny, nz)
                   PHASE_GRAPHITE (1) inside spheroids, PHASE_PORE (0) elsewhere.
    vf_carbon    : float — measured carbon volume fraction (label count / N_voxels).
                   Should be close to CompositionState.phi_carbon_pre × cr.
    n_particles  : number of particles rasterized.
    warnings     : list of diagnostic strings.
    """

    carbon_label: np.ndarray  # uint8 (nx, ny, nz)
    vf_carbon: float  # measured from label array
    n_particles: int
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        nx, ny, nz = self.carbon_label.shape
        lines = [
            "=" * 62,
            " CARBON RASTERIZER",
            "=" * 62,
            f"  Grid shape     : {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels",
            f"  Particles      : {self.n_particles}",
            f"  Carbon VF      : {self.vf_carbon:.4f}",
            f"  Carbon voxels  : {int(self.carbon_label.sum()):,}",
        ]
        if self.warnings:
            lines += ["", f"  ⚠ {len(self.warnings)} WARNING(s):"]
            lines += [f"    [{i+1}] {w}" for i, w in enumerate(self.warnings)]
        lines.append("=" * 62)
        return "\n".join(lines)
