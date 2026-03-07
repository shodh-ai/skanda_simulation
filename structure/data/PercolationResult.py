from dataclasses import dataclass, field
from typing import List


@dataclass
class PercolationResult:
    """
    Full percolation diagnostics for one microstructure.

    electronic_percolating   : True if elec. network percolates Z=0→Z=nz-1
    electronic_fraction      : fraction of solid in percolating components
    electronic_n_components  : total number of distinct solid components
    electronic_n_percolating : number of components that span Z

    ionic_percolating         : True if pore space percolates Z=0→Z=nz-1
    ionic_fraction            : fraction of pore in percolating pore components
    ionic_n_components        : total pore components
    actual_porosity           : measured pore fraction in this volume
    warnings                  : list of warning strings
    """

    # Electronic (conductive) network
    electronic_percolating: bool
    electronic_fraction: float
    electronic_n_components: int
    electronic_n_percolating: int

    # Ionic (pore) network
    ionic_percolating: bool
    ionic_fraction: float
    ionic_n_components: int
    ionic_n_percolating: int

    # Global
    actual_porosity: float
    min_threshold: float
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        e_ok = "✓  PASS" if self.electronic_percolating else "✗  FAIL"
        i_ok = "✓  PASS" if self.ionic_percolating else "✗  FAIL"
        lines = [
            "=" * 62,
            "  PERCOLATION VALIDATOR",
            "=" * 62,
            f"  Electronic network  [{e_ok}]",
            f"    percolating frac : {self.electronic_fraction:.4f}"
            f"  (threshold ≥ {self.min_threshold:.2f})",
            f"    components       : {self.electronic_n_components:,}"
            f"  ({self.electronic_n_percolating} span Z)",
            "",
            f"  Ionic (pore) path   [{i_ok}]",
            f"    percolating frac : {self.ionic_fraction:.4f}",
            f"    pore components  : {self.ionic_n_components:,}",
            f" ({self.ionic_n_percolating} span Z)",
            "",
            f"  Actual porosity    : {self.actual_porosity:.4f}",
        ]
        if self.warnings:
            lines += ["", f"  ⚠  {len(self.warnings)} WARNING(s):"]
            lines += [f"     [{i+1}] {w}" for i, w in enumerate(self.warnings)]
        lines.append("=" * 62)
        return "\n".join(lines)
