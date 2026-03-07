from dataclasses import dataclass, field
from structure.data import OblateSpheroid


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
    n_overlapping_pairs: int = 0
    n_pairs_checked: int = 0
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
                f" Inflation factor : {self.inflation_factor:.4f}"
                f" (XY basal axes scaled by this factor)"
            )
            overlap_str = (
                f"{self.n_overlapping_pairs} / {self.n_pairs_checked} pairs"
                f" ({100.0 * self.n_overlapping_pairs / max(self.n_pairs_checked, 1):.1f}%)"
                if self.n_pairs_checked > 0
                else "not checked"
            )
            lines.append(
                f" Overlapping pairs: {overlap_str} "
                f"{'⚠ UNPHYSICAL' if self.n_overlapping_pairs > 0 else '✓ none'}"
            )
        if self.warnings:
            lines += ["", f"  ⚠  {len(self.warnings)} WARNING(s):"]
            lines += [f"     [{i+1}] {w}" for i, w in enumerate(self.warnings)]
        lines.append("=" * 62)
        return "\n".join(lines)
