from structure.data import MicrostructureVolume
from dataclasses import dataclass, field


@dataclass
class PipelineResult:
    """
    Output of run().

    volume       : the assembled MicrostructureVolume
    seed_used    : the seed that produced a percolating structure
    attempts     : number of attempts (1 = first try succeeded)
    elapsed_s    : wall-clock seconds for the full run
    step_times_s : per-step wall-clock seconds {step_name: seconds}
    warnings     : pipeline-level warnings (step warnings are in volume.warnings)
    """

    volume: MicrostructureVolume
    seed_used: int
    attempts: int
    elapsed_s: float
    step_times_s: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 66,
            " PIPELINE RUN COMPLETE",
            "=" * 66,
            f"  Seed used   : {self.seed_used}",
            f"  Attempts    : {self.attempts}",
            f"  Total time  : {self.elapsed_s:.2f}s",
            "",
            "  Step times:",
        ]
        lines.extend(
            f"    {name:<30} {t:.3f}s" for name, t in self.step_times_s.items()
        )
        if self.warnings:
            lines += ["", f"  ⚠ {len(self.warnings)} pipeline WARNING(s):"]
            lines += [f"    [{i+1}] {w}" for i, w in enumerate(self.warnings)]
        lines.append("=" * 66)
        return "\n".join(lines)
