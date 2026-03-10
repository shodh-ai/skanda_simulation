"""
Result dataclasses for the simulation pipeline.

Hierarchy:
    SimulationResult
    ├── taufactor    : TauFactorResult
    ├── rate_cap     : RateCapabilityResult
    ├── dcir         : DCIRResult
    └── cycle_life   : CycleLifeResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TauFactorResult:
    # Ionic (pore) network
    tau_ionic: float
    epsilon_ionic: float
    D_eff_ionic_m2_s: float
    bruggeman_exponent: float

    # Electronic (carbon + CBD) — None if not requested
    tau_electronic: Optional[float] = None
    epsilon_electronic: Optional[float] = None

    # Solver metadata
    converged: bool = True
    n_iterations: int = 0
    residual: float = 0.0

    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "  TauFactor:",
            f"    τ_ionic        = {self.tau_ionic:.4f}",
            f"    ε_ionic        = {self.epsilon_ionic:.4f}",
            f"    D_eff_ionic    = {self.D_eff_ionic_m2_s:.4e} m²/s",
            f"    β (Bruggeman)  = {self.bruggeman_exponent:.4f}",
        ]
        if self.tau_electronic is not None:
            lines.append(
                f"    τ_electronic   = {self.tau_electronic:.4f}  "
                f"ε={self.epsilon_electronic:.4f}"
            )
        lines.append(
            f"    converged={self.converged}  "
            f"iter={self.n_iterations}  res={self.residual:.2e}"
        )
        if self.warnings:
            lines += [f"    ⚠ {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class RateCapabilityResult:
    c_rates: list[float]
    capacities_mAh_cm2: list[float]
    energy_densities_mWh_cm2: list[float]

    nominal_capacity_mAh_cm2: float
    energy_density_mWh_cm2: float

    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["  Rate Capability:"]
        for c, q, e in zip(
            self.c_rates,
            self.capacities_mAh_cm2,
            self.energy_densities_mWh_cm2,
        ):
            c_str = f"C/{1/c:.0f}" if c <= 1.0 else f"{c:.1f}C"
            lines.append(f"    {c_str:<8} : {q:.4f} mAh/cm²  {e:.4f} mWh/cm²")
        lines.append(
            f"    Nominal (C/5) : "
            f"{self.nominal_capacity_mAh_cm2:.4f} mAh/cm²  "
            f"{self.energy_density_mWh_cm2:.4f} mWh/cm²"
        )
        if self.warnings:
            lines += [f"    ⚠ {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class DCIRResult:
    dcir_mOhm_cm2: float
    soc_point: float
    pulse_c_rate: float
    pulse_duration_s: float
    delta_V_mV: float

    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "  DCIR:",
            f"    DCIR           = {self.dcir_mOhm_cm2:.2f} mΩ·cm²",
            f"    ΔV             = {self.delta_V_mV:.2f} mV",
            f"    @ SOC={self.soc_point:.0%}  "
            f"{self.pulse_c_rate}C  {self.pulse_duration_s:.0f}s",
        ]
        if self.warnings:
            lines += [f"    ⚠ {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class CycleLifeResult:
    cycles_run: int
    cycle_numbers: list[int]
    capacities_mAh_cm2: list[float]

    projected_cycle_life: Optional[int]
    capacity_fade_rate_pct_per_cycle: float

    initial_capacity_mAh_cm2: float
    final_capacity_mAh_cm2: float
    retention_at_final_cycle: float

    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        life_str = (
            f"{self.projected_cycle_life} cycles"
            if self.projected_cycle_life
            else f"> {self.cycles_run} cycles (EOL not reached)"
        )
        lines = [
            "  Cycle Life:",
            f"    Projected EOL  = {life_str}",
            f"    Fade rate      = "
            f"{self.capacity_fade_rate_pct_per_cycle:.4f} %/cycle",
            f"    Q_initial      = {self.initial_capacity_mAh_cm2:.4f} mAh/cm²",
            f"    Q_final        = {self.final_capacity_mAh_cm2:.4f} mAh/cm²"
            f"  (retention={self.retention_at_final_cycle:.1%})",
        ]
        if self.warnings:
            lines += [f"    ⚠ {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class SimulationResult:
    """Complete output of run_simulation()."""

    taufactor: Optional[TauFactorResult]
    rate_capability: Optional[RateCapabilityResult]
    dcir: Optional[DCIRResult]
    cycle_life: Optional[CycleLifeResult]

    elapsed_s: float
    step_times_s: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=" * 60, " SIMULATION RESULT", "=" * 60]
        lines.extend(
            result.summary()
            for result in [
                self.taufactor,
                self.rate_capability,
                self.dcir,
                self.cycle_life,
            ]
            if result is not None
        )
        if self.warnings:
            lines += ["", f"  ⚠ {len(self.warnings)} pipeline WARNING(s):"]
            lines += [f"    [{i+1}] {w}" for i, w in enumerate(self.warnings)]
        lines += ["", f"  Total time : {self.elapsed_s:.2f}s", "=" * 60]
        return "\n".join(lines)
