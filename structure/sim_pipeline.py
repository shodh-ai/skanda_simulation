"""
Simulation pipeline orchestrator — Steps A through C.

Single entry point:

    result = run_simulation(vol, sim)

Step dependency map:
    Step A   run_taufactor(vol, sim)              → TauFactorResult
    Step B   build_parameter_set(vol, sim, tau)   → pybamm.ParameterValues
    Step C-1 run_rate_capability(param, sim)       → RateCapabilityResult
    Step C-2 run_dcir_pulse(param, sim)            → DCIRResult
    Step C-3 run_cycle_life(param, sim)            → CycleLifeResult

Steps C-1, C-2, C-3 are independent — a failure in one does not abort others.
Step A failure raises immediately (τ is required for the parameter set).
"""

from __future__ import annotations

import time
from typing import Optional

from structure.data import (
    MicrostructureVolume,
    SimulationResult,
    TauFactorResult,
    RateCapabilityResult,
    DCIRResult,
    CycleLifeResult,
)
from structure.schema import ResolvedSimulation
from structure.simulation import (
    run_taufactor,
    build_parameter_set,
    run_rate_capability,
    run_dcir_pulse,
    run_cycle_life,
)


def run_simulation(
    vol: MicrostructureVolume,
    sim: ResolvedSimulation,
    verbose: bool = True,
) -> SimulationResult:
    """
    Run the full simulation pipeline.

    Args:
        vol     : MicrostructureVolume — from gen_pipeline.run() or .load()
        sim     : ResolvedSimulation   — from resolve_simulation()
        verbose : print per-step progress

    Returns:
        SimulationResult

    Raises:
        ImportError  if taufactor or pybamm are not installed
        RuntimeError if Step A (TauFactor) fails and is enabled
    """
    t0 = time.perf_counter()
    step_times: dict[str, float] = {}
    warns: list[str] = []

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    def _timed(name: str, fn, *args, **kwargs):
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        step_times[name] = time.perf_counter() - t
        _log(f"  [{name}] {step_times[name]:.3f}s")
        return out

    def _collect(step_warns: list[str], step: str) -> None:
        for w in step_warns:
            warns.append(f"[{step}] {w}")

    _log("\n── Simulation Pipeline ──")
    _log(sim.summary())

    # ── Step A: TauFactor ─────────────────────────────────────────────────
    tau_result: Optional[TauFactorResult] = None
    _needs_tau = sim.taufactor.enabled and any(
        [
            sim.outputs.tortuosity_factor,
            sim.outputs.electrode_porosity,
            sim.outputs.effective_diffusivity,
            sim.outputs.bruggeman_exponent,
        ]
    )

    if _needs_tau:
        _log("\n── Step A: TauFactor ──")
        try:
            tau_result = _timed("A_taufactor", run_taufactor, vol, sim)
            _log(tau_result.summary())
            _collect(tau_result.warnings, "A")
        except ImportError:
            raise
        except Exception as exc:
            raise RuntimeError(f"TauFactor step failed: {exc}") from exc
    else:
        _log("  Step A: skipped (taufactor disabled or no τ/ε outputs requested)")

    # ── Step B: PyBaMM parameter set ─────────────────────────────────────
    param = None
    _needs_pybamm = sim.pybamm.enabled and any(
        [
            sim.outputs.nominal_capacity_mAh_cm2,
            sim.outputs.energy_density_mWh_cm2,
            sim.outputs.initial_dcir_mOhm_cm2,
            sim.outputs.projected_cycle_life,
            sim.outputs.capacity_fade_rate_pct_per_cycle,
        ]
    )

    if _needs_pybamm:
        _log("\n── Step B: PyBaMM parameter set ──")
        try:
            param = _timed("B_params", build_parameter_set, vol, sim, tau_result)
            _log("  Parameter set assembled.")
        except ImportError:
            raise
        except Exception as exc:
            warns.append(
                f"[B] PyBaMM parameter assembly failed: {exc}. "
                f"All PyBaMM steps skipped."
            )
    else:
        _log("  Step B: skipped (pybamm disabled or no electrochemical outputs)")

    # ── Step C-1: Rate capability ─────────────────────────────────────────
    rate_cap: Optional[RateCapabilityResult] = None
    if (
        param is not None
        and sim.rate_capability
        and sim.rate_capability.enabled
        and (sim.outputs.nominal_capacity_mAh_cm2 or sim.outputs.energy_density_mWh_cm2)
    ):
        _log("\n── Step C-1: Rate Capability ──")
        try:
            rate_cap = _timed("C1_rate_cap", run_rate_capability, param, sim)
            _log(rate_cap.summary())
            _collect(rate_cap.warnings, "C-1")
        except Exception as exc:
            warns.append(f"[C-1] Rate capability failed: {exc}")

    # ── Step C-2: DCIR pulse ──────────────────────────────────────────────
    dcir: Optional[DCIRResult] = None
    if (
        param is not None
        and sim.dcir_pulse
        and sim.dcir_pulse.enabled
        and sim.outputs.initial_dcir_mOhm_cm2
    ):
        _log("\n── Step C-2: DCIR Pulse ──")
        try:
            dcir = _timed("C2_dcir", run_dcir_pulse, param, sim)
            _log(dcir.summary())
            _collect(dcir.warnings, "C-2")
        except Exception as exc:
            warns.append(f"[C-2] DCIR pulse failed: {exc}")

    # ── Step C-3: Cycle life ──────────────────────────────────────────────
    cycle_life: Optional[CycleLifeResult] = None
    if (
        param is not None
        and sim.cycle_life
        and sim.cycle_life.enabled
        and (
            sim.outputs.projected_cycle_life
            or sim.outputs.capacity_fade_rate_pct_per_cycle
        )
    ):
        _log("\n── Step C-3: Cycle Life ──")
        try:
            cycle_life = _timed("C3_cycle_life", run_cycle_life, param, sim)
            _log(cycle_life.summary())
            _collect(cycle_life.warnings, "C-3")
        except Exception as exc:
            warns.append(f"[C-3] Cycle life failed: {exc}")

    result = SimulationResult(
        taufactor=tau_result,
        rate_capability=rate_cap,
        dcir=dcir,
        cycle_life=cycle_life,
        elapsed_s=time.perf_counter() - t0,
        step_times_s=step_times,
        warnings=warns,
    )
    _log(result.summary())
    return result
