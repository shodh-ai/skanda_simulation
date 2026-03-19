"""
Step C — Run PyBaMM protocols.

C-1  run_rate_capability  → RateCapabilityResult
C-2  run_dcir_pulse       → DCIRResult
C-3  run_cycle_life       → CycleLifeResult

Each sub-step is independent — an exception in C-2 is caught and stored
as a warning; C-3 still runs.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import contextlib
from structure.schema import ResolvedSimulation
from structure.data import RateCapabilityResult, DCIRResult, CycleLifeResult
import pybamm

_NAN = float("nan")

_SEI_MAP: dict[str, str] = {
    "none": "none",
    "reaction_limited": "reaction limited",
    "solvent_diffusion": "solvent-diffusion limited",
}


def _make_model(sim: ResolvedSimulation) -> "pybamm.lithium_ion.BaseModel":
    opts: dict = {
        "particle mechanics": ("swelling and cracking", "none"),
        "SEI on cracks": "true",
        "loss of active material": ("stress-driven", "none"),
        "lithium plating": "none",
    }
    sei_key = sim.pybamm.sei_model
    if sei_key != "none":
        pybamm_sei = _SEI_MAP.get(sei_key)
        if pybamm_sei is None:
            raise ValueError(
                f"sei_model '{sei_key}' has no PyBaMM mapping. "
                f"Valid config values: {list(_SEI_MAP)}"
            )
        opts["SEI"] = pybamm_sei

    cls = {
        "SPM": pybamm.lithium_ion.SPM,
        "SPMe": pybamm.lithium_ion.SPMe,
        "DFN": pybamm.lithium_ion.DFN,
    }[sim.pybamm.model]
    return cls(options=opts or None)


def _make_solver(sim: ResolvedSimulation) -> "pybamm.BaseSolver":
    cls = {"IDAKLUSolver": pybamm.IDAKLUSolver, "CasadiSolver": pybamm.CasadiSolver}[
        sim.pybamm.solver
    ]

    # Use more robust solver settings for difficult cases
    solver_kwargs = {
        "atol": sim.pybamm.atol,
        "rtol": sim.pybamm.rtol,
        "max_steps": 10000,  # Increase from default
        "max_h": 100,  # Maximum step size
        "min_h": 1e-10,  # Minimum step size
        "suppress_solve": True,  # Suppress some solver output
    }

    # For IDAKLU, add additional robustness parameters
    if sim.pybamm.solver == "IDAKLUSolver":
        solver_kwargs.update({
            "linear_solver": "KLU",
            "max_err_test_fails": 10,  # More attempts before failure
            "max_nonlin_iters": 6,  # More nonlinear iterations
            "max_setups": 10,
        })

    return cls(**solver_kwargs)


def _solution_is_valid(sol: "pybamm.Solution") -> bool:
    """
    Return False when PyBaMM logged an error but didn't raise.
    Catches multiple failure modes:
      1. Empty solution — no timepoints at all
      2. Rest-only solution — experiment ran rest step but failed on
         the discharge/charge step, returning I≈0 throughout
      3. Unphysical voltage values
      4. Very short simulations
    """
    try:
        t = sol["Time [s]"].entries
        if len(t) <= 1 or float(t[-1]) <= 0.0:
            return False

        I = sol["Current [A]"].entries
        # If max |I| is effectively zero, only the rest step ran
        if float(np.max(np.abs(I))) < 1e-6:
            return False

        # Check for reasonable duration
        if float(t[-1]) < 1.0:  # Less than 1 second of simulation
            return False

        # Check voltage is physical (not negative or zero)
        try:
            V = sol["Terminal voltage [V]"].entries
            if len(V) > 0:
                max_V = float(np.max(V))
                min_V = float(np.min(V))
                if max_V <= 0.0 or min_V <= 0.0:
                    return False
        except Exception:
            pass  # Voltage variable might not exist in all solutions

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# C-1: Rate capability
# ---------------------------------------------------------------------------


def run_rate_capability(
    param: "pybamm.ParameterValues",
    sim: ResolvedSimulation,
) -> RateCapabilityResult:
    proto = sim.rate_capability
    area_cm2 = sim.cell_area_cm2
    warns: list[str] = []

    capacities: list[float] = []
    energy_densities: list[float] = []

    for c_rate in proto.c_rates:
        try:
            exp = pybamm.Experiment(
                [
                    (
                        f"Rest for {int(proto.rest_time_s)} seconds",
                        f"Discharge at {c_rate}C until {sim.voltage_cutoff_low_V}V",
                    )
                ]
            )
            psim = pybamm.Simulation(
                _make_model(sim),
                parameter_values=param,
                experiment=exp,
                solver=_make_solver(sim),
            )
            psim.solve()
            sol = psim.solution

            if not _solution_is_valid(sol):
                raise ValueError("PyBaMM experiment returned an empty solution.")

            # The discharge is the last step in the cycle — extract only this window
            if hasattr(sol, "cycles") and len(sol.cycles) > 0:
                step_sol = sol.cycles[-1].steps[-1]
            elif hasattr(sol, "sub_solutions") and len(sol.sub_solutions) > 0:
                step_sol = sol.sub_solutions[-1]
            else:
                step_sol = sol

            I = step_sol["Current [A]"].entries
            t = step_sol["Time [s]"].entries
            V = step_sol["Terminal voltage [V]"].entries

            Q_mAh_cm2 = float(np.trapezoid(np.abs(I), t)) / 3600.0 * 1000.0 / area_cm2
            E_mWh_cm2 = (
                float(np.trapezoid(V * np.abs(I), t)) / 3600.0 * 1000.0 / area_cm2
            )

            capacities.append(Q_mAh_cm2)
            energy_densities.append(E_mWh_cm2)

        except Exception as exc:
            exc_str = str(exc)
            # Provide more specific warnings for common failure modes
            if "Maximum voltage" in exc_str and "non-positive" in exc_str:
                warns.append(f"C-rate {c_rate}C failed: voltage parameters invalid - {exc}")
            elif "IDA_CONV_FAIL" in exc_str or "convergence" in exc_str.lower():
                warns.append(f"C-rate {c_rate}C failed: solver convergence issue - {exc}")
            elif "consistent states" in exc_str.lower():
                warns.append(f"C-rate {c_rate}C failed: inconsistent initial states - {exc}")
            else:
                warns.append(f"C-rate {c_rate}C failed: {exc}")
            capacities.append(_NAN)
            energy_densities.append(_NAN)

    # Nominal = result closest to 0.2C
    c_arr = np.array(proto.c_rates)
    idx_nom = int(np.argmin(np.abs(c_arr - 0.2)))
    Q_nom = capacities[idx_nom]
    E_nom = energy_densities[idx_nom]

    if math.isnan(Q_nom):
        warns.append("Nominal capacity (C/5) simulation failed.")

    return RateCapabilityResult(
        c_rates=proto.c_rates,
        capacities_mAh_cm2=capacities,
        energy_densities_mWh_cm2=energy_densities,
        nominal_capacity_mAh_cm2=Q_nom,
        energy_density_mWh_cm2=E_nom,
        warnings=warns,
    )


# ---------------------------------------------------------------------------
# C-2: DCIR pulse
# ---------------------------------------------------------------------------


def run_dcir_pulse(
    param: "pybamm.ParameterValues",
    sim: ResolvedSimulation,
) -> DCIRResult:
    proto = sim.dcir_pulse
    area_cm2 = sim.cell_area_cm2
    voltage_window = sim.voltage_cutoff_high_V - sim.voltage_cutoff_low_V
    warns: list[str] = []

    dcir = _NAN
    delta_V = _NAN

    try:
        # Cell is at 100% SOC. Discharge to target SOC at C/10.
        discharge_s = int((1.0 - proto.soc_point) * 36000)

        steps = []
        if discharge_s > 0:
            steps.append(f"Discharge at C/10 for {discharge_s} seconds")

        steps += [
            "Rest for 30 seconds",
            f"Discharge at {proto.pulse_c_rate}C for {int(proto.pulse_duration_s)} seconds",
        ]

        psim = pybamm.Simulation(
            _make_model(sim),
            parameter_values=param,
            experiment=pybamm.Experiment([tuple(steps)]),
            solver=_make_solver(sim),
        )
        psim.solve()
        sol = psim.solution

        if not _solution_is_valid(sol):
            raise ValueError("PyBaMM experiment returned an empty solution.")

        # Pulse step is the last step — extract only that window
        if hasattr(sol, "cycles") and len(sol.cycles) > 0:
            stepsol = sol.cycles[-1].steps[-1]
        elif hasattr(sol, "sub_solutions") and len(sol.sub_solutions) > 0:
            stepsol = sol.sub_solutions[-1]
        else:
            raise ValueError("Could not locate step solutions in PyBaMM Solution.")

        Vpulse = stepsol["Terminal voltage [V]"].entries
        Vbefore = float(Vpulse[0])
        Vafter = float(Vpulse[-1])
        delta_V = Vbefore - Vafter

        if delta_V > voltage_window:
            warns.append(f"delta_V={delta_V*1000:.2f} mV exceeds voltage window.")
            dcir = _NAN
            delta_V = _NAN
        else:
            Ipulse = stepsol["Current [A]"].entries
            Imean = float(np.mean(np.abs(Ipulse)))
            if Imean < 1e-12:
                raise ValueError("Near-zero pulse current.")
            dcir = delta_V / Imean * area_cm2 * 1000.0

            if dcir < 0:
                warns.append("Negative DCIR — voltage rose during discharge pulse.")

    except Exception as exc:
        warns.append(f"DCIR pulse failed: {exc}")
        dcir = _NAN
        delta_V = _NAN

    return DCIRResult(
        dcir_mOhm_cm2=dcir,
        soc_point=proto.soc_point,
        pulse_c_rate=proto.pulse_c_rate,
        pulse_duration_s=proto.pulse_duration_s,
        delta_V_mV=_NAN if math.isnan(delta_V) else delta_V * 1000.0,
        warnings=warns,
    )


# ---------------------------------------------------------------------------
# C-3: Cycle life
# ---------------------------------------------------------------------------


def run_cycle_life(
    param: "pybamm.ParameterValues",
    sim: ResolvedSimulation,
) -> CycleLifeResult:
    proto = sim.cycle_life
    area_cm2 = sim.cell_area_cm2
    warns: list[str] = []

    cycle_numbers = []
    capacities_mAh_cm2 = []
    Q_init = _NAN
    Q_final = _NAN
    retention = _NAN
    fade_pct = _NAN
    projected = None

    try:
        # We initialize at 100% SOC, so start with discharge.
        exp = pybamm.Experiment(
            [
                (
                    f"Discharge at {proto.discharge_c_rate}C until {sim.voltage_cutoff_low_V}V",
                    f"Charge at {proto.charge_c_rate}C until {sim.voltage_cutoff_high_V}V",
                    f"Hold at {sim.voltage_cutoff_high_V}V until C/20",
                )
            ]
            * proto.n_cycles
        )
        psim = pybamm.Simulation(
            _make_model(sim),
            parameter_values=param,
            experiment=exp,
            solver=_make_solver(sim),
        )
        psim.solve()
        sol = psim.solution

        if not _solution_is_valid(sol):
            raise ValueError("PyBaMM experiment returned an empty solution.")

        all_cycles = (
            sol.cycles if hasattr(sol, "cycles") and len(sol.cycles) > 0 else [sol]
        )

        for i, cyc in enumerate(all_cycles):
            if i % proto.record_every_n_cycles == 0 or i == len(all_cycles) - 1:
                try:
                    # In a cycle starting with discharge, the discharge is the FIRST step.
                    if hasattr(cyc, "steps") and len(cyc.steps) > 0:
                        step_sol = cyc.steps[0]
                    else:
                        step_sol = cyc

                    try:
                        Q = (
                            float(
                                step_sol["Discharge capacity [A.h]"].entries[-1]
                                - step_sol["Discharge capacity [A.h]"].entries[0]
                            )
                            * 1000.0
                            / area_cm2
                        )
                    except Exception:
                        # Fallback
                        I = step_sol["Current [A]"].entries
                        t = step_sol["Time[s]"].entries
                        mask = I > 0  # Discharge is positive current in PyBaMM
                        if mask.sum() < 2:
                            continue
                        Q = (
                            float(np.trapezoid(np.abs(I[mask]), t[mask]))
                            / 3600.0
                            * 1000.0
                            / area_cm2
                        )

                    cycle_numbers.append(i + 1)
                    capacities_mAh_cm2.append(Q)
                except Exception as exc:
                    warns.append(f"Cycle {i+1} extraction failed: {exc}")

        if len(capacities_mAh_cm2) < 2:
            raise ValueError("Fewer than 2 cycles extracted — cannot fit fade.")

        Q_init = capacities_mAh_cm2[0]
        Q_final = capacities_mAh_cm2[-1]
        retention = Q_final / Q_init if Q_init > 0 else _NAN
        eol_Q = Q_init * proto.end_of_life_retention

        eol_reached = Q_final <= eol_Q

        ns = np.array(cycle_numbers, dtype=float)
        qs = np.array(capacities_mAh_cm2, dtype=float)

        # Robust endpoint-based fade rate calculation instead of a linear fit,
        # which can be skewed if capacity trajectories are highly nonlinear.
        if Q_init > 0 and proto.n_cycles > 0:
            fade_pct = ((Q_init - Q_final) / Q_init) / proto.n_cycles * 100.0
        else:
            fade_pct = _NAN

        if eol_reached:
            projected = _interpolate_eol(cycle_numbers, capacities_mAh_cm2, eol_Q)
        else:
            # Linear projection using the robust endpoint slope
            drop_per_cycle = (Q_init - Q_final) / proto.n_cycles
            if drop_per_cycle > 0:
                projected = int((Q_init - eol_Q) / drop_per_cycle)

    except Exception as exc:
        warns.append(f"Cycle life failed: {exc}")

    return CycleLifeResult(
        cycles_run=proto.n_cycles,
        cycle_numbers=cycle_numbers,
        capacities_mAh_cm2=capacities_mAh_cm2,
        projected_cycle_life=projected,
        capacity_fade_rate_pct_per_cycle=fade_pct,
        initial_capacity_mAh_cm2=Q_init,
        final_capacity_mAh_cm2=Q_final,
        retention_at_final_cycle=retention,
        warnings=warns,
    )


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _interpolate_eol(
    cycles: list[int],
    capacities: list[float],
    eol_Q: float,
) -> Optional[int]:
    """Linear interpolation between the two cycles bracketing EOL."""
    for i in range(len(capacities) - 1):
        if capacities[i] >= eol_Q >= capacities[i + 1]:
            n0, n1 = cycles[i], cycles[i + 1]
            q0, q1 = capacities[i], capacities[i + 1]
            if abs(q1 - q0) < 1e-12:
                return n0
            frac = (q0 - eol_Q) / (q0 - q1)
            return int(n0 + frac * (n1 - n0))
    return cycles[-1]
