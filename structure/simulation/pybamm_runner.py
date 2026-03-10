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
    opts: dict = {}
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
    return cls(atol=sim.pybamm.atol, rtol=sim.pybamm.rtol)


def _solution_is_valid(sol: "pybamm.Solution") -> bool:
    """
    Return False when PyBaMM logged an error but didn't raise.
    Catches two failure modes:
      1. Empty solution — no timepoints at all
      2. Rest-only solution — experiment ran rest step but failed on
         the discharge/charge step, returning I≈0 throughout
    """
    try:
        t = sol["Time [s]"].entries
        if len(t) <= 1 or float(t[-1]) <= 0.0:
            return False
        I = sol["Current [A]"].entries
        # If max |I| is effectively zero, only the rest step ran
        return float(np.max(np.abs(I))) >= 1e-6
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
                raise ValueError(
                    "PyBaMM experiment returned an empty solution — "
                    "check parameter set and voltage window."
                )

            I = sol["Current [A]"].entries
            t = sol["Time [s]"].entries
            V = sol["Terminal voltage [V]"].entries

            Q_mAh_cm2 = float(np.trapezoid(np.abs(I), t)) / 3600.0 * 1000.0 / area_cm2
            E_mWh_cm2 = (
                float(np.trapezoid(V * np.abs(I), t)) / 3600.0 * 1000.0 / area_cm2
            )
            capacities.append(Q_mAh_cm2)
            energy_densities.append(E_mWh_cm2)

        except Exception as exc:
            warns.append(f"C-rate {c_rate}C failed: {exc}")
            capacities.append(_NAN)
            energy_densities.append(_NAN)

    # Nominal = result closest to 0.2C
    c_arr = np.array(proto.c_rates)
    idx_nom = int(np.argmin(np.abs(c_arr - 0.2)))
    Q_nom = capacities[idx_nom]
    E_nom = energy_densities[idx_nom]

    if math.isnan(Q_nom):
        warns.append(
            "Nominal capacity (C/5) simulation failed — "
            "check parameter set and voltage window."
        )

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
    warns: list[str] = []

    try:
        # Discharge time to reach target SOC from 100% at C/10
        discharge_s = int((1.0 - proto.soc_point) * 36000)

        steps = [
            f"Charge at C/10 until {sim.voltage_cutoff_high_V}V",
            "Rest for 60 seconds",
        ]
        if discharge_s > 0:
            steps.append(f"Discharge at C/10 for {discharge_s} seconds")
        steps += [
            "Rest for 30 seconds",
            f"Discharge at {proto.pulse_c_rate}C "
            f"for {int(proto.pulse_duration_s)} seconds",
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
        n_steps = len(steps)
        step_sol = sol.cycles[-1] if hasattr(sol, "cycles") else sol

        V_pulse = step_sol["Terminal voltage [V]"].entries
        I_pulse = step_sol["Current [A]"].entries

        V_before = float(V_pulse[0])
        V_after = float(V_pulse[-1])
        I_mean = float(np.mean(np.abs(I_pulse)))
        delta_V = V_before - V_after

        if I_mean < 1e-12:
            raise ValueError("Near-zero pulse current — check C-rate and capacity.")

        dcir = (delta_V / I_mean) * area_cm2 * 1000.0  # mΩ·cm²

        if dcir < 0:
            warns.append(
                f"Negative DCIR ({dcir:.2f} mΩ·cm²) — "
                f"voltage rose during discharge pulse (unexpected). "
                f"Check SOC targeting."
            )

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

    try:
        exp = pybamm.Experiment(
            [
                (
                    f"Charge at {proto.charge_c_rate}C until "
                    f"{sim.voltage_cutoff_high_V}V",
                    f"Discharge at {proto.discharge_c_rate}C until "
                    f"{sim.voltage_cutoff_low_V}V",
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

        cycle_numbers: list[int] = []
        capacities_mAh_cm2: list[float] = []

        all_cycles = sol.cycles if hasattr(sol, "cycles") else [sol]

        for i, cyc in enumerate(all_cycles):
            if i % proto.record_every_n_cycles == 0 or i == len(all_cycles) - 1:
                try:
                    I = cyc["Current [A]"].entries
                    t = cyc["Time [s]"].entries
                    mask = I > 0
                    if mask.sum() < 2:
                        mask = np.ones_like(I, dtype=bool)
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

        # ── Fit + projection ─────────────────────────────────────────────
        # Try linear first, then exponential.
        # Exponential: Q(n) = Q0 * exp(-k*n) → ln(Q/Q0) = -k*n
        # Use whichever gives a better R² on the recorded data.

        ns = np.array(cycle_numbers, dtype=float)
        qs = np.array(capacities_mAh_cm2, dtype=float)

        # Linear fit
        lin_coeffs = np.polyfit(ns, qs, 1)
        lin_slope = lin_coeffs[0]
        lin_q_pred = np.polyval(lin_coeffs, ns)
        lin_r2 = _r2(qs, lin_q_pred)
        fade_pct = abs(lin_slope) / Q_init * 100.0

        # Exponential fit (only valid when all Q > 0)
        exp_k = _NAN
        exp_r2 = -1.0
        exp_q_pred = None
        if np.all(qs > 0):
            try:
                log_qs = np.log(qs / Q_init)
                exp_coeffs = np.polyfit(ns, log_qs, 1)
                exp_k = -exp_coeffs[0]  # decay rate per cycle
                exp_q_pred = Q_init * np.exp(-exp_k * ns)
                exp_r2 = _r2(qs, exp_q_pred)
            except Exception:
                pass

        # Choose better fit
        use_exp = (not math.isnan(exp_k)) and (exp_r2 > lin_r2 + 0.01)

        if use_exp:
            warns.append(
                f"Exponential fit (R²={exp_r2:.4f}) chosen over "
                f"linear (R²={lin_r2:.4f}) for cycle life projection."
            )

        # Project EOL
        projected: Optional[int] = None

        if eol_reached:
            # EOL already hit — interpolate from recorded data
            projected = _interpolate_eol(cycle_numbers, capacities_mAh_cm2, eol_Q)
            warns.append(
                f"EOL ({proto.end_of_life_retention:.0%} retention) reached "
                f"within {proto.n_cycles} cycles."
            )
        else:
            # EOL not reached — extrapolate from chosen fit
            if use_exp and not math.isnan(exp_k) and exp_k > 0:
                # Q(n) = Q0 * exp(-k*n) = eol_Q → n = -ln(eol_frac) / k
                eol_frac = proto.end_of_life_retention
                projected = int(-math.log(eol_frac) / exp_k)
            elif lin_slope < 0:
                projected = int((Q_init - eol_Q) / abs(lin_slope))
            else:
                warns.append(
                    "Capacity is not declining — cannot project EOL. "
                    "Consider running more cycles."
                )

            if projected is not None:
                extrap_factor = projected / proto.n_cycles
                warns.append(
                    f"EOL not reached in {proto.n_cycles} cycles — "
                    f"projected {projected} cycles by "
                    f"{'exponential' if use_exp else 'linear'} extrapolation "
                    f"({extrap_factor:.1f}× beyond simulated range)."
                )
                if extrap_factor > 5.0:
                    warns.append(
                        f"Extrapolation factor {extrap_factor:.1f}× is large — "
                        f"consider increasing n_cycles for a more reliable projection."
                    )

    except Exception as exc:
        warns.append(f"Cycle life failed: {exc}")
        cycle_numbers = []
        capacities_mAh_cm2 = []
        Q_init = _NAN
        Q_final = _NAN
        retention = _NAN
        fade_pct = _NAN
        projected = None

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
