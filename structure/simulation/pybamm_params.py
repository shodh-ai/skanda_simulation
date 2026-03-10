"""
Step B — Assemble a PyBaMM ParameterValues from MicrostructureVolume + ResolvedSimulation.

Base: Chen2020 parameter set.
Overrides — anode geometry + transport from vol.metadata (set at generation time).
Overrides — cathode, separator, electrolyte from ResolvedSimulation.
τ and β overridden from TauFactorResult if available.

All fields used here exist in either:
    vol.metadata  (VolumeMetadata — generation + DB properties baked in)
    sim.*         (ResolvedSimulation — cell-level materials)
No fields are invented — all trace to MaterialsDB or GenConfig.
"""

from __future__ import annotations

from typing import Optional
import pybamm
from structure.data import MicrostructureVolume, TauFactorResult
from structure.schema import ResolvedSimulation

_C_REF_MOL_M3 = 1000.0  # 1.0 M reference concentration


def build_parameter_set(
    vol: MicrostructureVolume,
    sim: ResolvedSimulation,
    tau_result: Optional[TauFactorResult] = None,
) -> "pybamm.ParameterValues":
    """
    Build PyBaMM ParameterValues from vol + sim.

    Override policy — ONLY geometry and transport:
        SAFE to override  : thicknesses, porosities, tortuosity, Bruggeman,
                            diffusivity, conductivity, electrolyte transport,
                            temperature, cell area, voltage window.
        NOT overridden    : max concentration, particle radius, AM volume
                            fraction, OCV curves, exchange current densities.
                            These are all coupled to Chen2020's OCV polynomial
                            and DAE initial conditions — changing any one
                            without the others causes "Could not find
                            consistent states" at t=0.
    """
    try:
        import pybamm
    except ImportError:
        raise ImportError("pybamm is required. Install with: pip install pybamm")

    m = vol.metadata
    cat = sim.cathode.material
    sep = sim.separator.material

    param = pybamm.ParameterValues("Chen2020")

    # ── Geometry ──────────────────────────────────────────────────────────
    param["Negative electrode thickness [m]"] = m.electrode_thickness_um * 1e-6
    param["Positive electrode thickness [m]"] = sim.cathode.thickness_um * 1e-6
    param["Separator thickness [m]"] = sep.thickness_um * 1e-6

    side_m = (sim.cell_area_cm2 * 1e-4) ** 0.5
    param["Electrode height [m]"] = side_m
    param["Electrode width [m]"] = side_m

    # ── Porosity ──────────────────────────────────────────────────────────
    param["Negative electrode porosity"] = m.measured_porosity
    param["Positive electrode porosity"] = sim.cathode.porosity
    param["Separator porosity"] = sep.porosity

    # ── Tortuosity / Bruggeman ────────────────────────────────────────────
    # Ionic: use measured τ from TauFactor when available, else β=1.5
    if tau_result is not None and not _isnan(tau_result.tau_ionic):
        param["Negative electrode tortuosity factor"] = tau_result.tau_ionic
        param["Negative electrode Bruggeman coefficient (electrolyte)"] = (
            1.5
            if _isnan(tau_result.bruggeman_exponent)
            else tau_result.bruggeman_exponent
        )
    else:
        param["Negative electrode Bruggeman coefficient (electrolyte)"] = 1.5

    # Separator: µ-CT measured τ takes precedence (Lagadec 2016)
    param["Separator tortuosity factor"] = sep.tortuosity
    param["Separator Bruggeman coefficient (electrolyte)"] = sep.bruggeman_exponent
    param["Positive electrode Bruggeman coefficient (electrolyte)"] = 1.5

    # ── Anode transport — safe scalar overrides ────────────────────────────
    # Diffusivity and conductivity only — these don't affect initial conditions.
    # Particle radius, AM volume fraction, and max concentration are intentionally
    # left as Chen2020 defaults — they are coupled to the OCV polynomial and
    # DAE initial conditions. Overriding them independently causes
    # "Could not find consistent states" at t=0.
    param["Negative electrode diffusivity [m2.s-1]"] = m.carbon_li_diffusivity_m2_s
    param["Negative electrode conductivity [S.m-1]"] = (
        m.carbon_electrical_conductivity_S_m
    )

    # ── Cathode transport — safe scalar overrides only ─────────────────────
    param["Positive electrode diffusivity [m2.s-1]"] = cat.li_diffusivity_m2_s
    param["Positive electrode conductivity [S.m-1]"] = cat.electrical_conductivity_S_m

    # ── Electrolyte ────────────────────────────────────────────────────────
    param["Electrolyte conductivity [S.m-1]"] = sim.electrolyte.kappa(_C_REF_MOL_M3)
    param["Electrolyte diffusivity [m2.s-1]"] = sim.electrolyte.D(_C_REF_MOL_M3)
    param["Cation transference number"] = sim.electrolyte.t_plus(_C_REF_MOL_M3)
    param["Thermodynamic factor"] = sim.electrolyte.thermodynamic_factor(_C_REF_MOL_M3)

    # ── Temperature ───────────────────────────────────────────────────────
    param["Ambient temperature [K]"] = sim.electrolyte.temperature_K
    param["Initial temperature [K]"] = sim.electrolyte.temperature_K

    # ── Voltage window ────────────────────────────────────────────────────
    param["Lower voltage cut-off [V]"] = sim.voltage_cutoff_low_V
    param["Upper voltage cut-off [V]"] = sim.voltage_cutoff_high_V

    return param


def _isnan(x: float) -> bool:
    import math

    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return True
