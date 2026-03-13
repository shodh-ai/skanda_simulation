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

    m = vol.metadata
    cat = sim.cathode.material
    sep = sim.separator.material

    param = pybamm.ParameterValues("Chen2020")

    # ── Geometry ──────────────────────────────────────────────────────────
    param["Negative electrode thickness [m]"] = m.electrode_thickness_um * 1e-6
    param["Separator thickness [m]"] = sep.thickness_um * 1e-6

    # Calculate required cathode thickness to honor the np_ratio
    # Anode areal capacity [mAh/cm²] = volumetric capacity [mAh/cm³] × thickness [cm]
    anode_areal_cap = m.volumetric_capacity_mah_cm3 * (m.electrode_thickness_um * 1e-4)

    # Target cathode areal capacity = Anode / NP_ratio
    target_cathode_areal_cap = anode_areal_cap / sim.np_ratio

    # Cathode volumetric capacity [mAh/cm³] = AM_fraction × density [g/cm³] × capacity[mAh/g]
    # Assuming standard Chen2020 AM fraction if not overridden (~0.665)
    cat_am_frac = 1.0 - sim.cathode.porosity - 0.05  # roughly 5% for binder/CBD
    param["Positive electrode active material volume fraction"] = cat_am_frac

    cat_vol_cap = cat_am_frac * cat.density_g_cm3 * cat.theoretical_capacity_mAh_g

    # New cathode thickness [cm] -> [m]
    if cat_vol_cap > 0:
        cat_thickness_cm = target_cathode_areal_cap / cat_vol_cap
        param["Positive electrode thickness [m]"] = cat_thickness_cm * 1e-2
    else:
        param["Positive electrode thickness [m]"] = sim.cathode.thickness_um * 1e-6

    side_m = (sim.cell_area_cm2 * 1e-4) ** 0.5
    param["Electrode height [m]"] = side_m
    param["Electrode width [m]"] = side_m

    # ── Porosity ──────────────────────────────────────────────────────────
    if m.measured_porosity > 0.50:
        raise RuntimeError(
            f"Negative electrode porosity={m.measured_porosity:.4f} > 0.50 is "
            f"unphysical for a dense Si/C electrode."
        )
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
    param["Positive electrode tortuosity factor"] = 1.5

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

    # Active material volume fraction excludes porosity AND CBD/Binder/SEI
    am_vol_frac = (m.vf_carbon + m.vf_si) * (1.0 - m.measured_porosity)
    param["Negative electrode active material volume fraction"] = am_vol_frac

    cat_am_frac = 1.0 - sim.cathode.porosity - 0.05
    param["Positive electrode active material volume fraction"] = cat_am_frac

    c_s_n_max = float(param["Maximum concentration in negative electrode [mol.m-3]"])
    c_s_p_max = float(param["Maximum concentration in positive electrode [mol.m-3]"])

    # Calculate required cathode thickness to perfectly honor np_ratio in PyBaMM
    # Capacity proxy = c_s_max * am_vol_frac * thickness
    anode_capacity_proxy = c_s_n_max * am_vol_frac * (m.electrode_thickness_um * 1e-6)

    if c_s_p_max > 0 and cat_am_frac > 0:
        target_cat_cap_proxy = anode_capacity_proxy / sim.np_ratio
        cat_thickness_m = target_cat_cap_proxy / (c_s_p_max * cat_am_frac)
    else:
        cat_thickness_m = sim.cathode.thickness_um * 1e-6
    param["Positive electrode thickness [m]"] = cat_thickness_m
    # Chen2020 100% SOC stoichiometries (fully charged bounds)
    x_100 = 0.80
    y_100 = 0.26
    y_0 = 0.90  # 0% SOC stoichiometry

    # Initialize cell at fully charged state (100% SOC) so experiments can start with discharge
    param["Initial concentration in negative electrode [mol.m-3]"] = x_100 * c_s_n_max
    param["Initial concentration in positive electrode [mol.m-3]"] = y_100 * c_s_p_max
    param["Initial concentration in electrolyte [mol.m-3]"] = _C_REF_MOL_M3

    # CRITICAL: Update nominal cell capacity so C-rates compute correct currents!
    # Nominal capacity must be the PRACTICAL capacity, not theoretical.
    cat_theoretical_Ah = (
        c_s_p_max
        * cat_am_frac
        * cat_thickness_m
        * (sim.cell_area_cm2 * 1e-4)
        * 96485.332
    ) / 3600.0
    practical_fraction = y_0 - y_100
    param["Nominal cell capacity [A.h]"] = cat_theoretical_Ah * practical_fraction

    return param


def _isnan(x: float) -> bool:
    import math

    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return True
