"""
Maps a unit-hypercube vector (length N_SIM_DIMS) to a SimConfig-compatible
kwargs dict, matching the actual snake_case field names.

Dimension index table  (order is fixed — never reorder):
  0  cell.anode.current_collector_thickness_um
  1  cell.cathode.material          categorical
  2  cell.cathode.thickness_um
  3  cell.cathode.porosity
  4  cell.separator.material        categorical
  5  cell.np_ratio
  6  cell.cell_area_cm2
  7  electrolyte.material           categorical  (drives cathode override + Vhi range)
  8  electrolyte.temperature_K
  9  cycling.voltage_cutoff_low_V   range depends on cathode family
 10  cycling.voltage_cutoff_high_V  range depends on cathode family
 11  rate_capability.charge_c_rate
 12  dcir_pulse.soc_point
 13  dcir_pulse.pulse_c_rate
 14  dcir_pulse.pulse_duration_s
 15  cycle_life.charge_c_rate
 16  cycle_life.discharge_c_rate
 17  cycle_life.n_cycles            integer
 18  cycle_life.end_of_life_retention
 19  taufactor.pore_threshold
 20  taufactor.electronic_threshold
 21  taufactor.direction            categorical {z, all}
 22  pybamm.model                   categorical {SPM, SPMe, DFN}
"""

from __future__ import annotations
import numpy as np
from ._primitives import _cont, _cat, _bool, _int

N_SIM_DIMS: int = 23

# DB keys confirmed from materialsdb.yml
_HV_CATHODES = ["nmc811", "nmc622", "nmc532", "nmc111", "nca", "lco"]
_ALL_CATHODES = _HV_CATHODES + ["lfp"]
_ELECTROLYTES = [
    "LiPF6_EC_DMC_1M",
    "LiPF6_EC_DEC_1M",
    "LiPF6_EC_EMC_3_7_1M",
    "LiPF6_FEC_DMC_1M",
    "LiFSI_DME_1M",
]
_ETHER_ELECTROLYTES = frozenset(["LiFSI_DME_1M"])
_SEPARATORS = ["Celgard2325", "Celgard2500", "Celgard3501", "ceramic_PP"]
_DIRECTIONS = ["z", "all"]
_PYBAMM_MODELS = ["DFN"]


def map_sim_config(u: np.ndarray) -> dict:
    """
    Map a unit vector u of length N_SIM_DIMS to a dict accepted by
    SimConfig.model_validate().

    Args:
        u:  1-D array, length N_SIM_DIMS, values in [0, 1].

    Returns:
        dict ready for ``SimConfig.model_validate(d)``.
    """
    if len(u) != N_SIM_DIMS:
        raise ValueError(f"Expected u of length {N_SIM_DIMS}, got {len(u)}")

    i = iter(u)

    def take() -> float:
        return float(next(i))

    # ── Cell ──────────────────────────────────────────────────────────────────
    anode_cc_t = _cont(take(), 6.0, 14.0)
    cathode_key_raw = _cat(take(), _ALL_CATHODES)  # may be overridden below
    cathode_thickness = _cont(take(), 40.0, 120.0)
    cathode_porosity = _cont(take(), 0.25, 0.45)
    separator_key = _cat(take(), _SEPARATORS)
    np_ratio = _cont(take(), 0.85, 1.40)
    cell_area_cm2 = _cont(take(), 1.0, 50.0)

    # ── Electrolyte ───────────────────────────────────────────────────────────
    electrolyte_key = _cat(take(), _ELECTROLYTES)
    temperature_K = _cont(take(), 268.0, 323.0)  # −5 °C … +50 °C

    # ── Ether constraint: LiFSIDME1M → cathode must be lfp ───────────────────
    if electrolyte_key in _ETHER_ELECTROLYTES:
        cathode_key = "lfp"
        vhi_lo, vhi_hi = 3.20, 3.65
        vlo_lo, vlo_hi = 2.00, 2.80
    else:
        cathode_key = cathode_key_raw
        if cathode_key == "lfp":
            vhi_lo, vhi_hi = 3.20, 3.65
            vlo_lo, vlo_hi = 2.00, 2.80
        else:  # high-voltage: NMC / NCA / LCO
            vhi_lo, vhi_hi = 4.00, 4.35
            vlo_lo, vlo_hi = 2.50, 3.00

    vlo = _cont(take(), vlo_lo, vlo_hi)
    vhi_min = max(vhi_lo, vlo + 0.20)  # ensure voltage_cutoff_high_V > low with minimum 200mV window
    # Add buffer to avoid edge cases and ensure stable voltage ranges
    vhi_buffer = min(0.3, vhi_hi - vhi_min)  # Don't exceed upper bound
    vhi_range_start = vhi_min + vhi_buffer
    vhi = _cont(take(), vhi_range_start, vhi_hi)

    # Ensure reasonable absolute voltage values
    if vlo < 2.5:  # Minimum reasonable low voltage
        vlo = 2.5
    if vhi > 4.3:  # Maximum reasonable high voltage for most chemistries
        vhi = 4.3

    # ── Rate capability protocol ──────────────────────────────────────────────
    rc_charge_rate = _cont(take(), 0.10, 0.50)

    # ── DCIR protocol ─────────────────────────────────────────────────────────
    dcir_soc = _cont(take(), 0.10, 0.90)
    dcir_pulse_rate = _cont(take(), 0.5, 5.0)
    dcir_duration = _cont(take(), 2.0, 30.0)

    # ── Cycle life protocol ───────────────────────────────────────────────────
    cl_charge_rate = _cont(take(), 0.20, 1.00)
    cl_discharge_rate = _cont(take(), 0.20, 1.00)
    cl_n_cycles = _int(take(), 50, 500)
    cl_eol = _cont(take(), 0.70, 0.85)

    # ── TauFactor ─────────────────────────────────────────────────────────────
    tf_pore_thresh = _cont(take(), 0.30, 0.60)
    tf_elec_thresh = _cont(take(), 0.20, 0.40)
    tf_direction = _cat(take(), _DIRECTIONS)

    # ── PyBaMM ────────────────────────────────────────────────────────────────
    pybamm_model = _cat(take(), _PYBAMM_MODELS)

    return dict(
        cell=dict(
            anode=dict(current_collector_thickness_um=anode_cc_t),
            cathode=dict(
                material=cathode_key,
                thickness_um=cathode_thickness,
                porosity=cathode_porosity,
            ),
            separator=dict(material=separator_key),
            np_ratio=np_ratio,
            cell_area_cm2=cell_area_cm2,
        ),
        electrolyte=dict(
            material=electrolyte_key,
            temperature_K=temperature_K,
        ),
        cycling=dict(
            voltage_cutoff_low_V=vlo,
            voltage_cutoff_high_V=vhi,
            protocols=[
                dict(
                    name="rate_capability",
                    enabled=True,
                    c_rates=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                    charge_c_rate=rc_charge_rate,
                    rest_time_s=300.0,
                ),
                dict(
                    name="dcir_pulse",
                    enabled=True,
                    soc_point=dcir_soc,
                    pulse_c_rate=dcir_pulse_rate,
                    pulse_duration_s=dcir_duration,
                ),
                dict(
                    name="cycle_life",
                    enabled=True,
                    charge_c_rate=cl_charge_rate,
                    discharge_c_rate=cl_discharge_rate,
                    n_cycles=cl_n_cycles,
                    end_of_life_retention=cl_eol,
                    record_every_n_cycles=5,
                ),
            ],
        ),
        taufactor=dict(
            enabled=True,
            pore_threshold=tf_pore_thresh,
            electronic_threshold=tf_elec_thresh,
            direction=tf_direction,
        ),
        pybamm=dict(
            enabled=True,
            model=pybamm_model,
            solver="IDAKLUSolver",
            atol=1e-6,
            rtol=1e-6,
            sei_model="reaction_limited",
        ),
        outputs=dict(
            tortuosity_factor=True,
            electrode_porosity=True,
            effective_diffusivity=True,
            bruggeman_exponent=True,
            nominal_capacity_mAh_cm2=True,
            energy_density_mWh_cm2=True,
            initial_dcir_mOhm_cm2=True,
            projected_cycle_life=True,
            capacity_fade_rate_pct_per_cycle=True,
        ),
    )
