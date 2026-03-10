"""
Transport expressions for 1M LiFSI in DME (dimethoxyethane).

Ether-based electrolyte. Excellent ionic conductivity and low viscosity.
Compatible ONLY with low-voltage cathodes (LFP, ≤ 4.0V vs Li/Li+).

HARD CONSTRAINT enforced in resolve_simulation():
    Do NOT pair with NMC, NCA, LCO, or any cathode with voltage_max > 4.0V.
    DME oxidises above ~4.0V. Standard PyBaMM does not model ether oxidation.
    Pairing with high-voltage cathodes will produce unphysically optimistic
    cycle life and capacity predictions.

Source: Wan et al., Adv. Energy Mater. 10 (2020) 2000791.
        Onsager-Stefan-Maxwell transport model from Pesko et al. 2018.

Status: STUB — Wan 2020 values at 298.15 K, 1M used as constants.
        Temperature dependence (strong Arrhenius, low Tg of DME) to be added.
"""

from __future__ import annotations


# Ether electrolytes have notably different transport vs carbonate:
#   - Lower viscosity → higher D and higher κ at moderate c
#   - Stronger concentration dependence (ion pairing at > 1M)
#   - t⁺ measured by Hittorf method (Wan 2020, Table 1)


def kappa(c_mol_m3: float, T_K: float) -> float:
    """
    LiFSI/DME conductivity — peaks around 1.5M, high at 1M.
    Wan 2020: κ ≈ 1.6 S/m at 1M, 25°C.
    """
    return 1.60  # S/m


def D(c_mol_m3: float, T_K: float) -> float:
    """
    LiFSI/DME diffusivity — DME low viscosity gives fast transport.
    Wan 2020: D ≈ 6.0×10⁻¹⁰ m²/s at 1M, 25°C.
    """
    return 6.00e-10  # m²/s


def t_plus(c_mol_m3: float, T_K: float) -> float:
    """
    LiFSI/DME transference number (Hittorf, Wan 2020 Table 1).
    Higher than carbonate electrolytes — DME solvates Li⁺ weakly.
    """
    return 0.46


def thermodynamic_factor(c_mol_m3: float, T_K: float) -> float:
    """
    Dilute-limit approximation. LiFSI forms ion pairs at > 1.5M
    (Pesko 2018) — this simplification is only valid near 1M.
    """
    return 1.0
