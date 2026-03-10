"""
Transport expressions for 1M LiPF6 in EC:EMC 3:7 w/w (LP57).

Source: Petibon et al., J. Electrochem. Soc. 163(7) A1332 (2016).
        Schmalstieg et al. parameter set used in PyBaMM Si cells.

Status: STUB — constant values at 298.15 K used.
        Concentration and temperature dependence to be fitted from
        Petibon 2016 Table 2 extended dataset when available.
"""

from __future__ import annotations


def kappa(c_mol_m3: float, T_K: float) -> float:
    """LP57 conductivity ~0.9 S/m at 1M, 25°C (slightly lower than LP30)."""
    return 0.90


def D(c_mol_m3: float, T_K: float) -> float:
    """LP57 diffusivity at 1M, 25°C — EMC reduces viscosity vs DMC."""
    return 3.00e-10  # m²/s


def t_plus(c_mol_m3: float, T_K: float) -> float:
    """LP57 transference number ~0.38 (similar to LP30)."""
    return 0.38


def thermodynamic_factor(c_mol_m3: float, T_K: float) -> float:
    return 1.0
