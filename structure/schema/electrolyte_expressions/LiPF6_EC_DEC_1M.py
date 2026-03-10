"""
Transport expressions for 1M LiPF6 in EC:DEC 1:1 v/v (LP40).

Source: Ecker et al., J. Electrochem. Soc. 162(9) A1836 (2015).
        Table 3 polynomial fits at 25°C.

Status: STUB — temperature dependence not implemented.
        At T ≠ 298.15K, Arrhenius scaling is applied using activation
        energies from Nyman et al. JES 157(11) A1236 (2010).
"""

from __future__ import annotations
import math

# Reference values at 298.15 K, 1000 mol/m³
_T_REF = 298.15
_C_REF = 1000.0  # mol/m³

# Activation energies (J/mol) — Nyman 2010
_EA_KAPPA = 17100.0
_EA_D = 17000.0
_R = 8.314


def kappa(c_mol_m3: float, T_K: float) -> float:
    """Ecker 2015 polynomial at T_ref, Arrhenius-scaled to T."""
    c = c_mol_m3 / 1000.0  # mol/L
    # Polynomial at 25°C (Ecker 2015 Table 3)
    k0 = -0.0915 + 1.7523 * c - 0.8674 * c**2 + 0.1514 * c**3
    k0 = max(k0, 0.0)
    arr = math.exp(-_EA_KAPPA / _R * (1.0 / T_K - 1.0 / _T_REF))
    return k0 * arr


def D(c_mol_m3: float, T_K: float) -> float:
    """Ecker 2015 diffusivity at T_ref, Arrhenius-scaled."""
    # Weak concentration dependence near 1M — use reference value
    D0 = 2.60e-10  # m²/s at 298.15K, 1M (Ecker 2015)
    arr = math.exp(-_EA_D / _R * (1.0 / T_K - 1.0 / _T_REF))
    return D0 * arr


def t_plus(c_mol_m3: float, T_K: float) -> float:
    """EC:DEC t⁺ from Nyman 2010 — slightly higher than EC:DMC."""
    return 0.40  # weakly concentration-dependent; held constant


def thermodynamic_factor(c_mol_m3: float, T_K: float) -> float:
    """Dilute-limit approximation — valid near 1M."""
    return 1.0
