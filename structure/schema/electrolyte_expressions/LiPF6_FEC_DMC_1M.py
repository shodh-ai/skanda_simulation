"""
Transport expressions for 1M LiPF6 in FEC:DMC 1:4 v/v.

Fluoroethylene carbonate (FEC) additive electrolyte standard for Si anodes.
High FEC content stabilises the SEI on Si by forming LiF-rich films.

Source: Petibon et al., J. Electrochem. Soc. 161(6) A867 (2014).
        Dose et al., ACS Appl. Mater. Interfaces 13 (2021) for FEC-rich variant.

Note: FEC content reduces ionic conductivity ~15% vs LP30 due to higher
viscosity of FEC (η ≈ 4.1 mPas at 25°C vs. 0.59 mPas for DMC).

Status: STUB — constant values at 298.15 K, 1M.
"""

from __future__ import annotations


def kappa(c_mol_m3: float, T_K: float) -> float:
    """FEC:DMC 1:4 conductivity ~0.85 S/m at 1M, 25°C (Petibon 2014)."""
    return 0.85


def D(c_mol_m3: float, T_K: float) -> float:
    """FEC:DMC diffusivity — slightly lower than LP30 due to FEC viscosity."""
    return 2.50e-10  # m²/s


def t_plus(c_mol_m3: float, T_K: float) -> float:
    return 0.37  # slightly lower due to FEC solvation shell effect


def thermodynamic_factor(c_mol_m3: float, T_K: float) -> float:
    return 1.0
