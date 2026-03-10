"""
Transport expressions for 1M LiPF6 in EC:DMC 1:1 v/v (LP30).

Source: Valoen & Reimers, J. Electrochem. Soc. 152(5) A882–A891 (2005).
        Equations 33, 36, 40.

Valid range:
    c : 0 – 3000 mol/m³  (0 – 3.0 M)
    T : 253 – 333 K       (−20°C to +60°C)

All functions use SI units throughout:
    c  → mol/m³
    T  → K
    κ  → S/m
    D  → m²/s
    t⁺ → dimensionless
    Λ  → dimensionless  (thermodynamic factor)
"""

from __future__ import annotations
import math


def kappa(c_mol_m3: float, T_K: float) -> float:
    """
    Valoen & Reimers 2005, Eq. 36. Returns κ in S/m.

    V&R gives κ [mS/cm] = c_L * inner^2  (c_L in mol/L)
    S/m = mS/cm × 0.1  →  κ [S/m] = 0.1 × c_L × inner^2
                                    = 1e-4 × c_mol_m3 × inner^2   ← leading factor is mol/m³
    The polynomial `inner` is evaluated at c_L = c_mol_m3 / 1000.
    """
    c_L = c_mol_m3 / 1000.0  # mol/L — for polynomial only
    inner = (
        -10.5
        + 0.0740 * T_K
        - 6.96e-5 * T_K**2
        + c_L * (0.668 - 0.0178 * T_K + 2.80e-5 * T_K**2)
        + c_L**2 * (0.494 - 8.86e-4 * T_K)
    )
    return 0.0 if inner <= 0.0 else 1e-4 * c_mol_m3 * inner**2


def D(c_mol_m3: float, T_K: float) -> float:
    """
    Valoen & Reimers 2005, Eq. 33. Returns D in m²/s.
    V&R gives D [cm²/s]; 1 cm²/s = 1e-4 m²/s.
    """
    c_L = c_mol_m3 / 1000.0
    denom = T_K - 229.0 - 5.0 * c_L
    if denom <= 0.0:
        return 2.80e-10
    exponent = -4.43 - 54.0 / denom - 0.22 * c_L
    return 1e-4 * 10.0**exponent  # 1e-4 converts cm²/s → m²/s


def t_plus(c_mol_m3: float, T_K: float) -> float:
    """Valoen & Reimers 2005, Eq. 40."""
    c_L = c_mol_m3 / 1000.0
    return 0.4492 - 4.9e-4 * (T_K - 294.0) - 0.2096 * c_L + 0.0012 * c_L**2


def thermodynamic_factor(c_mol_m3: float, T_K: float) -> float:
    """
    Valoen & Reimers 2005, Eq. 41.
    TDF = 1 + d(ln f±)/d(ln c)
    At 1M, 25°C → ~2.32 (not 1.0 — LiPF6 is a concentrated solution).
    Chen2020 uses a constant 1.0; this is more accurate but may differ
    from Chen2020 baseline. Does not affect DAE initial conditions (only
    electrolyte flux equations, which are zero at rest).
    """
    c_L = c_mol_m3 / 1000.0
    return (
        1.0
        + 0.601
        - 0.24 * c_L**0.5
        + 0.982 * (1.0 - 0.0052 * (T_K - 294.0)) * c_L**1.5
    )
