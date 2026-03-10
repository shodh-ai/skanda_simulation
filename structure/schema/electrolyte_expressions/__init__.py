"""
Electrolyte transport expression modules.

Each file in this package exposes four callables:

    kappa(c_mol_m3: float, T_K: float) -> float
        Ionic conductivity in S/m.
        c_mol_m3: Li⁺ concentration in mol/m³ (1M = 1000 mol/m³)
        T_K:      temperature in Kelvin

    D(c_mol_m3: float, T_K: float) -> float
        Li⁺ diffusivity in m²/s.

    t_plus(c_mol_m3: float, T_K: float) -> float
        Li⁺ transference number (Hittorf convention, dimensionless).

    thermodynamic_factor(c_mol_m3: float, T_K: float) -> float
        Activity correction = 1 + d(ln f±)/d(ln c).
        Used in concentrated-solution theory (Doyle, Fuller, Newman 1993).
        Equals 1.0 in the dilute limit.

All functions must accept scalar float inputs and return scalar floats.
PyBaMM will call them on numpy arrays element-wise via its expression graph —
implementations must also be compatible with numpy array inputs if used
directly in PyBaMM parameter sets.

Reference concentration:   1000 mol/m³  (= 1.0 mol/L = 1.0 M)
Reference temperature:     298.15 K     (= 25°C)
"""
