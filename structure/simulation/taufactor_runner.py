"""
Step A — TauFactor tortuosity calculation.

Computes τ_ionic, D_eff, β (Bruggeman) from the pore_vf field.
Optionally computes τ_electronic from carbon_vf + cbd_vf.

Convention: D_eff = D_bulk × ε / τ  (TauFactor / PyBaMM convention).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from structure.data import MicrostructureVolume, TauFactorResult
from structure.schema import ResolvedSimulation
import taufactor as tf


def run_taufactor(
    vol: MicrostructureVolume,
    sim: ResolvedSimulation,
) -> TauFactorResult:
    """
    Run TauFactor on vol.pore_vf (ionic) and optionally
    (vol.carbon_vf + vol.cbd_vf) (electronic).

    Args:
        vol : MicrostructureVolume
        sim : ResolvedSimulation — reads sim.taufactor config

    Returns:
        TauFactorResult

    Raises:
        ImportError if taufactor is not installed.
    """

    cfg = sim.taufactor
    warns: list[str] = []

    # ── Ionic ─────────────────────────────────────────────────────────────
    pore_mask = vol.to_pore_mask(threshold=cfg.pore_threshold)
    epsilon_ionic = float(pore_mask.astype(np.float64).mean())

    if epsilon_ionic < 0.05:
        warns.append(
            f"Very low porosity ({epsilon_ionic:.3f}) — "
            f"τ may be unreliable or TauFactor may not converge."
        )
    if epsilon_ionic > 0.75:
        warns.append(
            f"Very high porosity ({epsilon_ionic:.3f}) — "
            f"Bruggeman exponent fit unreliable above ε=0.75."
        )

    tau_ionic, n_iter, residual, converged = _solve(
        mask=pore_mask,
        direction=cfg.direction,
        label="ionic",
        warns=warns,
    )

    # D_eff = D_bulk × ε / τ
    c_ref = sim.electrolyte.material.salt_concentration_mol_L * 1000.0  # mol/m³
    D_bulk = sim.electrolyte.D(c_ref)
    D_eff = D_bulk * epsilon_ionic / tau_ionic

    # Bruggeman: τ = ε^(1-β) → β = 1 - ln(τ)/ln(ε)
    if 0.0 < epsilon_ionic < 1.0:
        try:
            bruggeman = 1.0 - math.log(tau_ionic) / math.log(epsilon_ionic)
        except (ValueError, ZeroDivisionError):
            bruggeman = float("nan")
            warns.append("Bruggeman exponent computation failed (log domain error).")
    else:
        bruggeman = float("nan")
        warns.append(f"Cannot compute Bruggeman exponent: ε={epsilon_ionic:.4f}.")

    # ── Electronic (optional) ─────────────────────────────────────────────
    tau_electronic: Optional[float] = None
    epsilon_electronic: Optional[float] = None

    elec_mask = (
        vol.carbon_vf.astype(np.float32) + vol.cbd_vf.astype(np.float32)
    ) >= cfg.electronic_threshold
    epsilon_electronic = float(elec_mask.astype(np.float64).mean())

    if epsilon_electronic < 0.01:
        warns.append(
            f"Electronic phase fraction very low ({epsilon_electronic:.4f}) — "
            f"electronic τ unreliable. Check electronic_threshold={cfg.electronic_threshold}."
        )
    else:
        tau_electronic, _, _, _ = _solve(
            mask=elec_mask,
            direction=cfg.direction,
            label="electronic",
            warns=warns,
        )

    return TauFactorResult(
        tau_ionic=tau_ionic,
        epsilon_ionic=epsilon_ionic,
        D_eff_ionic_m2_s=D_eff,
        bruggeman_exponent=bruggeman,
        tau_electronic=tau_electronic,
        epsilon_electronic=epsilon_electronic,
        converged=converged,
        n_iterations=n_iter,
        residual=residual,
        warnings=warns,
    )


def _solve(
    mask: np.ndarray,
    direction: str,
    label: str,
    warns: list[str],
) -> tuple[float, int, float, bool]:
    img = mask.astype(np.float32)

    try:
        if direction == "z":
            solver = tf.Solver(img, device="cpu")
            solver.solve()
            # solver.tau is a 0-dim or 1-element numpy array — unwrap safely
            tau_val = float(np.asarray(solver.tau).flat[0])
            n_iter = int(np.asarray(getattr(solver, "iter", 0)).flat[0])
            _crit = getattr(
                solver,
                "conv_crit",
                getattr(solver, "convergence_criterion", getattr(solver, "res", 0.0)),
            )
            residual = float(np.asarray(_crit).flat[0])
            converged = bool(getattr(solver, "converged", True))
        else:
            tau_vals = []
            for ax in range(3):
                img_ax = np.moveaxis(img, ax, 0)
                s = tf.Solver(img_ax)
                s.solve()
                tau_vals.append(float(np.asarray(s.tau).flat[0]))
            tau_val = float(np.mean(tau_vals))
            n_iter = 0
            residual = 0.0
            converged = True

    except Exception as exc:
        warns.append(
            f"TauFactor {label} solve failed: {exc}. Returning τ=1.0 as fallback."
        )
        return 1.0, 0, 0.0, False

    if tau_val < 1.0:
        warns.append(
            f"TauFactor {label} τ={tau_val:.4f} < 1.0 (unphysical). "
            f"Clamping to 1.0. Check phase mask threshold and connectivity."
        )
        tau_val = 1.0

    return tau_val, n_iter, residual, converged
