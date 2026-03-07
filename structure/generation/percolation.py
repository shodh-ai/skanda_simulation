"""
Step 7 — Percolation Validator

Validates that the conductive network (carbon + Si + CBD) percolates
through the full electrode thickness (Z=0 → Z=nz-1).

Two checks performed:
  1. ELECTRONIC: carbon + Si + CBD solid mask percolates in Z
  2. IONIC:      pore mask percolates in Z (electrolyte accessible)

If electronic percolation fraction < min_threshold, raises
PercolationFailed — the caller retries generation with a new seed.

Implementation uses scipy.ndimage.label (C implementation, union-find),
which handles 128³ in ~50ms — much faster than pure-Python BFS.

Percolation fraction definition:
    N_solid_in_percolating_components / N_total_solid
A component "percolates" iff it touches both Z=0 and Z=nz-1 face.
"""

from __future__ import annotations


from typing import List
import numpy as np

from structure.schema import ResolvedSimulation
from structure.utils.percolation import run_percolation
from structure.phases import PHASE_GRAPHITE
from structure.data import (
    CompositionState,
    DomainGeometry,
    SiMapResult,
    CBDBinderResult,
    SEIResult,
    PercolationFailed,
    PercolationResult,
)


# ---------------------------------------------------------------------------
# PercolationValidator
# ---------------------------------------------------------------------------
class PercolationValidator:
    """
    Validates both electronic and ionic percolation of a microstructure.

    Usage:
        validator = PercolationValidator(comp, domain, sim)
        result    = validator.validate(carbon_label, si_result,
                                       cbd_result, sei_result)
        # Raises PercolationFailed if electronic < threshold
    """

    # Thresholds for calling a vf field "occupied"
    SI_VF_THRESHOLD: float = 0.01
    CBD_VF_THRESHOLD: float = 0.01

    def __init__(
        self,
        comp: CompositionState,
        domain: DomainGeometry,
        sim: ResolvedSimulation,
    ) -> None:
        self.comp = comp
        self.domain = domain
        self.sim = sim

    # ── Public ───────────────────────────────────────────────────────────

    def validate(
        self,
        carbon_label: np.ndarray,
        si_result: SiMapResult,
        cbd_result: CBDBinderResult,
        sei_result: SEIResult,
    ) -> PercolationResult:
        """
        Run percolation analysis.
        Raises PercolationFailed if electronic fraction < min_threshold.
        """
        sim = self.sim
        warns: List[str] = []

        # ------------------------------------------------------------------
        # Build masks
        # ------------------------------------------------------------------
        carbon_mask = carbon_label == PHASE_GRAPHITE
        si_mask = si_result.si_vf > self.SI_VF_THRESHOLD
        cbd_mask = cbd_result.cbd_vf > self.CBD_VF_THRESHOLD

        # Electronic solid = active material + conductive additive
        # Binder and SEI are ionic conductors / insulators — excluded
        elec_solid = carbon_mask | si_mask | cbd_mask

        # Ionic pore = anything not in electronic solid and not SEI
        sei_mask = sei_result.sei_vf > 0.001
        pore_mask = ~elec_solid & ~sei_mask

        # ------------------------------------------------------------------
        # Electronic percolation
        # ------------------------------------------------------------------
        e_frac, e_n_comp, e_n_perc, elec_perc = run_percolation(elec_solid)

        # ------------------------------------------------------------------
        # Ionic percolation
        # ------------------------------------------------------------------
        i_frac, i_n_comp, i_n_perc, ionic_perc = run_percolation(pore_mask)

        # ------------------------------------------------------------------
        # Warnings
        # ------------------------------------------------------------------
        actual_porosity = float(pore_mask.mean())

        if not ionic_perc:
            warns.append(
                "Pore space is NOT percolating in Z. "
                "Electrolyte cannot access the full electrode thickness. "
                "Consider increasing target_porosity."
            )

        if e_n_perc > 1:
            warns.append(
                f"Multiple percolating electronic components ({e_n_perc}). "
                f"Network is fragmented into multiple spanning paths"
            )

        if i_n_perc > 1:
            warns.append(
                f"Multiple percolating ionic components ({i_n_perc}). "
                f"Pore space is connected but fragmented across Z — "
                f"electrolyte may not access all regions uniformly. "
                f"Consider increasing target_porosity."
            )

        if abs(actual_porosity - self.comp.porosity) > 0.05:
            warns.append(
                f"Measured porosity {actual_porosity:.3f} deviates from "
                f"target {self.comp.porosity:.3f} by "
                f"{abs(actual_porosity-self.comp.porosity)*100:.1f} pp."
            )

        result = PercolationResult(
            electronic_percolating=elec_perc,
            electronic_fraction=e_frac,
            electronic_n_components=e_n_comp,
            electronic_n_percolating=e_n_perc,
            ionic_percolating=ionic_perc,
            ionic_fraction=i_frac,
            ionic_n_components=i_n_comp,
            ionic_n_percolating=i_n_perc,
            actual_porosity=actual_porosity,
            min_threshold=sim.percolation_min_threshold,
            warnings=warns,
        )

        # ------------------------------------------------------------------
        # Raise if electronic percolation below threshold
        # ------------------------------------------------------------------
        if sim.percolation_enforce and e_frac < sim.percolation_min_threshold:
            raise PercolationFailed(
                run_id=sim.run_id,
                seed=sim.seed,
                fraction=e_frac,
                threshold=sim.percolation_min_threshold,
            )

        return result


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def validate_percolation(
    comp: CompositionState,
    domain: DomainGeometry,
    sim,
    carbon_label: np.ndarray,
    si_result: SiMapResult,
    cbd_result: CBDBinderResult,
    sei_result: SEIResult,
) -> PercolationResult:
    """
    Canonical pipeline entry for Step 7.

    Args:
      comp         : CompositionState (Step 0)
      domain       : DomainGeometry (Step 1)
      sim          : ResolvedSimulation
      carbon_label : uint8 label map post-calendering (Step 2)
      si_result    : SiMapResult post-calendering (Step 3)
      cbd_result   : CBDBinderResult (Step 4)
      sei_result   : SEIResult (Step 6)

    Returns:
      PercolationResult with full diagnostics.

    Raises:
      PercolationFailed if electronic fraction < sim.percolation_min_threshold
      and sim.percolation_enforce is True.
    """
    return PercolationValidator(comp, domain, sim).validate(
        carbon_label=carbon_label,
        si_result=si_result,
        cbd_result=cbd_result,
        sei_result=sei_result,
    )
