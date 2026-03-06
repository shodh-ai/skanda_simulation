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

from dataclasses import dataclass, field
from typing import Set, List

import numpy as np
from scipy.ndimage import label as scipy_label

from structure.schema.resolved import ResolvedSimulation

from .composition import CompositionState
from .domain import DomainGeometry
from .si_mapper import SiMapResult
from .cbd_binder import CBDBinderResult
from .sei import SEIResult
from ..phases import PHASE_GRAPHITE
from ._percolation_utils import run_percolation


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class PercolationFailed(Exception):
    """
    Raised when the electronic percolation fraction is below min_threshold.
    Caller should catch this, increment seed, and regenerate from Step 2.

    Attributes:
      run_id  : run identifier from sim.run_id
      seed    : the seed that produced this failure
      fraction: the percolation fraction that was measured
      threshold: the required minimum
    """

    def __init__(
        self,
        run_id: int,
        seed: int,
        fraction: float,
        threshold: float,
    ) -> None:
        self.run_id = run_id
        self.seed = seed
        self.fraction = fraction
        self.threshold = threshold
        super().__init__(
            f"Run {run_id} seed={seed}: electronic percolation fraction "
            f"{fraction:.4f} < threshold {threshold:.4f}. Retry with new seed."
        )


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class PercolationResult:
    """
    Full percolation diagnostics for one microstructure.

    electronic_percolating   : True if elec. network percolates Z=0→Z=nz-1
    electronic_fraction      : fraction of solid in percolating components
    electronic_n_components  : total number of distinct solid components
    electronic_n_percolating : number of components that span Z

    ionic_percolating         : True if pore space percolates Z=0→Z=nz-1
    ionic_fraction            : fraction of pore in percolating pore components
    ionic_n_components        : total pore components
    actual_porosity           : measured pore fraction in this volume
    warnings                  : list of warning strings
    """

    # Electronic (conductive) network
    electronic_percolating: bool
    electronic_fraction: float
    electronic_n_components: int
    electronic_n_percolating: int

    # Ionic (pore) network
    ionic_percolating: bool
    ionic_fraction: float
    ionic_n_components: int

    # Global
    actual_porosity: float
    min_threshold: float
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        e_ok = "✓  PASS" if self.electronic_percolating else "✗  FAIL"
        i_ok = "✓  PASS" if self.ionic_percolating else "✗  FAIL"
        lines = [
            "=" * 62,
            "  PERCOLATION VALIDATOR",
            "=" * 62,
            f"  Electronic network  [{e_ok}]",
            f"    percolating frac : {self.electronic_fraction:.4f}"
            f"  (threshold ≥ {self.min_threshold:.2f})",
            f"    components       : {self.electronic_n_components:,}"
            f"  ({self.electronic_n_percolating} span Z)",
            "",
            f"  Ionic (pore) path   [{i_ok}]",
            f"    percolating frac : {self.ionic_fraction:.4f}",
            f"    pore components  : {self.ionic_n_components:,}",
            "",
            f"  Actual porosity    : {self.actual_porosity:.4f}",
        ]
        if self.warnings:
            lines += ["", f"  ⚠  {len(self.warnings)} WARNING(s):"]
            lines += [f"     [{i+1}] {w}" for i, w in enumerate(self.warnings)]
        lines.append("=" * 62)
        return "\n".join(lines)


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
        i_frac, i_n_comp, _, ionic_perc = run_percolation(pore_mask)

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
                f"Network is connected but fragmented — consider increasing CBD."
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
