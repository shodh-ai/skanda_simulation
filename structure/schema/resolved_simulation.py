"""
Combines SimConfig + MaterialsDB into ResolvedSimulation.
This is the ONLY object the simulation steps (TauFactor, PyBaMM) ever read.

What lives here:   resolved material objects, typed protocols, solver settings,
                   loaded electrolyte expression module
What lives in vol: anode geometry, measured porosity, phase fractions, capacity
What does NOT live here: MicrostructureVolume (passed directly to sim steps)
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Optional

from .materials import (
    MaterialsDB,
    CathodeMaterial,
    ElectrolyteMaterial,
    SeparatorMaterial,
    CurrentCollectorMaterial,
)
from .sim_config import (
    SimConfig,
    RateCapabilityProtocol,
    DCIRPulseProtocol,
    CycleLifeProtocol,
    TauFactorConfig,
    PyBaMMConfig,
    OutputsConfig,
)


# ---------------------------------------------------------------------------
# Ether electrolyte + high-voltage cathode guard
# ---------------------------------------------------------------------------

# LiFSI_DME_1M uses an ether solvent (DME). Ethers oxidise above ~4.0 V vs
# Li/Li+. Pairing with NMC/NCA/LCO (voltage_max > 4.0 V) would produce
# unphysically optimistic results because ether oxidation is not modelled in
# standard PyBaMM. Enforced as a hard error at resolve time.
_ETHER_ELECTROLYTES: frozenset[str] = frozenset({"LiFSI_DME_1M"})
_ETHER_VOLTAGE_LIMIT_V: float = 4.0


# ---------------------------------------------------------------------------
# Resolved sub-objects
# ---------------------------------------------------------------------------


@dataclass
class ResolvedCathode:
    material: CathodeMaterial
    key: str  # DB key e.g. "nmc811"
    thickness_um: float
    porosity: float

    @property
    def capacity_mAh_g(self) -> float:
        return self.material.capacity_mAh_g


@dataclass
class ResolvedSeparator:
    material: SeparatorMaterial
    key: str

    @property
    def tortuosity(self) -> float:
        # µ-CT measured tortuosity always takes precedence over Bruggeman
        # (Lagadec et al. 2016) — stored directly in material.tortuosity.
        return self.material.tortuosity


@dataclass
class ResolvedElectrolyte:
    material: ElectrolyteMaterial
    key: str
    temperature_K: float
    # Loaded expression module — exposes kappa(c,T), D(c,T),
    # t_plus(c,T), thermodynamic_factor(c,T).
    # None if expressions file is not found (DB reference values used instead).
    expressions: Optional[ModuleType]

    @property
    def temperature_C(self) -> float:
        return self.temperature_K - 273.15

    def kappa(self, c_mol_m3: float) -> float:
        """Ionic conductivity S/m at current temperature."""
        if self.expressions and hasattr(self.expressions, "kappa"):
            return self.expressions.kappa(c_mol_m3, self.temperature_K)
        return self.material.ionic_conductivity_S_m  # DB reference value

    def D(self, c_mol_m3: float) -> float:
        """Li+ diffusivity m²/s at current temperature."""
        if self.expressions and hasattr(self.expressions, "D"):
            return self.expressions.D(c_mol_m3, self.temperature_K)
        return self.material.li_diffusivity_m2_s

    def t_plus(self, c_mol_m3: float) -> float:
        """Li+ transference number at current temperature."""
        if self.expressions and hasattr(self.expressions, "t_plus"):
            return self.expressions.t_plus(c_mol_m3, self.temperature_K)
        return self.material.transference_number

    def thermodynamic_factor(self, c_mol_m3: float) -> float:
        """1 + d(ln f±)/d(ln c) at current temperature."""
        if self.expressions and hasattr(self.expressions, "thermodynamic_factor"):
            return self.expressions.thermodynamic_factor(c_mol_m3, self.temperature_K)
        return self.material.thermodynamic_factor


# ---------------------------------------------------------------------------
# ResolvedSimulation
# ---------------------------------------------------------------------------


@dataclass
class ResolvedSimulation:
    """
    Single resolved object handed to TauFactor and PyBaMM simulation steps.
    Never touches raw YAML — only reads from this object.
    """

    # ── Cell ─────────────────────────────────────────────────────────────
    cell_area_cm2: float
    np_ratio: float
    anode_cc_thickness_um: float
    cathode_cc_thickness_um: float  # from al_foil in DB (mean of min/max)
    cathode: ResolvedCathode
    separator: ResolvedSeparator
    electrolyte: ResolvedElectrolyte

    # ── Cycling ───────────────────────────────────────────────────────────
    voltage_cutoff_low_V: float
    voltage_cutoff_high_V: float
    rate_capability: Optional[RateCapabilityProtocol]
    dcir_pulse: Optional[DCIRPulseProtocol]
    cycle_life: Optional[CycleLifeProtocol]

    # ── Tools ─────────────────────────────────────────────────────────────
    taufactor: TauFactorConfig
    pybamm: PyBaMMConfig

    # ── Outputs ───────────────────────────────────────────────────────────
    outputs: OutputsConfig

    # ── Convenience ──────────────────────────────────────────────────────

    @property
    def any_protocol_enabled(self) -> bool:
        return any(
            [
                self.rate_capability and self.rate_capability.enabled,
                self.dcir_pulse and self.dcir_pulse.enabled,
                self.cycle_life and self.cycle_life.enabled,
            ]
        )

    def summary(self) -> str:
        lines = [
            "=" * 60,
            " RESOLVED SIMULATION",
            "=" * 60,
            f"  Cathode      : {self.cathode.key} "
            f"({self.cathode.thickness_um:.0f}µm, "
            f"ε={self.cathode.porosity:.2f})",
            f"  Separator    : {self.separator.key} "
            f"(τ={self.separator.tortuosity:.2f}, "
            f"ε={self.separator.material.porosity:.2f})",
            f"  Electrolyte  : {self.electrolyte.key} "
            f"@ {self.electrolyte.temperature_C:.0f}°C",
            f"  Expressions  : "
            f"{'loaded' if self.electrolyte.expressions else 'DB reference values'}",
            f"  N/P ratio    : {self.np_ratio:.2f}",
            f"  Cell area    : {self.cell_area_cm2:.1f} cm²",
            f"  Voltage      : {self.voltage_cutoff_low_V:.2f}–"
            f"{self.voltage_cutoff_high_V:.2f} V",
            "",
            "  Protocols:",
        ]
        for name, proto in [
            ("rate_capability", self.rate_capability),
            ("dcir_pulse", self.dcir_pulse),
            ("cycle_life", self.cycle_life),
        ]:
            status = (
                "✓ enabled"
                if proto and proto.enabled
                else "✗ disabled" if proto else "— not configured"
            )
            lines.append(f"    {name:<22} {status}")

        lines += [
            "",
            f"  TauFactor    : {'enabled' if self.taufactor.enabled else 'disabled'}"
            f"  dir={self.taufactor.direction}",
            f"  PyBaMM       : {'enabled' if self.pybamm.enabled else 'disabled'}"
            f"  model={self.pybamm.model}  sei={self.pybamm.sei_model}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Expression loader
# ---------------------------------------------------------------------------


def _load_expressions(
    key: str,
    expressions_dir: Path,
) -> Optional[ModuleType]:
    """
    Load electrolyte expression module from expressions_dir/{key}.py.
    Returns None (with a warning) if the file does not exist — the caller
    falls back to DB reference scalar values.
    """
    expr_path = expressions_dir / f"{key}.py"
    if not expr_path.exists():
        import warnings

        warnings.warn(
            f"Electrolyte expression file not found: {expr_path}. "
            f"Using DB reference values for κ, D, t⁺, f± at T_ref. "
            f"For accurate PyBaMM simulations, add "
            f"materialdb/electrolyte_expressions/{key}.py.",
            UserWarning,
            stacklevel=3,
        )
        return None

    spec = importlib.util.spec_from_file_location(f"electrolyte_expr_{key}", expr_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"electrolyte_expr_{key}"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def resolve_simulation(
    sim_cfg: SimConfig,
    db: MaterialsDB,
    expressions_dir: str | Path = Path("structure/schema/electrolyte_expressions"),
) -> ResolvedSimulation:
    """
    Merge SimConfig + MaterialsDB into one flat ResolvedSimulation.

    Args:
        sim_cfg         : validated SimConfig from load_sim_config()
        db              : validated MaterialsDB from load_materials_db()
        expressions_dir : directory containing {electrolyte_key}.py files.
                          Default: structure/schema/electrolyte_expressions/
                          (relative to CWD, i.e. the project root)

    Returns:
        ResolvedSimulation

    Raises:
        ValueError  on hard constraint violations (e.g. ether + NMC)
        AttributeError if a DB key doesn't exist (misconfigured YAML)
    """
    expressions_dir = Path(expressions_dir)

    # ── Cathode ───────────────────────────────────────────────────────────
    cathode_mat = db.get_cathode(sim_cfg.cell.cathode.material)
    cathode = ResolvedCathode(
        material=cathode_mat,
        key=sim_cfg.cell.cathode.material,
        thickness_um=sim_cfg.cell.cathode.thickness_um,
        porosity=sim_cfg.cell.cathode.porosity,
    )

    # ── Separator ─────────────────────────────────────────────────────────
    sep_mat = db.get_separator(sim_cfg.cell.separator.material)
    separator = ResolvedSeparator(
        material=sep_mat,
        key=sim_cfg.cell.separator.material,
    )

    # ── Electrolyte ───────────────────────────────────────────────────────
    elec_key = sim_cfg.electrolyte.material
    elec_mat = db.get_electrolyte(elec_key)

    # Hard constraint: ether electrolytes cannot pair with high-voltage cathodes.
    # Ether oxidation above 4.0 V is not modelled in standard PyBaMM DFN — using
    # LiFSI_DME_1M with NMC/NCA/LCO would silently produce unphysical results.
    if (
        elec_key in _ETHER_ELECTROLYTES
        and cathode_mat.voltage_max_V > _ETHER_VOLTAGE_LIMIT_V
    ):
        raise ValueError(
            f"Electrolyte '{elec_key}' (ether-based, max stable ~4.0V) is "
            f"incompatible with cathode '{cathode.key}' "
            f"(voltage_max={cathode_mat.voltage_max_V}V). "
            f"Ether oxidation above 4.0V is not modelled. "
            f"Use LFP cathode or switch to a carbonate electrolyte."
        )

    expressions = _load_expressions(elec_key, expressions_dir)

    electrolyte = ResolvedElectrolyte(
        material=elec_mat,
        key=elec_key,
        temperature_K=sim_cfg.electrolyte.temperature_K,
        expressions=expressions,
    )

    # ── Cathode current collector (use DB mean thickness) ────────────────
    cathode_cc_thickness = (
        db.al_foil.thickness_min_um + db.al_foil.thickness_max_um
    ) / 2.0

    # ── Protocols — unpack list into typed Optional fields ───────────────
    rate_capability: Optional[RateCapabilityProtocol] = None
    dcir_pulse: Optional[DCIRPulseProtocol] = None
    cycle_life: Optional[CycleLifeProtocol] = None

    for proto in sim_cfg.cycling.protocols:
        if proto.name == "rate_capability":
            rate_capability = proto
        elif proto.name == "dcir_pulse":
            dcir_pulse = proto
        elif proto.name == "cycle_life":
            cycle_life = proto

    return ResolvedSimulation(
        cell_area_cm2=sim_cfg.cell.cell_area_cm2,
        np_ratio=sim_cfg.cell.np_ratio,
        anode_cc_thickness_um=sim_cfg.cell.anode.current_collector_thickness_um,
        cathode_cc_thickness_um=cathode_cc_thickness,
        cathode=cathode,
        separator=separator,
        electrolyte=electrolyte,
        voltage_cutoff_low_V=sim_cfg.cycling.voltage_cutoff_low_V,
        voltage_cutoff_high_V=sim_cfg.cycling.voltage_cutoff_high_V,
        rate_capability=rate_capability,
        dcir_pulse=dcir_pulse,
        cycle_life=cycle_life,
        taufactor=sim_cfg.taufactor,
        pybamm=sim_cfg.pybamm,
        outputs=sim_cfg.outputs,
    )
