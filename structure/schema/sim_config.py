"""
Pydantic schema for sim_config.yml.
Validated cell assembly + cycling + solver settings.
Does NOT contain material properties — those live in materials_db.yml.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator
import yaml
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Protocol sub-models — discriminated on `name`
# ---------------------------------------------------------------------------


class RateCapabilityProtocol(BaseModel):
    name: Literal["rate_capability"]
    enabled: bool = True
    c_rates: list[float] = Field(..., min_length=1)
    charge_c_rate: float = Field(..., gt=0)
    rest_time_s: float = Field(300.0, ge=0)

    @property
    def enabled_c_rates(self) -> list[float]:
        return self.c_rates if self.enabled else []


class DCIRPulseProtocol(BaseModel):
    name: Literal["dcir_pulse"]
    enabled: bool = True
    soc_point: float = Field(..., ge=0.0, le=1.0)
    pulse_c_rate: float = Field(..., gt=0)
    pulse_duration_s: float = Field(..., gt=0)


class CycleLifeProtocol(BaseModel):
    name: Literal["cycle_life"]
    enabled: bool = True
    charge_c_rate: float = Field(..., gt=0)
    discharge_c_rate: float = Field(..., gt=0)
    n_cycles: int = Field(..., gt=0)
    end_of_life_retention: float = Field(..., ge=0.5, le=1.0)
    record_every_n_cycles: int = Field(5, ge=1)


# Pydantic v2 discriminated union — `name` is the tag
Protocol = Annotated[
    Union[RateCapabilityProtocol, DCIRPulseProtocol, CycleLifeProtocol],
    Field(discriminator="name"),
]


# ---------------------------------------------------------------------------
# Cell sub-models
# ---------------------------------------------------------------------------


class AnodeConfig(BaseModel):
    current_collector_thickness_um: float = Field(..., gt=0, le=20.0)


class CathodeConfig(BaseModel):
    material: str  # DB key — normalised to lowercase on validation
    thickness_um: float = Field(..., ge=30.0, le=200.0)
    porosity: float = Field(..., ge=0.20, le=0.45)

    @model_validator(mode="after")
    def _normalise_key(self) -> "CathodeConfig":
        self.material = self.material.lower()
        return self


class SeparatorConfig(BaseModel):
    material: str  # DB key — kept as-is (Celgard keys are mixed-case)


class CellConfig(BaseModel):
    anode: AnodeConfig
    cathode: CathodeConfig
    separator: SeparatorConfig
    np_ratio: float = Field(..., ge=0.05, le=2.0)
    cell_area_cm2: float = Field(..., gt=0, le=200.0)


# ---------------------------------------------------------------------------
# Electrolyte
# ---------------------------------------------------------------------------


class ElectrolyteConfig(BaseModel):
    material: str  # DB key
    temperature_K: float = Field(..., ge=233.0, le=333.0)

    @property
    def temperature_C(self) -> float:
        return self.temperature_K - 273.15


# ---------------------------------------------------------------------------
# Cycling
# ---------------------------------------------------------------------------


class CyclingConfig(BaseModel):
    voltage_cutoff_low_V: float = Field(..., ge=2.0, le=3.0)
    voltage_cutoff_high_V: float = Field(..., ge=3.0, le=4.35)
    protocols: list[Protocol] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _voltage_order(self) -> "CyclingConfig":
        if self.voltage_cutoff_high_V <= self.voltage_cutoff_low_V:
            raise ValueError(
                f"voltage_cutoff_high_V ({self.voltage_cutoff_high_V}) "
                f"must be > voltage_cutoff_low_V ({self.voltage_cutoff_low_V})"
            )
        # Ensure voltage window is large enough for stable simulation
        voltage_window = self.voltage_cutoff_high_V - self.voltage_cutoff_low_V
        if voltage_window < 0.2:  # At least 200mV window needed
            raise ValueError(
                f"Voltage window ({voltage_window:.3f}V) is too small for stable simulation. "
                f"Minimum 0.2V required."
            )
        # Ensure voltages are physically reasonable
        if self.voltage_cutoff_low_V <= 0.0:
            raise ValueError(
                f"voltage_cutoff_low_V ({self.voltage_cutoff_low_V}) must be positive"
            )
        if self.voltage_cutoff_high_V > 5.0:
            raise ValueError(
                f"voltage_cutoff_high_V ({self.voltage_cutoff_high_V}) is unreasonably high (>5.0V)"
            )
        return self

    def get_protocol(self, name: str) -> Optional[Protocol]:
        """Return the first protocol with the given name, or None."""
        return next((p for p in self.protocols if p.name == name), None)


# ---------------------------------------------------------------------------
# TauFactor
# ---------------------------------------------------------------------------


class TauFactorConfig(BaseModel):
    enabled: bool = True
    # Voxels with pore_vf >= pore_threshold are treated as open pore space
    pore_threshold: float = Field(0.5, ge=0.1, le=0.7)
    # Voxels with (carbon_vf + cbd_vf) >= electronic_threshold are conductive
    electronic_threshold: float = Field(0.3, ge=0.1, le=0.5)
    # Through-plane only ("z") or all three axes ("all")
    direction: Literal["z", "all"] = "z"


# ---------------------------------------------------------------------------
# PyBaMM
# ---------------------------------------------------------------------------


class PyBaMMConfig(BaseModel):
    enabled: bool = True
    model: Literal["SPM", "SPMe", "DFN"] = "DFN"
    solver: Literal["IDAKLUSolver", "CasadiSolver"] = "IDAKLUSolver"
    atol: float = Field(1e-6, gt=0)
    rtol: float = Field(1e-6, gt=0)
    # "none" → no degradation; "reaction_limited" → standard SEI model in DFN
    sei_model: Literal["none", "reaction_limited", "solvent_diffusion"] = (
        "reaction_limited"
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


class OutputsConfig(BaseModel):
    # TauFactor outputs
    tortuosity_factor: bool = True
    electrode_porosity: bool = True
    effective_diffusivity: bool = True
    bruggeman_exponent: bool = True
    # rate_capability outputs
    nominal_capacity_mAh_cm2: bool = True
    energy_density_mWh_cm2: bool = True
    # dcir_pulse output
    initial_dcir_mOhm_cm2: bool = True
    # cycle_life outputs
    projected_cycle_life: bool = True
    capacity_fade_rate_pct_per_cycle: bool = True


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class SimConfig(BaseModel):
    cell: CellConfig
    electrolyte: ElectrolyteConfig
    cycling: CyclingConfig
    taufactor: TauFactorConfig = Field(default_factory=TauFactorConfig)
    pybamm: PyBaMMConfig = Field(default_factory=PyBaMMConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @classmethod
    def sample(
        cls,
        *,
        u: "np.ndarray | None" = None,
        rng: "np.random.Generator | None" = None,
        seed: int | None = None,
    ) -> "SimConfig":
        """
        Create a sampled SimConfig from a unit-hypercube vector.

        Parameters
        ----------
        u    : Optional pre-computed unit vector of length N_SIM_DIMS.
               When None, one is drawn from ``rng`` or a fresh Generator.
        rng  : Optional numpy Generator (used when ``u`` is None).
        seed : Seed for a fresh Generator (used when both u and rng are None).

        Examples
        --------
        # Single random sample:
        sim = SimConfig.sample()

        # From a pre-computed LHS row:
        sim = SimConfig.sample(u=lhs_matrix[i, N_GEN_DIMS:])
        """
        import numpy as _np
        from structure.sampling._sim_map import N_SIM_DIMS, map_sim_config

        if u is None:
            if rng is None:
                rng = _np.random.default_rng(seed)
            u = rng.random(N_SIM_DIMS)

        kwargs = map_sim_config(_np.asarray(u, dtype=float))
        return cls.model_validate(kwargs)

    def to_flat_dict(self) -> dict:
        """
        Return a flat dict of all SimConfig fields prefixed with ``sim_``.
        Protocols are broken out by name, e.g.:
            sim_cycling_rate_capability_charge_c_rate,
            sim_cycling_dcir_pulse_soc_point,
            sim_cycling_cycle_life_n_cycles, ...

        Suitable as a set of columns in a CSV row.
        """
        from structure.sampling.flatten import flatten_dict

        raw = self.model_dump()

        # Pull protocols out and flatten them separately by name
        protocols_flat: dict = {}
        for proto in raw.get("cycling", {}).get("protocols", []):
            name = proto.get("name", "unknown")
            for k, v in proto.items():
                if k != "name":
                    key = f"sim_cycling_{name}_{k}"
                    protocols_flat[key] = (
                        ";".join(str(x) for x in v) if isinstance(v, list) else v
                    )

        # Flatten everything except the protocols list
        raw_no_proto = {
            k: (
                {kk: vv for kk, vv in v.items() if kk != "protocols"}
                if k == "cycling"
                else v
            )
            for k, v in raw.items()
        }
        base_flat = flatten_dict(raw_no_proto, prefix="sim")
        return {**base_flat, **protocols_flat}


def load_sim_config(path: str | Path) -> SimConfig:
    """Load and validate sim_config.yml. Raises ValidationError on any issue."""
    raw = yaml.safe_load(Path(path).read_text())
    return SimConfig.model_validate(raw)
