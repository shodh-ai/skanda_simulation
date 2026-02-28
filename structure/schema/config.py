"""
Pydantic schema for str_gen_config.yml.
Validates all 45 config fields with range checks and cross-field rules.
"""

from __future__ import annotations
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Enums — every string option in the config
# ---------------------------------------------------------------------------


class SiMorphology(str, Enum):
    SPHERICAL = "spherical"
    IRREGULAR = "irregular"
    POROUS = "porous"


class SiDistribution(str, Enum):
    EMBEDDED = "embedded"
    SURFACE_ANCHORED = "surface_anchored"
    CORE_SHELL = "core_shell"


class SiCoatingType(str, Enum):
    CARBON_COATING = "carbon_coating"
    SIOX_COATING = "siox_coating"


class CarbonType(str, Enum):
    GRAPHITE_ARTIFICIAL = "graphite_artificial"
    GRAPHITE_NATURAL = "graphite_natural"
    GRAPHITE_MCMB = "graphite_mcmb"


class ConductiveAdditiveType(str, Enum):
    CB_SUPERP = "cb_superp"
    CB_C65 = "cb_c65"
    CB_KETJENBLACK = "cb_ketjenblack"
    CNT_MULTIWALLED = "cnt_multiwalled"
    GRAPHENE_FLAKES = "graphene_flakes"


class ConductiveAdditiveDistribution(str, Enum):
    AGGREGATE = "aggregate"
    DISPERSED = "dispersed"
    NETWORK = "network"


class BinderType(str, Enum):
    PVDF_KYNAR = "pvdf_kynar"
    CMC_SBR = "cmc_sbr"


class BinderDistribution(str, Enum):
    UNIFORM = "uniform"
    PATCHY = "patchy"
    NECKS = "necks"


class SeiMaterial(str, Enum):
    SEI_GENERIC = "sei_generic"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class CalenderingConfig(BaseModel):
    compression_ratio: float = Field(
        ..., ge=0.50, le=0.90, description="final_z / initial_z"
    )
    particle_deformation: float = Field(
        ..., ge=0.00, le=1.00, description="0=rigid, 1=fully plastic"
    )
    orientation_enhancement: float = Field(
        ...,
        ge=0.00,
        le=0.50,
        description="Additional c-axis alignment from calendering",
    )


class SEIConfig(BaseModel):
    enabled: bool
    sei_material: SeiMaterial = SeiMaterial.SEI_GENERIC
    thickness_nm: float = Field(
        ..., ge=5.0, le=100.0, description="Fresh: 5–20 nm | Aged: 20–100 nm"
    )
    uniformity_cv: float = Field(
        ..., ge=0.00, le=1.00, description="Spatial thickness variation σ/μ"
    )


class PercolationConfig(BaseModel):
    enforce: bool
    min_threshold: float = Field(
        ...,
        ge=0.80,
        le=0.99,
        description="Values >0.99 risk infinite regeneration loops",
    )


class ContactsConfig(BaseModel):
    coordination_number: float = Field(..., ge=4.0, le=8.0)
    contact_area_fraction: float = Field(..., ge=0.05, le=0.20)


class ParticleCracksConfig(BaseModel):
    enabled: bool
    crack_probability: float = Field(..., ge=0.01, le=0.20)
    crack_width_nm: float = Field(..., ge=10.0, le=200.0)


class BinderAgglomerationConfig(BaseModel):
    enabled: bool
    agglomeration_probability: float = Field(..., ge=0.05, le=0.30)


class DelaminationConfig(BaseModel):
    enabled: bool
    delamination_fraction: float = Field(..., ge=0.01, le=0.20)


class PoreClusteringConfig(BaseModel):
    enabled: bool
    clustering_degree: float = Field(..., ge=0.00, le=1.00)


class DefectsConfig(BaseModel):
    particle_cracks: ParticleCracksConfig
    binder_agglomeration: BinderAgglomerationConfig
    delamination: DelaminationConfig
    pore_clustering: PoreClusteringConfig


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):

    # --- Metadata ---
    run_id: int = Field(..., ge=0)
    seed: int = Field(..., ge=0)

    # --- Domain ---
    coating_thickness_um: float = Field(
        ...,
        ge=20.0,
        le=150.0,
        description="Physical cube edge = electrode coating depth",
    )
    voxel_resolution: Literal[64, 128, 256] = Field(
        ...,
        description="Output cube edge in voxels. Memory: 64→0.25MB, 128→2MB, 256→16MB",
    )

    # --- Composition ---
    conductive_additive_wt_frac: float = Field(..., ge=0.01, le=0.10)
    binder_wt_frac: float = Field(..., ge=0.02, le=0.10)
    si_wt_frac_in_am: float = Field(
        ..., gt=0.0, lt=1.0, description="Si/(Si+C-matrix) within active material only"
    )
    target_porosity: float = Field(..., ge=0.20, le=0.50)

    # --- Silicon ---
    si_particle_d50_nm: float = Field(..., ge=30.0, le=3000.0)
    si_particle_size_cv: float = Field(..., ge=0.10, le=0.60)
    si_morphology: SiMorphology
    si_internal_porosity: float = Field(
        ..., ge=0.30, le=0.70, description="Only active when si_morphology == porous"
    )
    si_distribution: SiDistribution
    si_void_enabled: bool
    si_void_fraction: float = Field(
        ..., ge=0.20, le=0.50, description="Only active when si_void_enabled == true"
    )
    si_coating_enabled: bool
    si_coating_type: SiCoatingType
    si_coating_thickness_nm: float = Field(
        ...,
        ge=2.0,
        le=50.0,
        description="carbon_coating: 5–50 nm | siox_coating: 2–20 nm",
    )

    # --- Carbon Matrix ---
    carbon_particle_d50_nm: float = Field(
        ..., ge=5000.0, le=30000.0, description="Overrides DB d50_nm when set"
    )
    carbon_particle_size_cv: float = Field(
        ..., ge=0.15, le=0.45, description="Overrides DB size_cv when set"
    )
    carbon_type: CarbonType

    # --- Conductive Additive ---
    conductive_additive_type: ConductiveAdditiveType
    conductive_additive_distribution: ConductiveAdditiveDistribution

    # --- Binder ---
    binder_type: BinderType
    binder_distribution: BinderDistribution

    # --- Manufacturing ---
    calendering: CalenderingConfig
    sei: SEIConfig

    # --- Constraints ---
    percolation: PercolationConfig
    contacts: ContactsConfig

    # --- Defects ---
    defects: DefectsConfig

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def voxel_size_nm(self) -> float:
        """Physical size of one voxel in nm."""
        return (self.coating_thickness_um * 1000.0) / self.voxel_resolution

    @property
    def active_material_wt_frac(self) -> float:
        """Derived: 1 - CA - binder."""
        return 1.0 - self.conductive_additive_wt_frac - self.binder_wt_frac

    @property
    def silicon_wt_frac(self) -> float:
        """Si fraction of total electrode mass."""
        return self.active_material_wt_frac * self.si_wt_frac_in_am

    @property
    def carbon_matrix_wt_frac(self) -> float:
        """C-matrix fraction of total electrode mass."""
        return self.active_material_wt_frac * (1.0 - self.si_wt_frac_in_am)

    # -----------------------------------------------------------------------
    # Cross-field validators
    # -----------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_composition_sum(self) -> RunConfig:
        """CA + binder must leave ≥0.80 for active material."""
        non_active = self.conductive_additive_wt_frac + self.binder_wt_frac
        if non_active >= 0.20:
            raise ValueError(
                f"conductive_additive_wt_frac + binder_wt_frac = {non_active:.3f}. "
                f"Must be < 0.20 to leave ≥0.80 for active material."
            )
        return self

    @model_validator(mode="after")
    def validate_coating_thickness_vs_coating(self) -> RunConfig:
        """siox_coating range is 2–20 nm, carbon_coating is 5–50 nm."""
        if self.si_coating_enabled:
            t = self.si_coating_thickness_nm
            if self.si_coating_type == SiCoatingType.SIOX_COATING and not (
                2.0 <= t <= 20.0
            ):
                raise ValueError(
                    f"si_coating_thickness_nm={t} out of range for siox_coating (2–20 nm)"
                )
            if self.si_coating_type == SiCoatingType.CARBON_COATING and not (
                5.0 <= t <= 50.0
            ):
                raise ValueError(
                    f"si_coating_thickness_nm={t} out of range for carbon_coating (5–50 nm)"
                )
        return self

    @model_validator(mode="after")
    def validate_carbon_d50_vs_type(self) -> RunConfig:
        """Warn if d50 is outside the typical range for the chosen carbon type."""
        ranges = {
            CarbonType.GRAPHITE_ARTIFICIAL: (10000.0, 30000.0),
            CarbonType.GRAPHITE_NATURAL: (5000.0, 25000.0),
            CarbonType.GRAPHITE_MCMB: (5000.0, 40000.0),
        }
        lo, hi = ranges[self.carbon_type]
        d50 = self.carbon_particle_d50_nm
        if not (lo <= d50 <= hi):
            import warnings

            warnings.warn(
                f"carbon_particle_d50_nm={d50} is outside the typical range "
                f"({lo}–{hi} nm) for {self.carbon_type.value}",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def validate_si_coating_thickness_vs_particle(self) -> RunConfig:
        """Coating thickness must be << particle radius (at most 50% of radius)."""
        if self.si_coating_enabled:
            r = self.si_particle_d50_nm / 2.0
            if self.si_coating_thickness_nm > r * 0.5:
                raise ValueError(
                    f"si_coating_thickness_nm={self.si_coating_thickness_nm} is >50% of "
                    f"Si particle radius ({r:.1f} nm). Unphysical."
                )
        return self

    @model_validator(mode="after")
    def validate_void_fraction_with_flag(self) -> RunConfig:
        """si_void_fraction is meaningless and ignored when si_void_enabled is False."""
        # No hard error — just a reminder that the value is silently unused.
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a run config YAML. Raises ValidationError on any issue."""
    raw = yaml.safe_load(Path(path).read_text())
    return RunConfig.model_validate(raw)
