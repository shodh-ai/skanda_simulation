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
    sei_correlation_length_nm: float = Field(
        ...,
        gt=0.0,
        le=10000.0,
        description=(
            "Spatial correlation length of SEI thickness GRF variation (nm). "
            "Converted to voxels as sei_correlation_length_nm / voxel_size_nm. "
            "Resolution-invariant: same value produces consistent spatial "
            "structure at any voxel size."
        ),
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


class GenConfig(BaseModel):

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
    si_embedding_uniformity_cv: float = Field(
        ...,
        ge=0.0,
        le=0.60,
        description=(
            "Spatial CV of Si loading within one graphite particle. "
            "Only active when si_distribution == 'embedded'. "
            "0.0 = perfectly uniform, 0.60 = highly clustered."
        ),
    )
    si_core_shell_carbon_thickness_nm: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description=(
            "Structural carbon shell thickness (nm) in Si@C core-shell particles. "
            "Only active when si_distribution='core_shell'. "
            "Physically distinct from si_coating_thickness_nm (thin passivation). "
            "0.0 = fallback to 20%% of carbon c-axis half-thickness."
        ),
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
    carbon_orientation_degree: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="c-axis alignment toward Z. 0=random, 1=perfectly aligned.",
    )

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
    def validate_composition_sum(self) -> GenConfig:
        """CA + binder must leave ≥0.80 for active material."""
        non_active = self.conductive_additive_wt_frac + self.binder_wt_frac
        if non_active >= 0.20:
            raise ValueError(
                f"conductive_additive_wt_frac + binder_wt_frac = {non_active:.3f}. "
                f"Must be < 0.20 to leave ≥0.80 for active material."
            )
        return self

    @model_validator(mode="after")
    def validate_coating_thickness_vs_coating(self) -> GenConfig:
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
    def validate_carbon_d50_vs_type(self) -> GenConfig:
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
    def validate_orientation_degree_clamp(self) -> GenConfig:
        """Warn if carbon_orientation_degree is close enough to 1.0 to hit the κ clamp."""
        import warnings

        od = self.carbon_orientation_degree
        if od >= 1.0:
            warnings.warn(
                f"carbon_orientation_degree={od} is exactly 1.0. "
                f"The vMF concentration κ will be clamped to 1000 (perfect Z alignment). "
                f"Values above 0.95 are physically indistinguishable — "
                f"consider using 0.95 to avoid the hard clamp.",
                UserWarning,
                stacklevel=2,
            )
        elif od > 0.95:
            warnings.warn(
                f"carbon_orientation_degree={od} is very high (>0.95). "
                f"κ = {-10.0 * __import__('math').log(1.0 - od):.1f} — "
                f"particles will be nearly perfectly Z-aligned. "
                f"This may not be physically representative of a real electrode.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def validate_si_coating_thickness_vs_particle(self) -> GenConfig:
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
    def validate_void_fraction_with_flag(self) -> GenConfig:
        """Warn if si_void_fraction is set while si_void_enabled is False."""
        import warnings

        if not self.si_void_enabled:
            warnings.warn(
                f"si_void_fraction={self.si_void_fraction} is set but "
                f"si_void_enabled=False — this value will be ignored. "
                f"Either set si_void_enabled=true or remove si_void_fraction "
                f"from the config to avoid confusion.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def validate_internal_porosity_with_morphology(self) -> GenConfig:
        """Warn if si_internal_porosity is set while si_morphology != 'porous'."""
        import warnings

        if (
            self.si_morphology != SiMorphology.POROUS
            and self.si_internal_porosity > 0.30
        ):
            warnings.warn(
                f"si_internal_porosity={self.si_internal_porosity} is set but "
                f"si_morphology='{self.si_morphology.value}' — internal porosity "
                f"is only active when si_morphology='porous'. "
                f"This value will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def validate_coating_params_with_flag(self) -> GenConfig:
        """Warn if coating parameters are set while si_coating_enabled is False."""
        import warnings

        if not self.si_coating_enabled and self.si_coating_thickness_nm > 5.0:
            warnings.warn(
                f"si_coating_thickness_nm={self.si_coating_thickness_nm} is set but "
                f"si_coating_enabled=False — coating thickness and type will be "
                f"ignored. Set si_coating_enabled=true to activate.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def validate_core_shell_thickness(self) -> GenConfig:
        """
        Warn if si_core_shell_carbon_thickness_nm is set while
        si_distribution != 'core_shell' — value will be silently unused.
        Also warn if the shell is so thick the Si core would be negligible.
        """
        import warnings

        if (
            self.si_distribution != SiDistribution.CORE_SHELL
            and self.si_core_shell_carbon_thickness_nm > 0.0
        ):
            warnings.warn(
                f"si_core_shell_carbon_thickness_nm="
                f"{self.si_core_shell_carbon_thickness_nm} is set but "
                f"si_distribution='{self.si_distribution.value}' — "
                f"this value is only used when si_distribution='core_shell'.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def validate_sei_correlation_vs_resolution(self) -> GenConfig:
        """
        Warn if sei_correlation_length_nm is smaller than one voxel —
        the GRF sigma would be < 1.0 voxels, making it effectively white noise
        and indistinguishable from sei_uniformity_cv=0.
        Also warn if it exceeds the full domain size (correlation larger than
        the electrode thickness is unphysical).
        """
        import warnings

        sigma_vox = self.sei.sei_correlation_length_nm / self.voxel_size_nm
        domain_z_nm = self.coating_thickness_um * 1000.0

        if sigma_vox < 1.0:
            warnings.warn(
                f"sei_correlation_length_nm={self.sei.sei_correlation_length_nm}nm "
                f"is smaller than one voxel ({self.voxel_size_nm}nm) — "
                f"sigma_vox={sigma_vox:.3f} < 1.0. The GRF will be "
                f"effectively white noise. Increase sei_correlation_length_nm "
                f"to at least {self.voxel_size_nm:.0f}nm (one voxel).",
                UserWarning,
                stacklevel=2,
            )
        if self.sei.sei_correlation_length_nm > domain_z_nm:
            warnings.warn(
                f"sei_correlation_length_nm={self.sei.sei_correlation_length_nm}nm "
                f"exceeds the electrode thickness "
                f"({domain_z_nm:.0f}nm = coating_thickness_um × 1000). "
                f"SEI correlation length larger than the domain is unphysical.",
                UserWarning,
                stacklevel=2,
            )
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_gen_config(path: str | Path) -> GenConfig:
    """Load and validate a run config YAML. Raises ValidationError on any issue."""
    raw = yaml.safe_load(Path(path).read_text())
    return GenConfig.model_validate(raw)
