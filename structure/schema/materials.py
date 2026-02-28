"""
Pydantic schema for materials_db.yml.
Validates all 15 material entries and exposes typed accessors.
"""

from __future__ import annotations
import math
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Sub-models per material family
# ---------------------------------------------------------------------------


class GraphiteMaterial(BaseModel):
    family: str
    description: str
    density_g_cm3: float = Field(..., gt=0)
    young_modulus_GPa: float = Field(..., gt=0)
    poisson_ratio: float = Field(..., ge=0, le=0.5)
    aspect_ratio_mean: float = Field(..., ge=1.0)
    aspect_ratio_std: float = Field(..., ge=0.0)
    d002_nm: float = Field(..., ge=0.335, le=0.345)
    Lc_nm: float = Field(..., gt=0)
    La_nm: float = Field(..., gt=0)
    electrical_conductivity_S_m: float = Field(..., gt=0)
    li_diffusivity_m2_s: float = Field(..., gt=0)
    molar_mass_g_mol: float = Field(..., gt=0)
    theoretical_capacity_mAh_g: float = Field(..., gt=0)


class SiMorphologyDetail(BaseModel):
    roundness: Optional[float] = None
    roundness_range: Optional[list[float]] = None
    aspect_ratio_range: Optional[list[float]] = None
    internal_porosity_range: Optional[list[float]] = None
    pore_diameter_nm_range: Optional[list[float]] = None


class SiCorrelations(BaseModel):
    # BET: 6000 / (density * d50_nm)
    # li_diffusivity: D0 * exp(-alpha * ln(d50/d_ref))
    D0: float
    d_ref: float
    alpha: float
    # electrical_conductivity: sigma0 * (d_ref / d50) ** beta
    sigma0: float
    beta: float


class SiBaseMaterial(BaseModel):
    family: str
    description: str
    density_g_cm3: float = Field(..., gt=0)
    young_modulus_GPa: float = Field(..., gt=0)
    poisson_ratio: float = Field(..., ge=0, le=0.5)
    volume_expansion_factor: float = Field(..., ge=1.0)
    molar_mass_g_mol: float = Field(..., gt=0)
    theoretical_capacity_mAh_g: float = Field(..., gt=0)
    morphologies: dict[str, SiMorphologyDetail]
    correlations: SiCorrelations

    def compute_BET(self, d50_nm: float) -> float:
        """BET (m²/g) = 6000 / (density_g_cm3 * d50_nm)"""
        return 6000.0 / (self.density_g_cm3 * d50_nm)

    def compute_li_diffusivity(self, d50_nm: float) -> float:
        """D = D0 * exp(-alpha * ln(d50 / d_ref))"""
        c = self.correlations
        return c.D0 * math.exp(-c.alpha * math.log(d50_nm / c.d_ref))

    def compute_electrical_conductivity(self, d50_nm: float) -> float:
        """sigma = sigma0 * (d_ref / d50) ** beta"""
        c = self.correlations
        return c.sigma0 * (c.d_ref / d50_nm) ** c.beta


class CoatingMaterial(BaseModel):
    family: str
    description: str
    density_g_cm3: float = Field(..., gt=0)
    young_modulus_GPa: float = Field(..., gt=0)
    poisson_ratio: float = Field(..., ge=0, le=0.5)
    electrical_conductivity_S_m: float = Field(..., gt=0)
    thickness_min_nm: float = Field(..., gt=0)
    thickness_max_nm: float = Field(..., gt=0)
    x_min: Optional[float] = None  # SiOx only
    x_max: Optional[float] = None
    molar_mass_g_mol: float = Field(..., gt=0)


class ConductiveAdditiveMaterial(BaseModel):
    family: str
    description: str
    density_g_cm3: float = Field(..., gt=0)
    electrical_conductivity_S_m: float = Field(..., gt=0)
    # CB-specific (optional for CNT/graphene)
    primary_particle_nm: Optional[float] = None
    aggregate_size_nm: Optional[float] = None
    BET_m2_g: Optional[float] = None
    DBP_ml_100g: Optional[float] = None
    # CNT-specific
    diameter_nm_mean: Optional[float] = None
    length_um_mean: Optional[float] = None
    aspect_ratio_mean: Optional[float] = None
    # Graphene-specific
    lateral_size_nm_mean: Optional[float] = None
    thickness_nm_mean: Optional[float] = None
    molar_mass_g_mol: float = Field(..., gt=0)


class BinderMaterial(BaseModel):
    family: str
    description: str
    density_g_cm3: float = Field(..., gt=0)
    elastic_modulus_MPa: float = Field(..., gt=0)
    poisson_ratio: float = Field(..., ge=0, le=0.5)
    film_thickness_min_nm: float = Field(..., gt=0)
    film_thickness_max_nm: float = Field(..., gt=0)
    repeat_unit_mass_g_mol: float = Field(..., gt=0)


class SEIMaterial(BaseModel):
    family: str
    description: str
    density_g_cm3: float = Field(..., gt=0)
    li_ionic_conductivity_S_m: float = Field(..., gt=0)
    electronic_conductivity_S_m: float = Field(..., gt=0)
    thickness_fresh_min_nm: float
    thickness_fresh_max_nm: float
    thickness_aged_min_nm: float
    thickness_aged_max_nm: float
    molar_mass_g_mol: float = Field(..., gt=0)


class CurrentCollectorMaterial(BaseModel):
    family: str
    description: str
    density_g_cm3: float = Field(..., gt=0)
    electrical_conductivity_S_m: float = Field(..., gt=0)
    thickness_min_um: float
    thickness_max_um: float
    surface_roughness_Ra_nm_min: float
    surface_roughness_Ra_nm_max: float
    molar_mass_g_mol: float = Field(..., gt=0)


# ---------------------------------------------------------------------------
# Root DB model — typed accessors for every code
# ---------------------------------------------------------------------------


class MaterialsDB(BaseModel):
    # Graphite grades
    graphite_artificial: GraphiteMaterial
    graphite_natural: GraphiteMaterial
    graphite_mcmb: GraphiteMaterial

    # Silicon
    si_base: SiBaseMaterial

    # Coatings
    carbon_coating: CoatingMaterial
    siox_coating: CoatingMaterial

    # Conductive additives
    cb_superp: ConductiveAdditiveMaterial
    cb_c65: ConductiveAdditiveMaterial
    cb_ketjenblack: ConductiveAdditiveMaterial
    cnt_multiwalled: ConductiveAdditiveMaterial
    graphene_flakes: ConductiveAdditiveMaterial

    # Binders
    pvdf_kynar: BinderMaterial
    cmc_sbr: BinderMaterial

    # SEI
    sei_generic: SEIMaterial

    # Current collector
    cu_foil: CurrentCollectorMaterial

    def get_graphite(self, carbon_type: str) -> GraphiteMaterial:
        return getattr(self, carbon_type)

    def get_conductive_additive(self, ca_type: str) -> ConductiveAdditiveMaterial:
        return getattr(self, ca_type)

    def get_binder(self, binder_type: str) -> BinderMaterial:
        return getattr(self, binder_type)

    def get_coating(self, coating_type: str) -> CoatingMaterial:
        return getattr(self, coating_type)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_materials_db(path: str | Path) -> MaterialsDB:
    """Load and validate materials_db.yml. Raises ValidationError on any issue."""
    raw = yaml.safe_load(Path(path).read_text())
    return MaterialsDB.model_validate(raw)
