"""
Pydantic schema for materials_db.yml.
Validates all 15 material entries and exposes typed accessors.
"""

from __future__ import annotations
import math
from typing import Optional, List
from pydantic import BaseModel, Field
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
    vis_color_hex: str
    vis_color_rgb: list[int]


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
    vis_color_hex: str
    vis_color_rgb: list[int]
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
    vis_color_hex: str
    vis_color_rgb: list[int]


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
    vis_color_hex: str
    vis_color_rgb: list[int]


class BinderMaterial(BaseModel):
    family: str
    description: str
    density_g_cm3: float = Field(..., gt=0)
    elastic_modulus_MPa: float = Field(..., gt=0)
    poisson_ratio: float = Field(..., ge=0, le=0.5)
    film_thickness_min_nm: float = Field(..., gt=0)
    film_thickness_max_nm: float = Field(..., gt=0)
    repeat_unit_mass_g_mol: float = Field(..., gt=0)
    vis_color_hex: str
    vis_color_rgb: list[int]


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
    vis_color_hex: str
    vis_color_rgb: list[int]


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
    vis_color_hex: str
    vis_color_rgb: list[int]


# ---------------------------------------------------------------------------
# Cathode active materials
# ---------------------------------------------------------------------------


class CathodeMaterial(BaseModel):
    family: str
    formula: str
    description: str

    # Physical
    density_g_cm3: float = Field(..., gt=0)
    molar_mass_g_mol: float = Field(..., gt=0)
    theoretical_capacity_mAh_g: float = Field(..., gt=0)
    practical_capacity_mAh_g: Optional[float] = None  # LCO only
    max_concentration_mol_m3: float = Field(..., gt=0)

    # Stoichiometry window
    stoichiometry_charged: float = Field(..., ge=0.0, le=1.0)
    stoichiometry_discharged: float = Field(..., ge=0.0, le=1.0)

    # Electrochemical
    li_diffusivity_m2_s: float = Field(..., gt=0)
    electrical_conductivity_S_m: float = Field(..., gt=0)
    exchange_current_density_A_m2: float = Field(..., gt=0)
    charge_transfer_coefficient: float = Field(..., ge=0.0, le=1.0)

    # Voltage window
    voltage_min_V: float = Field(..., gt=0)
    voltage_max_V: float = Field(..., gt=0)

    # OCV curve reference key (resolves to file under materialdb/ocv_curves/)
    ocv_curve: str

    # Degradation
    capacity_fade_mechanism: str
    cycle_life_typical_cycles: int = Field(..., gt=0)

    vis_color_hex: str
    vis_color_rgb: list[int]

    @property
    def capacity_mAh_g(self) -> float:
        """Practical capacity if available, else theoretical."""
        return self.practical_capacity_mAh_g or self.theoretical_capacity_mAh_g


# ---------------------------------------------------------------------------
# Electrolytes
# ---------------------------------------------------------------------------


class ElectrolyteMaterial(BaseModel):
    family: str
    aliases: list[str] = Field(default_factory=list)
    description: str

    # Formulation
    solvent: dict[str, float]  # e.g. {"EC": 0.50, "DMC": 0.50}
    salt: str
    salt_concentration_mol_L: float = Field(..., gt=0)

    # Bulk transport at 298.15 K (Valoen & Reimers 2005 unless noted)
    ionic_conductivity_S_m: float = Field(..., gt=0)
    li_diffusivity_m2_s: float = Field(..., gt=0)
    transference_number: float = Field(..., ge=0.0, le=1.0)
    thermodynamic_factor: float = Field(..., gt=0)

    # Physical
    density_g_cm3: float = Field(..., gt=0)
    viscosity_mPas: float = Field(..., gt=0)
    dielectric_constant: float = Field(..., gt=0)

    # If true, concentration-dependent expressions are loaded from
    # materialdb/electrolyte_expressions/ at simulation time
    concentration_dependent: bool = False
    temperature_dependent: bool = False
    expression_key: Optional[str] = None

    electrochemical_window_V: list[float]
    compatible_anodes: list[str]
    compatible_cathodes: list[str]


# ---------------------------------------------------------------------------
# Separators
# ---------------------------------------------------------------------------


class SeparatorMaterial(BaseModel):
    family: str
    description: str
    construction: str  # e.g. "trilayer_PP_PE_PP"

    # Geometry
    thickness_um: float = Field(..., gt=0)
    porosity: float = Field(..., gt=0, le=1.0)
    ceramic_layer_thickness_um: Optional[float] = None
    mean_pore_size_nm: float = Field(..., gt=0)
    pore_size_distribution: str  # "unimodal" | "bimodal"

    # Transport
    tortuosity: float = Field(..., ge=1.0)
    macmullin_number: float = Field(..., ge=1.0)  # = tortuosity / porosity
    bruggeman_exponent: float = Field(..., gt=0)

    # Mechanical
    tensile_strength_MD_MPa: float = Field(..., gt=0)
    tensile_strength_TD_MPa: float = Field(..., gt=0)
    puncture_strength_gf: float = Field(..., gt=0)

    # Thermal
    # None for monolayer PP/PE — no shutdown layer present
    thermal_shutdown_temp_C: Optional[float] = None
    meltdown_temp_C: float = Field(..., gt=0)

    vis_color_hex: str
    vis_color_rgb: list[int]


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

    # Cathode active materials
    nmc811: CathodeMaterial
    nmc622: CathodeMaterial
    nmc532: CathodeMaterial
    nmc111: CathodeMaterial
    lfp: CathodeMaterial
    lfp_halfcell_dummy: CathodeMaterial
    nca: CathodeMaterial
    lco: CathodeMaterial

    # Electrolytes
    LiPF6_EC_DMC_1M: ElectrolyteMaterial
    LiPF6_EC_DEC_1M: ElectrolyteMaterial
    LiPF6_EC_EMC_3_7_1M: ElectrolyteMaterial
    LiPF6_FEC_DMC_1M: ElectrolyteMaterial
    LiFSI_DME_1M: ElectrolyteMaterial

    # Separators
    Celgard2325: SeparatorMaterial
    Celgard2500: SeparatorMaterial
    Celgard3501: SeparatorMaterial
    ceramic_PP: SeparatorMaterial

    # Cathode current collector
    al_foil: CurrentCollectorMaterial

    def get_graphite(self, carbon_type: str) -> GraphiteMaterial:
        return getattr(self, carbon_type)

    def get_conductive_additive(self, ca_type: str) -> ConductiveAdditiveMaterial:
        return getattr(self, ca_type)

    def get_binder(self, binder_type: str) -> BinderMaterial:
        return getattr(self, binder_type)

    def get_coating(self, coating_type: str) -> CoatingMaterial:
        return getattr(self, coating_type)

    def get_cathode(self, cathode_key: str) -> CathodeMaterial:
        return getattr(self, cathode_key)

    def get_electrolyte(self, electrolyte_key: str) -> ElectrolyteMaterial:
        return getattr(self, electrolyte_key)

    def get_separator(self, separator_key: str) -> SeparatorMaterial:
        return getattr(self, separator_key)

    def get_current_collector(self, cc_key: str) -> CurrentCollectorMaterial:
        return getattr(self, cc_key)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_materials_db(path: str | Path) -> MaterialsDB:
    """Load and validate materials_db.yml. Raises ValidationError on any issue."""
    raw = yaml.safe_load(Path(path).read_text())
    return MaterialsDB.model_validate(raw)
