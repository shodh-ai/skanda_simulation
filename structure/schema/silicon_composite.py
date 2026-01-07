from dataclasses import dataclass
from .enums import (
    SiliconMorphology,
    CarbonMatrixType,
    SiliconDistribution,
    CoatingType,
    ConductiveAdditiveType,
    BinderType,
    DistributionMode,
    BinderDistribution,
)


@dataclass
class SiliconCompositeParams:
    """Parameters for silicon-composite anode microstructures."""

    # Silicon content
    silicon_weight_fraction: float

    # Silicon particle parameters
    si_particle_size_nm: float
    si_size_distribution: float
    si_morphology: SiliconMorphology
    si_internal_porosity: float  # Only used if morphology is POROUS

    # Carbon matrix
    carbon_matrix_type: CarbonMatrixType
    carbon_particle_size_um: float

    # Silicon distribution
    si_distribution: SiliconDistribution

    # Void space design
    void_space_enabled: bool
    void_fraction: float

    # Silicon coating
    si_coating_enabled: bool
    si_coating_type: CoatingType
    si_coating_thickness_nm: float

    # Porosity
    target_porosity: float

    # Conductive additive parameters (inline)
    conductive_additive_type: ConductiveAdditiveType
    conductive_additive_wt_frac: float
    conductive_additive_particle_size_nm: float
    conductive_additive_distribution: DistributionMode

    # Binder parameters (inline)
    binder_type: BinderType
    binder_wt_frac: float
    binder_distribution: BinderDistribution
    binder_film_thickness_nm: float

    def __post_init__(self):
        # Validate silicon weight fraction
        if not (0.05 <= self.silicon_weight_fraction <= 0.50):
            raise ValueError(
                f"silicon_weight_fraction must be in [0.05, 0.50], got {self.silicon_weight_fraction}"
            )

        if self.silicon_weight_fraction > 0.30:
            print(
                f"Warning: silicon_weight_fraction={self.silicon_weight_fraction} > 0.30 is in research territory, may not be commercial"
            )

        # Validate Si particle size
        if not (30.0 <= self.si_particle_size_nm <= 3000.0):
            raise ValueError(
                f"si_particle_size_nm must be in [30, 3000], got {self.si_particle_size_nm}"
            )

        # Validate size distribution
        if not (0.1 <= self.si_size_distribution <= 0.8):
            raise ValueError(
                f"si_size_distribution must be in [0.1, 0.8], got {self.si_size_distribution}"
            )

        # Validate internal porosity
        if self.si_morphology == SiliconMorphology.POROUS:
            if not (0.3 <= self.si_internal_porosity <= 0.7):
                raise ValueError(
                    f"si_internal_porosity must be in [0.3, 0.7] for porous Si, got {self.si_internal_porosity}"
                )

        # Validate carbon particle size
        if not (5.0 <= self.carbon_particle_size_um <= 30.0):
            raise ValueError(
                f"carbon_particle_size_um must be in [5, 30] um, got {self.carbon_particle_size_um}"
            )

        # Validate void fraction
        if self.void_space_enabled:
            if not (0.2 <= self.void_fraction <= 0.5):
                raise ValueError(
                    f"void_fraction must be in [0.2, 0.5], got {self.void_fraction}"
                )

        # Validate Si coating
        if self.si_coating_enabled and self.si_coating_type != CoatingType.NONE:
            if self.si_coating_type == CoatingType.CARBON:
                if not (5.0 <= self.si_coating_thickness_nm <= 50.0):
                    raise ValueError(
                        f"Carbon coating thickness must be in [5, 50] nm, got {self.si_coating_thickness_nm}"
                    )
            elif self.si_coating_type == CoatingType.SILICON_OXIDE:
                if not (2.0 <= self.si_coating_thickness_nm <= 20.0):
                    raise ValueError(
                        f"SiOx coating thickness must be in [2, 20] nm, got {self.si_coating_thickness_nm}"
                    )

        # Validate porosity
        if not (0.35 <= self.target_porosity <= 0.50):
            raise ValueError(
                f"target_porosity must be in [0.35, 0.50], got {self.target_porosity}"
            )

        # Validate conductive additive
        if not (0.0 <= self.conductive_additive_wt_frac <= 0.1):
            raise ValueError(
                f"conductive_additive_wt_frac must be in [0.0, 0.1], got {self.conductive_additive_wt_frac}"
            )

        # Validate binder
        if not (0.02 <= self.binder_wt_frac <= 0.1):
            raise ValueError(
                f"binder_wt_frac must be in [0.02, 0.1], got {self.binder_wt_frac}"
            )

        if not (5.0 <= self.binder_film_thickness_nm <= 50.0):
            raise ValueError(
                f"binder_film_thickness_nm must be in [5.0, 50.0], got {self.binder_film_thickness_nm}"
            )
