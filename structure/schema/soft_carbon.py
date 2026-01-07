from dataclasses import dataclass
from .enums import (
    SoftCarbonPrecursor,
    ConductiveAdditiveType,
    BinderType,
    DistributionMode,
    BinderDistribution,
)


@dataclass
class SoftCarbonParams:
    """Parameters for soft carbon anode microstructures."""

    # Precursor and processing
    precursor_type: SoftCarbonPrecursor
    graphitization_temp_c: float

    # Particle parameters
    particle_size_um: float
    size_distribution: float

    # Crystallite parameters
    d002_spacing_nm: float
    crystallite_lc_nm: float
    crystallite_la_nm: float

    # Surface properties
    surface_roughness: float

    # Microporosity
    micropore_fraction: float

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
        # Validate graphitization temperature
        if not (2000.0 <= self.graphitization_temp_c <= 2800.0):
            raise ValueError(
                f"graphitization_temp_c must be in [2000, 2800], got {self.graphitization_temp_c}"
            )

        # Validate particle size
        if not (5.0 <= self.particle_size_um <= 40.0):
            raise ValueError(
                f"particle_size_um must be in [5, 40], got {self.particle_size_um}"
            )

        # Validate size distribution
        if not (0.15 <= self.size_distribution <= 0.6):
            raise ValueError(
                f"size_distribution must be in [0.15, 0.6], got {self.size_distribution}"
            )

        # Validate d002 spacing (between graphite and hard carbon)
        if not (0.340 <= self.d002_spacing_nm <= 0.355):
            raise ValueError(
                f"d002_spacing_nm must be in [0.340, 0.355] for soft carbon, got {self.d002_spacing_nm}"
            )

        # Validate crystallite sizes
        if not (10.0 <= self.crystallite_lc_nm <= 50.0):
            raise ValueError(
                f"crystallite_lc_nm must be in [10, 50], got {self.crystallite_lc_nm}"
            )

        if not (10.0 <= self.crystallite_la_nm <= 50.0):
            raise ValueError(
                f"crystallite_la_nm must be in [10, 50], got {self.crystallite_la_nm}"
            )

        # Validate surface roughness
        if not (0.0 <= self.surface_roughness <= 1.0):
            raise ValueError(
                f"surface_roughness must be in [0.0, 1.0], got {self.surface_roughness}"
            )

        # Validate micropore fraction
        if not (0.01 <= self.micropore_fraction <= 0.10):
            raise ValueError(
                f"micropore_fraction must be in [0.01, 0.10], got {self.micropore_fraction}"
            )

        # Validate porosity
        if not (0.25 <= self.target_porosity <= 0.45):
            raise ValueError(
                f"target_porosity must be in [0.25, 0.45], got {self.target_porosity}"
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
