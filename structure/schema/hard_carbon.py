from dataclasses import dataclass
from .enums import (
    PrecursorType,
    ParticleShape,
    ConductiveAdditiveType,
    BinderType,
    DistributionMode,
    BinderDistribution,
)


@dataclass
class HardCarbonParams:
    """Parameters for hard carbon anode microstructures."""

    # Precursor and processing
    precursor_type: PrecursorType
    pyrolysis_temp_c: float

    # Particle parameters
    particle_size_um: float
    size_distribution: float
    particle_shape: ParticleShape

    # Crystallite parameters
    d002_spacing_nm: float
    crystallite_lc_nm: float
    crystallite_la_nm: float

    # Micropore parameters
    micropore_volume_fraction: float
    micropore_avg_size_nm: float
    micropore_size_distribution: float

    # Closed pore parameters
    closed_pore_volume_fraction: float
    closed_pore_avg_size_nm: float

    # Surface area
    specific_surface_area: float

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
        # Validate pyrolysis temperature
        if not (800.0 <= self.pyrolysis_temp_c <= 2200.0):
            raise ValueError(
                f"pyrolysis_temp_c must be in [800, 2200], got {self.pyrolysis_temp_c}"
            )

        # Validate particle size
        if not (1.0 <= self.particle_size_um <= 30.0):
            raise ValueError(
                f"particle_size_um must be in [1, 30], got {self.particle_size_um}"
            )

        # Validate size distribution
        if not (0.15 <= self.size_distribution <= 0.8):
            raise ValueError(
                f"size_distribution must be in [0.15, 0.8], got {self.size_distribution}"
            )

        # Validate d002 spacing (hard carbon is disordered)
        if not (0.37 <= self.d002_spacing_nm <= 0.43):
            raise ValueError(
                f"d002_spacing_nm must be in [0.37, 0.43] for hard carbon, got {self.d002_spacing_nm}"
            )

        # Validate crystallite sizes (smaller than graphite)
        if not (1.0 <= self.crystallite_lc_nm <= 5.0):
            raise ValueError(
                f"crystallite_lc_nm must be in [1, 5], got {self.crystallite_lc_nm}"
            )

        if not (1.0 <= self.crystallite_la_nm <= 10.0):
            raise ValueError(
                f"crystallite_la_nm must be in [1, 10], got {self.crystallite_la_nm}"
            )

        # Validate micropores
        if not (0.05 <= self.micropore_volume_fraction <= 0.25):
            raise ValueError(
                f"micropore_volume_fraction must be in [0.05, 0.25], got {self.micropore_volume_fraction}"
            )

        if not (0.5 <= self.micropore_avg_size_nm <= 2.0):
            raise ValueError(
                f"micropore_avg_size_nm must be in [0.5, 2.0], got {self.micropore_avg_size_nm}"
            )

        if not (0.2 <= self.micropore_size_distribution <= 0.8):
            raise ValueError(
                f"micropore_size_distribution must be in [0.2, 0.8], got {self.micropore_size_distribution}"
            )

        # Validate closed pores
        if not (0.10 <= self.closed_pore_volume_fraction <= 0.30):
            raise ValueError(
                f"closed_pore_volume_fraction must be in [0.10, 0.30], got {self.closed_pore_volume_fraction}"
            )

        if not (0.5 <= self.closed_pore_avg_size_nm <= 1.5):
            raise ValueError(
                f"closed_pore_avg_size_nm must be in [0.5, 1.5], got {self.closed_pore_avg_size_nm}"
            )

        # Validate surface area
        if not (1.0 <= self.specific_surface_area <= 500.0):
            raise ValueError(
                f"specific_surface_area must be in [1, 500] m²/g, got {self.specific_surface_area}"
            )

        # Validate porosity
        if not (0.30 <= self.target_porosity <= 0.50):
            raise ValueError(
                f"target_porosity must be in [0.30, 0.50], got {self.target_porosity}"
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
