from dataclasses import dataclass
from .enums import (
    GraphiteType,
    ConductiveAdditiveType,
    BinderType,
    DistributionMode,
    BinderDistribution,
)


@dataclass
class GraphiteParams:
    """Parameters for graphite-based anode microstructures."""

    # Graphite type
    graphite_type: GraphiteType

    # Particle sizes
    primary_particle_size_um: float
    secondary_particle_size_um: float
    size_distribution: float

    # Particle morphology
    aspect_ratio: float

    # Crystallite parameters
    d002_spacing_nm: float
    crystallite_lc_nm: float
    crystallite_la_nm: float

    # Orientation
    orientation_degree: float

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
        # Validate primary particle size based on graphite type
        if self.graphite_type == GraphiteType.NATURAL:
            if not (5.0 <= self.primary_particle_size_um <= 25.0):
                print(
                    f"Warning: primary_particle_size_um={self.primary_particle_size_um} outside typical range [5, 25] for natural graphite"
                )
        elif self.graphite_type == GraphiteType.ARTIFICIAL:
            if not (10.0 <= self.primary_particle_size_um <= 30.0):
                print(
                    f"Warning: primary_particle_size_um={self.primary_particle_size_um} outside typical range [10, 30] for artificial graphite"
                )
        elif self.graphite_type == GraphiteType.MCMB:
            if not (5.0 <= self.primary_particle_size_um <= 40.0):
                print(
                    f"Warning: primary_particle_size_um={self.primary_particle_size_um} outside typical range [5, 40] for MCMB"
                )

        # Validate secondary particle size
        if not (5.0 <= self.secondary_particle_size_um <= 50.0):
            raise ValueError(
                f"secondary_particle_size_um must be in [5.0, 50.0], got {self.secondary_particle_size_um}"
            )

        if self.secondary_particle_size_um < self.primary_particle_size_um:
            raise ValueError(
                f"secondary_particle_size_um ({self.secondary_particle_size_um}) must be >= primary_particle_size_um ({self.primary_particle_size_um})"
            )

        # Validate size distribution
        if not (0.1 <= self.size_distribution <= 0.8):
            raise ValueError(
                f"size_distribution must be in [0.1, 0.8], got {self.size_distribution}"
            )

        # Validate aspect ratio
        if self.graphite_type == GraphiteType.MCMB:
            if not (1.0 <= self.aspect_ratio <= 1.2):
                print(
                    f"Warning: aspect_ratio={self.aspect_ratio} outside typical range [1.0, 1.2] for spherical MCMB"
                )
        else:
            if not (3.0 <= self.aspect_ratio <= 10.0):
                print(
                    f"Warning: aspect_ratio={self.aspect_ratio} outside typical range [3.0, 10.0] for flake graphite"
                )

        # Validate d002 spacing
        if not (0.335 <= self.d002_spacing_nm <= 0.340):
            raise ValueError(
                f"d002_spacing_nm must be in [0.335, 0.340], got {self.d002_spacing_nm}"
            )

        # Validate crystallite sizes
        if self.graphite_type == GraphiteType.NATURAL:
            if not (50.0 <= self.crystallite_lc_nm <= 200.0):
                print(
                    f"Warning: crystallite_lc_nm={self.crystallite_lc_nm} outside typical range [50, 200] for natural graphite"
                )
        elif self.graphite_type == GraphiteType.ARTIFICIAL:
            if not (100.0 <= self.crystallite_lc_nm <= 500.0):
                print(
                    f"Warning: crystallite_lc_nm={self.crystallite_lc_nm} outside typical range [100, 500] for artificial graphite"
                )

        if not (20.0 <= self.crystallite_la_nm <= 100.0):
            print(
                f"Warning: crystallite_la_nm={self.crystallite_la_nm} outside typical range [20, 100]"
            )

        # Validate orientation degree
        if not (0.0 <= self.orientation_degree <= 1.0):
            raise ValueError(
                f"orientation_degree must be in [0.0, 1.0], got {self.orientation_degree}"
            )

        # Validate porosity
        if not (0.20 <= self.target_porosity <= 0.50):
            raise ValueError(
                f"target_porosity must be in [0.20, 0.50], got {self.target_porosity}"
            )

        # Validate conductive additive
        if not (0.0 <= self.conductive_additive_wt_frac <= 0.1):
            raise ValueError(
                f"conductive_additive_wt_frac must be in [0.0, 0.1], got {self.conductive_additive_wt_frac}"
            )

        if self.conductive_additive_type in [
            ConductiveAdditiveType.CARBON_BLACK,
            ConductiveAdditiveType.SUPER_P,
        ]:
            if not (30.0 <= self.conductive_additive_particle_size_nm <= 100.0):
                print(
                    f"Warning: conductive_additive_particle_size_nm={self.conductive_additive_particle_size_nm} outside typical range [30, 100] for {self.conductive_additive_type.value}"
                )
        elif self.conductive_additive_type == ConductiveAdditiveType.CNT:
            if not (5.0 <= self.conductive_additive_particle_size_nm <= 50.0):
                print(
                    f"Warning: conductive_additive_particle_size_nm={self.conductive_additive_particle_size_nm} outside typical range [5, 50] for CNT diameter"
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
