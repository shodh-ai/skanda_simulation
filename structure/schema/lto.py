from dataclasses import dataclass
from .enums import (
    LTOSynthesis,
    ParticleShape,
    CarbonNanomaterialType,
    ConductiveAdditiveType,
    BinderType,
    DistributionMode,
    BinderDistribution,
)


@dataclass
class LTOParams:
    """Parameters for lithium titanate (LTO) anode microstructures."""

    # Synthesis method
    synthesis_method: LTOSynthesis

    # Particle parameters
    primary_particle_size_nm: float
    secondary_particle_size_um: float
    size_distribution: float
    particle_shape: ParticleShape

    # Crystal properties
    crystallinity: float
    ti3_plus_fraction: float

    # Carbon coating on LTO particles
    carbon_coating_enabled: bool
    carbon_coating_thickness_nm: float
    carbon_coating_coverage: float

    # Carbon nanomaterial incorporation
    carbon_nano_enabled: bool
    carbon_nano_type: CarbonNanomaterialType
    carbon_nano_wt_frac: float

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
        # Validate primary particle size
        if not (20.0 <= self.primary_particle_size_nm <= 2000.0):
            raise ValueError(
                f"primary_particle_size_nm must be in [20, 2000], got {self.primary_particle_size_nm}"
            )

        if self.primary_particle_size_nm > 500.0:
            print(
                f"Warning: primary_particle_size_nm={self.primary_particle_size_nm} > 500 nm may have poor rate performance"
            )

        # Validate secondary particle size
        if not (1.0 <= self.secondary_particle_size_um <= 20.0):
            raise ValueError(
                f"secondary_particle_size_um must be in [1, 20], got {self.secondary_particle_size_um}"
            )

        # Validate size distribution
        if not (0.15 <= self.size_distribution <= 0.6):
            raise ValueError(
                f"size_distribution must be in [0.15, 0.6], got {self.size_distribution}"
            )

        # Validate crystallinity
        if not (0.7 <= self.crystallinity <= 1.0):
            raise ValueError(
                f"crystallinity must be in [0.7, 1.0], got {self.crystallinity}"
            )

        # Validate Ti3+ fraction
        if not (0.0 <= self.ti3_plus_fraction <= 0.05):
            raise ValueError(
                f"ti3_plus_fraction must be in [0.0, 0.05], got {self.ti3_plus_fraction}"
            )

        # Validate carbon coating
        if self.carbon_coating_enabled:
            if not (2.0 <= self.carbon_coating_thickness_nm <= 20.0):
                raise ValueError(
                    f"carbon_coating_thickness_nm must be in [2, 20], got {self.carbon_coating_thickness_nm}"
                )

            if not (0.5 <= self.carbon_coating_coverage <= 1.0):
                raise ValueError(
                    f"carbon_coating_coverage must be in [0.5, 1.0], got {self.carbon_coating_coverage}"
                )

        # Validate carbon nanomaterial
        if self.carbon_nano_enabled:
            if not (0.01 <= self.carbon_nano_wt_frac <= 0.10):
                raise ValueError(
                    f"carbon_nano_wt_frac must be in [0.01, 0.10], got {self.carbon_nano_wt_frac}"
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
