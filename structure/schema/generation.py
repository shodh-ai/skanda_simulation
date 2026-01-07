from dataclasses import dataclass
from typing import Optional


@dataclass
class CalenderingParams:
    """Calendering effect simulation parameters."""

    compression_ratio: float
    particle_deformation: float
    orientation_enhancement: float

    def __post_init__(self):
        if not (0.5 <= self.compression_ratio <= 0.9):
            raise ValueError(
                f"compression_ratio must be in [0.5, 0.9], got {self.compression_ratio}"
            )

        if not (0.0 <= self.particle_deformation <= 1.0):
            raise ValueError(
                f"particle_deformation must be in [0.0, 1.0], got {self.particle_deformation}"
            )

        if not (0.0 <= self.orientation_enhancement <= 0.5):
            raise ValueError(
                f"orientation_enhancement must be in [0.0, 0.5], got {self.orientation_enhancement}"
            )


@dataclass
class SEILayerParams:
    """Solid Electrolyte Interphase layer parameters."""

    enabled: bool
    thickness_nm: float
    uniformity: float

    def __post_init__(self):
        if self.enabled:
            if not (5.0 <= self.thickness_nm <= 100.0):
                raise ValueError(
                    f"SEI thickness_nm must be in [5, 100], got {self.thickness_nm}"
                )

            if not (0.0 <= self.uniformity <= 1.0):
                raise ValueError(
                    f"uniformity must be in [0.0, 1.0], got {self.uniformity}"
                )


@dataclass
class ContactParams:
    """Particle contact mechanics parameters."""

    coordination_number: float
    contact_area_fraction: float

    def __post_init__(self):
        if not (4.0 <= self.coordination_number <= 8.0):
            raise ValueError(
                f"coordination_number must be in [4, 8], got {self.coordination_number}"
            )

        if not (0.05 <= self.contact_area_fraction <= 0.20):
            raise ValueError(
                f"contact_area_fraction must be in [0.05, 0.20], got {self.contact_area_fraction}"
            )


@dataclass
class PercolationParams:
    """Electrical network percolation parameters."""

    enforce_percolation: bool
    min_percolation: float

    def __post_init__(self):
        if self.enforce_percolation:
            if not (0.8 <= self.min_percolation <= 1.0):
                raise ValueError(
                    f"min_percolation must be in [0.8, 1.0], got {self.min_percolation}"
                )


@dataclass
class GenerationParams:
    """Microstructure generation control parameters."""

    calendering: CalenderingParams
    sei_layer: SEILayerParams
    contacts: ContactParams
    tortuosity_manual: Optional[float]
    percolation: PercolationParams

    def __post_init__(self):
        if self.tortuosity_manual is not None:
            if not (1.5 <= self.tortuosity_manual <= 4.0):
                raise ValueError(
                    f"tortuosity_manual must be in [1.5, 4.0] or None, got {self.tortuosity_manual}"
                )
