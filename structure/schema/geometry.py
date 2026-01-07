from dataclasses import dataclass
from typing import Tuple


@dataclass
class Geometry:
    """Spatial parameters for the simulation domain."""

    shape: Tuple[int, int, int]
    field_of_view_x_um: float
    coating_thickness_um: float
    current_collector: str = "Cu"

    def __post_init__(self):
        # Validate shape
        if len(self.shape) != 3:
            raise ValueError(f"shape must be 3D tuple (Z, Y, X), got {self.shape}")

        for i, dim in enumerate(self.shape):
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"shape[{i}] must be positive integer, got {dim}")

        # Validate FOV
        if self.field_of_view_x_um <= 0.0:
            raise ValueError(
                f"field_of_view_x_um must be positive, got {self.field_of_view_x_um}"
            )

        # Typical FOV range: 3.0 - 20.0 um for anodes
        if not (3.0 <= self.field_of_view_x_um <= 20.0):
            print(
                f"Warning: field_of_view_x_um={self.field_of_view_x_um} is outside typical range [3.0, 20.0] um"
            )

        # Validate coating thickness
        if self.coating_thickness_um <= 0.0:
            raise ValueError(
                f"coating_thickness_um must be positive, got {self.coating_thickness_um}"
            )

        # Typical coating thickness: 20.0 - 150.0 um
        if not (20.0 <= self.coating_thickness_um <= 150.0):
            print(
                f"Warning: coating_thickness_um={self.coating_thickness_um} is outside typical range [20.0, 150.0] um"
            )

        # Validate current collector
        if self.current_collector != "Cu":
            raise ValueError(
                f"current_collector must be 'Cu' for anodes, got '{self.current_collector}'"
            )

    @property
    def voxel_size_nm(self) -> float:
        """Calculate voxel size in nanometers based on FOV and shape."""
        return (self.field_of_view_x_um * 1000.0) / self.shape[2]
