import numpy as np
from typing import Union
from .schema import (
    AnodeType,
    Geometry,
    GraphiteParams,
    SiliconCompositeParams,
    HardCarbonParams,
    SoftCarbonParams,
    LTOParams,
    GenerationParams,
    DefectParams,
)
from .graphite import _generate_graphite
from .silicon_composite import _generate_silicon_composite
from .hard_carbon import _generate_hard_carbon
from .soft_carbon import _generate_soft_carbon
from .lto import _generate_lto


def generate_anode_microstructure(
    run_id: int,
    seed: int,
    geometry: Geometry,
    active_type: AnodeType,
    anode_params: Union[
        GraphiteParams,
        SiliconCompositeParams,
        HardCarbonParams,
        SoftCarbonParams,
        LTOParams,
    ],
    generation: GenerationParams,
    defects: DefectParams,
) -> np.ndarray:
    """
    Generate a 3D anode microstructure based on configuration.

    Args:
        run_id: Run identification number
        seed: Random seed for reproducibility
        geometry: Spatial parameters (shape, FOV, coating thickness)
        active_type: Type of anode material
        anode_params: Anode-specific parameters (type must match active_type)
        generation: Generation control parameters (calendering, SEI, contacts, etc.)
        defects: Defect parameters for microstructure diversity

    Returns:
        binary_volume: 3D numpy array with shape geometry.shape
                      where 0 = solid (active material, binder, additives)
                      and 1 = pore (electrolyte-filled space)

    Raises:
        ValueError: If configuration is invalid or anode_params type doesn't match active_type
        RuntimeError: If generation fails
    """
    # Validate that anode_params matches active_type
    expected_type_map = {
        AnodeType.GRAPHITE: GraphiteParams,
        AnodeType.SILICON_COMPOSITE: SiliconCompositeParams,
        AnodeType.HARD_CARBON: HardCarbonParams,
        AnodeType.SOFT_CARBON: SoftCarbonParams,
        AnodeType.LTO: LTOParams,
    }

    expected_type = expected_type_map[active_type]
    if not isinstance(anode_params, expected_type):
        raise ValueError(
            f"active_type is {active_type.value} but anode_params is {type(anode_params).__name__}. "
            f"Expected {expected_type.__name__}"
        )

    # Print generation info
    print(f"\n{'='*60}")
    print(f"Generating {active_type.value} anode microstructure")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Seed: {seed}")
    print(f"Shape: {geometry.shape}")
    print(f"FOV: {geometry.field_of_view_x_um} um")
    print(f"Voxel size: {geometry.voxel_size_nm:.2f} nm")
    print(f"Coating thickness: {geometry.coating_thickness_um} um")
    print(f"Target porosity: {anode_params.target_porosity:.3f}")
    print(f"{'='*60}\n")

    # Route to specific generator based on active_type
    if active_type == AnodeType.GRAPHITE:
        return _generate_graphite(
            run_id, seed, geometry, anode_params, generation, defects
        )
    elif active_type == AnodeType.SILICON_COMPOSITE:
        return _generate_silicon_composite(
            run_id, seed, geometry, anode_params, generation, defects
        )
    elif active_type == AnodeType.HARD_CARBON:
        return _generate_hard_carbon(
            run_id, seed, geometry, anode_params, generation, defects
        )
    elif active_type == AnodeType.SOFT_CARBON:
        return _generate_soft_carbon(
            run_id, seed, geometry, anode_params, generation, defects
        )
    elif active_type == AnodeType.LTO:
        return _generate_lto(run_id, seed, geometry, anode_params, generation, defects)
    else:
        raise ValueError(f"Unknown active_type: {active_type}")


__all__ = ["generate_anode_microstructure", "schema"]
