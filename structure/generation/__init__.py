from .composition import compute_composition
from .domain import DomainGeometry, build_domain
from .carbon_packer import pack_carbon_scaffold, OblateSpheroid, PHASE_CARBON
from .si_mapper import (
    map_si_distribution,
    PHASE_SI,
    PHASE_COATING,
    PHASE_GRAPHITE,
    PHASE_PORE,
)

__all__ = [
    "compute_composition",
    "DomainGeometry",
    "build_domain",
    "pack_carbon_scaffold",
    "OblateSpheroid",
    "PHASE_CARBON",
    "map_si_distribution",
    "PHASE_SI",
    "PHASE_COATING",
    "PHASE_GRAPHITE",
    "PHASE_PORE",
]
