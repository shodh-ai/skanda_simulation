"""
Canonical phase ID registry.

Colors are NOT hardcoded here. They come from the materials_db at
runtime via build_phase_colors(sim). This keeps colors consistent
with whatever materials are actually configured for this run.

Import PHASE_* constants everywhere; call build_phase_colors(sim) once
at the start of any visualization or voxelization step.
"""

from __future__ import annotations

from structure.schema.resolved import ResolvedSimulation

# ---------------------------------------------------------------------------
# Phase IDs (uint8 — stable, never reorder)
# ---------------------------------------------------------------------------
PHASE_PORE: int = 0
PHASE_GRAPHITE: int = 1
PHASE_SI: int = 2
PHASE_COATING: int = 3
PHASE_CBD: int = 4
PHASE_BINDER: int = 5
PHASE_SEI: int = 6

# Priority for discrete label assignment (higher overwrites lower)
LABEL_PRIORITY: dict[int, int] = {
    PHASE_PORE: 0,
    PHASE_BINDER: 1,
    PHASE_CBD: 2,
    PHASE_SI: 3,
    PHASE_COATING: 4,
    PHASE_GRAPHITE: 5,
    PHASE_SEI: 6,
}

# Human-readable names (used in summaries, legends, axis labels)
PHASE_NAMES: dict[int, str] = {
    PHASE_PORE: "Pore",
    PHASE_GRAPHITE: "Graphite",
    PHASE_SI: "Silicon",
    PHASE_COATING: "Coating",
    PHASE_CBD: "CBD",
    PHASE_BINDER: "Binder",
    PHASE_SEI: "SEI",
}


def _rgb_list_to_float(rgb: list[int]) -> tuple[float, float, float]:
    """Convert [R, G, B] in 0-255 range to (r, g, b) in 0.0-1.0 range."""
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def build_phase_colors(
    sim: ResolvedSimulation,
) -> dict[int, tuple[float, float, float]]:
    """
    Build the phase → float RGB color map from the actual resolved materials.

    Reads vis_color_rgb from:
      PHASE_GRAPHITE : sim.carbon.material.vis_color_rgb
      PHASE_SI       : sim.silicon.material.vis_color_rgb   (si_base)
      PHASE_COATING  : sim.silicon.coating_material.vis_color_rgb (if enabled)
      PHASE_CBD      : sim.additive.vis_color_rgb
      PHASE_BINDER   : sim.binder.vis_color_rgb
      PHASE_SEI      : sim.sei.material.vis_color_rgb
      PHASE_PORE     : fixed neutral (no material assigned to pore)

    Args:
      sim : ResolvedSimulation

    Returns:
      dict[phase_id → (r, g, b)] with floats in [0, 1]
    """
    colors: dict[int, tuple[float, float, float]] = {}

    # Pore — no material, fixed neutral background
    colors[PHASE_PORE] = (0.92, 0.92, 0.96)
    # Graphite
    colors[PHASE_GRAPHITE] = _rgb_list_to_float(sim.carbon.material.vis_color_rgb)
    # Silicon
    colors[PHASE_SI] = _rgb_list_to_float(sim.silicon.vis_color_rgb)
    # Coating (carbon_coating or siox_coating)
    colors[PHASE_COATING] = _rgb_list_to_float(
        sim.silicon.coating_material.vis_color_rgb
    )
    # Conductive additive
    colors[PHASE_CBD] = _rgb_list_to_float(sim.additive.vis_color_rgb)
    # Binder
    colors[PHASE_BINDER] = _rgb_list_to_float(sim.binder.vis_color_rgb)
    # SEI
    colors[PHASE_SEI] = _rgb_list_to_float(sim.sei_material.vis_color_rgb)

    return colors


def build_phase_colors_hex(sim: ResolvedSimulation) -> dict[int, str]:
    """
    Same as build_phase_colors but returns hex strings

    Returns dict[phase_id → '#RRGGBB']
    """
    hex_colors: dict[int, str] = {}

    _material_map = {
        PHASE_PORE: lambda: "#EBEBF5",
        PHASE_GRAPHITE: lambda: sim.carbon.material.vis_color_hex,
        PHASE_SI: lambda: sim.silicon.vis_color_hex,
        PHASE_COATING: lambda: sim.silicon.coating_material.vis_color_hex,
        PHASE_CBD: lambda: sim.additive.vis_color_hex,
        PHASE_BINDER: lambda: sim.binder.vis_color_hex,
        PHASE_SEI: lambda: sim.sei_material.vis_color_hex,
    }

    for phase_id in PHASE_NAMES:
        hex_colors[phase_id] = _material_map[phase_id]()

    return hex_colors
