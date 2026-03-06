"""
Full pipeline test — Steps 0-8.

Outputs:
  output/microstructure.npz
  output/microstructure_viewer.html

Run:
  python test.py
"""

from __future__ import annotations

import sys
import time
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from skimage.measure import marching_cubes

from structure.schema import load_run_config, load_materials_db, resolve
from structure.generation.composition import compute_composition
from structure.generation.domain import build_domain
from structure.generation.carbon_packer import pack_carbon_scaffold, OblateSpheroid
from structure.generation.si_mapper import map_si_distribution
from structure.generation.cbd_binder import fill_cbd_binder
from structure.generation.calendering import apply_calendering
from structure.generation.sei import add_sei_shell
from structure.generation.percolation import validate_percolation, PercolationFailed
from structure.generation.volume import assemble_volume, MicrostructureVolume
from structure.phases import (
    PHASE_PORE,
    PHASE_GRAPHITE,
    PHASE_SI,
    PHASE_CBD,
    PHASE_NAMES,
)

OUT = Path("output")
OUT.mkdir(exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================


def _voxelize_carbon_only(
    particles: list[OblateSpheroid],
    domain,
    compression_ratio: float,
) -> np.ndarray:
    """
    Lightweight carbon-only rasterizer used to feed Steps 3-6.
    Step 8 does the authoritative full voxelization.
    """
    nx, ny, nz = domain.nx, domain.ny, domain.nz
    vs = domain.voxel_size_nm
    Lx = domain.Lx_nm
    Ly = domain.Ly_nm
    vol = np.zeros((nx, ny, nz), dtype=np.uint8)

    for p in particles:
        cx = p.center[0]
        cy = p.center[1]
        cz = p.center[2] * compression_ratio
        half = int(np.ceil(p.a / vs)) + 1

        lx = np.arange(int(cx / vs) - half, int(cx / vs) + half + 1)
        ly = np.arange(int(cy / vs) - half, int(cy / vs) + half + 1)
        lz = np.arange(int(cz / vs) - half, int(cz / vs) + half + 1)
        lz = lz[(lz >= 0) & (lz < nz)]
        if lz.size == 0:
            continue

        xs = (lx + 0.5) * vs
        ys = (ly + 0.5) * vs
        zs = (lz + 0.5) * vs
        Xs, Ys, Zs = np.meshgrid(xs, ys, zs, indexing="ij")

        dx = Xs - cx
        dx -= Lx * np.round(dx / Lx)
        dy = Ys - cy
        dy -= Ly * np.round(dy / Ly)
        dz = Zs - cz

        RT = p.R.T
        dx_b = RT[0, 0] * dx + RT[0, 1] * dy + RT[0, 2] * dz
        dy_b = RT[1, 0] * dx + RT[1, 1] * dy + RT[1, 2] * dz
        dz_b = RT[2, 0] * dx + RT[2, 1] * dy + RT[2, 2] * dz

        inside = (dx_b / p.a) ** 2 + (dy_b / p.a) ** 2 + (dz_b / p.c) ** 2 <= 1.0
        lx_w = lx % nx
        ly_w = ly % ny

        ixg = lx_w[:, None, None] * np.ones((1, len(ly), len(lz)), dtype=np.intp)
        iyg = ly_w[None, :, None] * np.ones((len(lx), 1, len(lz)), dtype=np.intp)
        izg = lz[None, None, :] * np.ones((len(lx), len(ly), 1), dtype=np.intp)
        vol[ixg[inside], iyg[inside], izg[inside]] = PHASE_GRAPHITE

    return vol


# =============================================================================
# LOAD + RESOLVE
# =============================================================================
print("Loading config and materials DB...")
cfg = load_run_config("str_gen_config.yml")
db = load_materials_db("materials_db.yml")
sim = resolve(cfg, db)

# =============================================================================
# STEPS 0–1
# =============================================================================
comp = compute_composition(sim)
domain = build_domain(comp)
print(comp.summary())
print(domain.summary())
comp.raise_if_critical()

# =============================================================================
# STEPS 2–7  (with percolation retry loop)
# =============================================================================
MAX_RETRIES = 10
vol = None

for attempt in range(MAX_RETRIES):
    seed = sim.seed + attempt
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    print(f"\n[Attempt {attempt+1}/{MAX_RETRIES}]  seed={seed}")

    # Step 2 — Carbon scaffold
    print("  Step 2: Carbon packing...")
    packing = pack_carbon_scaffold(comp, domain, sim, rng)
    print(
        f"    {packing.N_placed}/{packing.N_target} particles  "
        f"φ={packing.phi_achieved:.3f}  inflated={packing.inflated}"
    )

    # Intermediate carbon label for Steps 3-6
    carbon_label = _voxelize_carbon_only(
        packing.particles, domain, comp.compression_ratio
    )

    # Step 3 — Si vf map
    print("  Step 3: Si distribution...")
    si_result = map_si_distribution(comp, domain, sim, carbon_label, packing, rng)
    print(f"    err={si_result.mass_error_pct:.3f}%  dist={si_result.distribution}")

    # Step 4 — CBD + Binder
    print("  Step 4: CBD + Binder fill...")
    cbd_result = fill_cbd_binder(comp, domain, sim, carbon_label, si_result, rng)
    print(
        f"    CBD err={cbd_result.cbd_mass_error_pct:.3f}%  "
        f"percolates={cbd_result.cbd_percolating}"
    )

    # Step 5 — Calendering
    print("  Step 5: Calendering transform...")
    si_result, cbd_result = apply_calendering(
        packing.particles, comp, domain, si_result, cbd_result, sim
    )

    # Step 6 — SEI shell
    print("  Step 6: SEI shell...")
    sei_result = add_sei_shell(comp, domain, sim, carbon_label, si_result, rng)
    print(
        f"    t_eff={sei_result.mean_thickness_nm:.2f}nm  "
        f"SA={sei_result.surface_area_nm2:.3e} nm²"
    )

    # Step 7 — Percolation validation
    print("  Step 7: Percolation check...")
    try:
        perc = validate_percolation(
            comp,
            domain,
            sim,
            carbon_label,
            si_result,
            cbd_result,
            sei_result,
        )
        print(
            f"    electronic={perc.electronic_fraction:.4f}  "
            f"ionic={perc.ionic_fraction:.4f}  ✓"
        )

        print(" Step 8: Assembling volume...")
        vol = assemble_volume(
            comp=comp,
            domain=domain,
            sim=sim,
            packing=packing,
            carbon_label=carbon_label,
            si_result=si_result,
            cbd_result=cbd_result,
            sei_result=sei_result,
            perc_result=perc,
        )
        print(f" Done in {time.perf_counter()-t0:.1f}s seed={seed}")
        print(vol.summary())
        break

    except PercolationFailed as e:
        print(
            f"    ✗ Percolation failed ({e.fraction:.4f} < {e.threshold:.4f}) "
            f"— retrying..."
        )
        continue

if vol is None:
    sys.exit(
        f"Failed to generate a percolating structure after {MAX_RETRIES} attempts."
    )

# =============================================================================
# SAVE .NPZ
# =============================================================================
vol.save(str(OUT / "microstructure.npz"))
print(f"\nSaved → output/microstructure.npz")
