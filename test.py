"""
Full pipeline test — Steps 0-8.

Outputs:
  output/microstructure.npz        final label_map uint8 + vf maps float16
  output/microstructure_viewer.html interactive Plotly viewer

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
from structure.generation.voxelizer import voxelize_microstructure
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


def _ellipsoid_surface(
    p: OblateSpheroid,
    cr: float,
    n_pts: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parametric ellipsoid surface in µm (post-calendering)."""
    u = np.linspace(0, 2 * np.pi, n_pts)
    v = np.linspace(0, np.pi, n_pts)
    xs = p.a * np.outer(np.cos(u), np.sin(v))
    ys = p.a * np.outer(np.sin(u), np.sin(v))
    zs = p.c * np.outer(np.ones_like(u), np.cos(v))

    pts = p.R @ np.stack([xs.ravel(), ys.ravel(), zs.ravel()])
    sh = xs.shape
    xl = (pts[0] + p.center[0]).reshape(sh) / 1000.0
    yl = (pts[1] + p.center[1]).reshape(sh) / 1000.0
    zl = (pts[2] * cr + p.center[2] * cr).reshape(sh) / 1000.0
    return xl, yl, zl


def _discrete_colorscale(
    phase_colors_hex: dict[int, str],
) -> tuple[list, list]:
    """
    Build a Plotly discrete colorscale for the 7 phase IDs.

    Returns (colorscale, tickvals, ticktext) for use with go.Surface surfacecolor.
    surfacecolor values should be phase_id / (n_phases - 1).
    """
    n = len(PHASE_NAMES)  # 7
    cs = []
    for i in range(n):
        col = phase_colors_hex.get(i, "#888888")
        lo = i / n
        hi = (i + 1) / n
        if i == 0:
            lo = 0.0
        if i == n - 1:
            hi = 1.0
        cs.append([lo, col])
        cs.append([hi - 1e-9 if i < n - 1 else 1.0, col])

    tickvals = [i / (n - 1) for i in range(n)]
    ticktext = [PHASE_NAMES[i] for i in range(n)]
    return cs, tickvals, ticktext


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
grid = None

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

        # Step 8 — Final voxelization
        print("  Step 8: Voxelizing...")
        grid = voxelize_microstructure(
            comp,
            domain,
            sim,
            packing.particles,
            si_result,
            cbd_result,
            sei_result,
        )
        print(f"    Done in {time.perf_counter()-t0:.1f}s  seed={seed}")
        print(grid.summary())
        break

    except PercolationFailed as e:
        print(
            f"    ✗ Percolation failed ({e.fraction:.4f} < {e.threshold:.4f}) "
            f"— retrying..."
        )
        continue

if grid is None:
    sys.exit(
        f"Failed to generate a percolating structure after {MAX_RETRIES} attempts."
    )

# =============================================================================
# SAVE .NPZ
# =============================================================================
grid.save(str(OUT / "microstructure.npz"))
print(f"\nSaved → output/microstructure.npz")

# =============================================================================
# PLOTLY 3D VIEWER
# =============================================================================
print("\nBuilding interactive viewer...")

vs_um = domain.voxel_size_nm / 1000.0
L_um = domain.Lx_nm / 1000.0
Lz_um = domain.Lz_final_nm / 1000.0
cr = comp.compression_ratio
n_ph = len(PHASE_NAMES)  # 7
L_nm = domain.Lx_nm

hex_col = grid.phase_colors_hex  # from materials_db
cscale, tickvals, ticktext = _discrete_colorscale(hex_col)

traces: list[go.BaseTraceType] = []

# ── Carbon ellipsoids ─────────────────────────────────────────────────────
c_col = hex_col[PHASE_GRAPHITE]
for i, p in enumerate(packing.particles):
    xl, yl, zl = _ellipsoid_surface(p, cr)
    traces.append(
        go.Surface(
            x=xl,
            y=yl,
            z=zl,
            colorscale=[[0, c_col], [1, c_col]],
            showscale=False,
            opacity=0.70,
            name="Graphite",
            legendgroup="Graphite",
            showlegend=(i == 0),
            lighting=dict(
                ambient=0.55, diffuse=0.80, specular=0.15, roughness=0.5, fresnel=0.1
            ),
            hoverinfo="skip",
        )
    )

# ── Si isosurface (marching cubes) ────────────────────────────────────────
si_arr = grid.si_vf.astype(np.float32)
si_nonzero = si_arr[si_arr > 0.001]
if si_nonzero.size > 0:
    si_iso = float(np.percentile(si_nonzero, 20))
    try:
        verts, faces, _, _ = marching_cubes(si_arr, level=si_iso)
        traces.append(
            go.Mesh3d(
                x=verts[:, 0] * vs_um,
                y=verts[:, 1] * vs_um,
                z=verts[:, 2] * vs_um,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=hex_col[PHASE_SI],
                opacity=0.35,
                name="Silicon",
                legendgroup="Silicon",
                showlegend=True,
                hoverinfo="skip",
            )
        )
    except Exception as e:
        print(f"  ⚠ Si isosurface skipped: {e}")

# ── CBD isosurface ────────────────────────────────────────────────────────
cbd_arr = grid.cbd_vf.astype(np.float32)
cbd_nonzero = cbd_arr[cbd_arr > 0.001]
if cbd_nonzero.size > 0:
    cbd_iso = float(np.percentile(cbd_nonzero, 35))
    try:
        verts_c, faces_c, _, _ = marching_cubes(cbd_arr, level=cbd_iso)
        traces.append(
            go.Mesh3d(
                x=verts_c[:, 0] * vs_um,
                y=verts_c[:, 1] * vs_um,
                z=verts_c[:, 2] * vs_um,
                i=faces_c[:, 0],
                j=faces_c[:, 1],
                k=faces_c[:, 2],
                color=hex_col[PHASE_CBD],
                opacity=0.25,
                name="CBD",
                legendgroup="CBD",
                showlegend=True,
                hoverinfo="skip",
            )
        )
    except Exception as e:
        print(f"  ⚠ CBD isosurface skipped: {e}")

# ── Three orthogonal cross-section planes (discrete phase colormap) ───────
#
# surfacecolor = phase_id / (n_ph - 1) so values span [0, 1]
# which maps to our discrete colorscale.
#
iz_mid = domain.nz // 2
iy_mid = domain.ny // 2
ix_mid = domain.nx // 2

# XY plane (Z mid-slice)
_xs = np.arange(domain.nx) * vs_um
_ys = np.arange(domain.ny) * vs_um
_Xs, _Ys = np.meshgrid(_xs, _ys, indexing="ij")
_z_plane = np.full_like(_Xs, iz_mid * vs_um)
_sc_xy = grid.label_map[:, :, iz_mid].astype(np.float32) / (n_ph - 1)

traces.append(
    go.Surface(
        x=_Xs,
        y=_Ys,
        z=_z_plane,
        surfacecolor=_sc_xy,
        colorscale=cscale,
        cmin=0.0,
        cmax=1.0,
        showscale=False,
        opacity=1.0,
        name="Slice XY",
        legendgroup="Slices",
        showlegend=True,
        hoverinfo="skip",
    )
)

# XZ plane (Y mid-slice)
_xs2 = np.arange(domain.nx) * vs_um
_zs2 = np.arange(domain.nz) * vs_um
_Xs2, _Zs2 = np.meshgrid(_xs2, _zs2, indexing="ij")
_y_plane = np.full_like(_Xs2, iy_mid * vs_um)
_sc_xz = grid.label_map[:, iy_mid, :].astype(np.float32) / (n_ph - 1)

traces.append(
    go.Surface(
        x=_Xs2,
        y=_y_plane,
        z=_Zs2,
        surfacecolor=_sc_xz,
        colorscale=cscale,
        cmin=0.0,
        cmax=1.0,
        showscale=False,
        opacity=1.0,
        name="Slice XZ",
        legendgroup="Slices",
        showlegend=False,
        hoverinfo="skip",
    )
)

# YZ plane (X mid-slice)
_ys3 = np.arange(domain.ny) * vs_um
_zs3 = np.arange(domain.nz) * vs_um
_Ys3, _Zs3 = np.meshgrid(_ys3, _zs3, indexing="ij")
_x_plane = np.full_like(_Ys3, ix_mid * vs_um)
_sc_yz = grid.label_map[ix_mid, :, :].astype(np.float32) / (n_ph - 1)

traces.append(
    go.Surface(
        x=_x_plane,
        y=_Ys3,
        z=_Zs3,
        surfacecolor=_sc_yz,
        colorscale=cscale,
        cmin=0.0,
        cmax=1.0,
        showscale=False,
        opacity=1.0,
        name="Slice YZ",
        legendgroup="Slices",
        showlegend=False,
        hoverinfo="skip",
    )
)

# ── Dummy scatter traces for clean legend entries ─────────────────────────
#
# Plotly Surface traces don't render a color swatch in the legend cleanly.
# Adding zero-point Scatter3d gives proper colored squares in the legend.
for phase_id, name in PHASE_NAMES.items():
    if phase_id == PHASE_PORE:
        continue
    frac = grid.phase_fractions.get(name, 0.0)
    traces.append(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(size=10, color=hex_col[phase_id], symbol="square"),
            name=f"{name}  ({frac*100:.1f}%)",
            legendgroup=name,
            showlegend=True,
        )
    )

# ── Phase color bar (visible colorscale on one dummy surface) ─────────────
# Add a thin colorbar as annotation rather than a trace
# Done via a 1D surface with showscale=True and the discrete colorscale
_dummy_x = np.array([[0, 0], [0, 0]])
_dummy_y = np.array([[0, 0], [0, 0]])
_dummy_z = np.array([[0, 0], [0, 0]])
_dummy_sc = np.array([[0.0, 1.0], [0.0, 1.0]])

traces.append(
    go.Surface(
        x=_dummy_x,
        y=_dummy_y,
        z=_dummy_z,
        surfacecolor=_dummy_sc,
        colorscale=cscale,
        cmin=0.0,
        cmax=1.0,
        showscale=True,
        opacity=0.0,  # invisible — colorbar only
        colorbar=dict(
            title=dict(text="Phase", side="right"),
            tickvals=tickvals,
            ticktext=ticktext,
            thickness=18,
            len=0.7,
            x=1.02,
        ),
        showlegend=False,
        hoverinfo="skip",
        name="_colorbar",
    )
)

# ── Composition annotation text ───────────────────────────────────────────
ann_lines = [
    f"<b>Composition</b>",
    f"Si:      {comp.wf_si*100:.1f} wt%  ({comp.vf_si*100:.1f} vol%)",
    f"C-matrix:{comp.wf_carbon*100:.1f} wt%  ({comp.vf_carbon*100:.1f} vol%)",
    f"CBD:     {comp.wf_additive*100:.1f} wt%",
    f"Binder:  {comp.wf_binder*100:.1f} wt%",
    f"Porosity:{comp.porosity*100:.0f}%  (target)",
    f"",
    f"<b>Domain:</b> {L_um:.0f}³ µm  |  {domain.nx}³ vx  |  {vs_um*1000:.0f}nm/vx",
    f"<b>Particles:</b> {packing.N_placed} graphite flakes",
    f"<b>Capacity:</b> {comp.capacity_total_mah:.2e} mAh  "
    f"(Si: {comp.capacity_si_fraction*100:.0f}%)",
]
ann_text = "<br>".join(ann_lines)

# ── Layout ────────────────────────────────────────────────────────────────
layout = go.Layout(
    title=dict(
        text=(
            f"Si-Graphite Microstructure  |  "
            f"seed={seed}  |  "
            f"{packing.N_placed} particles"
        ),
        font=dict(color="white", size=14),
        x=0.5,
    ),
    paper_bgcolor="#111111",
    scene=dict(
        bgcolor="#111111",
        xaxis=dict(
            title="X (µm)",
            range=[0, L_um],
            backgroundcolor="#1a1a1a",
            gridcolor="#2a2a2a",
            showbackground=True,
            color="#aaaaaa",
        ),
        yaxis=dict(
            title="Y (µm)",
            range=[0, L_um],
            backgroundcolor="#1a1a1a",
            gridcolor="#2a2a2a",
            showbackground=True,
            color="#aaaaaa",
        ),
        zaxis=dict(
            title="Z (µm)",
            range=[0, Lz_um],
            backgroundcolor="#1a1a1a",
            gridcolor="#2a2a2a",
            showbackground=True,
            color="#aaaaaa",
        ),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
    ),
    legend=dict(
        x=0.01,
        y=0.98,
        bgcolor="rgba(20,20,20,0.85)",
        bordercolor="#333",
        borderwidth=1,
        font=dict(color="white", size=11),
    ),
    annotations=[
        dict(
            text=ann_text,
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.18,
            y=0.30,
            bgcolor="rgba(20,20,20,0.80)",
            bordercolor="#444",
            borderwidth=1,
            font=dict(color="#dddddd", size=10, family="monospace"),
        )
    ],
    margin=dict(l=0, r=220, t=50, b=0),
    # Toggle buttons for layer visibility
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.01,
            y=1.07,
            xanchor="left",
            bgcolor="#222222",
            bordercolor="#444",
            font=dict(color="white", size=10),
            buttons=[
                dict(
                    label="All On",
                    method="restyle",
                    args=["visible", True],
                ),
                dict(
                    label="Particles Only",
                    method="restyle",
                    args=[
                        "visible",
                        [
                            True if "Graphite" in (t.name or "") else "legendonly"
                            for t in traces
                        ],
                    ],
                ),
                dict(
                    label="Slices Only",
                    method="restyle",
                    args=[
                        "visible",
                        [
                            (
                                True
                                if "Slice" in (t.name or "") or t.name == "_colorbar"
                                else "legendonly"
                            )
                            for t in traces
                        ],
                    ],
                ),
            ],
        ),
    ],
)

fig = go.Figure(data=traces, layout=layout)

# Save as self-contained HTML (no CDN dependency)
fig.write_html(
    str(OUT / "microstructure_viewer.html"),
    full_html=True,
    include_plotlyjs=True,  # embed Plotly JS — fully offline
    config={
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["toggleSpikelines"],
        "toImageButtonOptions": {
            "format": "svg",
            "filename": f"microstructure_seed{seed}",
        },
    },
)

print(f"Saved → output/microstructure_viewer.html")
print("\n" + "=" * 62)
print(f"  DONE  |  seed={seed}  |  {time.perf_counter()-t0:.1f}s total")
print(f"  output/microstructure.npz")
print(f"  output/microstructure_viewer.html")
print("=" * 62)
