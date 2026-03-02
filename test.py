"""
Test script for Steps 0–2 (Composition → Domain → Carbon Packing).
Outputs:
  - carbon_scaffold.npz      : 3D label volume (uint8) + particle metadata
  - slice_z_mid.png          : 2D mid-Z cross-section (BSE-style colors)
  - slice_z_mid_3panel.png   : Z / Y / X orthogonal slices
  - particles_3d.html        : interactive 3D Plotly visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import plotly.graph_objects as go
from pathlib import Path

from structure.schema import load_run_config, load_materials_db, resolve
from structure.generation.composition import compute_composition
from structure.generation.domain import build_domain
from structure.generation.carbon_packer import pack_carbon_scaffold, OblateSpheroid

# ── Load config + DB ─────────────────────────────────────────────────────────
cfg = load_run_config("str_gen_config.yml")
db = load_materials_db("materials_db.yml")
sim = resolve(cfg, db)

print(sim.voxel_size_nm)

# ── Steps 0–2 ────────────────────────────────────────────────────────────────
comp = compute_composition(sim)
domain = build_domain(comp)
rng = np.random.default_rng(sim.seed)

print(comp.summary())
print(domain.summary())

result = pack_carbon_scaffold(comp, domain, sim, rng)
print(result.summary())

# ── Voxelize carbon particles into a label volume ─────────────────────────────
# At this stage we only have carbon (phase 1) and pore (phase 0).
# Later steps will add Si, CBD, binder, SEI.
# We voxelize in FINAL domain coordinates: Z is compressed by compression_ratio.

PHASE_PORE = 0
PHASE_CARBON = 1


def voxelize_particles(
    particles: list[OblateSpheroid],
    domain,
    compression_ratio: float,
) -> np.ndarray:
    """
    Rasterize oblate spheroids into a uint8 label volume.

    Each voxel center is tested analytically against every particle.
    For each particle: transform voxel center into particle body frame,
    check if inside the oblate spheroid:
        (x'/a)² + (y'/a)² + (z'/c)² ≤ 1
    where x', y', z' are body-frame coordinates.

    Priority: last particle written wins (all same phase, so no conflict yet).

    Z coordinates are compressed: z_final = z_pre × compression_ratio
    """
    nx, ny, nz = domain.nx, domain.ny, domain.nz
    vs = domain.voxel_size_nm
    volume = np.zeros((nx, ny, nz), dtype=np.uint8)

    # Voxel center coordinates in final (post-calender) domain
    xs = (np.arange(nx) + 0.5) * vs
    ys = (np.arange(ny) + 0.5) * vs
    zs = (np.arange(nz) + 0.5) * vs
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")  # (nx, ny, nz)

    for p in particles:
        # Compress particle center Z to final coordinates
        cx = p.center[0]
        cy = p.center[1]
        cz = p.center[2] * compression_ratio

        # Displacement of every voxel center from this particle center
        dx = Xg - cx
        dy = Yg - cy
        dz = Zg - cz

        # Handle periodic X, Y (minimum image)
        Lx, Ly = domain.Lx_nm, domain.Ly_nm
        dx = dx - Lx * np.round(dx / Lx)
        dy = dy - Ly * np.round(dy / Ly)

        # Rotate into particle body frame: d_body = R.T @ d_lab
        # R columns are body axes in lab frame, so R.T rows are lab axes in body frame
        RT = p.R.T
        dx_b = RT[0, 0] * dx + RT[0, 1] * dy + RT[0, 2] * dz
        dy_b = RT[1, 0] * dx + RT[1, 1] * dy + RT[1, 2] * dz
        dz_b = RT[2, 0] * dx + RT[2, 1] * dy + RT[2, 2] * dz

        # Oblate spheroid interior test
        inside = (dx_b / p.a) ** 2 + (dy_b / p.a) ** 2 + (dz_b / p.c) ** 2 <= 1.0
        volume[inside] = PHASE_CARBON

    return volume


volume = voxelize_particles(
    result.particles,
    domain,
    comp.compression_ratio,
)

actual_porosity = 1.0 - (volume > 0).mean()
print(f"\nVoxelized volume:")
print(f"  Carbon voxels : {(volume == PHASE_CARBON).sum():,}")
print(f"  Pore voxels   : {(volume == PHASE_PORE).sum():,}")
print(f"  Actual porosity: {actual_porosity:.3f}  (target: {comp.porosity:.3f})")

# ── Save 3D volume ─────────────────────────────────────────────────────────────
out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

np.savez_compressed(
    out_dir / "carbon_scaffold.npz",
    label_map=volume,
    voxel_size_nm=np.float32(domain.voxel_size_nm),
    compression_ratio=np.float32(comp.compression_ratio),
    N_carbon=np.int32(result.N_placed),
)
print(f"\nSaved 3D volume → output/carbon_scaffold.npz")

# ── Colors from materials DB ──────────────────────────────────────────────────
PHASE_COLORS = {
    PHASE_PORE: np.array(db.graphite_artificial.vis_color_rgb or [220, 220, 255])
    / 255.0,  # pore = light blue
    PHASE_CARBON: np.array(db.graphite_artificial.vis_color_rgb) / 255.0,
}
# Override pore with a clean background
PHASE_COLORS[PHASE_PORE] = np.array([0.92, 0.92, 0.96])  # very light grey-blue


def phase_to_rgb(vol_slice: np.ndarray) -> np.ndarray:
    """Convert a 2D phase label slice → RGB image (H, W, 3)."""
    H, W = vol_slice.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    for phase_id, color in PHASE_COLORS.items():
        mask = vol_slice == phase_id
        img[mask] = color[:3]
    return img


# ── 3-panel orthogonal slice plot ─────────────────────────────────────────────
nx, ny, nz = volume.shape
ix_mid = nx // 2
iy_mid = ny // 2
iz_mid = nz // 2

slice_z = volume[:, :, iz_mid]  # XY plane (through-thickness mid)
slice_y = volume[:, iy_mid, :]  # XZ plane
slice_x = volume[ix_mid, :, :]  # YZ plane

vs_um = domain.voxel_size_nm / 1000.0  # nm → µm for axis labels

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor="#1a1a1a")
fig.suptitle(
    "Carbon Scaffold — Orthogonal Slices",
    color="white",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)

panels = [
    (slice_z, "XY  (Z mid-slice)", "X (µm)", "Y (µm)"),
    (slice_y, "XZ  (Y mid-slice)", "X (µm)", "Z (µm)"),
    (slice_x, "YZ  (X mid-slice)", "Y (µm)", "Z (µm)"),
]

for ax, (slc, title, xlabel, ylabel) in zip(axes, panels):
    img = phase_to_rgb(slc)
    L_um = domain.Lx_nm / 1000.0
    ax.imshow(img, origin="lower", extent=[0, L_um, 0, L_um], interpolation="nearest")
    ax.set_title(title, color="white", fontsize=11)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.set_facecolor("#1a1a1a")

# Legend
legend_handles = [
    mpatches.Patch(color=PHASE_COLORS[PHASE_CARBON], label="Graphite (C-matrix)"),
    mpatches.Patch(color=PHASE_COLORS[PHASE_PORE], label="Pore / void"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=10,
    labelcolor="white",
    bbox_to_anchor=(0.5, -0.06),
)

plt.tight_layout()
plt.savefig(
    out_dir / "slice_orthogonal_3panel.png",
    dpi=180,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.close()
print("Saved 2D slices → output/slice_orthogonal_3panel.png")

# ── Interactive 3D Plotly: particle ellipsoids ─────────────────────────────────
# Draw each particle as a wireframe ellipsoid surface for interactivity.
# At N=263 this is fast enough for Plotly.


def ellipsoid_surface(p: OblateSpheroid, compression_ratio: float, n_pts: int = 18):
    """
    Return (x, y, z) arrays for an ellipsoid surface mesh in lab frame,
    with Z compressed to final domain coordinates.
    """
    u = np.linspace(0, 2 * np.pi, n_pts)
    v = np.linspace(0, np.pi, n_pts)
    # Body-frame unit sphere scaled by (a, a, c)
    xs = p.a * np.outer(np.cos(u), np.sin(v))
    ys = p.a * np.outer(np.sin(u), np.sin(v))
    zs = p.c * np.outer(np.ones_like(u), np.cos(v))

    # Rotate to lab frame: [x_lab, y_lab, z_lab] = R @ [xs, ys, zs]
    shape = xs.shape
    pts_body = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=0)  # (3, N)
    pts_lab = p.R @ pts_body

    # Translate + compress Z
    xl = (pts_lab[0] + p.center[0]).reshape(shape)
    yl = (pts_lab[1] + p.center[1]).reshape(shape)
    zl = (pts_lab[2] * compression_ratio + p.center[2] * compression_ratio).reshape(
        shape
    )
    return xl / 1000.0, yl / 1000.0, zl / 1000.0  # → µm


cr = comp.compression_ratio
c_color = (
    f"rgb({db.graphite_artificial.vis_color_rgb[0]},"
    f"{db.graphite_artificial.vis_color_rgb[1]},"
    f"{db.graphite_artificial.vis_color_rgb[2]})"
)

traces = []
for p in result.particles:
    xl, yl, zl = ellipsoid_surface(p, cr)
    traces.append(
        go.Surface(
            x=xl,
            y=yl,
            z=zl,
            colorscale=[[0, c_color], [1, c_color]],
            showscale=False,
            opacity=0.75,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
            hoverinfo="skip",
        )
    )

L_um = domain.Lx_nm / 1000.0
Lz_um = domain.Lz_final_nm / 1000.0

layout = go.Layout(
    title=dict(
        text="Carbon Scaffold — 3D Particle View", font=dict(color="white"), x=0.5
    ),
    paper_bgcolor="#1a1a1a",
    scene=dict(
        bgcolor="#1a1a1a",
        xaxis=dict(
            title="X (µm)",
            range=[0, L_um],
            backgroundcolor="#1a1a1a",
            gridcolor="#333",
            showbackground=True,
            color="white",
        ),
        yaxis=dict(
            title="Y (µm)",
            range=[0, L_um],
            backgroundcolor="#1a1a1a",
            gridcolor="#333",
            showbackground=True,
            color="white",
        ),
        zaxis=dict(
            title="Z (µm)",
            range=[0, Lz_um],
            backgroundcolor="#1a1a1a",
            gridcolor="#333",
            showbackground=True,
            color="white",
        ),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    showlegend=False,
)

fig3d = go.Figure(data=traces, layout=layout)
fig3d.write_html(str(out_dir / "particles_3d.html"))
print("Saved 3D view     → output/particles_3d.html")
print("\nAll outputs in ./output/")
