"""
Single test script — runs Steps 0–3 sequentially and produces all outputs.

Steps:
  0. CompositionCalculator  → comp
  1. DomainGeometry         → domain
  2. CarbonScaffoldPacker   → packing result + carbon_label (uint8 128³)
  3. SiVfMapper             → si_vf, coating_vf, void_mask

Outputs (all in ./output/):
  carbon_scaffold.npz          3D label volume (uint8) + metadata
  si_map_result.npz            si_vf float32 + coating_vf + void_mask
  slice_orthogonal_3panel.png  Carbon-only XY/XZ/YZ BSE slices
  slice_si_3panel.png          Si+Carbon composite BSE slices
  si_vf_histogram.png          Distribution of non-zero si_vf values
  particles_3d.html            Interactive 3D: carbon ellipsoids only
  particles_3d_with_si.html    Interactive 3D: carbon + Si isosurface
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from pathlib import Path
from skimage.measure import marching_cubes

from structure.schema import load_run_config, load_materials_db, resolve
from structure.generation.composition import compute_composition
from structure.generation.domain import build_domain
from structure.generation.carbon_packer import (
    pack_carbon_scaffold,
    OblateSpheroid,
    PHASE_CARBON,
)
from structure.generation.si_mapper import (
    map_si_distribution,
    PHASE_PORE,
    PHASE_GRAPHITE,
    PHASE_SI,
    PHASE_COATING,
)

# ── Output directory ──────────────────────────────────────────────────────────
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# =============================================================================
# LOAD CONFIG + DB
# =============================================================================
cfg = load_run_config("str_gen_config.yml")
db = load_materials_db("materials_db.yml")
sim = resolve(cfg, db)

# =============================================================================
# STEP 0 — Composition
# =============================================================================
comp = compute_composition(sim)
print(comp.summary())
comp.raise_if_critical()

# =============================================================================
# STEP 1 — Domain
# =============================================================================
domain = build_domain(comp)
print(domain.summary())

# =============================================================================
# STEP 2 — Carbon Scaffold Packing
# =============================================================================
rng = np.random.default_rng(sim.seed)
packing = pack_carbon_scaffold(comp, domain, sim, rng)
print(packing.summary())


# ── Voxelize carbon particles → uint8 label map ───────────────────────────────
def voxelize_particles(
    particles: list[OblateSpheroid],
    domain,
    compression_ratio: float,
) -> np.ndarray:
    """
    Rasterize oblate spheroids into a uint8 label volume.
    Each voxel center tested analytically against every particle.
    Z compressed: z_final = z_pre × compression_ratio.
    Periodic boundary in X, Y.
    """
    nx, ny, nz = domain.nx, domain.ny, domain.nz
    vs = domain.voxel_size_nm
    volume = np.zeros((nx, ny, nz), dtype=np.uint8)

    xs = (np.arange(nx) + 0.5) * vs
    ys = (np.arange(ny) + 0.5) * vs
    zs = (np.arange(nz) + 0.5) * vs
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")

    Lx, Ly = domain.Lx_nm, domain.Ly_nm

    for p in particles:
        cx = p.center[0]
        cy = p.center[1]
        cz = p.center[2] * compression_ratio

        dx = Xg - cx
        dx = dx - Lx * np.round(dx / Lx)
        dy = Yg - cy
        dy = dy - Ly * np.round(dy / Ly)
        dz = Zg - cz

        RT = p.R.T
        dx_b = RT[0, 0] * dx + RT[0, 1] * dy + RT[0, 2] * dz
        dy_b = RT[1, 0] * dx + RT[1, 1] * dy + RT[1, 2] * dz
        dz_b = RT[2, 0] * dx + RT[2, 1] * dy + RT[2, 2] * dz

        inside = (dx_b / p.a) ** 2 + (dy_b / p.a) ** 2 + (dz_b / p.c) ** 2 <= 1.0
        volume[inside] = PHASE_CARBON

    return volume


carbon_label = voxelize_particles(packing.particles, domain, comp.compression_ratio)

carbon_vf = (carbon_label == PHASE_CARBON).mean()
actual_porosity = 1.0 - carbon_vf
print(f"\nVoxelized carbon label:")
print(
    f"  Carbon voxels  : {(carbon_label == PHASE_CARBON).sum():,}  (φ_C = {carbon_vf:.3f})"
)
print(
    f"  Pore   voxels  : {(carbon_label == PHASE_PORE).sum():,}  (φ_pore = {actual_porosity:.3f})"
)

np.savez_compressed(
    OUT / "carbon_scaffold.npz",
    label_map=carbon_label,
    voxel_size_nm=np.float32(domain.voxel_size_nm),
    compression_ratio=np.float32(comp.compression_ratio),
    N_carbon=np.int32(packing.N_placed),
)
print(f"Saved → output/carbon_scaffold.npz")

# =============================================================================
# STEP 3 — Si Vf Map
# =============================================================================
si_result = map_si_distribution(comp, domain, sim, carbon_label, packing, rng)
print(si_result.summary())

np.savez_compressed(
    OUT / "si_map_result.npz",
    si_vf=si_result.si_vf,
    coating_vf=si_result.coating_vf,
    void_mask=si_result.void_mask.astype(np.uint8),
    voxel_size_nm=np.float32(domain.voxel_size_nm),
)
print(f"Saved → output/si_map_result.npz")

# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

# Phase colors — BSE-SEM convention
COLORS = {
    PHASE_PORE: np.array([0.92, 0.92, 0.96]),  # light grey-blue — pore
    PHASE_GRAPHITE: np.array([0.29, 0.29, 0.29]),  # dark grey — graphite BSE
    PHASE_SI: np.array([0.69, 0.77, 0.87]),  # steel blue — Si bright in BSE
    PHASE_COATING: np.array([0.18, 0.18, 0.18]),  # near-black — carbon coating
}
VOID_COLOR = np.array([1.00, 0.95, 0.60])
L_um = domain.Lx_nm / 1000.0
vs_um = domain.voxel_size_nm / 1000.0


def base_rgb(label_slice: np.ndarray) -> np.ndarray:
    """Label map slice → float32 RGB (H, W, 3)."""
    H, W = label_slice.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    for pid, col in COLORS.items():
        img[label_slice == pid] = col
    return img


def composite_rgb(
    label_slice: np.ndarray,
    si_vf_slice: np.ndarray,
    void_slice: np.ndarray,
) -> np.ndarray:
    """Blend carbon base + Si vf overlay + void tint into BSE-style RGB."""
    img = base_rgb(label_slice)
    si_alpha = np.clip(si_vf_slice * 4.0, 0.0, 1.0)
    img = img * (1 - si_alpha[..., None]) + COLORS[PHASE_SI] * si_alpha[..., None]
    img[void_slice > 0] = img[void_slice > 0] * 0.7 + VOID_COLOR * 0.3
    return np.clip(img, 0.0, 1.0)


def save_3panel(
    vol_fn,  # callable(ix|iy|iz, plane) → slice
    title: str,
    fname: str,
    dark: bool = True,
    legend_handles=None,
) -> None:
    """Save a 3-panel orthogonal slice figure."""
    nx, ny, nz = carbon_label.shape
    bg = "#1a1a1a" if dark else "white"
    tc = "white" if dark else "black"
    ac = "#aaaaaa" if dark else "#333333"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=bg)
    fig.suptitle(title, color=tc, fontsize=11, fontweight="bold", y=1.01)

    panels = [
        (nz // 2, "xy", "XY (Z-mid)", "X (µm)", "Y (µm)"),
        (ny // 2, "xz", "XZ (Y-mid)", "X (µm)", "Z (µm)"),
        (nx // 2, "yz", "YZ (X-mid)", "Y (µm)", "Z (µm)"),
    ]
    for ax, (idx, plane, ptitle, xl, yl) in zip(axes, panels):
        img = vol_fn(idx, plane)
        ax.imshow(
            img, origin="lower", extent=[0, L_um, 0, L_um], interpolation="nearest"
        )
        ax.set_title(ptitle, color=tc, fontsize=10)
        ax.set_xlabel(xl, color=ac, fontsize=8)
        ax.set_ylabel(yl, color=ac, fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.set_facecolor(bg)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            frameon=False,
            fontsize=9,
            labelcolor=tc,
            bbox_to_anchor=(0.5, -0.07),
        )
    plt.tight_layout()
    plt.savefig(
        OUT / fname, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close()
    print(f"Saved → output/{fname}")


# =============================================================================
# PLOT 1 — Carbon-only 3-panel slices
# =============================================================================
def carbon_slice(idx, plane):
    if plane == "xy":
        return base_rgb(carbon_label[:, :, idx])
    if plane == "xz":
        return base_rgb(carbon_label[:, idx, :])
    if plane == "yz":
        return base_rgb(carbon_label[idx, :, :])


save_3panel(
    vol_fn=carbon_slice,
    title=f"Carbon Scaffold  |  N={packing.N_placed} particles"
    f"  |  φ_C={carbon_vf:.3f}  |  seed={sim.seed}",
    fname="slice_carbon_3panel.png",
    legend_handles=[
        mpatches.Patch(color=COLORS[PHASE_GRAPHITE], label="Graphite"),
        mpatches.Patch(color=COLORS[PHASE_PORE], label="Pore"),
    ],
)


# =============================================================================
# PLOT 2 — Si + Carbon composite 3-panel slices
# =============================================================================
def si_slice(idx, plane):
    if plane == "xy":
        return composite_rgb(
            carbon_label[:, :, idx],
            si_result.si_vf[:, :, idx],
            si_result.void_mask[:, :, idx],
        )
    if plane == "xz":
        return composite_rgb(
            carbon_label[:, idx, :],
            si_result.si_vf[:, idx, :],
            si_result.void_mask[:, idx, :],
        )
    if plane == "yz":
        return composite_rgb(
            carbon_label[idx, :, :],
            si_result.si_vf[idx, :, :],
            si_result.void_mask[idx, :, :],
        )


save_3panel(
    vol_fn=si_slice,
    title=f"Si-C Microstructure  |  dist={sim.silicon.distribution}"
    f"  |  wf_Si={comp.wf_si*100:.1f}%  |  err={si_result.mass_error_pct:.3f}%",
    fname="slice_si_3panel.png",
    legend_handles=[
        mpatches.Patch(color=COLORS[PHASE_GRAPHITE], label="Graphite"),
        mpatches.Patch(color=COLORS[PHASE_SI], label="Si (vf intensity)"),
        mpatches.Patch(color=COLORS[PHASE_PORE], label="Pore"),
        mpatches.Patch(color=VOID_COLOR, label="Void buffer"),
    ],
)

# =============================================================================
# PLOT 3 — Si vf histogram
# =============================================================================
nonzero = si_result.si_vf[si_result.si_vf > 0.001].ravel()
fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1a1a1a")
ax.set_facecolor("#1a1a1a")
ax.hist(nonzero, bins=60, color="#6699cc", edgecolor="none", alpha=0.85)
ax.axvline(
    nonzero.mean(), color="#ffcc44", lw=1.5, label=f"mean = {nonzero.mean():.4f}"
)
ax.axvline(
    float(np.median(nonzero)),
    color="#ff8844",
    lw=1.2,
    linestyle="--",
    label=f"median = {float(np.median(nonzero)):.4f}",
)
ax.set_xlabel("Si volume fraction", color="white", fontsize=10)
ax.set_ylabel("Voxel count", color="white", fontsize=10)
ax.set_title("Si vf Distribution (non-zero voxels)", color="white", fontsize=11)
ax.tick_params(colors="#888888")
ax.legend(frameon=False, labelcolor="white", fontsize=9)
for sp in ax.spines.values():
    sp.set_edgecolor("#333")
plt.tight_layout()
plt.savefig(
    OUT / "si_vf_histogram.png",
    dpi=160,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.close()
print("Saved → output/si_vf_histogram.png")


# =============================================================================
# HELPER — Ellipsoid surface mesh (for Plotly)
# =============================================================================
def ellipsoid_surface(
    p: OblateSpheroid,
    compression_ratio: float,
    n_pts: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parametric ellipsoid surface in µm, Z compressed to final domain."""
    u = np.linspace(0, 2 * np.pi, n_pts)
    v = np.linspace(0, np.pi, n_pts)
    xs = p.a * np.outer(np.cos(u), np.sin(v))
    ys = p.a * np.outer(np.sin(u), np.sin(v))
    zs = p.c * np.outer(np.ones_like(u), np.cos(v))

    pts_body = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=0)
    pts_lab = p.R @ pts_body

    shape = xs.shape
    xl = (pts_lab[0] + p.center[0]).reshape(shape) / 1000.0
    yl = (pts_lab[1] + p.center[1]).reshape(shape) / 1000.0
    zl = (pts_lab[2] * compression_ratio + p.center[2] * compression_ratio).reshape(
        shape
    ) / 1000.0
    return xl, yl, zl


def _base_3d_layout(title: str) -> go.Layout:
    Lz_um = domain.Lz_final_nm / 1000.0
    return go.Layout(
        title=dict(text=title, font=dict(color="white"), x=0.5),
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
        showlegend=True,
    )


def _carbon_traces() -> list:
    Lg = db.graphite_artificial.vis_color_rgb
    c_col = f"rgb({Lg[0]},{Lg[1]},{Lg[2]})"
    traces = []
    for i, p in enumerate(packing.particles):
        xl, yl, zl = ellipsoid_surface(p, comp.compression_ratio)
        traces.append(
            go.Surface(
                x=xl,
                y=yl,
                z=zl,
                colorscale=[[0, c_col], [1, c_col]],
                showscale=False,
                opacity=0.65,
                name="Graphite" if i == 0 else None,
                showlegend=i == 0,
                lighting=dict(ambient=0.55, diffuse=0.75, specular=0.2),
                hoverinfo="skip",
            )
        )
    return traces


# =============================================================================
# PLOT 4 — 3D: carbon ellipsoids only
# =============================================================================
fig_c = go.Figure(
    data=_carbon_traces(),
    layout=_base_3d_layout(
        f"Carbon Scaffold 3D  |  N={packing.N_placed}  |  seed={sim.seed}"
    ),
)
fig_c.write_html(str(OUT / "particles_3d.html"))
print("Saved → output/particles_3d.html")

# =============================================================================
# PLOT 5 — 3D: carbon + Si isosurface
# =============================================================================
traces_si = _carbon_traces()

si_smooth = si_result.si_vf
threshold = max(0.05, float(np.percentile(si_smooth[si_smooth > 0.001], 25)))

try:
    verts, faces, _, _ = marching_cubes(si_smooth, level=threshold)
    vx = verts[:, 0] * vs_um
    vy = verts[:, 1] * vs_um
    vz = verts[:, 2] * vs_um
    Ls = db.si_base.vis_color_rgb
    traces_si.append(
        go.Mesh3d(
            x=vx,
            y=vy,
            z=vz,
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=f"rgb({Ls[0]},{Ls[1]},{Ls[2]})",
            opacity=0.40,
            name=f"Si vf  (iso={threshold:.3f})",
            showlegend=True,
            hoverinfo="skip",
        )
    )
except Exception as e:
    print(f"  ⚠ Si isosurface skipped: {e}")

fig_si = go.Figure(
    data=traces_si,
    layout=_base_3d_layout(
        f"Si-C Microstructure 3D  |  dist={sim.silicon.distribution}"
        f"  |  wf_Si={comp.wf_si*100:.1f}%"
    ),
)
fig_si.write_html(str(OUT / "particles_3d_with_si.html"))
print("Saved → output/particles_3d_with_si.html")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 62)
print("  ALL STEPS COMPLETE")
print("=" * 62)
print(f"  output/carbon_scaffold.npz")
print(f"  output/si_map_result.npz")
print(f"  output/slice_carbon_3panel.png")
print(f"  output/slice_si_3panel.png")
print(f"  output/si_vf_histogram.png")
print(f"  output/particles_3d.html")
print(f"  output/particles_3d_with_si.html")
print("=" * 62)
