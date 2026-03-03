# Step 8: Voxelizer

**File:** `structure/voxelizer.py`

This is the final synthesis step of the geometry pipeline. It consolidates all the intermediate geometric representations (particle lists, continuous fields, surface masks) into a single **Discrete Voxel Grid** representing the final microstructure.

This grid is the "product" delivered to the user for visualization, and it serves as the input for meshing tools (like ScanIP or TetGen) for FEM analysis.

---

## 1. The Voxelization Process

The voxelizer assigns a single **Phase ID** (uint8) to every voxel $(x, y, z)$ in the domain. Since a voxel might contain overlapping signals (e.g., a bit of binder, a bit of CBD, and the edge of a particle), a strict **Priority System** determines the winner.

### 1.1. Priority Hierarchy

Higher priority overwrites lower priority.

| Priority        | Phase        | ID  | Logic                                      |
| :-------------- | :----------- | :-- | :----------------------------------------- |
| **0** (Lowest)  | **Pore**     | 0   | Default background.                        |
| **1**           | **Binder**   | 5   | Thresholded from `binder_vf`.              |
| **2**           | **CBD**      | 4   | Thresholded from `cbd_vf`.                 |
| **3**           | **Silicon**  | 2   | Thresholded from `si_vf`.                  |
| **4**           | **Coating**  | 3   | (Implicit in Si logic or separate).        |
| **5**           | **Graphite** | 1   | Inside the analytical spheroid boundary.   |
| **6** (Highest) | **SEI**      | 6   | Thresholded from `sei_vf` (surface layer). |

- **Note:** SEI has the highest priority because it is a surface layer that must coat the underlying solid. If Graphite overwrote SEI, the SEI would be buried inside the particle.

### 1.2. Continuous → Discrete Conversion

For field-based phases (Si, CBD, Binder, SEI), the continuous volume fraction (0.0–1.0) is converted to a discrete label using a threshold.

- **Thresholds:** defined in `_THRESHOLDS` (typically 0.02, i.e., 2%).
- **Preservation:** The original continuous fields (`si_vf`, `cbd_vf`) are **preserved** in the output object alongside the discrete label map. This allows downstream solvers (e.g., lithium diffusion) to use the accurate sub-voxel partial volume fractions instead of the stair-stepped integer grid.

---

## 2. Algorithm: Bounding-Box Rasterization

To voxelize millions of Graphite particles efficiently, the code avoids iterating over the entire $128^3$ grid for every particle.

**Optimization:**

1.  Compute the **Axis-Aligned Bounding Box (AABB)** of the particle in voxel coordinates.
2.  Iterate only over the voxels within this box.
3.  Check the analytical spheroid condition: $(x/a)^2 + (y/a)^2 + (z/c)^2 \le 1$.
4.  Write `PHASE_GRAPHITE` to matching voxels.

This technique is approximately **50x faster** than a naive full-grid scan.

---

## 3. Visualization Support

The `VoxelGrid` object is designed for immediate visualization.

### 3.1. Dynamic Color Mapping

Colors are **NOT** hardcoded in the voxelizer.

- **Source:** `materials_db` (via `ResolvedSimulation`).
- **Mechanism:** The `to_rgb()` method builds a 3D float array (Nx, Ny, Nz, 3) by looking up the RGB color of the material assigned to each voxel.
- **Benefit:** If you change "Graphite" to "Hard Carbon" in the config, the visualizer automatically updates to the new material's color (e.g., form dark grey to black).

### 3.2. Slicing

The `slice_rgb(axis, index)` method provides a fast way to generate 2D cross-sectional images for debugging without rendering the full 3D volume.

---

## 4. Output

The step returns a `VoxelGrid` object containing:

- **`label_map`:** `uint8` array [0...6]. The segmentation map.
- **`si_vf`, `cbd_vf`, ...:** `float16` arrays. The original partial volume fractions.
- **`voxel_size_nm`:** Physical scale.
- **`phase_colors`:** Dictionary mapping ID $\to$ (R, G, B).

### Example Usage

```python
from structure.voxelizer import voxelize_microstructure

# Inputs from previous steps
# comp, domain, sim, particles, si_result, cbd_result, sei_result

grid = voxelize_microstructure(
    comp, domain, sim, particles, si_result, cbd_result, sei_result
)

print(grid.summary())

# Save for visualization
grid.save("output/microstructure.npz")
```

**Sample Output:**

```text
==============================================================
  VOXEL GRID
==============================================================
  Shape         : 128×128×128 = 2,097,152 voxels
  Voxel size    : 78.12 nm
  Memory        : 18.0 MB

  Phase volume fractions:
    Pore        : 0.3105  #EAEAF5  ███████████
    Graphite    : 0.5520  #4A4A4A  █████████████████████
    Silicon     : 0.0510  #3498DB  ██
    CBD         : 0.0350  #000000  █
    Binder      : 0.0250  #ECF0F1  █
    SEI         : 0.0265  #F1C40F  █
==============================================================
```
