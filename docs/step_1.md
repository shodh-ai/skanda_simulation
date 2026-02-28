# Step 1: Domain Geometry

**File:** `structure/step_1_domain.py`

This module defines the physical stage upon which the simulation is enacted. It establishes the coordinate system, boundary conditions, and the spatial transformation representing the manufacturing process (calendering).

Unlike Step 0 (which calculates _what_ goes into the box), Step 1 defines the **Box itself**. It produces a `DomainGeometry` object that serves as the single source of truth for all spatial queries (distance checks, voxel mapping, boundary wrapping) in downstream steps.

---

## 1. Coordinate System & Units

The simulation uses a **Cartesian Coordinate System** $(x, y, z)$.

- **Units:** Nanometers ($nm$) throughout.
- **Origin $(0, 0, 0)$:** Bottom-Left-Back corner of the domain.
- **Axes:**
  - **X & Y:** Lateral dimensions (plane parallel to the current collector).
  - **Z:** Through-plane thickness.
    - $Z = 0$: Interface with Current Collector (Cu Foil).
    - $Z = L_z$: Interface with Electrolyte/Separator.

---

## 2. Dual-Domain Model (Calendering)

To simulate the physical compression of the electrode (calendering), the system maintains two definitions of the simulation box height.

### 2.1. The Pre-Calendering Box (Expanded)

Particles are initially generated and packed into this expanded volume. This allows the Random Sequential Adsorption (RSA) algorithm to pack particles at a lower density ($\phi_{pre}$) to avoid jamming, mimicking the uncompressed coating state.

$$
L_{z, pre} = \frac{L_{final}}{CR}
$$

- where $CR$ is the `compression_ratio` (typically 1.3 – 1.6).

**All geometric placement algorithms (Step 2 & 3) operate within $[0, L_{z, pre}]$.**

### 2.2. The Final Box (Target)

This is the final physical dimension of the electrode after compression. It is the target for voxelization and Finite Element Meshing.

$$
L_{z, final} = L_{final}
$$

The final domain is always a perfect **Cube** ($L \times L \times L$) to ensure isotropic voxel resolution when mapped to a regular grid ($N^3$).

---

## 3. Boundary Conditions

The domain applies different boundary conditions (BCs) to the lateral and vertical axes.

### 3.1. Z-Axis: Hard Walls

The Z-direction represents the finite thickness of the coating.

- **Condition:** Particles cannot cross $Z=0$ or $Z=L_z$.
- **Physics:** $Z=0$ is an impenetrable solid (copper). $Z=L_z$ is the open pore/separator interface.

### 3.2. X/Y-Axes: Periodic (Toroidal)

The electrode is assumed to be infinitely wide in the lateral directions. We simulate this using **Periodic Boundary Conditions (PBC)**. A particle leaving the right side ($X > L$) instantly re-enters on the left side ($X = 0$).

#### Minimum Image Convention

To calculate distances correctly in a periodic domain (e.g., for collision detection), we must find the shortest path between two points, which might cross the boundary.

For two points $p_1, p_2$ along a periodic axis of length $L$, the displacement $\Delta x$ is:

$$
\Delta x = (x_2 - x_1)
$$

The **Minimum Image** displacement $\Delta x_{mic}$ is:

$$
\Delta x_{mic} = \Delta x - L \cdot \text{round}\left(\frac{\Delta x}{L}\right)
$$

- If $\Delta x = 0.9L$ (points are far apart in Euclidean space), the term becomes $0.9L - 1.0L = -0.1L$. The points are actually very close, wrapping around the edge.

**Code Implementation:**

```python
def min_image_vector(self, p1, p2):
    d = p2 - p1
    d[0] -= self.Lx_nm * round(d[0] / self.Lx_nm)
    d[1] -= self.Ly_nm * round(d[1] / self.Ly_nm)
    return d  # Z is unchanged (hard wall)
```

---

## 4. Voxel Mapping

The simulation eventually rasterizes the continuous vector geometry (spheres, ellipsoids) into a discrete voxel grid for numerical analysis.

- **Resolution:** $N_x = N_y = N_z = N$ (e.g., 128 or 256).
- **Voxel Size:** $\delta = L / N$.

### Mapping Function

To map a continuous coordinate $p \in \mathbb{R}^3$ to a discrete index $(i, j, k)$:

$$
i = \text{clamp}\left( \lfloor \frac{x}{\delta} \rfloor, 0, N-1 \right)
$$

### Voxel Centers

The physical coordinate of the center of voxel $(i, j, k)$ is:

$$
x_{center} = (i + 0.5) \cdot \delta
$$

---

## 5. Usage in Pipeline

This step is lightweight but foundational. It does not perform heavy computation; it simply configures the space.

```python
from structure.step_0_composition import compute_composition
from structure.step_1_domain import build_domain

# 1. Get Physics State
comp_state = compute_composition(resolved_sim)

# 2. Build Geometry Container
domain = build_domain(comp_state)

print(domain.summary())
```

**Sample Output:**

```text
==============================================================
  DOMAIN GEOMETRY
==============================================================
  Pre-calender box  : 10.0 × 10.0 × 13.0 µm
  Final box         : 10.0 × 10.0 × 10.0 µm  (cubic)
  Voxel grid        : 128 × 128 × 128
  Voxel size        : 78.12 nm  (isotropic)
  Boundary X, Y     : periodic
  Boundary Z        : hard wall
  Compression       : 1.30  (13.0 µm → 10.0 µm)
==============================================================
```
