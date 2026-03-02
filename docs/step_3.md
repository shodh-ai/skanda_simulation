# Step 3: Silicon Volume-Fraction Mapper

**File:** `structure/si_mapper.py`

This module generates the spatial distribution of the active Silicon phase.

Unlike Graphite (which is explicitly modeled as discrete particles in Step 2), Silicon particles are often much smaller ($D_{50} \approx 50-100 \text{ nm}$) than the typical simulation voxel size ($\approx 100-200 \text{ nm}$). Explicitly meshing millions of tiny spheres would be computationally prohibitive.

Instead, we treat Silicon as a **Continuum Field** (`si_vf`). Each voxel contains a float value between $0.0$ and $1.0$ representing the local volume fraction of Silicon.

---

## 1. Physical Architectures

The mapper supports three distinct physical architectures for how Silicon is integrated into the Carbon matrix, controlled by the `si_distribution` config parameter.

### 1.1. Embedded (Homogeneous)

- **Description:** Silicon nanoparticles are mixed homogeneously throughout the bulk of the graphite/carbon particles.
- **Algorithm:**
  1.  Iterate through every carbon particle placed in Step 2.
  2.  Identify interior voxels.
  3.  Assign `si_vf` based on the ratio $V_{Si}/V_{C}$.
  4.  Apply spatial noise (Log-Normal) to simulate mixing inhomogeneity.

### 1.2. Surface Anchored

- **Description:** Silicon nanoparticles are chemically or mechanically anchored to the _surface_ of graphite flakes.
- **Algorithm:**
  1.  Compute the **Distance Transform** from the carbon surface.
  2.  Apply a Gaussian weight function centered at `distance = 0`.
  3.  The result is a "skin" of Silicon wrapping the carbon skeleton.

### 1.3. Core-Shell

- **Description:** Carbon is coated _around_ a Silicon core (or Silicon is encapsulated inside Carbon voids).
- **Algorithm:**
  1.  Define a "shell thickness" (from config or heuristic).
  2.  Identify the deep interior of carbon particles (distance > shell thickness).
  3.  Assign this deep core region as Silicon.

---

## 2. Advanced Features

### 2.1. Void Space (Expansion Buffer)

Silicon expands by ~300% during lithiation. Real electrodes often include engineered void space around Si to accommodate this expansion without pulverizing the matrix.

- **Model:** A boolean `void_mask` is generated around high-Si voxels.
- **Effect:** In the final geometry (Step 5), these voxels will be explicitly cleared of Binder/Conductive Additive to leave empty space.

### 2.2. Coating Layers

Modern Si anodes often use Carbon or SiOx coatings on the Si particles to stabilize the SEI.

- **Model:** Since the coating is sub-voxel ($< 10 \text{ nm}$), it is modeled as a fractional field `coating_vf`.
- **Calculation:**
  $$ \phi*{coating} = \phi*{Si} \times \frac{3 \cdot t*{coating}}{r*{Si}} $$
  This formula approximates the volume ratio of a thin shell on a sphere.

---

## 3. Mass Conservation (Normalization)

The most critical step is **Normalization**. The stochastic generation methods (Gaussian weights, noise) do not inherently guarantee that the total amount of Silicon matches the target weight fraction from Step 0.

To fix this, the code calculates a global scaling factor:

$$
\text{Scale} = \frac{V_{Si, target}}{\sum (\text{si\_vf}_{i,j,k} \times V_{voxel})}
$$

$$
\text{si\_vf}_{final} = \text{clamp}(\text{si\_vf}_{raw} \times \text{Scale}, 0.0, 1.0)
$$

This ensures that the final simulation contains **exactly** the correct mass of Silicon to within floating-point error (< 0.1%).

---

## 4. Output

The step returns a `SiMapResult` object containing:

- **`si_vf`:** 3D float array (0.0 - 1.0).
- **`coating_vf`:** 3D float array (0.0 - 1.0).
- **`void_mask`:** 3D boolean array.
- **Diagnostics:** Mass error percentage, distribution mode used.

### Example Usage

```python
from structure.si_mapper import map_si_distribution

# Inputs:
# comp: CompositionState, domain: DomainGeometry, sim: ResolvedSimulation
# carbon_label: 3D Label Matrix (Step 2 output), packing: PackingResult

result = map_si_distribution(comp, domain, sim, carbon_label, packing, rng)

print(result.summary())
# Output:
# ==============================================================
#   SI VF MAP
# ==============================================================
#   Distribution      : surface_anchored
#   V_Si target       : 1.5000e-10 nm³
#   V_Si actual       : 1.5000e-10 nm³
#   Mass error        : 0.0000%
#   si_vf  mean(>0)   : 0.2450
#   Voxels with Si>0  : 1,240,500
# ==============================================================
```
