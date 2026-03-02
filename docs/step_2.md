# Step 2: Carbon Scaffold Packer

**File:** `structure/carbon_packer.py`

This module generates the solid backbone of the electrode microstructure. It places the active material (Graphite/Carbon) into the simulation domain defined in Step 1.

Because graphite particles are flake-like (anisotropic) rather than spherical, this step is mathematically complex. It utilizes **Random Sequential Adsorption (RSA)** with exact overlap detection for ellipsoids.

---

## 1. Physical Representation

### 1.1. Particle Shape: Oblate Spheroids

Graphite flakes are modeled as **Oblate Spheroids** (flattened spheres).

- **Basal Semi-axis ($a$):** The radius of the flake in its plane. Derived from $D_{50}/2$.
- **Thickness Semi-axis ($c$):** Half the thickness. $c = a / \text{Aspect Ratio}$.
- **Geometry:** $a = b > c$.

### 1.2. Shape Matrix ($Q$)

To perform efficient overlap checks, each particle is represented by a positive definite matrix $Q$. For a spheroid rotated by rotation matrix $R$:

$$
Q = R \cdot \text{diag}(a^{-2}, a^{-2}, c^{-2}) \cdot R^T
$$

The equation of the ellipsoid surface is:

$$
(x - c)^T Q (x - c) = 1
$$

In the code, we store `A_inv` (which corresponds to $Q^{-1}$) to speed up the contact potential calculation.

---

## 2. The RSA Algorithm

**Random Sequential Adsorption** is a stochastic packing method:

1.  **Generate:** Create a particle with random position $(x, y, z)$, size (drawn from Log-Normal PSD), and orientation.
2.  **Check:** Does this particle overlap with any existing particle?
3.  **Accept/Reject:**
    - **No Overlap:** Place it in the list. Update the spatial grid.
    - **Overlap:** Reject. Increment retry counter.
4.  **Repeat:** Until $N_{target}$ is reached or the "Jamming Limit" is hit.

### 2.1. Spatial Optimization (Cell Lists)

A naive overlap check is $O(N^2)$. To make this tractable for thousands of particles, the domain is divided into a grid of cells (size $\approx 2 \times r_{max}$). We only check collisions against particles in the 27 neighboring cells (3x3x3 block). This reduces complexity to roughly $O(N)$.

---

## 3. Overlap Detection: Perram-Wertheim

Detecting if two rotated ellipsoids overlap is non-trivial. The code implements the **Perram-Wertheim (1985)** criterion.

We define a contact potential function $F(\lambda)$ for $\lambda \in [0, 1]$. Two ellipsoids $A$ and $B$ separated by distance vector $r_{AB}$ are **NOT** overlapping if:

$$
\max_{\lambda \in [0,1]} \left[ \lambda(1-\lambda) r_{AB}^T \left( (1-\lambda)A^{-1} + \lambda B^{-1} \right)^{-1} r_{AB} \right] > 1
$$

- **Fast Rejection:** First, we check if the bounding spheres overlap. If not, we skip the expensive matrix math.
- **Exact Check:** We use `scipy.optimize.minimize_scalar` to find the maximum of the contact potential.

### Boundary Conditions

- **Periodic (X/Y):** When calculating vector $r_{AB}$, we use the **Minimum Image Convention** (from Step 1) to account for particles wrapping around the edges.
- **Hard Wall (Z):** Particles are explicitly clipped so they cannot penetrate $Z=0$ or $Z=L_{pre}$.

---

## 4. Orientation Statistics

Graphite flakes align during the coating process due to shear forces and gravity. They are rarely perfectly random.

We model this using the **von Mises-Fisher (vMF)** distribution (analogous to a Gaussian on a sphere).

- **Mean Direction:** $\mu = [0, 0, 1]$ (Aligned with Z-axis).
- **Concentration ($\kappa$):** Controls the spread.
  - $\kappa = 0$: Isotropic (Random).
  - $\kappa \to \infty$: Perfect alignment (all flakes flat).

**Code Logic:**
The user inputs an `orientation_degree` (0 to 1). The code maps this to $\kappa$ and samples a rotation matrix $R$ where the c-axis follows the vMF distribution.

---

## 5. Jamming Escape (Inflation)

RSA typically "jams" (cannot find space for new particles) at solid volume fractions of ~30-40% for ellipsoids. However, real calendered electrodes have densities of 60-70%.

To bridge this gap, the packer includes a **Jamming Escape Mechanism**:

1.  **RSA Phase:** Pack particles until the rejection limit is hit.
2.  **Inflation Phase:**
    - Calculate the volume deficit ($V_{target} - V_{current}$).
    - Calculate a scale factor $f$.
    - **Expand** the basal radius ($a$) of **all** placed particles by $f$.
    - _Note:_ Thickness ($c$) is **not** scaled. This simulates the flakes spreading out or the domain compressing relative to them, without making them unphysically thick.

This ensures the simulation always hits the exact target porosity defined in Step 0.

---

## 6. Output

The step returns a `PackingResult` object containing:

- **Particles:** List of `OblateSpheroid` objects (center, rotation, dimensions).
- **Diagnostics:** Number of attempts, final volume fraction, and whether inflation was triggered.

### Example Usage

```python
from structure.carbon_packer import pack_carbon_scaffold

# Inputs from previous steps
# comp: CompositionState, domain: DomainGeometry, sim: ResolvedSimulation

result = pack_carbon_scaffold(comp, domain, sim, rng)

print(result.summary())
# Output:
# ==============================================================
#   CARBON SCAFFOLD PACKING
# ==============================================================
#   Status            : INFLATED (jamming escape)
#   Particles placed  : 412 / 412
#   φ_solid achieved  : 0.3770  (target 0.3770)
#   Inflation factor  : 1.0520
# ==============================================================
```
