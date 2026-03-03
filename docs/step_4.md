# Step 4: CBD & Binder Generation

**File:** `structure/cbd_binder.py`

This module fills the interstitial void space of the electrode with the secondary solid phases: **Conductive Additive (CBD)** and **Binder**.

Unlike the large Graphite flakes (Step 2), these materials consist of nanoparticles (Carbon Black $\approx 40 \text{ nm}$) or polymer chains. Explicitly meshing billions of nanoparticles is impossible. Instead, we model them as **Sub-Voxel Continuous Fields** generated via stochastic processes (Gaussian Random Fields).

---

## 1. Conductive Binder Domain (CBD)

The CBD (e.g., Carbon Black, CNTs) forms a percolating network that provides electrical conductivity between the active material particles.

### 1.1. Generation Algorithm: Biased Gaussian Random Field (GRF)

The distribution is generated using a multi-stage stochastic process:

1.  **Base Noise:** Generate a 3D grid of white noise $N(0, 1)$.
2.  **Correlation:** Smooth the noise using a Gaussian filter. The $\sigma$ (correlation length) is derived from the aggregate size (e.g., 40 nm) relative to the voxel size.
    $$ F\_{corr} = \text{GaussianFilter}(\text{Noise}, \sigma) $$
3.  **Surface Biasing:** In real electrodes, CBD tends to aggregate near the surface of active materials. We calculate the distance transform $d(x)$ from the nearest Carbon surface and apply a bias weight:
    $$ W(x) = 0.5 + \exp\left(-2 \cdot \frac{d(x)}{d*{max}}\right) $$
    $$ F*{biased} = F\_{corr} \times W(x) $$
4.  **Thresholding:** Ranking the values of $F_{biased}$ and selecting the top $k$ voxels until the target volume $V_{CBD}$ (from Step 0) is exactly met.

### 1.2. Percolation Enforcement

A critical physical constraint is that the solid phase must conduct electrons from the current collector ($Z=0$) to the separator ($Z=L$).

- **Check:** After generating the CBD field, the code runs a **Flood Fill (BFS)** algorithm on the combined mask (Carbon + CBD).
- **Retry:** If the network does not connect $Z_{min}$ to $Z_{max}$, the GRF is regenerated with a new random seed.
- **Fallback:** If percolation fails after `MAX_CBD_RETRIES` (default 5), the best attempt is kept, and a warning is raised.

---

## 2. Binder Generation

The Binder (e.g., PVDF, CMC/SBR) acts as the "glue" holding the electrode together. It is mechanically dominant at the contact points between particles.

### 2.1. Distribution Modes

The simulation supports three morphologies defined in the config:

1.  **Necks (Default/Physical):**
    - Identifies "Contact Voxels": Carbon voxels that have $\ge 2$ carbon neighbors.
    - Places binder at these contacts.
    - Smears the binder into the immediate pore vicinity using a Gaussian blur to simulate wetting/capillary action.
2.  **Uniform:** Binder simply coats all Carbon surfaces evenly.
3.  **Patchy:** Binder forms random "blobs" on the surface (using a masked GRF).

### 2.2. Mass Conservation

Like the other steps, the Binder field is normalized:

$$
\text{binder\_vf}_{final} = \text{binder\_vf}_{raw} \times \frac{V_{target}}{\sum (\text{binder\_vf} \times V_{voxel})}
$$

---

## 3. Inputs & Interactions

This step integrates data from all previous steps:

| Input                  | Source | Usage                                                                             |
| :--------------------- | :----- | :-------------------------------------------------------------------------------- |
| **`CompositionState`** | Step 0 | Defines exactly how much CBD/Binder volume ($nm^3$) to place.                     |
| **`DomainGeometry`**   | Step 1 | Defines grid size and resolution.                                                 |
| **`carbon_label`**     | Step 2 | Defines the "skeleton" that CBD/Binder attaches to.                               |
| **`si_result`**        | Step 3 | Defines Si zones and **Void Masks**. CBD is excluded from engineered void spaces. |

---

## 4. Output

The step returns a `CBDBinderResult` object containing:

- **`cbd_vf`:** 3D float array (0.0 - 1.0).
- **`binder_vf`:** 3D float array (0.0 - 1.0).
- **`cbd_percolating`:** Boolean flag indicating if the electronic network is valid.
- **Diagnostics:** Mass error percentages (usually $\approx 0.0\%$).

### Example Usage

```python
from structure.cbd_binder import fill_cbd_binder

# Inputs:
# comp, domain, sim, carbon_label, si_result (from previous steps)

result = fill_cbd_binder(comp, domain, sim, carbon_label, si_result, rng)

print(result.summary())
# Output:
# ==============================================================
#   CBD + BINDER FILL
# ==============================================================
#   CBD volume   : 5.0000e-11 nm³  (target 5.0000e-11 nm³, err=0.000%)
#   Binder volume: 3.0000e-11 nm³  (target 3.0000e-11 nm³, err=0.000%)
#   CBD percolates in 3D: True
#   CBD voxels>0       : 240,500
#   Binder voxels>0    : 150,200
# ==============================================================
```
