# Step 6: SEI Shell Adder

**File:** `structure/sei.py`

This module adds the **Solid Electrolyte Interphase (SEI)** to the electrode structure.

The SEI is a thin passivation layer (typically 5–20 nm) that forms on the active material surfaces (Graphite and Silicon) due to electrolyte decomposition. While volume-wise it is small, it is critical for electrochemical performance (impedance) and degradation (growth over time).

---

## 1. Physical Model

Since the SEI layer (e.g., 15 nm) is much thinner than the voxel size (e.g., 100 nm), it cannot be resolved as a separate layer of voxels. Instead, it is modeled as a **Sub-Voxel Volume Fraction** on the surface voxels of the active material.

### 1.1. Surface Identification

The code identifies "Surface Voxels" by convolving the solid mask with a 6-neighbor kernel.

- **Face Count ($N_f$):** For each solid voxel, we count how many of its 6 neighbors are pore space.
- **Surface Area Estimate:** $A_{surf} = N_f \times \Delta x^2$

### 1.2. Volume Calculation

The local volume of SEI in a voxel is:
$$ V*{SEI} = A*{surf} \times t*{local} $$
$$ \phi*{SEI} = \frac{V*{SEI}}{V*{voxel}} = \frac{N*f \times \Delta x^2 \times t*{local}}{\Delta x^3} = \frac{N*f \times t*{local}}{\Delta x} $$

### 1.3. Thickness Variation

Real SEI is not uniform. The code supports spatial variation using a **Gaussian Random Field (GRF)** to modulate the local thickness $t_{local}$.

- **Mean:** `sei_thickness_nm` (from config).
- **Variation:** `sei_uniformity_cv` (coefficient of variation).

$$ t*{local} = t*{mean} \times \max(0, 1 + CV \times \mathcal{N}(0, 1)\_{correlated}) $$

---

## 2. Formation Rules

SEI forms on:

1.  **Graphite Surfaces:** Voxels identified as `PHASE_GRAPHITE` with exposed pore faces.
2.  **Silicon Surfaces:** Voxels with significant Silicon volume fraction (`si_vf > 0.02`).

SEI does **NOT** form in:

- **Void Zones:** The engineered expansion gaps around Silicon are assumed to be empty/gas-filled or disconnected from the electrolyte, preventing SEI formation.

---

## 3. Output

The step returns an `SEIResult` object containing:

- **`sei_vf`:** 3D float array (0.0 - 1.0).
- **`V_sei_nm3`:** Total volume of SEI generated.
- **`surface_area_nm2`:** Total active surface area covered.
- **`mean_thickness_nm`:** Effective average thickness (calculated as $V_{total} / A_{total}$).

### Example Usage

```python
from structure.sei import add_sei_shell

# Inputs from previous steps
# comp, domain, sim, carbon_label, si_result, rng

result = add_sei_shell(comp, domain, sim, carbon_label, si_result, rng)

print(result.summary())
# Output:
# ==============================================================
#   SEI SHELL
# ==============================================================
#   SEI voxels > 0    : 450,200
#   V_SEI             : 1.2000e-11 nm³
#   Surface area      : 8.0000e+08 nm²
#   Effective thickness: 15.00 nm
#   sei_vf max        : 0.1500
# ==============================================================
```
