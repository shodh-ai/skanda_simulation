# Step 0: Composition Calculator

**File:** `structure/generation/composition.py`

The **Composition Calculator** is the physics and stoichiometry engine of the simulation pipeline. It runs **once** at the very beginning of the generation process. Its primary responsibility is to translate user-defined weight percentages and material properties into exact geometric constraints (volumes, particle counts) and electrochemical predictions.

This step ensures **Mass Conservation**. All downstream steps (Geometry Generation, Meshing, FEM) read from the immutable `CompositionState` produced here, guaranteeing that the final geometry matches the requested stoichiometry exactly.

---

## 1. Mathematical Models

### 1.1. Mass-to-Volume Conversion

User inputs are typically in **Weight Fraction** ($w_i$). The simulation domain requires **Volume Fraction** ($v_i$).

Given the densities ($\rho_i$) of Silicon, Carbon, Additive, and Binder, the specific volume contribution of each phase is calculated. The volume fraction of phase $i$ relative to the total _solid_ volume is:

$$
v_i = \frac{w_i / \rho_i}{\sum_{j} (w_j / \rho_j)}
$$

where $j \in \{Si, Carbon, Additive, Binder\}$.

### 1.2. Domain Volume & Porosity

The total simulation domain is a cube of side length $L$.

- **Total Volume:** $V_{domain} = L^3$
- **Target Solid Volume:** $V_{solid} = V_{domain} \times (1 - \epsilon_{target})$

The specific volume of each phase in the domain is then:

$$
V_i [nm^3] = v_i \times V_{solid}
$$

### 1.3. Particle Size Distribution (PSD) Correction

A critical feature of this calculator is the **Log-Normal Volume Correction**.

Users define particle size by the **Median Diameter** ($D_{50}$). However, calculating particle count $N$ as $V_{total} / V(D_{50})$ is incorrect for a distribution, as large particles in the "fat tail" of the distribution contribute disproportionately to the volume.

We assume a Log-Normal distribution. The relationship between the **Mean Volume** and the **Median Volume** depends on the Coefficient of Variation ($CV = \sigma / \mu$):

$$
\text{Correction Factor} = \frac{E[V]}{V_{median}} = (1 + CV^2)^{4.5}
$$

**Code Implementation:**

```python
correction = (1.0 + size_cv**2) ** (9.0 / 2.0)
```

- If $CV = 0$ (monodisperse), Correction = 1.0.
- If $CV = 0.3$ (typical), Correction $\approx 1.47$. This means using the median size would result in ~50% too many particles.

### 1.4. Particle Geometry

- **Silicon:** Modelled as spheres.
  $$ V*{median} = \frac{4}{3} \pi (D*{50}/2)^3 $$
- **Carbon (Graphite):** Modelled as **Oblate Spheroids** (flattened disks).
  - Basal semi-axis $a = D_{50} / 2$
  - Aspect Ratio $AR = a / c$
  - Thickness semi-axis $c = a / AR$
    $$ V\_{median} = \frac{4}{3} \pi a^2 c $$

**Final Particle Count:**

$$
N_i = \text{round}\left( \frac{V_{i, total}}{V_{median} \times \text{Correction Factor}} \right)
$$

### 1.5. Calendering & RSA Limits

Battery electrodes are compressed (calendered) during manufacturing. The simulation generates particles in an **uncompressed (expanded)** state and then compresses the domain.

- **Compression Ratio ($CR$):** $L_{z, final} / L_{z, initial}$
- **Pre-Calendering Height:** $L_{z, pre} = L / CR$
- **Pre-Calendering Solid Fraction:**
  $$ \phi*{pre} = (1 - \epsilon*{target}) \times CR $$

**The Jamming Limit:**
Random Sequential Adsorption (RSA)—the algorithm used to place particles—cannot exceed a specific density. For oblate spheroids, this mathematical limit is approximately 40-50%.

$$
\text{If } \phi_{pre} > 0.45 \implies \text{CRITICAL ERROR}
$$

_This check prevents the simulation from running an infinite loop trying to pack particles into a space that is physically too small._

---

## 2. Electrochemistry & Stoichiometry

The calculator derives theoretical cell properties using data from the `MaterialsDB`.

### 2.1. Moles

$$
n_i [\text{mol}] = \frac{\text{Mass}_i [g]}{\text{Molar Mass}_i [g/mol]}
$$

- For Binders (polymers), the molar mass of the **repeat unit** is used.

### 2.2. Capacity

$$
Q_i [\text{mAh}] = \text{Mass}_i [g] \times Q_{theoretical, i} [\text{mAh}/g]
$$

- **Total Capacity:** $Q_{total} = Q_{Si} + Q_{Graphite}$ (Additives and Binders are assumed inactive).
- **Volumetric Capacity:** $Q_{vol} = Q_{total} / V_{domain}$

---

## 3. Data Structures

### Input: `ResolvedSimulation`

A fully validated configuration object containing:

- `RunConfig`: User settings (weights, porosity, dimensions).
- `MaterialsDB`: Material constants (density, molar mass, theoretical capacity).

### Output: `CompositionState`

An immutable dataclass containing the "Truth" for the simulation.

| Category      | Key Fields                     | Description                            |
| :------------ | :----------------------------- | :------------------------------------- |
| **Volume**    | `V_si_nm3`, `V_carbon_nm3`     | Exact volume to generate.              |
| **Counts**    | `N_carbon`                     | Target count for RSA solver.           |
| **Counts**    | `N_si`                         | Target count (if explicitly resolved). |
| **Geometry**  | `carbon_a_nm`, `carbon_c_nm`   | Spheroid axes dimensions.              |
| **Physics**   | `phi_solid_pre`                | Packing density check.                 |
| **Chemistry** | `mol_si`, `capacity_total_mah` | Electrochemical predictions.           |

---

## 4. Validation Rules

The `_validate` function runs heuristics to ensure the simulation is viable before expensive generation starts.

1.  **Statistical Significance:**
    - **Rule:** `N_carbon >= 20`
    - **Reason:** If the domain is too small to hold at least 20 graphite particles, the resulting microstructure will not be statistically representative of a real electrode.

2.  **Computational Feasibility:**
    - **Rule:** `N_carbon <= 5000`
    - **Reason:** Explicit RSA packing scales poorly ($O(N^2)$ or worse depending on density). High counts suggest the user should decrease the domain size or increase voxel resolution.

3.  **Jamming Limit (Critical):**
    - **Rule:** `phi_solid_pre <= 0.45`
    - **Reason:** See Section 1.5. It is mathematically impossible to randomly place spheroids denser than this without overlaps.

4.  **Sub-Voxel Silicon:**
    - **Condition:** $D_{50, Si} < \text{Voxel Size}$
    - **Action:** Returns an `[INFO]` warning. The Generator will switch strategies from "Discrete Particle Placement" to "Continuum Field Generation" for silicon.

---

## 5. Usage Example

```python
from structure.schema.resolve import resolve
from structure import compute_composition

# 1. Load Configuration
resolved_sim = resolve(run_config, materials_db)

# 2. Compute Composition
state = compute_composition(resolved_sim)

# 3. Check for errors
state.raise_if_critical()

# 4. Print Report
print(state.summary())
```

**Sample Output:**

```text
==============================================================
  COMPOSITION STATE
==============================================================
  Domain          : 10.0 µm cube  (1.000e-09 cm³)
  Voxel size      : 78.1 nm/voxel  (128³)

  Weight fractions (of total dry electrode mass):
    Si            : 0.1500
    C-matrix      : 0.7700
    Additive      : 0.0500
    Binder        : 0.0300
    ─────────────   1.0000

  Particle counts:
    N_carbon      : 412  (explicit RSA, d50=5.0µm, AR=5.0)
    N_Si          : 2.14e+05  (statistical fill, d50=60nm)

  Theoretical capacity:
    Si            : 1.2400e-06 mAh  (61.5% of total)
    C-matrix      : 7.7500e-07 mAh
    Total         : 2.0150e-06 mAh
    Volumetric    : 2015.00 mAh/cm³

  Pre-calendering domain:
    compression   : 1.30
    φ_solid_pre   : 0.377  (37.7%)  [RSA limit ≈ 45%]
==============================================================
```
