# Step 5: Calendering Transformation

**File:** `structure/calendering.py`

This module simulates the **calendering process**, a critical manufacturing step where the electrode is compressed between heavy rollers to increase its density and improve electrical contact.

Physically, this process:

1.  **Compresses** the electrode thickness (Z-axis).
2.  **Reorients** particles (flakes align horizontally).
3.  **Deforms** softer particles (graphite flakes may spread or crack, though we model only plastic deformation here).

---

## 1. Geometric Transformation

The transformation is applied to the discrete `OblateSpheroid` particles generated in Step 2.

### 1.1. Z-Compression

The most direct effect is the reduction of the domain height. All particle centers are scaled:

$$
z_{new} = z_{old} \times \text{Compression Ratio}
$$

- **Note:** The simulation domain boundaries ($L_z$) were already defined in Step 1. This step moves the _content_ to fit that final boundary.

### 1.2. Particle Reorientation (`orientation_enhancement`)

Compression forces graphite flakes to lie flat. We model this by rotating the particle's orientation matrix $R$ towards the Z-axis.

The code performs a **Spherical Linear Interpolation (Slerp)** between the particle's current $c$-axis vector and the global vertical vector $\hat{k} = [0,0,1]$.

$$
\vec{c}_{new} = \text{Slerp}(\vec{c}_{old}, \hat{k}, t=\text{orientation\_enhancement})
$$

- $t=0$: No change.
- $t=1$: Perfect alignment (all flakes horizontal).

### 1.3. Plastic Deformation (`particle_deformation`)

Graphite is relatively soft and can deform under pressure. We approximate this as an isochoric (volume-conserving) deformation of the spheroid axes.

- **Flattening:** The thickness ($c$) is reduced.
- **Widening:** The basal radius ($a$) is increased to conserve volume ($V \propto a^2 c$).

$$
f_{deform} = 1 + \text{deformation} \times \left(\frac{1}{CR} - 1\right)
$$

$$
a_{new} = a_{old} \times \sqrt{f_{deform}}
$$

$$
c_{new} = c_{old} / f_{deform}
$$

---

## 2. Field Transformation

Steps 3 (Silicon) and 4 (CBD/Binder) generate volume fraction fields.

- **Current Implementation:** The pipeline currently generates these fields directly on the **Final (Compressed) Grid** ($N \times N \times N$). Therefore, no explicit resampling or interpolation is required in this step.
- **Future Proofing:** If the pipeline is updated to generate fields on the expanded grid (to capture pre-calendering porosity morphology), this step will include a `scipy.ndimage.zoom` operation to resample the fields to the final resolution.

---

## 3. Usage

This step is an **In-Place** operation on the particle list. It modifies the coordinate data directly.

```python
from structure.calendering import apply_calendering

# Inputs from previous steps:
# particles (List[OblateSpheroid]), comp, domain, si_result, cbd_result

apply_calendering(
    particles,
    comp,
    domain,
    si_result,
    cbd_result,
    sim  # Contains deformation/orientation params
)

# After this call, 'particles' have new Z-coordinates and shapes.
# The 'si_result' and 'cbd_result' remain valid for the final grid.
```
