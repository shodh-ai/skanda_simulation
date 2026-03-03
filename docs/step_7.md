# Step 7: Percolation Validator

**File:** `structure/percolation.py`

This module performs the final **Quality Control (QC)** check on the generated microstructure. It verifies that the electrode is functional by checking for continuous pathways for both electrons and lithium ions.

Unlike previous steps which might only issue warnings, this step has the authority to **Reject the Simulation**. If the generated geometry does not meet the connectivity threshold, it raises a `PercolationFailed` exception, triggering the main pipeline to discard the result and retry with a new random seed.

---

## 1. Physical Networks

The validator analyzes two distinct transport networks:

### 1.1. Electronic Network (Solid)

Electrons must travel from the Current Collector ($Z=0$) through the solid matrix to reach active particles.

- **Composition:** Graphite $\cup$ Silicon $\cup$ CBD (Conductive Additive).
- **Insulators:** Binder and SEI are treated as electronically insulating.
- **Condition:** A continuous cluster of solid voxels must connect the $Z=0$ face to the $Z=L_z$ face.

### 1.2. Ionic Network (Pore)

Lithium ions must travel from the Separator/Electrolyte interface ($Z=L_z$) through the electrolyte-filled pores.

- **Composition:** Void space (Pores).
- **Blockers:** Any solid phase, including SEI, blocks ionic transport in the pore phase (though SEI conducts ions _into_ particles, it takes up space in the pore).
- **Condition:** A continuous cluster of pore voxels must connect the $Z=L_z$ face to the $Z=0$ face.

---

## 2. Algorithm: Connected Component Analysis

The percolation check uses **6-connectivity** (face neighbors only) to define clusters.

1.  **Labeling:** Uses `scipy.ndimage.label` (a fast C-implementation of Union-Find) to assign a unique integer ID to every disconnected blob of material.
2.  **Face Identification:**
    - Identify the set of unique labels present at the bottom slice ($Z=0$): $S_{bottom}$.
    - Identify the set of unique labels present at the top slice ($Z=L_z$): $S_{top}$.
3.  **Intersection:**
    - Percolating Clusters = $S_{bottom} \cap S_{top}$.
    - If this intersection is empty, the network does not percolate.

### The Percolation Fraction

Mere connectivity is not enough; a single thin wire connecting top to bottom counts as "percolating" but is poor for performance. The metric used for validation is the **Percolating Fraction**:

$$
\phi_{perc} = \frac{\text{Volume of percolating clusters}}{\text{Total volume of phase}}
$$

- **Target:** Typically $> 0.95$ (95% of active material is electrically connected).
- **Config:** Controlled by `percolation_min_threshold` in `RunConfig`.

---

## 3. Failure & Retry Logic

If the **Electronic Percolation Fraction** falls below the configured threshold:

1.  A `PercolationFailed` exception is raised.
2.  This exception carries metadata (Run ID, Seed, Measured Fraction).
3.  The main generation loop catches this exception.
4.  The Seed is incremented (`seed += 1`).
5.  The entire generation pipeline restarts from Step 2 (Carbon Placement).

_Note: Ionic percolation failure generates a warning but does not automatically trigger a retry, as low porosity designs may intentionally have tortuous or semi-blocked paths._

---

## 4. Output

The step returns a `PercolationResult` object containing:

- **`electronic_percolating`:** (Bool) Does it connect?
- **`electronic_fraction`:** (Float) 0.0 to 1.0.
- **`ionic_percolating`:** (Bool) Does the pore space connect?
- **`actual_porosity`:** (Float) The final measured porosity after all steps (including SEI growth).

### Example Usage

```python
from structure.percolation import validate_percolation

# Inputs:
# comp, domain, sim, carbon_label, si_result, cbd_result, sei_result

try:
    qc_result = validate_percolation(
        comp, domain, sim, carbon_label, si_result, cbd_result, sei_result
    )
    print(qc_result.summary())

except PercolationFailed as e:
    print(f"QC FAILED: {e}")
    # Trigger retry logic here...

# Output (Success):
# ==============================================================
#   PERCOLATION VALIDATOR
# ==============================================================
#   Electronic network  [✓  PASS]
#     percolating frac : 0.9850  (threshold ≥ 0.95)
#     components       : 12  (1 spans Z)
#
#   Ionic (pore) path   [✓  PASS]
#     percolating frac : 0.9990
#
#   Actual porosity    : 0.3105
# ==============================================================
```
